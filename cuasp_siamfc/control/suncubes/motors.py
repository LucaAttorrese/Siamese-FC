"""Motor adapters for the two-axis Kollmorgen PCMM/KAS controller.

The TCP application protocol is deliberately small::

    abs[0,1]=axis1_position_deg,axis2_position_deg\n
    enc[0,1]?\n
    status?\n
    stop[0]=1\n

Only atomic absolute position pairs are sent to KAS.  KAS measures the interval
between references in its deterministic PLC task and closes the position loop
by continuously updating the motor velocity.  All public angular units in this
module are degrees, degrees/second and seconds.
"""

from __future__ import annotations

import asyncio
import logging
import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


logger = logging.getLogger(__name__)


# APPLICATION CONFIGURATION -------------------------------------------------
# Keep these safe command ranges aligned with Main.src:
#   hard axis ranges minus PositionLimitMarginDeg.
DEFAULT_AXIS1_POSITION_MIN_DEG = -29.5
DEFAULT_AXIS1_POSITION_MAX_DEG = 29.5
DEFAULT_AXIS2_POSITION_MIN_DEG = -19.5
DEFAULT_AXIS2_POSITION_MAX_DEG = 19.5

# TCP timeouts [s].  They affect connection and request/reply queries only;
# position-reference writes remain fire-and-forget.
DEFAULT_CONNECT_TIMEOUT_S = 1.0
DEFAULT_IO_TIMEOUT_S = 1.0

# Accepted for compatibility with existing callers. Reference timing is now
# measured by KAS, so this value is not used to compute a velocity in Python.
DEFAULT_REFERENCE_PERIOD_S = 1.0 / 30.0


__version__ = "5.0.0-pcmm-position-reference"
MOTOR_PROTOCOL_VERSION = "abs-position-v1"
MOTOR_BUILD_ID = "2026-08-03-minimal-position-to-velocity-v1"
PCMM_STATUS_SCHEMA_POSITION_V1 = "position-v1-15-fields"
PCMM_STATUS_SCHEMA_LEGACY_V6 = "legacy-error-servo-v6-12-fields"


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _format_float(value: float) -> str:
    return f"{_finite_float(value, 'motor command'):.6f}"


def _clip_range(value: float, minimum: float, maximum: float) -> float:
    value = _finite_float(value, "position")
    return max(minimum, min(maximum, value))


def _normalise_encoder_axes(axes: Iterable[int]) -> Tuple[int, ...]:
    out = tuple(int(axis) for axis in axes)
    if not out:
        raise ValueError("At least one encoder axis must be requested")
    if any(axis not in (0, 1) for axis in out):
        raise ValueError("Encoder axis must be 0 (azimuth) or 1 (polar)")
    if len(set(out)) != len(out):
        raise ValueError("Encoder axes must be unique")
    return out


def _parse_encoder_reply(reply: str, expected_axes: Iterable[int]) -> Tuple[float, ...]:
    expected_axes = _normalise_encoder_axes(expected_axes)
    text = reply.strip("\r\n ")
    if not text.startswith("enc[") or "]=" not in text:
        raise ValueError(f"Invalid encoder reply: {reply!r}")

    axis_text, value_text = text.split("]=", 1)
    reply_axes = _normalise_encoder_axes(
        part.strip() for part in axis_text[4:].split(",")
    )
    values = tuple(float(part.strip()) for part in value_text.split(","))
    if len(reply_axes) != len(values) or any(not math.isfinite(v) for v in values):
        raise ValueError(f"Invalid encoder reply: {reply!r}")

    data = dict(zip(reply_axes, values))
    missing = [axis for axis in expected_axes if axis not in data]
    if missing:
        raise ValueError(
            f"Encoder reply {reply!r} does not contain requested axes {missing}"
        )
    return tuple(data[axis] for axis in expected_axes)


def _parse_bool(value: str) -> bool:
    value = value.strip().upper()
    if value in {"TRUE", "1"}:
        return True
    if value in {"FALSE", "0"}:
        return False
    raise ValueError(f"Invalid PLC boolean value: {value!r}")


@dataclass(frozen=True)
class PCMMStatus:
    controller_state: int
    position_reference_sequence: int
    position_reference_fresh: bool
    position_reference_valid: bool
    tcp_connected: bool
    axis1_velocity_command_deg_s: float
    axis2_velocity_command_deg_s: float
    axis1_reference_deg: float
    axis2_reference_deg: float
    axis1_actual_deg: float
    axis2_actual_deg: float
    reference_dt_s: float
    reference_age_s: float
    position_reference_clamped: bool
    controller_fault: bool
    status_schema: str = PCMM_STATUS_SCHEMA_POSITION_V1


def _parse_status_reply(reply: str) -> PCMMStatus:
    text = reply.strip("\r\n ")
    prefix = "status="
    if not text.startswith(prefix):
        raise ValueError(f"Invalid status reply: {reply!r}")
    fields = [field.strip() for field in text[len(prefix) :].split(",")]
    if len(fields) == 15:
        return PCMMStatus(
            controller_state=int(fields[0]),
            position_reference_sequence=int(fields[1]),
            position_reference_fresh=_parse_bool(fields[2]),
            position_reference_valid=_parse_bool(fields[3]),
            tcp_connected=_parse_bool(fields[4]),
            axis1_velocity_command_deg_s=float(fields[5]),
            axis2_velocity_command_deg_s=float(fields[6]),
            axis1_reference_deg=float(fields[7]),
            axis2_reference_deg=float(fields[8]),
            axis1_actual_deg=float(fields[9]),
            axis2_actual_deg=float(fields[10]),
            reference_dt_s=float(fields[11]),
            reference_age_s=float(fields[12]),
            position_reference_clamped=_parse_bool(fields[13]),
            controller_fault=_parse_bool(fields[14]),
            status_schema=PCMM_STATUS_SCHEMA_POSITION_V1,
        )

    if len(fields) == 12:
        # Legacy 2026-07 error-servo status:
        # MotionMode, sequence, fresh, timed_out, tcp, v1, v2,
        # control_mode, error1, error2, sample_dt, sample_age.
        # Keep it parseable for connection diagnostics, but mark the schema so
        # position-reference applications cannot mistake these values for the
        # new reference/encoder fields.
        reference_fresh = _parse_bool(fields[2])
        reference_timed_out = _parse_bool(fields[3])
        return PCMMStatus(
            controller_state=int(fields[0]),
            position_reference_sequence=int(fields[1]),
            position_reference_fresh=reference_fresh,
            position_reference_valid=reference_fresh and not reference_timed_out,
            tcp_connected=_parse_bool(fields[4]),
            axis1_velocity_command_deg_s=float(fields[5]),
            axis2_velocity_command_deg_s=float(fields[6]),
            axis1_reference_deg=math.nan,
            axis2_reference_deg=math.nan,
            axis1_actual_deg=math.nan,
            axis2_actual_deg=math.nan,
            reference_dt_s=float(fields[10]),
            reference_age_s=float(fields[11]),
            position_reference_clamped=False,
            controller_fault=False,
            status_schema=PCMM_STATUS_SCHEMA_LEGACY_V6,
        )

    raise ValueError(
        "Invalid status reply: expected the 15-field position schema or the "
        f"12-field legacy schema, got {len(fields)} fields: {reply!r}"
    )


class MotorsControllerInterface:
    """Common motor-controller interface; angular units are degrees."""

    def __init__(self) -> None:
        self.isOK = False

    def connect_motors(self) -> bool:
        raise NotImplementedError

    def abs_azimuthal_rotate(self, az_deg: float) -> None:
        raise NotImplementedError

    def abs_polar_rotate(self, po_deg: float) -> None:
        raise NotImplementedError

    def azimuthal_rotate(self, delta_az_deg: float) -> None:
        raise NotImplementedError

    def polar_rotate(self, delta_po_deg: float) -> None:
        raise NotImplementedError

    def rel_rotate(self, delta_az_deg: float, delta_po_deg: float) -> None:
        raise NotImplementedError

    def get_encoder_position(self, motor_index: int) -> float:
        raise NotImplementedError

    def get_encoder_positions(self) -> Tuple[float, float]:
        raise NotImplementedError


class PCMMMotorAdapterUDP(MotorsControllerInterface):
    """Legacy indexed UDP adapter; it does not implement the new TCP loop."""

    def __init__(self, IP: str, PORT: int) -> None:
        super().__init__()
        self.ip = IP
        self.port = int(PORT)
        self.kas_address = None
        self.sock: Optional[socket.socket] = None

    def connect_motors(self) -> bool:
        self.isOK = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        logger.info("MOTORS UDP: listening on %s:%s", self.ip, self.port)
        data, self.kas_address = self.sock.recvfrom(1024)
        logger.info("MOTORS UDP: received %r from %s", data, self.kas_address)
        time.sleep(1.0)
        self.sock.sendto(b"I am ready to send", self.kas_address)
        self.isOK = True
        return True

    def _send(self, message: str) -> None:
        if not self.isOK or self.sock is None or self.kas_address is None:
            raise ConnectionError("MOTORS UDP: adapter is not connected")
        self.sock.sendto(message.encode("ascii"), self.kas_address)

    def abs_azimuthal_rotate(self, az_deg: float) -> None:
        self._send(f"var[0]={_format_float(az_deg)}\n")

    def abs_polar_rotate(self, po_deg: float) -> None:
        self._send(f"var[1]={_format_float(po_deg)}\n")

    def azimuthal_rotate(self, delta_az_deg: float) -> None:
        self._send(f"var[2]={_format_float(delta_az_deg)}\n")

    def polar_rotate(self, delta_po_deg: float) -> None:
        self._send(f"var[3]={_format_float(delta_po_deg)}\n")

    def rel_rotate(self, delta_az_deg: float, delta_po_deg: float) -> None:
        self._send(
            f"var[2,3]={_format_float(delta_az_deg)},"
            f"{_format_float(delta_po_deg)}\n"
        )

    def get_encoder_position(self, motor_index: int) -> float:
        raise NotImplementedError("Encoder request/reply is not implemented over UDP")

    def get_encoder_positions(self) -> Tuple[float, float]:
        raise NotImplementedError("Encoder request/reply is not implemented over UDP")


class _PCMMTCPBase(MotorsControllerInterface):
    """Synchronous core shared by the sync and asyncio-facing adapters."""

    def __init__(
        self,
        IP: str,
        PORT: int,
        *,
        reference_period_s: float = DEFAULT_REFERENCE_PERIOD_S,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        io_timeout_s: float = DEFAULT_IO_TIMEOUT_S,
        axis1_position_min_deg: float = DEFAULT_AXIS1_POSITION_MIN_DEG,
        axis1_position_max_deg: float = DEFAULT_AXIS1_POSITION_MAX_DEG,
        axis2_position_min_deg: float = DEFAULT_AXIS2_POSITION_MIN_DEG,
        axis2_position_max_deg: float = DEFAULT_AXIS2_POSITION_MAX_DEG,
    ) -> None:
        super().__init__()
        reference_period_s = _finite_float(reference_period_s, "reference_period_s")
        connect_timeout_s = _finite_float(connect_timeout_s, "connect_timeout_s")
        io_timeout_s = _finite_float(io_timeout_s, "io_timeout_s")
        if reference_period_s <= 0.0:
            raise ValueError("reference_period_s must be positive")
        if connect_timeout_s <= 0.0 or io_timeout_s <= 0.0:
            raise ValueError("TCP timeouts must be positive")

        limits = (
            _finite_float(axis1_position_min_deg, "axis1_position_min_deg"),
            _finite_float(axis1_position_max_deg, "axis1_position_max_deg"),
            _finite_float(axis2_position_min_deg, "axis2_position_min_deg"),
            _finite_float(axis2_position_max_deg, "axis2_position_max_deg"),
        )
        if limits[0] >= limits[1] or limits[2] >= limits[3]:
            raise ValueError("Each position minimum must be lower than its maximum")

        self.ip = str(IP)
        self.port = int(PORT)
        self.reference_period_s = reference_period_s
        self.connect_timeout_s = connect_timeout_s
        self.io_timeout_s = io_timeout_s
        self.axis1_position_min_deg = limits[0]
        self.axis1_position_max_deg = limits[1]
        self.axis2_position_min_deg = limits[2]
        self.axis2_position_max_deg = limits[3]

        self.sock: Optional[socket.socket] = None
        self._io_lock = threading.RLock()
        self._rx_buffer = ""
        self._last_tcp_request = ""
        self._last_send_monotonic_s: Optional[float] = None
        self._position_target_deg: Optional[Tuple[float, float]] = None

    def connect_motors(self) -> bool:
        with self._io_lock:
            self._close_locked(send_stop=False)
            try:
                sock = socket.create_connection(
                    (self.ip, self.port), timeout=self.connect_timeout_s
                )
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.settimeout(self.io_timeout_s)
                self.sock = sock
                self._rx_buffer = ""
                self._position_target_deg = None
                self.isOK = True
                logger.info(
                    "MOTORS: connected to %s:%s module=%s protocol=%s build=%s",
                    self.ip,
                    self.port,
                    __version__,
                    MOTOR_PROTOCOL_VERSION,
                    MOTOR_BUILD_ID,
                )
                return True
            except Exception as exc:
                self._mark_disconnected_locked()
                raise ConnectionError(
                    f"MOTORS: cannot connect to {self.ip}:{self.port}"
                ) from exc

    def _require_socket_locked(self) -> socket.socket:
        if not self.isOK or self.sock is None:
            raise ConnectionError("MOTORS: TCP socket is not connected")
        return self.sock

    def _mark_disconnected_locked(self) -> None:
        sock = self.sock
        self.sock = None
        self.isOK = False
        self._rx_buffer = ""
        self._position_target_deg = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _close_locked(self, *, send_stop: bool) -> None:
        sock = self.sock
        if sock is None:
            self.isOK = False
            self._rx_buffer = ""
            self._position_target_deg = None
            return
        if send_stop and self.isOK:
            try:
                sock.sendall(b"stop[0]=1\n")
            except OSError:
                pass
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        finally:
            self.sock = None
            self.isOK = False
            self._rx_buffer = ""
            self._position_target_deg = None

    def close_motors_sync(self, *, send_stop: bool = True) -> None:
        with self._io_lock:
            self._close_locked(send_stop=send_stop)
            logger.info("MOTORS: connection closed")

    def _send_message_locked(self, message: str) -> None:
        sock = self._require_socket_locked()
        try:
            sock.sendall(message.encode("ascii"))
            self._last_send_monotonic_s = time.monotonic()
        except (OSError, TimeoutError) as exc:
            self._mark_disconnected_locked()
            raise ConnectionError(
                f"MOTORS: TCP send failed for {message.strip()!r}: {exc}"
            ) from exc

    def _pop_rx_line_locked(self) -> Optional[str]:
        if "\n" not in self._rx_buffer:
            return None
        line, self._rx_buffer = self._rx_buffer.split("\n", 1)
        return line.rstrip("\r")

    def _recv_line_locked(self, timeout_s: float) -> str:
        sock = self._require_socket_locked()
        timeout_s = _finite_float(timeout_s, "timeout_s")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        old_timeout = sock.gettimeout()
        sock.settimeout(timeout_s)
        try:
            line = self._pop_rx_line_locked()
            while line is None:
                chunk = sock.recv(1024)
                if not chunk:
                    self._mark_disconnected_locked()
                    raise ConnectionError("MOTORS: PCMM closed the TCP connection")
                self._rx_buffer += chunk.decode("ascii", errors="replace")
                line = self._pop_rx_line_locked()
            return line
        except socket.timeout as exc:
            request = self._last_tcp_request or "<unknown>"
            # A late reply would otherwise be consumed by the next query and
            # desynchronise the request/reply stream. Force a clean reconnect.
            self._mark_disconnected_locked()
            raise TimeoutError(
                f"MOTORS: timeout waiting for reply to {request!r}"
            ) from exc
        except OSError as exc:
            self._mark_disconnected_locked()
            raise ConnectionError(f"MOTORS: TCP receive failed: {exc}") from exc
        finally:
            if self.sock is sock:
                try:
                    sock.settimeout(old_timeout)
                except OSError:
                    pass

    def _request_reply_sync(self, message: str, timeout_s: float) -> str:
        with self._io_lock:
            self._last_tcp_request = message.strip()
            self._send_message_locked(message)
            return self._recv_line_locked(timeout_s)

    def _clip_pair(self, axis1_deg: float, axis2_deg: float) -> Tuple[float, float]:
        return (
            _clip_range(
                axis1_deg,
                self.axis1_position_min_deg,
                self.axis1_position_max_deg,
            ),
            _clip_range(
                axis2_deg,
                self.axis2_position_min_deg,
                self.axis2_position_max_deg,
            ),
        )

    def _send_absolute_target_locked(
        self, axis1_deg: float, axis2_deg: float
    ) -> Tuple[float, float]:
        requested = (
            _finite_float(axis1_deg, "axis1 position"),
            _finite_float(axis2_deg, "axis2 position"),
        )
        target = self._clip_pair(*requested)
        if target != requested:
            logger.warning(
                "MOTORS: position reference clipped from (%.6f, %.6f) deg "
                "to (%.6f, %.6f) deg",
                requested[0], requested[1], target[0], target[1],
            )
        self._send_message_locked(
            f"abs[0,1]={_format_float(target[0])},{_format_float(target[1])}\n"
        )
        self._position_target_deg = target
        return target

    def abs_rotate_sync(self, axis1_deg: float, axis2_deg: float) -> None:
        """Send one bounded, atomic absolute reference pair [deg]."""
        with self._io_lock:
            self._send_absolute_target_locked(axis1_deg, axis2_deg)

    def _query_encoder_positions_locked(
        self, axes: Iterable[int], timeout_s: float
    ) -> Tuple[float, ...]:
        axes = _normalise_encoder_axes(axes)
        axis_text = ",".join(str(axis) for axis in axes)
        message = f"enc[{axis_text}]?\n"
        self._last_tcp_request = message.strip()
        self._send_message_locked(message)
        reply = self._recv_line_locked(timeout_s)
        return _parse_encoder_reply(reply, axes)

    def _ensure_position_target_locked(self, timeout_s: float) -> Tuple[float, float]:
        if self._position_target_deg is None:
            actual = self._query_encoder_positions_locked((0, 1), timeout_s)
            self._position_target_deg = (actual[0], actual[1])
            logger.debug(
                "MOTORS: relative-reference accumulator initialized from "
                "encoder (%.6f, %.6f) deg",
                actual[0], actual[1],
            )
        return self._position_target_deg

    def rel_rotate_sync(
        self,
        delta_axis1_deg: float,
        delta_axis2_deg: float,
        *,
        dt_s: Optional[float] = None,
    ) -> None:
        """Accumulate one correction and send the resulting absolute pair.

        This preserves the API used by ``pointing_run_nucleo_fused_fastbin.py``.
        ``dt_s`` remains accepted but is intentionally not used: KAS derives the
        reference interval from the number of deterministic PLC cycles.
        """
        delta1 = _finite_float(delta_axis1_deg, "delta_axis1_deg")
        delta2 = _finite_float(delta_axis2_deg, "delta_axis2_deg")
        if dt_s is not None:
            dt_s = _finite_float(dt_s, "dt_s")
            if dt_s <= 0.0:
                raise ValueError("dt_s must be positive")
        with self._io_lock:
            current = self._ensure_position_target_locked(self.io_timeout_s)
            self._send_absolute_target_locked(
                current[0] + delta1,
                current[1] + delta2,
            )

    def set_visual_error_sync(
        self,
        axis1_correction_deg: float,
        axis2_correction_deg: float,
        *,
        sample_dt_s: Optional[float] = None,
    ) -> None:
        """Compatibility alias; there is no dedicated visual protocol or mode."""
        self.rel_rotate_sync(
            axis1_correction_deg,
            axis2_correction_deg,
            dt_s=sample_dt_s,
        )

    def _single_axis_absolute_sync(
        self, axis: int, value_deg: float, timeout_s: float
    ) -> None:
        axis = int(axis)
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        value = _finite_float(value_deg, "position")
        with self._io_lock:
            current = self._ensure_position_target_locked(timeout_s)
            target = [current[0], current[1]]
            target[axis] = value
            self._send_absolute_target_locked(target[0], target[1])

    def stop_sync(self) -> None:
        with self._io_lock:
            self._send_message_locked("stop[0]=1\n")
            self._position_target_deg = None
            logger.debug("MOTORS: safe stop command sent")

    def get_encoder_position_sync(
        self, motor_index: int, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> float:
        with self._io_lock:
            return self._query_encoder_positions_locked((motor_index,), timeout_s)[0]

    def get_encoder_positions_sync(
        self, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> Tuple[float, float]:
        with self._io_lock:
            values = self._query_encoder_positions_locked((0, 1), timeout_s)
            return values[0], values[1]

    def get_status_sync(self, timeout_s: float = DEFAULT_IO_TIMEOUT_S) -> PCMMStatus:
        return _parse_status_reply(self._request_reply_sync("status?\n", timeout_s))

    def check_connection_sync(self, timeout_s: float = DEFAULT_IO_TIMEOUT_S) -> bool:
        try:
            status = self.get_status_sync(timeout_s)
        except (ConnectionError, TimeoutError, ValueError):
            return False
        return bool(status.tcp_connected and self.isOK)

    @property
    def position_target_deg(self) -> Optional[Tuple[float, float]]:
        with self._io_lock:
            return self._position_target_deg

    @property
    def last_send_monotonic_s(self) -> Optional[float]:
        with self._io_lock:
            return self._last_send_monotonic_s


class PCMMMotorAdapterTCP(_PCMMTCPBase):
    """Asyncio-facing adapter; blocking socket work runs in worker threads."""

    async def connect_motors_async(self) -> bool:
        return await asyncio.to_thread(super().connect_motors)

    async def close_motors(self, *, send_stop: bool = True) -> None:
        await asyncio.to_thread(self.close_motors_sync, send_stop=send_stop)

    async def disconnect_motors(self) -> None:
        await self.close_motors(send_stop=True)

    async def disconnect(self) -> None:
        await self.close_motors(send_stop=True)

    async def close(self) -> None:
        await self.close_motors(send_stop=True)

    async def rel_rotate(
        self,
        delta_axis1_deg: float,
        delta_axis2_deg: float,
        *,
        dt_s: Optional[float] = None,
    ) -> None:
        await asyncio.to_thread(
            self.rel_rotate_sync,
            delta_axis1_deg,
            delta_axis2_deg,
            dt_s=dt_s,
        )

    async def set_visual_error(
        self,
        axis1_correction_deg: float,
        axis2_correction_deg: float,
        *,
        sample_dt_s: Optional[float] = None,
    ) -> None:
        await asyncio.to_thread(
            self.set_visual_error_sync,
            axis1_correction_deg,
            axis2_correction_deg,
            sample_dt_s=sample_dt_s,
        )

    async def abs_rotate(self, axis1_deg: float, axis2_deg: float) -> None:
        await asyncio.to_thread(self.abs_rotate_sync, axis1_deg, axis2_deg)

    async def abs_azimuthal_rotate(
        self, axis1_deg: float, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> None:
        await asyncio.to_thread(
            self._single_axis_absolute_sync, 0, axis1_deg, timeout_s
        )

    async def abs_polar_rotate(
        self, axis2_deg: float, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> None:
        await asyncio.to_thread(
            self._single_axis_absolute_sync, 1, axis2_deg, timeout_s
        )

    async def azimuthal_rotate(
        self, delta_axis1_deg: float, *, dt_s: Optional[float] = None
    ) -> None:
        await self.rel_rotate(delta_axis1_deg, 0.0, dt_s=dt_s)

    async def polar_rotate(
        self, delta_axis2_deg: float, *, dt_s: Optional[float] = None
    ) -> None:
        await self.rel_rotate(0.0, delta_axis2_deg, dt_s=dt_s)

    async def stop(self) -> None:
        await asyncio.to_thread(self.stop_sync)

    async def get_encoder_position(
        self, motor_index: int, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> float:
        return await asyncio.to_thread(
            self.get_encoder_position_sync, motor_index, timeout_s
        )

    async def get_encoder_positions(
        self, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> Tuple[float, float]:
        return await asyncio.to_thread(self.get_encoder_positions_sync, timeout_s)

    async def get_status(self, timeout_s: float = DEFAULT_IO_TIMEOUT_S) -> PCMMStatus:
        return await asyncio.to_thread(self.get_status_sync, timeout_s)

    async def check_connection(self, timeout_s: float = DEFAULT_IO_TIMEOUT_S) -> bool:
        return await asyncio.to_thread(self.check_connection_sync, timeout_s)

    async def __aenter__(self) -> "PCMMMotorAdapterTCP":
        await self.connect_motors_async()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close_motors(send_stop=True)


class PCMMMotorAdapterTCP_sync(_PCMMTCPBase):
    """Synchronous adapter using the same position-reference protocol."""

    def close_motors(self, *, send_stop: bool = True) -> None:
        self.close_motors_sync(send_stop=send_stop)

    def disconnect_motors(self) -> None:
        self.close_motors(send_stop=True)

    def disconnect(self) -> None:
        self.close_motors(send_stop=True)

    def close(self) -> None:
        self.close_motors(send_stop=True)

    def rel_rotate(
        self,
        delta_axis1_deg: float,
        delta_axis2_deg: float,
        *,
        dt_s: Optional[float] = None,
    ) -> None:
        self.rel_rotate_sync(delta_axis1_deg, delta_axis2_deg, dt_s=dt_s)

    def set_visual_error(
        self,
        axis1_correction_deg: float,
        axis2_correction_deg: float,
        *,
        sample_dt_s: Optional[float] = None,
    ) -> None:
        self.set_visual_error_sync(
            axis1_correction_deg,
            axis2_correction_deg,
            sample_dt_s=sample_dt_s,
        )

    def abs_rotate(self, axis1_deg: float, axis2_deg: float) -> None:
        self.abs_rotate_sync(axis1_deg, axis2_deg)

    def abs_azimuthal_rotate(
        self, axis1_deg: float, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> None:
        self._single_axis_absolute_sync(0, axis1_deg, timeout_s)

    def abs_polar_rotate(
        self, axis2_deg: float, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> None:
        self._single_axis_absolute_sync(1, axis2_deg, timeout_s)

    def azimuthal_rotate(
        self, delta_axis1_deg: float, *, dt_s: Optional[float] = None
    ) -> None:
        self.rel_rotate_sync(delta_axis1_deg, 0.0, dt_s=dt_s)

    def polar_rotate(
        self, delta_axis2_deg: float, *, dt_s: Optional[float] = None
    ) -> None:
        self.rel_rotate_sync(0.0, delta_axis2_deg, dt_s=dt_s)

    def stop(self) -> None:
        self.stop_sync()

    def get_encoder_position(
        self, motor_index: int, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> float:
        return self.get_encoder_position_sync(motor_index, timeout_s)

    def get_encoder_positions(
        self, timeout_s: float = DEFAULT_IO_TIMEOUT_S
    ) -> Tuple[float, float]:
        return self.get_encoder_positions_sync(timeout_s)

    def get_status(self, timeout_s: float = DEFAULT_IO_TIMEOUT_S) -> PCMMStatus:
        return self.get_status_sync(timeout_s)

    def check_connection(self, timeout_s: float = DEFAULT_IO_TIMEOUT_S) -> bool:
        return self.check_connection_sync(timeout_s)

    def __enter__(self) -> "PCMMMotorAdapterTCP_sync":
        self.connect_motors()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close_motors(send_stop=True)
