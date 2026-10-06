"""Tests for cuasp.py's MotorCommandDispatcher latest-only dispatch semantics.

Uses a lightweight in-process spy instead of a real motors adapter so the
assertions are about *which Python method was called with what*, not about
inferring that from raw wire bytes -- cleaner for the "no duplicate
actuation" and "still calls set_visual_error, not abs_rotate" properties
below. Wire-level behavior (the actual `abs[0,1]=...` bytes on the socket) is
covered separately in test_motors_fake_server.py.
"""

import asyncio
import time

import pytest

from suncubes.motors import PCMMMotorAdapterTCP


class SpyMotors:
    """Records every dispatch call instead of touching a socket."""

    def __init__(self, encoder_positions: tuple[float, float] = (0.0, 0.0)):
        self.calls: list[tuple] = []
        self._encoder_positions = encoder_positions

    def connect_motors(self) -> bool:
        return True

    def set_visual_error(self, axis1_correction_deg: float, axis2_correction_deg: float) -> None:
        self.calls.append(("set_visual_error", axis1_correction_deg, axis2_correction_deg))

    def abs_rotate(self, axis1_deg: float, axis2_deg: float) -> None:
        self.calls.append(("abs_rotate", axis1_deg, axis2_deg))

    def get_encoder_positions(self) -> tuple[float, float]:
        return self._encoder_positions

    def stop(self) -> None:
        self.calls.append(("stop",))

    def disconnect_motors(self) -> None:
        pass


@pytest.fixture
def cuasp(cuasp_module):
    return cuasp_module


async def test_rapid_submissions_collapse_to_a_single_send_of_the_latest(cuasp, monkeypatch):
    monkeypatch.setattr(cuasp, "MOTOR_COMMAND_HZ", 20.0)
    spy = SpyMotors()
    dispatcher = cuasp.MotorCommandDispatcher(spy)
    await dispatcher.start()
    try:
        for i in range(5):
            correction = cuasp.CuasCorrection(
                reliable=True, reason="ok", d_az_motor_units=float(i), d_po_motor_units=0.0
            )
            dispatcher.submit(correction, frame_index=i, capture_ns=0)

        await asyncio.sleep(0.3)
        assert len(spy.calls) == 1, "no duplicate actuation for corrections replaced before being sent"
        assert spy.calls[0] == ("abs_rotate", 4.0, 0.0), "only the latest correction must be sent"
        assert dispatcher.status.replaced_count >= 4
    finally:
        await dispatcher.stop()


async def test_dispatcher_has_no_content_deduplication_fresh_sample_gating_is_upstream(cuasp, monkeypatch):
    """Documents CURRENT behavior, not a design flaw in the dispatcher.

    The dispatcher's only guarantee is latest-wins/no-backlog (see the test
    above): it has no notion of "camera sample identity", so submitting the
    *same* CuasCorrection values again after the first one already went out
    is indistinguishable from a genuinely new correction and gets sent again.
    Deciding whether a `CameraCorrection` represents a new visual sample --
    root README.md's "Non-negotiable interface semantics" -- is `cuasp.py`
    main()'s job, and main() does not yet implement that check (it calls
    `camera_api.get_latest()` and submits every loop tick regardless of
    whether the sample changed). This test is a marker for that still-open
    gap, not a claim that today's code already prevents duplicate actuation
    end to end.
    """
    monkeypatch.setattr(cuasp, "MOTOR_COMMAND_HZ", 20.0)
    spy = SpyMotors()
    dispatcher = cuasp.MotorCommandDispatcher(spy)
    await dispatcher.start()
    try:
        correction = cuasp.CuasCorrection(reliable=True, reason="ok", d_az_motor_units=1.0, d_po_motor_units=-2.0)
        dispatcher.submit(correction, frame_index=1, capture_ns=0)
        await asyncio.sleep(0.15)
        dispatcher.submit(correction, frame_index=2, capture_ns=0)  # same values, new submission
        await asyncio.sleep(0.15)
        assert spy.calls == [
            ("abs_rotate", 1.0, -2.0),
            ("abs_rotate", 1.0, -2.0),
        ], "no content-based dedup exists today -- see docstring"
    finally:
        await dispatcher.stop()


async def test_stale_command_is_dropped_not_sent(cuasp, monkeypatch):
    monkeypatch.setattr(cuasp, "MOTOR_COMMAND_STALE_TIMEOUT_S", 0.05)
    spy = SpyMotors()
    dispatcher = cuasp.MotorCommandDispatcher(spy)
    await dispatcher.start()
    try:
        correction = cuasp.CuasCorrection(reliable=True, reason="ok", d_az_motor_units=1.0, d_po_motor_units=2.0)
        # Inject an already-old request directly, sidestepping any race
        # between submission time and the dispatcher's own send-rate timing.
        stale_request = cuasp.MotorCommandRequest(
            correction=correction,
            frame_index=1,
            source_capture_ns=0,
            created_monotonic_ns=time.monotonic_ns() - int(1.0 * 1e9),
            reason="queued",
        )
        dispatcher._queue.put_latest(stale_request)
        await asyncio.sleep(0.2)
        assert spy.calls == [], "a command older than the stale timeout must never be sent"
        assert dispatcher.status.stale_count >= 1
    finally:
        await dispatcher.stop()


async def test_unreliable_correction_is_never_dispatched(cuasp, monkeypatch):
    monkeypatch.setattr(cuasp, "MOTOR_COMMAND_HZ", 20.0)
    spy = SpyMotors()
    dispatcher = cuasp.MotorCommandDispatcher(spy)
    await dispatcher.start()
    try:
        correction = cuasp.CuasCorrection(reliable=False, reason="confidence_low")
        queued = dispatcher.submit(correction, frame_index=1, capture_ns=0)
        assert queued is False
        await asyncio.sleep(0.15)
        assert spy.calls == []
    finally:
        await dispatcher.stop()


async def test_dispatcher_now_uses_abs_rotate_against_fresh_encoder(cuasp, monkeypatch):
    """2026-09-21: this assertion flipped -- see suncubes/README.md:335 ("the
    system supervisor should not use [set_visual_error] as an automatic
    interpretation of every CameraCorrection") and control/README.md's
    "Preferred baseline command formulation". set_visual_error was
    rel_rotate_sync under the hood (motors.py), accumulating onto an internal
    position target with no restoring force to hardware truth; _send() now
    reads a fresh motors.get_encoder_positions() and sends one bounded
    abs_rotate(current + correction) instead, so a residual error converges
    to a bounded offset instead of accumulating without bound. Validated live
    on real hardware 2026-09-18, 3400+ frames, zero marker loss (see
    calibration/roughtest/README.md's "Mitigation 3 validation" section)
    before being promoted from the roughtest test harness into this real
    dispatcher."""
    monkeypatch.setattr(cuasp, "MOTOR_COMMAND_HZ", 20.0)
    spy = SpyMotors(encoder_positions=(10.0, -5.0))
    dispatcher = cuasp.MotorCommandDispatcher(spy)
    await dispatcher.start()
    try:
        correction = cuasp.CuasCorrection(reliable=True, reason="ok", d_az_motor_units=3.0, d_po_motor_units=-1.0)
        dispatcher.submit(correction, frame_index=1, capture_ns=0)
        await asyncio.sleep(0.15)
        assert len(spy.calls) == 1
        assert spy.calls[0] == ("abs_rotate", 13.0, -6.0), "target = fresh encoder + correction"
    finally:
        await dispatcher.stop()


async def test_dispatcher_stops_once_on_tracked_to_lost_transition(cuasp, monkeypatch):
    """control/README.md's cuasp.py contract item 12: stop issuing new visual
    commands on stale/unreliable camera data. Must fire exactly once on the
    transition, not on every subsequent unreliable frame."""
    monkeypatch.setattr(cuasp, "MOTOR_COMMAND_HZ", 20.0)
    spy = SpyMotors()
    dispatcher = cuasp.MotorCommandDispatcher(spy)
    await dispatcher.start()
    try:
        reliable = cuasp.CuasCorrection(reliable=True, reason="ok", d_az_motor_units=1.0, d_po_motor_units=0.0)
        dispatcher.submit(reliable, frame_index=1, capture_ns=0)
        await asyncio.sleep(0.1)

        lost = cuasp.CuasCorrection(reliable=False, reason="marker_lost")
        dispatcher.submit(lost, frame_index=2, capture_ns=1)
        dispatcher.submit(lost, frame_index=3, capture_ns=2)
        dispatcher.submit(lost, frame_index=4, capture_ns=3)
        await asyncio.sleep(0.1)

        assert spy.calls.count(("stop",)) == 1
    finally:
        await dispatcher.stop()


async def test_dispatcher_shutdown_sends_stop_and_closes_the_connection(cuasp, fake_pcmm_server):
    """control/README.md 'Tests to add': shutdown sends stop and closes
    resources. Uses the real PCMMMotorAdapterTCP against FakePCMMServer
    (not the spy) because the stop[0]=1 send happens inside
    motors.py's close_motors_sync(send_stop=True), reached via
    MotorCommandDispatcher._close_if_supported() -> disconnect_motors()."""
    motors = PCMMMotorAdapterTCP(
        fake_pcmm_server.host, fake_pcmm_server.port, connect_timeout_s=1.0, io_timeout_s=0.3,
    )
    dispatcher = cuasp.MotorCommandDispatcher(motors)
    await dispatcher.start()
    correction = cuasp.CuasCorrection(reliable=True, reason="ok", d_az_motor_units=1.0, d_po_motor_units=0.0)
    dispatcher.submit(correction, frame_index=1, capture_ns=0)
    await asyncio.sleep(0.15)
    assert dispatcher.status.connected is True

    await dispatcher.stop()

    assert "stop[0]=1" in fake_pcmm_server.received_lines
    assert dispatcher.status.connected is False
    assert motors.isOK is False
