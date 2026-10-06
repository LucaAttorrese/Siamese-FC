"""CUAS supervisor: CameraAPI target/correction -> PBVS -> PCMM/KAS motor dispatch.

No CLI is used by design: edit the variables in the CONFIGURATION section.

Pipeline (see suncubes/camera.py for the vision side):
- CameraAPI drives IDS acquisition, KLT tracking, asynchronous ArUco detection,
  per-marker PnP and (for 2+ configured markers) rigid multi-marker target-point
  fusion, producing a CameraCorrection (plain radians, no sign map applied).
- This module only supervises: fresh-sample consumption (gated on
  CameraCorrection.ts_capture_ns identity, see main()'s _is_new_visual_sample
  check), rad -> deg conversion, explicit axis/sign mapping, an explicit
  visual gain, transport clamping, and a latest-only motor command dispatcher
  talking to suncubes.motors.PCMMMotorAdapterTCP via one bounded absolute
  abs_rotate() reference per accepted sample (not accumulated rel_rotate()).

All computer-vision acquisition/KLT/ArUco/PnP/target-fusion logic lives in
suncubes.camera; do not duplicate it here (see the repository TODO history for
why this split exists).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import asyncio
import inspect
import math
import sys
import time
from typing import Any, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SUNCUBES_DIR = SCRIPT_DIR / "suncubes"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from suncubes import settings
from suncubes.motors import PCMMMotorAdapterTCP
from suncubes.camera import (
    CameraAPI,
    CameraConfig,
    CameraCorrection,
    GatingConfig,
    RuntimeConfig,
    _import_cv2_or_raise,
)

CV_CUAS_CODE_VERSION = "2026-09-21-cuasp-supervisor-v2"


# =============================================================================
# CONFIGURATION - edit here, no CLI
# =============================================================================

# Aruco markers mounted on the target drone. All configured markers share one
# dictionary and physical size (matching calibration.yaml's schema); the
# camera-side pipeline (suncubes.camera.MultiArucoCorrectionSourceAdapter)
# fuses 2+ markers into one rigid target point, or a single marker is tracked
# directly when only one id is configured.
ARUCO_IDS = tuple(int(v) for v in getattr(settings, "CV_ARUCO_TARGET_IDS", (1, 2)))
ARUCO_DICT_NAME = str(getattr(settings, "CV_ARUCO_DICT_NAME", "DICT_4X4_100"))
ARUCO_MARKER_SIZE_M = float(getattr(settings, "CV_ARUCO_MARKER_SIZE_M", 0.05))

# Camera / calibration.
# 2026-09-21: resolved repo-root-relative (was SUNCUBES_DIR-relative), so the
# real runtime calibration artifact (calibration/calibration.yaml, populated
# via calibration/finalize_runtime_calibration.py) can be referenced directly
# -- closes root README.md's "Known integration gaps" wiring item. The old
# fallback default is kept reachable at its true repo-root-relative location
# (it did not exist on disk under the old SUNCUBES_DIR-relative resolution
# either -- this was already a broken default, not a working one this change
# disturbs).
REPO_ROOT_DIR = SCRIPT_DIR.parent
# 2026-09-23: TEMPORARY override for the next test -- forces the legacy
# 2026-09-21 intrinsics (fx/fy~24119/24482) instead of
# settings.CV_CAMERA_CALIBRATION_YAML's current calibration/calibration.yaml
# (the 2026-09-23 recalibration, fx/fy~21062/21051), PLUS a hand-tuned test
# boresight: R_lc=identity (no rotation -- the earlier 0.4443deg rotation
# test was reverted at the operator's request) and
# t_lc_m=(-0.05,-0.18,0) -- the existing -18cm vertical lever arm plus a new
# -5cm lateral term (operator-tuned; see
# calibration/calibration_legacy_20260921_x_test.yaml's header comment for
# the sign convention). This is a hand-tuned guess for this one test, not
# a measurement -- it does NOT touch calibration/boresight.yaml. Revert to
# the line below once the comparison test is done:
#   CALIBRATION_YAML = REPO_ROOT_DIR / str(
#       getattr(
#           settings,
#           "CV_CAMERA_CALIBRATION_YAML",
#           "control/suncubes/visual_servoing_calibration_interpolated.yaml",
#       )
#   )
CALIBRATION_YAML = REPO_ROOT_DIR / "calibration/calibration_legacy_20260921_x_test.yaml"
IDS_DEVICE_INDEX = int(getattr(settings, "CV_IDS_DEVICE_INDEX", 0))
IDS_PIXEL_FORMAT = str(getattr(settings, "CV_IDS_PIXEL_FORMAT", "Mono8"))
IDS_WIDTH = int(getattr(settings, "CV_IDS_WIDTH", 2592))
IDS_HEIGHT = int(getattr(settings, "CV_IDS_HEIGHT", 1944))
IDS_BUFFER_TIMEOUT_MS = int(getattr(settings, "CV_IDS_BUFFER_TIMEOUT_MS", 100))
IDS_ACQUISITION_MODE = str(getattr(settings, "CV_IDS_ACQUISITION_MODE", "Continuous"))
IDS_EXPOSURE_AUTO = str(getattr(settings, "CV_IDS_EXPOSURE_AUTO", "Off"))
IDS_GAIN_AUTO = str(getattr(settings, "CV_IDS_GAIN_AUTO", "Off"))
IDS_EXPOSURE_TIME_US = float(getattr(settings, "CV_IDS_EXPOSURE_TIME_US", 5000.0))
IDS_GAIN_DB = float(getattr(settings, "CV_IDS_GAIN_DB", 0.0))
IDS_FRAME_RATE_TARGET_ENABLE = bool(int(getattr(settings, "CV_IDS_FRAME_RATE_TARGET_ENABLE", 1)))
IDS_FRAME_RATE_TARGET_HZ = float(getattr(settings, "CV_IDS_FRAME_RATE_TARGET_HZ", 70.0))
IDS_GENTL_CTI_PATH = str(getattr(settings, "CV_IDS_GENTL_CTI_PATH", ""))

# Gating applied by CameraAPI before a sample is exposed as reliable. Detection
# speed/ROI/KLT/target-fusion tuning (CV_CUAS_*, CV_ARUCO_SCOUT_*, etc.) is read
# directly by suncubes.camera's own config loaders using the same setting
# names as before; this module no longer needs to read or forward them.
CONFIDENCE_MIN = float(getattr(settings, "CV_ARUCO_CONFIDENCE_MIN", 0.005))
Z_MIN_M = float(getattr(settings, "CV_ARUCO_Z_MIN_M", 0.25))
MAX_AGE_S = float(getattr(settings, "CV_CUAS_MAX_AGE_S", 0.25))

# PBVS geometry-to-axis mapping. Python applies an explicit visual gain
# (PBVS_GAIN, control/README.md's "Preferred baseline command formulation":
# q_cmd_deg = q_base_deg + K_visual * axis_sign_map(error_deg)) and sends the
# resulting angular error [deg]; KAS performs deadband, filtering, saturation
# and motion conditioning on top of that.
PBVS_RAW_TO_CMD_AZ_SIGN = float(getattr(settings, "CV_PBVS_RAW_TO_CMD_AZ_SIGN", -1.0))
PBVS_RAW_TO_CMD_PO_SIGN = float(getattr(settings, "CV_PBVS_RAW_TO_CMD_PO_SIGN", -1.0))
POINTING_FROM_RAD_TO_CMD = float(getattr(settings, "POINTING_FROM_RAD_TO_CMD", 180.0 / math.pi))
POINTING_AZ_DIR_CMD = float(getattr(settings, "POINTING_AZ_DIR_CMD", -1.0))
POINTING_PO_DIR_CMD = float(getattr(settings, "POINTING_PO_DIR_CMD", -1.0))
PBVS_MAX_ERROR_DEG = float(getattr(settings, "CV_CUAS_PBVS_MAX_ERROR_DEG", 30.0))
PBVS_GAIN = float(getattr(settings, "CV_CUAS_PBVS_GAIN", 1.0))

# Runtime.
CUAS_LOOP_HZ = float(getattr(settings, "CV_CUAS_LOOP_HZ", IDS_FRAME_RATE_TARGET_HZ))
LOOP_PERIOD_S = 0.0 if CUAS_LOOP_HZ <= 0.0 else 1.0 / float(CUAS_LOOP_HZ)
RUN_DURATION_S = 0.0  # 0: run until Ctrl+C
MAX_FRAMES = 0  # 0: unlimited
PRINT_EVERY_S = 0.5
SHOW_DEBUG_WINDOW = True
DEBUG_WINDOW_NAME = "CV CUAS"
DEBUG_WINDOW_SCALE = float(getattr(settings, "CV_CUAS_DEBUG_DISPLAY_SCALE", 0.33))
DEBUG_DRAW_HZ = float(getattr(settings, "CV_CUAS_DEBUG_DRAW_HZ", 2.0))
DEBUG_WINDOW_SIZE = (1280, 820)

# Motor safety switch. The compute path is always active; commands are sent only
# when this is True. Motor actuation is intentionally slower than camera processing:
# Python sends angular errors; KAS computes velocity deterministically.
ENABLE_MOTOR_COMMANDS = bool(int(getattr(settings, "CV_CUAS_ENABLE_MOTOR_COMMANDS", 1)))
MOTOR_IP = str(getattr(settings, "IP", "192.168.0.109"))
MOTOR_PORT = int(getattr(settings, "PORT", 8123))
MOTOR_COMMAND_HZ = float(getattr(settings, "CV_CUAS_MOTOR_COMMAND_HZ", 30.0))
MOTOR_COMMAND_STALE_TIMEOUT_S = float(
    getattr(settings, "CV_CUAS_MOTOR_COMMAND_STALE_TIMEOUT_S", 0.25)
)
MOTOR_RECONNECT_INITIAL_S = float(
    getattr(settings, "CV_CUAS_MOTOR_RECONNECT_INITIAL_S", 0.5)
)
MOTOR_RECONNECT_MAX_S = float(
    getattr(settings, "CV_CUAS_MOTOR_RECONNECT_MAX_S", 5.0)
)
# 0.0 == disabled (default). See roughtest/README.md's "Root cause found"
# section: raising this on the test rig did not fix a real-hardware polar
# oscillation (confirmed autonomous KAS/mechanical limit cycle, not a
# resend-rate artifact) -- ships off pending a real, separately-validated
# tuning pass against actual tracking-latency requirements.
MOTOR_COMMAND_MIN_INTERVAL_S = float(
    getattr(settings, "CV_CUAS_MOTOR_COMMAND_MIN_INTERVAL_S", 0.0)
)


# =============================================================================
# Data contracts
# =============================================================================


@dataclass(frozen=True)
class CuasCorrection:
    reliable: bool
    reason: str
    d_az_raw_rad: float = 0.0
    d_po_raw_rad: float = 0.0
    d_az_cmd_rad: float = 0.0
    d_po_cmd_rad: float = 0.0
    d_az_motor_units: float = 0.0
    d_po_motor_units: float = 0.0
    az_clamped: bool = False
    po_clamped: bool = False


# =============================================================================
# Helpers
# =============================================================================


def _clamp(value: float, limit: float) -> tuple[float, bool]:
    if not np.isfinite(value):
        return 0.0, True
    if limit <= 0.0:
        return float(value), False
    out = float(np.clip(float(value), -float(limit), float(limit)))
    return out, bool(abs(out - float(value)) > 1e-12)


def _is_new_visual_sample(last_capture_ns: Optional[int], capture_ns: int) -> bool:
    """root README.md's "Non-negotiable interface semantics": a command
    update must be gated by new visual sample identity, not dispatched
    unconditionally on every poll. ts_capture_ns is the documented interim
    key (control/README.md's "Critical visual-command semantic") pending a
    real frame/sample sequence field on CameraCorrection -- note this only
    catches a background-worker cache literally returning the same object
    before a new frame is processed; the IDS sensor produces a genuinely new
    ts_capture_ns roughly every ~14ms regardless of scene content (see
    control/suncubes/README.md's "Thread 2" description), so this does not
    by itself throttle a stationary-target correction rate.
    """
    return last_capture_ns is None or int(capture_ns) != int(last_capture_ns)


def _elapsed_ms(t0_ns: int, t1_ns: int) -> float:
    return (float(t1_ns) - float(t0_ns)) / 1e6


def _format_finite(value: Optional[float], suffix: str = "", precision: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        out = float(value)
    except Exception:
        return "N/A"
    if not np.isfinite(out):
        return "N/A"
    return f"{out:.{int(precision)}f}{suffix}"


def _period_s(hz: float) -> float:
    return 0.0 if hz <= 0.0 else 1.0 / float(hz)


# =============================================================================
# PBVS: rad -> deg, sign map, transport clamp (supervisor-owned, not camera.py)
# =============================================================================


def compute_pbvs_correction_from_camera(correction: CameraCorrection) -> CuasCorrection:
    """Convert an already-gated CameraCorrection into a signed, clamped PBVS command.

    All reliability/confidence/z/age gating already happened inside CameraAPI
    (see suncubes.camera's CorrectionCandidate validation); this function only
    performs the supervisor's own responsibilities: rad -> deg exactly once,
    the explicit axis/sign map, and the explicit visual gain PBVS_GAIN
    (control/README.md's "cuasp.py contract" / "Preferred baseline command
    formulation": q_cmd_deg = q_base_deg + K_visual * axis_sign_map(error_deg)).
    Gain is applied before the transport clamp below, so PBVS_MAX_ERROR_DEG
    stays the authoritative safety bound for any gain value, not just <= 1.0.
    """
    if not correction.reliable:
        return CuasCorrection(reliable=False, reason=correction.reason)

    d_az_raw = float(correction.d_az_rad)
    d_po_raw = float(correction.d_po_rad)
    if not np.isfinite(d_az_raw) or not np.isfinite(d_po_raw):
        return CuasCorrection(reliable=False, reason="pbvs_angles_not_finite")

    d_az_axis_rad = float(PBVS_RAW_TO_CMD_AZ_SIGN * POINTING_AZ_DIR_CMD * d_az_raw)
    d_po_axis_rad = float(PBVS_RAW_TO_CMD_PO_SIGN * POINTING_PO_DIR_CMD * d_po_raw)
    d_az_error_deg = float(d_az_axis_rad * POINTING_FROM_RAD_TO_CMD * PBVS_GAIN)
    d_po_error_deg = float(d_po_axis_rad * POINTING_FROM_RAD_TO_CMD * PBVS_GAIN)

    # Very broad transport sanity clamp only; normal control saturation is KAS.
    max_error_deg = max(0.1, float(PBVS_MAX_ERROR_DEG))
    d_az_error_limited, az_clamped = _clamp(d_az_error_deg, max_error_deg)
    d_po_error_limited, po_clamped = _clamp(d_po_error_deg, max_error_deg)

    return CuasCorrection(
        reliable=True,
        reason="ok",
        d_az_raw_rad=d_az_raw,
        d_po_raw_rad=d_po_raw,
        d_az_cmd_rad=math.radians(float(d_az_error_limited)),
        d_po_cmd_rad=math.radians(float(d_po_error_limited)),
        d_az_motor_units=float(d_az_error_limited),
        d_po_motor_units=float(d_po_error_limited),
        az_clamped=bool(az_clamped),
        po_clamped=bool(po_clamped),
    )


# =============================================================================
# Debug window
# =============================================================================


def _draw_cross(cv2_mod, img, x: int, y: int, size: int, color, thickness: int) -> None:
    cv2_mod.line(img, (x - size, y), (x + size, y), color, thickness, cv2_mod.LINE_AA)
    cv2_mod.line(img, (x, y - size), (x, y + size), color, thickness, cv2_mod.LINE_AA)


def _draw_debug_frame(
    cv2_mod,
    correction: CameraCorrection,
    pbvs: CuasCorrection,
    snapshot: dict[str, Any],
    frame_idx: int,
    processed_fps_hz: Optional[float],
    motor_status: Optional["MotorRuntimeStatus"],
) -> bool:
    if not snapshot.get("available"):
        return False
    gray = snapshot.get("frame_gray")
    if gray is None:
        return False
    vis = cv2_mod.cvtColor(np.asarray(gray), cv2_mod.COLOR_GRAY2BGR) if np.asarray(gray).ndim == 2 else np.asarray(gray).copy()
    h, w = vis.shape[:2]

    marker_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for idx, marker in enumerate(snapshot.get("markers") or []):
        pts = np.asarray(marker["corners_px"], dtype=np.float64).reshape(4, 2).astype(int)
        color = marker_colors[idx % len(marker_colors)]
        cv2_mod.polylines(vis, [pts], True, color, 2, cv2_mod.LINE_AA)
        center = np.mean(pts, axis=0).astype(int)
        cv2_mod.circle(vis, (int(center[0]), int(center[1])), 5, color, -1, cv2_mod.LINE_AA)
        cv2_mod.putText(
            vis,
            f"id={marker['marker_id']} conf={marker['confidence']:.2f} Z={marker['z_m']:.3f}m",
            (int(center[0]) + 8, int(center[1]) - 8),
            cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2_mod.LINE_AA,
        )

    # Pointing references, drawn under the PBVS target marker: the frame's own
    # geometric center, the calibration's optical axis (principal point -- these
    # two are NOT the same point and can sit far apart), and where the loaded
    # calibration's R_lc/t_lc_m put the laser beam (camera.py's
    # _laser_axis_point_in_camera; its offset from the optical axis is the
    # camera/laser lever-arm parallax, hence the range it was projected at).
    _draw_cross(cv2_mod, vis, w // 2, h // 2, 14, (255, 255, 255), 1)
    cv2_mod.putText(vis, "frame center", (w // 2 + 10, h // 2 + 20),
                     cv2_mod.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2_mod.LINE_AA)

    principal_point = snapshot.get("principal_point_px")
    if principal_point is not None:
        pp = np.asarray(principal_point, dtype=np.float64).reshape(2)
        if np.all(np.isfinite(pp)):
            ppx, ppy = int(round(pp[0])), int(round(pp[1]))
            if 0 <= ppx < w and 0 <= ppy < h:
                _draw_cross(cv2_mod, vis, ppx, ppy, 18, (0, 165, 255), 2)
                cv2_mod.putText(vis, "optical axis", (ppx + 12, ppy - 12),
                                 cv2_mod.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2_mod.LINE_AA)

    laser_pixel = snapshot.get("laser_axis_pixel_px")
    if laser_pixel is not None:
        lp = np.asarray(laser_pixel, dtype=np.float64).reshape(2)
        if np.all(np.isfinite(lp)):
            lx, ly = int(round(lp[0])), int(round(lp[1]))
            inside = 0 <= lx < w and 0 <= ly < h
            draw_lx = int(np.clip(lx, 0, max(0, w - 1)))
            draw_ly = int(np.clip(ly, 0, max(0, h - 1)))
            cv2_mod.circle(vis, (draw_lx, draw_ly), 10, (0, 0, 255), 2, cv2_mod.LINE_AA)
            _draw_cross(cv2_mod, vis, draw_lx, draw_ly, 16, (0, 0, 255), 1)
            label = "laser (calib) {}@{} m{}".format(
                "" if snapshot.get("laser_axis_range_source") == "target" else "nominal ",
                _format_finite(snapshot.get("laser_axis_range_m"), precision=2),
                "" if inside else " OUT",
            )
            cv2_mod.putText(vis, label, (min(draw_lx + 14, max(10, w - 260)), max(20, draw_ly + 26)),
                             cv2_mod.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2_mod.LINE_AA)

    target_pixel = snapshot.get("pbvs_target_pixel_px")
    if target_pixel is not None:
        tp = np.asarray(target_pixel, dtype=np.float64).reshape(2)
        if np.all(np.isfinite(tp)):
            tx, ty = int(round(tp[0])), int(round(tp[1]))
            inside = 0 <= tx < w and 0 <= ty < h
            draw_x, draw_y = int(np.clip(tx, 0, max(0, w - 1))), int(np.clip(ty, 0, max(0, h - 1)))
            cv2_mod.circle(vis, (draw_x, draw_y), 8, (255, 0, 0), -1, cv2_mod.LINE_AA)
            cv2_mod.circle(vis, (draw_x, draw_y), 12, (255, 255, 255), 2, cv2_mod.LINE_AA)
            cv2_mod.putText(
                vis, "PBVS target" if inside else "PBVS target OUT",
                (min(draw_x + 12, max(10, w - 180)), max(20, draw_y - 10)),
                cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2_mod.LINE_AA,
            )

    cv2_mod.putText(vis, "Press q to quit", (max(10, w - 180), 28),
                     cv2_mod.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2_mod.LINE_AA)

    y, step = 28, 24
    expected_ids = tuple(snapshot.get("expected_marker_ids") or ())
    cv2_mod.putText(
        vis,
        f"frame={frame_idx} obs={len(snapshot.get('markers') or [])}/{len(expected_ids)} "
        f"processed_fps={_format_finite(processed_fps_hz, precision=1)} "
        f"tile={snapshot.get('tile_state')} backend={snapshot.get('tracking_backend')}",
        (10, y), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2_mod.LINE_AA,
    )
    y += step

    target_point = snapshot.get("target_point")
    if target_point is not None:
        p_c = target_point["position_camera_m"]
        cv2_mod.putText(
            vis,
            "PBVS target camera [m] X={:+.3f} Y={:+.3f} Z={:+.3f} ids={} mode={}".format(
                float(p_c[0]), float(p_c[1]), float(p_c[2]),
                target_point["contributing_markers"], target_point["mode"],
            ),
            (10, y), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 0), 2, cv2_mod.LINE_AA,
        )
    else:
        cv2_mod.putText(vis, "PBVS target camera [m]: N/A", (10, y),
                         cv2_mod.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2_mod.LINE_AA)
    y += step

    cv2_mod.putText(
        vis,
        f"camera reason={correction.reason} conf={_format_finite(correction.confidence, precision=3)} "
        f"z={_format_finite(correction.z_m, ' m', 3)} age={_format_finite(correction.age_s, ' s', 3)}",
        (10, y), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2, cv2_mod.LINE_AA,
    )
    y += step

    cv2_mod.putText(
        vis,
        f"PBVS ok={int(pbvs.reliable)} dAz={pbvs.d_az_cmd_rad:+.5f} dPo={pbvs.d_po_cmd_rad:+.5f} {pbvs.reason}",
        (10, y), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.58,
        (0, 255, 0) if pbvs.reliable else (0, 0, 255), 2, cv2_mod.LINE_AA,
    )
    y += step

    if motor_status is not None:
        cv2_mod.putText(
            vis,
            f"motor_conn={int(motor_status.connected)} sent={motor_status.sent_count} "
            f"state={motor_status.last_reason}",
            (10, y), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 0), 2, cv2_mod.LINE_AA,
        )

    draw_scale = float(DEBUG_WINDOW_SCALE) if DEBUG_WINDOW_SCALE > 0.0 else 1.0
    if abs(draw_scale - 1.0) > 1e-6:
        vis = cv2_mod.resize(
            vis, (max(1, int(round(w * draw_scale))), max(1, int(round(h * draw_scale)))),
            interpolation=cv2_mod.INTER_AREA,
        )

    cv2_mod.imshow(DEBUG_WINDOW_NAME, vis)
    key = cv2_mod.waitKey(1) & 0xFF
    return key in (ord("q"), 27)


# =============================================================================
# Motor dispatch (unchanged by the camera.py migration)
# =============================================================================


@dataclass(frozen=True)
class MotorCommandRequest:
    correction: Optional[CuasCorrection]
    frame_index: int
    source_capture_ns: int
    created_monotonic_ns: int
    reason: str


@dataclass
class MotorRuntimeStatus:
    enabled: bool
    connected: bool = False
    queued_count: int = 0
    sent_count: int = 0
    dropped_count: int = 0
    replaced_count: int = 0
    stale_count: int = 0
    error_count: int = 0
    last_queued_frame: int = -1
    last_sent_frame: int = -1
    last_reason: str = "disabled"
    last_error: str = ""
    last_send_monotonic_ns: int = 0


class LatestMotorCommandQueue:
    """Single-slot queue: a newer visual command always replaces an older one."""

    def __init__(self):
        self._queue: asyncio.Queue[MotorCommandRequest] = asyncio.Queue(maxsize=1)
        self.dropped_count = 0

    def put_latest(self, request: MotorCommandRequest) -> None:
        while self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped_count += 1
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(request)

    async def get(self) -> MotorCommandRequest:
        return await self._queue.get()

    def get_nowait(self) -> MotorCommandRequest:
        return self._queue.get_nowait()

    def drain_latest(self, current: MotorCommandRequest) -> MotorCommandRequest:
        latest = current
        while True:
            try:
                latest = self._queue.get_nowait()
                self.dropped_count += 1
            except asyncio.QueueEmpty:
                return latest


async def _call_method_maybe_async(
    method, *args, run_sync_in_thread: bool = False, **kwargs
):
    """Call APIs that may be either synchronous or asynchronous.

    suncubes motor APIs may be coroutine or synchronous, while connect_motors()
    is synchronous. This adapter supports both without silently losing a coroutine.
    """
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)

    if run_sync_in_thread:
        result = await asyncio.to_thread(method, *args, **kwargs)
    else:
        result = method(*args, **kwargs)

    if inspect.isawaitable(result):
        return await result
    return result


class MotorCommandDispatcher:
    """Latest-only PBVS error dispatcher; timing and velocity are computed in PCMM."""

    def __init__(self, motors: PCMMMotorAdapterTCP):
        self.motors = motors
        self.status = MotorRuntimeStatus(enabled=True, last_reason="starting")
        self._queue = LatestMotorCommandQueue()
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._next_reconnect_s = 0.0
        self._reconnect_delay_s = max(0.05, float(MOTOR_RECONNECT_INITIAL_S))
        self._last_actual_send_monotonic_s: Optional[float] = None
        # False, not True: a stop-on-loss must fire on a genuine tracked ->
        # lost transition, not merely because the very first-ever submission
        # happens to be unreliable (nothing was ever actuated to stop yet).
        self._last_submit_valid = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="cuas-motor-dispatcher")
        await asyncio.sleep(0)

    def submit(self, correction: CuasCorrection, frame_index: int, capture_ns: int) -> bool:
        finite = bool(
            np.isfinite(correction.d_az_motor_units)
            and np.isfinite(correction.d_po_motor_units)
        )
        # Zero angular error is a valid sample: KAS deadband/stop logic needs it.
        valid = bool(correction.reliable and finite)
        reason = "queued" if valid else correction.reason
        request = MotorCommandRequest(
            correction=correction if valid else None,
            frame_index=int(frame_index),
            source_capture_ns=int(capture_ns),
            created_monotonic_ns=time.monotonic_ns(),
            reason=str(reason),
        )
        self._queue.put_latest(request)
        self.status.replaced_count = int(self._queue.dropped_count)
        self.status.dropped_count = self.status.replaced_count + self.status.stale_count
        self.status.last_queued_frame = int(frame_index)
        self.status.last_reason = str(reason)
        if valid:
            self.status.queued_count += 1
        # control/README.md's cuasp.py contract item 12: "stop issuing new
        # visual commands on stale/unreliable camera data." Fires exactly
        # once on the tracked -> lost transition (not on every subsequent
        # unreliable frame), and also resets motors.py's client-side
        # absolute-position accumulator so re-acquisition re-seeds from a
        # fresh encoder read. Validated live on real hardware 2026-09-18 (see
        # calibration/roughtest/README.md).
        if self._last_submit_valid and not valid:
            print(f"[MOTOR] marker lost (reason={correction.reason}) -> stop[0]=1")
            asyncio.create_task(
                _call_method_maybe_async(self.motors.stop, run_sync_in_thread=True)
            )
        self._last_submit_valid = valid
        return valid

    async def stop(self) -> None:
        self._stop_event.set()
        self._queue.put_latest(
            MotorCommandRequest(
                correction=None,
                frame_index=-1,
                source_capture_ns=0,
                created_monotonic_ns=time.monotonic_ns(),
                reason="shutdown",
            )
        )
        if self._task is not None:
            try:
                await self._task
            finally:
                self._task = None
        await self._close_if_supported()

    async def _connect(self) -> bool:
        now_s = time.monotonic()
        if self.status.connected:
            return True
        if now_s < self._next_reconnect_s:
            return False

        try:
            result = await _call_method_maybe_async(
                self.motors.connect_motors,
                run_sync_in_thread=True,
            )
            if result is False:
                raise ConnectionError("connect_motors returned False")
            self.status.connected = True
            self.status.last_error = ""
            self.status.last_reason = "connected"
            self._reconnect_delay_s = max(0.05, float(MOTOR_RECONNECT_INITIAL_S))
            print(f"[MOTOR] connected to {MOTOR_IP}:{MOTOR_PORT}")
            return True
        except Exception as exc:
            self.status.connected = False
            self.status.error_count += 1
            message = f"{type(exc).__name__}: {exc}"
            if message != self.status.last_error:
                print(f"[MOTOR] connection failed: {message}")
            self.status.last_error = message
            self.status.last_reason = "connect_failed"
            self._next_reconnect_s = now_s + self._reconnect_delay_s
            self._reconnect_delay_s = min(
                max(self._reconnect_delay_s * 2.0, 0.05),
                max(float(MOTOR_RECONNECT_MAX_S), 0.05),
            )
            return False

    async def _send(self, request: MotorCommandRequest) -> bool:
        correction = request.correction
        if correction is None:
            return False

        # Mitigation 4: optional minimum wall-clock gap between actual
        # hardware sends, on top of MOTOR_COMMAND_HZ's own period. Disabled
        # (0.0) by default -- see MOTOR_COMMAND_MIN_INTERVAL_S's definition.
        if MOTOR_COMMAND_MIN_INTERVAL_S > 0.0:
            now_s = time.monotonic()
            last_sent_s = self._last_actual_send_monotonic_s
            if last_sent_s is not None and (now_s - last_sent_s) < MOTOR_COMMAND_MIN_INTERVAL_S:
                self.status.last_reason = "rate_limited_skip"
                return False

        if not await self._connect():
            return False

        try:
            # Mitigation 3: one bounded abs_rotate() against a fresh encoder
            # read, instead of motors.set_visual_error() (a rel_rotate()
            # alias that accumulates onto motors.py's own internal position
            # target with no restoring force to hardware truth -- see
            # suncubes/README.md:335, "the system supervisor should not use
            # [set_visual_error] as an automatic interpretation of every
            # CameraCorrection", and control/README.md's "Preferred baseline
            # command formulation": q_cmd_deg = q_base_deg + K_visual *
            # axis_sign_map(error_deg)). Grounding q_base_deg in the real
            # encoder every send means per-cycle noise or a residual
            # boresight bias converges to a bounded offset instead of
            # accumulating without bound. Trade-off: one extra TCP
            # round-trip (enc[0,1]?) per send, inside the same
            # 1/MOTOR_COMMAND_HZ budget as the send itself -- validated live
            # on real hardware 2026-09-18, 3400+ frames, zero marker loss
            # (see calibration/roughtest/README.md's "Mitigation 3
            # validation" section).
            current_az, current_po = await _call_method_maybe_async(
                self.motors.get_encoder_positions,
                run_sync_in_thread=True,
            )
            target_az = float(current_az) + float(correction.d_az_motor_units)
            target_po = float(current_po) + float(correction.d_po_motor_units)
            await _call_method_maybe_async(
                self.motors.abs_rotate,
                target_az,
                target_po,
                run_sync_in_thread=True,
            )
            self.status.sent_count += 1
            self.status.last_sent_frame = int(request.frame_index)
            self.status.last_send_monotonic_ns = time.monotonic_ns()
            self.status.last_reason = "sent"
            self.status.last_error = ""
            self._last_actual_send_monotonic_s = time.monotonic()
            return True
        except Exception as exc:
            self.status.connected = False
            self.status.error_count += 1
            self.status.last_reason = "send_failed"
            self.status.last_error = f"{type(exc).__name__}: {exc}"
            self._next_reconnect_s = time.monotonic() + max(
                0.05, float(MOTOR_RECONNECT_INITIAL_S)
            )
            print(f"[MOTOR] send failed: {self.status.last_error}")
            return False

    async def _run(self) -> None:
        command_period_s = 1.0 / max(float(MOTOR_COMMAND_HZ), 1e-6)
        next_send_s = time.monotonic()
        pending: Optional[MotorCommandRequest] = None

        await self._connect()

        while not self._stop_event.is_set():
            now_s = time.monotonic()
            if pending is None:
                timeout_s = 0.10
            else:
                timeout_s = max(0.0, min(0.10, next_send_s - now_s))

            try:
                if timeout_s <= 0.0:
                    request = self._queue.get_nowait()
                else:
                    request = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
                pending = self._queue.drain_latest(request)
                self.status.replaced_count = int(self._queue.dropped_count)
                self.status.dropped_count = self.status.replaced_count + self.status.stale_count
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                pass

            # Always consume a just-arrived invalidation/newer command before a
            # relative move. This prevents a race at the rate-limit boundary.
            if pending is not None:
                try:
                    request = self._queue.get_nowait()
                    pending = self._queue.drain_latest(request)
                    self.status.replaced_count = int(self._queue.dropped_count)
                    self.status.dropped_count = self.status.replaced_count + self.status.stale_count
                except asyncio.QueueEmpty:
                    pass

            if self._stop_event.is_set():
                break

            if not self.status.connected and time.monotonic() >= self._next_reconnect_s:
                await self._connect()

            if pending is None:
                continue

            # An invalid frame explicitly cancels a previously pending command.
            if pending.correction is None:
                self.status.last_reason = str(pending.reason)
                pending = None
                continue

            now_s = time.monotonic()
            if now_s < next_send_s:
                continue

            age_s = (time.monotonic_ns() - pending.created_monotonic_ns) * 1e-9
            if age_s > float(MOTOR_COMMAND_STALE_TIMEOUT_S):
                self.status.stale_count += 1
                self.status.dropped_count = self.status.replaced_count + self.status.stale_count
                self.status.last_reason = "stale_command_dropped"
                pending = None
                next_send_s += command_period_s
                if next_send_s <= now_s:
                    next_send_s = now_s + command_period_s
                continue

            await self._send(pending)
            pending = None
            next_send_s += command_period_s
            now_after_send_s = time.monotonic()
            if next_send_s <= now_after_send_s:
                missed = math.floor((now_after_send_s - next_send_s) / command_period_s) + 1
                next_send_s += missed * command_period_s

    async def _close_if_supported(self) -> None:
        for name in ("disconnect_motors", "disconnect", "close"):
            method = getattr(self.motors, name, None)
            if method is None or not callable(method):
                continue
            try:
                await _call_method_maybe_async(method, run_sync_in_thread=True)
            except Exception:
                pass
            break
        self.status.connected = False


def _validate_configuration() -> None:
    errors = []
    try:
        settings.validate()
    except ValueError as exc:
        errors.append(str(exc))
    if not ARUCO_IDS:
        errors.append("ARUCO_IDS must be non-empty")
    if len(set(ARUCO_IDS)) != len(ARUCO_IDS):
        errors.append("ARUCO_IDS must not contain duplicates")
    if ARUCO_MARKER_SIZE_M <= 0.0:
        errors.append("ARUCO_MARKER_SIZE_M must be > 0")
    if not np.isfinite(CONFIDENCE_MIN) or CONFIDENCE_MIN < 0.0:
        errors.append("CONFIDENCE_MIN must be finite and >= 0")
    if not np.isfinite(Z_MIN_M) or Z_MIN_M <= 0.0:
        errors.append("Z_MIN_M must be finite and > 0")
    if not np.isfinite(MAX_AGE_S) or MAX_AGE_S <= 0.0:
        errors.append("MAX_AGE_S must be finite and > 0")
    if not np.isfinite(PBVS_MAX_ERROR_DEG) or PBVS_MAX_ERROR_DEG <= 0.0:
        errors.append("PBVS_MAX_ERROR_DEG must be finite and > 0")
    if not np.isfinite(PBVS_GAIN) or PBVS_GAIN <= 0.0:
        errors.append("PBVS_GAIN must be finite and > 0")
    if not np.isfinite(MOTOR_COMMAND_MIN_INTERVAL_S) or MOTOR_COMMAND_MIN_INTERVAL_S < 0.0:
        errors.append("MOTOR_COMMAND_MIN_INTERVAL_S must be finite and >= 0")
    if not np.isfinite(POINTING_FROM_RAD_TO_CMD) or abs(POINTING_FROM_RAD_TO_CMD) <= 1e-12:
        errors.append("POINTING_FROM_RAD_TO_CMD must be finite and non-zero")
    for name, value in (
        ("PBVS_RAW_TO_CMD_AZ_SIGN", PBVS_RAW_TO_CMD_AZ_SIGN),
        ("PBVS_RAW_TO_CMD_PO_SIGN", PBVS_RAW_TO_CMD_PO_SIGN),
        ("POINTING_AZ_DIR_CMD", POINTING_AZ_DIR_CMD),
        ("POINTING_PO_DIR_CMD", POINTING_PO_DIR_CMD),
    ):
        if not np.isfinite(value) or abs(float(value)) <= 1e-12:
            errors.append(f"{name} must be finite and non-zero")
    if ENABLE_MOTOR_COMMANDS:
        if MOTOR_COMMAND_HZ <= 0.0:
            errors.append("MOTOR_COMMAND_HZ must be > 0")
        if MOTOR_COMMAND_STALE_TIMEOUT_S <= 0.0:
            errors.append("MOTOR_COMMAND_STALE_TIMEOUT_S must be > 0")
        if not MOTOR_IP:
            errors.append("MOTOR_IP cannot be empty")
        if MOTOR_PORT <= 0 or MOTOR_PORT > 65535:
            errors.append("MOTOR_PORT must be in [1, 65535]")
    if errors:
        raise ValueError("Invalid CUAS configuration:\n- " + "\n- ".join(errors))


def _build_camera_config() -> CameraConfig:
    ids_cfg: dict[str, Any] = {
        "device_index": IDS_DEVICE_INDEX,
        "pixel_format": IDS_PIXEL_FORMAT,
        "width": IDS_WIDTH,
        "height": IDS_HEIGHT,
        "buffer_timeout_ms": IDS_BUFFER_TIMEOUT_MS,
        "acquisition_mode": IDS_ACQUISITION_MODE,
        "exposure_auto": IDS_EXPOSURE_AUTO,
        "gain_auto": IDS_GAIN_AUTO,
        "exposure_time_us": IDS_EXPOSURE_TIME_US,
        "gain_db": IDS_GAIN_DB,
    }
    if IDS_FRAME_RATE_TARGET_ENABLE and IDS_FRAME_RATE_TARGET_HZ > 0.0:
        ids_cfg["frame_rate_target_hz"] = IDS_FRAME_RATE_TARGET_HZ
    if IDS_GENTL_CTI_PATH:
        ids_cfg["gentl_cti_path"] = IDS_GENTL_CTI_PATH

    aruco_cfg = {
        "dict_name": ARUCO_DICT_NAME,
        "marker_size_m": ARUCO_MARKER_SIZE_M,
        "target_marker_ids": list(ARUCO_IDS),
    }

    return CameraConfig(
        enable=True,
        mode="ids_opencv",
        gating=GatingConfig(
            max_age_s=MAX_AGE_S,
            conf_min=CONFIDENCE_MIN,
            z_min_m=Z_MIN_M,
            max_abs_angle_rad=0.0,
        ),
        runtime=RuntimeConfig(background_worker=True, worker_hz=max(CUAS_LOOP_HZ, 1.0)),
        ids_opencv={
            "ids_peak": ids_cfg,
            "aruco": aruco_cfg,
            "calibration": {"path": str(CALIBRATION_YAML)},
            "debug_view": bool(SHOW_DEBUG_WINDOW),
        },
    )


# =============================================================================
# Runtime entry point
# =============================================================================


async def main() -> None:
    _validate_configuration()

    print(f"=== CV CUAS supervisor [{CV_CUAS_CODE_VERSION}] ===")
    print(f"markers: ids={ARUCO_IDS} dict={ARUCO_DICT_NAME} size_m={ARUCO_MARKER_SIZE_M}")
    print(f"calibration: {CALIBRATION_YAML}")
    print(
        "gating: "
        f"max_age_s={MAX_AGE_S:.3f} confidence_min={CONFIDENCE_MIN:.4f} z_min_m={Z_MIN_M:.3f}"
    )
    print(
        "pbvs: geometry_on_camera_py sign_map_and_clamp_on_cuasp "
        f"transport_error_limit={PBVS_MAX_ERROR_DEG:.2f}deg "
        f"effective_sign_az={PBVS_RAW_TO_CMD_AZ_SIGN * POINTING_AZ_DIR_CMD:+.0f} "
        f"effective_sign_po={PBVS_RAW_TO_CMD_PO_SIGN * POINTING_PO_DIR_CMD:+.0f}"
    )
    print(
        "motor commands: "
        f"{'ENABLED' if ENABLE_MOTOR_COMMANDS else 'DISABLED'} "
        f"endpoint={MOTOR_IP}:{MOTOR_PORT} rate={MOTOR_COMMAND_HZ:.1f}Hz "
        f"stale_timeout={MOTOR_COMMAND_STALE_TIMEOUT_S:.3f}s"
    )

    camera_cfg = _build_camera_config()
    camera_api = CameraAPI(camera_cfg)
    cv2_mod = _import_cv2_or_raise()

    motor_dispatcher: Optional[MotorCommandDispatcher] = None
    if ENABLE_MOTOR_COMMANDS:
        motors = PCMMMotorAdapterTCP(
            MOTOR_IP,
            MOTOR_PORT,
            reference_period_s=1.0 / max(float(MOTOR_COMMAND_HZ), 1e-6),
            connect_timeout_s=1.0,
            io_timeout_s=0.25,
        )
        motor_dispatcher = MotorCommandDispatcher(motors)
        await motor_dispatcher.start()

    camera_started = False
    try:
        await asyncio.to_thread(camera_api.start)
        camera_started = True

        if SHOW_DEBUG_WINDOW:
            cv2_mod.namedWindow(DEBUG_WINDOW_NAME, cv2_mod.WINDOW_NORMAL)
            cv2_mod.resizeWindow(DEBUG_WINDOW_NAME, int(DEBUG_WINDOW_SIZE[0]), int(DEBUG_WINDOW_SIZE[1]))

        start_s = time.monotonic()
        last_print_s = 0.0
        last_debug_s = 0.0
        last_capture_ns: Optional[int] = None
        last_submitted_capture_ns: Optional[int] = None
        processed_fps_hz: Optional[float] = None
        frame_idx = 0

        while True:
            now_s = time.monotonic()
            if RUN_DURATION_S > 0.0 and (now_s - start_s) >= float(RUN_DURATION_S):
                break
            if MAX_FRAMES > 0 and frame_idx >= int(MAX_FRAMES):
                break

            t_algo0 = time.monotonic_ns()
            correction = camera_api.get_latest()
            pbvs = compute_pbvs_correction_from_camera(correction)
            algorithm_update_ms = _elapsed_ms(t_algo0, time.monotonic_ns())

            capture_ns = correction.ts_capture_ns if correction.ts_capture_ns is not None else time.monotonic_ns()
            if last_capture_ns is not None and capture_ns != last_capture_ns:
                dt_capture_s = (int(capture_ns) - int(last_capture_ns)) * 1e-9
                if dt_capture_s > 1e-6:
                    fps_inst = 1.0 / dt_capture_s
                    processed_fps_hz = (
                        float(fps_inst)
                        if processed_fps_hz is None or not np.isfinite(float(processed_fps_hz))
                        else 0.20 * float(fps_inst) + 0.80 * float(processed_fps_hz)
                    )
            last_capture_ns = capture_ns

            # root README.md's "Non-negotiable interface semantics": gate
            # dispatch on the visual sample actually being new, rather than
            # submitting unconditionally on every ~14ms loop tick regardless
            # of whether camera_api.get_latest() returned the same
            # underlying measurement again.
            command_queued = False
            if motor_dispatcher is not None and _is_new_visual_sample(last_submitted_capture_ns, capture_ns):
                command_queued = motor_dispatcher.submit(pbvs, frame_index=frame_idx, capture_ns=capture_ns)
                last_submitted_capture_ns = capture_ns

            now_s = time.monotonic()
            if PRINT_EVERY_S > 0.0 and (now_s - last_print_s) >= float(PRINT_EVERY_S):
                last_print_s = now_s
                if motor_dispatcher is None:
                    motor_txt = "motor=disabled"
                else:
                    ms = motor_dispatcher.status
                    motor_txt = (
                        f"motor_conn={int(ms.connected)} queued={int(command_queued)} "
                        f"sent_total={ms.sent_count} last_sent_frame={ms.last_sent_frame} "
                        f"replaced={ms.replaced_count} stale={ms.stale_count} "
                        f"motor_state={ms.last_reason}"
                    )
                    if ms.last_error:
                        motor_txt += f" motor_error={ms.last_error}"

                print(
                    f"[CUAS] frame={frame_idx} camera_reason={correction.reason} "
                    f"processed_fps={_format_finite(processed_fps_hz, precision=1)} "
                    f"algo_ms={algorithm_update_ms:.1f} "
                    f"conf={_format_finite(correction.confidence, precision=3)} "
                    f"z={_format_finite(correction.z_m, ' m', 3)} "
                    f"pbvs_ok={int(pbvs.reliable)} reason={pbvs.reason} "
                    f"dAz={pbvs.d_az_cmd_rad:+.6f} dPo={pbvs.d_po_cmd_rad:+.6f} "
                    f"axis_error_deg=({pbvs.d_az_motor_units:+.5f},{pbvs.d_po_motor_units:+.5f}) "
                    f"{motor_txt}"
                )

            if SHOW_DEBUG_WINDOW:
                debug_period_s = _period_s(DEBUG_DRAW_HZ)
                if debug_period_s <= 0.0 or now_s - last_debug_s >= debug_period_s:
                    last_debug_s = now_s
                    snapshot = camera_api.get_debug_snapshot()
                    stop = _draw_debug_frame(
                        cv2_mod, correction, pbvs, snapshot, frame_idx, processed_fps_hz,
                        motor_dispatcher.status if motor_dispatcher is not None else None,
                    )
                else:
                    key = cv2_mod.waitKey(1) & 0xFF
                    stop = key in (ord("q"), 27)
                if stop:
                    break

            frame_idx += 1
            sleep_s = LOOP_PERIOD_S - (time.monotonic() - now_s) if LOOP_PERIOD_S > 0.0 else 0.0
            await asyncio.sleep(max(0.0, sleep_s))

    except asyncio.CancelledError:
        print("\n[CUAS] cancellation requested.")
        raise
    finally:
        if motor_dispatcher is not None:
            await motor_dispatcher.stop()

        if camera_started:
            try:
                await asyncio.to_thread(camera_api.stop)
            except Exception as exc:
                print(f"[CUAS] camera stop warning: {type(exc).__name__}: {exc}")

        if SHOW_DEBUG_WINDOW:
            try:
                cv2_mod.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CUAS] interrupted by user.")
