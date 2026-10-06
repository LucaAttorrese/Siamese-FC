"""suncubes.settings: engineer-editable runtime settings.

Single, authoritative configuration source for both `control/cuasp.py` and
`control/suncubes/camera.py`. Every value below is read dynamically via
`getattr(settings, "NAME", default)` (see `camera.py`'s `_setting()`/
`_bool_setting()`/`_setting_vec3()` and `cuasp.py`'s module-level constants) --
so every name here matches the effective default that was previously baked
into source as the fallback, meaning this file populating from empty is a
zero-behavior-change edit, not a re-tuning.

`camera.py` also exposes a separate YAML-driven config path
(`load_camera_config_yaml()` / `CameraAPI.from_yaml()`); that path is NOT used
by `cuasp.py`'s live runtime today (`cuasp.py` builds `CameraConfig` directly
in Python, sourced entirely from this module) -- it remains available only as
an alternate entry point for other consumers.

Call `validate()` once at startup (already wired into `cuasp.py`'s
`_validate_configuration()`) to fail fast on malformed values instead of
silently misbehaving at runtime.
"""

from __future__ import annotations

import math


# =============================================================================
# Marker identity (control/cuasp.py)
# =============================================================================

# ArUco marker IDs mounted on the target; 2+ ids enable rigid multi-marker
# target-point fusion in camera.py's MultiArucoCorrectionSourceAdapter.
# 2026-09-21: promoted from calibration/roughtest/run_cuasp_live_test.py's
# bench overrides to the real, final target marker identity (was
# (0, 1) / DICT_5X5_100 / 0.10m -- superseded, not a bench-only value anymore).
CV_ARUCO_TARGET_IDS = (1, 2)
CV_ARUCO_DICT_NAME = "DICT_4X4_100"
CV_ARUCO_MARKER_SIZE_M = 0.05  # metres

# Real calibration campaign now populated -- see calibration/README.md and
# calibration/calibration.yaml's own header (2026-09-21: finalize_runtime_
# calibration.py run for real, closing root README.md's "Known integration
# gaps" wiring item). Path is repo-root-relative (see cuasp.py's
# CALIBRATION_YAML resolution, changed alongside this to match). The camera-
# to-laser boresight embedded in that file is still partially provisional --
# see calibration/boresight.yaml's own header before treating any angle from
# it as fully validated.
CV_CAMERA_CALIBRATION_YAML = "calibration/calibration.yaml"


# =============================================================================
# IDS acquisition (control/cuasp.py ids_peak config dict)
# =============================================================================

CV_IDS_DEVICE_INDEX = 0
CV_IDS_PIXEL_FORMAT = "Mono8"
CV_IDS_WIDTH = 2592  # px
CV_IDS_HEIGHT = 1944  # px
CV_IDS_BUFFER_TIMEOUT_MS = 100
CV_IDS_ACQUISITION_MODE = "Continuous"
CV_IDS_EXPOSURE_AUTO = "Off"
CV_IDS_GAIN_AUTO = "Off"
CV_IDS_EXPOSURE_TIME_US = 20000.0
# 2026-09-21: was 5000.0 (tuned for a fast-moving outdoor target). Confirmed
# by the operator as the new real value -- the previous value left the
# current operating scene near-black (mean pixel ~18/255, confirmed by direct
# capture during roughtest smoke tests) and unusable for cv2.aruco detection.
CV_IDS_GAIN_DB = 0.0
CV_IDS_FRAME_RATE_TARGET_ENABLE = 1
CV_IDS_FRAME_RATE_TARGET_HZ = 70.0
CV_IDS_GENTL_CTI_PATH = ""


# =============================================================================
# ArUco detector tuning (camera.py _apply_aruco_detector_tuning_from_settings)
# =============================================================================

CV_ARUCO_CORNER_REFINEMENT = "SUBPIX"
CV_ARUCO_CORNER_REFINEMENT_WIN_SIZE = 5
CV_ARUCO_CORNER_REFINEMENT_MAX_ITERATIONS = 30
CV_ARUCO_CORNER_REFINEMENT_MIN_ACCURACY = 0.1
CV_ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MIN = 3
CV_ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MAX = 23
CV_ARUCO_ADAPTIVE_THRESH_WIN_SIZE_STEP = 10
CV_ARUCO_MIN_MARKER_PERIMETER_RATE = 0.015
CV_ARUCO_MAX_MARKER_PERIMETER_RATE = 4.0
CV_ARUCO_POLYGONAL_APPROX_ACCURACY_RATE = 0.03
CV_ARUCO_MIN_CORNER_DISTANCE_RATE = 0.02


# =============================================================================
# Fast-tracking / KLT / detection ROI (camera.py load_fast_tracking_config)
# =============================================================================

CV_CUAS_ARUCO_REDETECT_HZ = 3.0
CV_CUAS_MISSING_MARKER_DETECT_HZ = 8.0
CV_CUAS_FULL_FRAME_RECOVERY_HZ = 0.5
CV_CUAS_DETECTION_RESULT_MAX_AGE_S = 0.12
CV_CUAS_PREPROCESS_MODE = "clahe"  # raw | stretch | clahe
CV_CUAS_TRACK_DETECTION_ROI_SCALE = 2.5
CV_CUAS_TRACK_DETECTION_ROI_MIN_SIDE_PX = 320
CV_CUAS_MISSING_DETECTION_ROI_SCALE = 8.0
CV_CUAS_MISSING_DETECTION_ROI_MIN_SIDE_PX = 384
CV_ARUCO_SCOUT_GRID_COLS = 3
CV_ARUCO_SCOUT_GRID_ROWS = 3
CV_ARUCO_SCOUT_TILE_OVERLAP = 0.25

CV_CUAS_KLT_WIN_SIZE = 15  # px, odd
CV_CUAS_KLT_MAX_LEVEL = 2
CV_CUAS_KLT_MAX_ITERS = 10
CV_CUAS_KLT_EPS = 0.03
CV_CUAS_KLT_ROI_MARGIN_PX = 64
CV_CUAS_KLT_ROI_SCALE = 5.0
CV_CUAS_KLT_FB_CHECK_PERIOD = 4  # frames
CV_CUAS_KLT_USE_INITIAL_FLOW = 1
CV_CUAS_KLT_ROBUST_WIN_SIZE = 21  # px, odd
CV_CUAS_KLT_ROBUST_MAX_LEVEL = 3
CV_CUAS_KLT_ROBUST_MAX_ITERS = 15
CV_CUAS_KLT_ROBUST_EPS = 0.01
CV_CUAS_KLT_FB_MAX_ERROR_PX = 1.25
CV_CUAS_KLT_MAX_LK_ERROR = 30.0
CV_CUAS_KLT_AREA_RATIO_MIN = 0.35
CV_CUAS_KLT_AREA_RATIO_MAX = 2.85
CV_CUAS_KLT_MAX_SIDE_RATIO = 4.0
CV_CUAS_KLT_MIN_QUAD_AREA_PX2 = 16.0

CV_CUAS_CUDA_MODE = "auto"  # auto | off | force
CV_CUAS_CUDA_USE_KLT = 1
CV_CUAS_CUDA_BENCHMARK_FRAMES = 8

CV_CUAS_DEBUG_DRAW_HZ = 2.0
CV_CUAS_DEBUG_MAX_WIDTH = 960  # px, single-marker debug snapshot
# Range used to project the debug overlay's expected-laser-position marker
# when no target is tracked (with a target, that target's own range is used).
CV_CUAS_DEBUG_LASER_RANGE_M = 10.0


# =============================================================================
# Multi-marker tracking / rigid target-point fusion
# (camera.py load_multi_marker_tracking_config, _multi_marker_offset_cm,
#  _multi_marker_frame_rpy_deg)
# =============================================================================

CV_CUAS_STRETCH_PERCENTILE_LOW = 2.0
CV_CUAS_STRETCH_PERCENTILE_HIGH = 98.0
CV_CUAS_CLAHE_CLIP_LIMIT = 2.0
CV_CUAS_CLAHE_TILE_X = 8
CV_CUAS_CLAHE_TILE_Y = 8

CV_CUAS_TRACK_ROI_EXPAND_ON_MISS = 1.6
CV_CUAS_TRACK_ROI_MISSES_TO_SCOUT = 3
CV_CUAS_TRACK_EXTRA_SCOUT_TILES_PER_FRAME = 0
CV_CUAS_PARTIAL_EXTRA_SCOUT_TILES_PER_FRAME = 2
CV_CUAS_PARTIAL_HINT_SEARCH_ENABLE = 1
CV_CUAS_SCOUT_TILES_PER_FRAME = 2
CV_CUAS_SCOUT_ORDER = "last_center_first"  # last_center_first | center_out

CV_CUAS_CONFIDENCE_AREA_RATIO_REF = 0.00048
CV_CUAS_POSE_REPROJ_RMSE_CONF_PX = 4.0
CV_CUAS_POSE_REPROJ_RMSE_MAX_PX = 10.0
CV_CUAS_VELOCITY_FILTER_ALPHA = 0.35
CV_CUAS_COVARIANCE_MIN_STD_M = 0.003
CV_CUAS_COVARIANCE_DEPTH_FRACTION_MIN = 0.01
CV_CUAS_COVARIANCE_CONFIDENCE_FLOOR = 0.02

CV_CUAS_SINGLE_MARKER_FALLBACK_ENABLE = 1
CV_CUAS_SINGLE_MARKER_CONFIDENCE_SCALE = 0.65
CV_CUAS_SINGLE_MARKER_COVARIANCE_SCALE = 4.0
CV_CUAS_TARGET_OFFSET_STD_CM = 0.5

# Rigid-target-frame offsets, in centimetres, expressed in the OpenCV marker
# frame. RESTORED from the original standalone cv_CUAS.py (this file was
# empty before, so these three values were never actually in effect until
# now -- they were only literal fallback defaults inside cuasp.py's own
# source, and yesterday's migration into camera.py accidentally reset the
# *code-level* fallback defaults to (0,0,0) instead of preserving them).
# FLAG: these read as real physical rig measurements (markers ~18 cm either
# side of a +5 cm-offset midpoint), not placeholders -- but they have not
# been re-verified against the current physical target since being restored
# here. Treat as provisional until someone re-measures the rig; see
# README.md "Known integration gaps" (calibration.yaml) for the related,
# still-open physical-data gap.
CV_CUAS_TARGET_MIDPOINT_OFFSET_CM = (0.0, 5.0, 0.0)
# 2026-09-21: real target marker ids changed from (0, 1) to (1, 2) (see
# CV_ARUCO_TARGET_IDS above) -- these two fields are RENUMBERED, not
# re-measured: id 1 now carries the value the old id 1 already had (it keeps
# its own role, now as the frame-reference marker), and id 2 carries the
# value the old, now-retired id 0 had. Validated live on real hardware
# 2026-09-18 against the "estimated center jumps to the opposite side on
# single-marker loss" failure mode (see calibration/roughtest/README.md's
# "Marker-id offset mismatch" section) before being promoted here -- this
# exact mapping was confirmed correct by the operator, not re-derived.
CV_CUAS_ARUCO_1_TARGET_OFFSET_CM = (21.0, 5.0, 0.0)
# 2026-09-24: id 2's value above (-18.0, 5.0, 0.0), carried over unchanged
# from the renumbering, assumed the physical marker now printed/mounted as
# id 2 sits in the same rotational orientation the old id 0 was measured in.
# The operator physically inspected the rig and confirmed the id-2 marker is
# mounted rotated 180 degrees about its own boresight (Z) axis relative to
# that assumption. A 180deg in-plane rotation flips both local X and Y axes
# (Z unaffected), so the same fixed real-world aim-point direction must be
# re-expressed in the marker's actual (rotated) local frame by negating BOTH
# X and Y of the old value: (-(-18.0), -(5.0), 0.0) = (18.0, -5.0, 0.0).
# This only affects the single-marker fallback branch (compute_multi_marker_
# target_point's `else` branch, camera.py) -- the complete-observation branch
# never uses id 2's own rotation/offset, which is why the mismatch was
# invisible whenever id 1 was also visible and only showed up as a mirrored
# center when id 1 was lost and id 2 tracked alone. Not yet re-validated live
# against real hardware with a deliberate single-marker-loss transition.
CV_CUAS_ARUCO_2_TARGET_OFFSET_CM = (21.0, -5.0, 0.0)

CV_CUAS_TARGET_FRAME_REFERENCE_ARUCO_ID = 1  # was 0 -- see the renumbering note above
CV_CUAS_ARUCO_1_TARGET_FRAME_RPY_DEG = (0.0, 0.0, 0.0)
CV_CUAS_ARUCO_2_TARGET_FRAME_RPY_DEG = (0.0, 0.0, 0.0)


# =============================================================================
# PBVS Z resolution (camera.py load_pbvs_z_config)
# =============================================================================

CV_PBVS_Z_MODE = "measured"  # measured | filtered | fixed
CV_PBVS_Z_FIXED_M = 0.0
CV_PBVS_Z_FILTER_TAU_S = 0.35

# Optional legacy single-marker target offset override, FRD convention
# (forward/right/down, metres). Unset by default -- only read by the
# single-marker IdsOpenCvCorrectionSourceAdapter when all three are present.
# CV_PBVS_TARGET_DX_M = 0.0
# CV_PBVS_TARGET_DY_M = 0.0
# CV_PBVS_TARGET_DZ_M = 0.0


# =============================================================================
# PBVS sign/axis mapping + transport clamp (control/cuasp.py, supervisor-owned)
# =============================================================================

CV_PBVS_RAW_TO_CMD_AZ_SIGN = 1.0
CV_PBVS_RAW_TO_CMD_PO_SIGN = -1.0
POINTING_FROM_RAD_TO_CMD = 180.0 / math.pi
# Empirically verified 2026-09-17 with real motor hardware (camera mounted on
# the pointer, PBVS-driven correction against a stationary marker at a known
# on-frame offset): with the previous -1.0 here (net effective_sign_az=+1),
# a target to the right of frame center produced a net azimuth command that
# moved the real pointer LEFT (away from the target) in Kollmorgen Automation
# Suite -- confirmed by starting both axes at 0.0 and reading the net absolute
# position after a run of real, non-clamped corrections. The polar axis's
# sign was independently confirmed correct in the same test (target below
# center moved the pointer down, toward the target), so only this constant
# was flipped, not CV_PBVS_RAW_TO_CMD_AZ_SIGN or POINTING_PO_DIR_CMD. See
# control/README.md's "Safety expectations" checklist item this closes:
# "verify correct axis mapping and signs at low authority."
POINTING_AZ_DIR_CMD = 1.0
POINTING_PO_DIR_CMD = -1.0
CV_CUAS_PBVS_MAX_ERROR_DEG = 30.0  # deg, transport sanity clamp only
CV_CUAS_PBVS_GAIN = 1.0  # visual-error gain (control/README.md's "Preferred
# baseline command formulation": q_cmd_deg = q_base_deg + K_visual *
# axis_sign_map(error_deg)). 1.0 == no scaling, exactly matching real
# behavior before this constant existed. Applied before the transport clamp
# above, so PBVS_MAX_ERROR_DEG remains the authoritative safety bound for any
# gain, including > 1.0. Prototyped as PBVS_TEST_GAIN in
# calibration/roughtest/run_cuasp_live_test.py; not yet tuned/validated for
# real closed-loop operation -- see roughtest/README.md.


# =============================================================================
# Gating (control/cuasp.py -> SunCubes.camera.GatingConfig)
# =============================================================================

CV_ARUCO_CONFIDENCE_MIN = 0.0005
CV_ARUCO_Z_MIN_M = 0.25
CV_CUAS_MAX_AGE_S = 0.25  # s


# =============================================================================
# Runtime / debug (control/cuasp.py)
# =============================================================================

CV_CUAS_LOOP_HZ = CV_IDS_FRAME_RATE_TARGET_HZ
CV_CUAS_DEBUG_DISPLAY_SCALE = 0.33
# Note: CV_CUAS_DEBUG_DRAW_HZ is shared with the fast-tracking block above;
# both cuasp.py's own debug loop and camera.py's per-source debug snapshot
# gating read the same name.


# =============================================================================
# Motor safety switch / network / dispatch timing (control/cuasp.py)
# =============================================================================

CV_CUAS_ENABLE_MOTOR_COMMANDS = 1
IP = "192.168.0.109"
PORT = 8123
CV_CUAS_MOTOR_COMMAND_HZ = 50.0
CV_CUAS_MOTOR_COMMAND_STALE_TIMEOUT_S = 0.25
CV_CUAS_MOTOR_RECONNECT_INITIAL_S = 0.5
CV_CUAS_MOTOR_RECONNECT_MAX_S = 5.0
CV_CUAS_MOTOR_COMMAND_MIN_INTERVAL_S = 0.0  # 0.0 == disabled (default): no
# minimum wall-clock gap enforced between actual hardware sends beyond
# MOTOR_COMMAND_HZ's own period. Prototyped as CORRECTION_MIN_INTERVAL_S in
# calibration/roughtest/run_cuasp_live_test.py to isolate a real-hardware
# vertical-oscillation incident -- see roughtest/README.md's "Root cause
# found" section: raising this on the test rig did NOT fix the polar ringing
# (confirmed autonomous KAS/mechanical limit cycle, not a resend-rate
# artifact), so it ships disabled. Only raise it after a real, separately
# validated tuning pass against real tracking-latency requirements.


# =============================================================================
# Validation
# =============================================================================


def validate() -> None:
    """Fail fast on malformed settings instead of silently misbehaving.

    Deliberately narrow: only checks the kinds of mistakes a manual edit is
    likely to introduce (wrong type, non-finite, non-positive rate/size).
    `cuasp.py`'s own `_validate_configuration()` performs the complementary,
    safety-focused checks (signs non-zero, motor endpoint set, etc.) and
    calls this function as part of that startup validation.
    """
    errors: list[str] = []

    def _positive(name: str, value: object) -> None:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric, got {value!r}")
            return
        if not math.isfinite(numeric) or numeric <= 0.0:
            errors.append(f"{name} must be finite and > 0, got {value!r}")

    def _non_negative(name: str, value: object) -> None:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric, got {value!r}")
            return
        if not math.isfinite(numeric) or numeric < 0.0:
            errors.append(f"{name} must be finite and >= 0, got {value!r}")

    def _vec3(name: str, value: object) -> None:
        try:
            length = len(value)  # type: ignore[arg-type]
        except TypeError:
            errors.append(f"{name} must be a 3-sequence, got {value!r}")
            return
        if length != 3 or not all(math.isfinite(float(v)) for v in value):  # type: ignore[union-attr]
            errors.append(f"{name} must contain exactly three finite numbers, got {value!r}")

    for name in ("CV_IDS_WIDTH", "CV_IDS_HEIGHT", "CV_ARUCO_MARKER_SIZE_M"):
        _positive(name, globals()[name])

    for name in (
        "CV_CUAS_ARUCO_REDETECT_HZ",
        "CV_CUAS_MISSING_MARKER_DETECT_HZ",
        "CV_CUAS_MOTOR_COMMAND_HZ",
        "CV_CUAS_MOTOR_COMMAND_STALE_TIMEOUT_S",
        "CV_CUAS_PBVS_MAX_ERROR_DEG",
        "CV_CUAS_PBVS_GAIN",
        "CV_ARUCO_Z_MIN_M",
        "CV_CUAS_MAX_AGE_S",
    ):
        _positive(name, globals()[name])

    for name in (
        "CV_CUAS_FULL_FRAME_RECOVERY_HZ",
        "CV_ARUCO_CONFIDENCE_MIN",
        "CV_CUAS_MOTOR_COMMAND_MIN_INTERVAL_S",
    ):
        _non_negative(name, globals()[name])

    if CV_CUAS_KLT_WIN_SIZE < 3 or CV_CUAS_KLT_WIN_SIZE % 2 == 0:
        errors.append("CV_CUAS_KLT_WIN_SIZE must be an odd integer >= 3")
    if CV_CUAS_KLT_ROBUST_WIN_SIZE < 3 or CV_CUAS_KLT_ROBUST_WIN_SIZE % 2 == 0:
        errors.append("CV_CUAS_KLT_ROBUST_WIN_SIZE must be an odd integer >= 3")

    if CV_CUAS_CUDA_MODE not in {"auto", "off", "force"}:
        errors.append("CV_CUAS_CUDA_MODE must be one of: auto, off, force")
    if CV_PBVS_Z_MODE not in {"measured", "filtered", "fixed"}:
        errors.append("CV_PBVS_Z_MODE must be one of: measured, filtered, fixed")
    if CV_CUAS_PREPROCESS_MODE not in {"raw", "stretch", "clahe"}:
        errors.append("CV_CUAS_PREPROCESS_MODE must be one of: raw, stretch, clahe")

    for name in (
        "CV_PBVS_RAW_TO_CMD_AZ_SIGN",
        "CV_PBVS_RAW_TO_CMD_PO_SIGN",
        "POINTING_AZ_DIR_CMD",
        "POINTING_PO_DIR_CMD",
    ):
        value = globals()[name]
        if not math.isfinite(float(value)) or abs(float(value)) <= 1e-12:
            errors.append(f"{name} must be finite and non-zero, got {value!r}")

    if not (0 < PORT <= 65535):
        errors.append(f"PORT must be in [1, 65535], got {PORT!r}")
    if not IP:
        errors.append("IP must not be empty")

    for name in (
        "CV_CUAS_TARGET_MIDPOINT_OFFSET_CM",
        "CV_CUAS_ARUCO_1_TARGET_OFFSET_CM",
        "CV_CUAS_ARUCO_2_TARGET_OFFSET_CM",
        "CV_CUAS_ARUCO_1_TARGET_FRAME_RPY_DEG",
        "CV_CUAS_ARUCO_2_TARGET_FRAME_RPY_DEG",
    ):
        _vec3(name, globals()[name])

    if len(set(CV_ARUCO_TARGET_IDS)) != len(CV_ARUCO_TARGET_IDS):
        errors.append("CV_ARUCO_TARGET_IDS must not contain duplicate ids")

    if errors:
        raise ValueError("Invalid suncubes.settings:\n- " + "\n- ".join(errors))
