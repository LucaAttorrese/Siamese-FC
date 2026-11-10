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
# Target tracker: SiamFC + drone detector (cuasp_siamfc, PLAN.md Phases 3-4)
# =============================================================================

# "siamfc": visual tracking of the drone itself (camera.py provider_mode
# "ids_siamfc", suncubes/siamfc_source.py). "aruco": the original C-UASP
# ArUco/PnP pipeline, unchanged. Default "siamfc" (decision of 2026-11-10).
CV_TARGET_TRACKER = "siamfc"

# Weights. Paths are relative to cuasp_siamfc/ (like CV_CAMERA_CALIBRATION_YAML)
# or absolute. When a SHA256 is set, start-up refuses a file whose hash differs;
# "" disables the check. The SiamFC .pth hash is the one written by
# suncubes.siamfc.convert_matconvnet on the training PC (2026-11-06); a .pth
# converted on another machine may differ byte-wise -- then update it here.
CV_SIAMFC_WEIGHTS_PATH = "../pretrained/networks/baseline-conv5_gray_e100.pth"
CV_SIAMFC_WEIGHTS_SHA256 = "d858fec57757d1fa6eb5c9c3163b826d0337298aa91f68247bf9522562cfbb6f"
CV_DETECTOR_WEIGHTS_PATH = "../runs/detector/yolo11s_uav_gray_1280_e40/weights/best.pt"
CV_DETECTOR_WEIGHTS_SHA256 = "3a7c173244899df1ac5185916cbb8196bd0959b3d5626acc9f9cc3976d8040d5"

# SiamFC tracker.
CV_SIAMFC_DEVICE = ""  # "" = cuda when available, else cpu
CV_SIAMFC_PARAMS = "paper"  # "paper" (CFNet Table 2) or a dict of SiamFCParams overrides (from V4b)
# Fixed camera depth Z used to back-project the target pixel (no range sensor):
# centre of the expected 20-22 m. Parallax error for Z_true in 20..22 m is
# |t_lc| * |1/Z_true - 1/Z| <= ~0.43 mrad with the 18 cm lever arm.
CV_SIAMFC_TARGET_RANGE_M = 21.0
# Gating confidence for SiamFC samples (replaces CV_ARUCO_CONFIDENCE_MIN, which
# is calibrated on the ArUco marker-area confidence, in siamfc mode).
CV_SIAMFC_CONF_MIN = 0.15
CV_SIAMFC_BBOX_OFFSET_PX = (0.0, 0.0)  # aim pixel = tracked box centre + this offset

# Tracking state machine (siamfc_source.SessionConfig; see its comments).
# To be tuned in V4 on IDS recordings with tools/replay_siamfc_source.py.
CV_SIAMFC_LOST_CONF = 0.25
CV_SIAMFC_LOST_FRAMES = 5
CV_SIAMFC_MAX_LOST_S = 1.0
CV_SIAMFC_REF_FRAMES = 5
CV_SIAMFC_MISSING_HZ = 10.0  # full-frame detection while searching / lost
CV_SIAMFC_ACQUIRE_MIN_SCORE = 0.5
CV_SIAMFC_REDETECT_HZ = 5.0  # verification while tracking
CV_SIAMFC_VERIFY_RADIUS = 1.5
CV_SIAMFC_VERIFY_RADIUS_MIN_PX = 128.0
CV_SIAMFC_VERIFY_MAX_SIZE_RATIO = 1.5
CV_SIAMFC_REINIT_IOU = 0.5
CV_SIAMFC_REINIT_CONFIRM = 2
CV_SIAMFC_VERIFY_MISS_LIMIT = 2

# Drone detector (siamfc.detector.DetectorConfig / ConfirmerConfig).
CV_DETECTOR_DEVICE = ""  # "" = Ultralytics default (first CUDA GPU when available)
CV_DETECTOR_CONF_MIN = 0.15
CV_DETECTOR_IMGSZ = 1280
CV_DETECTOR_NMS_IOU = 0.7
CV_DETECTOR_MAX_DET = 10
CV_DETECTOR_FP16 = 1
CV_DETECTOR_MAX_BOX_FRAC = 0.3  # boxes larger than this fraction of the frame's larger side are dropped
CV_DETECTOR_MIN_BOX_PX = 4.0
CV_DETECTOR_CONFIRM_FRAMES = 3  # consistent full-frame detections needed to start a track
CV_DETECTOR_CONFIRM_MAX_SHIFT = 1.0
CV_DETECTOR_CONFIRM_MAX_SIZE_RATIO = 1.5
CV_DETECTOR_CONFIRM_MAX_GAP_S = 0.5


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

# cuasp_siamfc: 0 by default (decision of 2026-11-10; was 1 in C-UASP), so a first
# SiamFC run cannot move the motors by mistake. The C-UASP safety rules require live
# tests to start with motor commands disabled. Enable only for the motors-on bench
# test (PLAN.md, V8), deliberately.
CV_CUAS_ENABLE_MOTOR_COMMANDS = 0
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

    # Target tracker / SiamFC block: types and obvious ranges only. The complete,
    # cross-field validation is SiamFCSourceConfig.validate(), run by cuasp.py's
    # _validate_configuration() in siamfc mode. The weight files themselves are
    # checked (existence, SHA256) when the camera source starts.
    if CV_TARGET_TRACKER not in {"aruco", "siamfc"}:
        errors.append(f"CV_TARGET_TRACKER must be 'aruco' or 'siamfc', got {CV_TARGET_TRACKER!r}")
    for name in ("CV_SIAMFC_WEIGHTS_PATH", "CV_DETECTOR_WEIGHTS_PATH"):
        if not isinstance(globals()[name], str) or not globals()[name].strip():
            errors.append(f"{name} must be a non-empty path string")
    for name in ("CV_SIAMFC_WEIGHTS_SHA256", "CV_DETECTOR_WEIGHTS_SHA256"):
        value = globals()[name]
        if not isinstance(value, str) or (value and (len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value))):
            errors.append(f"{name} must be '' or a 64-character hex SHA256, got {value!r}")
    if not (CV_SIAMFC_PARAMS == "paper" or isinstance(CV_SIAMFC_PARAMS, dict)):
        errors.append("CV_SIAMFC_PARAMS must be 'paper' or a dict of SiamFCParams overrides")
    for name in (
        "CV_SIAMFC_TARGET_RANGE_M", "CV_SIAMFC_MAX_LOST_S", "CV_SIAMFC_MISSING_HZ", "CV_SIAMFC_VERIFY_RADIUS",
        "CV_SIAMFC_VERIFY_MAX_SIZE_RATIO", "CV_SIAMFC_REINIT_IOU", "CV_DETECTOR_IMGSZ", "CV_DETECTOR_NMS_IOU",
        "CV_DETECTOR_MAX_BOX_FRAC", "CV_DETECTOR_CONFIRM_MAX_SHIFT", "CV_DETECTOR_CONFIRM_MAX_SIZE_RATIO",
        "CV_DETECTOR_CONFIRM_MAX_GAP_S",
    ):
        _positive(name, globals()[name])
    for name in ("CV_SIAMFC_REDETECT_HZ", "CV_SIAMFC_VERIFY_RADIUS_MIN_PX", "CV_SIAMFC_ACQUIRE_MIN_SCORE", "CV_DETECTOR_MIN_BOX_PX"):
        _non_negative(name, globals()[name])
    for name in ("CV_SIAMFC_CONF_MIN", "CV_SIAMFC_LOST_CONF", "CV_DETECTOR_CONF_MIN"):
        value = globals()[name]
        try:
            ok = 0.0 < float(value) < 1.0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            errors.append(f"{name} must be in (0, 1), got {value!r}")
    for name in (
        "CV_SIAMFC_LOST_FRAMES", "CV_SIAMFC_REF_FRAMES", "CV_SIAMFC_REINIT_CONFIRM", "CV_SIAMFC_VERIFY_MISS_LIMIT",
        "CV_DETECTOR_MAX_DET", "CV_DETECTOR_CONFIRM_FRAMES",
    ):
        value = globals()[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"{name} must be an integer >= 1, got {value!r}")
    try:
        offset_ok = len(CV_SIAMFC_BBOX_OFFSET_PX) == 2 and all(math.isfinite(float(v)) for v in CV_SIAMFC_BBOX_OFFSET_PX)
    except (TypeError, ValueError):
        offset_ok = False
    if not offset_ok:
        errors.append(f"CV_SIAMFC_BBOX_OFFSET_PX must contain exactly two finite numbers, got {CV_SIAMFC_BBOX_OFFSET_PX!r}")

    if errors:
        raise ValueError("Invalid suncubes.settings:\n- " + "\n- ".join(errors))
