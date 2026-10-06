"""End-to-end tests of camera.py's multi-marker pipeline using real
cv2.aruco-rendered synthetic images.

Pure-math tests alone would not catch an integration bug like a wrong
argument order into cv2.aruco.estimatePoseSingleMarkers or a coordinate-frame
mismatch between detection and PnP -- this exercises detection, PnP,
covariance, fusion and gating together, exactly like the real pipeline would,
without needing a physical camera. The IDS acquisition producer is bypassed
(constructed by hand) since no camera hardware is available.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
import numpy as np
import pytest

from suncubes import camera as cam

IMG_W, IMG_H = 1280, 960
DICT_NAME = "DICT_5X5_100"
CAMERA_MATRIX = np.array([[900.0, 0.0, IMG_W / 2.0], [0.0, 900.0, IMG_H / 2.0], [0.0, 0.0, 1.0]])
DIST_COEFFS = np.zeros(5)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))


def render_marker(canvas, marker_id, cx, cy, side_px=120):
    tag = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, side_px)
    x0, y0 = int(cx - side_px / 2), int(cy - side_px / 2)
    canvas[y0:y0 + side_px, x0:x0 + side_px] = tag


def make_frame(ids_and_positions):
    canvas = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
    for marker_id, (cx, cy) in ids_and_positions.items():
        render_marker(canvas, marker_id, cx, cy)
    return canvas


@pytest.fixture
def adapter():
    """A MultiArucoCorrectionSourceAdapter built by hand (no IDS/calibration
    file), mirroring exactly what start() would populate from a calibration +
    aruco config -- see camera.py's MultiArucoCorrectionSourceAdapter.start()."""
    a = cam.MultiArucoCorrectionSourceAdapter.__new__(cam.MultiArucoCorrectionSourceAdapter)
    a._cfg = {}
    a._gating = cam.GatingConfig(max_age_s=5.0, conf_min=0.01, z_min_m=0.05, max_abs_angle_rad=0.0)
    a._ids_cfg = {}
    a._aruco_cfg = {}
    a._calib_cfg = {}
    a._fast_cfg = cam.load_fast_tracking_config()
    a._track_cfg = cam.MultiMarkerTrackingConfig(confidence_area_ratio_ref=0.001)
    a._started = True
    a._cv2 = cv2
    a._producer = None
    a._detector_executor = ThreadPoolExecutor(max_workers=1)
    a._detector_future = None
    a._preprocessor = cam.MultiMarkerTiledPreprocessor(cv2, a._fast_cfg, a._track_cfg)
    a._detector, a._aruco_dict, a._aruco_params = cam._init_aruco_detector(cv2, DICT_NAME)
    a._marker_specs = (
        cam.ArucoMarkerSpec(marker_id=0, dict_name=DICT_NAME, marker_size_m=0.10),
        cam.ArucoMarkerSpec(marker_id=1, dict_name=DICT_NAME, marker_size_m=0.10),
    )
    a._k_calib = CAMERA_MATRIX
    a._dist = DIST_COEFFS
    a._r_lc = np.eye(3)
    a._t_lc = np.zeros((3, 1))
    a._calib_w = IMG_W
    a._calib_h = IMG_H
    a._corner_tracker = cam.SparseCornerTracker(cv2, a._fast_cfg)
    a._last_sequence = 0
    a._prev_gray = None
    a._tracked_corners = {}
    a._last_marker_centers_px = {}
    a._track_states = {}
    a._last_detection_submit_ns = 0
    a._last_candidate = None
    a._debug_enable = False
    a._debug_lock = Lock()
    a._last_debug_ns = 0
    a._last_debug = {"available": False, "reason": "debug_disabled"}
    a._stats_lock = Lock()
    a._stats = {
        "version": "test", "frames": 0, "new_frames": 0, "detections_submitted": 0,
        "detections_ok": 0, "detection_failures": 0, "last_detection_ms": 0.0,
        "tracking_backend": "none",
    }
    yield a
    a._detector_executor.shutdown(wait=False, cancel_futures=True)


def make_producer(gray, ts_ns, sequence):
    return type("FakeProducer", (), {
        "latest": lambda self: cam._FramePacket(gray=gray, ts_capture_ns=ts_ns, sequence=sequence),
        "stats": lambda self: {"published": sequence, "replaced": 0, "sequence": sequence},
    })()


def _wait_for_detection(adapter, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while (
        adapter._detector_future is not None
        and not adapter._detector_future.done()
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert adapter._detector_future is not None and adapter._detector_future.done()


def test_detect_job_finds_both_configured_markers(adapter):
    frame = make_frame({0: (560, 480), 1: (720, 480)})
    result = adapter._detect_job(frame, ts_capture_ns=1, sequence=1, plans=[(0, 0, frame)], mode="scout")
    assert set(result.markers.keys()) == {0, 1}
    assert result.elapsed_ms >= 0.0


def test_make_observation_produces_valid_pnp_for_each_marker(adapter):
    frame = make_frame({0: (560, 480), 1: (720, 480)})
    result = adapter._detect_job(frame, ts_capture_ns=1, sequence=1, plans=[(0, 0, frame)], mode="scout")
    now_ns = time.monotonic_ns()
    for marker_id, info in result.markers.items():
        obs = adapter._make_observation(
            adapter._marker_specs[marker_id], info["corners"], CAMERA_MATRIX, IMG_W, IMG_H, now_ns
        )
        assert obs is not None
        assert np.isfinite(obs.position_camera_m[2]) and obs.position_camera_m[2] > 0.0
        assert obs.reproj_rmse_px < 2.0, "a clean synthetic render should reproject almost exactly"
        assert 0.0 < obs.confidence <= 1.0


def test_marker_x_ordering_matches_image_layout(adapter):
    """Marker 0 rendered left of marker 1 in the image must land at a smaller
    camera-frame X, and the fused midpoint must sit between them."""
    frame = make_frame({0: (560, 480), 1: (720, 480)})
    result = adapter._detect_job(frame, ts_capture_ns=1, sequence=1, plans=[(0, 0, frame)], mode="scout")
    now_ns = time.monotonic_ns()
    observations = [
        adapter._make_observation(
            adapter._marker_specs[mid], info["corners"], CAMERA_MATRIX, IMG_W, IMG_H, now_ns
        )
        for mid, info in result.markers.items()
    ]
    assert len(observations) == 2
    by_id = {o.marker_id: o for o in observations}
    assert by_id[0].position_camera_m[0] < by_id[1].position_camera_m[0]

    target = cam.compute_multi_marker_target_point(observations, expected_marker_count=2, cfg=adapter._track_cfg)
    assert target is not None
    x0, x1 = by_id[0].position_camera_m[0], by_id[1].position_camera_m[0]
    assert min(x0, x1) <= target.position_camera_m[0] <= max(x0, x1)


def test_read_candidate_becomes_reliable_once_async_detection_is_consumed(adapter):
    frame = make_frame({0: (560, 480), 1: (720, 480)})
    now0 = time.monotonic_ns()
    adapter._producer = make_producer(frame, now0, 1)

    first = adapter.read_candidate(now_ns=now0)
    assert not first.reliable, "the very first frame only schedules detection asynchronously"
    assert adapter._detector_future is not None
    _wait_for_detection(adapter)

    now1 = time.monotonic_ns()
    adapter._producer = make_producer(frame, now1, 2)
    candidate = adapter.read_candidate(now_ns=now1)
    assert candidate.reliable
    assert candidate.reason == "ok"
    assert candidate.marker_id in (0, 1)
    assert np.isfinite(candidate.d_az_rad)
    assert np.isfinite(candidate.d_po_rad)
    assert candidate.z_m is not None and candidate.z_m > 0.0


def test_read_candidate_reuses_cached_result_for_unchanged_sequence(adapter):
    frame = make_frame({0: (560, 480), 1: (720, 480)})
    now0 = time.monotonic_ns()
    adapter._producer = make_producer(frame, now0, 1)
    adapter.read_candidate(now_ns=now0)
    _wait_for_detection(adapter)

    now1 = time.monotonic_ns()
    adapter._producer = make_producer(frame, now1, 2)
    candidate = adapter.read_candidate(now_ns=now1)

    reused = adapter.read_candidate(now_ns=time.monotonic_ns())
    assert reused.reason == candidate.reason
    assert reused.marker_id == candidate.marker_id
    assert reused.age_s is not None and reused.age_s > 0.0


def test_read_candidate_single_visible_marker_uses_fallback(adapter):
    """After the pair is known, losing one marker from the frame should keep
    tracking the survivor directly via KLT (no async wait needed) and fall
    back to the single-marker target-point path."""
    frame_both = make_frame({0: (560, 480), 1: (720, 480)})
    now0 = time.monotonic_ns()
    adapter._producer = make_producer(frame_both, now0, 1)
    adapter.read_candidate(now_ns=now0)
    _wait_for_detection(adapter)
    now1 = time.monotonic_ns()
    adapter._producer = make_producer(frame_both, now1, 2)
    adapter.read_candidate(now_ns=now1)

    frame_one = make_frame({0: (560, 480)})
    now2 = time.monotonic_ns()
    adapter._producer = make_producer(frame_one, now2, 3)
    candidate = adapter.read_candidate(now_ns=now2)
    assert candidate.reliable
    assert candidate.marker_id == 0
