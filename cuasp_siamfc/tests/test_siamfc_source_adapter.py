"""suncubes.siamfc_source: geometry, tracking state machine and camera adapter, with fakes only
(no IDS camera, no weights, no GPU)."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
import pytest

from suncubes import camera as cam
from suncubes.siamfc.detector import ConfirmerConfig, Detection, DetectionConfirmer
from suncubes.siamfc_source import (
    REASON_LOST,
    REASON_SEARCHING,
    REASON_TRACKER_ERROR,
    STATE_LOST,
    STATE_SEARCHING,
    STATE_TRACKING,
    DetectionOutcome,
    IdsSiamFCCorrectionSourceAdapter,
    SessionConfig,
    SiamFCSourceConfig,
    SiamFCTrackingSession,
    bbox_iou,
    pixel_to_laser_angles,
)

MS = 1_000_000
W, H = 640, 480
K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])


# ============================================================================ geometry


def test_principal_point_gives_zero_angles_and_signs_follow_the_image():
    zero = np.zeros(5)
    eye, t0 = np.eye(3), np.zeros((3, 1))
    az, po, p_l = pixel_to_laser_angles(cv2, (320.0, 240.0), K, zero, eye, t0, 21.0)
    assert az == pytest.approx(0.0) and po == pytest.approx(0.0) and p_l[2] == pytest.approx(21.0)
    az, po, _ = pixel_to_laser_angles(cv2, (420.0, 240.0), K, zero, eye, t0, 21.0)
    assert az == pytest.approx(math.atan(0.1)) and po == pytest.approx(0.0)  # right of centre -> +az
    az, po, _ = pixel_to_laser_angles(cv2, (320.0, 140.0), K, zero, eye, t0, 21.0)
    assert po == pytest.approx(math.atan(0.1)) and az == pytest.approx(0.0)  # above centre -> +po


def test_laser_axis_pixel_gives_zero_angles_with_realistic_extrinsics_and_distortion():
    # Small camera->laser rotation, ~18 cm lever arm, radial/tangential distortion.
    r_lc = cam._rotation_from_rpy_deg((0.3, -0.2, 0.1))
    t_lc = np.array([[0.01], [-0.18], [0.02]])
    dist = np.array([-0.12, 0.05, 0.0004, -0.0003, 0.0])
    p_c_axis = cam._laser_axis_point_in_camera(r_lc, t_lc, 21.0)
    pixel = cam._project_camera_point_to_pixel(cv2, p_c_axis, K, dist)
    az, po, p_l = pixel_to_laser_angles(cv2, pixel, K, dist, r_lc, t_lc, float(p_c_axis[2]))
    assert abs(az) < 1e-6 and abs(po) < 1e-6
    assert p_l[2] == pytest.approx(21.0, abs=1e-4)


def test_parallax_error_of_the_fixed_range_matches_the_lever_arm_formula():
    # Target on the laser axis at 20 m, but Z fixed at 21 m: error ~ |t_lc| * |1/20 - 1/21|.
    eye, t_lc = np.eye(3), np.array([[0.0], [-0.18], [0.0]])
    p_c = cam._laser_axis_point_in_camera(eye, t_lc, 20.0)
    pixel = cam._project_camera_point_to_pixel(cv2, p_c, K, np.zeros(5))
    _, po, _ = pixel_to_laser_angles(cv2, pixel, K, np.zeros(5), eye, t_lc, 21.0)
    assert abs(po) == pytest.approx(0.18 * (1 / 20 - 1 / 21), rel=0.02)


def test_geometry_rejects_non_finite_input():
    zero, eye, t0 = np.zeros(5), np.eye(3), np.zeros((3, 1))
    assert pixel_to_laser_angles(cv2, (float("nan"), 1.0), K, zero, eye, t0, 21.0) is None
    assert pixel_to_laser_angles(cv2, (1.0, 1.0), K, zero, eye, t0, 0.0) is None


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert bbox_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


# ============================================================================ fakes


@dataclass
class _Res:
    center_xy: tuple[float, float]
    size_wh: tuple[float, float]
    peak: float
    apce: float


class FakeTracker:
    """Tracks a programmable target: ``center``, ``size``, ``peak``, ``apce`` can be changed between frames."""

    def __init__(self):
        self.inits = []
        self.updates = 0
        self.center, self.size, self.peak, self.apce = (320.0, 240.0), (100.0, 60.0), 10.0, 20.0
        self.fail_update = False

    def init(self, frame, center_xy, size_wh):
        self.inits.append((frame, tuple(center_xy), tuple(size_wh)))
        self.center, self.size = tuple(center_xy), tuple(size_wh)

    def update(self, frame):
        self.updates += 1
        if self.fail_update:
            raise RuntimeError("boom")
        return _Res(self.center, self.size, self.peak, self.apce)


def _det(cx=320.0, cy=240.0, w=100.0, h=60.0, score=0.9):
    return Detection((cx - w / 2, cy - h / 2, w, h), score)


def _session(**cfg):
    tracker = FakeTracker()
    s = SiamFCTrackingSession(tracker, DetectionConfirmer(ConfirmerConfig(frames_required=2)), SessionConfig(**cfg))
    return s, tracker


def _frame(v=0):
    return np.full((H, W), v, np.uint8)


def _acquire(s, frame, ts, dets):
    req = s.detection_request((H, W), ts, None)
    assert req is not None and req.purpose == "acquire" and req.roi_xywh is None
    s.apply_detection(DetectionOutcome(req, tuple(dets), frame, ts, 0))


def _lock_on(s, t0=0):
    f_det = _frame(1)
    _acquire(s, f_det, t0, [_det()])
    _acquire(s, f_det, t0 + 100 * MS, [_det()])
    return s.on_frame(_frame(2), t0 + 120 * MS), f_det


# ============================================================================ session


def test_session_starts_searching_and_rate_limits_full_frame_detection():
    s, _ = _session(missing_hz=10.0)
    obs = s.on_frame(_frame(), 0)
    assert obs.state == STATE_SEARCHING and obs.reason == REASON_SEARCHING and obs.bbox_xywh is None
    assert s.detection_request((H, W), 50 * MS, 0) is None  # 50 ms < 100 ms period
    req = s.detection_request((H, W), 100 * MS, 0)
    assert req.purpose == "acquire" and req.roi_xywh is None


def test_confirmed_detection_initialises_on_the_detection_frame_then_updates_on_the_current_one():
    s, tr = _session()
    f_det = _frame(1)
    _acquire(s, f_det, 0, [_det(300, 200)])
    assert s.on_frame(_frame(2), 10 * MS).state == STATE_SEARCHING  # one detection is not enough
    _acquire(s, f_det, 100 * MS, [_det(305, 202)])
    obs = s.on_frame(_frame(3), 120 * MS)
    assert obs.state == STATE_TRACKING and obs.reason == "tracking"
    frame, center, size = tr.inits[0]
    assert frame is f_det and center == (305.0, 202.0) and size == (100.0, 60.0)
    assert tr.updates == 1  # the catch-up update on the current frame
    assert s.epoch == 1 and obs.confidence == pytest.approx(1.0)


def test_weak_confidence_leads_to_lost_then_recovery():
    s, tr = _session(lost_frames=3, lost_conf=0.25, ref_frames=1)
    _lock_on(s)
    tr.peak = 1.0  # 1/10 of the reference -> confidence sqrt(0.1 * 1) ~ 0.32 > 0.25: still fine
    assert s.on_frame(_frame(), 200 * MS).state == STATE_TRACKING
    tr.peak = 0.1  # confidence ~0.1
    for i in range(2):
        assert s.on_frame(_frame(), (210 + i) * MS).state == STATE_TRACKING
    obs = s.on_frame(_frame(), 220 * MS)
    assert obs.state == STATE_LOST and obs.reason == REASON_LOST and obs.bbox_xywh is not None
    tr.peak = 10.0
    for i in range(2):
        assert s.on_frame(_frame(), (230 + i) * MS).state == STATE_LOST
    assert s.on_frame(_frame(), 240 * MS).state == STATE_TRACKING
    assert s.counters["lost"] == 1 and s.counters["recovered"] == 1


def test_lost_too_long_drops_the_tracker():
    s, tr = _session(lost_frames=1, max_lost_s=0.5, ref_frames=1)
    _lock_on(s)
    tr.peak = -3.0  # negative raw peak -> confidence 0
    assert s.on_frame(_frame(), 200 * MS).state == STATE_LOST
    assert s.on_frame(_frame(), 600 * MS).state == STATE_LOST
    obs = s.on_frame(_frame(), 800 * MS)  # 0.6 s after entering "lost"
    assert obs.state == STATE_SEARCHING and obs.reason == REASON_SEARCHING
    assert s.counters["dropped"] == 1


def test_target_leaving_the_frame_is_lost_immediately():
    s, tr = _session(lost_frames=5)
    _lock_on(s)
    tr.center = (W + 10.0, 240.0)
    assert s.on_frame(_frame(), 200 * MS).state == STATE_LOST


def test_verification_runs_on_the_full_frame_at_its_own_rate():
    s, tr = _session(redetect_hz=2.0)
    _lock_on(s)
    assert s.detection_request((H, W), 300 * MS, 200 * MS) is None  # 0.1 s < 0.5 s period
    req = s.detection_request((H, W), 800 * MS, 200 * MS)
    assert req.purpose == "verify" and req.epoch == 1 and req.roi_xywh is None
    assert req.tracked_bbox_xywh == (270.0, 210.0, 100.0, 60.0)


def test_verification_candidate_filters_by_distance_and_size():
    s, _ = _session(verify_radius=1.5, verify_radius_min_px=50, verify_max_size_ratio=1.5)
    tracked = (270.0, 210.0, 100.0, 60.0)  # centre (320, 240), larger side 100 -> radius 150 px
    near_big = _det(330, 240, 300, 200, score=0.9)  # too large (the false positives seen in the replay)
    far = _det(600, 240, score=0.8)  # 280 px away
    near = _det(350, 250, score=0.4)
    assert s.verification_candidate([near_big, far, near], tracked) is near
    assert s.verification_candidate([near_big, far], tracked) is None


def test_verification_with_a_drifted_tracker_reinitialises_it_after_confirmation():
    s, tr = _session(redetect_hz=2.0, reinit_iou=0.5, reinit_confirm=2)
    _lock_on(s)
    req = s.detection_request((H, W), 1000 * MS, 0)
    f1, f2 = _frame(7), _frame(8)
    s.apply_detection(DetectionOutcome(req, (_det(400, 240),), f1, 1000 * MS, 0))  # IoU with (320, 240) box = 0.11
    s.on_frame(_frame(), 1010 * MS)
    assert len(tr.inits) == 1  # one disagreeing verification is not enough
    s.apply_detection(DetectionOutcome(req, (_det(405, 241),), f2, 1500 * MS, 0))
    s.on_frame(_frame(), 1510 * MS)
    assert len(tr.inits) == 2 and tr.inits[1][0] is f2 and tr.inits[1][1] == (405.0, 241.0)
    assert s.epoch == 2 and s.counters["reinits_from_verify"] == 1


def test_inconsistent_disagreeing_verifications_do_not_reinitialise():
    s, tr = _session(redetect_hz=2.0, reinit_confirm=2, verify_radius=3.0)
    _lock_on(s)
    req = s.detection_request((H, W), 1000 * MS, 0)
    s.apply_detection(DetectionOutcome(req, (_det(420, 240),), _frame(), 1000 * MS, 0))
    s.apply_detection(DetectionOutcome(req, (_det(220, 240),), _frame(), 1500 * MS, 0))  # jumped 2 sides
    s.on_frame(_frame(), 1510 * MS)
    assert len(tr.inits) == 1


def test_consistent_verification_keeps_the_tracker_and_misses_lead_to_lost():
    s, tr = _session(redetect_hz=2.0, verify_miss_limit=2)
    _lock_on(s)
    req = s.detection_request((H, W), 1000 * MS, 0)
    s.apply_detection(DetectionOutcome(req, (_det(325, 241),), _frame(), 1000 * MS, 0))
    s.on_frame(_frame(), 1010 * MS)
    assert len(tr.inits) == 1 and s.state == STATE_TRACKING
    s.apply_detection(DetectionOutcome(req, (), _frame(), 1500 * MS, 0))
    assert s.state == STATE_TRACKING
    s.apply_detection(DetectionOutcome(req, (_det(620, 60),), _frame(), 2000 * MS, 0))  # far away: still a miss
    assert s.state == STATE_LOST


def test_stale_verification_results_are_ignored():
    s, tr = _session(redetect_hz=2.0, reinit_confirm=1)
    _lock_on(s)
    req = s.detection_request((H, W), 1000 * MS, 0)
    # tracker re-initialised (epoch 2) before the verify result arrives
    s.apply_detection(DetectionOutcome(req, (_det(400, 240),), _frame(), 1000 * MS, 0))
    s.on_frame(_frame(), 1010 * MS)
    assert s.epoch == 2
    s.apply_detection(DetectionOutcome(req, (_det(330, 240),), _frame(), 1020 * MS, 0))  # epoch 1 request
    s.on_frame(_frame(), 1030 * MS)
    assert len(tr.inits) == 2


def test_weak_detections_never_start_a_track():
    s, tr = _session(acquire_min_score=0.5)
    f = _frame()
    for k in range(4):
        _acquire(s, f, k * 100 * MS, [_det(score=0.45)])
    assert s.on_frame(_frame(), 500 * MS).state == STATE_SEARCHING and not tr.inits


def test_loss_from_verification_is_not_recovered_by_the_tracker():
    s, tr = _session(redetect_hz=2.0, verify_miss_limit=1, lost_frames=1, max_lost_s=5.0)
    _lock_on(s)
    req = s.detection_request((H, W), 1000 * MS, 0)
    s.apply_detection(DetectionOutcome(req, (), _frame(), 1000 * MS, 0))  # detector sees no drone there
    assert s.state == STATE_LOST
    for k in range(5):  # tracker confidence stays at 1.0 on whatever it follows
        assert s.on_frame(_frame(), (1010 + k) * MS).state == STATE_LOST
    _acquire(s, _frame(), 1100 * MS, [_det(200, 200)])
    _acquire(s, _frame(), 1200 * MS, [_det(200, 200)])
    assert s.on_frame(_frame(), 1210 * MS).state == STATE_TRACKING  # only a confirmed detection resumes
    assert s.counters["recovered"] == 0 and len(tr.inits) == 2


def test_target_out_of_frame_is_not_recovered_by_the_tracker():
    s, tr = _session(lost_frames=1, max_lost_s=5.0)
    _lock_on(s)
    tr.center = (W + 10.0, 240.0)
    s.on_frame(_frame(), 200 * MS)
    tr.center = (320.0, 240.0)  # tracker jumps back with high confidence
    for k in range(3):
        assert s.on_frame(_frame(), (210 + k) * MS).state == STATE_LOST


def test_tracker_exception_returns_to_searching():
    s, tr = _session()
    _lock_on(s)
    tr.fail_update = True
    obs = s.on_frame(_frame(), 200 * MS)
    assert obs.state == STATE_SEARCHING and obs.reason == REASON_TRACKER_ERROR
    assert s.counters["tracker_errors"] == 1


def test_lost_state_reacquires_through_the_detector():
    s, tr = _session(lost_frames=1, max_lost_s=5.0, ref_frames=1)
    _lock_on(s)
    tr.peak = -1.0
    s.on_frame(_frame(), 200 * MS)
    assert s.state == STATE_LOST
    _acquire(s, _frame(), 300 * MS, [_det(200, 200)])
    _acquire(s, _frame(), 400 * MS, [_det(202, 200)])
    tr.peak = 10.0
    obs = s.on_frame(_frame(), 410 * MS)
    assert obs.state == STATE_TRACKING and len(tr.inits) == 2 and s.epoch == 2


def test_confidence_uses_the_median_of_the_first_frames_as_reference():
    s, tr = _session(ref_frames=3, lost_frames=10)
    tr.peak, tr.apce = 20.0, 20.0  # inflated first update
    _lock_on(s)
    tr.peak = 10.0
    s.on_frame(_frame(), 200 * MS)
    s.on_frame(_frame(), 210 * MS)  # references: median(20, 10, 10) = 10 for the peak
    obs = s.on_frame(_frame(), 220 * MS)
    assert obs.confidence == pytest.approx(1.0)


# ============================================================================ adapter


class FakeProducer:
    """Frames 20 ms apart (50 fps), starting at the current monotonic time."""

    def __init__(self):
        self.packet = None
        self.started = False
        self.t0 = time.monotonic_ns()

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def latest(self):
        return self.packet

    def push(self, gray, ts_ns=None):
        seq = 1 if self.packet is None else self.packet.sequence + 1
        ts = self.t0 + seq * 20 * MS if ts_ns is None else ts_ns
        self.packet = cam._FramePacket(gray=gray, ts_capture_ns=int(ts), sequence=seq)

    def stats(self):
        return {"sequence": 0 if self.packet is None else self.packet.sequence}


class FakeDetector:
    def __init__(self, dets=()):
        self.dets = list(dets)
        self.calls = []

    def detect(self, gray, roi_xywh=None):
        self.calls.append(roi_xywh)
        return list(self.dets)


class InlineExecutor:
    """Runs the job at submit time; like ThreadPoolExecutor, a job exception goes into the Future."""

    def submit(self, fn, *args):
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            f.set_exception(exc)
        return f

    def shutdown(self, wait=False, cancel_futures=False):
        pass


CALIB = {
    "camera_matrix": K,
    "dist_coeffs": np.zeros(5),
    "R_lc": np.eye(3),
    "t_lc_m": np.zeros((3, 1)),
    "image_width": W,
    "image_height": H,
}


def _adapter(gating=cam.GatingConfig(max_age_s=0.25, conf_min=0.15, z_min_m=0.25), debug=False, dets=(_det(),), **session):
    cfg = {
        "debug_view": debug,
        "siamfc": {"weights_path": "", "target_range_m": 21.0, "missing_hz": 1000.0, "redetect_hz": 0.0, "ref_frames": 1, **session},
        "detector": {"confirm_frames": 2, "confirm_max_gap_s": 1.0},
    }
    tracker, detector, producer = FakeTracker(), FakeDetector(dets), FakeProducer()
    a = IdsSiamFCCorrectionSourceAdapter(
        cfg, gating, producer=producer, tracker=tracker, detector=detector, calibration=CALIB,
        executor_factory=InlineExecutor,
    )
    return a, tracker, detector, producer


def _step(a, producer, gray=None):
    producer.push(_frame() if gray is None else gray)
    return a.read_candidate(time.monotonic_ns())


def test_adapter_not_started():
    a, *_ = _adapter()
    assert a.read_candidate(time.monotonic_ns()).reason == "source_not_started"
    assert a.mode == "ids_siamfc"


def test_adapter_end_to_end_searching_then_tracking_with_angles():
    a, tr, det, producer = _adapter()
    a.start()
    assert producer.started
    assert a.read_candidate(time.monotonic_ns()).reason == "frame_not_available"
    # frame 1: searching, detection submitted (inline executor -> done immediately)
    c1 = _step(a, producer)
    assert not c1.reliable and c1.reason == REASON_SEARCHING
    # frame 2: first detection consumed (1/2), second submitted
    assert _step(a, producer).reason == REASON_SEARCHING
    # frame 3: second detection consumed -> confirmed -> tracker initialised and updated
    c_lock = _step(a, producer)
    assert c_lock.reliable and c_lock.reason == "ok"
    assert c_lock.d_az_rad == pytest.approx(0.0) and c_lock.d_po_rad == pytest.approx(0.0)  # detection at the principal point
    # frame 4: the tracker follows the target to (420, 140)
    tr.center = (420.0, 140.0)
    c3 = _step(a, producer)
    assert c3.reliable and c3.reason == "ok"
    az, po, p_l = pixel_to_laser_angles(cv2, (420.0, 140.0), K, np.zeros(5), np.eye(3), np.zeros((3, 1)), 21.0)
    assert c3.d_az_rad == pytest.approx(az) and c3.d_po_rad == pytest.approx(po)
    assert c3.d_az_rad > 0 and c3.d_po_rad > 0  # right of and above the principal point
    assert c3.z_m == pytest.approx(21.0) and c3.confidence == pytest.approx(1.0)
    assert c3.marker_id is None and c3.age_s is not None
    stats = a.fast_stats()
    assert stats["tracker_state"] == STATE_TRACKING and stats["detections_submitted"] >= 2
    assert stats["session"]["inits"] == 1


def test_adapter_reuses_the_last_candidate_and_marks_it_stale():
    a, tr, _, producer = _adapter()
    a.start()
    for _ in range(3):
        c = _step(a, producer)
    assert c.reliable
    reused = a.read_candidate(time.monotonic_ns())  # same sequence
    assert reused.reliable and reused.d_az_rad == c.d_az_rad
    stale = a.read_candidate(int(c.ts_capture_ns) + int(0.3e9))
    assert not stale.reliable and stale.reason == "sample_stale"


def test_adapter_low_confidence_is_not_reliable_before_the_target_is_lost():
    a, tr, _, producer = _adapter(lost_frames=10)
    a.start()
    for _ in range(3):
        _step(a, producer)
    tr.peak = 0.1  # confidence ~0.1 < conf_min 0.15, but only 1 weak frame
    c = _step(a, producer)
    assert not c.reliable and c.reason == "confidence_low" and c.confidence == pytest.approx(0.1, abs=0.01)


def test_adapter_reports_lost():
    a, tr, _, producer = _adapter(lost_frames=1)
    a.start()
    for _ in range(3):
        _step(a, producer)
    tr.center = (-50.0, 10.0)
    c = _step(a, producer)
    assert not c.reliable and c.reason == REASON_LOST


def test_adapter_z_below_min_is_refused():
    a, tr, _, producer = _adapter(gating=cam.GatingConfig(z_min_m=30.0))
    a.start()
    for _ in range(3):
        c = _step(a, producer)
    assert not c.reliable and c.reason == "z_below_min" and c.z_m == pytest.approx(21.0)


def test_adapter_detector_failure_is_counted_and_does_not_crash():
    a, tr, det, producer = _adapter()

    def boom(gray, roi=None):
        raise RuntimeError("detector down")

    det.detect = boom
    a.start()
    for _ in range(3):
        c = _step(a, producer)
    assert c.reason == REASON_SEARCHING
    assert a.fast_stats()["detection_failures"] >= 2


def test_adapter_debug_snapshot_has_the_keys_used_by_cuasp():
    a, tr, _, producer = _adapter(debug=True)
    a.start()
    for _ in range(3):
        _step(a, producer)
    a._last_debug_ns = 0  # bypass the debug rate limit for the test
    _step(a, producer)
    snap = a.get_debug_snapshot()
    assert snap["available"] and snap["tracking_backend"] == "siamfc" and snap["z_mode"] == "fixed"
    for key in ("frame_gray", "principal_point_px", "laser_axis_pixel_px", "laser_axis_range_m", "pbvs_target_pixel_px",
                "markers", "expected_marker_ids", "tile_state", "target_point", "siamfc_bbox_xywh", "siamfc_score",
                "tracker_state", "detections"):
        assert key in snap, key
    assert snap["markers"] == [] and snap["laser_axis_range_source"] == "fixed"
    # identity extrinsics: the laser axis projects onto the principal point
    np.testing.assert_allclose(snap["laser_axis_pixel_px"], snap["principal_point_px"], atol=1e-6)


def test_adapter_stop_resets_state():
    a, tr, _, producer = _adapter()
    a.start()
    _step(a, producer)
    a.stop()
    assert not a.started and not producer.started
    assert a.read_candidate(time.monotonic_ns()).reason == "source_not_started"


# ============================================================================ configuration and routing


def _full_cfg(**siamfc):
    return {
        "siamfc": {"weights_path": "w.pth", "target_range_m": 21.0, **siamfc},
        "detector": {"weights_path": "d.pt"},
    }


def test_config_from_dicts_and_validation():
    c = SiamFCSourceConfig.from_dicts(*_full_cfg(lost_frames=7, params="paper").values())
    c.validate()
    assert c.session.lost_frames == 7 and c.params == {} and c.detector.weights_path == "d.pt"
    with pytest.raises(ValueError, match="unknown siamfc"):
        SiamFCSourceConfig.from_dicts({"weights_path": "w", "target_range_m": 21, "lost_frame": 3}, {"weights_path": "d"})
    with pytest.raises(ValueError, match="unknown detector"):
        SiamFCSourceConfig.from_dicts({"weights_path": "w", "target_range_m": 21}, {"weights_path": "d", "conf": 0.2})
    with pytest.raises(ValueError, match="target_range_m"):
        SiamFCSourceConfig.from_dicts({"weights_path": "w"}, {"weights_path": "d"}).validate()
    with pytest.raises(ValueError, match="detector configuration"):
        SiamFCSourceConfig.from_dicts({"weights_path": "w", "target_range_m": 21}, {}).validate()
    with pytest.raises(ValueError, match="max_gap_s"):  # 2 Hz full-frame runs can never be 0.5 s apart
        SiamFCSourceConfig.from_dicts(
            {"weights_path": "w", "target_range_m": 21, "missing_hz": 2.0}, {"weights_path": "d", "confirm_max_gap_s": 0.5}
        ).validate()


def test_routing_ids_siamfc_builds_the_new_adapter():
    cfg = cam.CameraConfig(enable=True, mode="ids_siamfc", ids_opencv={"calibration": {"path": "x.yaml"}, **_full_cfg()})
    api = cam.CameraAPI(cfg)
    assert type(api._app._source).__name__ == "IdsSiamFCCorrectionSourceAdapter"
    assert api._app._source.mode == "ids_siamfc"


def test_routing_ids_siamfc_refuses_an_invalid_configuration():
    cfg = cam.CameraConfig(enable=True, mode="ids_siamfc", ids_opencv={"siamfc": {"weights_path": "w.pth"}, "detector": {"weights_path": "d.pt"}})
    with pytest.raises(ValueError, match="target_range_m"):
        cam.CameraAPI(cfg)
