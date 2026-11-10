"""suncubes.siamfc.detector: geometry, fake-model plumbing, confirmation logic, and one
real-weights check (skipped when the trained weights or the dataset are not present)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from suncubes.siamfc.detector import (
    ConfirmerConfig,
    Detection,
    DetectionConfirmer,
    DetectorConfig,
    UltralyticsDroneDetector,
    clip_roi,
    detections_from_xyxy,
)

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "runs" / "detector" / "yolo11s_uav_gray_1280_e40" / "weights" / "best.pt"
TEST_IMAGES = REPO / "datasets" / "yolo_uav_gray" / "images" / "test"


# ---------------------------------------------------------------- geometry helpers


def test_detection_properties():
    d = Detection((10.0, 20.0, 30.0, 40.0), 0.8)
    assert d.center_xy == (25.0, 40.0)
    assert d.size_wh == (30.0, 40.0)
    assert d.side == pytest.approx(np.sqrt(1200.0))


@pytest.mark.parametrize(
    ("roi", "expected"),
    [
        ((10, 5, 30, 40), (10, 5, 30, 40)),  # fully inside (frame 100 x 50)
        ((-5, -5, 20, 20), (0, 0, 15, 15)),  # clipped at the top-left
        ((90, 40, 50, 50), (90, 40, 10, 10)),  # clipped at the bottom-right (frame 100 x 50)
        ((10.4, 20.6, 5.2, 5.2), (10, 20, 6, 6)),  # floor / ceil to cover the requested area
        ((200, 10, 5, 5), None),  # outside
        ((10, 10, 0, 5), None),  # empty
        ((float("nan"), 0, 5, 5), None),
    ],
)
def test_clip_roi(roi, expected):
    assert clip_roi(roi, (50, 100)) == expected


def test_detections_from_xyxy_shifts_sorts_and_filters():
    xyxy = np.array([[0, 0, 10, 10], [5, 5, 25, 15], [3, 3, 3, 9], [1, 1, 2, 2]], float)
    scores = np.array([0.3, 0.9, 0.95, 0.05])
    dets = detections_from_xyxy(xyxy, scores, offset_xy=(100, 200), conf_min=0.1)
    # degenerate box (zero width) and the below-threshold one are dropped; sorted by score
    assert [d.score for d in dets] == [0.9, 0.3]
    # edge coordinates -> pixel centres: -0.5 on the top-left corner, width/height unchanged
    assert dets[0].bbox_xywh == (104.5, 204.5, 20.0, 10.0)
    assert dets[1].bbox_xywh == (99.5, 199.5, 10.0, 10.0)


def test_detections_from_xyxy_undoes_the_resize():
    # The region starting at (40, 10) was resized by 0.5 in x and 0.25 in y before inference.
    dets = detections_from_xyxy(np.array([[10, 4, 30, 8]], float), np.array([0.8]), offset_xy=(40, 10), scale_xy=(0.5, 0.25))
    # edges in the frame: x 20..60 -> 60..100, y 16..32 -> 26..42
    assert dets[0].bbox_xywh == (59.5, 25.5, 40.0, 16.0)
    assert dets[0].center_xy == (79.5, 33.5)


def test_pixel_centre_convention_of_a_known_box():
    # A box covering pixel columns 10..19 and rows 0..9 is [10, 0, 20, 10] in edge coordinates.
    d = detections_from_xyxy(np.array([[10, 0, 20, 10]], float), np.array([1.0]))[0]
    assert d.bbox_xywh == (9.5, -0.5, 10.0, 10.0)
    assert d.center_xy == (14.5, 4.5)  # mean of the covered pixel indices


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights_path": ""},
        {"weights_path": "w.pt", "conf_min": 0.0},
        {"weights_path": "w.pt", "conf_min": 1.0},
        {"weights_path": "w.pt", "imgsz": 1000},
        {"weights_path": "w.pt", "max_det": 0},
        {"weights_path": "w.pt", "nms_iou": 0.0},
        {"weights_path": "w.pt", "weights_sha256": "abc"},
    ],
)
def test_detector_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DetectorConfig(**kwargs).validate()


# ---------------------------------------------------------------- Ultralytics wrapper with a fake model


class _FakeTensor:
    def __init__(self, a):
        self._a = np.asarray(a, float)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = _FakeTensor(xyxy), _FakeTensor(conf)

    def __len__(self):
        return len(self.conf.numpy())


class _FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(xyxy, conf)


class _FakeModel:
    def __init__(self, xyxy=((10, 20, 50, 60),), conf=(0.9,)):
        self.calls = []
        self._xyxy, self._conf = np.asarray(xyxy, float).reshape(-1, 4), conf

    def predict(self, img, **kwargs):
        self.calls.append((img, kwargs))
        return [_FakeResult(self._xyxy, self._conf)]


def _fake_detector(model, **cfg):
    cfg.setdefault("max_box_frac", 1.0)  # the fake boxes are large relative to the tiny test frames
    return UltralyticsDroneDetector(DetectorConfig(weights_path="unused.pt", device="cpu", **cfg), model=model)


def test_wrapper_replicates_gray_to_three_channels_and_passes_settings():
    model = _FakeModel()
    det = _fake_detector(model, conf_min=0.2, imgsz=640, max_det=5)
    gray = np.random.default_rng(0).integers(0, 255, (100, 120), dtype=np.uint8)
    out = det.detect(gray)
    img, kw = model.calls[0]
    assert img.shape == (100, 120, 3) and img.dtype == np.uint8
    assert np.array_equal(img[:, :, 0], gray) and np.array_equal(img[:, :, 2], gray)
    assert kw["imgsz"] == 640 and kw["conf"] == 0.2 and kw["max_det"] == 5
    assert kw["quantize"] is None  # fp16 is never used on the CPU
    assert out == [Detection((9.5, 19.5, 40.0, 40.0), 0.9)]


def test_wrapper_roi_crops_and_shifts_back():
    model = _FakeModel(xyxy=((1, 2, 11, 22),), conf=(0.7,))
    det = _fake_detector(model)
    gray = np.zeros((100, 200), np.uint8)
    out = det.detect(gray, roi_xywh=(50, 30, 40, 60))
    img, _ = model.calls[0]
    assert img.shape == (60, 40, 3)
    assert out[0].bbox_xywh == (50.5, 31.5, 10.0, 20.0)


def test_wrapper_downscales_large_frames_before_replicating_channels():
    # imgsz 64 and a 128 x 200 frame: the single channel is resized to 64 x 41 (scale 0.32) first.
    model = _FakeModel(xyxy=((16, 8, 32, 16),), conf=(0.9,))
    det = _fake_detector(model, imgsz=64)
    out = det.detect(np.full((128, 200), 77, np.uint8))
    img, kw = model.calls[0]
    assert img.shape == (41, 64, 3) and kw["imgsz"] == 64
    assert np.all(img == 77)
    sx, sy = 64 / 200, 41 / 128
    x, y, w, h = out[0].bbox_xywh
    assert (x, y) == pytest.approx((16 / sx - 0.5, 8 / sy - 0.5))
    assert (w, h) == pytest.approx((16 / sx, 8 / sy))


def test_wrapper_drops_implausibly_large_or_small_boxes():
    # frame 100 x 200: larger side 200 -> max box side 0.3 * 200 = 60 px
    model = _FakeModel(xyxy=((0, 0, 150, 80), (10, 10, 50, 40), (5, 5, 8, 30)), conf=(0.9, 0.8, 0.7))
    det = _fake_detector(model, max_box_frac=0.3, min_box_px=4)
    out = det.detect(np.zeros((100, 200), np.uint8))
    assert [d.score for d in out] == [0.8]  # 150 px wide is too large, 3 px wide too small


def test_filter_box_size_uses_the_full_frame_even_with_a_roi():
    model = _FakeModel(xyxy=((0, 0, 50, 50),), conf=(0.9,))
    det = _fake_detector(model, max_box_frac=0.3)
    # ROI of 60 x 60 inside a 200 x 400 frame: a 50 px box is fine (limit 0.3 * 400 = 120 px)
    assert len(det.detect(np.zeros((200, 400), np.uint8), roi_xywh=(0, 0, 60, 60))) == 1


def test_wrapper_empty_roi_does_not_call_the_model():
    model = _FakeModel()
    det = _fake_detector(model)
    assert det.detect(np.zeros((100, 200), np.uint8), roi_xywh=(500, 500, 10, 10)) == []
    assert model.calls == []


def test_wrapper_rejects_wrong_frames():
    det = _fake_detector(_FakeModel())
    with pytest.raises(ValueError):
        det.detect(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        det.detect(np.zeros((10, 10), np.float32))
    assert det.detect(np.zeros((100, 120, 1), np.uint8))  # (H, W, 1) is accepted


def test_wrapper_refuses_missing_weights_and_wrong_hash(tmp_path):
    with pytest.raises(FileNotFoundError):
        UltralyticsDroneDetector(DetectorConfig(weights_path=str(tmp_path / "missing.pt")))
    w = tmp_path / "w.pt"
    w.write_bytes(b"not the right weights")
    with pytest.raises(ValueError, match="SHA256"):
        UltralyticsDroneDetector(DetectorConfig(weights_path=str(w), weights_sha256="0" * 64))


# ---------------------------------------------------------------- confirmation logic


def _d(cx, cy, side=100.0, score=0.9):
    return Detection((cx - side / 2, cy - side / 2, side, side), score)


MS = 1_000_000


def test_confirmer_needs_consecutive_consistent_detections():
    c = DetectionConfirmer(ConfirmerConfig(frames_required=3))
    assert c.update([_d(500, 500)], 0) is None
    assert c.update([_d(540, 500)], 100 * MS) is None  # moved 0.4 sides: consistent
    confirmed = c.update([_d(600, 520)], 200 * MS)
    assert confirmed is not None and confirmed.center_xy == (600.0, 520.0)
    assert c.streak == 3


def test_confirmer_picks_best_score_and_resets_on_empty_run():
    c = DetectionConfirmer(ConfirmerConfig(frames_required=2))
    c.update([_d(100, 100, score=0.3), _d(500, 500, score=0.9)], 0)
    assert c.update([_d(510, 500)], 100 * MS) is not None  # the 0.9 detection was the candidate
    assert c.update([], 200 * MS) is None and c.streak == 0
    assert c.update([_d(510, 500)], 300 * MS) is None  # streak restarted at 1


def test_confirmer_restarts_on_jump_size_change_or_gap():
    cfg = ConfirmerConfig(frames_required=2, max_center_shift=1.0, max_size_ratio=1.5, max_gap_s=0.5)
    c = DetectionConfirmer(cfg)
    c.update([_d(500, 500)], 0)
    assert c.update([_d(650, 500)], 100 * MS) is None  # jumped 1.5 sides
    assert c.streak == 1
    assert c.update([_d(650, 500, side=200)], 200 * MS) is None  # size doubled
    assert c.update([_d(650, 500, side=200)], 900 * MS) is None  # 0.7 s later: too old
    assert c.update([_d(650, 500, side=200)], 1000 * MS) is not None


def test_confirmer_ignores_non_monotonic_time():
    c = DetectionConfirmer(ConfirmerConfig(frames_required=2))
    c.update([_d(500, 500)], 1000 * MS)
    assert c.update([_d(500, 500)], 900 * MS) is None
    assert c.streak == 1


def test_confirmer_single_frame_mode():
    c = DetectionConfirmer(ConfirmerConfig(frames_required=1))
    assert c.update([_d(500, 500)], 0) is not None


def test_confirmer_config_validation():
    for bad in ({"frames_required": 0}, {"max_center_shift": 0.0}, {"max_size_ratio": 0.9}, {"max_gap_s": 0.0}):
        with pytest.raises(ValueError):
            ConfirmerConfig(**bad).validate()


# ---------------------------------------------------------------- real weights (optional)


@pytest.mark.skipif(not WEIGHTS.is_file() or not TEST_IMAGES.is_dir(), reason="trained weights or dataset not present")
def test_real_weights_find_the_drone_on_a_test_image_with_and_without_roi():
    pytest.importorskip("ultralytics")
    cv2 = pytest.importorskip("cv2")
    name = "dut_test_00001.jpg" if (TEST_IMAGES / "dut_test_00001.jpg").exists() else sorted(p.name for p in TEST_IMAGES.glob("dut_*.jpg"))[0]
    gray = cv2.imread(str(TEST_IMAGES / name), cv2.IMREAD_GRAYSCALE)
    label = (TEST_IMAGES.parent.parent / "labels" / "test" / f"{Path(name).stem}.txt").read_text().split()
    h, w = gray.shape
    cx, cy, bw, bh = (float(v) for v in label[1:5])
    gt_center = (cx * w, cy * h)
    det = UltralyticsDroneDetector(DetectorConfig(weights_path=str(WEIGHTS)))
    full = det.detect(gray)
    assert full, "no detection on the full frame"
    assert np.hypot(full[0].center_xy[0] - gt_center[0], full[0].center_xy[1] - gt_center[1]) < max(bw * w, bh * h)
    side = int(4 * max(bw * w, bh * h, 32))
    roi = (int(gt_center[0] - side / 2), int(gt_center[1] - side / 2), side, side)
    in_roi = det.detect(gray, roi_xywh=roi)
    assert in_roi, "no detection inside the ROI"
    assert np.hypot(in_roi[0].center_xy[0] - full[0].center_xy[0], in_roi[0].center_xy[1] - full[0].center_xy[1]) < 0.25 * max(bw * w, bh * h, 8)
