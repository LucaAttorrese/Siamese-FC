"""Pure helpers of tools/replay_siamfc_source.py (no models, no camera)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "replay_siamfc_source.py"


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("replay_siamfc_source", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_gt_marks_absent_targets(rp, tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text("10 20 30 40\n-100 -100 -100 -100\n5,6,7,8\n1 2 0 4\n\n")
    gt = rp.read_gt(p)
    assert gt.shape == (4, 4)
    np.testing.assert_array_equal(gt[0], [10, 20, 30, 40])
    assert np.isnan(gt[1]).all() and np.isnan(gt[3]).all()
    np.testing.assert_array_equal(gt[2], [5, 6, 7, 8])
    assert rp.read_gt(p, n_frames=2).shape == (2, 4)


def test_gt_to_pixel_centre(rp):
    assert rp.gt_to_pixel_centre((10, 0, 10, 10)) == (9.5, -0.5, 10.0, 10.0)


def test_classify(rp):
    gt = (100.0, 100.0, 40.0, 20.0)  # centre (120, 110), tolerance 40 px
    assert rp.classify(False, (100, 100, 40, 20), gt, True) is None  # not reliable
    assert rp.classify(True, (100, 100, 40, 20), gt, False) is None  # no ground truth at all
    assert rp.classify(True, (130, 100, 40, 20), gt, True) == "correct"  # 30 px off
    assert rp.classify(True, (150, 100, 40, 20), gt, True) == "wrong_target"  # 50 px off
    assert rp.classify(True, (150, 100, 40, 20), None, True) == "drone_absent"


def test_longest_run(rp):
    assert rp.longest_run([False, True, True, False, True]) == 2
    assert rp.longest_run([]) == 0


def _rec(i, state, reliable, category=None, gt=None, err=None, iou=None, reason=None):
    return {"frame": i, "state": state, "reason": reason or ("ok" if reliable else "siamfc_searching"),
            "reliable": reliable, "category": category, "gt": gt, "err_px": err, "iou": iou, "read_ms": 7.0}


def test_summarize_counts_and_dangerous_segments(rp):
    g = (0.0, 0.0, 10.0, 10.0)
    recs = [
        _rec(0, "searching", False, gt=g),
        _rec(1, "tracking", True, "correct", g, 2.0, 0.8),
        _rec(2, "tracking", True, "wrong_target", g, 50.0, 0.0),
        _rec(3, "tracking", True, "wrong_target", g, 60.0, 0.0),
        _rec(4, "tracking", True, "drone_absent", None),
        _rec(5, "lost", False, gt=g, reason="siamfc_lost"),
    ]
    s = rp.summarize(recs, fps=50.0, has_gt=True)
    assert s["first_reliable_frame"] == 1 and s["first_correct_frame"] == 1
    assert (s["correct"], s["wrong_target"], s["drone_absent"]) == (1, 2, 1)
    assert s["frames_target_present"] == 5 and s["coverage"] == pytest.approx(0.2)
    assert s["longest_dangerous_segment_s"] == pytest.approx(3 / 50)
    assert s["after_first_track"]["tracking"] == pytest.approx(4 / 5)
    assert s["centre_error_px"]["max"] == 60.0 and s["iou"]["above_0.5"] == pytest.approx(1 / 3)
    assert s["reasons"] == {"ok": 4, "siamfc_lost": 1, "siamfc_searching": 1}


def test_summarize_without_ground_truth(rp):
    s = rp.summarize([_rec(0, "tracking", True)], fps=50.0, has_gt=False)
    assert s["reliable_frames"] == 1 and "coverage" not in s


def test_scale_gt_scales_edge_coordinates_and_keeps_absent_rows(rp):
    gt = np.array([[100.0, 50.0, 40.0, 20.0], [np.nan] * 4])
    out = rp.scale_gt(gt, 2040 / 1920, 1536 / 1080)
    np.testing.assert_allclose(out[0], [100 * 2040 / 1920, 50 * 1536 / 1080, 40 * 2040 / 1920, 20 * 1536 / 1080])
    assert np.isnan(out[1]).all()
    np.testing.assert_array_equal(gt[0], [100.0, 50.0, 40.0, 20.0])  # input untouched


def test_resize_frames_produces_the_requested_size(rp):
    frames = [np.zeros((1080, 1920), np.uint8), np.zeros((1536, 2040), np.uint8)]
    out = list(rp.resize_frames(iter(frames), (2040, 1536)))
    assert [f.shape for f in out] == [(1536, 2040), (1536, 2040)]
    assert out[1] is frames[1]  # already the right size: not copied


def test_summarize_detections_counts_boxes_per_score_level(rp):
    dets = [{"job": 1, "score": 0.2}, {"job": 1, "score": 0.6}, {"job": 2, "score": 0.3}]
    s = rp.summarize_detections(dets, n_frames=626, fps=62.6)  # 10 s
    assert s["boxes"] == 3 and s["jobs_with_boxes"] == 2 and s["max_score"] == 0.6
    assert (s["boxes_score_ge_0.15"], s["boxes_score_ge_0.25"], s["boxes_score_ge_0.5"]) == (3, 2, 1)
    assert s["boxes_score_ge_0.5_per_min"] == pytest.approx(6.0)
    empty = rp.summarize_detections([], n_frames=626, fps=62.6)
    assert empty["boxes"] == 0 and empty["max_score"] is None
