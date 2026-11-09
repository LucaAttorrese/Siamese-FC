"""Matching, size buckets and F1 threshold of tools/eval_detector.py (no Ultralytics needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "eval_detector.py"


@pytest.fixture(scope="module")
def ed():
    spec = importlib.util.spec_from_file_location("eval_detector", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_size_bucket_edges(ed):
    assert ed.size_bucket(0.0) == "0-16"
    assert ed.size_bucket(15.99) == "0-16"
    assert ed.size_bucket(16.0) == "16-32"
    assert ed.size_bucket(127.9) == "64-128"
    assert ed.size_bucket(128.0) == ">=128"
    assert ed.size_bucket(5000.0) == ">=128"


def test_box_iou_known_values(ed):
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(ed.box_iou(a, b), [[1.0, 50 / 150, 0.0]])
    assert ed.box_iou(np.zeros((0, 4)), b).shape == (0, 3)


def test_match_highest_confidence_wins_and_gt_used_once(ed):
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float)  # both overlap the same gt
    conf = np.array([0.3, 0.9])
    tp, best = ed.match_image(pred, conf, gt)
    assert tp.tolist() == [False, True]  # the 0.9 prediction takes the gt, the 0.3 one is a false positive
    assert best.tolist() == [0.9]


def test_match_iou_below_threshold_is_miss(ed):
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[6, 0, 16, 10]], float)  # IoU = 4/16 = 0.25
    tp, best = ed.match_image(pred, np.array([0.8]), gt)
    assert tp.tolist() == [False]
    assert best.tolist() == [-1.0]


def test_match_without_gt_or_predictions(ed):
    tp, best = ed.match_image(np.array([[0, 0, 5, 5]], float), np.array([0.5]), np.zeros((0, 4)))
    assert tp.tolist() == [False] and len(best) == 0
    tp, best = ed.match_image(np.zeros((0, 4)), np.zeros(0), np.array([[0, 0, 5, 5]], float))
    assert len(tp) == 0 and best.tolist() == [-1.0]


def test_pr_curve_and_best_f1(ed):
    # 3 ground truths; predictions sorted by confidence: TP, TP, FP, TP
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    tp = np.array([True, True, False, True])
    p, r, thr = ed.pr_curve(conf, tp, n_gt=3)
    np.testing.assert_allclose(r, [1 / 3, 2 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(p, [1.0, 1.0, 2 / 3, 3 / 4])
    t, f1, pf, rf = ed.best_f1_threshold(conf, tp, n_gt=3)
    # F1 at 0.8: 2*1*(2/3)/(1+2/3) = 0.8 ; at 0.6: 2*0.75*1/1.75 = 0.857 -> best
    assert t == pytest.approx(0.6) and f1 == pytest.approx(6 / 7)
    assert pf == pytest.approx(0.75) and rf == pytest.approx(1.0)


def test_yolo_labels_to_pixels(ed):
    boxes = ed._yolo_to_xyxy(["0 0.5 0.5 0.1 0.2"], 1920, 1080)
    np.testing.assert_allclose(boxes, [[864, 432, 1056, 648]])
    assert ed._yolo_to_xyxy([], 1920, 1080).shape == (0, 4)
