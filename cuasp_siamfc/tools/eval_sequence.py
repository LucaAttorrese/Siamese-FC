"""Run the SiamFC port on a VOT-format sequence and report accuracy and speed (validation V2).

Usage (from cuasp_siamfc/)::

    .venv/Scripts/python tools/eval_sequence.py ../demo-sequences/vot15_bag \
        --weights ../pretrained/networks/baseline-conv5_gray_e100.pth --gray

Initialisation follows the reference: the frame-1 groundtruth polygon is turned
into an axis-aligned box with ``get_axis_aligned_BB.m`` (1-based MATLAB centre,
converted to 0-based here). Accuracy is measured against the same axis-aligned
groundtruth boxes, as in CFNet ``run_tracker_evaluation.m``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))

import cv2  # noqa: E402
import torch  # noqa: E402

from suncubes.siamfc.net import SiamFCNet  # noqa: E402
from suncubes.siamfc.tracker import SiamFCTracker, default_device  # noqa: E402


def axis_aligned_bb(region: np.ndarray) -> tuple[float, float, float, float]:
    """Port of get_axis_aligned_BB.m -> (cx, cy, w, h) in MATLAB 1-based coordinates."""
    r = np.asarray(region, np.float64)
    if r.size == 4:
        x, y, w, h = r
        return x + w / 2, y + h / 2, w, h
    xs, ys = r[0::2], r[1::2]
    cx, cy = xs.mean(), ys.mean()
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    a1 = np.linalg.norm(r[0:2] - r[2:4]) * np.linalg.norm(r[2:4] - r[4:6])
    a2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(a1 / a2)
    return cx, cy, s * (x2 - x1) + 1, s * (y2 - y1) + 1


def iou(a: tuple, b: tuple) -> float:
    """IoU of two (cx, cy, w, h) boxes."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--gray", action="store_true", help="feed single-channel frames (IDS Mono8 path)")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", type=Path, help="optional CSV of per-frame results")
    args = ap.parse_args()

    gt = np.loadtxt(args.sequence / "groundtruth.txt", delimiter=",")
    files = sorted((args.sequence / "imgs").glob("*.jpg"))
    assert len(files) == len(gt), "frame/groundtruth count mismatch"

    tracker = SiamFCTracker(SiamFCNet.from_file(args.weights), device=args.device)

    def load(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if args.gray else cv2.IMREAD_COLOR)
        return img if args.gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # reference nets are RGB

    cx, cy, w, h = axis_aligned_bb(gt[0])
    frame = load(files[0])
    tracker.init(frame, (cx - 1.0, cy - 1.0), (w, h))

    ious, errs, times, rows = [], [], [], []
    for i in range(1, len(files)):
        frame = load(files[i])
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = tracker.update(frame)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        pred = (res.center_xy[0] + 1.0, res.center_xy[1] + 1.0, res.size_wh[0], res.size_wh[1])
        ref = axis_aligned_bb(gt[i])
        ious.append(iou(pred, ref))
        errs.append(float(np.hypot(pred[0] - ref[0], pred[1] - ref[1])))
        rows.append((i + 1, *pred, res.peak, res.apce, res.scale_index, ious[-1], errs[-1]))

    t = np.asarray(times[5:] if len(times) > 10 else times) * 1e3
    ious_a, errs_a = np.asarray(ious), np.asarray(errs)
    print(f"sequence={args.sequence.name} frames={len(files)} weights={args.weights.name} gray={args.gray} device={args.device}")
    print(f"mean IoU={ious_a.mean():.3f}  IoU>0.5: {np.mean(ious_a > 0.5) * 100:.1f}%  "
          f"centre err mean={errs_a.mean():.1f}px median={np.median(errs_a):.1f}px  prec@20px={np.mean(errs_a <= 20) * 100:.1f}%")
    print(f"update ms: mean={t.mean():.2f} p50={np.percentile(t, 50):.2f} p99={np.percentile(t, 99):.2f}  (~{1e3 / t.mean():.0f} Hz)")
    if args.out:
        header = "frame,cx1,cy1,w,h,peak,apce,scale_index,iou,center_err_px"
        np.savetxt(args.out, np.asarray(rows), delimiter=",", header=header, comments="", fmt="%.4f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
