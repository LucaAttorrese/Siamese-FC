"""SiamMask port on the demo sequence (same protocol and metric as tools/eval_sequence.py).

Checks that the fast path (GPU crop, mask post-processing in the mask window, FP16) gives the
same trajectory as the faithful path (cv2 crop, mask warped onto the whole frame), and compares
the accuracy with SiamFC (V2: gray 0.796 IoU / 5.9 px, RGB 0.773 / 6.4 px).

Usage (from cuasp_siamfc/):
  .venv/Scripts/python experiments/siammask/compare_vot.py ../demo-sequences/vot15_bag \
      --weights ../pretrained/siammask/SiamMask_VOT.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))

from eval_sequence import axis_aligned_bb, iou  # noqa: E402
from net import load_siammask  # noqa: E402
from tracker import SiamMaskTracker  # noqa: E402

CONFIGS = [
    ("original: cv2 crop, full-frame mask, fp32", dict(crop="cv2", mask="full", fp16=False)),
    ("fast: GPU crop, window mask, fp32", dict(crop="torch", mask="window", fp16=False)),
    ("fast: GPU crop, window mask, fp16", dict(crop="torch", mask="window", fp16=True)),
    ("box only: GPU crop, fp32", dict(crop="torch", mask=None, fp16=False)),
]


def run(weights, files, gt, gray, cfg):
    net, _ = load_siammask(str(weights), "cuda")
    torch.backends.cudnn.benchmark = True
    tr = SiamMaskTracker(net, **cfg)
    load = (lambda p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) if gray else (lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR))
    cx, cy, w, h = axis_aligned_bb(gt[0])
    tr.init(load(files[0]), (cx - 1.0, cy - 1.0), (w, h))
    out = []
    for i in range(1, len(files)):
        frame = load(files[i])
        torch.cuda.synchronize()
        r = tr.update(frame, sync_timing=True)
        torch.cuda.synchronize()
        ref = axis_aligned_bb(gt[i])
        pred = (r.center_xy[0] + 1, r.center_xy[1] + 1, *r.size_wh)
        mb = None
        if r.mask_box_xywh is not None:
            x, y, bw, bh = r.mask_box_xywh  # pixel-edge box of the blob -> centre/size in 1-based centre coords
            mb = (x + bw / 2 + 0.5, y + bh / 2 + 0.5, bw, bh)
        out.append(dict(pred=pred, iou=iou(pred, ref), err=float(np.hypot(pred[0] - ref[0], pred[1] - ref[1])),
                        miou=iou(mb, ref) if mb else np.nan, peak=r.peak, t=r.timing_ms))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    args = ap.parse_args()
    gt = np.loadtxt(args.sequence / "groundtruth.txt", delimiter=",")
    files = sorted((args.sequence / "imgs").glob("*.jpg"))
    results = {}
    for gray in (False, True):
        for name, cfg in CONFIGS:
            res = run(args.weights, files, gt, gray, cfg)
            results[(gray, name)] = res
            ious = np.array([r["iou"] for r in res]); errs = np.array([r["err"] for r in res])
            miou = np.array([r["miou"] for r in res])
            tt = np.array([r["t"]["total"] for r in res[5:]])
            print(f"{'gray' if gray else 'BGR '} {name:44s} IoU {ious.mean():.3f}  IoU>0.5 {np.mean(ious > 0.5) * 100:5.1f}%  "
                  f"err mean {errs.mean():5.1f} median {np.median(errs):5.1f} px  mask-box IoU {np.nanmean(miou):.3f}  "
                  f"update {np.median(tt):5.2f} ms median (480x360)")
        base = results[(gray, CONFIGS[0][0])]
        for name, _ in CONFIGS[1:]:
            other = results[(gray, name)]
            d = [np.hypot(a["pred"][0] - b["pred"][0], a["pred"][1] - b["pred"][1]) for a, b in zip(base, other)]
            print(f"      centre difference vs original, {name}: median {np.median(d):.2f} px, max {np.max(d):.2f} px")


if __name__ == "__main__":
    main()
