"""Tracker-only comparison of SiamFC and SiamMask on labelled IDS recordings (P7 step 0).

No detector and no state machine: each tracker is initialised from the label box, runs on every
frame while the drone is present, and after a failure (IoU 0 with the label) waits 5 frames and
is re-initialised from the label (VOT-style). When the label says "absent" (drone out of frame)
the tracker is stopped and re-initialised on the next present frame. Both trackers see the same
decoded frames.

Per tracker: failures, IoU (boxes clipped to the frame), centre error in px and mrad (fx of the
calibration in use), signed centre bias, box shape (height / width ratio to the label), and how
well the confidence (same formula as siamfc_source: sqrt(clip(peak/peak_ref) clip(apce/apce_ref)),
references from the first frames after each init) separates good frames (IoU >= 0.5) from
drifting ones (IoU < 0.3): AUC, 0.5 = useless, 1 = perfect.

Usage (from cuasp_siamfc/):
  .venv/Scripts/python experiments/siammask/compare_ids.py --out ../runs/siammask_step0 REC [REC ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "control"))
sys.path.insert(0, str(HERE.parents[1] / "tools"))

from bench import fold_bn  # noqa: E402
from net import load_siammask  # noqa: E402
from replay_siamfc_source import read_gt  # noqa: E402
from suncubes.siamfc.net import SiamFCNet  # noqa: E402
from suncubes.siamfc.tracker import SiamFCTracker  # noqa: E402
from tracker import SiamMaskTracker  # noqa: E402

FX = 24119.0  # calibration_legacy_20260921_x_test.yaml, the calibration in use
REF_FRAMES = 5
SKIP_AFTER_FAILURE = 5


def clip_box(b, w, h):
    x0, y0 = max(b[0], 0.0), max(b[1], 0.0)
    x1, y1 = min(b[0] + b[2], float(w)), min(b[1] + b[3], float(h))
    return (x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))


def iou_edges(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def auc(good, bad):
    good, bad = np.asarray(good), np.asarray(bad)
    if len(good) == 0 or len(bad) == 0:
        return float("nan")
    allv = np.concatenate([good, bad])
    ranks = allv.argsort().argsort().astype(float) + 1
    return float((ranks[:len(good)].sum() - len(good) * (len(good) + 1) / 2) / (len(good) * len(bad)))


class Runner:
    """One tracker under the evaluation protocol. ``outputs`` maps an output name to a function
    (result -> (edge box or None, pixel-centre point or None))."""

    def __init__(self, name, tracker, outputs):
        self.name, self.t, self.outputs = name, tracker, outputs
        self.active, self.wait_until, self.failures, self.inits = False, -1, 0, 0
        self.refs = []
        self.rows = []
        self.times = []

    def step(self, f, img, g_edge, W, H):
        if g_edge is None:
            self.active = False
            return
        if not self.active:
            if f >= self.wait_until:
                self.t.init(img, (g_edge[0] + g_edge[2] / 2 - 0.5, g_edge[1] + g_edge[3] / 2 - 0.5), (g_edge[2], g_edge[3]))
                self.active, self.refs = True, []
                self.inits += 1
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = self.t.update(img)
        torch.cuda.synchronize()
        self.times.append((time.perf_counter() - t0) * 1e3)
        if len(self.refs) < REF_FRAMES:
            self.refs.append((r.peak, r.apce))
        pr = np.median([p for p, _ in self.refs]); ar = np.median([a for _, a in self.refs])
        conf = float(np.sqrt(np.clip(r.peak / max(pr, 1e-12), 0, 1) * np.clip(r.apce / max(ar, 1e-12), 0, 1)))
        gc = (g_edge[0] + g_edge[2] / 2 - 0.5, g_edge[1] + g_edge[3] / 2 - 0.5)
        gclip = clip_box(g_edge, W, H)
        row = {"frame": f, "conf": conf, "peak": r.peak, "gw": g_edge[2], "gh": g_edge[3],
               "border": int(gclip[2] < g_edge[2] - 1 or gclip[3] < g_edge[3] - 1)}
        for out, fn in self.outputs.items():
            box, pt = fn(r)
            if box is None:
                row.update({f"{out}_iou": np.nan, f"{out}_dx": np.nan, f"{out}_dy": np.nan, f"{out}_w": np.nan, f"{out}_h": np.nan})
                continue
            if pt is None:
                pt = (box[0] + box[2] / 2 - 0.5, box[1] + box[3] / 2 - 0.5)
            row.update({f"{out}_iou": iou_edges(clip_box(box, W, H), gclip), f"{out}_dx": pt[0] - gc[0], f"{out}_dy": pt[1] - gc[1],
                        f"{out}_w": box[2], f"{out}_h": box[3]})
        self.rows.append(row)
        if row["box_iou"] == 0.0:  # failure on the tracking box
            self.failures += 1
            self.active, self.wait_until = False, f + SKIP_AFTER_FAILURE


def rpn_box(r):
    return (r.center_xy[0] - r.size_wh[0] / 2 + 0.5, r.center_xy[1] - r.size_wh[1] / 2 + 0.5, r.size_wh[0], r.size_wh[1]), None


def summarise(name, rows, runner):
    out = {"tracker": name, "frames": len(rows), "failures": runner.failures, "inits": runner.inits,
           "update_ms_median": float(np.median(runner.times)) if runner.times else float("nan")}
    keys = [k[:-4] for k in rows[0] if k.endswith("_iou")] if rows else []
    for k in keys:
        iou = np.array([r[f"{k}_iou"] for r in rows], float)
        ok = ~np.isnan(iou)
        dx = np.array([r[f"{k}_dx"] for r in rows], float)[ok]
        dy = np.array([r[f"{k}_dy"] for r in rows], float)[ok]
        err = np.hypot(dx, dy)
        hr = np.array([r[f"{k}_h"] / r["gh"] for r in rows], float)[ok]
        wr = np.array([r[f"{k}_w"] / r["gw"] for r in rows], float)[ok]
        good = iou[ok] >= 0.5
        out[k] = {"valid": int(ok.sum()), "iou_mean": float(iou[ok].mean()), "iou_gt_0.5": float(good.mean()),
                  "err_px_median": float(np.median(err)), "err_px_p90": float(np.percentile(err, 90)),
                  "err_mrad_median": float(np.median(err) / FX * 1e3), "err_mrad_p90": float(np.percentile(err, 90) / FX * 1e3),
                  "bias_dx_px": float(np.median(dx)), "bias_dy_px": float(np.median(dy)),
                  "h_ratio_median": float(np.median(hr)), "w_ratio_median": float(np.median(wr))}
    conf = np.array([r["conf"] for r in rows], float)
    iou = np.array([r["box_iou"] for r in rows], float)
    out["conf_auc_good_vs_drift"] = auc(conf[iou >= 0.5], conf[iou < 0.3])
    out["drift_frames"] = int((iou < 0.3).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--siamfc", type=Path, default=ROOT / "pretrained/networks/baseline-conv5_gray_e100.pth")
    ap.add_argument("--siammask", type=Path, default=ROOT / "pretrained/siammask/SiamMask_VOT.pth")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    fc = SiamFCTracker(SiamFCNet.from_file(args.siamfc), device="cuda")
    mnet, _ = load_siammask(str(args.siammask), "cuda")
    fold_bn(mnet)
    sm = SiamMaskTracker(mnet, crop="torch", mask="window", fp16=True)
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}  # merged across runs
    for rec in args.recordings:
        t_rec = time.perf_counter()
        frames = sorted((ROOT / "recordings" / rec / "frames").glob("*.jpg"))
        gt = read_gt(ROOT / "labels" / "ids" / f"{rec}.csv", len(frames))
        runners = [
            Runner("siamfc", fc, {"box": rpn_box}),
            Runner("siammask", sm, {"box": rpn_box,
                                    "maskbox": lambda r: (r.mask_box_xywh, r.mask_center_xy),
                                    "maskboxc": lambda r: (r.mask_box_xywh, None)}),
        ]
        H = W = None
        for f, path in enumerate(frames):
            g = gt[f]
            g_edge = None if np.isnan(g[0]) else tuple(float(v) for v in g)
            if g_edge is None and not any(rn.active for rn in runners):
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
            for rn in runners:
                rn.step(f, img, g_edge, W, H)
        summary[rec] = {}
        for rn in runners:
            with open(args.out / f"{rec}_{rn.name}.csv", "w", newline="") as fh:
                if rn.rows:
                    wr = csv.DictWriter(fh, fieldnames=list(rn.rows[0].keys()))
                    wr.writeheader()
                    wr.writerows(rn.rows)
            summary[rec][rn.name] = summarise(rn.name, rn.rows, rn)
        print(f"== {rec} ({time.perf_counter() - t_rec:.0f} s)", flush=True)
        for name, s in summary[rec].items():
            b = s["box"]
            line = (f"   {name:8s} failures {s['failures']:3d}  IoU {b['iou_mean']:.3f} (>0.5 {b['iou_gt_0.5'] * 100:5.1f}%)  "
                    f"err {b['err_mrad_median']:.2f}/{b['err_mrad_p90']:.2f} mrad  bias {b['bias_dx_px']:+.1f},{b['bias_dy_px']:+.1f} px  "
                    f"h/w ratio {b['h_ratio_median']:.2f}/{b['w_ratio_median']:.2f}  drift {s['drift_frames']}  conf AUC {s['conf_auc_good_vs_drift']:.2f}  "
                    f"{s['update_ms_median']:.1f} ms")
            print(line, flush=True)
            for k in ("maskbox", "maskboxc"):
                if k in s:
                    m = s[k]
                    print(f"            {k:8s} IoU {m['iou_mean']:.3f} (>0.5 {m['iou_gt_0.5'] * 100:5.1f}%)  err {m['err_mrad_median']:.2f}/{m['err_mrad_p90']:.2f} mrad  "
                          f"bias {m['bias_dx_px']:+.1f},{m['bias_dy_px']:+.1f} px  h/w ratio {m['h_ratio_median']:.2f}/{m['w_ratio_median']:.2f}  valid {m['valid']}", flush=True)
        summary_path.write_text(json.dumps(summary, indent=1))  # after every recording


if __name__ == "__main__":
    main()
