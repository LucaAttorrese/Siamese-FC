"""Aggregate the per-recording CSVs of compare_ids.py: totals and per-scenario comparison.

Usage (from cuasp_siamfc/):
  .venv/Scripts/python experiments/siammask/summarize_ids.py ../runs/siammask_step0
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_ids import FX, auc  # noqa: E402


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) for k in (rows[0] if rows else [])}


def scenario(rec):
    m = re.match(r"\d{8}_\d{6}_(.*?)(_r\d)?$", rec)
    return m.group(1) if m else rec


def stats(d, out="box"):
    iou = d[f"{out}_iou"]
    ok = ~np.isnan(iou)
    err = np.hypot(d[f"{out}_dx"], d[f"{out}_dy"])[ok]
    asp = np.abs(np.log((d[f"{out}_h"] / d[f"{out}_w"]) / (d["gh"] / d["gw"])))[ok]
    return {"n": int(ok.sum()), "iou": float(iou[ok].mean()), "iou50": float((iou[ok] >= 0.5).mean()),
            "err50": float(np.median(err) / FX * 1e3), "err90": float(np.percentile(err, 90) / FX * 1e3),
            "dx": float(np.median(d[f"{out}_dx"][ok])), "dy": float(np.median(d[f"{out}_dy"][ok])),
            "aspect_err": float(np.median(asp)), "drift": int((iou[ok] < 0.3).sum())}


def main():
    run = Path(sys.argv[1])
    summary = json.loads((run / "summary.json").read_text())
    recs = sorted(summary)
    data = {t: {} for t in ("siamfc", "siammask")}
    for rec in recs:
        for t in data:
            p = run / f"{rec}_{t}.csv"
            if p.exists() and p.stat().st_size > 0:
                data[t][rec] = load(p)

    def cat(t, recs_):
        ds = [data[t][r] for r in recs_ if r in data[t]]
        return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]} if ds else None

    print(f"{len(recs)} recordings; tracker only, initialised from the labels, re-initialised 5 frames after a failure (IoU 0)\n")
    print(f"{'':30s} {'fail':>5s} {'IoU':>6s} {'>0.5':>6s} {'err50':>6s} {'err90':>6s} {'bias dx,dy px':>14s} {'aspect':>7s} {'drift':>6s} {'confAUC':>7s}")
    for label, t, out in (("SiamFC", "siamfc", "box"), ("SiamMask RPN box", "siammask", "box"),
                          ("SiamMask mask box", "siammask", "maskbox"), ("SiamMask mask box, centre", "siammask", "maskboxc")):
        d = cat(t, recs)
        s = stats(d, out)
        fails = sum(summary[r][t]["failures"] for r in recs if t in summary[r])
        a = auc(d["conf"][d["box_iou"] >= 0.5], d["conf"][d["box_iou"] < 0.3]) if out == "box" else float("nan")
        print(f"{label:30s} {fails:5d} {s['iou']:6.3f} {s['iou50'] * 100:5.1f}% {s['err50']:6.2f} {s['err90']:6.2f} "
              f"{s['dx']:+6.1f},{s['dy']:+6.1f} {s['aspect_err']:7.2f} {s['drift']:6d} {a:7.2f}")
    print("\nerr in mrad (median / p90); aspect: median |log(predicted h/w : label h/w)|; drift: frames with IoU < 0.3;")
    print("confAUC: confidence separating IoU >= 0.5 from IoU < 0.3 (0.5 useless, 1 perfect)\n")

    scen = {}
    for r in recs:
        scen.setdefault(scenario(r), []).append(r)
    print(f"{'scenario':28s} {'SiamFC fail/IoU/err50/drift':>30s}   {'SiamMask fail/IoU/err50/drift':>30s}")
    for sname, rs in scen.items():
        cells = []
        for t in ("siamfc", "siammask"):
            d = cat(t, rs)
            s = stats(d)
            f = sum(summary[r][t]["failures"] for r in rs)
            cells.append(f"{f:4d} {s['iou']:.3f} {s['err50']:.2f} {s['drift']:5d}")
        print(f"{sname + f' ({len(rs)})':28s} {cells[0]:>30s}   {cells[1]:>30s}")
    print("\nper recording (failures SiamFC -> SiamMask, IoU, drift frames):")
    for r in recs:
        a, b = summary[r]["siamfc"], summary[r]["siammask"]
        print(f"  {r[16:]:36s} fail {a['failures']:3d} -> {b['failures']:3d}   IoU {a['box']['iou_mean']:.3f} -> {b['box']['iou_mean']:.3f}   "
              f"drift {a['drift_frames']:4d} -> {b['drift_frames']:4d}   err50 {a['box']['err_mrad_median']:.2f} -> {b['box']['err_mrad_median']:.2f} mrad")


if __name__ == "__main__":
    main()
