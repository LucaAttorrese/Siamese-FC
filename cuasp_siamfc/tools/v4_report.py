"""V4 summary of replays against the IDS labels (phase 2 baseline, phase 3 comparisons).

Reads the per-frame CSVs written by ``tools/replay_siamfc_source.py --gt labels/ids/<rec>.csv
--out <dir>/<rec>.csv`` (the ground truth is in the CSV) and reports, per recording, per
scenario (recording name without timestamp and ``_rN``) and in total:

- safety: reliable frames on the wrong target or with the drone absent, longest such segment.
  Pointing at nothing right after the drone left the frame ("after exit": the time the
  runtime needs to declare the loss) is separated from the other absent-drone frames
  (locked on something while the drone is out of the frame);
- coverage: correct reliable frames / frames with the drone present, also without the
  "border" frames (label box within ``--edge-margin`` px of a frame edge: drone partly out);
- accuracy while correct: IoU of the boxes clipped to the frame (label convention: the
  drone partly out of the frame is labelled by the visible part), centre error in px and in
  mrad (``--calibration`` fx, fy);
- timing: first correct frame, re-acquisition after each re-entry of the drone into the
  frame (stretches that end without a correct frame are counted as missed), loss delay after
  each exit.

Replays made without ``--gt`` can be evaluated with ``--labels labels/ids`` (same rule as the
replay). Empty-scene replays (``--empty``: folders with the ``.json`` summaries of ``--empty-scene``
runs) add the false-alarm count: every reliable frame there points at nothing.

Usage (from cuasp_siamfc/)::

    .venv/Scripts/python tools/v4_report.py ../runs/v4_baseline --empty ../runs/v4_blockA --out ../runs/v4_baseline/report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Optional, Sequence

import numpy as np

_CONTROL = Path(__file__).resolve().parents[1] / "control"
DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "calibration" / "calibration_legacy_20260921_x_test.yaml"
REPORT_VERSION = "2026-11-18-v4-report-v1"


# ============================================================================ loading


def scenario_of(name: str) -> str:
    """'20261117_093341_B3_rapido_in_volo_r2' -> 'B3_rapido_in_volo'."""
    s = re.sub(r"^\d{8}_\d{6}_", "", name)
    return re.sub(r"_r\d+$", "", s)


def _f(v: str) -> Optional[float]:
    return float(v) if v not in ("", "None", "nan") else None


def load_replay(path: Path) -> dict[str, Any]:
    """Per-frame arrays of a replay CSV. Boxes are in pixel-centre coordinates (NaN when missing)."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError(f"{path}: empty replay CSV")
    if "gx" not in rows[0]:
        raise ValueError(f"{path}: not a replay CSV")
    n = len(rows)
    out = {
        "frame": np.array([int(r["frame"]) for r in rows]),
        "reliable": np.array([r["reliable"] in ("1", "True") for r in rows], bool),
        "category": [r["category"] for r in rows],
        "box": np.full((n, 4), np.nan),
        "gt": np.full((n, 4), np.nan),
    }
    if not np.array_equal(out["frame"], np.arange(n)):
        raise ValueError(f"{path}: frames are not 0..{n - 1}")
    for i, r in enumerate(rows):
        b = [_f(r[k]) for k in ("bx", "by", "bw", "bh")]
        g = [_f(r[k]) for k in ("gx", "gy", "gw", "gh")]
        if None not in b:
            out["box"][i] = b
        if None not in g:
            out["gt"][i] = g
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    out["fps"] = float(meta.get("fps", 62.6))
    size = (meta.get("settings") or {}).get("frame_size")
    out["frame_wh"] = tuple(size) if size else None
    return out


def attach_labels(rep: dict[str, Any], label_path: Path) -> None:
    """Ground truth and categories of a replay made without ``--gt``, from a label file.

    Same conversion and same rule as ``replay_siamfc_source.py`` (``read_gt``,
    ``gt_to_pixel_centre``, ``classify``), so the result equals a replay with ``--gt``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import replay_siamfc_source as rp  # noqa: PLC0415

    n = len(rep["frame"])
    gt = rp.read_gt(label_path)
    if len(gt) != n:
        raise ValueError(f"{label_path.name}: {len(gt)} label rows for {n} replay frames")
    cats = []
    for i in range(n):
        g = None if np.isnan(gt[i, 0]) else rp.gt_to_pixel_centre(gt[i])
        rep["gt"][i] = g if g is not None else (np.nan,) * 4
        b = None if np.isnan(rep["box"][i, 0]) else tuple(rep["box"][i])
        cats.append(rp.classify(bool(rep["reliable"][i]), b, g, True) or "")
    rep["category"] = cats


def load_intrinsics(path: Path) -> tuple[float, float]:
    sys.path.insert(0, str(_CONTROL))
    from suncubes import camera as cam  # noqa: PLC0415

    k = np.asarray(cam._load_vs_calibration_for_camera(Path(path))["camera_matrix"], float)
    return float(k[0, 0]), float(k[1, 1])


# ============================================================================ metrics


def clip_box_pc(box: Sequence[float], frame_wh: tuple[int, int]) -> Optional[tuple[float, float, float, float]]:
    """Pixel-centre box clipped to the image ([-0.5, W-0.5] x [-0.5, H-0.5]); None when nothing is left."""
    x, y, w, h = (float(v) for v in box)
    x0, y0 = max(x, -0.5), max(y, -0.5)
    x1, y1 = min(x + w, frame_wh[0] - 0.5), min(y + h, frame_wh[1] - 0.5)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def border_frames(gt: np.ndarray, frame_wh: tuple[int, int], margin: float) -> np.ndarray:
    """Label box (pixel-centre) within ``margin`` px of a frame edge (or crossing it)."""
    x, y, w, h = gt[:, 0] + 0.5, gt[:, 1] + 0.5, gt[:, 2], gt[:, 3]  # edge coordinates
    d = np.minimum.reduce([x, y, frame_wh[0] - (x + w), frame_wh[1] - (y + h)])
    return np.where(np.isnan(d), False, d < margin)


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end inclusive) of the True runs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def recording_metrics(rep: dict[str, Any], fx: float, fy: float, margin: float, frame_wh: tuple[int, int]) -> dict[str, Any]:
    n = len(rep["frame"])
    fps = rep["fps"]
    present = ~np.isnan(rep["gt"][:, 0])
    border = border_frames(rep["gt"], frame_wh, margin) & present
    cat = np.array(rep["category"], object)
    correct = cat == "correct"
    wrong = cat == "wrong_target"
    absent = cat == "drone_absent"

    # absent-drone frames that continue the pointing from the moment the drone left
    after_exit = np.zeros(n, bool)
    exit_delays = []
    for s, e in runs_of(~present):
        if s == 0:
            continue  # the recording starts without the drone: no exit
        k = s
        while k <= e and rep["reliable"][k]:
            after_exit[k] = True
            k += 1
        exit_delays.append((k - s) / fps)
    absent_other = absent & ~after_exit

    reacq, missed = [], 0
    first_correct = int(np.argmax(correct)) if correct.any() else None
    for s, e in runs_of(present):
        if s == 0:
            continue  # initial lock, reported as first_correct
        hit = np.where(correct[s:e + 1])[0]
        if len(hit):
            reacq.append(hit[0] / fps)
        else:
            missed += 1

    ious, err_px, err_mrad = [], [], []
    for i in np.where(correct)[0]:
        b, g = rep["box"][i], rep["gt"][i]
        if np.isnan(b[0]):
            continue
        cb, cg = clip_box_pc(b, frame_wh), clip_box_pc(g, frame_wh)
        ious.append(iou(cb, cg) if cb and cg else 0.0)
        dx = (b[0] + b[2] / 2) - (g[0] + g[2] / 2)
        dy = (b[1] + b[3] / 2) - (g[1] + g[3] / 2)
        err_px.append(math.hypot(dx, dy))
        err_mrad.append(1e3 * math.hypot(math.atan(dx / fx), math.atan(dy / fy)))

    def longest(mask: np.ndarray) -> float:
        return max((e - s + 1 for s, e in runs_of(mask)), default=0) / fps

    inner = present & ~border
    return {
        "frames": n,
        "fps": fps,
        "present": int(present.sum()),
        "border": int(border.sum()),
        "correct": int(correct.sum()),
        "correct_inner": int((correct & inner).sum()),
        "inner": int(inner.sum()),
        "wrong_target": int(wrong.sum()),
        "drone_absent_after_exit": int((absent & after_exit).sum()),
        "drone_absent_other": int(absent_other.sum()),
        "longest_wrong_or_other_absent_s": longest(wrong | absent_other),
        "longest_dangerous_s": longest(wrong | absent),
        "first_correct_s": None if first_correct is None else first_correct / fps,
        "reacquisition_s": reacq,
        "missed_stretches": missed,
        "exit_delay_s": exit_delays,
        "iou_clipped": ious,
        "err_px": err_px,
        "err_mrad": err_mrad,
    }


def combine(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum of counts, concatenation of lists, max of the longest segments."""
    out: dict[str, Any] = {}
    for k in items[0]:
        vals = [it[k] for it in items]
        if isinstance(vals[0], list):
            out[k] = [v for vs in vals for v in vs]
        elif k.startswith("longest"):
            out[k] = max(vals)
        elif k == "first_correct_s":
            ok = [v for v in vals if v is not None]
            out[k] = max(ok) if len(ok) == len(vals) else None  # worst case; None if one never locked
        elif k == "fps":
            out[k] = vals[0]
        else:
            out[k] = sum(vals)
    return out


def digest(m: dict[str, Any]) -> dict[str, Any]:
    """Rates and percentiles of a (combined) metrics dict; drops the per-frame lists."""
    def stats(v: list, q=(50, 90)) -> Optional[dict[str, float]]:
        if not v:
            return None
        a = np.asarray(v, float)
        return {**{f"p{p}": float(np.percentile(a, p)) for p in q}, "max": float(a.max()), "n": int(len(a))}

    d = {k: v for k, v in m.items() if not isinstance(v, list)}
    d["coverage"] = m["correct"] / m["present"] if m["present"] else None
    d["coverage_inner"] = m["correct_inner"] / m["inner"] if m["inner"] else None
    d["iou_clipped"] = None if not m["iou_clipped"] else {
        "median": float(np.median(m["iou_clipped"])), "above_0.5": float(np.mean(np.asarray(m["iou_clipped"]) > 0.5))}
    d["err_px"] = stats(m["err_px"])
    d["err_mrad"] = stats(m["err_mrad"])
    d["reacquisition_s"] = stats(m["reacquisition_s"])
    d["exit_delay_s"] = stats(m["exit_delay_s"])
    return d


def empty_scene_summary(dirs: Sequence[Path]) -> Optional[dict[str, Any]]:
    files = sorted(p for d in dirs for p in Path(d).glob("*.json"))
    if not files:
        return None
    per = {}
    for p in files:
        s = json.loads(p.read_text())
        if "reliable_frames" not in s:
            continue
        per[p.stem] = {"frames": s.get("frames"), "reliable_frames": s.get("reliable_frames"), "fps": s.get("fps")}
    frames = sum(v["frames"] or 0 for v in per.values())
    return {"recordings": len(per), "frames": frames, "reliable_frames": sum(v["reliable_frames"] or 0 for v in per.values()),
            "minutes": sum((v["frames"] or 0) / (v["fps"] or 62.6) for v in per.values()) / 60.0, "per_recording": per}


# ============================================================================ output


def _fmt(v: Optional[float], spec: str = ".2f", none: str = "-") -> str:
    return none.rjust(int(re.match(r"\d*", spec).group() or 0)) if v is None else format(v, spec)


def table_row(name: str, d: dict[str, Any]) -> str:
    re_ = d["reacquisition_s"]
    ex = d["exit_delay_s"]
    io = d["iou_clipped"]
    em = d["err_mrad"]
    return (f"{name[:34]:34s} {d['present']:6d} {d['border']:5d} {_fmt(d['coverage'], '6.1%')} {_fmt(d['coverage_inner'], '6.1%')} "
            f"{d['wrong_target']:5d} {d['drone_absent_other']:5d} {d['drone_absent_after_exit']:5d} {d['longest_wrong_or_other_absent_s']:5.2f} "
            f"{_fmt(io and io['median'], '5.3f')} {_fmt(em and em['p50'], '5.2f')} {_fmt(em and em['p90'], '5.2f')} "
            f"{_fmt(d['first_correct_s'], '5.2f')} {_fmt(re_ and re_['p50'], '5.2f')} {_fmt(re_ and re_['max'], '5.2f')} {(re_ or {}).get('n', 0):3d} "
            f"{d['missed_stretches']:3d} {_fmt(ex and ex['p50'], '5.2f')} {_fmt(ex and ex['max'], '5.2f')}")


HEADER = (f"{'recording / scenario':34s} {'pres':>6s} {'bord':>5s} {'cov':>6s} {'cov_in':>6s} {'wrong':>5s} {'abs_o':>5s} {'abs_x':>5s} {'dng_s':>5s} "
          f"{'IoU':>5s} {'mr50':>5s} {'mr90':>5s} {'lck_s':>5s} {'rq50':>5s} {'rqmx':>5s} {'nrq':>3s} {'mis':>3s} {'ex50':>5s} {'exmx':>5s}")
LEGEND = """pres: frames with the drone (labels); bord: of which at the frame border; cov: correct reliable / pres; cov_in: same without border frames
wrong: reliable on the wrong target; abs_o: reliable with the drone absent, not right after an exit; abs_x: right after an exit (loss delay)
dng_s: longest wrong / abs_o segment (s); IoU: median, boxes clipped to the frame; mr50/mr90: centre error (mrad) while correct
lck_s: first correct frame (s); rq50/rqmx/nrq: re-acquisition after re-entry (s) median / max / count; mis: re-entries never re-acquired
ex50/exmx: loss delay after an exit (s) median / max"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("replays", type=Path, nargs="+", help="replay CSVs or folders of them")
    ap.add_argument("--empty", type=Path, nargs="*", default=[], help="folders of empty-scene replay JSONs")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="for the mrad conversion")
    ap.add_argument("--edge-margin", type=float, default=8.0, help="border frames: label box this close to an edge (px)")
    ap.add_argument("--labels", type=Path, help="label folder (labels/ids) for replays made without --gt")
    ap.add_argument("--out", type=Path, help="JSON report")
    args = ap.parse_args()

    files = []
    for p in args.replays:
        files += sorted(q for q in p.glob("*.csv") if not q.stem.endswith("_detections")) if p.is_dir() else [p]
    fx, fy = load_intrinsics(args.calibration)
    per: dict[str, dict[str, Any]] = {}
    for p in files:
        try:
            rep = load_replay(p)
        except ValueError as exc:
            print(f"skipped {p.name}: {exc}")
            continue
        no_gt = not (~np.isnan(rep["gt"][:, 0])).any() and not any(c for c in rep["category"])
        if no_gt and args.labels is not None:
            label_path = args.labels / f"{p.stem}.csv"
            if not label_path.is_file():
                print(f"skipped {p.name}: no ground truth in the replay and no {label_path.name}")
                continue
            try:
                attach_labels(rep, label_path)
            except ValueError as exc:
                print(f"skipped {p.name}: {exc}")
                continue
            no_gt = False
        if no_gt:
            print(f"skipped {p.name}: no ground truth in the replay")
            continue
        frame_wh = rep["frame_wh"] or (2040, 1536)
        per[p.stem] = recording_metrics(rep, fx, fy, args.edge_margin, frame_wh)
    if not per:
        print("no replays with ground truth")
        return 1

    scen: dict[str, list] = {}
    for name, m in per.items():
        scen.setdefault(scenario_of(name), []).append(m)
    total = combine(list(per.values()))

    print(HEADER)
    for name, m in per.items():
        print(table_row(name, digest(m)))
    print("-" * len(HEADER))
    for s, ms in scen.items():
        print(table_row(f"{s} ({len(ms)})", digest(combine(ms))))
    print("-" * len(HEADER))
    print(table_row(f"TOTAL ({len(per)})", digest(total)))
    print(LEGEND)
    empty = empty_scene_summary(args.empty) if args.empty else None
    if empty:
        print(f"empty scene: {empty['recordings']} recordings, {empty['minutes']:.1f} min, reliable frames (false alarms): {empty['reliable_frames']}")

    if args.out is not None:
        report = {"report_version": REPORT_VERSION, "fx": fx, "fy": fy, "edge_margin_px": args.edge_margin,
                  "replays": [str(p) for p in files],
                  "per_recording": {k: digest(v) for k, v in per.items()},
                  "per_scenario": {k: digest(combine(v)) for k, v in scen.items()},
                  "total": digest(total), "empty_scene": empty}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
