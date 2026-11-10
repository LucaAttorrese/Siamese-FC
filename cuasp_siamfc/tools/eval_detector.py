"""Break down the drone detector's test results by source and by drone size (Phase 2, step 2).

Usage (from cuasp_siamfc/)::

    .venv/Scripts/python tools/eval_detector.py --weights ../runs/detector/yolo11s_uav_gray_1280_e40/weights/best.pt --data ../datasets/yolo_uav_gray/data.yaml

What it reports:
- per source (image name prefixes ``dut_`` and ``auav_``): Ultralytics' own validation
  (mAP50, mAP50-95, P, R) on each subset of the test split, so the numbers are
  directly comparable with the overall test result of ``train_detector.py``;
- per drone size: recall at IoU >= 0.5 for ground-truth boxes grouped by their side
  sqrt(w*h) in pixels of the original image, at two confidence thresholds: 0.25
  (Ultralytics' predict default) and the threshold that maximises F1 on the whole split.
  Ultralytics has no size breakdown, so this uses its own greedy matching (highest
  confidence first, each ground-truth box matched once, IoU >= 0.5). The overall AP50 of
  this matching is printed as a consistency check against Ultralytics;
- false alarms: images without a drone (empty label file) that still get a detection.

Outputs ``<project>/<run>_test_breakdown/breakdown.json`` plus the printed tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

SOURCES = ("dut", "auav")
SIZE_EDGES_PX = (0, 16, 32, 64, 128, float("inf"))
MATCH_IOU = 0.5


def size_bucket(side_px: float, edges=SIZE_EDGES_PX) -> str:
    """Label of the size bucket containing ``side_px`` (lower edge inclusive)."""
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= side_px < hi:
            return f"{int(lo)}-{int(hi)}" if hi != float("inf") else f">={int(lo)}"
    raise ValueError(side_px)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes ``a`` (N, 4) and ``b`` (M, 4), both xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_image(pred_xyxy: np.ndarray, pred_conf: np.ndarray, gt_xyxy: np.ndarray, iou_thr: float = MATCH_IOU):
    """Greedy matching, highest confidence first.

    Returns (pred_tp, gt_best_conf): ``pred_tp[i]`` is True when prediction i matched a
    ground-truth box; ``gt_best_conf[j]`` is the confidence of the prediction that matched
    ground-truth box j, or -1 when none did. A ground-truth box is detected at threshold t
    exactly when ``gt_best_conf[j] >= t``.
    """
    order = np.argsort(-pred_conf, kind="stable")
    iou = box_iou(pred_xyxy, gt_xyxy)
    pred_tp = np.zeros(len(pred_conf), dtype=bool)
    gt_best_conf = np.full(len(gt_xyxy), -1.0)
    taken = np.zeros(len(gt_xyxy), dtype=bool)
    for i in order:
        if len(gt_xyxy) == 0:
            break
        cand = np.where(~taken & (iou[i] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(iou[i, cand])]
            taken[j] = True
            pred_tp[i] = True
            gt_best_conf[j] = pred_conf[i]
    return pred_tp, gt_best_conf


def pr_curve(conf: np.ndarray, tp: np.ndarray, n_gt: int):
    """Precision, recall and thresholds of all predictions sorted by confidence (descending)."""
    order = np.argsort(-conf, kind="stable")
    tp_sorted = tp[order].astype(float)
    ctp = np.cumsum(tp_sorted)
    cfp = np.cumsum(1.0 - tp_sorted)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    return precision, recall, conf[order]


def best_f1_threshold(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> tuple[float, float, float, float]:
    """(threshold, F1, precision, recall) at the prediction confidence that maximises F1."""
    p, r, thr = pr_curve(conf, tp, n_gt)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    k = int(np.argmax(f1))
    return float(thr[k]), float(f1[k]), float(p[k]), float(r[k])


def _yolo_to_xyxy(lines: list[str], w: int, h: int) -> np.ndarray:
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def _source_of(name: str) -> str:
    prefix = name.split("_", 1)[0]
    return prefix if prefix in SOURCES else "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="data.yaml with a 'test' split")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8, help="batch of Ultralytics' val (per-source part)")
    ap.add_argument("--predict-batch", type=int, default=4, help="batch of the prediction pass (size breakdown)")
    ap.add_argument("--out", type=Path, help="output folder (default: <runs/detector>/<run>_test_breakdown)")
    args = ap.parse_args()

    import yaml  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415
    from ultralytics.utils.metrics import compute_ap  # noqa: PLC0415

    cfg = yaml.safe_load(args.data.read_text())
    root = Path(cfg.get("path") or args.data.parent)
    img_dir = root / cfg["test"]  # YOLO layout: images/<split> with labels in labels/<split>
    lbl_dir = root / "labels" / Path(cfg["test"]).name
    if not lbl_dir.is_dir():
        print(f"label folder not found: {lbl_dir}", file=sys.stderr)
        return 2
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    run_name = args.weights.resolve().parents[1].name
    out_dir = (args.out or args.weights.resolve().parents[2] / f"{run_name}_test_breakdown").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    # 1. Ultralytics validation per source, on list files that select each subset.
    per_source_official = {}
    with tempfile.TemporaryDirectory() as tmp:
        for src in SOURCES:
            subset = [p for p in images if _source_of(p.name) == src]
            if not subset:
                continue
            lst = Path(tmp) / f"test_{src}.txt"
            lst.write_text("\n".join(p.as_posix() for p in subset) + "\n")
            sub_yaml = Path(tmp) / f"data_{src}.yaml"
            sub_yaml.write_text(yaml.safe_dump({"path": root.as_posix(), "train": cfg["train"], "val": lst.as_posix(), "test": lst.as_posix(), "names": cfg["names"]}))
            m = model.val(data=str(sub_yaml), split="test", imgsz=args.imgsz, batch=args.batch, plots=False, verbose=False, project=str(out_dir), name=f"val_{src}", exist_ok=True)
            per_source_official[src] = {"images": len(subset), "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "P": float(m.box.mp), "R": float(m.box.mr)}

    # 2. Own matching on all test images for the size and false-alarm breakdown.
    all_conf, all_tp, all_src = [], [], []
    gt_rows = []  # (source, side_px, best_conf)
    neg_max_conf = {src: [] for src in SOURCES}
    for start in range(0, len(images), 64):
        chunk = images[start:start + 64]
        # stream=True: results are yielded per batch instead of being accumulated; fp16 (quantize=16) like
        # Ultralytics' val (``half`` is deprecated in Ultralytics 8.4.2xx).
        # Without these, an 8 GB GPU filled up and Windows fell back to system memory (~30x slower).
        results = model.predict([str(p) for p in chunk], imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300, batch=args.predict_batch, quantize=16, stream=True, verbose=False)
        for path, res in zip(chunk, results):
            w, h = Image.open(path).size
            lbl = lbl_dir / f"{path.stem}.txt"
            gt = _yolo_to_xyxy(lbl.read_text().splitlines() if lbl.exists() else [], w, h)
            boxes = res.boxes
            pred = boxes.xyxy.cpu().numpy().astype(float) if boxes is not None else np.zeros((0, 4))
            conf = boxes.conf.cpu().numpy().astype(float) if boxes is not None else np.zeros(0)
            src = _source_of(path.name)
            tp, gt_best = match_image(pred, conf, gt)
            all_conf.append(conf)
            all_tp.append(tp)
            all_src.extend([src] * len(conf))
            for box, best in zip(gt, gt_best):
                side = float(np.sqrt((box[2] - box[0]) * (box[3] - box[1])))
                gt_rows.append((src, side, float(best)))
            if len(gt) == 0 and src in neg_max_conf:
                neg_max_conf[src].append(float(conf.max()) if len(conf) else 0.0)
        print(f"  predicted {min(start + 64, len(images))}/{len(images)}", flush=True)

    conf = np.concatenate(all_conf) if all_conf else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)
    pred_src = np.asarray(all_src)
    n_gt = len(gt_rows)
    p_curve, r_curve, _ = pr_curve(conf, tp, n_gt)
    ap50_own = float(compute_ap(r_curve, p_curve)[0])
    thr_f1, f1, p_f1, r_f1 = best_f1_threshold(conf, tp, n_gt)
    thresholds = {"0.25": 0.25, f"F1-best {thr_f1:.3f}": thr_f1}

    def recall_at(rows, thr):
        return (sum(1 for r in rows if r[2] >= thr) / len(rows)) if rows else float("nan")

    def precision_at(mask, thr):
        sel = (conf >= thr) & mask
        return float(tp[sel].mean()) if sel.any() else float("nan")

    buckets = [size_bucket(e) for e in SIZE_EDGES_PX[:-1]]
    by_size = {}
    for src in ("all", *SOURCES):
        rows_src = [r for r in gt_rows if src == "all" or r[0] == src]
        mask = np.ones(len(conf), dtype=bool) if src == "all" else pred_src == src
        entry = {"gt_boxes": len(rows_src), "precision": {k: precision_at(mask, t) for k, t in thresholds.items()}, "recall": {k: recall_at(rows_src, t) for k, t in thresholds.items()}, "buckets": {}}
        for b in buckets:
            rows_b = [r for r in rows_src if size_bucket(r[1]) == b]
            entry["buckets"][b] = {"gt_boxes": len(rows_b), **{f"recall@{k}": recall_at(rows_b, t) for k, t in thresholds.items()}}
        by_size[src] = entry

    false_alarms = {
        src: {"negative_images": len(v), **{f"with_detection@{k}": int(sum(c >= t for c in v)) for k, t in thresholds.items()}}
        for src, v in neg_max_conf.items()
    }
    report = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "imgsz": args.imgsz,
        "test_images": len(images),
        "per_source_ultralytics": per_source_official,
        "own_matching": {"iou": MATCH_IOU, "AP50_all": ap50_own, "F1_best": {"threshold": thr_f1, "F1": f1, "P": p_f1, "R": r_f1}},
        "by_size": by_size,
        "false_alarms_on_negative_images": false_alarms,
        "size_edges_px": [e if e != float("inf") else None for e in SIZE_EDGES_PX],
    }
    (out_dir / "breakdown.json").write_text(json.dumps(report, indent=2))

    print("\n== Per source (Ultralytics val on each subset of the test split) ==")
    print(f"{'source':8} {'images':>7} {'mAP50':>7} {'mAP50-95':>9} {'P':>6} {'R':>6}")
    for src, m in per_source_official.items():
        print(f"{src:8} {m['images']:7d} {m['mAP50']:7.3f} {m['mAP50-95']:9.3f} {m['P']:6.3f} {m['R']:6.3f}")
    print(f"\n== Own matching (IoU >= {MATCH_IOU}) == AP50 all = {ap50_own:.3f} (consistency check vs Ultralytics)")
    print(f"F1-best threshold {thr_f1:.3f}: F1 {f1:.3f}, P {p_f1:.3f}, R {r_f1:.3f}")
    for src, e in by_size.items():
        print(f"\n-- recall by drone size, source={src} ({e['gt_boxes']} boxes; side = sqrt(w*h) in original px) --")
        keys = list(thresholds)
        print(f"{'side px':>10} {'boxes':>6} " + " ".join(f"{('R@' + k):>18}" for k in keys))
        for b, v in e["buckets"].items():
            print(f"{b:>10} {v['gt_boxes']:6d} " + " ".join(f"{v['recall@' + k]:18.3f}" for k in keys))
        print(f"{'all':>10} {e['gt_boxes']:6d} " + " ".join(f"{e['recall'][k]:18.3f}" for k in keys) + "   precision: " + ", ".join(f"{k} {e['precision'][k]:.3f}" for k in keys))
    print("\n== False alarms on images without a drone ==")
    for src, v in false_alarms.items():
        print(f"{src}: {v}")
    print(f"\nwritten {out_dir / 'breakdown.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
