"""Automatic pre-labels of IDS recordings, to be corrected with tools/review_labels.py (V4).

For every frame: where is the drone (box) or is it absent. Offline, no real-time pacing:

1. the detector runs on every frame (``--det-every``), on a digitally brightened copy
   when the frame is dark (``ids_labels.auto_gain``, a labelling aid only: the runtime
   does not brighten). Raw detections are cached in ``runs/v4_labels/<name>_dets.csv``;
2. anchors: strong detections (>= 0.5) confirmed by another strong detection within
   +/- 3 frames;
3. between anchors SiamFC tracks forward from the left anchor and backward from the right
   one (raw frames, runtime confidence rule). Agreeing directions are blended; a weak
   detection that agrees with the tracker replaces its box; disagreements, one-direction
   tracking further than ~1 s from its anchor and absences between two anchors are flagged
   for review;
4. tracker and weak-detection boxes within 8 px of a frame edge, and tracker boxes with
   no detection nearby within +/- 16 frames, are flagged too (``ids_labels.flag_for_review``):
   when the drone leaves the frame SiamFC can stay on the border or on a wall stain.

Output: ``labels/ids/<recording>.csv`` (+ ``.json`` provenance), format in ``ids_labels.py``.
A label file that already contains review work is never overwritten unless ``--force``,
and then a timestamped backup is kept.

Usage (from cuasp_siamfc/)::

    .venv/Scripts/python tools/prelabel_ids.py ../recordings/20261117_093557_B3_rapido_in_volo_r3
    .venv/Scripts/python tools/prelabel_ids.py ../recordings/2026111*_B* ../recordings/2026111*_C* --exclude 20261116_145249_B1_stazionario_a_terra
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Iterator, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ids_labels as L  # noqa: E402

TOOL_VERSION = "2026-11-17-prelabel-ids-v2"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "labels" / "ids"
DEFAULT_CACHE = REPO / "runs" / "v4_labels"
DEFAULT_SIAMFC = REPO / "pretrained" / "networks" / "baseline-conv5_gray_e100.pth"
DEFAULT_DETECTOR = REPO / "runs" / "detector" / "yolo11s_uav_gray_1280_e40" / "weights" / "best.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def recording_dirs(paths: Sequence[Path]) -> list[Path]:
    """Recording folders (containing frames/) from recording or frames paths, sorted, unique."""
    out = []
    for p in paths:
        p = Path(p)
        rec = p.parent if p.name == "frames" else p
        if not (rec / "frames").is_dir():
            raise FileNotFoundError(f"{p}: no frames/ folder")
        out.append(rec.resolve())
    return sorted(set(out))


def list_frames(rec: Path) -> list[Path]:
    return sorted(p for p in (rec / "frames").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_gray(path: Path) -> np.ndarray:
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def detect_all(frames: Sequence[Path], detector, enhance: bool, target_mean: float, max_gain: float, det_every: int) -> tuple[list, list]:
    dets: list[list[tuple[L.Box, float]]] = []
    gains: list[float] = []
    t0 = time.perf_counter()
    for f, path in enumerate(frames):
        if f % det_every:
            dets.append([])
            gains.append(gains[-1] if gains else 1.0)
            continue
        gray = read_gray(path)
        g = L.auto_gain(gray, target_mean, max_gain) if enhance else 1.0
        found = detector.detect(L.apply_gain(gray, g))
        dets.append([(L.box_from_pixel_centre(d.bbox_xywh), float(d.score)) for d in found])
        gains.append(g)
        if f and f % 500 == 0:
            rate = f / (time.perf_counter() - t0)
            print(f"  detection {f}/{len(frames)} ({rate:.0f} frames/s)", flush=True)
    return dets, gains


def tracker_outputs(tracker, frames: Sequence[Path], init_frame: int, init_box: L.Box, order: Sequence[int]) -> Iterator[tuple[L.Box, float, float]]:
    """Initialise on ``init_frame`` with ``init_box`` (edge) and yield (edge box, peak, apce) along ``order``."""
    x, y, w, h = L.box_to_pixel_centre(init_box)
    tracker.init(read_gray(frames[init_frame]), (x + w / 2.0, y + h / 2.0), (w, h))
    for f in order:
        res = tracker.update(read_gray(frames[f]))
        (cx, cy), (rw, rh) = res.center_xy, res.size_wh
        yield L.box_from_pixel_centre((cx - rw / 2.0, cy - rh / 2.0, rw, rh)), float(res.peak), float(res.apce)


def prelabel(frames: Sequence[Path], dets: list, tracker, p: L.PrelabelParams, frame_wh: tuple[int, int]) -> L.LabelTrack:
    n = len(frames)
    track = L.LabelTrack(n)
    anchors = L.select_anchors(dets, p)
    for f, (box, score) in anchors.items():
        track.set_box(f, box, "det", score, 0, undoable=False)
    for first, last, left, right in L.gap_ranges(anchors, n):
        size = last - first + 1
        empty = L.TrackStep(None, float("nan"))
        fwd = [empty] * size
        bwd = [empty] * size
        if left is not None:
            order = list(range(first, min(last, left + p.max_track_frames) + 1))
            steps = L.track_run(anchors[left][0], tracker_outputs(tracker, frames, left, anchors[left][0], order), frame_wh, p)
            for f, s in zip(order, steps):
                fwd[f - first] = s
        if right is not None:
            order = list(range(last, max(first, right - p.max_track_frames) - 1, -1))
            steps = L.track_run(anchors[right][0], tracker_outputs(tracker, frames, right, anchors[right][0], order), frame_wh, p)
            for f, s in zip(order, steps):
                bwd[f - first] = s
        for f, (box, source, score, flag) in zip(range(first, last + 1), L.fuse_gap(first, last, left, right, fwd, bwd, dets, p)):
            if box is None:
                track.set_absent(f, "absent", flag, undoable=False)
            else:
                track.set_box(f, box, source, score, flag, undoable=False)
    L.flag_for_review(track, dets, frame_wh, p)
    return track


def recording_meta(rec: Path) -> dict:
    meta_path = rec / "meta.json"
    if not meta_path.is_file():
        return {}
    m = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    keep = ("name", "block", "notes", "distance_m", "drone", "lighting", "started_at", "tool_version")
    out = {k: m.get(k) for k in keep}
    rb = (m.get("camera") or {}).get("readback") or {}
    out["camera"] = {k: rb.get(k) for k in ("ExposureTime", "Gain", "AcquisitionFrameRate", "Width", "Height", "OffsetX", "OffsetY")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recordings", type=Path, nargs="+", help="recording folders (or their frames/ folders)")
    ap.add_argument("--exclude", nargs="*", default=[], help="recording folder names to skip")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--siamfc-weights", type=Path, default=DEFAULT_SIAMFC)
    ap.add_argument("--detector-weights", type=Path, default=DEFAULT_DETECTOR)
    ap.add_argument("--det-every", type=int, default=1, help="run the detector every N frames")
    ap.add_argument("--no-enhance", action="store_true", help="feed the detector the raw frames")
    ap.add_argument("--target-mean", type=float, default=80.0, help="mean gray level of the brightened detector input")
    ap.add_argument("--max-gain", type=float, default=3.0, help="max digital gain of the detector input")
    ap.add_argument("--redetect", action="store_true", help="ignore cached detections")
    ap.add_argument("--force", action="store_true", help="overwrite label files that contain review work (a backup is kept)")
    args = ap.parse_args()
    if args.det_every < 1:
        ap.error("--det-every must be >= 1")

    recs = [r for r in recording_dirs(args.recordings) if r.name not in set(args.exclude)]
    p = L.PrelabelParams()
    p.validate()
    todo = []
    for rec in recs:
        out = args.out_dir / f"{rec.name}.csv"
        if out.is_file():
            try:
                existing = L.load_labels(out)
            except ValueError as exc:
                print(f"{rec.name}: SKIPPED, unreadable label file ({exc})")
                continue
            if L.has_user_work(existing) and not args.force:
                print(f"{rec.name}: SKIPPED, {out.name} contains review work (use --force to overwrite, a backup is kept)")
                continue
        todo.append(rec)
    if not todo:
        print("nothing to do")
        return 0

    import torch  # noqa: PLC0415

    from suncubes.siamfc.detector import DetectorConfig, UltralyticsDroneDetector, sha256_file  # noqa: PLC0415
    from suncubes.siamfc.net import SiamFCNet  # noqa: PLC0415
    from suncubes.siamfc.tracker import SiamFCTracker, default_device  # noqa: PLC0415

    device = default_device()
    detector = None
    tracker = SiamFCTracker(SiamFCNet.from_file(args.siamfc_weights), device=device)
    weights = {
        "siamfc": {"path": str(args.siamfc_weights), "sha256": sha256_file(args.siamfc_weights)},
        "detector": {"path": str(args.detector_weights), "sha256": sha256_file(args.detector_weights)},
    }
    print(f"device {device}; {len(todo)} recording(s)")

    for rec in todo:
        t0 = time.perf_counter()
        frames = list_frames(rec)
        if not frames:
            print(f"{rec.name}: SKIPPED, no frames")
            continue
        first = read_gray(frames[0])
        frame_wh = (first.shape[1], first.shape[0])
        cache = args.cache_dir / f"{rec.name}_dets.csv"
        if cache.is_file() and not args.redetect:
            dets, gains = L.load_detections(cache, len(frames))
            print(f"{rec.name}: {len(frames)} frames, detections from cache")
        else:
            if detector is None:
                detector = UltralyticsDroneDetector(DetectorConfig(weights_path=str(args.detector_weights), conf_min=p.weak_score))
                detector.warmup((frame_wh[1], frame_wh[0]))
            print(f"{rec.name}: {len(frames)} frames, detecting ...", flush=True)
            dets, gains = detect_all(frames, detector, not args.no_enhance, args.target_mean, args.max_gain, args.det_every)
            L.save_detections(cache, dets, gains)
        with torch.no_grad():
            track = prelabel(frames, dets, tracker, p, frame_wh)
        out = args.out_dir / f"{rec.name}.csv"
        if out.is_file() and not args.force:
            try:  # checked again here: the file may have been reviewed while this batch was running
                reviewed_meanwhile = L.has_user_work(L.load_labels(out))
            except ValueError:
                reviewed_meanwhile = True
            if reviewed_meanwhile:
                print(f"  NOT WRITTEN: {out.name} now contains review work")
                continue
        if out.is_file():
            backup_dir = args.cache_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / f"{out.stem}.{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
            shutil.copy2(out, backup)
            print(f"  previous label file kept as {backup}")
        meta = {
            "tool_version": TOOL_VERSION,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "recording": str(rec),
            "frames": len(frames),
            "frame_size": list(frame_wh),
            "coordinates": "edge (x, y top-left; image spans [0, W] x [0, H])",
            "detections_cache": str(cache),
            "detector_input": {"enhance": not args.no_enhance, "target_mean": args.target_mean, "max_gain": args.max_gain,
                               "gain_median": float(np.median(gains)), "det_every": args.det_every},
            "params": asdict(p),
            "weights": weights,
            "recording_meta": recording_meta(rec),
        }
        L.save_labels(track, out, meta)
        c = track.counts()
        print(f"  -> {out} in {time.perf_counter() - t0:.0f} s: present {c['present']}, absent {c['absent']}, "
              f"flagged {c['flagged']} ({100 * c['flagged'] / c['frames']:.1f}%), sources {c['sources']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
