"""Replay a recorded sequence through the full ``ids_siamfc`` camera source (validation V4).

Runs ``IdsSiamFCCorrectionSourceAdapter`` exactly as at runtime: SiamFC inside ``read_candidate``,
the drone detector on its own asynchronous thread, the same gating. Only the IDS camera is
replaced by a producer that publishes the recorded frames, paced in real time at ``--fps``,
so the detector delays relative to the frames are realistic.

Usage (from cuasp_siamfc/)::

    # DUT-Anti-UAV tracking video with ground truth
    .venv/Scripts/python tools/replay_siamfc_source.py ../datasets/DUT-Anti-UAV/tracking/Anti-UAV-Tracking-V0/video01 --gt ../datasets/DUT-Anti-UAV/tracking/Anti-UAV-Tracking-V0GT/video01_gt.txt
    # IDS recording with the real calibration, per-frame CSV
    .venv/Scripts/python tools/replay_siamfc_source.py D:/ids_rec/run3 --calibration calibration/calibration.yaml --out ../runs/replay/run3.csv
    # empty scene (no drone): every reliable frame is a false alarm
    .venv/Scripts/python tools/replay_siamfc_source.py D:/ids_rec/empty_hall --empty-scene
    # override settings of the source
    .venv/Scripts/python tools/replay_siamfc_source.py <seq> --siamfc redetect_hz=2 verify_miss_limit=3 --detector conf_min=0.2
    # latency V5: DUT frames resized to the IDS camera size, paced at the camera rate, preloaded
    .venv/Scripts/python tools/replay_siamfc_source.py <dut video> --gt <gt> --resize 2040 1536 --fps 62.6 --preload --max-frames 1000

Input: a folder of images (sorted by name; jpg/png/bmp/tif) or a video file. Frames are
converted to Mono8. Ground truth (``--gt``): one line per frame, ``x y w h`` (spaces or
commas), top-left corner in edge coordinates as in DUT-Anti-UAV and YOLO; a line with
w <= 0, h <= 0 or x < 0 marks the target as absent. The IDS label files of
``tools/prelabel_ids.py`` / ``tools/review_labels.py`` (``labels/ids/<recording>.csv``)
are accepted as they are.

Metrics. Every reliable frame (``reason == "ok"``) with ground truth falls in one of:
``correct`` (target present and tracked centre within max(gt w, gt h) of the true centre),
``wrong_target`` (target present, centre further away) or ``drone_absent`` (no target in the
frame). The last two are the dangerous cases: the system would point elsewhere. ``coverage``
is correct / frames with the target present. Also reported: first reliable / correct
frame, longest wrong segment, centre error and IoU while tracking, ``read_candidate`` time.

Without ``--calibration`` the intrinsics are synthetic (``--fx``, principal point at the
image centre, identity extrinsics): tracking metrics are valid, the angles are not.
Notes: the detector runs asynchronously, so two replays of the same sequence can differ
slightly. ``--fast`` drops the real-time pacing (the detector then lags more frames behind).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Iterator, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SIAMFC = REPO / "pretrained" / "networks" / "baseline-conv5_gray_e100.pth"
DEFAULT_DETECTOR = REPO / "runs" / "detector" / "yolo11s_uav_gray_1280_e40" / "weights" / "best.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================================ input


def read_gt(path: Path, n_frames: Optional[int] = None) -> np.ndarray:
    """(N, 4) array of x, y, w, h (edge coordinates); rows of absent targets are NaN.

    Also reads the label files of ``tools/prelabel_ids.py`` / ``tools/review_labels.py``
    (CSV with a ``frame,present,x,y,w,h,...`` header, edge coordinates).
    """
    text = Path(path).read_text()
    if text.startswith("frame,present,"):
        rows = []
        for i, r in enumerate(csv.DictReader(text.splitlines())):
            if int(r["frame"]) != i:
                raise ValueError(f"{path}: row {i} has frame {r['frame']}")
            present = int(r["present"])
            rows.append([float(r[k]) for k in ("x", "y", "w", "h")] if present else [np.nan] * 4)
        gt = np.asarray(rows, float).reshape(-1, 4)
        return gt if n_frames is None else gt[:n_frames]
    rows = []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        x, y, w, h = (float(v) for v in parts[:4])
        rows.append([np.nan] * 4 if (w <= 0 or h <= 0 or x < 0 or y < 0) else [x, y, w, h])
    gt = np.asarray(rows, float).reshape(-1, 4)
    return gt if n_frames is None else gt[:n_frames]


def scale_gt(gt: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Ground truth (edge coordinates) of frames resized by ``sx``, ``sy``: edges scale exactly."""
    out = np.array(gt, float, copy=True)
    out[:, [0, 2]] *= float(sx)
    out[:, [1, 3]] *= float(sy)
    return out


def resize_frames(frames: Iterator[np.ndarray], size_wh: tuple[int, int]) -> Iterator[np.ndarray]:
    import cv2  # noqa: PLC0415

    for gray in frames:
        yield gray if gray.shape[1::-1] == tuple(size_wh) else cv2.resize(gray, tuple(size_wh), interpolation=cv2.INTER_LINEAR)


def gt_to_pixel_centre(box_edges: Sequence[float]) -> tuple[float, float, float, float]:
    """Edge-coordinate box -> pixel-centre convention used by the tracker (-0.5 on x, y)."""
    x, y, w, h = (float(v) for v in box_edges)
    return (x - 0.5, y - 0.5, w, h)


def list_frames(source: Path) -> list[Path]:
    return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def iter_frames(source: Path, max_frames: Optional[int]) -> Iterator[np.ndarray]:
    import cv2  # noqa: PLC0415

    count = 0
    if source.is_dir():
        for p in list_frames(source):
            if max_frames is not None and count >= max_frames:
                return
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"cannot read {p}")
            count += 1
            yield gray
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {source}")
    try:
        while max_frames is None or count < max_frames:
            ok, frame = cap.read()
            if not ok:
                return
            count += 1
            yield frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


class _Prefetcher:
    """Decodes frames on a background thread so decoding does not disturb the real-time pacing."""

    _END = object()

    def __init__(self, frames: Iterator[np.ndarray], depth: int = 32) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=depth)
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, args=(frames,), name="replay-reader", daemon=True)
        self._thread.start()

    def _run(self, frames: Iterator[np.ndarray]) -> None:
        try:
            for f in frames:
                self._q.put(f)
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
        finally:
            self._q.put(self._END)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            item = self._q.get()
            if item is self._END:
                if self._error is not None:
                    raise self._error
                return
            yield item


class ReplayProducer:
    """Stands in for ``_IdsLatestFrameProducer``: latest-only slot fed by the replay loop."""

    def __init__(self) -> None:
        self.packet = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self):
        return self.packet

    def publish(self, gray: np.ndarray) -> None:
        from suncubes.camera import _FramePacket  # noqa: PLC0415

        seq = 1 if self.packet is None else self.packet.sequence + 1
        self.packet = _FramePacket(gray=gray, ts_capture_ns=time.monotonic_ns(), sequence=seq)

    def stats(self) -> dict[str, int]:
        return {"sequence": 0 if self.packet is None else int(self.packet.sequence)}


# ============================================================================ metrics


def classify(reliable: bool, bbox: Optional[Sequence[float]], gt_pc: Optional[Sequence[float]], has_gt: bool) -> Optional[str]:
    """Category of a frame: correct / wrong_target / drone_absent for reliable frames, else None."""
    if not reliable or not has_gt:
        return None
    if gt_pc is None:
        return "drone_absent"
    if bbox is None:
        return "wrong_target"
    gx, gy, gw, gh = gt_pc
    bx, by, bw, bh = bbox
    err = math.hypot(bx + bw / 2 - (gx + gw / 2), by + bh / 2 - (gy + gh / 2))
    return "correct" if err <= max(gw, gh) else "wrong_target"


def longest_run(flags: Sequence[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def summarize(records: list[dict[str, Any]], fps: float, has_gt: bool) -> dict[str, Any]:
    n = len(records)
    states = np.array([r["state"] for r in records])
    reliable = np.array([r["reliable"] for r in records], bool)
    cats = [r["category"] for r in records]
    rc = np.array([r["read_ms"] for r in records], float)
    out: dict[str, Any] = {"frames": n, "fps": fps}
    tracking_idx = np.where(states == "tracking")[0]
    first_track = int(tracking_idx[0]) if len(tracking_idx) else None
    rel_idx = np.where(reliable)[0]
    out["first_reliable_frame"] = int(rel_idx[0]) if len(rel_idx) else None
    after = states[first_track:] if first_track is not None else states[:0]
    out["after_first_track"] = {s: float(np.mean(after == s)) if len(after) else 0.0 for s in ("tracking", "lost", "searching")}
    out["reliable_frames"] = int(reliable.sum())
    if has_gt:
        present = np.array([r["gt"] is not None for r in records], bool)
        correct = np.array([c == "correct" for c in cats])
        wrong = np.array([c == "wrong_target" for c in cats])
        absent = np.array([c == "drone_absent" for c in cats])
        corr_idx = np.where(correct)[0]
        errs = np.array([r["err_px"] for r in records if r["err_px"] is not None], float)
        ious = np.array([r["iou"] for r in records if r["iou"] is not None], float)
        out.update({
            "frames_target_present": int(present.sum()),
            "correct": int(correct.sum()),
            "wrong_target": int(wrong.sum()),
            "drone_absent": int(absent.sum()),
            "coverage": float(correct.sum() / max(present.sum(), 1)),
            "first_correct_frame": int(corr_idx[0]) if len(corr_idx) else None,
            "longest_dangerous_segment_s": longest_run(list(wrong | absent)) / fps,
            "centre_error_px": None if not len(errs) else {
                "median": float(np.median(errs)), "p90": float(np.percentile(errs, 90)), "max": float(errs.max())},
            "iou": None if not len(ious) else {"median": float(np.median(ious)), "above_0.5": float(np.mean(ious > 0.5))},
        })
    out["read_candidate_ms"] = None if not len(rc) else {
        "median": float(np.median(rc)), "p99": float(np.percentile(rc, 99)), "max": float(rc.max())}
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    out["reasons"] = dict(sorted(reasons.items()))
    return out


SCORE_LEVELS = (0.15, 0.25, 0.5)  # detector score levels counted in the summary (0.5 = CV_SIAMFC_ACQUIRE_MIN_SCORE)


def summarize_detections(dets: list[dict[str, Any]], n_frames: int, fps: float) -> dict[str, Any]:
    """Raw detector boxes (one row per box of each consumed detector job): how many, how strong.
    In an empty scene every box is a false positive."""
    scores = np.array([d["score"] for d in dets], float)
    out: dict[str, Any] = {"boxes": int(len(dets)), "jobs_with_boxes": int(len({d["job"] for d in dets})),
                           "max_score": float(scores.max()) if len(scores) else None}
    for level in SCORE_LEVELS:
        sel = scores >= level
        out[f"boxes_score_ge_{level:g}"] = int(sel.sum())
        out[f"boxes_score_ge_{level:g}_per_min"] = float(sel.sum() / max(n_frames / fps / 60.0, 1e-9))
    return out


def _parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    out = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"override must be key=value: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


# ============================================================================ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="folder of images or video file")
    ap.add_argument("--gt", type=Path, help="ground truth, one 'x y w h' line per frame")
    ap.add_argument("--empty-scene", action="store_true", help="the sequence contains no drone: every reliable frame is a false alarm")
    ap.add_argument("--calibration", type=Path, help="camera calibration YAML (else synthetic intrinsics)")
    ap.add_argument("--fx", type=float, default=2000.0, help="focal length of the synthetic intrinsics (px)")
    ap.add_argument("--siamfc-weights", type=Path, default=DEFAULT_SIAMFC)
    ap.add_argument("--detector-weights", type=Path, default=DEFAULT_DETECTOR)
    ap.add_argument("--target-range", type=float, default=21.0, help="fixed camera depth Z (m)")
    ap.add_argument("--fps", type=float, default=50.0, help="replay rate (real time)")
    ap.add_argument("--fast", action="store_true", help="no real-time pacing")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                    help="resize every frame (and the ground truth) to W x H, e.g. 2040 1536 = IDS camera (latency V5)")
    ap.add_argument("--preload", action="store_true",
                    help="decode (and resize) all frames into memory before the replay, so decoding does not compete "
                         "with the measured pipeline (about 3.1 MB per 2040x1536 frame; cap with --max-frames)")
    ap.add_argument("--siamfc", nargs="*", default=[], metavar="KEY=VALUE", help="overrides of the siamfc settings")
    ap.add_argument("--detector", nargs="*", default=[], metavar="KEY=VALUE", help="overrides of the detector settings")
    ap.add_argument("--out", type=Path, help="per-frame CSV (a .json summary is written next to it)")
    args = ap.parse_args()
    if args.gt is not None and args.empty_scene:
        ap.error("--gt and --empty-scene are mutually exclusive")

    import cv2  # noqa: PLC0415

    from suncubes import camera as cam  # noqa: PLC0415
    from suncubes.siamfc_source import IdsSiamFCCorrectionSourceAdapter, bbox_iou  # noqa: PLC0415

    if args.source.is_dir():
        n_avail = len(list_frames(args.source))
        if n_avail == 0:
            print(f"no images in {args.source}", file=sys.stderr)
            return 2
    else:
        n_avail = None
    first = next(iter_frames(args.source, 1))
    h_src, w_src = first.shape[:2]
    w0, h0 = (int(args.resize[0]), int(args.resize[1])) if args.resize else (w_src, h_src)
    n_frames = n_avail if args.max_frames is None else min(args.max_frames, n_avail or args.max_frames)
    gt = read_gt(args.gt, n_frames) if args.gt is not None else None
    if gt is not None and (w0, h0) != (w_src, h_src):
        gt = scale_gt(gt, w0 / w_src, h0 / h_src)
    has_gt = gt is not None or args.empty_scene

    frames: Any = iter_frames(args.source, n_frames)
    if (w0, h0) != (w_src, h_src):
        frames = resize_frames(frames, (w0, h0))
    if args.preload:
        t0 = time.perf_counter()
        frames = list(frames)
        print(f"preloaded {len(frames)} frames {w0}x{h0} ({sum(f.nbytes for f in frames) / 1e9:.1f} GB) in {time.perf_counter() - t0:.1f} s")
    else:
        frames = _Prefetcher(frames)

    cfg: dict[str, Any] = {
        "siamfc": {"weights_path": str(args.siamfc_weights), "target_range_m": float(args.target_range), **_parse_overrides(args.siamfc)},
        "detector": {"weights_path": str(args.detector_weights), **_parse_overrides(args.detector)},
    }
    calibration = None
    if args.calibration is not None:
        cfg["calibration"] = {"path": str(args.calibration)}
    else:
        calibration = {
            "camera_matrix": np.array([[args.fx, 0.0, w0 / 2.0], [0.0, args.fx, h0 / 2.0], [0.0, 0.0, 1.0]]),
            "dist_coeffs": np.zeros(5), "R_lc": np.eye(3), "t_lc_m": np.zeros((3, 1)),
            "image_width": w0, "image_height": h0,
        }
    producer = ReplayProducer()
    adapter = IdsSiamFCCorrectionSourceAdapter(cfg, cam.GatingConfig(), producer=producer, calibration=calibration)
    t0 = time.perf_counter()
    adapter.start()
    print(f"start (load + warm-up): {time.perf_counter() - t0:.1f} s")
    for name, info in adapter.weights_info.items():
        print(f"  {name}: {info['path']} sha256={info['sha256']}")
    resized = f" (resized from {w_src}x{h_src})" if (w0, h0) != (w_src, h_src) else ""
    print(f"replaying {args.source} ({n_frames if n_frames is not None else '?'} frames {w0}x{h0}{resized}) at {'max speed' if args.fast else f'{args.fps:g} fps'}")

    records: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    last_dets: Any = adapter.last_detections
    job = 0
    period = 1.0 / float(args.fps)
    next_t = time.perf_counter()
    try:
        for i, gray in enumerate(frames):
            producer.publish(gray)
            t_rc = time.perf_counter()
            cand = adapter.read_candidate(time.monotonic_ns())
            read_ms = (time.perf_counter() - t_rc) * 1e3
            obs = adapter.last_observation
            bbox = obs.bbox_xywh if obs is not None else None
            dets = adapter.last_detections
            if dets is not last_dets:  # a detector job with boxes was consumed in this frame
                last_dets = dets
                if dets:
                    job += 1
                    for d in dets:
                        detections.append({"frame": i, "job": job, "x": float(d.bbox_xywh[0]), "y": float(d.bbox_xywh[1]),
                                           "w": float(d.bbox_xywh[2]), "h": float(d.bbox_xywh[3]), "score": float(d.score)})
            gt_pc = None
            if gt is not None and i < len(gt) and not np.isnan(gt[i, 0]):
                gt_pc = gt_to_pixel_centre(gt[i])
            err = iou = None
            if gt_pc is not None and bbox is not None and obs.state == "tracking":
                err = math.hypot(bbox[0] + bbox[2] / 2 - (gt_pc[0] + gt_pc[2] / 2), bbox[1] + bbox[3] / 2 - (gt_pc[1] + gt_pc[3] / 2))
                iou = bbox_iou(bbox, gt_pc)
            records.append({
                "frame": i, "state": obs.state if obs is not None else "none", "reason": cand.reason,
                "reliable": bool(cand.reliable), "confidence": cand.confidence,
                "bbox": None if bbox is None else tuple(float(v) for v in bbox),
                "d_az_rad": cand.d_az_rad if cand.reliable else None, "d_po_rad": cand.d_po_rad if cand.reliable else None,
                "gt": gt_pc, "err_px": err, "iou": iou, "read_ms": read_ms,
                "category": classify(cand.reliable, bbox, gt_pc, has_gt),
            })
            if not args.fast:
                next_t += period
                time.sleep(max(0.0, next_t - time.perf_counter()))
    finally:
        stats = adapter.fast_stats()
        adapter.stop()

    summary = summarize(records, float(args.fps), has_gt)
    summary["detector_boxes"] = summarize_detections(detections, len(records), float(args.fps))
    summary["source_stats"] = {k: stats.get(k) for k in ("detections_submitted", "detections_ok", "detection_failures", "session")}
    summary["settings"] = {"siamfc": cfg["siamfc"], "detector": cfg["detector"], "calibration": str(args.calibration) if args.calibration else "synthetic",
                           "frame_size": [w0, h0], "source_size": [w_src, h_src], "preload": bool(args.preload)}

    print(f"\nframes {summary['frames']}, first reliable frame {summary['first_reliable_frame']}, reliable frames {summary['reliable_frames']}")
    a = summary["after_first_track"]
    print(f"after the first lock: tracking {a['tracking']:.1%}, lost {a['lost']:.1%}, searching {a['searching']:.1%}")
    if has_gt:
        print(f"SAFETY: target present in {summary['frames_target_present']} frames | reliable: correct {summary['correct']}, "
              f"WRONG TARGET {summary['wrong_target']}, DRONE ABSENT {summary['drone_absent']} | coverage {summary['coverage']:.1%} | "
              f"longest dangerous segment {summary['longest_dangerous_segment_s']:.2f} s")
        if summary.get("centre_error_px"):
            e, io = summary["centre_error_px"], summary["iou"]
            print(f"while tracking: centre error median {e['median']:.1f} px, p90 {e['p90']:.1f}, max {e['max']:.1f} | "
                  f"IoU median {io['median']:.3f}, IoU > 0.5 {io['above_0.5']:.1%}")
    r = summary["read_candidate_ms"]
    if r:
        print(f"read_candidate: median {r['median']:.1f} ms, p99 {r['p99']:.1f}, max {r['max']:.1f}")
    print(f"reasons: {summary['reasons']}")
    print(f"session: {summary['source_stats'].get('session')}")
    db = summary["detector_boxes"]
    print(f"detector boxes: {db['boxes']} in {db['jobs_with_boxes']} jobs (of {summary['source_stats'].get('detections_submitted')} submitted), "
          f"max score {db['max_score'] if db['max_score'] is None else round(db['max_score'], 3)}; "
          + ", ".join(f"score >= {lv:g}: {db[f'boxes_score_ge_{lv:g}']}" for lv in SCORE_LEVELS))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "state", "reason", "reliable", "confidence", "bx", "by", "bw", "bh", "d_az_rad", "d_po_rad",
                        "gx", "gy", "gw", "gh", "err_px", "iou", "read_ms", "category"])
            for rec in records:
                b = rec["bbox"] or ("",) * 4
                g = rec["gt"] or ("",) * 4
                w.writerow([rec["frame"], rec["state"], rec["reason"], int(rec["reliable"]), rec["confidence"], *b,
                            rec["d_az_rad"], rec["d_po_rad"], *g, rec["err_px"], rec["iou"], f"{rec['read_ms']:.3f}", rec["category"] or ""])
        args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
        det_path = args.out.with_name(args.out.stem + "_detections.csv")
        with open(det_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["frame", "job", "x", "y", "w", "h", "score"])
            w.writeheader()
            w.writerows(detections)
        print(f"written {args.out}, {args.out.with_suffix('.json')} and {det_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
