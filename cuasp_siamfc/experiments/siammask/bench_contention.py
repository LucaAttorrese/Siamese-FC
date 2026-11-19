"""Tracker update latency with the YOLO verification running on the same GPU (P7 step 0).

Real IDS frames are preloaded and fed at the camera rate (62.6 fps). The tracker runs in the main
thread on every frame (initialised from the label); a second thread runs the drone detector on
the latest frame at ``--det-hz`` (5 Hz = verification while tracking, as in the runtime). Only
the tracker update is timed, so SiamFC and SiamMask are compared under the same contention.

Usage (from cuasp_siamfc/):
  .venv/Scripts/python experiments/siammask/bench_contention.py 20261117_102552_B6_sfondi_6db_r3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
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
from suncubes import settings  # noqa: E402
from suncubes.siamfc.detector import DetectorConfig, UltralyticsDroneDetector  # noqa: E402
from suncubes.siamfc.net import SiamFCNet  # noqa: E402
from suncubes.siamfc.tracker import SiamFCTracker  # noqa: E402
from tracker import SiamMaskTracker  # noqa: E402


def run(tracker, frames, box, fps, detector, det_hz):
    latest = {"img": frames[0]}
    stop = threading.Event()
    det_ms = []

    def det_loop():
        period = 1.0 / det_hz
        nxt = time.perf_counter()
        while not stop.is_set():
            nxt += period
            t0 = time.perf_counter()
            detector.detect(latest["img"])
            det_ms.append((time.perf_counter() - t0) * 1e3)
            time.sleep(max(0.0, nxt - time.perf_counter()))

    x, y, w, h = box
    tracker.init(frames[0], (x + w / 2 - 0.5, y + h / 2 - 0.5), (w, h))
    th = threading.Thread(target=det_loop, daemon=True)
    if detector is not None:
        th.start()
    lat = []
    period = 1.0 / fps
    t_next = time.perf_counter()
    for img in frames[1:]:
        t_next += period
        latest["img"] = img
        t0 = time.perf_counter()
        tracker.update(img)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
        time.sleep(max(0.0, t_next - time.perf_counter()))
    stop.set()
    if detector is not None:
        th.join()
    return np.array(lat[20:]), np.array(det_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=float, default=62.6)
    ap.add_argument("--det-hz", type=float, default=5.0)
    args = ap.parse_args()
    paths = sorted((ROOT / "recordings" / args.recording / "frames").glob("*.jpg"))[:args.frames]
    gt = read_gt(ROOT / "labels" / "ids" / f"{args.recording}.csv", len(paths))
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    box = tuple(float(v) for v in gt[0])
    torch.backends.cudnn.benchmark = True
    det = UltralyticsDroneDetector(DetectorConfig(weights_path=str((HERE.parents[1] / settings.CV_DETECTOR_WEIGHTS_PATH).resolve())))
    det.warmup(frames[0].shape)
    budget = 1e3 / args.fps

    def make(name):
        if name == "siamfc":
            return SiamFCTracker(SiamFCNet.from_file(ROOT / "pretrained/networks/baseline-conv5_gray_e100.pth"), device="cuda")
        net, _ = load_siammask(str(ROOT / "pretrained/siammask/SiamMask_VOT.pth"), "cuda")
        fold_bn(net)
        return SiamMaskTracker(net, crop="torch", mask=None if name == "siammask box" else "window", fp16=True)

    print(f"{args.recording}: {len(frames)} frames at {args.fps} fps, detector at {args.det_hz} Hz on the same GPU; budget {budget:.1f} ms")
    for name in ("siamfc", "siammask box", "siammask mask"):
        for with_det in (False, True):
            lat, dms = run(make(name), frames, box, args.fps, det if with_det else None, args.det_hz)
            over = np.mean(lat > budget) * 100
            print(f"  {name:14s} {'with detector' if with_det else 'alone        '}  median {np.median(lat):5.2f}  p95 {np.percentile(lat, 95):5.2f}  "
                  f"p99 {np.percentile(lat, 99):5.2f}  max {lat.max():5.2f} ms  over budget {over:4.1f} %"
                  + (f"   (detector {np.median(dms):.0f} ms median)" if with_det else ""), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
