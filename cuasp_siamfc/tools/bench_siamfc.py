"""Latency benchmark of the SiamFC port on IDS-sized Mono8 frames (validation V5).

Usage (from cuasp_siamfc/)::

    .venv/Scripts/python tools/bench_siamfc.py --weights ../pretrained/networks/baseline-conv5_gray_e100.pth

Measures ``SiamFCTracker.update()`` end to end (CUDA-synchronised) on synthetic
2592x1944 Mono8 frames, for several target sizes, with and without the
per-frame template update (zLR). A stage breakdown (crop+H2D+resize, backbone,
xcorr+upsample+peak, template update) is measured on the same inputs.
Run it on every target machine: numbers depend on GPU, CPU and driver.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))

import torch  # noqa: E402

from suncubes.siamfc.net import SiamFCNet  # noqa: E402
from suncubes.siamfc.tracker import SiamFCTracker, default_device, params_with  # noqa: E402


def _frames(n: int, size_wh: tuple[int, int], target_px: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    w, h = size_wh
    background = rng.normal(120, 10, (h, w)).clip(0, 255).astype(np.uint8)
    target = rng.integers(0, 255, (target_px, target_px), dtype=np.uint8)
    frames, centres = [], []
    for i in range(n):
        cx, cy = w / 2 + 2.0 * np.sin(i / 7) * target_px / 10, h / 2 + 1.5 * i % 7
        img = background.copy()
        x0, y0 = int(cx - target_px / 2), int(cy - target_px / 2)
        img[y0:y0 + target_px, x0:x0 + target_px] = target
        frames.append(img)
        centres.append((x0 + (target_px - 1) / 2, y0 + (target_px - 1) / 2))
    return frames, centres


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timed(fn, device: str) -> float:
    _sync(device)
    t0 = time.perf_counter()
    fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1e3


def bench_update(net, device, target_px, z_lr, n=300, warmup=30) -> np.ndarray:
    frames, centres = _frames(n + warmup + 1, (2592, 1944), target_px)
    tr = SiamFCTracker(net, params_with(z_lr=z_lr), device=device)
    tr.init(frames[0], centres[0], (float(target_px), float(target_px)))
    times = [_timed(lambda f=f: tr.update(f), device) for f in frames[1:]]
    return np.asarray(times[warmup:])


@torch.inference_mode()
def bench_stages(net, device, target_px, n=200) -> dict[str, float]:
    frames, centres = _frames(n + 1, (2592, 1944), target_px)
    tr = SiamFCTracker(net, params_with(), device=device)
    tr.init(frames[0], centres[0], (float(target_px), float(target_px)))
    p = tr.p
    out = {"crop+H2D+resize": [], "backbone x3": [], "xcorr+upsample+peak": [], "template update": []}
    for f in frames[1:]:
        hwc = tr._as_hwc(f)
        holder = {}
        out["crop+H2D+resize"].append(_timed(lambda: holder.__setitem__("x", tr._scale_pyramid(hwc, tr._pos, tr._s_x * tr._scales, p.instance_size)), device))
        out["backbone x3"].append(_timed(lambda: holder.__setitem__("f", net.features(holder["x"])), device))

        def head():
            r = net.score(tr._z_feat, holder["f"])
            up = tr._resize(r, p.score_size * p.response_up)
            int(torch.argmax((up.amax(dim=(1, 2)) * tr._penalty)).item())

        out["xcorr+upsample+peak"].append(_timed(head, device))
        out["template update"].append(_timed(lambda: tr._exemplar_features(hwc), device))
    return {k: float(np.median(v)) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--targets", type=int, nargs="+", default=[16, 48, 150, 400])
    args = ap.parse_args()

    net = SiamFCNet.from_file(args.weights)
    name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
    print(f"device={args.device} ({name}) torch={torch.__version__} frame=2592x1944 Mono8")
    print(f"{'target px':>9} {'zLR':>7} {'mean ms':>8} {'p50':>7} {'p99':>7} {'max':>7} {'Hz@mean':>8}")
    for t in args.targets:
        for z_lr in (0.0102, 0.0):
            ms = bench_update(net, args.device, t, z_lr)
            print(f"{t:>9} {z_lr:>7.4f} {ms.mean():>8.2f} {np.percentile(ms, 50):>7.2f} {np.percentile(ms, 99):>7.2f} {ms.max():>7.2f} {1e3 / ms.mean():>8.0f}")
    print("stage breakdown (median ms, target 48 px, zLR on):")
    for k, v in bench_stages(net, args.device, 48).items():
        print(f"  {k:<22} {v:6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
