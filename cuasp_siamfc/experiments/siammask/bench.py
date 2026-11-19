"""Latency of the SiamMask port on IDS-sized frames (2040x1536 Mono8), with a stage breakdown.

A real IDS frame is used when given (``--frame``), otherwise a synthetic one with a dark blob.
The tracker is re-initialised on the same target every ``--reinit`` frames so that the search
region keeps the requested size.

Usage (from cuasp_siamfc/):
  .venv/Scripts/python experiments/siammask/bench.py --weights ../pretrained/siammask/SiamMask_VOT.pth
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

from net import load_siammask  # noqa: E402
from tracker import SiamMaskTracker  # noqa: E402


def fold_bn(net: torch.nn.Module) -> torch.nn.Module:
    """Fold every BatchNorm2d that follows a bias-free Conv2d into the conv (eval only)."""
    import torch.nn as nn

    def fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        w = conv.weight
        std = torch.sqrt(bn.running_var + bn.eps)
        new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                        conv.dilation, conv.groups, bias=True).to(w.device, w.dtype)
        new.weight.data = w * (bn.weight / std).view(-1, 1, 1, 1)
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        new.bias.data = (b - bn.running_mean) * bn.weight / std + bn.bias
        return new

    for module in net.modules():
        names = list(module._modules.keys())
        for i, name in enumerate(names[:-1]):
            a, b = module._modules[name], module._modules[names[i + 1]]
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                module._modules[name] = fold(a, b)
                module._modules[names[i + 1]] = nn.Identity()
        # Bottleneck: conv1/bn1, conv2/bn2, conv3/bn3 are attributes, not a Sequential
        for cn, bn_name in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
            if isinstance(getattr(module, cn, None), nn.Conv2d) and isinstance(getattr(module, bn_name, None), nn.BatchNorm2d):
                setattr(module, cn, fold(getattr(module, cn), getattr(module, bn_name)))
                setattr(module, bn_name, nn.Identity())
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--frame", type=Path, help="a real 2040x1536 IDS frame (JPEG/PNG)")
    ap.add_argument("--box", type=float, nargs=4, default=[900, 700, 290, 91], help="target x y w h in the frame")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--reinit", type=int, default=50)
    args = ap.parse_args()

    if args.frame:
        frame = cv2.imread(str(args.frame), cv2.IMREAD_GRAYSCALE)
    else:
        rng = np.random.default_rng(0)
        frame = (rng.normal(90, 12, (1536, 2040))).clip(0, 255).astype(np.uint8)
        x, y, w, h = (int(v) for v in args.box)
        frame[y:y + h, x:x + w] = 30
    x, y, w, h = args.box
    center, size = (x + w / 2, y + h / 2), (w, h)
    torch.backends.cudnn.benchmark = True
    configs = [
        ("box only, fp32", dict(mask=None, fp16=False), False),
        ("box only, fp16", dict(mask=None, fp16=True), False),
        ("box only, fp16, BN folded", dict(mask=None, fp16=True), True),
        ("mask (window), fp32", dict(mask="window", fp16=False), False),
        ("mask (window), fp16", dict(mask="window", fp16=True), False),
        ("mask (window), fp16, BN folded", dict(mask="window", fp16=True), True),
        ("mask (full frame, original), fp32", dict(mask="full", fp16=False), False),
    ]
    print(f"frame {frame.shape[1]}x{frame.shape[0]}, target {w:.0f}x{h:.0f} px, {args.frames} updates")
    for name, cfg, folded in configs:
        net, _ = load_siammask(str(args.weights), "cuda")
        if folded:
            fold_bn(net)
        tr = SiamMaskTracker(net, crop="torch", **cfg)
        totals, stages = [], {}
        for i in range(args.frames + 20):
            if i % args.reinit == 0:
                tr.init(frame, center, size)
            torch.cuda.synchronize()
            r = tr.update(frame, sync_timing=True)
            torch.cuda.synchronize()
            if i >= 20:
                totals.append(r.timing_ms["total"])
                for k, v in r.timing_ms.items():
                    stages.setdefault(k, []).append(v)
        t = np.array(totals)
        st = "  ".join(f"{k} {np.median(v):.2f}" for k, v in stages.items() if k != "total")
        print(f"{name:36s} median {np.median(t):5.2f} ms  p99 {np.percentile(t, 99):5.2f}  max {t.max():5.2f}   [{st}]")
        del tr, net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
