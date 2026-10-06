"""SiamFC (CFNet baseline-conv5) network in PyTorch, loaded from converted weights.

Input images are float tensors in [0, 255], with no mean subtraction (``subMean = false``
in the reference). Single-channel input is supported exactly: the reference
replicates a gray frame into 3 identical channels, which is equivalent to one
channel convolved with conv1 summed over its input channels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SiamFCNet(nn.Module):
    def __init__(self, payload: dict[str, Any]):
        super().__init__()
        state = payload["state_dict"]
        self.groups = dict(payload["groups"])
        self.strides = dict(payload["strides"])
        self.pool_sizes = dict(payload["pool_sizes"])
        self.adjust = {k: float(v) for k, v in payload["adjust"].items()}
        self.meta = dict(payload.get("meta", {}))
        self.n_convs = len(self.groups)
        for i in range(1, self.n_convs + 1):
            self.register_buffer(f"w{i}", state[f"conv{i}.weight"].float().clone())
            self.register_buffer(f"b{i}", state[f"conv{i}.bias"].float().clone())
        self.register_buffer("w1_gray", self.w1.sum(dim=1, keepdim=True))

    @classmethod
    def from_file(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "SiamFCNet":
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        return cls(payload)

    def features(self, img: torch.Tensor) -> torch.Tensor:
        """(N, 1|3, H, W) in [0, 255] -> (N, C, h, w) branch output (no ReLU after the last conv)."""
        x = img
        for i in range(1, self.n_convs + 1):
            w = getattr(self, f"w{i}")
            if i == 1 and x.shape[1] == 1:
                w = self.w1_gray
            x = F.conv2d(x, w, getattr(self, f"b{i}"), stride=self.strides[f"conv{i}"], groups=self.groups[f"conv{i}"])
            if i < self.n_convs:
                x = F.relu(x)
                pool = f"pool{i}"
                if pool in self.pool_sizes:
                    x = F.max_pool2d(x, self.pool_sizes[pool], self.strides[pool])
        return x

    def score(self, z_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Cross-correlate one exemplar (1, C, hz, wz) with N instances (N, C, hx, wx) -> (N, hs, ws).

        Then apply the final scalar BatchNorm ``fin_adjust_bn``.
        """
        if z_feat.shape[0] != 1:
            raise ValueError("score() expects a single exemplar")
        raw = F.conv2d(x_feat, z_feat)[:, 0]
        a = self.adjust
        return a["mult"] * (raw - a["mean"]) / a["sigma"] + a["bias"]
