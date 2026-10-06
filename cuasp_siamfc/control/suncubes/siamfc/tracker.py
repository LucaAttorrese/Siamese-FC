"""SiamFC tracker: port of CFNet ``tracker.m`` / ``tracker_step.m`` / ``make_scale_pyramid.m``.

Reference sources: ``../../../../reference/cfnet/*.m`` ('xcorr' join method).

Coordinate convention (public API): 0-based pixel-centre coordinates, i.e.
pixel (0, 0) is centred at (0.0, 0.0), as in OpenCV. The reference MATLAB code
uses 1-based coordinates; the only place where that matters is the crop
placement in ``crop_with_padding`` (``round(pos - (sz + 1) / 2)`` in 1-based
terms), and it is converted there explicitly. Positions inside this module are
stored as (y, x) like the reference ``targetPosition``, and converted to (x, y)
at the public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from .imresize import resize_matrix
from .net import SiamFCNet


@dataclass(frozen=True)
class SiamFCParams:
    """Defaults are CFNet ``run_baseline5_evaluation.m`` (Baseline-conv5, improved SiamFC)."""

    num_scale: int = 3
    scale_step: float = 1.0470
    scale_penalty: float = 0.9825
    scale_lr: float = 0.68
    response_up: int = 8
    w_influence: float = 0.175
    z_lr: float = 0.0102
    min_s_factor: float = 0.2
    max_s_factor: float = 5.0
    # Network geometry (must match the trained network).
    exemplar_size: int = 127
    instance_size: int = 255
    score_size: int = 33
    total_stride: int = 4
    context_amount: float = 0.5

    def validate(self) -> None:
        if self.num_scale < 1 or self.num_scale % 2 == 0:
            raise ValueError("num_scale must be odd and >= 1")
        for name in ("scale_step", "scale_penalty", "scale_lr", "w_influence", "min_s_factor", "max_s_factor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not 0.0 <= self.z_lr < 1.0:
            raise ValueError("z_lr must be in [0, 1)")
        if self.response_up < 1:
            raise ValueError("response_up must be >= 1")


@dataclass(frozen=True)
class SiamFCResult:
    center_xy: tuple[float, float]  # 0-based pixel coordinates
    size_wh: tuple[float, float]  # px
    peak: float  # raw max of the chosen (upsampled) response map, before normalisation
    apce: float  # average peak-to-correlation energy of the chosen response map
    scale_index: int  # 0..num_scale-1; num_scale // 2 means "no scale change"
    s_x: float  # search-region side in frame px


def matlab_round(value: float) -> int:
    """MATLAB round(): half away from zero (Python's round() is half-to-even)."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def crop_with_padding(
    img: np.ndarray, pos_yx: tuple[float, float], side: int, fill: np.ndarray
) -> np.ndarray:
    """Square ``side`` x ``side`` crop centred on ``pos_yx`` (0-based), with out-of-image
    pixels set to ``fill``. Equivalent to the indexing part of
    ``get_subwindow_tracking.m``. Returns an (H, W, C) float32 array.
    """
    side = int(side)
    if side < 1:
        raise ValueError("crop side must be >= 1")
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    c = (side + 1) / 2.0
    # 1-based MATLAB: context_min = round(pos1 - c), pos1 = pos0 + 1 -> 0-based start = that - 1.
    y0 = matlab_round(pos_yx[0] + 1.0 - c) - 1
    x0 = matlab_round(pos_yx[1] + 1.0 - c) - 1
    out = np.empty((side, side, channels), np.float32)
    out[...] = np.asarray(fill, np.float32).reshape(1, 1, channels)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(h, y0 + side), min(w, x0 + side)
    if sy1 > sy0 and sx1 > sx0:
        src = img[sy0:sy1, sx0:sx1]
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = src.reshape(sy1 - sy0, sx1 - sx0, channels)
    return out


def hann_window(n: int) -> np.ndarray:
    """MATLAB ``hann(n) * hann(n)'`` normalised to sum 1 (symmetric Hann)."""
    w = np.hanning(n)
    window = np.outer(w, w)
    return window / window.sum()


def displacement_in_frame(
    r0: int, c0: int, p: SiamFCParams, s_x: float
) -> tuple[float, float]:
    """Peak (0-based row/col in the upsampled response) -> (dy, dx) in frame px.

    Reference: ``disp = p_corr - (scoreSize*responseUp + 1)/2`` with 1-based p_corr.
    """
    centre = (p.score_size * p.response_up + 1) / 2.0
    k = p.total_stride / p.response_up * s_x / p.instance_size
    return ((r0 + 1) - centre) * k, ((c0 + 1) - centre) * k


class SiamFCTracker:
    def __init__(self, net: SiamFCNet, params: SiamFCParams = SiamFCParams(), device: str | torch.device = "cpu"):
        params.validate()
        self.p = params
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self._mats: dict[tuple[int, int], torch.Tensor] = {}
        n_up = params.score_size * params.response_up
        self._window = torch.as_tensor(hann_window(n_up), dtype=torch.float32, device=self.device)
        mid = params.num_scale // 2
        self._scales = params.scale_step ** (np.arange(params.num_scale) - mid)
        self._penalty = torch.full((params.num_scale,), params.scale_penalty, device=self.device)
        self._penalty[mid] = 1.0
        self._initialised = False

    # ------------------------------------------------------------------ helpers

    def _mat(self, n_in: int, n_out: int) -> torch.Tensor:
        key = (int(n_in), int(n_out))
        mat = self._mats.get(key)
        if mat is None:
            mat = torch.as_tensor(resize_matrix(*key), dtype=torch.float32, device=self.device)
            self._mats[key] = mat
        return mat

    def _resize(self, chw: torch.Tensor, out_side: int) -> torch.Tensor:
        """(..., H, W) -> (..., out, out), MATLAB bicubic imresize."""
        h, w = chw.shape[-2:]
        if h == out_side and w == out_side:
            return chw
        return self._mat(h, out_side) @ chw @ self._mat(w, out_side).T

    def _scale_pyramid(self, frame: np.ndarray, pos_yx: tuple[float, float], in_sides: np.ndarray, out_side: int) -> torch.Tensor:
        """``make_scale_pyramid.m``: (S, C, out, out) float tensor in [0, 255]."""
        in_sides = np.array([matlab_round(v) for v in in_sides], dtype=np.int64)
        max_side, min_side = int(in_sides[-1]), int(in_sides[0])
        search_side = matlab_round(out_side * max_side / min_side)
        region = crop_with_padding(frame, pos_yx, max_side, self._avg)
        region_t = torch.from_numpy(region).to(self.device).permute(2, 0, 1)
        region_t = self._resize(region_t, search_side)  # (C, search, search)
        centre = (search_side - 1) / 2.0  # 0-based form of (1 + search_side) / 2
        # Reference quirk, reproduced on purpose: get_subwindow_tracking.m places a
        # crop of side sz at round(pos - (sz + 1) / 2), i.e. centred one pixel
        # before pos. For the largest scale this starts at index -1 of the
        # search region, and MATLAB pads that row/column with avgChans.
        # See crop_with_padding() and README "Known reference quirks".
        fill = torch.as_tensor(self._avg, dtype=region_t.dtype, device=self.device).view(-1, 1, 1)
        padded = fill.expand(-1, search_side + 2, search_side + 2).clone()
        padded[:, 1:-1, 1:-1] = region_t
        crops = []
        for side in in_sides:
            target_side = matlab_round(out_side * int(side) / min_side)
            c = (target_side + 1) / 2.0
            start = matlab_round(centre + 1.0 - c) - 1
            if start < -1 or start + target_side > search_side + 1:
                raise RuntimeError("scale pyramid crop exceeds search region")  # cannot happen for scale_step >= 1
            sub = padded[:, start + 1:start + 1 + target_side, start + 1:start + 1 + target_side]
            crops.append(self._resize(sub, out_side))
        return torch.stack(crops, 0)

    def _exemplar_features(self, frame: np.ndarray) -> torch.Tensor:
        z_crops = self._scale_pyramid(frame, self._pos, self._s_z * self._scales, self.p.exemplar_size)
        mid = self.p.num_scale // 2
        return self.net.features(z_crops[mid:mid + 1])

    @staticmethod
    def _as_hwc(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            return arr[:, :, None]
        if arr.ndim == 3 and arr.shape[2] in (1, 3):
            return arr
        raise ValueError(f"unsupported frame shape {arr.shape}")

    # ------------------------------------------------------------------ API

    @property
    def initialised(self) -> bool:
        return self._initialised

    @torch.inference_mode()
    def init(self, frame: np.ndarray, center_xy: tuple[float, float], size_wh: tuple[float, float]) -> None:
        frame = self._as_hwc(frame)
        w, h = float(size_wh[0]), float(size_wh[1])
        if not (w > 0.0 and h > 0.0):
            raise ValueError("target size must be positive")
        p = self.p
        self._avg = frame.reshape(-1, frame.shape[2]).mean(axis=0).astype(np.float32)
        self._pos = (float(center_xy[1]), float(center_xy[0]))
        self._size = (h, w)
        wc_z = w + p.context_amount * (w + h)
        hc_z = h + p.context_amount * (w + h)
        self._s_z = math.sqrt(wc_z * hc_z)
        self._s_x = p.instance_size / p.exemplar_size * self._s_z
        self._min_s_x, self._max_s_x = p.min_s_factor * self._s_x, p.max_s_factor * self._s_x
        self._min_s_z, self._max_s_z = p.min_s_factor * self._s_z, p.max_s_factor * self._s_z
        self._z_feat = self._exemplar_features(frame)
        self._initialised = True

    @torch.inference_mode()
    def update(self, frame: np.ndarray) -> SiamFCResult:
        if not self._initialised:
            raise RuntimeError("tracker not initialised")
        frame = self._as_hwc(frame)
        p = self.p
        scaled_instance = self._s_x * self._scales
        x_crops = self._scale_pyramid(frame, self._pos, scaled_instance, p.instance_size)
        response = self.net.score(self._z_feat, self.net.features(x_crops))  # (S, 33, 33)
        n_up = p.score_size * p.response_up
        up = self._resize(response, n_up)  # (S, n_up, n_up)

        peaks = up.amax(dim=(1, 2)) * self._penalty
        best = int(torch.argmax(peaks).item())  # first maximum, like the strict '>' loop
        chosen = up[best]
        raw_peak = float(chosen.max().item())
        r = chosen - chosen.min()
        apce = float((r.max() ** 2 / torch.clamp((r ** 2).mean(), min=1e-12)).item())
        r = r / r.sum()
        r = (1.0 - p.w_influence) * r + p.w_influence * self._window
        # MATLAB find(..., 1) scans column-major: first max in column order.
        flat = int(torch.argmax(r.T.reshape(-1)).item())
        c0, r0 = divmod(flat, n_up)
        dy, dx = displacement_in_frame(r0, c0, p, self._s_x)
        self._pos = (self._pos[0] + dy, self._pos[1] + dx)

        self._s_x = max(self._min_s_x, min(self._max_s_x, (1 - p.scale_lr) * self._s_x + p.scale_lr * scaled_instance[best]))

        if p.z_lr > 0.0:
            scaled_exemplar = self._s_z * self._scales
            z_new = self._exemplar_features(frame)
            self._z_feat = (1.0 - p.z_lr) * self._z_feat + p.z_lr * z_new
            self._s_z = max(self._min_s_z, min(self._max_s_z, (1 - p.scale_lr) * self._s_z + p.scale_lr * scaled_exemplar[best]))

        s = self._scales[best]
        self._size = (
            (1 - p.scale_lr) * self._size[0] + p.scale_lr * self._size[0] * s,
            (1 - p.scale_lr) * self._size[1] + p.scale_lr * self._size[1] * s,
        )
        return SiamFCResult(
            center_xy=(self._pos[1], self._pos[0]),
            size_wh=(self._size[1], self._size[0]),
            peak=raw_peak,
            apce=apce,
            scale_index=best,
            s_x=float(self._s_x),
        )


def params_with(**overrides) -> SiamFCParams:
    return replace(SiamFCParams(), **overrides)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = [
    "SiamFCParams",
    "SiamFCResult",
    "SiamFCTracker",
    "crop_with_padding",
    "displacement_in_frame",
    "hann_window",
    "matlab_round",
    "params_with",
    "default_device",
]
