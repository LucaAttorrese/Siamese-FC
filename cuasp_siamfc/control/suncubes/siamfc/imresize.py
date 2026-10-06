"""MATLAB-compatible bicubic ``imresize`` expressed as separable matrices.

The reference tracker (CFNet ``tracker_step.m`` / ``get_subwindow_tracking.m``)
uses MATLAB ``imresize(..., 'bicubic')``: Keys cubic kernel with a = -0.5,
antialiasing when shrinking, symmetric border handling. OpenCV's INTER_CUBIC
differs (a = -0.75, no antialiasing), which would shift the response peak by a
fraction of a pixel, so this module rebuilds MATLAB's contribution weights
exactly. A resize from (H, W) to (h, w) becomes ``A_h @ img @ A_w.T``. The
matrices are cached per (in_len, out_len), so the per-frame cost is two matmuls
that can run on the GPU.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np


def _cubic(x: np.ndarray) -> np.ndarray:
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    return (
        (1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1.0)
        + (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0) * ((absx > 1.0) & (absx <= 2.0))
    )


@lru_cache(maxsize=256)
def resize_matrix(in_len: int, out_len: int, antialias: bool = True) -> np.ndarray:
    """(out_len, in_len) float64 weights, equal to MATLAB's ``contributions``.

    The scale is ``out_len / in_len``, as when MATLAB is given an explicit output
    size. The tracker always calls it with sizes whose ratio is the intended
    scale, so the result is identical to ``imresize(img, scale)``.
    """
    in_len = int(in_len)
    out_len = int(out_len)
    if in_len <= 0 or out_len <= 0:
        raise ValueError(f"invalid resize lengths {in_len} -> {out_len}")
    scale = out_len / in_len
    kernel_width = 4.0
    if scale < 1.0 and antialias:
        kernel = lambda x: scale * _cubic(scale * x)  # noqa: E731
        kernel_width = kernel_width / scale
    else:
        kernel = _cubic

    # MATLAB 1-based output coordinates mapped into input space.
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - kernel_width / 2.0)
    taps = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(taps, dtype=np.float64)[None, :]  # 1-based
    weights = kernel(u[:, None] - indices)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    # Symmetric (mirror) border handling, as MATLAB's
    # aux = [1:in_len, in_len:-1:1].
    aux = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    idx0 = aux[np.mod(indices.astype(np.int64) - 1, 2 * in_len)]

    matrix = np.zeros((out_len, in_len), dtype=np.float64)
    rows = np.repeat(np.arange(out_len), taps)
    np.add.at(matrix, (rows, idx0.reshape(-1)), weights.reshape(-1))
    return matrix


def imresize_np(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D (H, W) or 3-D (H, W, C) array like MATLAB bicubic imresize."""
    arr = np.asarray(img, dtype=np.float64)
    a_h = resize_matrix(arr.shape[0], int(out_hw[0]))
    a_w = resize_matrix(arr.shape[1], int(out_hw[1]))
    if arr.ndim == 2:
        return a_h @ arr @ a_w.T
    return np.einsum("ih,hwc,jw->ijc", a_h, arr, a_w)
