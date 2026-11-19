"""SiamMask tracking step (VOT setting: RPN box + refined mask), ported from tools/test.py
(``siamese_init`` / ``siamese_track``) of SiamMask, MIT License, Copyright (c) 2019 Qiang Wang.

Feasibility study only (proposal P7 in PLAN.md, step 0). Two options select the faithful
original path or the faster one; the faster one is checked against the original by
``compare_vot.py``:

- ``crop="cv2"``: crops as the original (NumPy padding + ``cv2.resize`` INTER_LINEAR, uint8
  rounding); ``crop="torch"``: same window, padding and bilinear resize on the GPU
  (``align_corners=False``, no antialiasing, float output).
- ``mask=None``: box only (the original with ``mask_enable=False``); ``"full"``: the original
  mask post-processing (mask warped onto the whole frame, contours, rotated rectangle);
  ``"window"``: the same warp restricted to the mask's own window in the frame (identical
  result: outside it the original warp only writes the padding value).

The tracking state (position, size) comes from the RPN box in every mode, as in the original;
the mask only adds an output. Coordinates: 0-based pixel centres, (x, y), like
``siamfc.tracker``. The original crop placement ``round(pos - (sz + 1) / 2)`` is kept as is.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SiamMaskParams:
    """experiments/siammask_sharp/config_vot.json (hp + anchors) and TrackerConfig defaults."""

    penalty_k: float = 0.04
    window_influence: float = 0.42
    lr: float = 0.25
    seg_thr: float = 0.30
    exemplar_size: int = 127
    instance_size: int = 255
    total_stride: int = 8
    base_size: int = 8
    out_size: int = 127
    context_amount: float = 0.5
    ratios: tuple = (0.33, 0.5, 1, 2, 3)
    scales: tuple = (8,)

    @property
    def score_size(self) -> int:
        return (self.instance_size - self.exemplar_size) // self.total_stride + 1 + self.base_size

    @property
    def anchor_num(self) -> int:
        return len(self.ratios) * len(self.scales)


@dataclass
class SiamMaskResult:
    center_xy: tuple[float, float]  # RPN box (tracking state)
    size_wh: tuple[float, float]
    peak: float  # raw foreground probability of the chosen anchor
    apce: float  # APCE of the raw score map of the chosen anchor shape
    mask_box_xywh: Optional[tuple[float, float, float, float]] = None  # axis-aligned box of the largest mask blob
    mask_center_xy: Optional[tuple[float, float]] = None  # centroid of the largest mask blob
    polygon: Optional[np.ndarray] = None  # rotated rectangle (VOT output); RPN box corners if the mask is empty
    timing_ms: Optional[dict] = None


def generate_anchor(p: SiamMaskParams) -> np.ndarray:
    """``utils.anchors.Anchors.generate_anchors`` + ``tools.test.generate_anchor``:
    (anchor_num * score_size^2, 4) as (cx, cy, w, h), index a * S^2 + y * S + x."""
    size = p.total_stride * p.total_stride
    base = []
    for r in p.ratios:
        ws = int(math.sqrt(size * 1.0 / r))
        hs = int(ws * r)
        for s in p.scales:
            w, h = ws * s, hs * s
            base.append([-w * 0.5, -h * 0.5, w * 0.5, h * 0.5])
    base = np.array(base, np.float32)
    x1, y1, x2, y2 = base[:, 0], base[:, 1], base[:, 2], base[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
    S, n = p.score_size, anchor.shape[0]
    anchor = np.tile(anchor, S * S).reshape((-1, 4))
    ori = -(S // 2) * p.total_stride
    xx, yy = np.meshgrid([ori + p.total_stride * dx for dx in range(S)], [ori + p.total_stride * dy for dy in range(S)])
    anchor[:, 0] = np.tile(xx.flatten(), (n, 1)).flatten().astype(np.float32)
    anchor[:, 1] = np.tile(yy.flatten(), (n, 1)).flatten().astype(np.float32)
    return anchor


def crop_window(pos_xy, original_sz: int):
    """Window of ``get_subwindow_tracking``: (xmin, ymin) of an ``original_sz`` square."""
    c = (original_sz + 1) / 2
    return round(pos_xy[0] - c), round(pos_xy[1] - c)  # Python round, as the original (Python 3)


def get_subwindow_cv2(im: np.ndarray, pos_xy, model_sz: int, original_sz: int, avg_chans: np.ndarray) -> np.ndarray:
    """``get_subwindow_tracking`` of the original (HxWx3 uint8 in, model_sz^2 x 3 uint8 out)."""
    sz = original_sz
    r, c, k = im.shape
    xmin, ymin = crop_window(pos_xy, sz)
    xmax, ymax = xmin + sz - 1, ymin + sz - 1
    left, top = int(max(0.0, -xmin)), int(max(0.0, -ymin))
    right, bottom = int(max(0.0, xmax - c + 1)), int(max(0.0, ymax - r + 1))
    xmin, xmax, ymin, ymax = xmin + left, xmax + left, ymin + top, ymax + top
    if any([top, bottom, left, right]):
        te = np.zeros((r + top + bottom, c + left + right, k), np.uint8)
        te[top:top + r, left:left + c, :] = im
        if top:
            te[0:top, left:left + c, :] = avg_chans
        if bottom:
            te[r + top:, left:left + c, :] = avg_chans
        if left:
            te[:, 0:left, :] = avg_chans
        if right:
            te[:, c + left:, :] = avg_chans
        patch = te[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        patch = im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    if model_sz != original_sz:
        patch = cv2.resize(patch, (model_sz, model_sz))
    return patch


class SiamMaskTracker:
    def __init__(self, net, params: SiamMaskParams = SiamMaskParams(), device="cuda", crop: str = "torch",
                 mask: Optional[str] = "window", fp16: bool = False):
        if crop not in ("cv2", "torch") or mask not in (None, "full", "window"):
            raise ValueError("crop must be cv2|torch, mask None|full|window")
        self.net, self.p, self.device = net, params, torch.device(device)
        self.crop_mode, self.mask_mode, self.fp16 = crop, mask, bool(fp16)
        self.dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            self.net.half()
        self.anchor = generate_anchor(params)
        S = params.score_size
        window = np.outer(np.hanning(S), np.hanning(S))
        self.window = np.tile(window.flatten(), params.anchor_num)
        self.pos = None
        self.sz = None
        self.avg = None
        self.im_hw = None

    # ------------------------------------------------------------------ crops

    @staticmethod
    def _as_bgr(frame: np.ndarray) -> np.ndarray:
        return np.repeat(frame[:, :, None], 3, axis=2) if frame.ndim == 2 else frame

    def _crop(self, frame: np.ndarray, pos_xy, model_sz: int, original_sz: int) -> torch.Tensor:
        if self.crop_mode == "cv2":
            patch = get_subwindow_cv2(self._as_bgr(frame), pos_xy, model_sz, original_sz, self.avg)
            t = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).to(self.device)
            return t.to(self.dtype).unsqueeze(0)
        h, w = frame.shape[:2]
        xmin, ymin = crop_window(pos_xy, original_sz)
        sx0, sy0 = max(xmin, 0), max(ymin, 0)
        sx1, sy1 = min(xmin + original_sz, w), min(ymin + original_sz, h)
        out = self._fill.expand(3, original_sz, original_sz).clone()
        if sx1 > sx0 and sy1 > sy0:
            src = torch.from_numpy(np.ascontiguousarray(frame[sy0:sy1, sx0:sx1])).to(self.device)
            src = src.unsqueeze(0).expand(3, -1, -1) if src.ndim == 2 else src.permute(2, 0, 1)
            out[:, sy0 - ymin:sy1 - ymin, sx0 - xmin:sx1 - xmin] = src.to(torch.float32)
        out = out.unsqueeze(0)
        if model_sz != original_sz:
            out = F.interpolate(out, size=(model_sz, model_sz), mode="bilinear", align_corners=False)
        return out.to(self.dtype)

    # ------------------------------------------------------------------ API

    def init(self, frame: np.ndarray, center_xy, size_wh) -> None:
        p = self.p
        self.im_hw = frame.shape[:2]
        self.avg = np.mean(frame, axis=(0, 1)) if frame.ndim == 3 else np.full(3, np.mean(frame))
        self._fill = torch.tensor(self.avg, dtype=torch.float32, device=self.device).view(3, 1, 1)
        self.pos = np.array(center_xy, np.float64)
        self.sz = np.array(size_wh, np.float64)
        wc_z = self.sz[0] + p.context_amount * sum(self.sz)
        hc_z = self.sz[1] + p.context_amount * sum(self.sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        self.net.template(self._crop(frame, self.pos, p.exemplar_size, s_z))

    def update(self, frame: np.ndarray, sync_timing: bool = False) -> SiamMaskResult:
        p = self.p
        t = {}
        t0 = time.perf_counter()
        wc_x = self.sz[1] + p.context_amount * sum(self.sz)
        hc_x = self.sz[0] + p.context_amount * sum(self.sz)
        s_x = np.sqrt(wc_x * hc_x)
        scale_x = p.exemplar_size / s_x
        d_search = (p.instance_size - p.exemplar_size) / 2
        s_x = s_x + 2 * (d_search / scale_x)
        crop_box = [self.pos[0] - round(s_x) / 2, self.pos[1] - round(s_x) / 2, round(s_x), round(s_x)]
        x_crop = self._crop(frame, self.pos, p.instance_size, round(s_x))
        if sync_timing:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if self.mask_mode is None:
            score, delta = self.net.track(x_crop)
            mask_state = None
        else:
            score, delta, mask_state = self.net.track_mask(x_crop)
        delta = delta.float().permute(1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        score = F.softmax(score.float().permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1)[:, 1].cpu().numpy()
        t2 = time.perf_counter()

        a = self.anchor
        delta[0, :] = delta[0, :] * a[:, 2] + a[:, 0]
        delta[1, :] = delta[1, :] * a[:, 3] + a[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * a[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * a[:, 3]

        def change(r):
            return np.maximum(r, 1.0 / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        tsz = self.sz * scale_x
        s_c = change(sz(delta[2, :], delta[3, :]) / sz(tsz[0], tsz[1]))
        r_c = change((tsz[0] / tsz[1]) / (delta[2, :] / delta[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * score
        pscore = pscore * (1 - p.window_influence) + self.window * p.window_influence
        best = int(np.argmax(pscore))
        pred = delta[:, best] / scale_x
        lr = penalty[best] * score[best] * p.lr
        res_x, res_y = pred[0] + self.pos[0], pred[1] + self.pos[1]
        res_w = self.sz[0] * (1 - lr) + pred[2] * lr
        res_h = self.sz[1] * (1 - lr) + pred[3] * lr
        pos = np.array([res_x, res_y])
        size = np.array([res_w, res_h])

        S = p.score_size
        a_idx, dy, dx = np.unravel_index(best, (p.anchor_num, S, S))
        fmap = score.reshape(p.anchor_num, S, S)[a_idx]
        fmin = float(fmap.min())
        apce = (float(fmap.max()) - fmin) ** 2 / max(float(np.mean((fmap - fmin) ** 2)), 1e-12)

        res = SiamMaskResult((0.0, 0.0), (0.0, 0.0), float(score[best]), apce)
        if self.mask_mode is not None:
            m = self.net.track_refine(mask_state, (dy, dx)).float().sigmoid().squeeze().view(p.out_size, p.out_size).cpu().numpy()
            t3 = time.perf_counter()
            self._mask_outputs(m, crop_box, int(dx), int(dy), pos, size, res)
            t["mask_post"] = (time.perf_counter() - t3) * 1e3
            t["refine"] = (t3 - t2) * 1e3

        h, w = self.im_hw
        pos[0] = max(0, min(w, pos[0]))
        pos[1] = max(0, min(h, pos[1]))
        size[0] = max(10, min(w, size[0]))
        size[1] = max(10, min(h, size[1]))
        self.pos, self.sz = pos, size
        res.center_xy = (float(pos[0]), float(pos[1]))
        res.size_wh = (float(size[0]), float(size[1]))
        t["crop"] = (t1 - t0) * 1e3
        t["net"] = (t2 - t1) * 1e3
        t["total"] = (time.perf_counter() - t0) * 1e3
        res.timing_ms = t
        return res

    # ------------------------------------------------------------------ mask

    def _mask_outputs(self, mask: np.ndarray, crop_box, dx: int, dy: int, pos, size, res: SiamMaskResult) -> None:
        p = self.p
        h, w = self.im_hw
        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (dx - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (dy - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, w * s, h * s]
        # crop_back of the original: mask (x_m, y_m) -> frame (a x_m + c, b y_m + d)
        a = (w - 1) / back_box[2]
        b = (h - 1) / back_box[3]
        c = -a * back_box[0]
        d = -b * back_box[1]
        if self.mask_mode == "full":
            ox, oy, ww, hh = 0, 0, w, h
        else:  # only the window covered by the mask (the rest of the original warp is padding)
            ox = int(max(0, math.floor(c) - 1))
            oy = int(max(0, math.floor(d) - 1))
            ww = int(min(w, math.ceil(c + a * (p.out_size - 1)) + 2)) - ox
            hh = int(min(h, math.ceil(d + b * (p.out_size - 1)) + 2)) - oy
        target = None
        if ww > 0 and hh > 0:
            mapping = np.array([[a, 0, c - ox], [0, b, d - oy]], np.float64)
            warped = cv2.warpAffine(mask, mapping, (ww, hh), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
            target = (warped > p.seg_thr).astype(np.uint8)
        contours = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] if target is not None else []
        areas = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(areas) > 100:
            cnt = contours[int(np.argmax(areas))]
            poly = cnt.reshape(-1, 2).astype(np.float64) + (ox, oy)
            res.polygon = cv2.boxPoints(cv2.minAreaRect(poly.astype(np.float32)))
            bx, by, bw, bh = cv2.boundingRect(cnt)
            res.mask_box_xywh = (float(bx + ox), float(by + oy), float(bw), float(bh))
            mo = cv2.moments(cnt)
            if mo["m00"] > 0:
                res.mask_center_xy = (mo["m10"] / mo["m00"] + ox, mo["m01"] / mo["m00"] + oy)
        else:
            x0, y0 = pos[0] - size[0] / 2, pos[1] - size[1] / 2
            res.polygon = np.array([[x0, y0], [x0 + size[0], y0], [x0 + size[0], y0 + size[1]], [x0, y0 + size[1]]])
