"""Drone detector used to initialise and re-acquire the SiamFC tracker (PLAN Phase 2, step 3).

Two parts:
- ``DroneDetector`` (Protocol) and ``UltralyticsDroneDetector``: run a YOLO model on a
  Mono8 frame, optionally restricted to a region of interest, and return detections in
  full-frame pixel coordinates. The protocol keeps the camera adapter independent of
  Ultralytics, so tests can use a fake detector;
- ``DetectionConfirmer``: requires several consistent detections in a row before a
  target is confirmed, so a single false alarm cannot initialise the tracker.

Coordinates: ``bbox_xywh = (x, y, w, h)`` with (x, y) the top-left corner, in pixels of
the full frame, OpenCV pixel-centre convention (pixel (0, 0) is centred at (0.0, 0.0)),
the same as ``SiamFCTracker`` and ``cv2.undistortPoints``: a box covering pixel columns
10..19 has x = 9.5 and w = 10. Ultralytics (and YOLO labels) use edge coordinates
instead, where the image spans [0, W] and pixel i is centred at i + 0.5; the conversion
(-0.5 px) happens in ``detections_from_xyxy``.

The default inference is on the whole frame resized to ``imgsz``: with the expected
target (DJI Mavic Mini at 20-22 m, ~130-340 px in a 2592 px wide frame, ~65-170 px at
imgsz 1280) tiling is not needed (PLAN, Phase 2, step 3). A frame (or ROI) larger than
``imgsz`` is downscaled here, on the single Mono8 channel and with ``INTER_AREA`` as
Ultralytics does for validation images, before being replicated to 3 channels:
replicating the full 2592x1944 frame first took ~31 ms of a ~55 ms detection on the
training PC, against ~20 ms in total this way.

``torch`` / ``ultralytics`` are imported only when ``UltralyticsDroneDetector`` is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox_xywh: tuple[float, float, float, float]  # top-left x, y, width, height (full-frame px)
    score: float  # detector confidence in [0, 1]

    @property
    def center_xy(self) -> tuple[float, float]:
        x, y, w, h = self.bbox_xywh
        return (x + w / 2.0, y + h / 2.0)

    @property
    def size_wh(self) -> tuple[float, float]:
        return (self.bbox_xywh[2], self.bbox_xywh[3])

    @property
    def side(self) -> float:
        """sqrt(w * h): the box side used for size comparisons."""
        return math.sqrt(max(self.bbox_xywh[2], 0.0) * max(self.bbox_xywh[3], 0.0))


class DroneDetector(Protocol):
    def detect(self, gray: np.ndarray, roi_xywh: Optional[Sequence[int]] = None) -> list[Detection]:
        """Detections sorted by decreasing score, in full-frame coordinates.

        ``roi_xywh`` restricts the search to that region (clipped to the frame); an
        empty region gives no detections.
        """
        ...


@dataclass(frozen=True)
class DetectorConfig:
    weights_path: str
    weights_sha256: Optional[str] = None  # when set, the file is refused if its SHA256 differs
    conf_min: float = 0.15
    imgsz: int = 1280
    nms_iou: float = 0.7
    max_det: int = 10
    device: Optional[str] = None  # None = Ultralytics default (first CUDA GPU if available)
    fp16: bool = True  # fp16 inference on CUDA; ignored on CPU
    # Plausible box size. The model produces occasional huge false positives over buildings
    # (more than half the frame, replay of DUT videos 2026-11-10); the expected target is
    # ~5-13 % of the frame's larger side (Mavic Mini at 20-22 m).
    max_box_frac: float = 0.3  # max box side (larger of w, h) as a fraction of the frame's larger side
    min_box_px: float = 4.0  # min box side (smaller of w, h) in frame px

    def validate(self) -> None:
        if not str(self.weights_path).strip():
            raise ValueError("weights_path must be set")
        if not 0.0 < float(self.conf_min) < 1.0:
            raise ValueError("conf_min must be in (0, 1)")
        if not 0.0 < float(self.nms_iou) <= 1.0:
            raise ValueError("nms_iou must be in (0, 1]")
        if int(self.imgsz) < 32 or int(self.imgsz) % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32 and >= 32")
        if int(self.max_det) < 1:
            raise ValueError("max_det must be >= 1")
        if self.weights_sha256 is not None and len(str(self.weights_sha256).strip()) != 64:
            raise ValueError("weights_sha256 must be a 64-character hex digest")
        if not 0.0 < float(self.max_box_frac) <= 1.0:
            raise ValueError("max_box_frac must be in (0, 1]")
        if float(self.min_box_px) < 0.0:
            raise ValueError("min_box_px must be >= 0")


def filter_box_size(dets: Sequence["Detection"], frame_hw: tuple[int, int], max_box_frac: float, min_box_px: float) -> list["Detection"]:
    """Drop detections whose larger side exceeds ``max_box_frac`` of the frame's larger side,
    or whose smaller side is below ``min_box_px``."""
    max_side = float(max_box_frac) * max(int(frame_hw[0]), int(frame_hw[1]))
    return [d for d in dets if max(d.size_wh) <= max_side and min(d.size_wh) >= float(min_box_px)]


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_roi(roi_xywh: Sequence[float], frame_hw: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
    """Integer ROI clipped to the frame, or None when nothing of it is inside."""
    h, w = int(frame_hw[0]), int(frame_hw[1])
    x, y, rw, rh = (float(v) for v in roi_xywh)
    if not all(math.isfinite(v) for v in (x, y, rw, rh)):
        return None
    x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    x1, y1 = min(w, int(math.ceil(x + rw))), min(h, int(math.ceil(y + rh)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def detections_from_xyxy(
    xyxy_edges: np.ndarray,
    scores: np.ndarray,
    offset_xy: tuple[float, float] = (0.0, 0.0),
    scale_xy: tuple[float, float] = (1.0, 1.0),
    conf_min: float = 0.0,
) -> list[Detection]:
    """Detections sorted by decreasing score, in full-frame pixel-centre coordinates.

    ``xyxy_edges`` are Ultralytics boxes in edge coordinates of the image that was fed to
    the model; that image is the frame region starting at ``offset_xy`` (integer pixel
    indices), resized by ``scale_xy`` = (fed width / region width, fed height / region height).
    """
    out = []
    ox, oy = float(offset_xy[0]), float(offset_xy[1])
    sx, sy = float(scale_xy[0]), float(scale_xy[1])
    for (x0, y0, x1, y1), s in zip(np.asarray(xyxy_edges, float).reshape(-1, 4), np.asarray(scores, float).reshape(-1)):
        if not (math.isfinite(s) and s >= conf_min) or x1 <= x0 or y1 <= y0:
            continue
        fx0, fy0 = x0 / sx + ox, y0 / sy + oy  # frame edge coordinates
        fw, fh = (x1 - x0) / sx, (y1 - y0) / sy
        out.append(Detection((fx0 - 0.5, fy0 - 0.5, fw, fh), float(s)))  # edges -> pixel centres
    out.sort(key=lambda d: d.score, reverse=True)
    return out


class UltralyticsDroneDetector:
    """``DroneDetector`` backed by an Ultralytics YOLO model trained on grayscale images.

    The model was trained on grayscale JPEGs read as 3 identical channels, so a Mono8
    frame is replicated to 3 channels before inference. Not thread-safe: use one
    instance per thread (the camera adapter runs it in its single detector thread).
    """

    def __init__(self, cfg: DetectorConfig, model: Any = None) -> None:
        cfg.validate()
        self.cfg = cfg
        path = Path(cfg.weights_path)
        if model is None:
            if not path.is_file():
                raise FileNotFoundError(f"detector weights not found: {path}")
            if cfg.weights_sha256 is not None:
                got = sha256_file(path)
                if got.lower() != str(cfg.weights_sha256).strip().lower():
                    raise ValueError(f"detector weights SHA256 mismatch for {path}: {got} != {cfg.weights_sha256}")
            from ultralytics import YOLO  # noqa: PLC0415

            model = YOLO(str(path))
        self._model = model
        self._quantize = 16 if cfg.fp16 and self._is_cuda() else None

    def _is_cuda(self) -> bool:
        device = self.cfg.device
        if device is not None:
            return str(device).lower() not in {"cpu", "mps"}
        try:
            import torch  # noqa: PLC0415

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def warmup(self, frame_hw: tuple[int, int] = (1944, 2592)) -> None:
        """One inference on a blank frame, so the first real call does not pay the CUDA start-up."""
        self.detect(np.zeros(frame_hw, np.uint8))

    def detect(self, gray: np.ndarray, roi_xywh: Optional[Sequence[int]] = None) -> list[Detection]:
        img = np.asarray(gray)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        if img.ndim != 2:
            raise ValueError(f"expected a single-channel frame, got shape {img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"expected a uint8 frame, got {img.dtype}")
        offset = (0, 0)
        frame_hw = img.shape
        if roi_xywh is not None:
            roi = clip_roi(roi_xywh, img.shape)
            if roi is None:
                return []
            x, y, w, h = roi
            img = img[y:y + h, x:x + w]
            offset = (x, y)
        h, w = img.shape
        scale = float(self.cfg.imgsz) / max(h, w)
        scale_xy = (1.0, 1.0)
        if scale < 1.0:  # downscale the single channel first (see the module docstring)
            import cv2  # noqa: PLC0415

            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            scale_xy = (nw / w, nh / h)
        bgr = np.ascontiguousarray(np.repeat(img[:, :, None], 3, axis=2))
        results = self._model.predict(
            bgr,
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf_min),
            iou=float(self.cfg.nms_iou),
            max_det=int(self.cfg.max_det),
            device=self.cfg.device,
            quantize=self._quantize,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        dets = detections_from_xyxy(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            offset_xy=offset,
            scale_xy=scale_xy,
            conf_min=float(self.cfg.conf_min),
        )
        return filter_box_size(dets, frame_hw, self.cfg.max_box_frac, self.cfg.min_box_px)


@dataclass(frozen=True)
class ConfirmerConfig:
    frames_required: int = 3  # consecutive consistent detections needed to confirm a target
    max_center_shift: float = 1.0  # max centre displacement between two detections, in units of the previous box side
    max_size_ratio: float = 1.5  # max ratio between the sides of two consecutive detections
    max_gap_s: float = 0.5  # detections further apart in time do not count as consecutive

    def validate(self) -> None:
        if int(self.frames_required) < 1:
            raise ValueError("frames_required must be >= 1")
        if not float(self.max_center_shift) > 0.0:
            raise ValueError("max_center_shift must be > 0")
        if not float(self.max_size_ratio) >= 1.0:
            raise ValueError("max_size_ratio must be >= 1")
        if not float(self.max_gap_s) > 0.0:
            raise ValueError("max_gap_s must be > 0")


class DetectionConfirmer:
    """Confirms a target only after ``frames_required`` consistent detections in a row.

    Call ``update`` once per completed detection run (not per camera frame: the detector
    runs asynchronously at a lower rate). The best-scoring detection of each run is
    compared with the previous candidate: it continues the streak when its centre moved
    by at most ``max_center_shift`` box sides, its size changed by at most
    ``max_size_ratio`` and less than ``max_gap_s`` elapsed; otherwise it starts a new
    streak. A run without detections resets the streak. Displacement is measured relative
    to the box size because the detector rate is low: a drone moving quickly can shift by
    more than its own size between two runs, which an IoU test would reject.
    """

    def __init__(self, cfg: ConfirmerConfig = ConfirmerConfig()) -> None:
        cfg.validate()
        self.cfg = cfg
        self._candidate: Optional[Detection] = None
        self._candidate_ns: Optional[int] = None
        self._count = 0

    @property
    def streak(self) -> int:
        return self._count

    def reset(self) -> None:
        self._candidate, self._candidate_ns, self._count = None, None, 0

    def consistent(self, prev: Detection, cur: Detection) -> bool:
        side = max(prev.side, 1e-6)
        (px, py), (cx, cy) = prev.center_xy, cur.center_xy
        if math.hypot(cx - px, cy - py) > float(self.cfg.max_center_shift) * side:
            return False
        ratio = max(cur.side, 1e-6) / side
        return 1.0 / float(self.cfg.max_size_ratio) <= ratio <= float(self.cfg.max_size_ratio)

    def update(self, detections: Sequence[Detection], now_ns: int) -> Optional[Detection]:
        """Return the confirmed detection (the latest one of the streak) or None."""
        if not detections:
            self.reset()
            return None
        best = max(detections, key=lambda d: d.score)
        # A non-monotonic timestamp (dt < 0) never continues a streak.
        fresh = self._candidate_ns is not None and 0 <= (int(now_ns) - self._candidate_ns) <= self.cfg.max_gap_s * 1e9
        if self._candidate is not None and fresh and self.consistent(self._candidate, best):
            self._count += 1
        else:
            self._count = 1
        self._candidate, self._candidate_ns = best, int(now_ns)
        return best if self._count >= int(self.cfg.frames_required) else None
