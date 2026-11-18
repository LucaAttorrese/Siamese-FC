"""Shared logic of the IDS labelling tools (validation V4): label track, files, pre-label fusion.

Used by ``tools/prelabel_ids.py`` (automatic pre-labels) and ``tools/review_labels.py``
(manual review). Nothing here loads models or opens windows, so it is fully unit-tested.

Coordinates. Label boxes are ``x, y, w, h`` in **edge coordinates** (the image spans
[0, W] x [0, H], pixel i is centred at i + 0.5), the convention of DUT-Anti-UAV, YOLO and
``replay_siamfc_source.py --gt``. The detector and the tracker use pixel-centre
coordinates (pixel 0 centred at 0.0): ``box_from_pixel_centre`` / ``box_to_pixel_centre``
convert (+/- 0.5 px on x and y).

Label file: ``labels/ids/<recording>.csv``, one row per frame::

    frame,present,x,y,w,h,source,score,flag,reviewed

- ``present``: 1 drone in the frame (box set), 0 absent;
- ``source``: where the row comes from (``SOURCES``);
- ``score``: detector score or tracker confidence when known, else empty;
- ``flag``: 1 = the pre-label is uncertain, look at it;
- ``reviewed``: 1 = shown to the user in the review tool.

A JSON sidecar (``<recording>.json``) keeps the provenance: tool version, recording
path, weights hashes, parameters, counts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

LABEL_FIELDS = ("frame", "present", "x", "y", "w", "h", "source", "score", "flag", "reviewed")

# Row sources. det: strong detection (anchor); det_weak: weak detection that agrees with the
# tracker; blend: forward and backward tracking agree; fwd / bwd: one tracking direction only;
# absent: no evidence of the drone; manual / manual_absent / interp / retrack / confirmed: review tool.
SOURCES = ("none", "det", "det_weak", "blend", "fwd", "bwd", "absent", "manual", "manual_absent", "interp", "retrack", "confirmed")
TRACKER_SOURCES = {"fwd", "bwd"}
MANUAL_SOURCES = {"manual", "manual_absent", "interp", "retrack", "confirmed"}

Box = tuple[float, float, float, float]


# ============================================================================ geometry


def box_from_pixel_centre(box_pc: Sequence[float]) -> Box:
    """Pixel-centre (detector, tracker) -> edge coordinates."""
    x, y, w, h = (float(v) for v in box_pc)
    return (x + 0.5, y + 0.5, w, h)


def box_to_pixel_centre(box_edge: Sequence[float]) -> Box:
    x, y, w, h = (float(v) for v in box_edge)
    return (x - 0.5, y - 0.5, w, h)


def box_centre(box: Sequence[float]) -> tuple[float, float]:
    x, y, w, h = (float(v) for v in box)
    return (x + w / 2.0, y + h / 2.0)


def box_side(box: Sequence[float]) -> float:
    return math.sqrt(max(float(box[2]), 0.0) * max(float(box[3]), 0.0))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def centre_inside(box: Sequence[float], frame_wh: tuple[int, int]) -> bool:
    cx, cy = box_centre(box)
    return 0.0 <= cx <= float(frame_wh[0]) and 0.0 <= cy <= float(frame_wh[1])


def interpolate_box(b0: Sequence[float], b1: Sequence[float], t: float) -> Box:
    """Linear interpolation of centre and size, ``t`` in [0, 1]."""
    c0, c1 = box_centre(b0), box_centre(b1)
    w = (1.0 - t) * float(b0[2]) + t * float(b1[2])
    h = (1.0 - t) * float(b0[3]) + t * float(b1[3])
    cx = (1.0 - t) * c0[0] + t * c1[0]
    cy = (1.0 - t) * c0[1] + t * c1[1]
    return (cx - w / 2.0, cy - h / 2.0, w, h)


# ============================================================================ display / detector aid


def auto_gain(gray: np.ndarray, target_mean: float = 80.0, max_gain: float = 3.0, step: int = 8) -> float:
    """Digital gain that brings a dark frame to ``target_mean`` (labelling aid only, never < 1).

    Limited so that the 99.5th percentile stays at or below 250 (no new saturation) and by
    ``max_gain``. Statistics on a subsampled grid. This is NOT used by the runtime.
    """
    sample = np.asarray(gray)[::step, ::step]
    mean = float(sample.mean()) if sample.size else 0.0
    if mean <= 0.0:
        return float(max_gain)
    p995 = float(np.percentile(sample, 99.5))
    g = target_mean / mean
    if p995 > 0.0:
        g = min(g, 250.0 / p995)
    return float(min(max(g, 1.0), max_gain))


def apply_gain(gray: np.ndarray, gain: float) -> np.ndarray:
    if gain == 1.0:
        return gray
    lut = np.clip(np.arange(256, dtype=np.float32) * float(gain) + 0.5, 0, 255).astype(np.uint8)
    return lut[np.asarray(gray, np.uint8)]


# ============================================================================ label track


@dataclass
class LabelTrack:
    """Per-frame labels of one recording, with an undo stack."""

    n: int
    present: np.ndarray = field(init=False)  # int8: 1 present, 0 absent
    box: np.ndarray = field(init=False)  # (n, 4) float, NaN when absent
    source: list = field(init=False)
    score: np.ndarray = field(init=False)
    flag: np.ndarray = field(init=False)
    reviewed: np.ndarray = field(init=False)
    _undo: list = field(init=False, default_factory=list, repr=False)
    dirty: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("a label track needs at least one frame")
        self.present = np.zeros(self.n, np.int8)
        self.box = np.full((self.n, 4), np.nan)
        self.source = ["none"] * self.n
        self.score = np.full(self.n, np.nan)
        self.flag = np.zeros(self.n, np.int8)
        self.reviewed = np.zeros(self.n, np.int8)

    # -------------------------------------------------------------- edits (with undo)

    def _push(self, f0: int, f1: int) -> None:
        sl = slice(f0, f1 + 1)
        self._undo.append((f0, f1, self.present[sl].copy(), self.box[sl].copy(), list(self.source[sl]),
                           self.score[sl].copy(), self.flag[sl].copy(), self.reviewed[sl].copy()))
        if len(self._undo) > 200:
            self._undo.pop(0)
        self.dirty = True

    def undo(self) -> Optional[tuple[int, int]]:
        """Revert the last edit; returns its frame range, or None if there is nothing to undo."""
        if not self._undo:
            return None
        f0, f1, present, box, source, score, flag, reviewed = self._undo.pop()
        sl = slice(f0, f1 + 1)
        self.present[sl], self.box[sl], self.score[sl], self.flag[sl], self.reviewed[sl] = present, box, score, flag, reviewed
        self.source[sl] = source
        self.dirty = True
        return (f0, f1)

    def _check(self, f: int) -> int:
        f = int(f)
        if not 0 <= f < self.n:
            raise IndexError(f"frame {f} outside 0..{self.n - 1}")
        return f

    def _set_box(self, f: int, box: Sequence[float], source: str, score: float, flag: int) -> None:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        b = [float(v) for v in box]
        if not all(math.isfinite(v) for v in b) or b[2] <= 0.0 or b[3] <= 0.0:
            raise ValueError(f"invalid box {box!r}")
        self.present[f] = 1
        self.box[f] = b
        self.source[f] = source
        self.score[f] = float(score) if score is not None else np.nan
        self.flag[f] = int(flag)

    def _set_absent(self, f: int, source: str, flag: int) -> None:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        self.present[f] = 0
        self.box[f] = np.nan
        self.source[f] = source
        self.score[f] = np.nan
        self.flag[f] = int(flag)

    def set_box(self, f: int, box: Sequence[float], source: str, score: Optional[float] = None, flag: int = 0, undoable: bool = True) -> None:
        f = self._check(f)
        if undoable:
            self._push(f, f)
        self._set_box(f, box, source, score, flag)

    def set_absent(self, f: int, source: str = "absent", flag: int = 0, undoable: bool = True) -> None:
        f = self._check(f)
        if undoable:
            self._push(f, f)
        self._set_absent(f, source, flag)

    def set_range_absent(self, f0: int, f1: int, source: str = "manual_absent") -> None:
        f0, f1 = sorted((self._check(f0), self._check(f1)))
        self._push(f0, f1)
        for f in range(f0, f1 + 1):
            self._set_absent(f, source, 0)
            self.reviewed[f] = 1

    def interpolate(self, f0: int, f1: int) -> int:
        """Linear boxes between two frames that both have a box; returns the frames written."""
        f0, f1 = sorted((self._check(f0), self._check(f1)))
        if f1 - f0 < 2:
            return 0
        if not (self.present[f0] and self.present[f1]):
            raise ValueError("both ends of an interpolation need a box")
        self._push(f0 + 1, f1 - 1)
        b0, b1 = self.box[f0].copy(), self.box[f1].copy()
        for f in range(f0 + 1, f1):
            self._set_box(f, interpolate_box(b0, b1, (f - f0) / (f1 - f0)), "interp", None, 0)
            self.reviewed[f] = 1
        return f1 - f0 - 1

    def confirm(self, f0: int, f1: int) -> None:
        """The user checked the range: clear the flags, mark reviewed (boxes unchanged)."""
        f0, f1 = sorted((self._check(f0), self._check(f1)))
        self._push(f0, f1)
        self.flag[f0:f1 + 1] = 0
        self.reviewed[f0:f1 + 1] = 1

    def write_run(self, frames: Sequence[int], boxes: Sequence[Optional[Sequence[float]]], source: str, scores: Sequence[Optional[float]], flags: Sequence[int]) -> None:
        """Several frames in one undoable edit (re-tracking). ``None`` box = absent."""
        if not frames:
            return
        idx = [self._check(f) for f in frames]
        self._push(min(idx), max(idx))
        for f, b, s, fl in zip(idx, boxes, scores, flags):
            if b is None:
                self._set_absent(f, "absent", int(fl))
            else:
                self._set_box(f, b, source, s, int(fl))

    def mark_reviewed(self, f0: int, f1: int) -> None:
        """Not an edit (no undo): frames shown to the user."""
        f0, f1 = sorted((self._check(f0), self._check(f1)))
        if not self.reviewed[f0:f1 + 1].all():
            self.reviewed[f0:f1 + 1] = 1
            self.dirty = True

    # -------------------------------------------------------------- queries

    def next_flagged(self, f: int, forward: bool = True, unreviewed_only: bool = False) -> Optional[int]:
        """First frame of the next (previous) flagged segment after (before) ``f``."""
        mask = self.flag.astype(bool)
        if unreviewed_only:
            mask &= ~self.reviewed.astype(bool)
        if forward:
            i = int(f) + 1
            while i < self.n and mask[i] and mask[i - 1]:
                i += 1  # leave the current segment
            hits = np.flatnonzero(mask[i:])
            return int(hits[0] + i) if hits.size else None
        i = int(f) - 1
        hits = np.flatnonzero(mask[: max(i + 1, 0)])
        if not hits.size:
            return None
        j = int(hits[-1])
        while j > 0 and mask[j - 1]:
            j -= 1  # start of that segment
        return j

    def counts(self) -> dict:
        src = {s: 0 for s in SOURCES}
        for s in self.source:
            src[s] += 1
        return {
            "frames": self.n,
            "present": int(self.present.sum()),
            "absent": int(self.n - self.present.sum()),
            "flagged": int(self.flag.sum()),
            "reviewed": int(self.reviewed.sum()),
            "flagged_not_reviewed": int((self.flag.astype(bool) & ~self.reviewed.astype(bool)).sum()),
            "sources": {k: v for k, v in src.items() if v},
        }


# ============================================================================ files


def _fmt(v: float) -> str:
    return "" if not math.isfinite(float(v)) else f"{float(v):.2f}"


def save_labels(track: LabelTrack, path: Path, meta: Optional[dict] = None) -> None:
    """Write the CSV (atomically) and, if given, the JSON sidecar with up-to-date counts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LABEL_FIELDS)
        for f in range(track.n):
            if track.present[f]:
                x, y, bw, bh = (_fmt(v) for v in track.box[f])
            else:
                x = y = bw = bh = ""
            score = "" if not math.isfinite(track.score[f]) else f"{track.score[f]:.3f}"
            w.writerow((f, int(track.present[f]), x, y, bw, bh, track.source[f], score, int(track.flag[f]), int(track.reviewed[f])))
    tmp.replace(path)
    if meta is not None:
        meta = dict(meta)
        meta["counts"] = track.counts()
        side = path.with_suffix(".json")
        tmp = side.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(side)
    track.dirty = False


def load_labels(path: Path) -> LabelTrack:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or tuple(rows[0].keys())[: len(LABEL_FIELDS)] != LABEL_FIELDS:
        raise ValueError(f"{path}: not a label file (expected columns {','.join(LABEL_FIELDS)})")
    track = LabelTrack(len(rows))
    for i, r in enumerate(rows):
        if int(r["frame"]) != i:
            raise ValueError(f"{path}: row {i} has frame {r['frame']}")
        flag = int(r["flag"] or 0)
        if int(r["present"]):
            track._set_box(i, (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])), r["source"],
                           float(r["score"]) if r["score"] else None, flag)
        else:
            track._set_absent(i, r["source"], flag)
        track.reviewed[i] = int(r["reviewed"] or 0)
    track.dirty = False
    return track


def load_meta(path: Path) -> dict:
    side = Path(path).with_suffix(".json")
    return json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}


def has_user_work(track: LabelTrack) -> bool:
    return bool(track.reviewed.any()) or any(s in MANUAL_SOURCES for s in track.source)


# ============================================================================ detections cache


def save_detections(path: Path, dets: Sequence[Sequence[tuple[Box, float]]], gains: Sequence[float]) -> None:
    """Raw detections per frame (edge boxes, scores) and the digital gain used."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("frame", "gain", "x", "y", "w", "h", "score"))
        for f, (frame_dets, g) in enumerate(zip(dets, gains)):
            if not frame_dets:
                w.writerow((f, f"{g:.3f}", "", "", "", "", ""))
            for (x, y, bw, bh), s in frame_dets:
                w.writerow((f, f"{g:.3f}", f"{x:.2f}", f"{y:.2f}", f"{bw:.2f}", f"{bh:.2f}", f"{s:.4f}"))


def load_detections(path: Path, n: int) -> tuple[list[list[tuple[Box, float]]], list[float]]:
    dets: list[list[tuple[Box, float]]] = [[] for _ in range(n)]
    gains = [1.0] * n
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            if f >= n:
                raise ValueError(f"{path}: frame {f} beyond the {n} frames of the recording")
            gains[f] = float(r["gain"])
            if r["score"]:
                dets[f].append(((float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])), float(r["score"])))
    for d in dets:
        d.sort(key=lambda t: t[1], reverse=True)
    return dets, gains


# ============================================================================ pre-label fusion


@dataclass(frozen=True)
class PrelabelParams:
    strong_score: float = 0.5  # a detection at or above this score can be an anchor
    weak_score: float = 0.15  # weaker detections only confirm the tracker
    anchor_window: int = 3  # an anchor needs another strong detection within +/- this many frames ...
    anchor_max_shift: float = 0.5  # ... whose centre is within this fraction of the box side
    agree_iou: float = 0.5  # forward / backward / weak detection agreement
    trust_frames: int = 63  # one-direction tracking further than this from its anchor is flagged (~1 s)
    max_track_frames: int = 626  # tracking runs stop after this many frames (~10 s at 62.6 fps)
    lost_conf: float = 0.25  # tracker frames below this confidence are invalid (as at runtime)
    lost_frames: int = 5  # invalid frames in a row that end a tracking run
    # Review flags added after the fusion (flag_for_review). Recordings B3/B4 (2026-11-17): when
    # the drone leaves the frame SiamFC keeps a box on the border or on a wall stain with a good
    # confidence, and both tracking directions can agree on it.
    edge_margin_px: float = 8.0  # tracker / det_weak boxes this close to a frame edge are flagged
    support_window: int = 16  # tracker boxes with no detection (any cached score) within +/- this many frames ...
    support_dist: float = 0.75  # ... whose centre is within this fraction of the box side are flagged

    def validate(self) -> None:
        if not 0.0 < self.weak_score <= self.strong_score < 1.0:
            raise ValueError("need 0 < weak_score <= strong_score < 1")
        for name in ("anchor_window", "trust_frames", "max_track_frames", "lost_frames"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.edge_margin_px < 0 or self.support_window < 0 or self.support_dist <= 0:
            raise ValueError("need edge_margin_px >= 0, support_window >= 0, support_dist > 0")


def best_detection(frame_dets: Sequence[tuple[Box, float]], min_score: float, near: Optional[Box] = None) -> Optional[tuple[Box, float]]:
    """Highest-scoring detection >= ``min_score``; with ``near``, the strong one closest to it."""
    cand = [d for d in frame_dets if d[1] >= min_score]
    if not cand:
        return None
    if near is None:
        return max(cand, key=lambda d: d[1])
    nc = box_centre(near)
    return min(cand, key=lambda d: math.hypot(box_centre(d[0])[0] - nc[0], box_centre(d[0])[1] - nc[1]))


def select_anchors(dets: Sequence[Sequence[tuple[Box, float]]], p: PrelabelParams) -> dict[int, tuple[Box, float]]:
    """Frames with a strong detection confirmed by another strong detection nearby in time."""
    n = len(dets)
    strong = [best_detection(d, p.strong_score) for d in dets]
    anchors: dict[int, tuple[Box, float]] = {}
    for f, d in enumerate(strong):
        if d is None:
            continue
        c, side = box_centre(d[0]), box_side(d[0])
        for g in range(max(0, f - p.anchor_window), min(n, f + p.anchor_window + 1)):
            if g == f or strong[g] is None:
                continue
            cg = box_centre(strong[g][0])
            if math.hypot(cg[0] - c[0], cg[1] - c[1]) <= p.anchor_max_shift * side:
                anchors[f] = d
                break
    return anchors


def gap_ranges(anchor_frames: Iterable[int], n: int) -> list[tuple[int, int, Optional[int], Optional[int]]]:
    """Maximal runs of non-anchor frames: (first, last, left anchor or None, right anchor or None)."""
    a = sorted(set(int(f) for f in anchor_frames))
    out = []
    prev = None
    for f in a + [n]:
        start = 0 if prev is None else prev + 1
        if f - 1 >= start:
            out.append((start, f - 1, prev, f if f < n else None))
        prev = f
    return out


@dataclass(frozen=True)
class TrackStep:
    box: Optional[Box]  # edge coordinates; None when the frame is not valid or the run has ended
    conf: float


class TrackerConfidence:
    """Tracker confidence as at runtime (``siamfc_source.TrackingSession._confidence``):
    sqrt(clip(peak / peak_ref) * clip(apce / apce_ref)), references = median of the first
    ``ref_frames`` updates after initialisation."""

    def __init__(self, ref_frames: int = 5) -> None:
        self.ref_frames = int(ref_frames)
        self._peaks: list[float] = []
        self._apces: list[float] = []

    def __call__(self, peak: float, apce: float) -> float:
        if not (math.isfinite(peak) and math.isfinite(apce)):
            return float("nan")
        if len(self._peaks) < self.ref_frames:
            self._peaks.append(float(peak))
            self._apces.append(float(apce))
        peak_ref = float(np.median(self._peaks))
        apce_ref = float(np.median(self._apces))
        c_peak = min(max(peak / peak_ref, 0.0), 1.0) if peak_ref > 1e-9 else (1.0 if peak > 0.0 else 0.0)
        c_apce = min(max(apce / apce_ref, 0.0), 1.0) if apce_ref > 1e-9 else 0.0
        return math.sqrt(c_peak * c_apce)


def track_run(init: Box, steps: Iterable[tuple[Box, float, float]], frame_wh: tuple[int, int], p: PrelabelParams) -> list[TrackStep]:
    """Validity of a tracking run from per-frame tracker outputs (edge box, peak, apce).

    A frame is valid when the box centre is inside the frame and the confidence is at least
    ``lost_conf``. The run ends (all later frames None) when the centre leaves the frame or
    after ``lost_frames`` invalid frames in a row, as the runtime goes to "lost".
    ``steps`` is consumed lazily, so the caller can stop tracking when the run ends.
    """
    del init  # the tracker was initialised on it by the caller; kept for readability of calls
    conf_fn = TrackerConfidence()
    out: list[TrackStep] = []
    weak = 0
    for box, peak, apce in steps:
        conf = conf_fn(peak, apce)
        inside = centre_inside(box, frame_wh)
        valid = inside and math.isfinite(conf) and conf >= p.lost_conf
        weak = 0 if valid else weak + 1
        out.append(TrackStep(box if valid else None, conf))
        if not inside or weak >= p.lost_frames:
            break
    return out


def fuse_gap(first: int, last: int, left: Optional[int], right: Optional[int],
             fwd: Sequence[TrackStep], bwd: Sequence[TrackStep],
             dets: Sequence[Sequence[tuple[Box, float]]], p: PrelabelParams) -> list[tuple[Optional[Box], str, Optional[float], int]]:
    """Label of every frame of a gap from the two tracking runs and the weak detections.

    ``fwd[i]`` is frame ``first + i`` tracked from the left anchor, ``bwd[i]`` the same frame
    tracked from the right anchor (empty sequences when that anchor does not exist).
    Returns (box or None, source, score, flag) per frame.
    """
    out = []
    for i, f in enumerate(range(first, last + 1)):
        fs = fwd[i] if i < len(fwd) else TrackStep(None, float("nan"))
        bs = bwd[i] if i < len(bwd) else TrackStep(None, float("nan"))
        d_left = f - left if left is not None else None
        d_right = right - f if right is not None else None
        if fs.box is not None and bs.box is not None:
            if iou(fs.box, bs.box) >= p.agree_iou:
                t = d_left / (d_left + d_right)  # both anchors exist here
                box, source, score, flag = interpolate_box(fs.box, bs.box, t), "blend", min(fs.conf, bs.conf), 0
            else:
                near_fwd = d_left <= d_right
                pick = fs if near_fwd else bs
                box, source, score, flag = pick.box, ("fwd" if near_fwd else "bwd"), pick.conf, 1
        elif fs.box is not None:
            box, source, score, flag = fs.box, "fwd", fs.conf, int(d_left > p.trust_frames)
        elif bs.box is not None:
            box, source, score, flag = bs.box, "bwd", bs.conf, int(d_right > p.trust_frames)
        else:
            weak = best_detection(dets[f], p.weak_score)
            # no tracking evidence: absent; flagged when the drone is probably there (between
            # two anchors) or when the detector sees something
            out.append((None, "absent", None, int(left is not None and right is not None or weak is not None)))
            continue
        weak = best_detection(dets[f], p.weak_score, near=box)
        if weak is not None and iou(weak[0], box) >= p.agree_iou:
            out.append((weak[0], "det_weak", weak[1], 0))  # detector and tracker agree: trusted
        else:
            out.append((box, source, score, flag))
    return out


def border_distance(box: Sequence[float], frame_wh: tuple[int, int]) -> float:
    """Smallest distance in px from the box to a frame edge, negative when the box crosses it."""
    x, y, w, h = (float(v) for v in box)
    return min(x, y, float(frame_wh[0]) - (x + w), float(frame_wh[1]) - (y + h))


def detection_support(box: Sequence[float], dets: Sequence[Sequence[tuple[Box, float]]], f: int, window: int, dist: float) -> bool:
    """True when a detection of frames f-window..f+window has its centre within dist * side of the box centre."""
    cx, cy = box_centre(box)
    limit = dist * box_side(box)
    for g in range(max(0, f - window), min(len(dets), f + window + 1)):
        for dbox, _ in dets[g]:
            dx, dy = box_centre(dbox)
            if math.hypot(dx - cx, dy - cy) < limit:
                return True
    return False


def flag_for_review(track: LabelTrack, dets: Sequence[Sequence[tuple[Box, float]]], frame_wh: tuple[int, int], p: PrelabelParams) -> int:
    """Flags pre-labels that the fusion trusted but that match a known tracker failure.

    - Tracker and ``det_weak`` boxes near a frame edge: the drone may be partly or fully out
      of the frame, and SiamFC can stay on the border or on a stain after it has left.
    - Tracker boxes with no detection nearby in time: nothing but the tracker supports them.
    Strong anchors (``det``) are left alone. Returns the number of frames newly flagged.
    """
    tracked = TRACKER_SOURCES | {"blend"}
    added = 0
    for f in range(track.n):
        src = track.source[f]
        if not track.present[f] or track.flag[f] or src not in tracked | {"det_weak"}:
            continue
        box = track.box[f]
        near_edge = border_distance(box, frame_wh) < p.edge_margin_px
        unsupported = src in tracked and not detection_support(box, dets, f, p.support_window, p.support_dist)
        if near_edge or unsupported:
            track.flag[f] = 1
            added += 1
    return added
