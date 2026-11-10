"""IDS camera correction source based on the SiamFC tracker and the drone detector (PLAN Phase 3).

``IdsSiamFCCorrectionSourceAdapter`` implements ``CorrectionSourcePort`` (``mode = "ids_siamfc"``)
and replaces the ArUco/PnP measurement with visual tracking of the drone itself. Downstream
nothing changes: it produces the same ``CorrectionCandidate`` (pointing error in rad, laser
frame) consumed by ``CameraApplicationService``, PBVS and the motor dispatcher.

Structure, each part testable on its own:
- ``pixel_to_laser_angles``: target pixel -> undistort -> point at a fixed camera depth Z ->
  laser frame -> (d_az, d_po). There is no range measurement: Z is a configured constant
  (``target_range_m``), and the parallax error this causes is |t_lc| * |1/Z_true - 1/Z|;
- ``SiamFCTrackingSession``: the tracking state machine, without threads or camera. States:
  ``searching`` (no target; full-frame detection, a target is confirmed by
  ``DetectionConfirmer`` and initialises SiamFC), ``tracking`` (SiamFC on every frame; the
  detector periodically verifies the tracker: on the full frame, keeping only detections
  near the tracked box and of similar size; ``reinit_confirm`` consistent disagreeing
  verifications re-initialise the tracker) and ``lost`` (confidence low for ``lost_frames``
  frames, the target left the frame, or verification found no candidate
  ``verify_miss_limit`` times in a row; the detector searches the full frame, and after
  ``max_lost_s`` the tracker is dropped -> ``searching``). SiamFC keeps running in
  ``lost`` and may resume by itself only when the loss was a confidence drop: after a
  detector disagreement or with the target out of frame its confidence says nothing about
  the drone, and only a confirmed detection can start tracking again. Detections below
  ``acquire_min_score`` never start a track;

Replay of DUT-Anti-UAV video15 (2026-11-10): with the drone out of frame, a static false
positive (score 0.49) was confirmed and tracked with confidence 1.0; when the drone came
back the verification correctly declared the track lost, but the tracker "recovered" on
the same object before the detector could confirm the drone. Hence the non-recoverable
loss and ``acquire_min_score``.

Why verification is not run on a ROI crop (replay of DUT-Anti-UAV video01, 2026-11-10):
the detector was trained on whole frames downscaled to imgsz, and a ~300 px ROI upscaled
to 1280 produced large false positives (scores up to 0.7). Re-initialising on any of them
grew the tracked box, hence the next ROI, until the tracker followed the background with
high confidence. Hence full-frame verification, the size check and ``reinit_confirm``.
- ``IdsSiamFCCorrectionSourceAdapter``: IDS latest-frame producer, SiamFC on the
  ``camera-api-worker`` thread (inside ``read_candidate``), YOLO only on the single
  ``drone-detector`` thread, results handed over through a ``Future`` as for the ArUco
  detection jobs. Calibration, gating order, stale-sample handling and the debug snapshot
  follow ``MultiArucoCorrectionSourceAdapter``.

Delayed detections: a detection result refers to the frame it ran on, a few frames old by
the time it is consumed. The tracker is (re)initialised on that frame and then updated on
the current one, so the delay only adds one larger SiamFC step.

Confidence (PLAN Phase 3; proposal P4 is not applied): geometric mean of
``peak / peak_ref`` and ``apce / apce_ref``, each clipped to [0, 1], where the references
are the medians over the first ``ref_frames`` updates after each (re)initialisation. The
first update alone is not used as reference because its peak is inflated (the template
was just taken on an almost identical frame). The raw peak can be negative away from the
target; the clipping maps that to 0.

``torch`` and ``ultralytics`` are imported only in ``start()``.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import statistics
import time
from threading import Lock
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from .camera import (
    CAMERA_CUDA_VERSION,
    CorrectionCandidate,
    GatingConfig,
    _IdsLatestFrameProducer,
    _angles_from_point_laser,
    _clone_debug_payload,
    _import_cv2_or_raise,
    _laser_axis_point_in_camera,
    _load_vs_calibration_for_camera,
    _period_ns,
    _point_c_to_laser,
    _project_camera_point_to_pixel,
    _scale_intrinsics_to_runtime,
    load_fast_tracking_config,
)
from .siamfc.detector import (
    ConfirmerConfig,
    Detection,
    DetectionConfirmer,
    DetectorConfig,
    DroneDetector,
    sha256_file,
)

STATE_SEARCHING = "searching"
STATE_TRACKING = "tracking"
STATE_LOST = "lost"

REASON_SEARCHING = "siamfc_searching"
REASON_LOST = "siamfc_lost"
REASON_TRACKER_ERROR = "siamfc_tracker_error"


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class SessionConfig:
    lost_conf: float = 0.25  # a tracking frame below this confidence counts as weak
    lost_frames: int = 5  # weak frames in a row: tracking -> lost; good frames in a row: lost -> tracking
    max_lost_s: float = 1.0  # time in "lost" before the tracker is dropped (-> searching)
    ref_frames: int = 5  # updates after (re)initialisation used for the peak/APCE references
    missing_hz: float = 10.0  # full-frame detection rate while searching or lost
    acquire_min_score: float = 0.5  # detections below this score never start a track (to be tuned in V4)
    # Verification rate while tracking (0 disables verification). With verify_miss_limit it bounds how long
    # a tracker that slid onto the background stays "reliable": ~verify_miss_limit / redetect_hz. DUT replay
    # (2026-11-10): 2 Hz x 3 misses -> 1.5 s wrong-target segments; 5 Hz x 2 -> ~0.4 s, and wrong-target
    # reliable frames went 76 -> 20 (video01) and 143 -> 12 (video10), with no change in read_candidate time.
    redetect_hz: float = 5.0
    verify_radius: float = 1.5  # verification considers detections whose centre is within this many
    #                             tracked-box larger sides of the tracked centre ...
    verify_radius_min_px: float = 128.0  # ... or within this many px, whichever is larger
    verify_max_size_ratio: float = 1.5  # ... and whose side is within this ratio of the tracked side
    reinit_iou: float = 0.5  # a verification candidate with a lower IoU than this disagrees with the tracker
    reinit_confirm: int = 2  # consistent disagreeing verifications in a row needed to re-initialise the tracker
    verify_miss_limit: int = 2  # verification runs in a row without a candidate: tracking -> lost

    def validate(self) -> None:
        if not 0.0 < float(self.lost_conf) < 1.0:
            raise ValueError("lost_conf must be in (0, 1)")
        for name in ("lost_frames", "ref_frames", "verify_miss_limit", "reinit_confirm"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not float(self.max_lost_s) > 0.0:
            raise ValueError("max_lost_s must be > 0")
        if not float(self.missing_hz) > 0.0:
            raise ValueError("missing_hz must be > 0")
        if float(self.redetect_hz) < 0.0:
            raise ValueError("redetect_hz must be >= 0")
        if not float(self.verify_radius) > 0.0 or float(self.verify_radius_min_px) < 0.0:
            raise ValueError("verify_radius must be > 0 and verify_radius_min_px >= 0")
        if not float(self.verify_max_size_ratio) >= 1.0:
            raise ValueError("verify_max_size_ratio must be >= 1")
        if not 0.0 < float(self.reinit_iou) <= 1.0:
            raise ValueError("reinit_iou must be in (0, 1]")
        if not 0.0 <= float(self.acquire_min_score) < 1.0:
            raise ValueError("acquire_min_score must be in [0, 1)")


@dataclass(frozen=True)
class SiamFCSourceConfig:
    weights_path: str
    target_range_m: float  # fixed camera-frame depth Z used to back-project the target pixel
    weights_sha256: Optional[str] = None
    device: Optional[str] = None  # None = "cuda" when available, else "cpu"
    params: dict[str, Any] = field(default_factory=dict)  # SiamFCParams overrides; empty = paper values
    bbox_offset_px: tuple[float, float] = (0.0, 0.0)  # added to the box centre to get the aim pixel
    detector: Optional[DetectorConfig] = None
    confirmer: ConfirmerConfig = ConfirmerConfig()
    session: SessionConfig = SessionConfig()

    def validate(self) -> None:
        if not str(self.weights_path).strip():
            raise ValueError("siamfc.weights_path must be set")
        z = float(self.target_range_m)
        if not (math.isfinite(z) and z > 0.0):
            raise ValueError("siamfc.target_range_m must be finite and > 0")
        if self.weights_sha256 is not None and len(str(self.weights_sha256).strip()) != 64:
            raise ValueError("siamfc.weights_sha256 must be a 64-character hex digest")
        if not all(math.isfinite(float(v)) for v in self.bbox_offset_px):
            raise ValueError("siamfc.bbox_offset_px must be finite")
        if self.detector is None:
            raise ValueError("detector configuration is required (weights_path, ...)")
        self.detector.validate()
        self.confirmer.validate()
        self.session.validate()
        # Detections are confirmed only if consecutive full-frame runs are close enough in time.
        if float(self.confirmer.max_gap_s) * float(self.session.missing_hz) < 1.5:
            raise ValueError("confirmer max_gap_s must be at least 1.5 / missing_hz, or no target can ever be confirmed")

    @classmethod
    def from_dicts(cls, siamfc: dict[str, Any], detector: dict[str, Any]) -> "SiamFCSourceConfig":
        """Build from the ``siamfc`` and ``detector`` dicts of the camera configuration.

        ``siamfc``: weights_path, weights_sha256, device, params, target_range_m,
        bbox_offset_px, plus the ``SessionConfig`` fields. ``detector``: the
        ``DetectorConfig`` fields, plus ``confirm_frames``, ``confirm_max_shift``,
        ``confirm_max_size_ratio``, ``confirm_max_gap_s``. Unknown keys are refused, so a
        misspelt setting cannot be silently ignored.
        """
        s, d = dict(siamfc or {}), dict(detector or {})
        session_keys = set(SessionConfig.__dataclass_fields__)
        top_keys = {"weights_path", "weights_sha256", "device", "params", "target_range_m", "bbox_offset_px"}
        unknown = set(s) - session_keys - top_keys
        if unknown:
            raise ValueError(f"unknown siamfc settings: {sorted(unknown)}")
        confirm_map = {
            "confirm_frames": "frames_required",
            "confirm_max_shift": "max_center_shift",
            "confirm_max_size_ratio": "max_size_ratio",
            "confirm_max_gap_s": "max_gap_s",
        }
        det_keys = set(DetectorConfig.__dataclass_fields__)
        unknown = set(d) - det_keys - set(confirm_map)
        if unknown:
            raise ValueError(f"unknown detector settings: {sorted(unknown)}")
        params = s.get("params") or {}
        if isinstance(params, str):
            if params.strip().lower() != "paper":
                raise ValueError("siamfc.params must be 'paper' or a dict of SiamFCParams overrides")
            params = {}
        offset = tuple(float(v) for v in (s.get("bbox_offset_px") or (0.0, 0.0)))
        if len(offset) != 2:
            raise ValueError("siamfc.bbox_offset_px must have 2 values")
        return cls(
            weights_path=str(s.get("weights_path", "") or ""),
            target_range_m=float(s.get("target_range_m", float("nan"))),
            weights_sha256=s.get("weights_sha256") or None,
            device=s.get("device") or None,
            params=dict(params),
            bbox_offset_px=offset,  # type: ignore[arg-type]
            detector=DetectorConfig(**{k: v for k, v in d.items() if k in det_keys}) if d.get("weights_path") else None,
            confirmer=ConfirmerConfig(**{confirm_map[k]: v for k, v in d.items() if k in confirm_map}),
            session=SessionConfig(**{k: v for k, v in s.items() if k in session_keys}),
        )


# ============================================================================
# Geometry
# ============================================================================


def pixel_to_laser_angles(
    cv2_mod,
    pixel_xy: Sequence[float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    r_lc: np.ndarray,
    t_lc: np.ndarray,
    target_range_m: float,
) -> Optional[tuple[float, float, np.ndarray]]:
    """(d_az, d_po, p_L) for the target seen at ``pixel_xy`` (pixel-centre convention).

    The pixel is undistorted to normalised camera coordinates (x_n, y_n), placed at the
    fixed camera depth Z: p_C = [x_n Z, y_n Z, Z], moved to the laser frame
    p_L = R_lc p_C + t_lc, and converted to angles with the same function as the ArUco
    adapters. Returns None for non-finite input or invalid geometry.
    """
    u, v = (float(c) for c in pixel_xy)
    z = float(target_range_m)
    if not all(math.isfinite(c) for c in (u, v, z)) or z <= 0.0:
        return None
    pts = cv2_mod.undistortPoints(
        np.array([[[u, v]]], np.float64),
        np.asarray(camera_matrix, np.float64).reshape(3, 3),
        np.asarray(dist_coeffs, np.float64),
    )
    x_n, y_n = (float(c) for c in np.asarray(pts, np.float64).reshape(2))
    if not (math.isfinite(x_n) and math.isfinite(y_n)):
        return None
    p_l = _point_c_to_laser(np.array([x_n * z, y_n * z, z], np.float64), r_lc, t_lc)
    angles = _angles_from_point_laser(p_l)
    if angles is None:
        return None
    return float(angles[0]), float(angles[1]), p_l


# ============================================================================
# Tracking session (state machine, no threads)
# ============================================================================


class TrackerLike(Protocol):
    def init(self, frame: np.ndarray, center_xy: tuple[float, float], size_wh: tuple[float, float]) -> None: ...
    def update(self, frame: np.ndarray) -> Any: ...  # object with center_xy, size_wh, peak, apce


@dataclass(frozen=True)
class DetectionRequest:
    purpose: str  # "acquire" (full frame) | "verify" (ROI around the tracked box)
    roi_xywh: Optional[tuple[int, int, int, int]]
    epoch: int  # tracker initialisation count at request time
    tracked_bbox_xywh: Optional[tuple[float, float, float, float]]  # tracker box on the submitted frame


@dataclass(frozen=True)
class DetectionOutcome:
    request: DetectionRequest
    detections: tuple[Detection, ...]
    frame: np.ndarray  # the frame the detector ran on
    ts_capture_ns: int
    sequence: int
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class TrackObservation:
    state: str
    reason: str  # "tracking" while tracking, else REASON_SEARCHING / REASON_LOST / REASON_TRACKER_ERROR
    bbox_xywh: Optional[tuple[float, float, float, float]] = None  # pixel-centre convention
    confidence: Optional[float] = None
    peak: Optional[float] = None
    apce: Optional[float] = None

    @property
    def center_xy(self) -> Optional[tuple[float, float]]:
        if self.bbox_xywh is None:
            return None
        x, y, w, h = self.bbox_xywh
        return (x + w / 2.0, y + h / 2.0)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def _bbox_from_center(center_xy: Sequence[float], size_wh: Sequence[float]) -> tuple[float, float, float, float]:
    cx, cy = (float(v) for v in center_xy)
    w, h = (float(v) for v in size_wh)
    return (cx - w / 2.0, cy - h / 2.0, w, h)


class SiamFCTrackingSession:
    """Tracking state machine: see the module docstring for the states and transitions.

    Call order for every new camera frame: ``apply_detection`` for a finished detection
    job (if any), then ``on_frame``, then ``detection_request`` to know whether a new
    detection job should be submitted. Timestamps are frame capture times (monotonic ns).
    """

    def __init__(self, tracker: TrackerLike, confirmer: DetectionConfirmer, cfg: SessionConfig = SessionConfig()) -> None:
        cfg.validate()
        self.cfg = cfg
        self._tracker = tracker
        self._confirmer = confirmer
        self._state = STATE_SEARCHING
        self._epoch = 0
        self._pending_init: Optional[tuple[np.ndarray, Detection]] = None
        self._weak = 0
        self._good = 0
        self._lost_since_ns: Optional[int] = None
        self._recoverable = True
        self._verify_misses = 0
        self._disagree: Optional[tuple[Detection, int]] = None  # last disagreeing verification and its streak
        self._peaks: list[float] = []
        self._apces: list[float] = []
        self._last_obs = TrackObservation(STATE_SEARCHING, REASON_SEARCHING)
        self.counters = {"inits": 0, "reinits_from_verify": 0, "lost": 0, "dropped": 0, "recovered": 0, "tracker_errors": 0}

    @property
    def state(self) -> str:
        return self._state

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def last_observation(self) -> TrackObservation:
        return self._last_obs

    # ------------------------------------------------------------------ detection side

    def detection_request(self, frame_hw: tuple[int, int], now_ns: int, last_submit_ns: Optional[int]) -> Optional[DetectionRequest]:
        """The detection job due now, or None. The caller checks that no job is in flight."""
        if self._state == STATE_TRACKING:
            if float(self.cfg.redetect_hz) <= 0.0 or self._last_obs.bbox_xywh is None:
                return None
            period, purpose = _period_ns(self.cfg.redetect_hz), "verify"
        else:
            period, purpose = _period_ns(self.cfg.missing_hz), "acquire"
        if last_submit_ns is not None and period > 0 and int(now_ns) - int(last_submit_ns) < period:
            return None
        bbox = self._last_obs.bbox_xywh if self._state == STATE_TRACKING else None
        # Always the full frame, also for verification: the detector was trained on whole frames
        # downscaled to imgsz, and a small ROI upscaled to imgsz produced large false positives
        # (replay of DUT video01, 2026-11-10). The verification "region" is applied to the result.
        return DetectionRequest(purpose, None, self._epoch, bbox)

    def verification_candidate(self, dets: Sequence[Detection], tracked: Sequence[float]) -> Optional[Detection]:
        """Best-scoring detection near the tracked box and of similar size, or None."""
        tx, ty, tw, th = (float(v) for v in tracked)
        tcx, tcy = tx + tw / 2.0, ty + th / 2.0
        radius = max(float(self.cfg.verify_radius) * max(tw, th), float(self.cfg.verify_radius_min_px))
        t_side = math.sqrt(max(tw, 1e-6) * max(th, 1e-6))
        ratio_max = float(self.cfg.verify_max_size_ratio)
        for d in sorted(dets, key=lambda d: d.score, reverse=True):
            cx, cy = d.center_xy
            ratio = max(d.side, 1e-6) / t_side
            if math.hypot(cx - tcx, cy - tcy) <= radius and 1.0 / ratio_max <= ratio <= ratio_max:
                return d
        return None

    def apply_detection(self, outcome: DetectionOutcome) -> None:
        req = outcome.request
        dets = list(outcome.detections)
        if req.purpose == "acquire":
            if self._state == STATE_TRACKING:
                return  # the tracker recovered while the job ran; nothing to do
            strong = [d for d in dets if d.score >= float(self.cfg.acquire_min_score)]
            confirmed = self._confirmer.update(strong, outcome.ts_capture_ns)
            if confirmed is not None:
                self._schedule_init(outcome.frame, confirmed)
            return
        # verify
        if self._state != STATE_TRACKING or req.epoch != self._epoch or req.tracked_bbox_xywh is None:
            return  # stale: the tracker was re-initialised or lost since the request
        cand = self.verification_candidate(dets, req.tracked_bbox_xywh)
        if cand is None:
            self._verify_misses += 1
            self._disagree = None
            if self._verify_misses >= int(self.cfg.verify_miss_limit):
                # The detector sees no drone where the tracker is: the tracker's own confidence
                # says nothing about the target any more, so it must not be allowed to recover.
                self._enter_lost(outcome.ts_capture_ns, recoverable=False)
            return
        self._verify_misses = 0
        if bbox_iou(cand.bbox_xywh, req.tracked_bbox_xywh) >= float(self.cfg.reinit_iou):
            self._disagree = None  # detector and tracker agree
            return
        # Detector and tracker disagree: re-initialise only after reinit_confirm consistent verifications,
        # so that one false positive near the drone cannot move the tracker.
        prev = self._disagree
        count = prev[1] + 1 if prev is not None and self._confirmer.consistent(prev[0], cand) else 1
        self._disagree = (cand, count)
        if count >= int(self.cfg.reinit_confirm):
            self.counters["reinits_from_verify"] += 1
            self._schedule_init(outcome.frame, cand)

    def _schedule_init(self, frame: np.ndarray, det: Detection) -> None:
        self._pending_init = (frame, det)

    # ------------------------------------------------------------------ frame side

    def on_frame(self, gray: np.ndarray, ts_capture_ns: int) -> TrackObservation:
        if self._pending_init is not None:
            frame, det = self._pending_init
            self._pending_init = None
            try:
                self._tracker.init(frame, det.center_xy, det.size_wh)
            except Exception:
                self.counters["tracker_errors"] += 1
                return self._drop(REASON_TRACKER_ERROR)
            self._epoch += 1
            self.counters["inits"] += 1
            self._state = STATE_TRACKING
            self._weak = self._good = self._verify_misses = 0
            self._disagree = None
            self._lost_since_ns = None
            self._peaks, self._apces = [], []
            self._confirmer.reset()

        if self._state == STATE_SEARCHING:
            self._last_obs = TrackObservation(STATE_SEARCHING, REASON_SEARCHING)
            return self._last_obs

        try:
            res = self._tracker.update(gray)
        except Exception:
            self.counters["tracker_errors"] += 1
            return self._drop(REASON_TRACKER_ERROR)
        bbox = _bbox_from_center(res.center_xy, res.size_wh)
        conf = self._confidence(float(res.peak), float(res.apce))
        h, w = gray.shape[:2]
        cx, cy = (float(v) for v in res.center_xy)
        inside = 0.0 <= cx <= w - 1.0 and 0.0 <= cy <= h - 1.0 and math.isfinite(conf)

        if self._state == STATE_TRACKING:
            if not inside:
                self._enter_lost(ts_capture_ns, recoverable=False)
            elif conf < float(self.cfg.lost_conf):
                self._weak += 1
                if self._weak >= int(self.cfg.lost_frames):
                    self._enter_lost(ts_capture_ns)
            else:
                self._weak = 0
        elif self._state == STATE_LOST:
            assert self._lost_since_ns is not None
            if self._recoverable and inside and conf >= float(self.cfg.lost_conf):
                self._good += 1
                if self._good >= int(self.cfg.lost_frames):
                    self._state, self._weak, self._good, self._lost_since_ns = STATE_TRACKING, 0, 0, None
                    self._verify_misses = 0
                    self.counters["recovered"] += 1
            else:
                self._good = 0
            if self._state == STATE_LOST and int(ts_capture_ns) - self._lost_since_ns > float(self.cfg.max_lost_s) * 1e9:
                self.counters["dropped"] += 1
                return self._drop(REASON_SEARCHING)

        reason = "tracking" if self._state == STATE_TRACKING else REASON_LOST
        self._last_obs = TrackObservation(self._state, reason, bbox, conf, float(res.peak), float(res.apce))
        return self._last_obs

    def _enter_lost(self, ts_ns: int, recoverable: bool = True) -> None:
        """``recoverable``: the tracker may resume by itself if its confidence returns. False when
        the loss was not a confidence drop (target out of frame, detector disagreement): then only
        a confirmed detection can start tracking again."""
        if self._state != STATE_LOST:
            self.counters["lost"] += 1
        self._state, self._good, self._weak = STATE_LOST, 0, 0
        self._recoverable = bool(recoverable)
        self._lost_since_ns = int(ts_ns)
        self._confirmer.reset()

    def _drop(self, reason: str) -> TrackObservation:
        self._state = STATE_SEARCHING
        self._pending_init = None
        self._disagree = None
        self._weak = self._good = self._verify_misses = 0
        self._lost_since_ns = None
        self._confirmer.reset()
        self._last_obs = TrackObservation(STATE_SEARCHING, reason)
        return self._last_obs

    def _confidence(self, peak: float, apce: float) -> float:
        if not (math.isfinite(peak) and math.isfinite(apce)):
            return float("nan")
        if len(self._peaks) < int(self.cfg.ref_frames):
            self._peaks.append(peak)
            self._apces.append(apce)
        peak_ref = statistics.median(self._peaks)
        apce_ref = statistics.median(self._apces)
        c_peak = min(max(peak / peak_ref, 0.0), 1.0) if peak_ref > 1e-9 else (1.0 if peak > 0.0 else 0.0)
        c_apce = min(max(apce / apce_ref, 0.0), 1.0) if apce_ref > 1e-9 else 0.0
        return math.sqrt(c_peak * c_apce)


# ============================================================================
# Camera adapter
# ============================================================================


class IdsSiamFCCorrectionSourceAdapter:
    """``CorrectionSourcePort`` for the IDS camera with SiamFC + drone detector (``mode = "ids_siamfc"``).

    ``cfg`` is the camera adapter dict (``ids_peak``, ``calibration``, ``debug_view``) plus
    ``siamfc`` and ``detector`` (see ``SiamFCSourceConfig.from_dicts``). The keyword-only
    arguments inject replacements for tests: ``producer`` (latest-frame source with
    ``start/stop/latest/stats``), ``tracker`` and ``detector`` (skip loading weights),
    ``calibration`` (dict as returned by ``_load_vs_calibration_for_camera``) and
    ``executor_factory`` (returns an executor with ``submit/shutdown``).
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        gating: GatingConfig,
        *,
        producer: Any = None,
        tracker: Optional[TrackerLike] = None,
        detector: Optional[DroneDetector] = None,
        calibration: Optional[dict[str, Any]] = None,
        executor_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._cfg = dict(cfg or {})
        self._gating = gating
        self._ids_cfg = dict(self._cfg.get("ids_peak", {}) or {})
        self._calib_cfg = dict(self._cfg.get("calibration", {}) or {})
        self._src_cfg = SiamFCSourceConfig.from_dicts(self._cfg.get("siamfc", {}), self._cfg.get("detector", {}))
        if detector is None or tracker is None:
            self._src_cfg.validate()  # injected test doubles may run without weights paths
        self._fast_cfg = load_fast_tracking_config()  # debug_hz / debug_max_width, shared with the ArUco adapters

        self._inj_producer, self._inj_tracker, self._inj_detector = producer, tracker, detector
        self._inj_calibration = calibration
        self._executor_factory = executor_factory

        self._started = False
        self._cv2 = None
        self._producer: Any = None
        self._detector: Optional[DroneDetector] = None
        self._session: Optional[SiamFCTrackingSession] = None
        self._executor: Any = None
        self._future: Optional[Future] = None
        self._last_submit_ns: Optional[int] = None

        self._k_calib = np.eye(3, dtype=np.float64)
        self._dist = np.zeros((1, 5), dtype=np.float64)
        self._r_lc = np.eye(3, dtype=np.float64)
        self._t_lc = np.zeros((3, 1), dtype=np.float64)
        self._calib_w = 0
        self._calib_h = 0
        self._weights_info: dict[str, Any] = {}

        self._last_sequence = 0
        self._last_candidate: Optional[CorrectionCandidate] = None
        self._last_detections: tuple[Detection, ...] = ()

        self._debug_enable = bool(self._cfg.get("debug_view", False))
        self._debug_lock = Lock()
        self._last_debug_ns = 0
        self._last_debug: dict[str, Any] = {"available": False, "reason": "debug_disabled"}

        self._stats_lock = Lock()
        self._stats: dict[str, Any] = {
            "version": CAMERA_CUDA_VERSION,
            "frames": 0,
            "new_frames": 0,
            "detections_submitted": 0,
            "detections_ok": 0,
            "detection_failures": 0,
            "last_detection_ms": 0.0,
            "last_update_ms": 0.0,
            "tracking_backend": "siamfc",
            "tracker_state": STATE_SEARCHING,
        }

    # ------------------------------------------------------------------ port

    @property
    def started(self) -> bool:
        return bool(self._started)

    @property
    def mode(self) -> str:
        return "ids_siamfc"

    @property
    def weights_info(self) -> dict[str, Any]:
        """Paths and SHA256 of the loaded weights (for the start-up banner)."""
        return dict(self._weights_info)

    @property
    def last_observation(self) -> Optional[TrackObservation]:
        """Tracker state, box and confidence of the last processed frame (None before start)."""
        return None if self._session is None else self._session.last_observation

    @property
    def last_detections(self) -> tuple[Detection, ...]:
        """Detections of the last consumed detector job."""
        return self._last_detections

    def start(self) -> None:
        if self._started:
            return
        self._cv2 = _import_cv2_or_raise()
        calib = self._inj_calibration
        if calib is None:
            path = str(self._calib_cfg.get("path", "") or "").strip()
            if not path:
                raise RuntimeError("ids_siamfc calibration.path is missing")
            calib = _load_vs_calibration_for_camera(Path(path))
        self._k_calib = np.asarray(calib["camera_matrix"], np.float64).reshape(3, 3)
        self._dist = np.asarray(calib["dist_coeffs"], np.float64)
        self._r_lc = np.asarray(calib["R_lc"], np.float64).reshape(3, 3)
        self._t_lc = np.asarray(calib["t_lc_m"], np.float64).reshape(3, 1)
        self._calib_w = int(calib["image_width"])
        self._calib_h = int(calib["image_height"])

        tracker = self._inj_tracker if self._inj_tracker is not None else self._load_tracker()
        self._detector = self._inj_detector if self._inj_detector is not None else self._load_detector()
        self._session = SiamFCTrackingSession(tracker, DetectionConfirmer(self._src_cfg.confirmer), self._src_cfg.session)
        self._executor = (
            self._executor_factory() if self._executor_factory is not None
            else ThreadPoolExecutor(max_workers=1, thread_name_prefix="drone-detector")
        )
        self._producer = self._inj_producer if self._inj_producer is not None else _IdsLatestFrameProducer(self._ids_cfg)
        self._producer.start()
        self._started = True

    def stop(self) -> None:
        self._started = False
        if self._producer is not None:
            self._producer.stop()
        self._producer = None
        if self._future is not None:
            self._future.cancel()
        self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        self._session = None
        self._last_submit_ns = None
        self._last_sequence = 0
        self._last_candidate = None

    def get_debug_snapshot(self) -> dict[str, Any]:
        if not self._debug_enable:
            return {"available": False, "reason": "debug_disabled"}
        with self._debug_lock:
            return _clone_debug_payload(self._last_debug)

    def fast_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            out = dict(self._stats)
        if self._session is not None:
            out["session"] = dict(self._session.counters)
        if self._producer is not None and hasattr(self._producer, "stats"):
            out["capture"] = self._producer.stats()
        return out

    def read_candidate(self, now_ns: int) -> CorrectionCandidate:
        if not self._started or self._producer is None or self._session is None:
            return CorrectionCandidate(False, "source_not_started")
        packet = self._producer.latest()
        if packet is None:
            return CorrectionCandidate(False, "frame_not_available")
        with self._stats_lock:
            self._stats["frames"] += 1
        if int(packet.sequence) == int(self._last_sequence):
            return self._reuse_last_candidate(now_ns)

        self._last_sequence = int(packet.sequence)
        gray = np.asarray(packet.gray, dtype=np.uint8)
        ts_capture_ns = int(packet.ts_capture_ns)
        h, w = gray.shape[:2]
        with self._stats_lock:
            self._stats["new_frames"] += 1

        self._consume_detection()
        t0 = time.perf_counter_ns()
        obs = self._session.on_frame(gray, ts_capture_ns)
        update_ms = (time.perf_counter_ns() - t0) * 1e-6
        self._schedule_detection(gray, ts_capture_ns, int(packet.sequence))
        with self._stats_lock:
            self._stats["last_update_ms"] = float(update_ms)
            self._stats["tracker_state"] = obs.state

        k_runtime, _, _ = _scale_intrinsics_to_runtime(self._k_calib, self._calib_w, self._calib_h, w, h)
        candidate, aim_px, p_l = self._candidate_from_observation(obs, k_runtime, ts_capture_ns)
        self._last_candidate = candidate
        self._update_debug(gray, candidate, obs, aim_px, k_runtime)
        return candidate

    # ------------------------------------------------------------------ internals

    def _candidate_from_observation(
        self, obs: TrackObservation, k_runtime: np.ndarray, ts_capture_ns: int
    ) -> tuple[CorrectionCandidate, Optional[tuple[float, float]], Optional[np.ndarray]]:
        if obs.state != STATE_TRACKING or obs.center_xy is None:
            return CorrectionCandidate(False, obs.reason, confidence=obs.confidence, ts_capture_ns=ts_capture_ns), None, None
        conf = float(obs.confidence) if obs.confidence is not None else 0.0
        ox, oy = self._src_cfg.bbox_offset_px
        aim = (obs.center_xy[0] + float(ox), obs.center_xy[1] + float(oy))
        if conf < float(self._gating.conf_min):
            return CorrectionCandidate(False, "confidence_low", confidence=conf, ts_capture_ns=ts_capture_ns), aim, None
        geo = pixel_to_laser_angles(
            self._cv2, aim, k_runtime, self._dist, self._r_lc, self._t_lc, self._src_cfg.target_range_m
        )
        if geo is None:
            return CorrectionCandidate(False, "invalid_target_geometry", confidence=conf, ts_capture_ns=ts_capture_ns), aim, None
        d_az, d_po, p_l = geo
        z_m = float(p_l[2])
        if z_m < float(self._gating.z_min_m):
            return CorrectionCandidate(False, "z_below_min", confidence=conf, z_m=z_m, ts_capture_ns=ts_capture_ns), aim, p_l
        age_s = max((time.monotonic_ns() - int(ts_capture_ns)) * 1e-9, 0.0)
        if age_s > float(self._gating.max_age_s):
            return CorrectionCandidate(
                False, "sample_stale", confidence=conf, z_m=z_m, age_s=age_s, ts_capture_ns=ts_capture_ns
            ), aim, p_l
        return CorrectionCandidate(
            True, "ok", d_az_rad=d_az, d_po_rad=d_po, confidence=conf, age_s=age_s, z_m=z_m, ts_capture_ns=ts_capture_ns
        ), aim, p_l

    def _reuse_last_candidate(self, now_ns: int) -> CorrectionCandidate:
        previous = self._last_candidate
        if previous is None or previous.ts_capture_ns is None:
            return CorrectionCandidate(False, "no_new_frame")
        age_s = max((int(now_ns) - int(previous.ts_capture_ns)) * 1e-9, 0.0)
        if age_s > float(self._gating.max_age_s):
            return CorrectionCandidate(
                False, "sample_stale", confidence=previous.confidence, marker_id=previous.marker_id,
                z_m=previous.z_m, age_s=age_s, ts_capture_ns=previous.ts_capture_ns,
            )
        return replace(previous, age_s=age_s)

    def _consume_detection(self) -> None:
        future = self._future
        if future is None or not future.done():
            return
        self._future = None
        try:
            outcome: DetectionOutcome = future.result()
        except Exception:
            with self._stats_lock:
                self._stats["detection_failures"] += 1
            return
        self._last_detections = outcome.detections
        with self._stats_lock:
            self._stats["last_detection_ms"] = float(outcome.elapsed_ms)
            if outcome.detections:
                self._stats["detections_ok"] += 1
        assert self._session is not None
        self._session.apply_detection(outcome)

    def _schedule_detection(self, gray: np.ndarray, ts_capture_ns: int, sequence: int) -> None:
        if self._executor is None or self._session is None or self._detector is None:
            return
        if self._future is not None and not self._future.done():
            return
        request = self._session.detection_request(gray.shape[:2], ts_capture_ns, self._last_submit_ns)
        if request is None:
            return
        frame = np.array(gray, copy=True)  # the producer may reuse buffers; the job owns its frame
        self._future = self._executor.submit(self._detect_job, request, frame, int(ts_capture_ns), int(sequence))
        self._last_submit_ns = int(ts_capture_ns)
        with self._stats_lock:
            self._stats["detections_submitted"] += 1

    def _detect_job(self, request: DetectionRequest, frame: np.ndarray, ts_capture_ns: int, sequence: int) -> DetectionOutcome:
        assert self._detector is not None
        t0 = time.perf_counter_ns()
        dets = self._detector.detect(frame, request.roi_xywh)
        return DetectionOutcome(
            request, tuple(dets), frame, ts_capture_ns, sequence, elapsed_ms=(time.perf_counter_ns() - t0) * 1e-6
        )

    def _load_tracker(self) -> TrackerLike:
        import torch  # noqa: PLC0415

        from .siamfc.net import SiamFCNet  # noqa: PLC0415
        from .siamfc.tracker import SiamFCTracker, params_with  # noqa: PLC0415

        cfg = self._src_cfg
        path = Path(cfg.weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"SiamFC weights not found: {path}")
        digest = sha256_file(path)
        if cfg.weights_sha256 is not None and digest.lower() != str(cfg.weights_sha256).strip().lower():
            raise ValueError(f"SiamFC weights SHA256 mismatch for {path}: {digest} != {cfg.weights_sha256}")
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        params = params_with(**cfg.params)
        params.validate()
        net = SiamFCNet.from_file(path, map_location=device).to(device).eval()
        tracker = SiamFCTracker(net, params, device=device)
        # Warm-up so the first real initialisation does not pay the CUDA start-up.
        blank = np.full((256, 256), 128, np.uint8)
        tracker.init(blank, (128.0, 128.0), (32.0, 32.0))
        tracker.update(blank)
        self._weights_info["siamfc"] = {"path": str(path), "sha256": digest, "device": device}
        return tracker

    def _load_detector(self) -> DroneDetector:
        from .siamfc.detector import UltralyticsDroneDetector  # noqa: PLC0415

        dcfg = self._src_cfg.detector
        assert dcfg is not None
        det = UltralyticsDroneDetector(dcfg)
        det.warmup()
        self._weights_info["detector"] = {"path": str(dcfg.weights_path), "sha256": sha256_file(dcfg.weights_path)}
        return det

    def _update_debug(
        self,
        gray: np.ndarray,
        candidate: CorrectionCandidate,
        obs: TrackObservation,
        aim_px: Optional[tuple[float, float]],
        camera_matrix: np.ndarray,
    ) -> None:
        if not self._debug_enable:
            return
        now_ns = time.monotonic_ns()
        period = _period_ns(self._fast_cfg.debug_hz)
        if period > 0 and now_ns - self._last_debug_ns < period:
            return
        self._last_debug_ns = now_ns
        h, w = gray.shape[:2]
        scale = min(1.0, float(self._fast_cfg.debug_max_width) / max(float(w), 1.0))
        if scale < 1.0:
            preview = self._cv2.resize(
                gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=self._cv2.INTER_AREA
            )
        else:
            preview = np.array(gray, copy=True)

        z = float(self._src_cfg.target_range_m)
        laser_px = _project_camera_point_to_pixel(
            self._cv2, _laser_axis_point_in_camera(self._r_lc, self._t_lc, z), camera_matrix, self._dist
        )

        def scaled_box(b):
            return None if b is None else tuple(float(v) * scale for v in b)

        payload = {
            "available": True,
            "version": CAMERA_CUDA_VERSION,
            "reason": candidate.reason,
            "frame_gray": preview,
            "preview_scale": float(scale),
            "ts_capture_ns": candidate.ts_capture_ns,
            # keys read by cuasp._draw_debug_frame (shared with the ArUco adapters)
            "markers": [],
            "expected_marker_ids": (),
            "target_point": None,
            "pbvs_target_pixel_px": None if aim_px is None else np.asarray(aim_px, np.float64) * float(scale),
            "principal_point_px": np.asarray([float(camera_matrix[0, 2]), float(camera_matrix[1, 2])], np.float64) * float(scale),
            "laser_axis_pixel_px": None if laser_px is None else laser_px * float(scale),
            "laser_axis_range_m": z,
            "laser_axis_range_source": "fixed",
            "tile_state": obs.state,
            "tracking_backend": "siamfc",
            "confidence": candidate.confidence,
            "z_m": candidate.z_m,
            "z_mode": "fixed",
            "d_az_rad": candidate.d_az_rad,
            "d_po_rad": candidate.d_po_rad,
            # SiamFC-specific keys
            "tracker_state": obs.state,
            "siamfc_bbox_xywh": scaled_box(obs.bbox_xywh),
            "siamfc_score": obs.confidence,
            "siamfc_peak": obs.peak,
            "siamfc_apce": obs.apce,
            "detections": [{"bbox_xywh": scaled_box(d.bbox_xywh), "score": float(d.score)} for d in self._last_detections],
            "stats": self.fast_stats(),
        }
        with self._debug_lock:
            self._last_debug = payload


__all__ = [
    "DetectionOutcome",
    "DetectionRequest",
    "IdsSiamFCCorrectionSourceAdapter",
    "SessionConfig",
    "SiamFCSourceConfig",
    "SiamFCTrackingSession",
    "TrackObservation",
    "bbox_iou",
    "pixel_to_laser_angles",
]
