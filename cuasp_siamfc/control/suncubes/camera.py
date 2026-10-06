"""suncubes camera facade with high-rate IDS/OpenCV PBVS tracking.

Key properties:
- preserves the public ``CameraAPI`` / ``CameraCorrection`` interface;
- acquires IDS frames in a dedicated latest-only producer thread;
- never builds a FIFO backlog of stale camera frames;
- performs sparse KLT corner tracking on every new frame;
- runs ArUco decoding asynchronously and at a lower configurable frequency;
- performs forward/backward optical-flow validation and geometric gating;
- uses CUDA sparse PyrLK only when OpenCV exposes it and a runtime benchmark
  shows that it is faster than the CPU implementation;
- automatically falls back to CPU if CUDA is unavailable or fails;
- applies PBVS target offsets in the marker frame;
- keeps full-frame recovery time-based rather than frame-count based;
- decimates and downsizes debug snapshots.

The effective high-rate path is:
    IDS capture -> latest frame -> KLT -> solvePnP -> target offset -> PBVS

ArUco decoding is used for initialization, drift correction and recovery, not on
all 70 Hz tracking iterations.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Lock, Thread
import math
import os
import re
import time
from typing import Any, Optional, Protocol, Sequence

import numpy as np


CAMERA_CUDA_VERSION = "2026-08-03-camera-cuda-v3-cuas-detection-debug"


# ============================================================================
# Domain contracts
# ============================================================================


@dataclass(frozen=True)
class CorrectionCandidate:
    reliable: bool
    reason: str
    d_az_rad: float = 0.0
    d_po_rad: float = 0.0
    confidence: Optional[float] = None
    age_s: Optional[float] = None
    marker_id: Optional[int] = None
    z_m: Optional[float] = None
    ts_capture_ns: Optional[int] = None


@dataclass(frozen=True)
class CameraCorrection:
    reliable: bool
    reason: str
    d_az_rad: float
    d_po_rad: float
    confidence: Optional[float]
    age_s: Optional[float]
    marker_id: Optional[int]
    z_m: Optional[float]
    ts_capture_ns: Optional[int]
    ts_consumed_ns: int


@dataclass(frozen=True)
class CameraHealth:
    running: bool
    mode: str
    source_started: bool
    last_reason: str
    last_update_ns: Optional[int]
    accepted_samples: int
    rejected_samples: int


@dataclass(frozen=True)
class GatingConfig:
    max_age_s: float = 0.25
    conf_min: float = 0.15
    z_min_m: float = 0.25
    max_abs_angle_rad: float = 0.0


@dataclass(frozen=True)
class RuntimeConfig:
    # High-rate operation is enabled by default in this replacement module.
    background_worker: bool = True
    worker_hz: float = 70.0


@dataclass(frozen=True)
class CameraConfig:
    enable: bool = False
    mode: str = "csv"  # csv | ids_opencv
    csv_path: str = ""
    gating: GatingConfig = GatingConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    ids_opencv: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class FastTrackingConfig:
    redetect_hz: float = 3.0
    missing_redetect_hz: float = 8.0
    full_recovery_hz: float = 0.5
    detection_result_max_age_s: float = 0.12
    detection_preprocess: str = "raw"  # raw | stretch | clahe
    detection_roi_scale: float = 2.5
    detection_roi_min_side_px: int = 320
    missing_detection_roi_scale: float = 8.0
    missing_detection_roi_min_side_px: int = 384
    scout_grid_cols: int = 3
    scout_grid_rows: int = 3
    scout_overlap: float = 0.25
    klt_win_size: int = 15
    klt_max_level: int = 2
    klt_max_iters: int = 10
    klt_eps: float = 0.03
    klt_roi_margin_px: int = 64
    klt_roi_scale: float = 5.0
    klt_fb_check_period: int = 4
    klt_use_initial_flow: bool = True
    klt_robust_win_size: int = 21
    klt_robust_max_level: int = 3
    klt_robust_max_iters: int = 15
    klt_robust_eps: float = 0.01
    fb_max_error_px: float = 1.25
    lk_max_error: float = 30.0
    area_ratio_min: float = 0.35
    area_ratio_max: float = 2.85
    max_side_ratio: float = 4.0
    min_quad_area_px2: float = 16.0
    cuda_mode: str = "auto"  # auto | off | force
    cuda_use_klt: bool = True
    cuda_benchmark_frames: int = 8
    debug_hz: float = 2.0
    debug_max_width: int = 960
    # Range at which the debug overlay's "where the laser boresight axis
    # should land" marker is projected when no target is being tracked (with
    # a target, its own laser-frame range is used instead). The marker's
    # offset from the optical axis is a parallax term -- it scales with
    # 1/range -- so this value is reported on screen alongside it, never
    # silently assumed.
    debug_laser_range_m: float = 10.0


@dataclass(frozen=True)
class PbvsZConfig:
    mode: str = "measured"  # measured | filtered | fixed
    fixed_m: float = 0.0
    filter_tau_s: float = 0.35


@dataclass(frozen=True)
class PbvsZState:
    ok: bool
    mode: str
    z_raw_m: Optional[float]
    z_used_m: Optional[float]
    status: str


@dataclass(frozen=True)
class _FramePacket:
    gray: np.ndarray
    ts_capture_ns: int
    sequence: int


@dataclass(frozen=True)
class _DetectionResult:
    ok: bool
    reason: str
    frame: np.ndarray
    ts_capture_ns: int
    sequence: int
    corners: Optional[np.ndarray] = None
    marker_id: Optional[int] = None
    mode: str = "none"
    roi_xywh: Optional[tuple[int, int, int, int]] = None
    elapsed_ms: float = 0.0


# ============================================================================
# Settings helpers
# ============================================================================


def _setting(name: str, default: Any) -> Any:
    try:
        from . import settings
        return getattr(settings, name, default)
    except (ImportError, ModuleNotFoundError):
        return default


def _float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _bool_setting(name: str, default: bool) -> bool:
    value = _setting(name, int(default))
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def _period_ns(rate_hz: float) -> int:
    return 0 if float(rate_hz) <= 0.0 else int(round(1e9 / float(rate_hz)))


def load_fast_tracking_config() -> FastTrackingConfig:
    return FastTrackingConfig(
        redetect_hz=float(_setting("CV_CUAS_ARUCO_REDETECT_HZ", 3.0)),
        missing_redetect_hz=float(_setting("CV_CUAS_MISSING_MARKER_DETECT_HZ", 8.0)),
        full_recovery_hz=float(_setting("CV_CUAS_FULL_FRAME_RECOVERY_HZ", 0.5)),
        detection_result_max_age_s=float(
            _setting("CV_CUAS_DETECTION_RESULT_MAX_AGE_S", 0.12)
        ),
        detection_preprocess=str(_setting("CV_CUAS_PREPROCESS_MODE", "raw")).strip().lower(),
        detection_roi_scale=float(
            _setting(
                "CV_CUAS_TRACK_DETECTION_ROI_SCALE",
                _setting("CV_CUAS_DETECTION_ROI_SCALE", 2.5),
            )
        ),
        detection_roi_min_side_px=int(
            _setting(
                "CV_CUAS_TRACK_DETECTION_ROI_MIN_SIDE_PX",
                _setting("CV_CUAS_DETECTION_ROI_MIN_SIDE_PX", 320),
            )
        ),
        missing_detection_roi_scale=float(
            _setting("CV_CUAS_MISSING_DETECTION_ROI_SCALE", 8.0)
        ),
        missing_detection_roi_min_side_px=int(
            _setting("CV_CUAS_MISSING_DETECTION_ROI_MIN_SIDE_PX", 384)
        ),
        scout_grid_cols=int(_setting("CV_ARUCO_SCOUT_GRID_COLS", 3)),
        scout_grid_rows=int(_setting("CV_ARUCO_SCOUT_GRID_ROWS", 3)),
        scout_overlap=float(_setting("CV_ARUCO_SCOUT_TILE_OVERLAP", 0.25)),
        klt_win_size=int(_setting("CV_CUAS_KLT_WIN_SIZE", 15)),
        klt_max_level=int(_setting("CV_CUAS_KLT_MAX_LEVEL", 2)),
        klt_max_iters=int(_setting("CV_CUAS_KLT_MAX_ITERS", 10)),
        klt_eps=float(_setting("CV_CUAS_KLT_EPS", 0.03)),
        klt_roi_margin_px=int(_setting("CV_CUAS_KLT_ROI_MARGIN_PX", 64)),
        klt_roi_scale=float(_setting("CV_CUAS_KLT_ROI_SCALE", 5.0)),
        klt_fb_check_period=max(1, int(_setting("CV_CUAS_KLT_FB_CHECK_PERIOD", 4))),
        klt_use_initial_flow=_bool_setting("CV_CUAS_KLT_USE_INITIAL_FLOW", True),
        klt_robust_win_size=int(_setting("CV_CUAS_KLT_ROBUST_WIN_SIZE", 21)),
        klt_robust_max_level=int(_setting("CV_CUAS_KLT_ROBUST_MAX_LEVEL", 3)),
        klt_robust_max_iters=int(_setting("CV_CUAS_KLT_ROBUST_MAX_ITERS", 15)),
        klt_robust_eps=float(_setting("CV_CUAS_KLT_ROBUST_EPS", 0.01)),
        fb_max_error_px=float(_setting("CV_CUAS_KLT_FB_MAX_ERROR_PX", 1.25)),
        lk_max_error=float(_setting("CV_CUAS_KLT_MAX_LK_ERROR", 30.0)),
        area_ratio_min=float(_setting("CV_CUAS_KLT_AREA_RATIO_MIN", 0.35)),
        area_ratio_max=float(_setting("CV_CUAS_KLT_AREA_RATIO_MAX", 2.85)),
        max_side_ratio=float(_setting("CV_CUAS_KLT_MAX_SIDE_RATIO", 4.0)),
        min_quad_area_px2=float(
            _setting("CV_CUAS_KLT_MIN_QUAD_AREA_PX2", 16.0)
        ),
        cuda_mode=str(_setting("CV_CUAS_CUDA_MODE", "auto")).strip().lower(),
        cuda_use_klt=_bool_setting("CV_CUAS_CUDA_USE_KLT", True),
        cuda_benchmark_frames=int(_setting("CV_CUAS_CUDA_BENCHMARK_FRAMES", 8)),
        debug_hz=float(_setting("CV_CUAS_DEBUG_DRAW_HZ", 2.0)),
        debug_max_width=int(_setting("CV_CUAS_DEBUG_MAX_WIDTH", 960)),
        debug_laser_range_m=float(_setting("CV_CUAS_DEBUG_LASER_RANGE_M", 10.0)),
    )


def load_pbvs_z_config() -> PbvsZConfig:
    return PbvsZConfig(
        mode=str(_setting("CV_PBVS_Z_MODE", "measured")).strip().lower(),
        fixed_m=float(_setting("CV_PBVS_Z_FIXED_M", 0.0)),
        filter_tau_s=float(_setting("CV_PBVS_Z_FILTER_TAU_S", 0.35)),
    )


# ============================================================================
# Public source protocol
# ============================================================================


class CorrectionSourcePort(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_candidate(self, now_ns: int) -> CorrectionCandidate: ...

    @property
    def started(self) -> bool: ...

    @property
    def mode(self) -> str: ...


# ============================================================================
# CSV source
# ============================================================================


class CsvCorrectionSourceAdapter:
    def __init__(self, csv_path: str, gating: GatingConfig):
        self._csv_path = str(csv_path or "").strip()
        self._gating = gating
        self._stream = None
        self._started = False

    @property
    def started(self) -> bool:
        return bool(self._started)

    @property
    def mode(self) -> str:
        return "csv"

    def start(self) -> None:
        # cv_aruco_helper.py is not part of this repository yet (see
        # suncubes/README.md "camera technical debt"); CSV replay mode remains
        # unavailable until it is added.
        from .cv_aruco_helper import CvArucoStream
        self._stream = CvArucoStream(self._csv_path)
        self._started = True

    def stop(self) -> None:
        self._stream = None
        self._started = False

    def read_candidate(self, now_ns: int) -> CorrectionCandidate:
        if not self._started or self._stream is None:
            return CorrectionCandidate(False, "source_not_started")
        d_az, d_po, meta = self._stream.get_raw_delta(
            now_s=float(now_ns) * 1e-9,
            max_age_s=float(self._gating.max_age_s),
            conf_min=float(self._gating.conf_min),
            z_min_m=float(self._gating.z_min_m),
        )
        meta = dict(meta or {})
        return CorrectionCandidate(
            reliable=bool(meta.get("reliable", False)),
            reason=str(meta.get("reason", "unknown")),
            d_az_rad=float(d_az),
            d_po_rad=float(d_po),
            confidence=_float_or_none(meta.get("confidence")),
            age_s=_float_or_none(meta.get("age_s")),
            marker_id=_int_or_none(meta.get("marker_id")),
            z_m=_float_or_none(meta.get("z_m")),
            ts_capture_ns=_int_or_none(meta.get("ts_rx_monotonic_ns")),
        )

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {"available": False, "reason": "debug_not_supported_csv"}


# ============================================================================
# Latest-only capture slot and IDS producer
# ============================================================================


class _LatestFrameSlot:
    def __init__(self):
        self._cond = Condition()
        self._packet: Optional[_FramePacket] = None
        self._published = 0
        self._replaced = 0

    def publish(self, gray: np.ndarray, ts_capture_ns: int) -> None:
        frame = np.asarray(gray, dtype=np.uint8)
        with self._cond:
            if self._packet is not None:
                self._replaced += 1
            self._published += 1
            self._packet = _FramePacket(
                gray=frame,
                ts_capture_ns=int(ts_capture_ns),
                sequence=int(self._published),
            )
            self._cond.notify_all()

    def latest(self) -> Optional[_FramePacket]:
        with self._cond:
            return self._packet

    def wait_newer(self, last_sequence: int, timeout_s: float) -> Optional[_FramePacket]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cond:
            while self._packet is None or self._packet.sequence <= int(last_sequence):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._cond.wait(timeout=remaining)
            return self._packet

    def stats(self) -> dict[str, int]:
        with self._cond:
            return {
                "published": int(self._published),
                "replaced": int(self._replaced),
                "sequence": 0 if self._packet is None else int(self._packet.sequence),
            }


class _IdsLatestFrameProducer:
    """Owns the complete IDS SDK lifecycle in one dedicated thread."""

    def __init__(self, ids_cfg: dict[str, Any]):
        self._cfg = dict(ids_cfg or {})
        self._slot = _LatestFrameSlot()
        self._thread: Optional[Thread] = None
        self._stop = Event()
        self._ready = Event()
        self._error: Optional[BaseException] = None
        self._started = False

    @property
    def started(self) -> bool:
        return bool(self._started)

    def start(self, timeout_s: float = 10.0) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._ready.clear()
        self._error = None
        self._thread = Thread(target=self._run, name="ids-latest-frame", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=max(0.1, float(timeout_s))):
            self.stop()
            raise RuntimeError("IDS capture thread start timeout")
        if self._error is not None:
            error = self._error
            self.stop()
            raise RuntimeError(f"IDS capture start failed: {error}") from error
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._started = False

    def latest(self) -> Optional[_FramePacket]:
        return self._slot.latest()

    def wait_newer(self, last_sequence: int, timeout_s: float) -> Optional[_FramePacket]:
        return self._slot.wait_newer(last_sequence, timeout_s)

    def stats(self) -> dict[str, int]:
        return self._slot.stats()

    def _run(self) -> None:
        ids_peak = ids_ipl = None
        datastream = nodemap = device = None
        library_initialized = False
        try:
            gentl = str(self._cfg.get("gentl_cti_path", "") or "").strip()
            if gentl:
                existing = os.environ.get("GENICAM_GENTL64_PATH", "")
                paths = [p for p in existing.split(os.pathsep) if p]
                if gentl not in paths:
                    os.environ["GENICAM_GENTL64_PATH"] = os.pathsep.join([gentl, *paths])

            ids_peak, ids_ipl = _load_ids_modules()
            ids_peak.Library.Initialize()
            library_initialized = True
            dm = ids_peak.DeviceManager.Instance()
            dm.Update()
            if dm.Devices().empty():
                raise RuntimeError("No IDS camera detected")

            index = int(self._cfg.get("device_index", 0))
            count = int(dm.Devices().size())
            if index < 0 or index >= count:
                raise RuntimeError(f"Invalid IDS device_index={index}; detected={count}")

            device = dm.Devices()[index].OpenDevice(ids_peak.DeviceAccessType_Exclusive)
            datastream = device.DataStreams()[0].OpenDataStream()
            nodemap = device.RemoteDevice().NodeMaps()[0]

            _try_set_node(nodemap, "PixelFormat", lambda n: n.SetCurrentEntry(str(self._cfg.get("pixel_format", "Mono8"))))
            if "width" in self._cfg:
                _try_set_node(nodemap, "Width", lambda n: n.SetValue(int(self._cfg["width"])))
            if "height" in self._cfg:
                _try_set_node(nodemap, "Height", lambda n: n.SetValue(int(self._cfg["height"])))
            _try_set_node(
                nodemap,
                "AcquisitionMode",
                lambda n: n.SetCurrentEntry(str(self._cfg.get("acquisition_mode", "Continuous"))),
            )

            target_hz = _float_or_none(self._cfg.get("frame_rate_target_hz"))
            settings_hz = _ids_frame_rate_target_hz_from_settings()
            if settings_hz is not None:
                target_hz = settings_hz
            if target_hz is not None and target_hz > 0.0:
                _set_acquisition_frame_rate_target_hz(nodemap, target_hz)

            _try_set_node(
                nodemap,
                "ExposureAuto",
                lambda n: n.SetCurrentEntry(str(self._cfg.get("exposure_auto", "Off"))),
            )
            _try_set_node(
                nodemap,
                "GainAuto",
                lambda n: n.SetCurrentEntry(str(self._cfg.get("gain_auto", "Off"))),
            )
            exposure_us = _ids_exposure_time_us_from_settings()
            if exposure_us is None:
                exposure_us = _float_or_none(self._cfg.get("exposure_time_us"))
            if exposure_us is not None and exposure_us > 0.0:
                _set_exposure_time_us(nodemap, exposure_us)
            if "gain_db" in self._cfg:
                _try_set_node(nodemap, "Gain", lambda n: n.SetValue(float(self._cfg["gain_db"])))

            payload_size = int(nodemap.FindNode("PayloadSize").Value())
            min_required = int(datastream.NumBuffersAnnouncedMinRequired())
            # A few extra buffers reduce acquisition starvation during OS jitter.
            buffer_count = max(min_required, int(self._cfg.get("buffer_count", min_required + 2)))
            for _ in range(buffer_count):
                buf = datastream.AllocAndAnnounceBuffer(payload_size)
                datastream.QueueBuffer(buf)

            datastream.StartAcquisition()
            _try_set_node(nodemap, "AcquisitionStart", lambda n: n.Execute())
            self._ready.set()

            timeout_ms = max(1, int(self._cfg.get("buffer_timeout_ms", 100)))
            while not self._stop.is_set():
                buffer = None
                try:
                    buffer = datastream.WaitForFinishedBuffer(timeout_ms)
                    ts_ns = time.monotonic_ns()
                    sdk_view = _buffer_to_mono8_numpy(ids_ipl, buffer)
                    # Required before QueueBuffer: the SDK buffer may be reused immediately.
                    gray = np.array(sdk_view, dtype=np.uint8, copy=True)
                    self._slot.publish(gray, ts_ns)
                except Exception:
                    if self._stop.is_set():
                        break
                    continue
                finally:
                    if buffer is not None:
                        try:
                            datastream.QueueBuffer(buffer)
                        except Exception:
                            pass
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            try:
                if nodemap is not None:
                    _try_set_node(nodemap, "AcquisitionStop", lambda n: n.Execute())
            except Exception:
                pass
            try:
                if datastream is not None:
                    datastream.StopAcquisition()
            except Exception:
                pass
            try:
                if datastream is not None and ids_peak is not None:
                    mode = getattr(ids_peak, "DataStreamFlushMode_DiscardAll", None)
                    if mode is not None:
                        datastream.Flush(mode)
            except Exception:
                pass
            try:
                if library_initialized and ids_peak is not None:
                    ids_peak.Library.Close()
            except Exception:
                pass
            self._ready.set()


# ============================================================================
# Sparse KLT backend with CUDA auto-selection
# ============================================================================


class SparseCornerTracker:
    """ROI sparse LK with predicted initial flow and periodic FB validation."""

    def __init__(self, cv2_mod, cfg: FastTrackingConfig):
        self.cv2 = cv2_mod
        self.cfg = cfg
        self._selected = "cpu"
        self._cuda_lk = None
        self._cpu_samples: list[float] = []
        self._cuda_samples: list[float] = []
        self._call_count = 0
        self._last_displacement: Optional[np.ndarray] = None
        self._init_cuda()

    @property
    def backend(self) -> str:
        return str(self._selected)

    def reset_motion_model(self) -> None:
        self._last_displacement = None

    def _init_cuda(self) -> None:
        mode = str(self.cfg.cuda_mode).strip().lower()
        if mode == "off" or not bool(self.cfg.cuda_use_klt):
            return
        try:
            count = int(self.cv2.cuda.getCudaEnabledDeviceCount())
            factory = getattr(self.cv2.cuda, "SparsePyrLKOpticalFlow_create", None)
            if factory is None:
                cls = getattr(self.cv2, "cuda_SparsePyrLKOpticalFlow", None)
                factory = getattr(cls, "create", None) if cls is not None else None
            if count <= 0 or factory is None:
                if mode == "force":
                    raise RuntimeError("OpenCV CUDA sparse PyrLK unavailable")
                return
            self._cuda_lk = factory(
                winSize=(int(self.cfg.klt_win_size), int(self.cfg.klt_win_size)),
                maxLevel=int(self.cfg.klt_max_level),
                iters=int(self.cfg.klt_max_iters),
                useInitialFlow=False,
            )
            self._selected = "cuda" if mode == "force" else "benchmark"
        except Exception:
            if mode == "force":
                raise
            self._cuda_lk = None
            self._selected = "cpu"

    def track(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) == 0:
            return pts, np.zeros(0, dtype=bool), np.zeros(0), "none"
        self._call_count += 1
        if self._last_displacement is not None and self._last_displacement.shape != pts.shape:
            self._last_displacement = None

        prev_roi, curr_roi, local_pts, origin = self._crop_tracking_roi(previous, current, pts)

        if self._selected == "cuda" and self._cuda_lk is not None:
            try:
                p, valid, fb = self._track_cuda(prev_roi, curr_roi, local_pts)
                self._remember(local_pts, p, valid)
                return p + origin, valid, fb, "cuda"
            except Exception:
                self._selected = "cpu"
                self._cuda_lk = None

        if self._selected == "benchmark" and self._cuda_lk is not None:
            t0 = time.perf_counter_ns()
            cpu = self._track_cpu_adaptive(prev_roi, curr_roi, local_pts)
            self._cpu_samples.append((time.perf_counter_ns() - t0) * 1e-6)
            try:
                t1 = time.perf_counter_ns()
                self._track_cuda(prev_roi, curr_roi, local_pts)
                self._cuda_samples.append((time.perf_counter_ns() - t1) * 1e-6)
                if len(self._cpu_samples) >= max(1, int(self.cfg.cuda_benchmark_frames)):
                    cpu_med = float(np.median(self._cpu_samples))
                    cuda_med = float(np.median(self._cuda_samples))
                    self._selected = "cuda" if cuda_med < 0.90 * cpu_med else "cpu"
                p, valid, fb = cpu
                self._remember(local_pts, p, valid)
                return p + origin, valid, fb, "benchmark_cpu"
            except Exception:
                self._selected = "cpu"
                self._cuda_lk = None

        p, valid, fb = self._track_cpu_adaptive(prev_roi, curr_roi, local_pts)
        self._remember(local_pts, p, valid)
        return p + origin, valid, fb, "cpu"

    def _remember(self, old, new, valid) -> None:
        valid = np.asarray(valid, bool).reshape(-1)
        if len(valid) == len(old) and bool(np.all(valid)):
            self._last_displacement = np.asarray(new - old, np.float32)
        else:
            self._last_displacement = None

    def _crop_tracking_roi(self, previous, current, points):
        h, w = previous.shape[:2]
        sides = []
        if len(points) >= 4:
            for i in range(0, len(points) - 3, 4):
                quad = points[i : i + 4]
                sides.extend(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).tolist())
        dynamic = (
            int(round(float(self.cfg.klt_roi_scale) * float(np.median(sides))))
            if sides
            else 0
        )
        margin = max(16, int(self.cfg.klt_roi_margin_px), dynamic)
        x0 = max(0, int(math.floor(float(np.min(points[:, 0])))) - margin)
        y0 = max(0, int(math.floor(float(np.min(points[:, 1])))) - margin)
        x1 = min(w, int(math.ceil(float(np.max(points[:, 0])))) + margin + 1)
        y1 = min(h, int(math.ceil(float(np.max(points[:, 1])))) + margin + 1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return previous, current, points.copy(), np.array([0.0, 0.0], np.float32)
        origin = np.array([float(x0), float(y0)], dtype=np.float32)
        return (
            np.ascontiguousarray(previous[y0:y1, x0:x1]),
            np.ascontiguousarray(current[y0:y1, x0:x1]),
            points - origin,
            origin,
        )

    def _track_cpu_adaptive(self, previous, current, points):
        run_fb = (self._call_count % max(1, int(self.cfg.klt_fb_check_period))) == 0
        result = self._track_cpu_profile(
            previous,
            current,
            points,
            int(self.cfg.klt_win_size),
            int(self.cfg.klt_max_level),
            int(self.cfg.klt_max_iters),
            float(self.cfg.klt_eps),
            run_fb,
            bool(self.cfg.klt_use_initial_flow),
        )
        if bool(np.all(result[1])):
            return result
        return self._track_cpu_profile(
            previous,
            current,
            points,
            int(self.cfg.klt_robust_win_size),
            int(self.cfg.klt_robust_max_level),
            int(self.cfg.klt_robust_max_iters),
            float(self.cfg.klt_robust_eps),
            True,
            bool(self.cfg.klt_use_initial_flow),
        )

    def _track_cpu_profile(
        self,
        previous,
        current,
        points,
        win_size,
        max_level,
        max_iters,
        eps,
        run_fb,
        use_initial,
    ):
        p0 = points.reshape(-1, 1, 2).astype(np.float32)
        criteria = (
            self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT,
            int(max_iters),
            float(eps),
        )
        flags = 0
        initial = None
        if use_initial and self._last_displacement is not None:
            initial = (points + self._last_displacement).reshape(-1, 1, 2).astype(np.float32)
            flags = int(self.cv2.OPTFLOW_USE_INITIAL_FLOW)
        p1, st1, err1 = self.cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            p0,
            initial,
            winSize=(int(win_size), int(win_size)),
            maxLevel=int(max_level),
            criteria=criteria,
            flags=flags,
            minEigThreshold=1e-4,
        )
        if p1 is None or st1 is None:
            return points.copy(), np.zeros(len(points), bool), np.full(len(points), np.inf)
        next_points = p1.reshape(-1, 2)
        lk = np.asarray(err1, dtype=np.float32).reshape(-1)
        valid = (
            np.asarray(st1).reshape(-1).astype(bool)
            & np.isfinite(lk)
            & (lk <= float(self.cfg.lk_max_error))
        )
        if not run_fb:
            return next_points, valid, np.zeros(len(points), np.float32)
        back, st2, _ = self.cv2.calcOpticalFlowPyrLK(
            current,
            previous,
            p1,
            None,
            winSize=(int(win_size), int(win_size)),
            maxLevel=int(max_level),
            criteria=criteria,
            flags=0,
            minEigThreshold=1e-4,
        )
        if back is None or st2 is None:
            return next_points, np.zeros(len(points), bool), np.full(len(points), np.inf)
        fb = np.linalg.norm(back.reshape(-1, 2) - points, axis=1)
        valid &= (
            np.asarray(st2).reshape(-1).astype(bool)
            & np.isfinite(fb)
            & (fb <= float(self.cfg.fb_max_error_px))
        )
        return next_points, valid, fb

    def _gpu_mat(self):
        ctor = getattr(self.cv2, "cuda_GpuMat", None)
        if ctor is not None:
            return ctor()
        return self.cv2.cuda.GpuMat()

    def _track_cuda(self, previous, current, points):
        gpu_prev = self._gpu_mat()
        gpu_curr = self._gpu_mat()
        gpu_pts = self._gpu_mat()
        gpu_prev.upload(np.asarray(previous, dtype=np.uint8))
        gpu_curr.upload(np.asarray(current, dtype=np.uint8))
        gpu_pts.upload(points.reshape(1, -1, 2).astype(np.float32))
        nxt, status1, error = self._cuda_lk.calc(gpu_prev, gpu_curr, gpu_pts, None)
        next_points = np.asarray(nxt.download(), dtype=np.float32).reshape(-1, 2)
        gpu_next = self._gpu_mat()
        gpu_next.upload(next_points.reshape(1, -1, 2))
        back, status2, _ = self._cuda_lk.calc(gpu_curr, gpu_prev, gpu_next, None)
        back_points = np.asarray(back.download(), dtype=np.float32).reshape(-1, 2)
        fb = np.linalg.norm(back_points - points, axis=1)
        lk = np.zeros(len(points), np.float32)
        if error is not None:
            lk = np.asarray(error.download(), dtype=np.float32).reshape(-1)
        valid = (
            np.asarray(status1.download()).reshape(-1).astype(bool)
            & np.asarray(status2.download()).reshape(-1).astype(bool)
            & np.isfinite(fb)
            & (fb <= float(self.cfg.fb_max_error_px))
            & np.isfinite(lk)
            & (lk <= float(self.cfg.lk_max_error))
        )
        return next_points, valid, fb


# ============================================================================
# PBVS depth resolver
# ============================================================================


class PbvsZResolver:
    def __init__(self, cfg: PbvsZConfig):
        self.cfg = cfg
        self._filtered: Optional[float] = None
        self._last_ns: Optional[int] = None

    def resolve(self, z_raw_m: float, now_ns: int) -> PbvsZState:
        z = _float_or_none(z_raw_m)
        if z is None or z <= 1e-6:
            return PbvsZState(False, self.cfg.mode, z, None, "z_raw_invalid")
        mode = str(self.cfg.mode).strip().lower()
        if mode == "fixed":
            fixed = float(self.cfg.fixed_m)
            if np.isfinite(fixed) and fixed > 1e-6:
                return PbvsZState(True, "fixed", z, fixed, "ok")
            return PbvsZState(True, "measured", z, z, "fixed_invalid_fallback_raw")
        if mode == "filtered":
            if self._filtered is None or self._last_ns is None:
                self._filtered = z
            else:
                dt = max(0.0, (int(now_ns) - int(self._last_ns)) * 1e-9)
                tau = max(float(self.cfg.filter_tau_s), 1e-9)
                alpha = 1.0 - math.exp(-dt / tau)
                self._filtered += alpha * (z - self._filtered)
            self._last_ns = int(now_ns)
            return PbvsZState(True, "filtered", z, float(self._filtered), "ok")
        return PbvsZState(True, "measured", z, z, "ok")


# ============================================================================
# High-rate IDS/OpenCV source
# ============================================================================


class IdsOpenCvCorrectionSourceAdapter:
    """High-rate latest-frame IDS source with asynchronous ArUco correction."""

    def __init__(self, cfg: dict[str, Any], gating: GatingConfig):
        self._cfg = dict(cfg or {})
        self._gating = gating
        self._ids_cfg = dict(self._cfg.get("ids_peak", {}) or {})
        self._aruco_cfg = dict(self._cfg.get("aruco", {}) or {})
        self._calib_cfg = dict(self._cfg.get("calibration", {}) or {})
        self._fast_cfg = load_fast_tracking_config()
        self._z_resolver = PbvsZResolver(load_pbvs_z_config())

        self._started = False
        self._reason = "init"
        self._cv2 = None
        self._producer: Optional[_IdsLatestFrameProducer] = None
        self._detector_executor: Optional[ThreadPoolExecutor] = None
        self._detector_future: Optional[Future] = None
        self._corner_tracker: Optional[SparseCornerTracker] = None
        self._clahe = None

        self._detector = None
        self._aruco_dict = None
        self._aruco_params = None
        self._aruco_dict_name = "DICT_7X7_100"
        self._marker_size_m = 0.10
        self._target_ids = [1]
        self._conf_area_ratio_ref = 0.02
        self._pose_reproj_rmse_conf_px = 4.0
        self._pose_reproj_rmse_max_px = 10.0
        self._pose_reproj_max_err_px = 10.0

        self._k_calib = np.eye(3, dtype=np.float64)
        self._dist = np.zeros((1, 5), dtype=np.float64)
        self._r_lc = np.eye(3, dtype=np.float64)
        self._t_lc = np.zeros((3, 1), dtype=np.float64)
        self._calib_w = 0
        self._calib_h = 0

        self._pbvs_use_marker_center = True
        self._default_target_point_m = np.zeros(3, dtype=np.float64)

        self._last_sequence = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._tracked_corners: Optional[np.ndarray] = None
        self._tracked_marker_id: Optional[int] = None
        self._last_known_corners: Optional[np.ndarray] = None
        self._last_detection_submit_ns = 0
        self._last_full_recovery_ns = 0
        self._scout_index = 0
        self._last_candidate: Optional[CorrectionCandidate] = None
        self._last_detection_mode = "none"
        self._last_detection_roi: Optional[tuple[int, int, int, int]] = None

        self._debug_enable = bool(self._cfg.get("debug_view", False))
        self._debug_lock = Lock()
        self._last_debug_ns = 0
        self._last_debug: dict[str, Any] = {"available": False, "reason": "debug_disabled"}

        self._stats_lock = Lock()
        self._stats: dict[str, Any] = {
            "version": CAMERA_CUDA_VERSION,
            "frames": 0,
            "new_frames": 0,
            "tracked": 0,
            "track_failures": 0,
            "detections_submitted": 0,
            "detections_ok": 0,
            "detection_failures": 0,
            "tracking_backend": "none",
            "last_track_ms": 0.0,
            "last_pose_ms": 0.0,
            "last_detection_ms": 0.0,
        }

    @property
    def started(self) -> bool:
        return bool(self._started)

    @property
    def mode(self) -> str:
        return "ids_opencv"

    def start(self) -> None:
        if self._started:
            return
        self._cv2 = _import_cv2_or_raise()
        calibration_path = str(self._calib_cfg.get("path", "") or "").strip()
        if not calibration_path:
            raise RuntimeError("ids_opencv calibration.path is missing")
        calib = _load_vs_calibration_for_camera(Path(calibration_path))
        self._k_calib = np.asarray(calib["camera_matrix"], np.float64)
        self._dist = np.asarray(calib["dist_coeffs"], np.float64)
        self._r_lc = np.asarray(calib["R_lc"], np.float64).reshape(3, 3)
        self._t_lc = np.asarray(calib["t_lc_m"], np.float64).reshape(3, 1)
        self._calib_w = int(calib["image_width"])
        self._calib_h = int(calib["image_height"])
        self._aruco_dict_name = str(calib.get("aruco_dict_name", "DICT_7X7_100"))
        self._marker_size_m = float(calib.get("marker_size_m", 0.10))
        self._target_ids = [int(v) for v in calib.get("target_marker_ids", [1])]

        if str(self._aruco_cfg.get("dict_name", "")).strip():
            self._aruco_dict_name = str(self._aruco_cfg["dict_name"]).strip()
        marker_size = _float_or_none(self._aruco_cfg.get("marker_size_m"))
        if marker_size is not None and marker_size > 0.0:
            self._marker_size_m = marker_size
        if self._aruco_cfg.get("target_marker_ids") is not None:
            parsed = [int(v) for v in list(self._aruco_cfg.get("target_marker_ids") or [])]
            if parsed:
                self._target_ids = parsed

        self._pbvs_use_marker_center = bool(self._aruco_cfg.get("pbvs_use_marker_center", True))
        configured_target = self._aruco_cfg.get("target_point_marker_m")
        if configured_target is not None:
            try:
                self._default_target_point_m = np.asarray(configured_target, np.float64).reshape(3)
                self._pbvs_use_marker_center = bool(
                    np.linalg.norm(self._default_target_point_m) <= 1e-12
                )
            except Exception:
                self._default_target_point_m = np.zeros(3, np.float64)
        legacy_target = _pbvs_target_point_marker_from_settings()
        if legacy_target is not None:
            self._default_target_point_m = legacy_target
            self._pbvs_use_marker_center = False

        area_ref = _float_or_none(self._aruco_cfg.get("confidence_area_ratio_ref"))
        if area_ref is not None and area_ref > 1e-8:
            self._conf_area_ratio_ref = area_ref
        reproj_conf = _float_or_none(
            self._aruco_cfg.get("pose_reproj_rmse_conf_px", 4.0)
        )
        if reproj_conf is not None and reproj_conf > 0.0:
            self._pose_reproj_rmse_conf_px = reproj_conf
        reproj_rmse_max = _float_or_none(
            self._aruco_cfg.get("pose_reproj_rmse_max_px", 10.0)
        )
        if reproj_rmse_max is not None and reproj_rmse_max > 0.0:
            self._pose_reproj_rmse_max_px = reproj_rmse_max
        reproj_max_err = _float_or_none(
            self._aruco_cfg.get("pose_reproj_max_err_px", 10.0)
        )
        if reproj_max_err is not None and reproj_max_err > 0.0:
            self._pose_reproj_max_err_px = reproj_max_err

        self._detector, self._aruco_dict, self._aruco_params = _init_aruco_detector(
            self._cv2, self._aruco_dict_name
        )
        self._clahe = self._cv2.createCLAHE(
            clipLimit=float(_setting("CV_CUAS_CLAHE_CLIP_LIMIT", 2.0)),
            tileGridSize=(
                int(_setting("CV_CUAS_CLAHE_TILE_X", 8)),
                int(_setting("CV_CUAS_CLAHE_TILE_Y", 8)),
            ),
        )
        self._corner_tracker = SparseCornerTracker(self._cv2, self._fast_cfg)
        self._detector_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aruco-detector"
        )
        self._producer = _IdsLatestFrameProducer(self._ids_cfg)
        self._producer.start()
        self._started = True
        self._reason = "ok"

    def stop(self) -> None:
        self._started = False
        if self._producer is not None:
            self._producer.stop()
        self._producer = None
        if self._detector_future is not None:
            self._detector_future.cancel()
        self._detector_future = None
        if self._detector_executor is not None:
            self._detector_executor.shutdown(wait=False, cancel_futures=True)
        self._detector_executor = None
        self._prev_gray = None
        self._tracked_corners = None
        self._tracked_marker_id = None
        if self._corner_tracker is not None:
            self._corner_tracker.reset_motion_model()
        self._corner_tracker = None

    def get_debug_snapshot(self) -> dict[str, Any]:
        if not self._debug_enable:
            return {"available": False, "reason": "debug_disabled"}
        with self._debug_lock:
            return _clone_debug_payload(self._last_debug)

    def fast_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            out = dict(self._stats)
        if self._producer is not None:
            out["capture"] = self._producer.stats()
        return out

    def read_candidate(self, now_ns: int) -> CorrectionCandidate:
        if not self._started or self._producer is None:
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
        with self._stats_lock:
            self._stats["new_frames"] += 1

        h, w = gray.shape[:2]
        k_runtime, _, _ = _scale_intrinsics_to_runtime(
            self._k_calib, self._calib_w, self._calib_h, w, h
        )

        t_track0 = time.perf_counter_ns()
        tracked_ok = self._track_existing(gray)
        track_ms = (time.perf_counter_ns() - t_track0) * 1e-6
        with self._stats_lock:
            self._stats["last_track_ms"] = float(track_ms)

        # A completed asynchronous detection can correct drift or initialize KLT.
        detection = self._consume_detection_result(gray, ts_capture_ns, packet.sequence)
        if detection:
            tracked_ok = self._tracked_corners is not None

        self._schedule_detection_if_due(gray, ts_capture_ns, packet.sequence, tracked_ok)
        self._prev_gray = gray

        if self._tracked_corners is None or self._tracked_marker_id is None:
            candidate = CorrectionCandidate(
                reliable=False,
                reason="detection_pending" if self._detector_future is not None else "marker_not_found",
                age_s=max((time.monotonic_ns() - ts_capture_ns) * 1e-9, 0.0),
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray, candidate, None, None, None, camera_matrix=k_runtime
            )
            return candidate

        corners = self._tracked_corners.reshape(1, 4, 2).astype(np.float32)
        marker_id = int(self._tracked_marker_id)
        confidence = _compute_area_confidence(
            self._cv2, corners, w, h, self._conf_area_ratio_ref
        )
        if confidence < float(self._gating.conf_min):
            candidate = CorrectionCandidate(
                False,
                "confidence_low",
                confidence=confidence,
                marker_id=marker_id,
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray, candidate, corners, None, None, camera_matrix=k_runtime
            )
            return candidate

        t_pose0 = time.perf_counter_ns()
        rvec, tvec = _estimate_pose_single_marker(
            self._cv2,
            corners,
            float(self._marker_size_m),
            k_runtime,
            self._dist,
        )
        pose_ms = (time.perf_counter_ns() - t_pose0) * 1e-6
        with self._stats_lock:
            self._stats["last_pose_ms"] = float(pose_ms)
        if rvec is None or tvec is None:
            self._invalidate_track()
            candidate = CorrectionCandidate(
                False,
                "pose_estimation_failed",
                confidence=confidence,
                marker_id=marker_id,
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray, candidate, corners, None, None, camera_matrix=k_runtime
            )
            return candidate

        reproj_rmse_px, reproj_max_err_px = _marker_reprojection_errors_px(
            self._cv2,
            corners,
            float(self._marker_size_m),
            rvec,
            tvec,
            k_runtime,
            self._dist,
        )
        marker_center_c = np.asarray(tvec, np.float64).reshape(3)
        if (
            not np.isfinite(reproj_rmse_px)
            or not np.isfinite(reproj_max_err_px)
            or reproj_rmse_px > float(self._pose_reproj_rmse_max_px)
            or reproj_max_err_px > float(self._pose_reproj_max_err_px)
        ):
            candidate = CorrectionCandidate(
                False,
                "pose_reprojection_error",
                confidence=confidence,
                marker_id=marker_id,
                z_m=float(marker_center_c[2]),
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray,
                candidate,
                corners,
                marker_center_c,
                None,
                camera_matrix=k_runtime,
                reproj_rmse_px=reproj_rmse_px,
                reproj_max_err_px=reproj_max_err_px,
            )
            return candidate

        reproj_confidence = math.exp(
            -max(0.0, float(reproj_rmse_px))
            / max(float(self._pose_reproj_rmse_conf_px), 1e-6)
        )
        confidence = float(np.clip(confidence * reproj_confidence, 0.0, 1.0))
        if confidence < float(self._gating.conf_min):
            candidate = CorrectionCandidate(
                False,
                "confidence_low_after_reprojection",
                confidence=confidence,
                marker_id=marker_id,
                z_m=float(marker_center_c[2]),
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray,
                candidate,
                corners,
                marker_center_c,
                None,
                camera_matrix=k_runtime,
                reproj_rmse_px=reproj_rmse_px,
                reproj_max_err_px=reproj_max_err_px,
            )
            return candidate

        rotation_cm, _ = self._cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
        offset_marker_m = self._target_offset_for_marker(marker_id)
        target_c = marker_center_c + rotation_cm @ offset_marker_m
        target_l = _point_c_to_laser(target_c, self._r_lc, self._t_lc)

        z_state = self._z_resolver.resolve(float(target_l[2]), ts_capture_ns)
        if not z_state.ok or z_state.z_used_m is None:
            candidate = CorrectionCandidate(
                False,
                z_state.status,
                confidence=confidence,
                marker_id=marker_id,
                z_m=float(target_l[2]),
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray,
                candidate,
                corners,
                marker_center_c,
                target_c,
                camera_matrix=k_runtime,
                reproj_rmse_px=reproj_rmse_px,
                reproj_max_err_px=reproj_max_err_px,
            )
            return candidate

        target_for_angles = np.asarray(target_l, np.float64).copy()
        target_for_angles[2] = float(z_state.z_used_m)
        angles = _angles_from_point_laser(target_for_angles)
        if angles is None:
            candidate = CorrectionCandidate(
                False,
                "invalid_target_geometry",
                confidence=confidence,
                marker_id=marker_id,
                z_m=float(target_l[2]),
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray,
                candidate,
                corners,
                marker_center_c,
                target_c,
                camera_matrix=k_runtime,
                reproj_rmse_px=reproj_rmse_px,
                reproj_max_err_px=reproj_max_err_px,
            )
            return candidate

        age_s = max((time.monotonic_ns() - ts_capture_ns) * 1e-9, 0.0)
        if float(target_l[2]) < float(self._gating.z_min_m):
            candidate = CorrectionCandidate(
                False,
                "z_below_min",
                confidence=confidence,
                marker_id=marker_id,
                z_m=float(target_l[2]),
                age_s=age_s,
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray,
                candidate,
                corners,
                marker_center_c,
                target_c,
                camera_matrix=k_runtime,
                reproj_rmse_px=reproj_rmse_px,
                reproj_max_err_px=reproj_max_err_px,
            )
            return candidate
        if age_s > float(self._gating.max_age_s):
            candidate = CorrectionCandidate(
                False,
                "sample_stale",
                confidence=confidence,
                marker_id=marker_id,
                z_m=float(target_l[2]),
                age_s=age_s,
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(
                gray,
                candidate,
                corners,
                marker_center_c,
                target_c,
                camera_matrix=k_runtime,
                reproj_rmse_px=reproj_rmse_px,
                reproj_max_err_px=reproj_max_err_px,
            )
            return candidate

        d_az, d_po = [float(v) for v in angles]
        candidate = CorrectionCandidate(
            True,
            "ok",
            d_az_rad=d_az,
            d_po_rad=d_po,
            confidence=confidence,
            marker_id=marker_id,
            z_m=float(target_l[2]),
            age_s=age_s,
            ts_capture_ns=ts_capture_ns,
        )
        self._last_candidate = candidate
        self._reason = "ok"
        self._update_debug(
            gray,
            candidate,
            corners,
            marker_center_c,
            target_c,
            camera_matrix=k_runtime,
            reproj_rmse_px=reproj_rmse_px,
            reproj_max_err_px=reproj_max_err_px,
        )
        return candidate

    def _reuse_last_candidate(self, now_ns: int) -> CorrectionCandidate:
        previous = self._last_candidate
        if previous is None or previous.ts_capture_ns is None:
            return CorrectionCandidate(False, "no_new_frame")
        age_s = max((int(now_ns) - int(previous.ts_capture_ns)) * 1e-9, 0.0)
        if age_s > float(self._gating.max_age_s):
            return CorrectionCandidate(
                False,
                "sample_stale",
                confidence=previous.confidence,
                marker_id=previous.marker_id,
                z_m=previous.z_m,
                age_s=age_s,
                ts_capture_ns=previous.ts_capture_ns,
            )
        return CorrectionCandidate(
            reliable=previous.reliable,
            reason=previous.reason,
            d_az_rad=previous.d_az_rad,
            d_po_rad=previous.d_po_rad,
            confidence=previous.confidence,
            age_s=age_s,
            marker_id=previous.marker_id,
            z_m=previous.z_m,
            ts_capture_ns=previous.ts_capture_ns,
        )

    def _track_existing(self, gray: np.ndarray) -> bool:
        if (
            self._prev_gray is None
            or self._tracked_corners is None
            or self._corner_tracker is None
        ):
            return False
        try:
            new_pts, valid, _fb, backend = self._corner_tracker.track(
                self._prev_gray, gray, self._tracked_corners
            )
            with self._stats_lock:
                self._stats["tracking_backend"] = str(backend)
            if len(new_pts) != 4 or not bool(np.all(valid)):
                raise RuntimeError("KLT point validation failed")
            if not self._valid_quad(self._tracked_corners, new_pts, gray.shape[1], gray.shape[0]):
                raise RuntimeError("KLT quadrilateral validation failed")
            self._tracked_corners = new_pts.astype(np.float32)
            self._last_known_corners = self._tracked_corners.copy()
            with self._stats_lock:
                self._stats["tracked"] += 1
            return True
        except Exception:
            with self._stats_lock:
                self._stats["track_failures"] += 1
            self._invalidate_track(keep_last_known=True)
            return False

    def _valid_quad(self, old: np.ndarray, new: np.ndarray, width: int, height: int) -> bool:
        old = np.asarray(old, np.float32).reshape(4, 2)
        new = np.asarray(new, np.float32).reshape(4, 2)
        if not np.all(np.isfinite(new)):
            return False
        if (
            np.any(new[:, 0] < 0)
            or np.any(new[:, 0] >= int(width))
            or np.any(new[:, 1] < 0)
            or np.any(new[:, 1] >= int(height))
        ):
            return False
        if not bool(self._cv2.isContourConvex(new.reshape(-1, 1, 2))):
            return False
        a0 = abs(float(self._cv2.contourArea(old)))
        a1 = abs(float(self._cv2.contourArea(new)))
        if a0 <= 1e-6 or a1 < float(self._fast_cfg.min_quad_area_px2):
            return False
        ratio = a1 / a0
        if not (
            float(self._fast_cfg.area_ratio_min)
            <= ratio
            <= float(self._fast_cfg.area_ratio_max)
        ):
            return False
        sides = np.linalg.norm(np.roll(new, -1, axis=0) - new, axis=1)
        if np.min(sides) <= 1.0:
            return False
        return bool(
            float(np.max(sides) / np.min(sides))
            <= float(self._fast_cfg.max_side_ratio)
        )

    def _invalidate_track(self, keep_last_known: bool = True) -> None:
        if keep_last_known and self._tracked_corners is not None:
            self._last_known_corners = self._tracked_corners.copy()
        self._tracked_corners = None
        self._tracked_marker_id = None

    def _consume_detection_result(self, gray: np.ndarray, now_ns: int, sequence: int) -> bool:
        future = self._detector_future
        if future is None or not future.done():
            return False
        self._detector_future = None
        try:
            result: _DetectionResult = future.result()
        except Exception:
            with self._stats_lock:
                self._stats["detection_failures"] += 1
            return False
        self._last_detection_mode = str(result.mode)
        self._last_detection_roi = (
            None
            if result.roi_xywh is None
            else tuple(int(value) for value in result.roi_xywh)
        )
        with self._stats_lock:
            self._stats["last_detection_ms"] = float(result.elapsed_ms)
        if not result.ok or result.corners is None or result.marker_id is None:
            with self._stats_lock:
                self._stats["detection_failures"] += 1
            return False
        age_s = max((int(now_ns) - int(result.ts_capture_ns)) * 1e-9, 0.0)
        if age_s > float(self._fast_cfg.detection_result_max_age_s):
            return False

        corners = np.asarray(result.corners, np.float32).reshape(4, 2)
        if int(result.sequence) != int(sequence) and self._corner_tracker is not None:
            propagated, valid, _fb, backend = self._corner_tracker.track(
                result.frame, gray, corners
            )
            with self._stats_lock:
                self._stats["tracking_backend"] = str(backend)
            if len(propagated) != 4 or not bool(np.all(valid)):
                return False
            if not self._valid_quad(corners, propagated, gray.shape[1], gray.shape[0]):
                return False
            corners = propagated.astype(np.float32)

        self._tracked_corners = corners.copy()
        self._last_known_corners = corners.copy()
        self._tracked_marker_id = int(result.marker_id)
        with self._stats_lock:
            self._stats["detections_ok"] += 1
        return True

    def _schedule_detection_if_due(
        self,
        gray: np.ndarray,
        ts_capture_ns: int,
        sequence: int,
        tracked_ok: bool,
    ) -> None:
        if self._detector_executor is None:
            return
        if self._detector_future is not None and not self._detector_future.done():
            return
        rate = self._fast_cfg.redetect_hz if tracked_ok else self._fast_cfg.missing_redetect_hz
        period = _period_ns(rate)
        if period > 0 and int(ts_capture_ns) - int(self._last_detection_submit_ns) < period:
            return

        use_full = False
        if not tracked_ok:
            full_period = _period_ns(self._fast_cfg.full_recovery_hz)
            use_full = (
                self._last_full_recovery_ns == 0
                or full_period <= 0
                or int(ts_capture_ns) - int(self._last_full_recovery_ns) >= full_period
            )

        if tracked_ok and self._tracked_corners is not None:
            roi = self._roi_around_corners(
                self._tracked_corners,
                gray.shape[1],
                gray.shape[0],
                scale=float(self._fast_cfg.detection_roi_scale),
                min_side=int(self._fast_cfg.detection_roi_min_side_px),
            )
            mode = "track_roi"
        elif not use_full and self._last_known_corners is not None:
            roi = self._roi_around_corners(
                self._last_known_corners,
                gray.shape[1],
                gray.shape[0],
                scale=float(self._fast_cfg.missing_detection_roi_scale),
                min_side=int(self._fast_cfg.missing_detection_roi_min_side_px),
            )
            mode = "recover_roi"
        elif use_full:
            roi = (0, 0, int(gray.shape[1]), int(gray.shape[0]))
            mode = "full_recovery"
            self._last_full_recovery_ns = int(ts_capture_ns)
        else:
            roi = self._next_scout_roi(gray.shape[1], gray.shape[0])
            mode = "scout"

        frame_ref = gray
        self._detector_future = self._detector_executor.submit(
            self._detect_job,
            frame_ref,
            int(ts_capture_ns),
            int(sequence),
            roi,
            mode,
        )
        self._last_detection_submit_ns = int(ts_capture_ns)
        with self._stats_lock:
            self._stats["detections_submitted"] += 1

    def _detect_job(
        self,
        frame: np.ndarray,
        ts_capture_ns: int,
        sequence: int,
        roi: tuple[int, int, int, int],
        mode: str,
    ) -> _DetectionResult:
        t0 = time.perf_counter_ns()
        x, y, w, h = [int(v) for v in roi]
        crop = np.ascontiguousarray(frame[y : y + h, x : x + w])
        detect_img = self._preprocess_detection(crop)
        corners, ids, _rejected = _detect_markers(
            self._cv2, detect_img, self._detector, self._aruco_dict, self._aruco_params
        )
        elapsed = (time.perf_counter_ns() - t0) * 1e-6
        if ids is None or len(ids) == 0:
            return _DetectionResult(
                False,
                "marker_not_found",
                frame,
                ts_capture_ns,
                sequence,
                mode=mode,
                roi_xywh=roi,
                elapsed_ms=elapsed,
            )
        shifted = _shift_marker_corners(corners, x, y)
        idx = _choose_target_marker(self._cv2, shifted, ids, self._target_ids)
        if idx is None:
            return _DetectionResult(
                False,
                "target_marker_not_found",
                frame,
                ts_capture_ns,
                sequence,
                mode=mode,
                roi_xywh=roi,
                elapsed_ms=elapsed,
            )
        return _DetectionResult(
            True,
            "ok",
            frame,
            ts_capture_ns,
            sequence,
            corners=np.asarray(shifted[idx], np.float32).reshape(4, 2),
            marker_id=int(ids[idx].item()),
            mode=mode,
            roi_xywh=roi,
            elapsed_ms=elapsed,
        )

    def _preprocess_detection(self, gray: np.ndarray) -> np.ndarray:
        mode = str(self._fast_cfg.detection_preprocess).strip().lower()
        src = np.asarray(gray, np.uint8)
        if mode == "clahe" and self._clahe is not None:
            return self._clahe.apply(src)
        if mode == "stretch":
            sample = src[::4, ::4] if min(src.shape[:2]) >= 64 else src
            low = float(np.percentile(sample, 2.0))
            high = float(np.percentile(sample, 98.0))
            if np.isfinite(low) and np.isfinite(high) and high > low + 1e-6:
                return np.clip((src.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        return src

    def _roi_around_corners(
        self,
        corners: np.ndarray,
        image_w: int,
        image_h: int,
        scale: float,
        min_side: int,
    ) -> tuple[int, int, int, int]:
        pts = np.asarray(corners, np.float64).reshape(-1, 2)
        min_xy = np.min(pts, axis=0)
        max_xy = np.max(pts, axis=0)
        center = 0.5 * (min_xy + max_xy)
        side = max(float(min_side), float(scale) * max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 1.0))
        side_i = max(1, min(int(round(side)), int(image_w), int(image_h)))
        x = int(round(float(center[0]) - 0.5 * side_i))
        y = int(round(float(center[1]) - 0.5 * side_i))
        x = max(0, min(x, int(image_w) - side_i))
        y = max(0, min(y, int(image_h) - side_i))
        return x, y, side_i, side_i

    def _next_scout_roi(self, image_w: int, image_h: int) -> tuple[int, int, int, int]:
        cols = max(1, int(self._fast_cfg.scout_grid_cols))
        rows = max(1, int(self._fast_cfg.scout_grid_rows))
        overlap = min(max(float(self._fast_cfg.scout_overlap), 0.0), 0.95)
        tile_w = int(math.ceil(image_w / max(cols - overlap * (cols - 1), 1.0)))
        tile_h = int(math.ceil(image_h / max(rows - overlap * (rows - 1), 1.0)))
        step_x = max(1, int(round(tile_w * (1.0 - overlap))))
        step_y = max(1, int(round(tile_h * (1.0 - overlap))))
        tiles = []
        for row in range(rows):
            for col in range(cols):
                x = min(col * step_x, max(0, image_w - tile_w))
                y = min(row * step_y, max(0, image_h - tile_h))
                tiles.append((int(x), int(y), min(tile_w, image_w - x), min(tile_h, image_h - y)))
        if not tiles:
            return 0, 0, image_w, image_h
        if self._last_known_corners is not None:
            center = np.mean(self._last_known_corners, axis=0)
        else:
            center = np.array([0.5 * image_w, 0.5 * image_h])
        tiles.sort(key=lambda r: (r[0] + 0.5 * r[2] - center[0]) ** 2 + (r[1] + 0.5 * r[3] - center[1]) ** 2)
        roi = tiles[self._scout_index % len(tiles)]
        self._scout_index = (self._scout_index + 1) % len(tiles)
        return roi

    def _target_offset_for_marker(self, marker_id: int) -> np.ndarray:
        # Preferred configuration used by cv_CUAS_verified_offsets_reacquisition_v2.py:
        # marker-local OpenCV coordinates, expressed in centimetres.
        name = f"CV_CUAS_ARUCO_{int(marker_id)}_TARGET_OFFSET_CM"
        raw = _setting(name, None)
        if raw is not None:
            try:
                value = np.asarray(raw, np.float64).reshape(3) * 0.01
                if np.all(np.isfinite(value)):
                    return value
            except Exception:
                pass
        if not self._pbvs_use_marker_center:
            return np.asarray(self._default_target_point_m, np.float64).reshape(3)
        return np.zeros(3, np.float64)

    def _update_debug(
        self,
        gray: np.ndarray,
        candidate: CorrectionCandidate,
        corners: Optional[np.ndarray],
        marker_center_c: Optional[np.ndarray],
        target_c: Optional[np.ndarray],
        *,
        camera_matrix: Optional[np.ndarray] = None,
        reproj_rmse_px: Optional[float] = None,
        reproj_max_err_px: Optional[float] = None,
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
            preview = self._cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=self._cv2.INTER_AREA)
        else:
            preview = np.array(gray, copy=True)
        scaled_corners = None
        if corners is not None:
            scaled_corners = np.asarray(corners, np.float32) * float(scale)
        marker_center_pixel = None
        if scaled_corners is not None:
            marker_center_pixel = np.mean(
                np.asarray(scaled_corners, np.float64).reshape(4, 2), axis=0
            )
        target_pixel = None
        if target_c is not None and camera_matrix is not None:
            projected = _project_camera_point_to_pixel(
                self._cv2,
                target_c,
                camera_matrix,
                self._dist,
            )
            if projected is not None:
                target_pixel = projected * float(scale)
        detection_roi = None
        if self._last_detection_roi is not None:
            x, y, roi_w, roi_h = self._last_detection_roi
            detection_roi = tuple(
                int(round(float(value) * float(scale)))
                for value in (x, y, roi_w, roi_h)
            )
        payload = {
            "available": True,
            "version": CAMERA_CUDA_VERSION,
            "reason": candidate.reason,
            "frame_gray": preview,
            "preview_scale": float(scale),
            "ts_capture_ns": candidate.ts_capture_ns,
            "marker_corners_px": scaled_corners,
            "marker_center_pixel_px": marker_center_pixel,
            "pbvs_target_pixel_px": target_pixel,
            "marker_id": candidate.marker_id,
            "marker_size_m": float(self._marker_size_m),
            "confidence": candidate.confidence,
            "z_m": candidate.z_m,
            "d_az_rad": candidate.d_az_rad,
            "d_po_rad": candidate.d_po_rad,
            "reproj_rmse_px": reproj_rmse_px,
            "reproj_max_err_px": reproj_max_err_px,
            "p_c_marker_center_m": None if marker_center_c is None else np.asarray(marker_center_c, np.float64),
            "p_c_target_m": None if target_c is None else np.asarray(target_c, np.float64),
            "detection_mode": self._last_detection_mode,
            "detection_roi_xywh": detection_roi,
            "tracking_backend": None if self._corner_tracker is None else self._corner_tracker.backend,
            "stats": self.fast_stats(),
        }
        with self._debug_lock:
            self._last_debug = payload


# Alias retained for code written against the previous split adapter.
FastIdsOpenCvCorrectionSourceAdapter = IdsOpenCvCorrectionSourceAdapter


# ============================================================================
# Multi-marker ArUco tracking and rigid target-point fusion
# ============================================================================
#
# Generalizes the single-marker pipeline above to N ArUco markers mounted on a
# rigid target, fusing their individually tracked positions into one PBVS
# target point (geometric center + rigid target-frame offset, with a
# single-marker fallback when only one of N markers is visible). This reuses
# the same IDS acquisition (_IdsLatestFrameProducer), KLT backend
# (SparseCornerTracker) and low-level ArUco/PnP helpers as the single-marker
# adapter above; only ROI planning, detection and target fusion are new.
#
# The single-marker IdsOpenCvCorrectionSourceAdapter class above is untouched
# by this section; _build_source() routes to MultiArucoCorrectionSourceAdapter
# only when more than one target_marker_ids entry is configured, so existing
# single-marker deployments keep their exact current behavior.


@dataclass(frozen=True)
class ArucoMarkerSpec:
    marker_id: int
    dict_name: str
    marker_size_m: float


@dataclass(frozen=True)
class MultiMarkerTrackingConfig:
    preprocess_mode: str = "raw"  # raw | stretch | clahe
    stretch_percentile_low: float = 2.0
    stretch_percentile_high: float = 98.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    track_roi_expand_on_miss: float = 1.6
    track_roi_misses_to_scout: int = 3
    track_extra_scout_tiles_per_frame: int = 0
    partial_extra_scout_tiles_per_frame: int = 2
    partial_hint_search_enable: bool = True
    scout_tiles_per_frame: int = 2
    scout_order: str = "last_center_first"  # last_center_first | center_out
    confidence_area_ratio_ref: float = 0.00048
    pose_reproj_rmse_conf_px: float = 4.0
    pose_reproj_rmse_max_px: float = 10.0
    velocity_filter_alpha: float = 0.35
    covariance_min_std_m: float = 0.003
    covariance_depth_fraction_min: float = 0.01
    covariance_confidence_floor: float = 0.02
    single_marker_fallback_enable: bool = True
    single_marker_confidence_scale: float = 0.65
    single_marker_covariance_scale: float = 4.0
    target_offset_std_cm: float = 0.5
    target_midpoint_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_frame_reference_marker_id: int = 0


def load_multi_marker_tracking_config() -> MultiMarkerTrackingConfig:
    return MultiMarkerTrackingConfig(
        preprocess_mode=str(_setting("CV_CUAS_PREPROCESS_MODE", "raw")).strip().lower(),
        stretch_percentile_low=float(_setting("CV_CUAS_STRETCH_PERCENTILE_LOW", 2.0)),
        stretch_percentile_high=float(_setting("CV_CUAS_STRETCH_PERCENTILE_HIGH", 98.0)),
        clahe_clip_limit=float(_setting("CV_CUAS_CLAHE_CLIP_LIMIT", 2.0)),
        clahe_tile_grid=(
            int(_setting("CV_CUAS_CLAHE_TILE_X", 8)),
            int(_setting("CV_CUAS_CLAHE_TILE_Y", 8)),
        ),
        track_roi_expand_on_miss=float(_setting("CV_CUAS_TRACK_ROI_EXPAND_ON_MISS", 1.6)),
        track_roi_misses_to_scout=int(_setting("CV_CUAS_TRACK_ROI_MISSES_TO_SCOUT", 3)),
        track_extra_scout_tiles_per_frame=int(
            _setting("CV_CUAS_TRACK_EXTRA_SCOUT_TILES_PER_FRAME", 0)
        ),
        partial_extra_scout_tiles_per_frame=int(
            _setting("CV_CUAS_PARTIAL_EXTRA_SCOUT_TILES_PER_FRAME", 2)
        ),
        partial_hint_search_enable=_bool_setting(
            "CV_CUAS_PARTIAL_HINT_SEARCH_ENABLE", True
        ),
        scout_tiles_per_frame=int(_setting("CV_CUAS_SCOUT_TILES_PER_FRAME", 2)),
        scout_order=str(_setting("CV_CUAS_SCOUT_ORDER", "last_center_first")).strip().lower(),
        confidence_area_ratio_ref=float(
            _setting("CV_CUAS_CONFIDENCE_AREA_RATIO_REF", 0.00048)
        ),
        pose_reproj_rmse_conf_px=float(_setting("CV_CUAS_POSE_REPROJ_RMSE_CONF_PX", 4.0)),
        pose_reproj_rmse_max_px=float(_setting("CV_CUAS_POSE_REPROJ_RMSE_MAX_PX", 10.0)),
        velocity_filter_alpha=float(_setting("CV_CUAS_VELOCITY_FILTER_ALPHA", 0.35)),
        covariance_min_std_m=float(_setting("CV_CUAS_COVARIANCE_MIN_STD_M", 0.003)),
        covariance_depth_fraction_min=float(
            _setting("CV_CUAS_COVARIANCE_DEPTH_FRACTION_MIN", 0.01)
        ),
        covariance_confidence_floor=float(
            _setting("CV_CUAS_COVARIANCE_CONFIDENCE_FLOOR", 0.02)
        ),
        single_marker_fallback_enable=_bool_setting(
            "CV_CUAS_SINGLE_MARKER_FALLBACK_ENABLE", True
        ),
        single_marker_confidence_scale=float(
            _setting("CV_CUAS_SINGLE_MARKER_CONFIDENCE_SCALE", 0.65)
        ),
        single_marker_covariance_scale=float(
            _setting("CV_CUAS_SINGLE_MARKER_COVARIANCE_SCALE", 4.0)
        ),
        target_offset_std_cm=float(_setting("CV_CUAS_TARGET_OFFSET_STD_CM", 0.5)),
        target_midpoint_offset_cm=_setting_vec3(
            "CV_CUAS_TARGET_MIDPOINT_OFFSET_CM", (0.0, 0.0, 0.0)
        ),
        target_frame_reference_marker_id=int(
            _setting("CV_CUAS_TARGET_FRAME_REFERENCE_ARUCO_ID", 0)
        ),
    )


def _setting_vec3(name: str, default: Sequence[float]) -> tuple[float, float, float]:
    raw = _setting(name, default)
    try:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{name} must be a sequence of three numbers") from exc
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _multi_marker_offset_cm(marker_id: int) -> tuple[float, float, float]:
    return _setting_vec3(f"CV_CUAS_ARUCO_{int(marker_id)}_TARGET_OFFSET_CM", (0.0, 0.0, 0.0))


def _multi_marker_frame_rpy_deg(marker_id: int) -> tuple[float, float, float]:
    return _setting_vec3(f"CV_CUAS_ARUCO_{int(marker_id)}_TARGET_FRAME_RPY_DEG", (0.0, 0.0, 0.0))


@dataclass(frozen=True)
class ArucoObservation:
    marker_id: int
    dict_name: str
    position_camera_m: np.ndarray
    velocity_camera_mps: np.ndarray
    covariance_camera_m2: np.ndarray
    confidence: float
    reproj_rmse_px: float
    rvec_cm: np.ndarray
    corners_px: np.ndarray
    ts_capture_ns: int


@dataclass(frozen=True)
class MultiMarkerTargetPoint:
    position_camera_m: np.ndarray
    covariance_camera_m2: np.ndarray
    confidence: float
    contributing_markers: tuple[int, ...]
    mode: str
    reference_marker_id: Optional[int]


@dataclass
class _MultiTrackState:
    position_camera_m: Optional[np.ndarray] = None
    velocity_camera_mps: Optional[np.ndarray] = None
    ts_ns: Optional[int] = None


@dataclass(frozen=True)
class _MultiAsyncDetectionResult:
    frame: np.ndarray
    ts_capture_ns: int
    sequence: int
    markers: dict[int, dict[str, Any]]
    mode: str
    tile_count: int
    elapsed_ms: float


def _rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = r / theta
    kx, ky, kz = axis
    k_mat = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * k_mat + (1.0 - math.cos(theta)) * (k_mat @ k_mat)


def _rotation_from_rpy_deg(rpy_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _multi_marker_position_covariance_m2(
    marker_corners: np.ndarray,
    p_c_m: np.ndarray,
    camera_matrix: np.ndarray,
    confidence: float,
    reproj_rmse_px: float,
    cfg: MultiMarkerTrackingConfig,
) -> np.ndarray:
    p_c = np.asarray(p_c_m, dtype=np.float64).reshape(3)
    z = max(abs(float(p_c[2])), 0.05)
    fx = max(abs(float(camera_matrix[0, 0])), 1.0)
    fy = max(abs(float(camera_matrix[1, 1])), 1.0)
    pts = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
    side_px = max(float(np.mean(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1))), 1.0)
    conf = max(float(confidence), float(cfg.covariance_confidence_floor))
    sigma_px = max(float(reproj_rmse_px), 0.25) / math.sqrt(conf)
    sigma_x = max(float(cfg.covariance_min_std_m), z * sigma_px / fx)
    sigma_y = max(float(cfg.covariance_min_std_m), z * sigma_px / fy)
    sigma_z = max(
        float(cfg.covariance_min_std_m),
        float(cfg.covariance_depth_fraction_min) * z,
        2.0 * z * sigma_px / side_px,
    )
    return np.diag([sigma_x * sigma_x, sigma_y * sigma_y, sigma_z * sigma_z]).astype(np.float64)


def _transform_covariance_to_laser(cov_camera_m2: np.ndarray, r_lc: np.ndarray) -> np.ndarray:
    r = np.asarray(r_lc, dtype=np.float64).reshape(3, 3)
    c = np.asarray(cov_camera_m2, dtype=np.float64).reshape(3, 3)
    return (r @ c @ r.T).astype(np.float64)


def compute_multi_marker_target_point(
    observations: Sequence[ArucoObservation],
    expected_marker_count: int,
    cfg: MultiMarkerTrackingConfig,
) -> Optional[MultiMarkerTargetPoint]:
    """Fuse per-marker observations of a rigid multi-marker target into one point.

    Complete observation (all configured markers visible): geometric center of
    all marker positions, offset by a rigid target-frame vector oriented
    through one configured reference marker's pose (rotations are never
    averaged across markers).

    Partial observation: with exactly one of N markers visible and fallback
    enabled, use that marker's own center plus its marker-local offset, with
    reduced confidence and inflated covariance. Any other partial count (only
    possible when more than two markers are configured) is not fused, matching
    the discontinuity-avoidance rule that motivated requiring either the full
    set or a single, explicitly-scaled fallback marker.
    """
    valid = [obs for obs in observations if np.all(np.isfinite(obs.position_camera_m))]
    if not valid:
        return None
    expected = max(1, int(expected_marker_count))
    complete = len(valid) >= expected
    offset_sigma_m = max(0.0, float(cfg.target_offset_std_cm)) * 0.01
    offset_cov = np.eye(3, dtype=np.float64) * (offset_sigma_m ** 2)
    marker_ids = tuple(int(obs.marker_id) for obs in valid)

    if complete:
        positions = np.vstack([obs.position_camera_m.reshape(1, 3) for obs in valid])
        p_mid = np.mean(positions, axis=0)
        ref_obs = next(
            (o for o in valid if int(o.marker_id) == int(cfg.target_frame_reference_marker_id)),
            valid[0],
        )
        ref_id = int(ref_obs.marker_id)
        r_camera_marker = _rotation_from_rotvec(ref_obs.rvec_cm)
        r_marker_target = _rotation_from_rpy_deg(_multi_marker_frame_rpy_deg(ref_id))
        r_camera_target = r_camera_marker @ r_marker_target
        offset_target_m = np.asarray(cfg.target_midpoint_offset_cm, dtype=np.float64).reshape(3) * 0.01
        p_c = p_mid + r_camera_target @ offset_target_m
        n = float(len(valid))
        cov_c = sum(
            (obs.covariance_camera_m2 for obs in valid), np.zeros((3, 3), dtype=np.float64)
        ) / (n * n)
        cov_c = cov_c + offset_cov
        confidence = float(np.clip(np.mean([obs.confidence for obs in valid]), 0.0, 1.0))
        mode = f"midpoint_target_frame_ref_{ref_id}"
        reference_marker_id: Optional[int] = ref_id
    else:
        if not cfg.single_marker_fallback_enable or len(valid) != 1:
            return None
        obs = valid[0]
        marker_id = int(obs.marker_id)
        offset_local_m = np.asarray(_multi_marker_offset_cm(marker_id), dtype=np.float64).reshape(3) * 0.01
        r_camera_marker = _rotation_from_rotvec(obs.rvec_cm)
        p_c = np.asarray(obs.position_camera_m, dtype=np.float64).reshape(3) + r_camera_marker @ offset_local_m
        cov_c = (
            np.asarray(obs.covariance_camera_m2, dtype=np.float64).reshape(3, 3)
            * max(1.0, float(cfg.single_marker_covariance_scale))
            + offset_cov
        )
        confidence = float(
            np.clip(float(obs.confidence) * float(cfg.single_marker_confidence_scale), 0.0, 1.0)
        )
        mode = f"single_marker_{marker_id}_offset"
        reference_marker_id = marker_id

    return MultiMarkerTargetPoint(
        position_camera_m=np.asarray(p_c, dtype=np.float64).reshape(3),
        covariance_camera_m2=np.asarray(cov_c, dtype=np.float64).reshape(3, 3),
        confidence=confidence,
        contributing_markers=marker_ids,
        mode=mode,
        reference_marker_id=reference_marker_id,
    )


class MultiMarkerTiledPreprocessor:
    """ROI planner with robust partial-marker reacquisition for N markers.

    In ``track`` mode all currently-visible markers are covered by one ROI.
    In ``partial`` mode (some but not all configured markers visible) that ROI
    stays wide while rotating scout tiles, a predicted missing-marker hint and
    the ordinary scout rotation run alongside it to reacquire the rest of the
    configured set. In ``scout``/``recover`` no reliable ROI exists yet, so the
    image is searched tile by tile.
    """

    def __init__(
        self,
        cv2_mod,
        fast_cfg: FastTrackingConfig,
        multi_cfg: MultiMarkerTrackingConfig,
    ):
        self._cv2 = cv2_mod
        self._fast_cfg = fast_cfg
        self._cfg = multi_cfg
        self._mode = "scout"
        self._miss_count = 0
        self._roi_xywh: Optional[tuple[int, int, int, int]] = None
        self._last_center_xy: Optional[tuple[float, float]] = None
        self._scout_idx = 0
        self._partial_search_hint_xy: Optional[tuple[float, float]] = None
        self._clahe = cv2_mod.createCLAHE(
            clipLimit=max(0.1, float(multi_cfg.clahe_clip_limit)),
            tileGridSize=(
                max(1, int(multi_cfg.clahe_tile_grid[0])),
                max(1, int(multi_cfg.clahe_tile_grid[1])),
            ),
        )

    @property
    def mode(self) -> str:
        return str(self._mode)

    def set_partial_search_hint(self, hint_xy: Optional[tuple[float, float]]) -> None:
        if hint_xy is None:
            self._partial_search_hint_xy = None
            return
        try:
            hx, hy = float(hint_xy[0]), float(hint_xy[1])
        except Exception:
            self._partial_search_hint_xy = None
            return
        self._partial_search_hint_xy = (hx, hy) if np.isfinite(hx) and np.isfinite(hy) else None

    def prepare(self, gray: np.ndarray) -> list["_MultiDetectionTile"]:
        src = self._ensure_gray_u8(gray)
        h, w = src.shape[:2]

        if self._mode in {"track", "partial", "recover"} and self._roi_xywh is not None:
            x, y, rw, rh = self._clamp_roi(self._roi_xywh, w, h)
            out = [
                _MultiDetectionTile(
                    gray=self._preprocess(src[y:y + rh, x:x + rw]),
                    offset_x=x, offset_y=y, state=self._mode, roi_xywh=(x, y, rw, rh),
                )
            ]
            extra_count = (
                max(1, int(self._cfg.partial_extra_scout_tiles_per_frame))
                if self._mode == "partial"
                else max(0, int(self._cfg.track_extra_scout_tiles_per_frame))
            )
            if extra_count > 0:
                scout_tiles = self._scout_tiles(
                    w, h,
                    prefer_partial_hint=bool(
                        self._mode == "partial" and self._cfg.partial_hint_search_enable
                    ),
                )
                checked = added = 0
                if scout_tiles:
                    start_idx = self._scout_idx % len(scout_tiles)
                    while checked < len(scout_tiles) and added < extra_count:
                        idx = (start_idx + checked) % len(scout_tiles)
                        checked += 1
                        sx, sy, sw, sh = scout_tiles[idx]
                        if self._roi_overlap_ratio((x, y, rw, rh), (sx, sy, sw, sh)) > 0.85:
                            continue
                        out.append(
                            _MultiDetectionTile(
                                gray=self._preprocess(src[sy:sy + sh, sx:sx + sw]),
                                offset_x=sx, offset_y=sy,
                                state=f"{self._mode}_scout", roi_xywh=(sx, sy, sw, sh),
                            )
                        )
                        added += 1
                    self._scout_idx = (start_idx + max(1, checked)) % len(scout_tiles)
            return out

        scout_tiles = self._scout_tiles(w, h)
        if not scout_tiles:
            scout_tiles = [(0, 0, int(w), int(h))]
        count = max(1, min(int(self._cfg.scout_tiles_per_frame), len(scout_tiles)))
        out: list[_MultiDetectionTile] = []
        for i in range(count):
            idx = (self._scout_idx + i) % len(scout_tiles)
            x, y, rw, rh = scout_tiles[idx]
            out.append(
                _MultiDetectionTile(
                    gray=self._preprocess(src[y:y + rh, x:x + rw]),
                    offset_x=x, offset_y=y, state="scout", roi_xywh=(x, y, rw, rh),
                )
            )
        self._scout_idx = (self._scout_idx + count) % len(scout_tiles)
        return out

    def update(
        self,
        img_shape: tuple[int, int],
        detected_corners: Sequence[np.ndarray],
        detected_marker_count: int,
        expected_marker_count: int,
    ) -> None:
        img_h, img_w = int(img_shape[0]), int(img_shape[1])
        detected_count = max(0, int(detected_marker_count))
        expected_count = max(1, int(expected_marker_count))

        if detected_corners:
            pts = np.concatenate(
                [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in detected_corners], axis=0
            )
            x0, y0 = np.min(pts, axis=0)
            x1, y1 = np.max(pts, axis=0)
            self._last_center_xy = (float(0.5 * (x0 + x1)), float(0.5 * (y0 + y1)))
            complete = detected_count >= expected_count
            if complete:
                self._mode = "track"
                self._partial_search_hint_xy = None
                scale = float(self._fast_cfg.detection_roi_scale)
                min_side = int(self._fast_cfg.detection_roi_min_side_px)
                self._scout_idx = 0
            else:
                self._mode = "partial"
                scale = float(self._fast_cfg.missing_detection_roi_scale)
                min_side = int(self._fast_cfg.missing_detection_roi_min_side_px)
            self._roi_xywh = self._roi_from_bbox(
                (float(x0), float(y0), float(x1), float(y1)), img_w, img_h, scale, min_side
            )
            self._miss_count = 0
            return

        self._miss_count += 1
        if self._mode in {"track", "partial", "recover"} and self._roi_xywh is not None:
            if self._miss_count < int(self._cfg.track_roi_misses_to_scout):
                self._roi_xywh = self._expand_roi(self._roi_xywh, img_w, img_h)
                self._mode = "recover"
                return
        self._mode = "scout"
        total = max(1, int(self._fast_cfg.scout_grid_cols) * int(self._fast_cfg.scout_grid_rows))
        self._scout_idx = (self._scout_idx + max(1, int(self._cfg.scout_tiles_per_frame))) % total

    def _ensure_gray_u8(self, gray: np.ndarray) -> np.ndarray:
        src = np.asarray(gray)
        if src.ndim == 3:
            src = self._cv2.cvtColor(src, self._cv2.COLOR_BGR2GRAY)
        if src.dtype != np.uint8:
            src = np.clip(src, 0, 255).astype(np.uint8)
        return src

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        mode = str(self._cfg.preprocess_mode or "raw").strip().lower()
        src = self._ensure_gray_u8(gray)
        if mode in {"", "raw"}:
            return src
        if mode == "stretch":
            low = float(self._cfg.stretch_percentile_low)
            high = float(self._cfg.stretch_percentile_high)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = 2.0, 98.0
            p_low, p_high = np.percentile(src, (low, high))
            if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low + 1e-6:
                return src
            out = (src.astype(np.float32) - p_low) * (255.0 / (p_high - p_low))
            return np.clip(out, 0, 255).astype(np.uint8)
        if mode == "clahe":
            return self._clahe.apply(src)
        return src

    def _scout_tiles(
        self, img_w: int, img_h: int, prefer_partial_hint: bool = False
    ) -> list[tuple[int, int, int, int]]:
        cols = max(1, int(self._fast_cfg.scout_grid_cols))
        rows = max(1, int(self._fast_cfg.scout_grid_rows))
        overlap = min(max(float(self._fast_cfg.scout_overlap), 0.0), 0.95)
        tile_w = int(math.ceil(float(img_w) / (float(cols) - overlap * float(cols - 1))))
        tile_h = int(math.ceil(float(img_h) / (float(rows) - overlap * float(rows - 1))))
        step_x = max(1, int(round(tile_w * (1.0 - overlap))))
        step_y = max(1, int(round(tile_h * (1.0 - overlap))))
        tiles: list[tuple[int, int, int, int]] = []
        seen = set()
        for row in range(rows):
            for col in range(cols):
                x = min(col * step_x, max(0, img_w - tile_w))
                y = min(row * step_y, max(0, img_h - tile_h))
                item = (int(x), int(y), min(tile_w, img_w - x), min(tile_h, img_h - y))
                if item[2] > 0 and item[3] > 0 and item not in seen:
                    seen.add(item)
                    tiles.append(item)
        if prefer_partial_hint and self._partial_search_hint_xy is not None:
            ref = self._partial_search_hint_xy
        elif str(self._cfg.scout_order).strip().lower() == "center_out" or self._last_center_xy is None:
            ref = (0.5 * img_w, 0.5 * img_h)
        else:
            ref = self._last_center_xy
        tiles.sort(key=lambda r: (r[0] + 0.5 * r[2] - ref[0]) ** 2 + (r[1] + 0.5 * r[3] - ref[1]) ** 2)
        return tiles

    def _roi_from_bbox(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        img_w: int,
        img_h: int,
        scale: float,
        min_side_px: int,
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox_xyxy
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        side = max(float(min_side_px), float(scale) * max(max(1.0, x1 - x0), max(1.0, y1 - y0)))
        rw, rh = int(round(min(float(img_w), side))), int(round(min(float(img_h), side)))
        return self._clamp_roi(
            (int(round(cx - 0.5 * rw)), int(round(cy - 0.5 * rh)), rw, rh), img_w, img_h
        )

    def _expand_roi(self, roi_xywh, img_w: int, img_h: int):
        x, y, rw, rh = roi_xywh
        cx, cy = x + 0.5 * rw, y + 0.5 * rh
        scale = max(1.0, float(self._cfg.track_roi_expand_on_miss))
        new_w = int(round(min(float(img_w), rw * scale)))
        new_h = int(round(min(float(img_h), rh * scale)))
        return self._clamp_roi(
            (int(round(cx - 0.5 * new_w)), int(round(cy - 0.5 * new_h)), new_w, new_h), img_w, img_h
        )

    @staticmethod
    def _clamp_roi(roi_xywh, img_w: int, img_h: int):
        x, y, rw, rh = [int(v) for v in roi_xywh]
        rw, rh = max(1, min(rw, img_w)), max(1, min(rh, img_h))
        x, y = max(0, min(x, img_w - rw)), max(0, min(y, img_h - rh))
        return x, y, rw, rh

    @staticmethod
    def _roi_overlap_ratio(a_xywh, b_xywh) -> float:
        ax, ay, aw, ah = [int(v) for v in a_xywh]
        bx, by, bw, bh = [int(v) for v in b_xywh]
        iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        ih = max(0, min(ay + ah, by + bh) - max(ay, by))
        return float(iw * ih) / max(float(min(aw * ah, bw * bh)), 1.0)


@dataclass(frozen=True)
class _MultiDetectionTile:
    gray: np.ndarray
    offset_x: int
    offset_y: int
    state: str
    roi_xywh: tuple[int, int, int, int]


class MultiArucoCorrectionSourceAdapter:
    """High-rate latest-frame IDS source tracking N rigid-mounted ArUco markers.

    Mirrors IdsOpenCvCorrectionSourceAdapter's architecture (latest-only IDS
    acquisition, sparse KLT propagation, single-worker asynchronous ArUco
    detection with delayed-result KLT propagation) generalized to a configured
    set of marker IDs: every currently-tracked marker's four corners are KLT
    tracked together in one combined ROI, MultiMarkerTiledPreprocessor plans
    the detection ROI/scout tiles, and per-marker PnP/confidence/covariance/
    velocity feed compute_multi_marker_target_point() to produce one fused
    PBVS target exposed through the same CorrectionCandidate contract as the
    single-marker adapter.
    """

    def __init__(self, cfg: dict[str, Any], gating: GatingConfig):
        self._cfg = dict(cfg or {})
        self._gating = gating
        self._ids_cfg = dict(self._cfg.get("ids_peak", {}) or {})
        self._aruco_cfg = dict(self._cfg.get("aruco", {}) or {})
        self._calib_cfg = dict(self._cfg.get("calibration", {}) or {})
        self._fast_cfg = load_fast_tracking_config()
        self._track_cfg = load_multi_marker_tracking_config()

        self._started = False
        self._cv2 = None
        self._producer: Optional[_IdsLatestFrameProducer] = None
        self._detector_executor: Optional[ThreadPoolExecutor] = None
        self._detector_future: Optional[Future] = None
        self._corner_tracker: Optional[SparseCornerTracker] = None
        self._preprocessor: Optional[MultiMarkerTiledPreprocessor] = None

        self._detector = None
        self._aruco_dict = None
        self._aruco_params = None
        self._marker_specs: tuple[ArucoMarkerSpec, ...] = ()

        self._k_calib = np.eye(3, dtype=np.float64)
        self._dist = np.zeros((1, 5), dtype=np.float64)
        self._r_lc = np.eye(3, dtype=np.float64)
        self._t_lc = np.zeros((3, 1), dtype=np.float64)
        self._calib_w = 0
        self._calib_h = 0

        self._last_sequence = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._tracked_corners: dict[int, np.ndarray] = {}
        self._last_marker_centers_px: dict[int, np.ndarray] = {}
        self._track_states: dict[int, _MultiTrackState] = {}
        self._last_detection_submit_ns = 0
        self._last_candidate: Optional[CorrectionCandidate] = None

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
            "tracking_backend": "none",
        }

    @property
    def started(self) -> bool:
        return bool(self._started)

    @property
    def mode(self) -> str:
        return "ids_opencv"

    def start(self) -> None:
        if self._started:
            return
        self._cv2 = _import_cv2_or_raise()
        calibration_path = str(self._calib_cfg.get("path", "") or "").strip()
        if not calibration_path:
            raise RuntimeError("ids_opencv calibration.path is missing")
        calib = _load_vs_calibration_for_camera(Path(calibration_path))
        self._k_calib = np.asarray(calib["camera_matrix"], np.float64)
        self._dist = np.asarray(calib["dist_coeffs"], np.float64)
        self._r_lc = np.asarray(calib["R_lc"], np.float64).reshape(3, 3)
        self._t_lc = np.asarray(calib["t_lc_m"], np.float64).reshape(3, 1)
        self._calib_w = int(calib["image_width"])
        self._calib_h = int(calib["image_height"])
        dict_name = str(calib.get("aruco_dict_name", "DICT_7X7_100"))
        marker_size_m = float(calib.get("marker_size_m", 0.10))
        target_ids = [int(v) for v in calib.get("target_marker_ids", [])]

        if str(self._aruco_cfg.get("dict_name", "")).strip():
            dict_name = str(self._aruco_cfg["dict_name"]).strip()
        cfg_size = _float_or_none(self._aruco_cfg.get("marker_size_m"))
        if cfg_size is not None and cfg_size > 0.0:
            marker_size_m = cfg_size
        if self._aruco_cfg.get("target_marker_ids") is not None:
            parsed = [int(v) for v in list(self._aruco_cfg.get("target_marker_ids") or [])]
            if parsed:
                target_ids = parsed
        if len(target_ids) < 2:
            raise ValueError(
                "MultiArucoCorrectionSourceAdapter requires 2+ target_marker_ids; "
                "configure a single id to use IdsOpenCvCorrectionSourceAdapter instead"
            )
        seen: set[int] = set()
        specs: list[ArucoMarkerSpec] = []
        for marker_id in target_ids:
            if marker_id in seen:
                raise ValueError(f"Duplicate target_marker_ids entry: {marker_id}")
            seen.add(marker_id)
            specs.append(
                ArucoMarkerSpec(marker_id=marker_id, dict_name=dict_name, marker_size_m=marker_size_m)
            )
        self._marker_specs = tuple(specs)

        self._detector, self._aruco_dict, self._aruco_params = _init_aruco_detector(self._cv2, dict_name)
        self._preprocessor = MultiMarkerTiledPreprocessor(self._cv2, self._fast_cfg, self._track_cfg)
        self._corner_tracker = SparseCornerTracker(self._cv2, self._fast_cfg)
        self._detector_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aruco-detector-multi"
        )
        self._producer = _IdsLatestFrameProducer(self._ids_cfg)
        self._producer.start()
        self._started = True

    def stop(self) -> None:
        self._started = False
        if self._producer is not None:
            self._producer.stop()
        self._producer = None
        if self._detector_future is not None:
            self._detector_future.cancel()
        self._detector_future = None
        if self._detector_executor is not None:
            self._detector_executor.shutdown(wait=False, cancel_futures=True)
        self._detector_executor = None
        self._prev_gray = None
        self._tracked_corners = {}
        if self._corner_tracker is not None:
            self._corner_tracker.reset_motion_model()
        self._corner_tracker = None

    def get_debug_snapshot(self) -> dict[str, Any]:
        if not self._debug_enable:
            return {"available": False, "reason": "debug_disabled"}
        with self._debug_lock:
            return _clone_debug_payload(self._last_debug)

    def fast_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            out = dict(self._stats)
        if self._producer is not None:
            out["capture"] = self._producer.stats()
        return out

    def read_candidate(self, now_ns: int) -> CorrectionCandidate:
        if not self._started or self._producer is None:
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
        with self._stats_lock:
            self._stats["new_frames"] += 1
        h, w = gray.shape[:2]
        k_runtime, _, _ = _scale_intrinsics_to_runtime(
            self._k_calib, self._calib_w, self._calib_h, w, h
        )

        tracked_corners = self._track_existing(gray)
        detected_corners = self._consume_detection_result(gray, ts_capture_ns, packet.sequence)
        tracked_corners.update(detected_corners)

        self._schedule_detection_if_due(gray, ts_capture_ns, packet.sequence, set(tracked_corners.keys()))

        assert self._preprocessor is not None
        self._preprocessor.update(
            (h, w),
            list(tracked_corners.values()),
            detected_marker_count=len(tracked_corners),
            expected_marker_count=len(self._marker_specs),
        )
        self._update_partial_search_hint(tracked_corners, w, h)

        self._tracked_corners = {
            k: np.asarray(v, np.float32).reshape(4, 2).copy() for k, v in tracked_corners.items()
        }
        self._prev_gray = gray

        observations: list[ArucoObservation] = []
        for spec in self._marker_specs:
            quad = tracked_corners.get(spec.marker_id)
            if quad is None:
                continue
            obs = self._make_observation(spec, quad, k_runtime, w, h, ts_capture_ns)
            if obs is not None:
                observations.append(obs)

        target = compute_multi_marker_target_point(
            observations, expected_marker_count=len(self._marker_specs), cfg=self._track_cfg
        )
        if target is None:
            candidate = CorrectionCandidate(
                False,
                "target_point_unavailable" if observations else "marker_not_found",
                ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(gray, candidate, observations, None, k_runtime)
            return candidate

        if target.confidence < float(self._gating.conf_min):
            candidate = CorrectionCandidate(
                False, "confidence_low", confidence=target.confidence,
                marker_id=target.reference_marker_id, ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(gray, candidate, observations, target, k_runtime)
            return candidate

        target_l = _point_c_to_laser(target.position_camera_m, self._r_lc, self._t_lc)
        z_m = float(target_l[2])
        if z_m < float(self._gating.z_min_m):
            candidate = CorrectionCandidate(
                False, "z_below_min", confidence=target.confidence,
                marker_id=target.reference_marker_id, z_m=z_m, ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(gray, candidate, observations, target, k_runtime)
            return candidate

        angles = _angles_from_point_laser(target_l)
        age_s = max((time.monotonic_ns() - ts_capture_ns) * 1e-9, 0.0)
        if angles is None:
            candidate = CorrectionCandidate(
                False, "invalid_target_geometry", confidence=target.confidence,
                marker_id=target.reference_marker_id, z_m=z_m, age_s=age_s, ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(gray, candidate, observations, target, k_runtime)
            return candidate

        if age_s > float(self._gating.max_age_s):
            candidate = CorrectionCandidate(
                False, "sample_stale", confidence=target.confidence,
                marker_id=target.reference_marker_id, z_m=z_m, age_s=age_s, ts_capture_ns=ts_capture_ns,
            )
            self._last_candidate = candidate
            self._update_debug(gray, candidate, observations, target, k_runtime)
            return candidate

        d_az, d_po = [float(v) for v in angles]
        candidate = CorrectionCandidate(
            True, "ok", d_az_rad=d_az, d_po_rad=d_po, confidence=target.confidence,
            marker_id=target.reference_marker_id, z_m=z_m, age_s=age_s, ts_capture_ns=ts_capture_ns,
        )
        self._last_candidate = candidate
        self._update_debug(gray, candidate, observations, target, k_runtime)
        return candidate

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
        return CorrectionCandidate(
            reliable=previous.reliable, reason=previous.reason, d_az_rad=previous.d_az_rad,
            d_po_rad=previous.d_po_rad, confidence=previous.confidence, age_s=age_s,
            marker_id=previous.marker_id, z_m=previous.z_m, ts_capture_ns=previous.ts_capture_ns,
        )

    def _track_existing(self, gray: np.ndarray) -> dict[int, np.ndarray]:
        if (
            self._prev_gray is None
            or not self._tracked_corners
            or self._prev_gray.shape != gray.shape
            or self._corner_tracker is None
        ):
            return {}
        keys = list(self._tracked_corners.keys())
        quads = [np.asarray(self._tracked_corners[k], np.float32).reshape(4, 2) for k in keys]
        all_points = np.vstack(quads)
        new_points, valid, _fb, backend = self._corner_tracker.track(self._prev_gray, gray, all_points)
        with self._stats_lock:
            self._stats["tracking_backend"] = str(backend)
        tracked: dict[int, np.ndarray] = {}
        cursor = 0
        h, w = gray.shape[:2]
        for marker_id, old_quad in zip(keys, quads):
            new_quad = new_points[cursor:cursor + 4]
            quad_valid = valid[cursor:cursor + 4]
            cursor += 4
            if len(new_quad) != 4 or not bool(np.all(quad_valid)):
                continue
            if not self._valid_quad(old_quad, new_quad, w, h):
                continue
            tracked[marker_id] = new_quad.astype(np.float32)
        return tracked

    def _valid_quad(self, old: np.ndarray, new: np.ndarray, width: int, height: int) -> bool:
        old = np.asarray(old, np.float32).reshape(4, 2)
        new = np.asarray(new, np.float32).reshape(4, 2)
        if not np.all(np.isfinite(new)):
            return False
        if (
            np.any(new[:, 0] < 0) or np.any(new[:, 0] >= int(width))
            or np.any(new[:, 1] < 0) or np.any(new[:, 1] >= int(height))
        ):
            return False
        if not bool(self._cv2.isContourConvex(new.reshape(-1, 1, 2))):
            return False
        a0 = abs(float(self._cv2.contourArea(old)))
        a1 = abs(float(self._cv2.contourArea(new)))
        if a0 <= 1e-6 or a1 < float(self._fast_cfg.min_quad_area_px2):
            return False
        ratio = a1 / a0
        if not (float(self._fast_cfg.area_ratio_min) <= ratio <= float(self._fast_cfg.area_ratio_max)):
            return False
        sides = np.linalg.norm(np.roll(new, -1, axis=0) - new, axis=1)
        if np.min(sides) <= 1.0:
            return False
        return bool(float(np.max(sides) / np.min(sides)) <= float(self._fast_cfg.max_side_ratio))

    def _consume_detection_result(self, gray: np.ndarray, now_ns: int, sequence: int) -> dict[int, np.ndarray]:
        future = self._detector_future
        if future is None or not future.done():
            return {}
        self._detector_future = None
        try:
            result: _MultiAsyncDetectionResult = future.result()
        except Exception:
            with self._stats_lock:
                self._stats["detection_failures"] += 1
            return {}
        with self._stats_lock:
            self._stats["last_detection_ms"] = float(result.elapsed_ms)
        if not result.markers:
            with self._stats_lock:
                self._stats["detection_failures"] += 1
            return {}
        age_s = max((int(now_ns) - int(result.ts_capture_ns)) * 1e-9, 0.0)
        if age_s > float(self._fast_cfg.detection_result_max_age_s):
            return {}
        if int(result.sequence) == int(sequence) or self._corner_tracker is None:
            with self._stats_lock:
                self._stats["detections_ok"] += 1
            return {mid: np.asarray(v["corners"], np.float32) for mid, v in result.markers.items()}

        # Detector frame is older than the current frame: propagate its corners
        # forward with KLT before accepting them, like the single-marker path.
        keys = list(result.markers.keys())
        quads = [np.asarray(result.markers[k]["corners"], np.float32).reshape(4, 2) for k in keys]
        old_points = np.vstack(quads)
        propagated, valid, _fb, backend = self._corner_tracker.track(result.frame, gray, old_points)
        with self._stats_lock:
            self._stats["tracking_backend"] = str(backend)
        out: dict[int, np.ndarray] = {}
        cursor = 0
        h, w = gray.shape[:2]
        for marker_id, old_quad in zip(keys, quads):
            quad = propagated[cursor:cursor + 4]
            quad_valid = valid[cursor:cursor + 4]
            cursor += 4
            if len(quad) != 4 or not bool(np.all(quad_valid)):
                continue
            if not self._valid_quad(old_quad, quad, w, h):
                continue
            out[marker_id] = quad.astype(np.float32)
        if out:
            with self._stats_lock:
                self._stats["detections_ok"] += 1
        return out

    def _schedule_detection_if_due(
        self, gray: np.ndarray, ts_capture_ns: int, sequence: int, tracked_ids: set[int]
    ) -> None:
        if self._detector_executor is None or self._preprocessor is None:
            return
        if self._detector_future is not None and not self._detector_future.done():
            return
        expected = len(self._marker_specs)
        visible = len(tracked_ids)
        rate = self._fast_cfg.redetect_hz if visible >= expected else self._fast_cfg.missing_redetect_hz
        period = _period_ns(rate)
        if period > 0 and int(ts_capture_ns) - int(self._last_detection_submit_ns) < period:
            return

        tiles = self._preprocessor.prepare(gray)
        plans = [(t.offset_x, t.offset_y, t.gray) for t in tiles]
        mode = self._preprocessor.mode
        self._detector_future = self._detector_executor.submit(
            self._detect_job, gray, int(ts_capture_ns), int(sequence), plans, mode
        )
        self._last_detection_submit_ns = int(ts_capture_ns)
        with self._stats_lock:
            self._stats["detections_submitted"] += 1

    def _detect_job(
        self,
        frame: np.ndarray,
        ts_capture_ns: int,
        sequence: int,
        plans: list[tuple[int, int, np.ndarray]],
        mode: str,
    ) -> _MultiAsyncDetectionResult:
        t0 = time.perf_counter_ns()
        found: dict[int, dict[str, Any]] = {}
        wanted = {spec.marker_id for spec in self._marker_specs}
        for offset_x, offset_y, tile_gray in plans:
            corners, ids, _rejected = _detect_markers(
                self._cv2, tile_gray, self._detector, self._aruco_dict, self._aruco_params
            )
            if ids is None or len(ids) == 0:
                continue
            shifted = _shift_marker_corners(corners, offset_x, offset_y)
            for i, item in enumerate(shifted):
                marker_id = int(ids[i].item())
                if marker_id not in wanted:
                    continue
                quad = np.asarray(item, np.float32).reshape(4, 2)
                area = abs(float(self._cv2.contourArea(quad)))
                prev = found.get(marker_id)
                if prev is None or area > float(prev["area_px"]):
                    found[marker_id] = {"corners": quad, "area_px": area}
        elapsed = (time.perf_counter_ns() - t0) * 1e-6
        return _MultiAsyncDetectionResult(
            frame=frame, ts_capture_ns=int(ts_capture_ns), sequence=int(sequence),
            markers=found, mode=str(mode), tile_count=len(plans), elapsed_ms=elapsed,
        )

    def _make_observation(
        self,
        spec: ArucoMarkerSpec,
        corners_px: np.ndarray,
        camera_matrix: np.ndarray,
        img_w: int,
        img_h: int,
        ts_capture_ns: int,
    ) -> Optional[ArucoObservation]:
        confidence_area = _compute_area_confidence(
            self._cv2, corners_px, img_w, img_h, float(self._track_cfg.confidence_area_ratio_ref)
        )
        rvec, tvec = _estimate_pose_single_marker(
            self._cv2, corners_px, float(spec.marker_size_m), camera_matrix, self._dist
        )
        if rvec is None or tvec is None:
            return None
        p_c = np.asarray(tvec, np.float64).reshape(3)
        if not np.all(np.isfinite(p_c)) or float(p_c[2]) < float(self._gating.z_min_m):
            return None
        reproj_rmse_px, _max_err = _marker_reprojection_errors_px(
            self._cv2, corners_px, float(spec.marker_size_m), rvec, p_c, camera_matrix, self._dist
        )
        if not np.isfinite(reproj_rmse_px) or reproj_rmse_px > float(self._track_cfg.pose_reproj_rmse_max_px):
            return None
        reproj_conf = float(
            math.exp(-max(0.0, reproj_rmse_px) / max(float(self._track_cfg.pose_reproj_rmse_conf_px), 1e-6))
        )
        confidence = float(np.clip(float(confidence_area) * reproj_conf, 0.0, 1.0))
        if confidence < float(self._gating.conf_min):
            return None
        cov_c = _multi_marker_position_covariance_m2(
            corners_px, p_c, camera_matrix, confidence, reproj_rmse_px, self._track_cfg
        )
        vel_c = self._update_velocity(int(spec.marker_id), p_c, ts_capture_ns)
        return ArucoObservation(
            marker_id=int(spec.marker_id), dict_name=str(spec.dict_name),
            position_camera_m=p_c.astype(np.float64), velocity_camera_mps=vel_c,
            covariance_camera_m2=cov_c, confidence=confidence, reproj_rmse_px=float(reproj_rmse_px),
            rvec_cm=np.asarray(rvec, np.float64).reshape(3),
            corners_px=np.asarray(corners_px, np.float64).reshape(4, 2),
            ts_capture_ns=int(ts_capture_ns),
        )

    def _update_velocity(self, marker_id: int, p_c: np.ndarray, ts_capture_ns: int) -> np.ndarray:
        state = self._track_states.setdefault(marker_id, _MultiTrackState())
        p_c = np.asarray(p_c, np.float64).reshape(3)
        vel = np.zeros(3, np.float64)
        if state.position_camera_m is not None and state.ts_ns is not None:
            dt = (int(ts_capture_ns) - int(state.ts_ns)) * 1e-9
            if dt > 1e-6:
                raw = (p_c - state.position_camera_m) / dt
                alpha = min(max(float(self._track_cfg.velocity_filter_alpha), 0.0), 1.0)
                prev = state.velocity_camera_mps if state.velocity_camera_mps is not None else raw
                vel = alpha * raw + (1.0 - alpha) * prev
        state.position_camera_m = p_c
        state.velocity_camera_mps = vel
        state.ts_ns = int(ts_capture_ns)
        return vel.astype(np.float64)

    def _update_partial_search_hint(
        self, tracked_corners: dict[int, np.ndarray], img_w: int, img_h: int
    ) -> None:
        """Predict the missing marker in image space from the last rigid layout."""
        visible_ids = list(tracked_corners.keys())
        hint_xy: Optional[tuple[float, float]] = None
        if len(visible_ids) == 1:
            visible_id = visible_ids[0]
            current_visible = np.mean(
                np.asarray(tracked_corners[visible_id], np.float64).reshape(4, 2), axis=0
            )
            previous_visible = self._last_marker_centers_px.get(visible_id)
            predictions: list[np.ndarray] = []
            for spec in self._marker_specs:
                if spec.marker_id == visible_id or spec.marker_id in tracked_corners:
                    continue
                previous_missing = self._last_marker_centers_px.get(spec.marker_id)
                if previous_missing is None:
                    continue
                predicted = (
                    current_visible + (previous_missing - previous_visible)
                    if previous_visible is not None
                    else previous_missing.copy()
                )
                predicted = np.asarray(predicted, np.float64).reshape(2)
                predicted[0] = np.clip(predicted[0], 0.0, max(0.0, img_w - 1.0))
                predicted[1] = np.clip(predicted[1], 0.0, max(0.0, img_h - 1.0))
                predictions.append(predicted)
            if predictions:
                hint = np.mean(np.vstack(predictions), axis=0)
                hint_xy = (float(hint[0]), float(hint[1]))
        if self._preprocessor is not None:
            self._preprocessor.set_partial_search_hint(hint_xy)
        for marker_id, corners in tracked_corners.items():
            self._last_marker_centers_px[marker_id] = np.mean(
                np.asarray(corners, np.float64).reshape(4, 2), axis=0
            )

    def _update_debug(
        self,
        gray: np.ndarray,
        candidate: CorrectionCandidate,
        observations: list[ArucoObservation],
        target: Optional[MultiMarkerTargetPoint],
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
                gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=self._cv2.INTER_AREA,
            )
        else:
            preview = np.array(gray, copy=True)

        markers_payload = []
        for obs in observations:
            markers_payload.append({
                "marker_id": int(obs.marker_id),
                "corners_px": np.asarray(obs.corners_px, np.float32) * float(scale),
                "confidence": float(obs.confidence),
                "z_m": float(obs.position_camera_m[2]),
                "reproj_rmse_px": float(obs.reproj_rmse_px),
            })

        target_pixel = None
        target_payload = None
        if target is not None:
            target_payload = {
                "position_camera_m": np.asarray(target.position_camera_m, np.float64),
                "confidence": float(target.confidence),
                "contributing_markers": target.contributing_markers,
                "mode": target.mode,
                "reference_marker_id": target.reference_marker_id,
            }
            projected = _project_camera_point_to_pixel(
                self._cv2, target.position_camera_m, camera_matrix, self._dist
            )
            if projected is not None:
                target_pixel = projected * float(scale)

        # Where the laser's own boresight axis should land in this frame,
        # per the loaded calibration's R_lc/t_lc_m. Projected at the tracked
        # target's own laser-frame range when there is one (that is the range
        # the beam would actually be hitting), otherwise at the configured
        # nominal range -- which one was used is reported in the payload, so
        # the overlay can never present a nominal-range guess as a measurement.
        laser_axis_pixel = None
        laser_axis_range_m = None
        laser_axis_range_source = "none"
        if target is not None:
            target_l = _point_c_to_laser(target.position_camera_m, self._r_lc, self._t_lc)
            if np.isfinite(target_l[2]) and float(target_l[2]) > 1e-6:
                laser_axis_range_m = float(target_l[2])
                laser_axis_range_source = "target"
        if laser_axis_range_m is None and float(self._fast_cfg.debug_laser_range_m) > 0.0:
            laser_axis_range_m = float(self._fast_cfg.debug_laser_range_m)
            laser_axis_range_source = "nominal"
        if laser_axis_range_m is not None:
            projected = _project_camera_point_to_pixel(
                self._cv2,
                _laser_axis_point_in_camera(self._r_lc, self._t_lc, laser_axis_range_m),
                camera_matrix,
                self._dist,
            )
            if projected is not None:
                laser_axis_pixel = projected * float(scale)

        payload = {
            "available": True,
            "version": CAMERA_CUDA_VERSION,
            "reason": candidate.reason,
            "frame_gray": preview,
            "preview_scale": float(scale),
            "ts_capture_ns": candidate.ts_capture_ns,
            "markers": markers_payload,
            "expected_marker_ids": tuple(int(s.marker_id) for s in self._marker_specs),
            "target_point": target_payload,
            "pbvs_target_pixel_px": target_pixel,
            "principal_point_px": np.asarray(
                [float(camera_matrix[0, 2]), float(camera_matrix[1, 2])], np.float64
            ) * float(scale),
            "laser_axis_pixel_px": laser_axis_pixel,
            "laser_axis_range_m": laser_axis_range_m,
            "laser_axis_range_source": laser_axis_range_source,
            "confidence": candidate.confidence,
            "z_m": candidate.z_m,
            "d_az_rad": candidate.d_az_rad,
            "d_po_rad": candidate.d_po_rad,
            "tile_state": self._preprocessor.mode if self._preprocessor is not None else "none",
            "tracking_backend": self._corner_tracker.backend if self._corner_tracker is not None else "none",
            "stats": self.fast_stats(),
        }
        with self._debug_lock:
            self._last_debug = payload


# ============================================================================
# Application service and facade
# ============================================================================


class CameraApplicationService:
    def __init__(self, source: CorrectionSourcePort, gating: GatingConfig):
        self._source = source
        self._gating = gating
        self._accepted = 0
        self._rejected = 0
        self._last_reason = "init"
        self._last_update_ns: Optional[int] = None
        self._last_sample = self._reject(time.monotonic_ns(), "init")

    def start(self) -> None:
        self._source.start()
        self._last_sample = self._reject(time.monotonic_ns(), "source_started")

    def stop(self) -> None:
        self._source.stop()
        self._last_sample = self._reject(time.monotonic_ns(), "source_stopped")

    def process_once(self, now_ns: Optional[int] = None) -> CameraCorrection:
        if now_ns is None:
            now_ns = time.monotonic_ns()
        candidate = self._source.read_candidate(int(now_ns))
        sample = self._validate_candidate(candidate, int(now_ns))
        self._last_sample = sample
        self._last_update_ns = int(now_ns)
        self._last_reason = sample.reason
        return sample

    def latest(self) -> CameraCorrection:
        return self._last_sample

    def health(self, running: bool) -> CameraHealth:
        return CameraHealth(
            running=bool(running),
            mode=self._source.mode,
            source_started=self._source.started,
            last_reason=self._last_reason,
            last_update_ns=self._last_update_ns,
            accepted_samples=int(self._accepted),
            rejected_samples=int(self._rejected),
        )

    def _validate_candidate(self, candidate: CorrectionCandidate, now_ns: int) -> CameraCorrection:
        if not candidate.reliable:
            self._rejected += 1
            return self._reject(
                now_ns,
                candidate.reason,
                candidate.confidence,
                candidate.age_s,
                candidate.marker_id,
                candidate.z_m,
                candidate.ts_capture_ns,
            )
        d_az = float(candidate.d_az_rad)
        d_po = float(candidate.d_po_rad)
        if not np.isfinite(d_az) or not np.isfinite(d_po):
            self._rejected += 1
            return self._reject(now_ns, "delta_not_finite")
        max_abs = float(self._gating.max_abs_angle_rad)
        if max_abs > 0.0 and (abs(d_az) > max_abs or abs(d_po) > max_abs):
            self._rejected += 1
            return self._reject(
                now_ns,
                "delta_above_max_abs_angle",
                candidate.confidence,
                candidate.age_s,
                candidate.marker_id,
                candidate.z_m,
                candidate.ts_capture_ns,
            )
        self._accepted += 1
        return CameraCorrection(
            True,
            "ok",
            d_az,
            d_po,
            candidate.confidence,
            candidate.age_s,
            candidate.marker_id,
            candidate.z_m,
            candidate.ts_capture_ns,
            int(now_ns),
        )

    @staticmethod
    def _reject(
        now_ns: int,
        reason: str,
        confidence: Optional[float] = None,
        age_s: Optional[float] = None,
        marker_id: Optional[int] = None,
        z_m: Optional[float] = None,
        ts_capture_ns: Optional[int] = None,
    ) -> CameraCorrection:
        return CameraCorrection(
            False,
            str(reason),
            0.0,
            0.0,
            confidence,
            age_s,
            marker_id,
            z_m,
            ts_capture_ns,
            int(now_ns),
        )


class CameraAPI:
    """Stable facade consumed by pointing.py."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._app = CameraApplicationService(_build_source(cfg), cfg.gating)
        self._running = False
        self._lock = Lock()
        self._worker: Optional[Thread] = None
        self._stop_event = Event()
        self._latest_sample: Optional[CameraCorrection] = None

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "CameraAPI":
        return cls(load_camera_config_yaml(yaml_path))

    def start(self) -> None:
        if not self.cfg.enable:
            self._running = False
            return
        self._app.start()
        self._running = True
        self._stop_event.clear()
        with self._lock:
            self._latest_sample = self._app.latest()
        if self.cfg.runtime.background_worker:
            hz = max(float(self.cfg.runtime.worker_hz), 1.0)
            self._worker = Thread(
                target=self._worker_loop,
                args=(1.0 / hz,),
                name="camera-api-worker",
                daemon=True,
            )
            self._worker.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._worker = None
        self._app.stop()
        with self._lock:
            self._latest_sample = self._app.latest()

    def get_latest(self) -> CameraCorrection:
        if not self.cfg.enable:
            return CameraApplicationService._reject(time.monotonic_ns(), "camera_disabled")
        if self.cfg.runtime.background_worker:
            with self._lock:
                return self._latest_sample if self._latest_sample is not None else self._app.latest()
        return self._app.process_once(time.monotonic_ns())

    def health(self) -> CameraHealth:
        return self._app.health(self._running)

    def get_debug_snapshot(self) -> dict[str, Any]:
        source = getattr(self._app, "_source", None)
        getter = getattr(source, "get_debug_snapshot", None)
        if not callable(getter):
            return {"available": False, "reason": "debug_not_supported"}
        try:
            return dict(getter() or {})
        except Exception as exc:
            return {"available": False, "reason": f"debug_get_failed:{exc}"}

    def fast_stats(self) -> dict[str, Any]:
        source = getattr(self._app, "_source", None)
        getter = getattr(source, "fast_stats", None)
        return dict(getter() or {}) if callable(getter) else {}

    def _worker_loop(self, period_s: float) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            sample = self._app.process_once(time.monotonic_ns())
            with self._lock:
                self._latest_sample = sample
            next_tick += float(period_s)
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                self._stop_event.wait(timeout=sleep_s)
            else:
                # Do not try to catch up by running multiple stale iterations.
                next_tick = time.monotonic()
                time.sleep(0.0)


def from_yaml(yaml_path: str | Path) -> CameraAPI:
    return CameraAPI.from_yaml(yaml_path)


# ============================================================================
# YAML config loader
# ============================================================================


def load_camera_config_yaml(yaml_path: str | Path) -> CameraConfig:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"camera config yaml not found: {path}")
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load camera config yaml") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    node = raw.get("camera", raw)
    base_dir = path.resolve().parent

    gating_node = dict(node.get("gating", {}) or {})
    runtime_node = dict(node.get("runtime", {}) or {})
    ids_node = dict(node.get("ids_peak", {}) or {})
    aruco_node = dict(node.get("aruco", {}) or {})
    calib_node = dict(node.get("calibration", {}) or {})
    csv_node = dict(node.get("csv", {}) or {})

    if "gentl_cti_path" in ids_node:
        ids_node["gentl_cti_path"] = _resolve_optional_path(ids_node["gentl_cti_path"], base_dir)
    if "path" in calib_node:
        calib_node["path"] = _resolve_optional_path(calib_node["path"], base_dir)

    return CameraConfig(
        enable=bool(node.get("enable", False)),
        mode=str(node.get("provider_mode", "csv")).strip().lower(),
        csv_path=_resolve_optional_path(csv_node.get("path", ""), base_dir),
        gating=GatingConfig(
            max_age_s=float(gating_node.get("max_age_s", 0.25)),
            conf_min=float(gating_node.get("confidence_min", 0.15)),
            z_min_m=float(gating_node.get("z_min_m", 0.25)),
            max_abs_angle_rad=float(gating_node.get("max_abs_angle_rad", 0.0)),
        ),
        runtime=RuntimeConfig(
            background_worker=bool(runtime_node.get("background_worker", True)),
            worker_hz=float(runtime_node.get("worker_hz", 70.0)),
        ),
        ids_opencv={
            "ids_peak": ids_node,
            "aruco": aruco_node,
            "calibration": calib_node,
            "debug_view": bool(node.get("debug_view", False)),
        },
    )


def _build_source(cfg: CameraConfig) -> CorrectionSourcePort:
    if cfg.mode == "csv":
        return CsvCorrectionSourceAdapter(cfg.csv_path, cfg.gating)
    if cfg.mode == "ids_opencv":
        ids_cfg = cfg.ids_opencv or {}
        # Mode selection is driven by the YAML aruco.target_marker_ids list, not by
        # calibration.yaml's own target_marker_ids (calibration is only loaded once
        # the chosen adapter's start() runs). Configure aruco.target_marker_ids
        # explicitly with 2+ ids to enable multi-marker rigid target-point fusion.
        aruco_cfg = dict(ids_cfg.get("aruco", {}) or {})
        target_ids = aruco_cfg.get("target_marker_ids")
        if target_ids is not None and len(list(target_ids)) > 1:
            return MultiArucoCorrectionSourceAdapter(ids_cfg, cfg.gating)
        return IdsOpenCvCorrectionSourceAdapter(ids_cfg, cfg.gating)
    raise ValueError(f"Unsupported camera provider_mode: {cfg.mode}")


# ============================================================================
# OpenCV / IDS helpers
# ============================================================================


def _import_cv2_or_raise():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) is required for ids_opencv mode") from exc
    return cv2


def _load_ids_modules():
    try:
        from ids_peak import ids_peak as ids_peak_mod
        import ids_peak.ids_peak_ipl_extension as ids_ipl_mod
    except Exception as exc:
        raise RuntimeError("ids_peak Python modules are required") from exc
    return ids_peak_mod, ids_ipl_mod


def _try_set_node(nodemap, name: str, setter) -> bool:
    try:
        setter(nodemap.FindNode(name))
        return True
    except Exception:
        return False


def _find_first_node(nodemap, names: tuple[str, ...]):
    for name in names:
        try:
            return nodemap.FindNode(name)
        except Exception:
            continue
    return None


def _set_exposure_time_us(nodemap, target_us: float) -> bool:
    node = _find_first_node(nodemap, ("ExposureTime", "ExposureTimeAbs"))
    if node is None:
        return False
    value = float(target_us)
    try:
        value = max(float(node.Minimum()), value)
    except Exception:
        pass
    try:
        value = min(float(node.Maximum()), value)
    except Exception:
        pass
    try:
        node.SetValue(value)
        return True
    except Exception:
        return False


def _set_acquisition_frame_rate_target_hz(nodemap, target_hz: float) -> bool:
    enabled = (
        _try_set_node(nodemap, "AcquisitionFrameRateTargetEnable", lambda n: n.SetValue(True))
        or _try_set_node(nodemap, "AcquisitionFrameRateTargetEnable", lambda n: n.SetCurrentEntry("On"))
    )
    node = _find_first_node(nodemap, ("AcquisitionFrameRateTarget", "AcquisitionFrameRate"))
    if node is None:
        return False
    value = float(target_hz)
    try:
        value = max(float(node.Minimum()), value)
    except Exception:
        pass
    try:
        value = min(float(node.Maximum()), value)
    except Exception:
        pass
    try:
        node.SetValue(value)
        return bool(enabled or node is not None)
    except Exception:
        return False


def _buffer_to_mono8_numpy(ids_ipl_mod, buffer) -> np.ndarray:
    img = None
    for name in ("BufferToImage", "ConvertToImage", "BufferToPeakImage", "BufferToIplImage"):
        if hasattr(ids_ipl_mod, name):
            try:
                img = getattr(ids_ipl_mod, name)(buffer)
                break
            except Exception:
                pass
    if img is None:
        raise RuntimeError("Failed to convert IDS buffer")
    converted = img
    for name in ("ConvertTo", "ConvertToPixelFormat", "Convert"):
        if hasattr(img, name):
            try:
                converted = getattr(img, name)("Mono8")
                break
            except Exception:
                pass
    frame = converted.get_numpy()
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    return np.asarray(frame)


def _apply_aruco_detector_tuning_from_settings(cv2_mod, params) -> None:
    refinement = str(_setting("CV_ARUCO_CORNER_REFINEMENT", "SUBPIX")).strip().upper()
    enum_name = f"CORNER_REFINE_{refinement}"
    if hasattr(params, "cornerRefinementMethod") and hasattr(cv2_mod.aruco, enum_name):
        params.cornerRefinementMethod = getattr(cv2_mod.aruco, enum_name)
    values = {
        "cornerRefinementWinSize": int(_setting("CV_ARUCO_CORNER_REFINEMENT_WIN_SIZE", 5)),
        "cornerRefinementMaxIterations": int(_setting("CV_ARUCO_CORNER_REFINEMENT_MAX_ITERATIONS", 30)),
        "cornerRefinementMinAccuracy": float(_setting("CV_ARUCO_CORNER_REFINEMENT_MIN_ACCURACY", 0.1)),
        "adaptiveThreshWinSizeMin": int(_setting("CV_ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MIN", 3)),
        "adaptiveThreshWinSizeMax": int(_setting("CV_ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MAX", 23)),
        "adaptiveThreshWinSizeStep": int(_setting("CV_ARUCO_ADAPTIVE_THRESH_WIN_SIZE_STEP", 10)),
        "minMarkerPerimeterRate": float(_setting("CV_ARUCO_MIN_MARKER_PERIMETER_RATE", 0.015)),
        "maxMarkerPerimeterRate": float(_setting("CV_ARUCO_MAX_MARKER_PERIMETER_RATE", 4.0)),
        "polygonalApproxAccuracyRate": float(_setting("CV_ARUCO_POLYGONAL_APPROX_ACCURACY_RATE", 0.03)),
        "minCornerDistanceRate": float(_setting("CV_ARUCO_MIN_CORNER_DISTANCE_RATE", 0.02)),
    }
    for key, value in values.items():
        if hasattr(params, key):
            setattr(params, key, value)


def _init_aruco_detector(cv2_mod, dictionary_name: str):
    if not hasattr(cv2_mod, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable")
    if not hasattr(cv2_mod.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2_mod.aruco.getPredefinedDictionary(getattr(cv2_mod.aruco, dictionary_name))
    params = (
        cv2_mod.aruco.DetectorParameters()
        if hasattr(cv2_mod.aruco, "DetectorParameters")
        else cv2_mod.aruco.DetectorParameters_create()
    )
    _apply_aruco_detector_tuning_from_settings(cv2_mod, params)
    if hasattr(cv2_mod.aruco, "ArucoDetector"):
        return cv2_mod.aruco.ArucoDetector(dictionary, params), None, None
    return None, dictionary, params


def _detect_markers(cv2_mod, gray, detector, dictionary, params):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2_mod.aruco.detectMarkers(gray, dictionary, parameters=params)


def _shift_marker_corners(corners, offset_x: int, offset_y: int):
    shifted = []
    for item in corners or []:
        pts = np.asarray(item, np.float32).copy()
        pts[..., 0] += float(offset_x)
        pts[..., 1] += float(offset_y)
        shifted.append(pts)
    return shifted


def _choose_target_marker(cv2_mod, corners, ids, target_ids: list[int]):
    if ids is None or len(ids) == 0:
        return None
    target_set = set(int(v) for v in target_ids)
    best = None
    best_area = -1.0
    for i, item in enumerate(corners):
        marker_id = int(ids[i].item())
        if target_set and marker_id not in target_set:
            continue
        area = abs(float(cv2_mod.contourArea(np.asarray(item, np.float32).reshape(4, 2))))
        if area > best_area:
            best = i
            best_area = area
    return best


def _estimate_pose_single_marker(cv2_mod, marker_corners, marker_size_m, camera_matrix, dist_coeffs):
    if marker_size_m <= 0.0:
        return None, None
    corners_4x2 = np.asarray(marker_corners, np.float32).reshape(4, 2)
    if hasattr(cv2_mod.aruco, "estimatePoseSingleMarkers"):
        # OpenCV expects one (4, 2) corner array per marker. Passing an already
        # batched (1, 4, 2) array inside another list creates an invalid
        # (1, 1, 4, 2) nesting on some OpenCV builds.
        rvecs, tvecs, _ = cv2_mod.aruco.estimatePoseSingleMarkers(
            [corners_4x2], float(marker_size_m), camera_matrix, dist_coeffs
        )
        if rvecs is not None and len(rvecs):
            return rvecs[0].reshape(3), tvecs[0].reshape(3)
    half = float(marker_size_m) * 0.5
    obj = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        np.float64,
    )
    flag = getattr(cv2_mod, "SOLVEPNP_IPPE_SQUARE", cv2_mod.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2_mod.solvePnP(
        obj,
        corners_4x2.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=flag,
    )
    return (rvec.reshape(3), tvec.reshape(3)) if ok else (None, None)


def _marker_reprojection_errors_px(
    cv2_mod,
    marker_corners,
    marker_size_m,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
) -> tuple[float, float]:
    """Return marker reprojection RMSE and maximum corner error in pixels."""
    try:
        half = float(marker_size_m) * 0.5
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )
        projected, _ = cv2_mod.projectPoints(
            object_points,
            np.asarray(rvec, np.float64).reshape(3, 1),
            np.asarray(tvec, np.float64).reshape(3, 1),
            np.asarray(camera_matrix, np.float64).reshape(3, 3),
            np.asarray(dist_coeffs, np.float64),
        )
        actual = np.asarray(marker_corners, np.float64).reshape(4, 2)
        errors = np.linalg.norm(projected.reshape(4, 2) - actual, axis=1)
        if not np.all(np.isfinite(errors)):
            return float("inf"), float("inf")
        return (
            float(np.sqrt(np.mean(np.square(errors)))),
            float(np.max(errors)),
        )
    except Exception:
        return float("inf"), float("inf")


def _project_camera_point_to_pixel(
    cv2_mod,
    point_camera_m,
    camera_matrix,
    dist_coeffs,
) -> Optional[np.ndarray]:
    try:
        point = np.asarray(point_camera_m, np.float64).reshape(3)
        if not np.all(np.isfinite(point)) or float(point[2]) <= 1e-9:
            return None
        projected, _ = cv2_mod.projectPoints(
            point.reshape(1, 1, 3),
            np.zeros((3, 1), np.float64),
            np.zeros((3, 1), np.float64),
            np.asarray(camera_matrix, np.float64).reshape(3, 3),
            np.asarray(dist_coeffs, np.float64),
        )
        pixel = np.asarray(projected, np.float64).reshape(2)
        return pixel if np.all(np.isfinite(pixel)) else None
    except Exception:
        return None


def _compute_area_confidence(cv2_mod, marker_corners, img_w, img_h, area_ref):
    area = abs(float(cv2_mod.contourArea(np.asarray(marker_corners, np.float32).reshape(4, 2))))
    return float(np.clip((area / max(float(img_w * img_h), 1.0)) / max(float(area_ref), 1e-8), 0.0, 1.0))


def _scale_intrinsics_to_runtime(k, calib_w, calib_h, runtime_w, runtime_h):
    if calib_w <= 0 or calib_h <= 0:
        return np.asarray(k, np.float64), 1.0, 1.0
    sx = float(runtime_w) / float(calib_w)
    sy = float(runtime_h) / float(calib_h)
    out = np.asarray(k, np.float64).copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out, sx, sy


def _point_c_to_laser(p_c, r_lc, t_lc):
    return (
        np.asarray(r_lc, np.float64).reshape(3, 3) @ np.asarray(p_c, np.float64).reshape(3, 1)
        + np.asarray(t_lc, np.float64).reshape(3, 1)
    ).reshape(3)


def _laser_axis_point_in_camera(r_lc, t_lc, range_m):
    """Point on the laser's own boresight axis (x_L=y_L=0) at range_m,
    expressed in the CAMERA frame -- the inverse of _point_c_to_laser's
    p_L = R_lc @ p_C + t_lc. Projecting it gives the pixel where the beam
    should land at that range; its offset from the principal point is the
    camera/laser lever-arm parallax, which shrinks as 1/range."""
    r = np.asarray(r_lc, np.float64).reshape(3, 3)
    t = np.asarray(t_lc, np.float64).reshape(3, 1)
    p_l = np.array([[0.0], [0.0], [float(range_m)]], np.float64)
    return (r.T @ (p_l - t)).reshape(3)


def _angles_from_point_laser(p_l):
    x, y, z = [float(v) for v in np.asarray(p_l, np.float64).reshape(3)]
    if not all(np.isfinite(v) for v in (x, y, z)) or z <= 1e-6:
        return None
    return float(np.arctan2(x, z)), float(np.arctan2(-y, math.sqrt(x * x + z * z)))


# ============================================================================
# Calibration and misc helpers
# ============================================================================


def _coerce_rotmat(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, np.float64).reshape(3, 3))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _matrix_to_int_list(mat) -> list[int]:
    if mat is None:
        return []
    return [int(round(float(v))) for v in np.asarray(mat).reshape(-1)]


def _parse_opencv_matrix_from_text(text: str, key: str) -> np.ndarray:
    pattern = (
        rf"(?ms)^\s*{re.escape(key)}\s*:\s*!!opencv-matrix\s*"
        rf"\n\s*rows\s*:\s*(\d+)\s*"
        rf"\n\s*cols\s*:\s*(\d+)\s*"
        rf"\n\s*dt\s*:\s*\w+\s*"
        rf"\n\s*data\s*:\s*\[(.*?)\]"
    )
    match = re.search(pattern, text)
    if match is None:
        raise KeyError(key)
    rows, cols = int(match.group(1)), int(match.group(2))
    values = [float(v.strip()) for v in match.group(3).replace("\n", " ").split(",") if v.strip()]
    return np.asarray(values, np.float64).reshape(rows, cols)


def _parse_scalar(text: str, key: str, default: Any = None):
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
    if match is None:
        return default
    return match.group(1).strip().strip("\"'")


def _load_vs_calibration_for_camera(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    cv2 = _import_cv2_or_raise()
    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            width = int(fs.getNode("image_width").real())
            height = int(fs.getNode("image_height").real())
            k = fs.getNode("camera_matrix").mat()
            dist = fs.getNode("dist_coeffs").mat()
            dictionary = fs.getNode("aruco_dict_name").string()
            marker_size = float(fs.getNode("marker_size_m").real())
            ids = fs.getNode("target_marker_ids").mat()
            r_lc = fs.getNode("R_lc").mat()
            t_lc = fs.getNode("t_lc_m").mat()
            r_cl = fs.getNode("R_cl").mat()
            t_cl = fs.getNode("t_cl_m").mat()
        finally:
            fs.release()
        if k is not None and dist is not None:
            if r_lc is None or t_lc is None:
                if r_cl is None or t_cl is None:
                    raise RuntimeError("Calibration must contain R_lc/t_lc_m or R_cl/t_cl_m")
                r_cl = _coerce_rotmat(r_cl)
                t_cl = np.asarray(t_cl, np.float64).reshape(3, 1)
                r_lc = r_cl.T
                t_lc = -r_lc @ t_cl
            return {
                "image_width": width,
                "image_height": height,
                "camera_matrix": np.asarray(k, np.float64),
                "dist_coeffs": np.asarray(dist, np.float64),
                "aruco_dict_name": dictionary or "DICT_7X7_100",
                "marker_size_m": marker_size if marker_size > 0 else 0.10,
                "target_marker_ids": _matrix_to_int_list(ids) or [1],
                "R_lc": _coerce_rotmat(r_lc),
                "t_lc_m": np.asarray(t_lc, np.float64).reshape(3, 1),
            }
    except Exception:
        pass

    # Text fallback for OpenCV YAML variants that FileStorage cannot parse.
    text = path.read_text(encoding="utf-8")
    k = _parse_opencv_matrix_from_text(text, "camera_matrix")
    dist = _parse_opencv_matrix_from_text(text, "dist_coeffs")
    try:
        r_lc = _parse_opencv_matrix_from_text(text, "R_lc")
        t_lc = _parse_opencv_matrix_from_text(text, "t_lc_m")
    except KeyError:
        r_cl = _parse_opencv_matrix_from_text(text, "R_cl")
        t_cl = _parse_opencv_matrix_from_text(text, "t_cl_m").reshape(3, 1)
        r_lc = _coerce_rotmat(r_cl).T
        t_lc = -r_lc @ t_cl
    ids_text = _parse_scalar(text, "target_marker_ids", "[1]")
    ids = [int(v) for v in re.findall(r"-?\d+", str(ids_text))] or [1]
    return {
        "image_width": int(float(_parse_scalar(text, "image_width", 0))),
        "image_height": int(float(_parse_scalar(text, "image_height", 0))),
        "camera_matrix": k,
        "dist_coeffs": dist,
        "aruco_dict_name": str(_parse_scalar(text, "aruco_dict_name", "DICT_7X7_100")),
        "marker_size_m": float(_parse_scalar(text, "marker_size_m", 0.10)),
        "target_marker_ids": ids,
        "R_lc": _coerce_rotmat(r_lc),
        "t_lc_m": np.asarray(t_lc, np.float64).reshape(3, 1),
    }


def _pbvs_target_point_marker_from_settings() -> Optional[np.ndarray]:
    dx = _float_or_none(_setting("CV_PBVS_TARGET_DX_M", None))
    dy = _float_or_none(_setting("CV_PBVS_TARGET_DY_M", None))
    dz = _float_or_none(_setting("CV_PBVS_TARGET_DZ_M", None))
    if dx is None and dy is None and dz is None:
        return None
    dx = 0.0 if dx is None else dx
    dy = 0.0 if dy is None else dy
    dz = 0.0 if dz is None else dz
    if abs(dx) + abs(dy) + abs(dz) <= 1e-12:
        return None
    # Settings FRD -> OpenCV marker frame: F=-Z, R=+X, D=-Y.
    return np.array([dy, -dz, -dx], np.float64)


def _ids_exposure_time_us_from_settings() -> Optional[float]:
    value = _float_or_none(_setting("CV_IDS_EXPOSURE_TIME_US", None))
    return value if value is not None and value > 0.0 else None


def _ids_frame_rate_target_hz_from_settings() -> Optional[float]:
    if not _bool_setting("CV_IDS_FRAME_RATE_TARGET_ENABLE", True):
        return None
    value = _float_or_none(_setting("CV_IDS_FRAME_RATE_TARGET_HZ", 70.0))
    return value if value is not None and value > 0.0 else None


def _resolve_optional_path(raw: Any, base_dir: Path) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _clone_debug_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        out[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return out


__all__ = [
    "CAMERA_CUDA_VERSION",
    "CorrectionCandidate",
    "CameraCorrection",
    "CameraHealth",
    "GatingConfig",
    "RuntimeConfig",
    "CameraConfig",
    "CameraAPI",
    "CameraApplicationService",
    "CsvCorrectionSourceAdapter",
    "IdsOpenCvCorrectionSourceAdapter",
    "FastIdsOpenCvCorrectionSourceAdapter",
    "MultiArucoCorrectionSourceAdapter",
    "ArucoMarkerSpec",
    "ArucoObservation",
    "MultiMarkerTargetPoint",
    "MultiMarkerTrackingConfig",
    "load_multi_marker_tracking_config",
    "compute_multi_marker_target_point",
    "load_camera_config_yaml",
    "from_yaml",
]
