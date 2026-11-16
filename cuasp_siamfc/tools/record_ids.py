"""Record IDS camera sequences for validation V4 (procedure: ../../RECORDING_PROTOCOL.md).

Every frame delivered by the camera is saved, with its capture timestamp, using exactly
the IDS settings of the runtime (built by ``cuasp._build_camera_config()`` from
``suncubes/settings.py``: resolution, Mono8, exposure, gain, frame-rate target).

Usage (from cuasp_siamfc/)::

    # one sequence, live preview, stop with q (or after --duration seconds)
    .venv/Scripts/python tools/record_ids.py --name B1_hover_center --block B --distance-m 21.0 --notes "hover, lights on"
    # empty scene (protocol block A), 60 s, no preview
    .venv/Scripts/python tools/record_ids.py --name A3_wall_east --block A --duration 60 --no-preview
    # dry run without camera (synthetic frames): checks disk speed and the whole pipeline
    .venv/Scripts/python tools/record_ids.py --name dryrun --fake --duration 20

Output: ``<out>/<YYYYMMDD_HHMMSS>_<name>/`` (default out: ``../recordings``, gitignored) with
- ``frames/000000.jpg ...`` grayscale JPEG (``--jpeg-quality``, default 95 like the
  detector training set) or ``--png`` (lossless). The folder can be given directly to
  ``tools/replay_siamfc_source.py``;
- ``timestamps.csv``: ``frame,sequence,ts_capture_ns`` (written while recording);
- ``meta.json``: notes, distance, block, camera settings, calibration in use, software
  version, and statistics: frames, losses, measured frame rate, timestamp gaps, the
  **actual** frame size and the gray levels (sampled). For the real camera, ``camera.readback``
  holds the node values read back (read-only) after the recording: the settings are only a
  request, and the camera silently keeps its own value when it rejects one (2026-11-16: the
  U3-33DxXCP-M, Sony IMX900 2064x1552, keeps its 2040x1536 ROI when 2592x1944 is requested).
  ``checks`` lists mismatches: frame size against the requested size and against the
  calibration size, size changes during the recording, saturated or very dark frames.

Losses are counted, never hidden:
- ``dropped_capture``: frames the camera delivered that this tool did not take (gaps in the
  producer sequence; the recorder was too slow);
- ``dropped_writer``: frames taken but not saved because the write queue was full (the
  disk or the JPEG encoding was too slow). Capture never blocks on writing;
- ``timestamp_gaps``: intervals longer than 1.5x the median frame interval, which also
  reveal frames lost before reaching the host. Timestamps are host arrival times
  (``time.monotonic_ns()`` in ``_IdsLatestFrameProducer``), not sensor timestamps.

Requirements for real recording: the IDS peak SDK and its Python bindings (``ids_peak``,
``ids_peak_ipl``) in this venv (README section 8). Motor commands are not involved.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "recordings"
TOOL_VERSION = "2026-11-16-record-ids-v2"
BLOCKS = ("A", "B", "C", "D", "other")
# Camera nodes read back after the recording; for the *_MAX ones the node maximum is stored too
# (AcquisitionFrameRate's maximum is the frame rate the camera allows at the current exposure).
READBACK_NODES = (
    "DeviceModelName", "DeviceSerialNumber", "DeviceFirmwareVersion", "SensorName", "SensorWidth", "SensorHeight",
    "SensorPixelWidth", "SensorPixelHeight", "SensorShutterMode", "UserSetDefault", "PixelFormat", "Width", "Height",
    "OffsetX", "OffsetY", "BinningHorizontal", "BinningVertical", "ExposureTime", "Gain", "BlackLevel",
    "AcquisitionFrameRate", "AcquisitionFrameRateTarget", "AcquisitionFrameRateTargetEnable",
    "DeviceLinkThroughputLimit", "DeviceLinkSpeed", "AcquisitionMode", "TriggerMode",
)
READBACK_NODES_MAX = ("AcquisitionFrameRate", "Gain", "Width", "Height")
GRAY_SAMPLE_EVERY = 25  # frames between gray-level samples
DARK_P99 = 64  # 99th percentile below this gray level: flagged as very dark
SATURATED_PCT = 1.0  # more than this % of pixels at 255 in a sample: flagged as saturated


# ============================================================================ producers


class FakeProducer:
    """Synthetic camera with the ``_IdsLatestFrameProducer`` interface (dry runs and tests).

    Publishes ``width`` x ``height`` Mono8 frames at ``fps`` from its own thread: a noisy
    background with a bright moving blob. ``skip_every`` > 0 publishes a sequence gap every
    N frames (to test loss accounting).
    """

    def __init__(self, width: int = 2592, height: int = 1944, fps: float = 50.0, skip_every: int = 0, seed: int = 0):
        from suncubes.camera import _FramePacket  # noqa: PLC0415

        self._packet_cls = _FramePacket
        self.width, self.height, self.fps, self.skip_every = int(width), int(height), float(fps), int(skip_every)
        rng = np.random.default_rng(seed)
        self._background = rng.normal(110, 12, (self.height, self.width)).clip(0, 255).astype(np.uint8)
        self._cond = threading.Condition()
        self._packet = None
        self._sequence = 0
        self._published = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fake-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        i = 0
        while not self._stop.is_set():
            frame = self._background.copy()
            cx = int(self.width / 2 + 0.3 * self.width * math.sin(i / 40.0))
            cy = int(self.height / 2 + 0.2 * self.height * math.cos(i / 55.0))
            s = max(8, self.width // 40)
            frame[max(0, cy - s):cy + s, max(0, cx - s):cx + s] = 230
            with self._cond:
                self._sequence += 1
                if self.skip_every > 0 and self._sequence % self.skip_every == 0:
                    self._sequence += 1  # a frame "lost" before publication
                self._published += 1
                self._packet = self._packet_cls(gray=frame, ts_capture_ns=time.monotonic_ns(), sequence=self._sequence)
                self._cond.notify_all()
            i += 1
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def latest(self):
        with self._cond:
            return self._packet

    def wait_newer(self, last_sequence: int, timeout_s: float):
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
            return {"published": int(self._published), "sequence": int(self._sequence)}


# ============================================================================ recorder


class Recorder:
    """Takes every new frame from ``producer`` and saves it with a pool of writer threads.

    ``run()`` blocks until ``stop()`` is called, ``max_frames`` frames were taken, or
    ``duration_s`` elapsed. ``encode`` is the image writer, replaceable in tests.
    """

    def __init__(
        self,
        producer: Any,
        seq_dir: Path,
        *,
        jpeg_quality: int = 95,
        png: bool = False,
        writers: int = 4,
        queue_size: int = 128,
        duration_s: Optional[float] = None,
        max_frames: Optional[int] = None,
        encode: Optional[Callable[[Path, np.ndarray], bool]] = None,
    ) -> None:
        self.producer = producer
        self.seq_dir = Path(seq_dir)
        self.frames_dir = self.seq_dir / "frames"
        self.ext = ".png" if png else ".jpg"
        self.jpeg_quality = int(jpeg_quality)
        self.png = bool(png)
        self.writers = max(1, int(writers))
        self.duration_s = duration_s
        self.max_frames = max_frames
        self._encode = encode or self._encode_cv2
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "frames_taken": 0, "frames_written": 0, "dropped_capture": 0, "dropped_writer": 0,
            "write_errors": 0, "bytes_written": 0, "frame_size_changes": 0,
        }
        self.frame_hw: Optional[tuple[int, int]] = None  # (height, width) of the first frame
        self._gray_samples: list[tuple[float, float, float, float, float]] = []
        self._ts: list[int] = []
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None

    def stop(self) -> None:
        self._stop.set()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _encode_cv2(self, path: Path, gray: np.ndarray) -> bool:
        import cv2  # noqa: PLC0415

        params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if self.png else [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        return bool(cv2.imwrite(str(path), gray, params))

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            index, gray = item
            path = self.frames_dir / f"{index:06d}{self.ext}"
            try:
                ok = self._encode(path, gray)
                size = path.stat().st_size if ok and path.exists() else 0
            except Exception:  # noqa: BLE001
                ok, size = False, 0
            with self._lock:
                if ok:
                    self.stats["frames_written"] += 1
                    self.stats["bytes_written"] += int(size)
                else:
                    self.stats["write_errors"] += 1

    def run(self) -> dict[str, Any]:
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        pool = [threading.Thread(target=self._writer, name=f"writer-{k}", daemon=True) for k in range(self.writers)]
        for t in pool:
            t.start()
        last_seq = 0
        index = 0
        self.t_start = time.monotonic()
        with open(self.seq_dir / "timestamps.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "sequence", "ts_capture_ns"])
            try:
                while not self._stop.is_set():
                    if self.duration_s is not None and time.monotonic() - self.t_start >= float(self.duration_s):
                        break
                    if self.max_frames is not None and index >= int(self.max_frames):
                        break
                    packet = self.producer.wait_newer(last_seq, 0.5)
                    if packet is None:
                        continue
                    seq = int(packet.sequence)
                    if last_seq > 0 and seq > last_seq + 1:
                        with self._lock:
                            self.stats["dropped_capture"] += seq - last_seq - 1
                    last_seq = seq
                    with self._lock:
                        self.stats["frames_taken"] += 1
                    hw = (int(packet.gray.shape[0]), int(packet.gray.shape[1]))
                    if self.frame_hw is None:
                        self.frame_hw = hw
                    elif hw != self.frame_hw:
                        with self._lock:
                            self.stats["frame_size_changes"] += 1
                    if index % GRAY_SAMPLE_EVERY == 0:
                        self._gray_samples.append(gray_levels(packet.gray))
                    try:
                        # The producer publishes a fresh array per frame and never mutates it: no copy needed.
                        self._queue.put_nowait((index, packet.gray))
                    except queue.Full:
                        with self._lock:
                            self.stats["dropped_writer"] += 1
                        continue
                    w.writerow([index, seq, int(packet.ts_capture_ns)])
                    self._ts.append(int(packet.ts_capture_ns))
                    if index % 50 == 0:
                        fh.flush()
                    index += 1
            finally:
                self.t_end = time.monotonic()
                for _ in pool:
                    self._queue.put(None)
                for t in pool:
                    t.join()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        out = dict(self.stats)
        ts = np.asarray(self._ts, np.int64)
        out["duration_s"] = None if self.t_start is None or self.t_end is None else round(self.t_end - self.t_start, 3)
        if len(ts) >= 2:
            dt = np.diff(ts) * 1e-9
            med = float(np.median(dt))
            out["measured_fps"] = round((len(ts) - 1) / float((ts[-1] - ts[0]) * 1e-9), 3)
            out["frame_interval_ms"] = {"median": round(med * 1e3, 3), "p99": round(float(np.percentile(dt, 99)) * 1e3, 3),
                                        "max": round(float(dt.max()) * 1e3, 3)}
            out["timestamp_gaps"] = int(np.sum(dt > 1.5 * med)) if med > 0 else 0
        else:
            out["measured_fps"], out["frame_interval_ms"], out["timestamp_gaps"] = None, None, 0
        written = out["frames_written"]
        out["mean_frame_mb"] = round(out["bytes_written"] / written / 1e6, 3) if written else None
        out["frame_width"] = None if self.frame_hw is None else self.frame_hw[1]
        out["frame_height"] = None if self.frame_hw is None else self.frame_hw[0]
        if self._gray_samples:
            g = np.asarray(self._gray_samples, np.float64)
            out["gray_levels"] = {
                "samples": int(len(g)), "mean": round(float(np.median(g[:, 0])), 1),
                "p1": round(float(np.median(g[:, 1])), 1), "p50": round(float(np.median(g[:, 2])), 1),
                "p99": round(float(np.median(g[:, 3])), 1), "saturated_pct_max": round(float(g[:, 4].max()), 3),
            }
        else:
            out["gray_levels"] = None
        return out


def gray_levels(gray: np.ndarray, step: int = 8) -> tuple[float, float, float, float, float]:
    """``(mean, p1, p50, p99, % of pixels at 255)`` of an 8-bit frame subsampled by ``step``
    (a histogram, so well under 1 ms on a 3 MP frame: safe in the capture loop)."""
    sub = np.ascontiguousarray(np.asarray(gray, np.uint8)[::step, ::step]).ravel()
    n = max(int(sub.size), 1)
    hist = np.bincount(sub, minlength=256).astype(np.float64)
    cdf = np.cumsum(hist) / n
    p1, p50, p99 = (float(np.searchsorted(cdf, q)) for q in (0.01, 0.5, 0.99))
    return float(sub.mean()) if sub.size else 0.0, p1, p50, p99, float(hist[255] * 100.0 / n)


# ============================================================================ helpers


def git_commit(repo: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        return r.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return None


def runtime_camera_settings() -> dict[str, Any]:
    """IDS dict and calibration exactly as cuasp.py builds them for the runtime."""
    import cuasp  # noqa: PLC0415
    from suncubes import settings  # noqa: PLC0415

    cfg = cuasp._build_camera_config()
    return {
        "ids_peak": dict(cfg.ids_opencv["ids_peak"]),
        "calibration": str(cuasp.CALIBRATION_YAML),
        "settings": {n: getattr(settings, n) for n in dir(settings) if n.startswith("CV_IDS_")},
    }


def calibration_summary(path: Path) -> dict[str, Any]:
    """Image size and focal lengths of the calibration the runtime uses (same loader as camera.py)."""
    from suncubes.camera import _load_vs_calibration_for_camera  # noqa: PLC0415

    cal = _load_vs_calibration_for_camera(Path(path))
    k = np.asarray(cal["camera_matrix"], np.float64).reshape(3, 3)
    return {"image_width": int(cal["image_width"]), "image_height": int(cal["image_height"]),
            "fx": round(float(k[0, 0]), 1), "fy": round(float(k[1, 1]), 1),
            "cx": round(float(k[0, 2]), 1), "cy": round(float(k[1, 2]), 1)}


def read_nodes(nodemap: Any, names=READBACK_NODES, max_names=READBACK_NODES_MAX) -> dict[str, Any]:
    """Current value of each GenICam node in ``names`` (enumerations by symbolic name), plus
    ``<name>_max`` for ``max_names``. Read-only: nothing is set. Missing nodes are skipped."""
    out: dict[str, Any] = {}
    for name in names:
        try:
            node = nodemap.FindNode(name)
        except Exception:  # noqa: BLE001
            continue
        for getter in (lambda n: n.CurrentEntry().SymbolicValue(), lambda n: n.Value()):
            try:
                value = getter(node)
            except Exception:  # noqa: BLE001
                continue
            out[name] = value if isinstance(value, (bool, int, float, str)) else str(value)
            break
        if name in max_names:
            try:
                out[f"{name}_max"] = float(node.Maximum())
            except Exception:  # noqa: BLE001
                pass
    return out


def read_ids_camera_state(device_index: int = 0) -> dict[str, Any]:
    """Opens the camera read-only, after the producer has released it, and reads ``READBACK_NODES``.
    The values set by the producer persist in the camera until it is reset or powered off."""
    from suncubes.camera import _load_ids_modules  # noqa: PLC0415

    ids_peak, _ = _load_ids_modules()
    ids_peak.Library.Initialize()
    try:
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        count = int(dm.Devices().size())
        if not 0 <= int(device_index) < count:
            return {"error": f"device_index {device_index} not found ({count} devices)"}
        device = dm.Devices()[int(device_index)].OpenDevice(ids_peak.DeviceAccessType_ReadOnly)
        try:
            return read_nodes(device.RemoteDevice().NodeMaps()[0])
        finally:
            del device
    finally:
        ids_peak.Library.Close()


def consistency_checks(summary: dict[str, Any], requested_wh: Optional[tuple[int, int]],
                       calibration_wh: Optional[tuple[int, int]]) -> list[str]:
    """Human-readable warnings about the recording; empty when everything is consistent."""
    notes: list[str] = []
    fw, fh = summary.get("frame_width"), summary.get("frame_height")
    if fw is None or fh is None:
        return notes
    if requested_wh is not None and tuple(requested_wh) != (fw, fh):
        notes.append(f"frames are {fw}x{fh} but the settings request {requested_wh[0]}x{requested_wh[1]}: "
                     "the camera keeps its own size when it rejects the requested one (see camera.readback)")
    if calibration_wh is not None and tuple(calibration_wh) != (fw, fh):
        notes.append(f"frames are {fw}x{fh} but the calibration is {calibration_wh[0]}x{calibration_wh[1]}: "
                     "the runtime would rescale the intrinsics to the frame size")
    if summary.get("frame_size_changes"):
        notes.append(f"the frame size changed {summary['frame_size_changes']} times during the recording")
    g = summary.get("gray_levels")
    if g:
        if g["saturated_pct_max"] > SATURATED_PCT:
            notes.append(f"saturated frames: up to {g['saturated_pct_max']:.1f}% of pixels at 255")
        if g["p99"] < DARK_P99:
            notes.append(f"very dark frames: 99% of pixels at or below gray level {g['p99']:.0f}/255 (median of samples)")
    return notes


def _preview_loop(recorder: Recorder, producer: Any, name: str, worker: threading.Thread, max_width: int = 960) -> None:
    """Main-thread live view (OpenCV windows must be driven by one thread) while ``worker``
    records: q or Esc stops the recording; the view closes when the recording ends."""
    import cv2  # noqa: PLC0415

    win = f"record_ids - {name} (q: stop)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while worker.is_alive():
            packet = producer.latest()
            if packet is not None:
                gray = packet.gray
                h, w = gray.shape[:2]
                s = min(1.0, max_width / float(w))
                img = cv2.cvtColor(cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
                ph, pw = img.shape[:2]
                cv2.line(img, (pw // 2 - 20, ph // 2), (pw // 2 + 20, ph // 2), (0, 255, 255), 1)
                cv2.line(img, (pw // 2, ph // 2 - 20), (pw // 2, ph // 2 + 20), (0, 255, 255), 1)
                st = recorder.stats
                el = time.monotonic() - (recorder.t_start or time.monotonic())
                fps = st["frames_taken"] / el if el > 0 else 0.0
                lost = st["dropped_capture"] + st["dropped_writer"]
                cv2.putText(img, f"REC {name}  {el:6.1f} s  frames {st['frames_written']}  {fps:5.1f} fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f"lost {lost} (capture {st['dropped_capture']}, writer {st['dropped_writer']})  queue {recorder.queue_depth}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if lost else (0, 200, 0), 2, cv2.LINE_AA)
                cv2.imshow(win, img)
            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), 27):
                recorder.stop()
    finally:
        cv2.destroyWindow(win)


# ============================================================================ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="sequence name, e.g. B1_hover_center (protocol code + description)")
    ap.add_argument("--block", choices=BLOCKS, default="other", help="block of RECORDING_PROTOCOL.md")
    ap.add_argument("--notes", default="", help="what happens in the sequence")
    ap.add_argument("--distance-m", type=float, help="measured drone distance (m), if any")
    ap.add_argument("--drone", default="", help="drone model / configuration (e.g. 'Mavic Mini, ArUco markers')")
    ap.add_argument("--lighting", default="", help="lighting configuration")
    ap.add_argument("--duration", type=float, help="stop after this many seconds (else q or Ctrl+C)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--png", action="store_true", help="lossless PNG instead of JPEG (about 3-4 times larger)")
    ap.add_argument("--writers", type=int, default=4)
    ap.add_argument("--queue-size", type=int, default=128, help="frames buffered for writing (~5 MB each)")
    ap.add_argument("--buffers", type=int, default=16, help="IDS SDK acquisition buffers")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--fake", action="store_true", help="synthetic frames instead of the camera (dry run)")
    ap.add_argument("--fake-fps", type=float, default=50.0)
    args = ap.parse_args()

    if not all(c.isalnum() or c in "-_" for c in args.name):
        ap.error("--name may contain only letters, digits, '-' and '_'")
    if not 1 <= args.jpeg_quality <= 100:
        ap.error("--jpeg-quality must be in 1..100")
    args.out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.out).free / 1e9
    if free_gb < args.min_free_gb:
        print(f"only {free_gb:.1f} GB free in {args.out} (< --min-free-gb {args.min_free_gb})", file=sys.stderr)
        return 2
    started = datetime.now()
    seq_dir = args.out / f"{started:%Y%m%d_%H%M%S}_{args.name}"
    if seq_dir.exists():
        print(f"{seq_dir} already exists", file=sys.stderr)
        return 2

    camera = runtime_camera_settings()
    calibration_wh: Optional[tuple[int, int]] = None
    try:
        camera["calibration_summary"] = calibration_summary(Path(camera["calibration"]))
        calibration_wh = (camera["calibration_summary"]["image_width"], camera["calibration_summary"]["image_height"])
    except Exception as exc:  # noqa: BLE001
        camera["calibration_summary"] = {"error": repr(exc)}
    requested_wh = None
    if "width" in camera["ids_peak"] and "height" in camera["ids_peak"]:
        requested_wh = (int(camera["ids_peak"]["width"]), int(camera["ids_peak"]["height"]))
    if args.fake:
        w, h = int(camera["ids_peak"].get("width", 2592)), int(camera["ids_peak"].get("height", 1944))
        producer: Any = FakeProducer(w, h, fps=args.fake_fps)
    else:
        from suncubes.camera import _IdsLatestFrameProducer  # noqa: PLC0415

        producer = _IdsLatestFrameProducer({**camera["ids_peak"], "buffer_count": int(args.buffers)})

    recorder = Recorder(producer, seq_dir, jpeg_quality=args.jpeg_quality, png=args.png, writers=args.writers,
                        queue_size=args.queue_size, duration_s=args.duration, max_frames=args.max_frames)
    print(f"recording {'SYNTHETIC frames' if args.fake else 'IDS camera'} -> {seq_dir}")
    print(f"  requested {camera['ids_peak'].get('width')}x{camera['ids_peak'].get('height')} {camera['ids_peak'].get('pixel_format')}, "
          f"exposure {camera['ids_peak'].get('exposure_time_us')} us, target {camera['ids_peak'].get('frame_rate_target_hz')} Hz; "
          f"{free_gb:.0f} GB free")
    producer.start()
    worker = threading.Thread(target=recorder.run, name="recorder", daemon=True)
    try:
        worker.start()
        if args.no_preview:
            while worker.is_alive():
                worker.join(timeout=0.5)
        else:
            _preview_loop(recorder, producer, args.name, worker)
            recorder.stop()
            worker.join()
    except KeyboardInterrupt:
        print("\nstopping...")
        recorder.stop()
        worker.join()
    finally:
        producer.stop()

    if not args.fake:
        try:
            camera["readback"] = read_ids_camera_state(int(camera["ids_peak"].get("device_index", 0)))
        except Exception as exc:  # noqa: BLE001
            camera["readback"] = {"error": repr(exc)}
    summary = recorder.summary()
    checks = consistency_checks(summary, requested_wh, calibration_wh)
    meta = {
        "tool_version": TOOL_VERSION,
        "name": args.name, "block": args.block, "notes": args.notes, "distance_m": args.distance_m,
        "drone": args.drone, "lighting": args.lighting, "synthetic": bool(args.fake),
        "started_at": started.isoformat(timespec="seconds"), "ended_at": datetime.now().isoformat(timespec="seconds"),
        "format": "png" if args.png else f"jpeg q{args.jpeg_quality}", "frames_dir": "frames",
        "camera": camera, "producer_stats": producer.stats() if hasattr(producer, "stats") else None,
        "git_commit": git_commit(REPO), "statistics": summary, "checks": checks,
    }
    (seq_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    lost = summary["dropped_capture"] + summary["dropped_writer"] + summary["write_errors"]
    print(f"\nsaved {summary['frames_written']} frames in {summary['duration_s']} s, measured {summary['measured_fps']} fps, "
          f"{summary['mean_frame_mb']} MB/frame")
    print(f"lost: capture {summary['dropped_capture']}, writer queue {summary['dropped_writer']}, write errors "
          f"{summary['write_errors']}, timestamp gaps {summary['timestamp_gaps']}")
    g = summary.get("gray_levels")
    if summary.get("frame_width") is not None:
        print(f"frames {summary['frame_width']}x{summary['frame_height']}"
              + (f"; gray levels: mean {g['mean']:.0f}, p1 {g['p1']:.0f}, p99 {g['p99']:.0f}, "
                 f"saturated up to {g['saturated_pct_max']:.2f}%" if g else ""))
    rb = camera.get("readback")
    if isinstance(rb, dict) and "error" not in rb:
        print(f"camera: {rb.get('DeviceModelName')} ({rb.get('SensorName')}), ROI {rb.get('Width')}x{rb.get('Height')} "
              f"at ({rb.get('OffsetX')}, {rb.get('OffsetY')}), exposure {rb.get('ExposureTime', float('nan')):.0f} us, "
              f"gain {rb.get('Gain')}, max frame rate {rb.get('AcquisitionFrameRate_max', float('nan')):.2f} Hz")
    elif isinstance(rb, dict):
        print(f"camera readback failed: {rb['error']}")
    if lost:
        print("WARNING: frames were lost; see meta.json and RECORDING_PROTOCOL.md (troubleshooting)")
    for note in checks:
        print(f"NOTE: {note}")
    print(f"meta: {seq_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
