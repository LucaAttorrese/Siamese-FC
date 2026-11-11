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
  version, and statistics: frames, losses, measured frame rate, timestamp gaps.

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
TOOL_VERSION = "2026-11-11-record-ids-v1"
BLOCKS = ("A", "B", "C", "D", "other")


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
            "write_errors": 0, "bytes_written": 0,
        }
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
        return out


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
    if args.fake:
        w, h = int(camera["ids_peak"].get("width", 2592)), int(camera["ids_peak"].get("height", 1944))
        producer: Any = FakeProducer(w, h, fps=args.fake_fps)
    else:
        from suncubes.camera import _IdsLatestFrameProducer  # noqa: PLC0415

        producer = _IdsLatestFrameProducer({**camera["ids_peak"], "buffer_count": int(args.buffers)})

    recorder = Recorder(producer, seq_dir, jpeg_quality=args.jpeg_quality, png=args.png, writers=args.writers,
                        queue_size=args.queue_size, duration_s=args.duration, max_frames=args.max_frames)
    print(f"recording {'SYNTHETIC frames' if args.fake else 'IDS camera'} -> {seq_dir}")
    print(f"  {camera['ids_peak'].get('width')}x{camera['ids_peak'].get('height')} {camera['ids_peak'].get('pixel_format')}, "
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

    summary = recorder.summary()
    meta = {
        "tool_version": TOOL_VERSION,
        "name": args.name, "block": args.block, "notes": args.notes, "distance_m": args.distance_m,
        "drone": args.drone, "lighting": args.lighting, "synthetic": bool(args.fake),
        "started_at": started.isoformat(timespec="seconds"), "ended_at": datetime.now().isoformat(timespec="seconds"),
        "format": "png" if args.png else f"jpeg q{args.jpeg_quality}", "frames_dir": "frames",
        "camera": camera, "producer_stats": producer.stats() if hasattr(producer, "stats") else None,
        "git_commit": git_commit(REPO), "statistics": summary,
    }
    (seq_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    lost = summary["dropped_capture"] + summary["dropped_writer"] + summary["write_errors"]
    print(f"\nsaved {summary['frames_written']} frames in {summary['duration_s']} s, measured {summary['measured_fps']} fps, "
          f"{summary['mean_frame_mb']} MB/frame")
    print(f"lost: capture {summary['dropped_capture']}, writer queue {summary['dropped_writer']}, write errors "
          f"{summary['write_errors']}, timestamp gaps {summary['timestamp_gaps']}")
    if lost:
        print("WARNING: frames were lost; see meta.json and RECORDING_PROTOCOL.md (troubleshooting)")
    print(f"meta: {seq_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
