"""tools/record_ids.py with the synthetic producer (no camera): frame accounting, losses, outputs."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rec():
    return _load("record_ids")


def _csv_rows(seq_dir: Path) -> list[list[str]]:
    with open(seq_dir / "timestamps.csv", newline="") as fh:
        return list(csv.reader(fh))[1:]


def _run(rec, tmp_path, producer, **kw):
    seq_dir = tmp_path / "seq"
    r = rec.Recorder(producer, seq_dir, **kw)
    producer.start()
    try:
        summary = r.run()
    finally:
        producer.stop()
    return r, seq_dir, summary


def test_every_frame_is_saved_with_its_timestamp(rec, tmp_path):
    prod = rec.FakeProducer(320, 240, fps=100.0)
    _, seq_dir, s = _run(rec, tmp_path, prod, duration_s=0.6, writers=2)
    files = sorted((seq_dir / "frames").glob("*.jpg"))
    rows = _csv_rows(seq_dir)
    assert s["frames_written"] == len(files) == len(rows) == s["frames_taken"] > 40
    assert s["dropped_capture"] == s["dropped_writer"] == s["write_errors"] == 0
    assert [int(r[0]) for r in rows] == list(range(len(rows)))  # frame indices match file names
    assert files[0].name == "000000.jpg"
    assert s["measured_fps"] == pytest.approx(100.0, rel=0.15)
    assert s["timestamp_gaps"] == 0 and s["mean_frame_mb"] > 0


def test_sequence_gaps_are_counted_as_capture_losses(rec, tmp_path):
    prod = rec.FakeProducer(160, 120, fps=100.0, skip_every=5)
    _, _, s = _run(rec, tmp_path, prod, duration_s=0.5)
    assert s["dropped_capture"] >= s["frames_taken"] // 6  # one gap every ~5 frames


def test_a_slow_disk_drops_frames_instead_of_blocking_capture(rec, tmp_path):
    def slow_encode(path, gray):
        time.sleep(0.05)
        path.write_bytes(b"x")
        return True

    prod = rec.FakeProducer(160, 120, fps=100.0)
    _, seq_dir, s = _run(rec, tmp_path, prod, duration_s=0.5, writers=1, queue_size=2, encode=slow_encode)
    assert s["dropped_writer"] > 0
    assert s["frames_written"] + s["dropped_writer"] == s["frames_taken"]
    assert len(list((seq_dir / "frames").iterdir())) == s["frames_written"] == len(_csv_rows(seq_dir))


def test_write_errors_are_counted(rec, tmp_path):
    prod = rec.FakeProducer(160, 120, fps=100.0)
    _, _, s = _run(rec, tmp_path, prod, max_frames=10, encode=lambda path, gray: False)
    assert s["write_errors"] == 10 and s["frames_written"] == 0


def test_existing_sequence_folder_is_never_overwritten(rec, tmp_path):
    (tmp_path / "seq" / "frames").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        rec.Recorder(rec.FakeProducer(64, 48), tmp_path / "seq", max_frames=1).run()


def test_recorded_frames_are_readable_by_the_replay_tool(rec, tmp_path):
    prod = rec.FakeProducer(160, 120, fps=100.0)
    _, seq_dir, s = _run(rec, tmp_path, prod, max_frames=5)
    replay = _load("replay_siamfc_source")
    frames = replay.list_frames(seq_dir / "frames")
    assert len(frames) == 5
    grays = list(replay.iter_frames(seq_dir / "frames", None))
    assert grays[0].shape == (120, 160) and grays[0].dtype.name == "uint8"


def test_command_line_dry_run_writes_meta(rec, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "record_ids.py", "--name", "dry_A1", "--block", "A", "--fake", "--fake-fps", "30", "--duration", "1",
        "--no-preview", "--out", str(tmp_path), "--min-free-gb", "0", "--notes", "test", "--distance-m", "21",
    ])
    assert rec.main() == 0
    seq_dirs = list(tmp_path.glob("*_dry_A1"))
    assert len(seq_dirs) == 1
    meta = json.loads((seq_dirs[0] / "meta.json").read_text())
    st = meta["statistics"]
    assert meta["synthetic"] is True and meta["block"] == "A" and meta["distance_m"] == 21.0
    assert meta["camera"]["ids_peak"]["pixel_format"] == "Mono8"  # runtime settings from cuasp
    assert st["frames_written"] == len(list((seq_dirs[0] / "frames").glob("*.jpg"))) > 15
    assert st["dropped_capture"] == st["dropped_writer"] == 0
    # the synthetic frames use the requested size, so the size reported is the frame's own
    req = meta["camera"]["ids_peak"]
    assert (st["frame_width"], st["frame_height"]) == (req["width"], req["height"])
    cal = meta["camera"]["calibration_summary"]
    assert cal["image_width"] > 0 and cal["fx"] > 0
    calib_mismatch = (cal["image_width"], cal["image_height"]) != (req["width"], req["height"])
    assert any("calibration" in n for n in meta["checks"]) == calib_mismatch
    assert "readback" not in meta["camera"]  # synthetic: no camera to read back
    out = capsys.readouterr().out
    assert "saved" in out and "gray levels" in out


class _ListProducer:
    """Delivers the given frames once, with consecutive sequence numbers."""

    def __init__(self, frames):
        self._frames = frames

    def start(self):
        pass

    def stop(self):
        pass

    def wait_newer(self, last_sequence, timeout_s):
        if last_sequence >= len(self._frames):
            time.sleep(0.01)
            return None
        return SimpleNamespace(sequence=last_sequence + 1, gray=self._frames[last_sequence],
                               ts_capture_ns=(last_sequence + 1) * 20_000_000)


def test_summary_reports_the_actual_frame_size_and_gray_levels(rec, tmp_path):
    frames = [np.full((48, 64), 40, np.uint8) for _ in range(30)]
    _, _, s = _run(rec, tmp_path, _ListProducer(frames), max_frames=30)
    assert (s["frame_width"], s["frame_height"]) == (64, 48)
    assert s["frame_size_changes"] == 0
    g = s["gray_levels"]
    assert g["samples"] == 2  # frames 0 and 25
    assert g["mean"] == g["p1"] == g["p50"] == g["p99"] == 40 and g["saturated_pct_max"] == 0


def test_frame_size_changes_are_counted(rec, tmp_path):
    frames = [np.zeros((48, 64), np.uint8)] * 3 + [np.zeros((40, 64), np.uint8)] * 2
    _, _, s = _run(rec, tmp_path, _ListProducer(frames), max_frames=5)
    assert (s["frame_width"], s["frame_height"]) == (64, 48)  # the first frame's size
    assert s["frame_size_changes"] == 2


def test_gray_levels_of_a_half_saturated_frame(rec):
    gray = np.zeros((64, 64), np.uint8)
    gray[:, 32:] = 255
    mean, p1, p50, p99, sat = rec.gray_levels(gray, step=1)
    assert mean == pytest.approx(127.5) and p1 == 0 and p99 == 255 and sat == pytest.approx(50.0)


class _Node:
    def __init__(self, value=None, entry=None, maximum=None):
        self._value, self._entry, self._max = value, entry, maximum

    def Value(self):
        if self._value is None:
            raise RuntimeError("not a value node")
        return self._value

    def CurrentEntry(self):
        if self._entry is None:
            raise RuntimeError("not an enumeration")
        return SimpleNamespace(SymbolicValue=lambda: self._entry)

    def Maximum(self):
        if self._max is None:
            raise RuntimeError("no maximum")
        return self._max


class _NodeMap:
    def __init__(self, nodes):
        self._nodes = nodes

    def FindNode(self, name):
        if name not in self._nodes:
            raise KeyError(name)
        return self._nodes[name]


def test_read_nodes_reads_values_enumerations_and_maxima_without_setting_anything(rec):
    nm = _NodeMap({"Width": _Node(2040, maximum=2040), "PixelFormat": _Node(entry="Mono8"),
                   "AcquisitionFrameRate": _Node(43.22, maximum=43.22), "DeviceModelName": _Node("U3-33DxXCP-M")})
    out = rec.read_nodes(nm)
    assert out["Width"] == 2040 and out["Width_max"] == 2040.0
    assert out["PixelFormat"] == "Mono8" and out["DeviceModelName"] == "U3-33DxXCP-M"
    assert out["AcquisitionFrameRate_max"] == pytest.approx(43.22)
    assert "Height" not in out  # missing nodes are skipped


def test_consistency_checks(rec):
    ok = {"frame_width": 2040, "frame_height": 1536, "frame_size_changes": 0,
          "gray_levels": {"p99": 200.0, "saturated_pct_max": 0.0}}
    assert rec.consistency_checks(ok, (2040, 1536), (2040, 1536)) == []
    notes = rec.consistency_checks(ok, (2592, 1944), (2040, 1536))  # the 2026-11-16 camera test
    assert len(notes) == 1 and "2592x1944" in notes[0] and "readback" in notes[0]
    dark = {**ok, "gray_levels": {"p99": 35.0, "saturated_pct_max": 2.5}}
    notes = rec.consistency_checks(dark, (2040, 1536), (2592, 1944))
    assert len(notes) == 3
    assert any("calibration" in n for n in notes) and any("dark" in n for n in notes) and any("saturated" in n for n in notes)


def test_invalid_names_are_refused(rec, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["record_ids.py", "--name", "../escape", "--fake", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        rec.main()
