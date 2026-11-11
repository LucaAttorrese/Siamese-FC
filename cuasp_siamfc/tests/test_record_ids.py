"""tools/record_ids.py with the synthetic producer (no camera): frame accounting, losses, outputs."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

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
    assert "saved" in capsys.readouterr().out


def test_invalid_names_are_refused(rec, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["record_ids.py", "--name", "../escape", "--fake", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        rec.main()
