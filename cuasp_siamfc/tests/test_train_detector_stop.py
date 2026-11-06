"""Stop/resume rules of tools/train_detector.py (no Ultralytics needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "train_detector.py"


@pytest.fixture(scope="module")
def td():
    spec = importlib.util.spec_from_file_location("train_detector", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_stop_file_never_stops(td, tmp_path):
    assert not td.should_stop(tmp_path / "STOP", 1, 40, False)


def test_stop_file_stops_between_epochs(td, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("x")
    assert td.should_stop(stop, 1, 40, False)
    assert td.should_stop(stop, 39, 40, False)


def test_stop_ignored_after_final_epoch(td, tmp_path):
    # A checkpoint of the final epoch cannot be resumed: let the run finish normally.
    stop = tmp_path / "STOP"
    stop.write_text("x")
    assert not td.should_stop(stop, 40, 40, False)


def test_stop_ignored_when_run_already_ending(td, tmp_path):
    # Early stopping already ends the run; raising would skip the final evaluation for nothing.
    stop = tmp_path / "STOP"
    stop.write_text("x")
    assert not td.should_stop(stop, 12, 40, True)


def test_request_stop_writes_file_in_run_folder(td, tmp_path, monkeypatch):
    (tmp_path / "myrun").mkdir()
    monkeypatch.setattr("sys.argv", ["train_detector.py", "--project", str(tmp_path), "--name", "myrun", "--request-stop"])
    assert td.main() == 0
    assert (tmp_path / "myrun" / "STOP").exists()


def test_request_stop_without_run_folder_fails(td, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_detector.py", "--project", str(tmp_path), "--name", "missing", "--request-stop"])
    assert td.main() == 2


def test_new_run_refuses_existing_folder(td, tmp_path, monkeypatch):
    (tmp_path / "myrun").mkdir()
    monkeypatch.setattr("sys.argv", ["train_detector.py", "--data", "x.yaml", "--project", str(tmp_path), "--name", "myrun"])
    assert td.main() == 2


def test_resume_without_checkpoint_fails(td, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_detector.py", "--data", "x.yaml", "--project", str(tmp_path), "--name", "myrun", "--resume"])
    assert td.main() == 2
