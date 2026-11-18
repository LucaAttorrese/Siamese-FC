"""tools/v4_report.py: V4 metrics from replay CSVs (synthetic replays, no models)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("v4_report", _TOOLS / "v4_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load()
FIELDS = ["frame", "state", "reason", "reliable", "confidence", "bx", "by", "bw", "bh", "d_az_rad", "d_po_rad",
          "gx", "gy", "gw", "gh", "err_px", "iou", "read_ms", "category"]


def _write(path: Path, rows: list[dict], fps: float = 10.0, size=(1000, 800)) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({k: r.get(k, "") for k in FIELDS} | {"frame": i})
    path.with_suffix(".json").write_text(json.dumps({"fps": fps, "settings": {"frame_size": list(size)}}))
    return path


def _row(gt=None, box=None, reliable=False, cat=""):
    r = {"reliable": int(reliable), "category": cat, "state": "tracking" if reliable else "searching", "reason": "ok" if reliable else "x"}
    if gt:
        r.update(zip(("gx", "gy", "gw", "gh"), gt))
    if box:
        r.update(zip(("bx", "by", "bw", "bh"), box))
    return r


def test_scenario_names():
    assert R.scenario_of("20261117_093341_B3_rapido_in_volo_r2") == "B3_rapido_in_volo"
    assert R.scenario_of("20261117_103240_B7_distanze_21_m") == "B7_distanze_21_m"


def test_clipping_and_border():
    assert R.clip_box_pc((-20.5, 10, 40, 20), (1000, 800)) == (-0.5, 10, 20, 20)
    assert R.clip_box_pc((-100, 10, 40, 20), (1000, 800)) is None
    gt = np.array([[100, 100, 50, 50], [2, 100, 50, 50], [np.nan] * 4])
    assert R.border_frames(gt, (1000, 800), 8.0).tolist() == [False, True, False]


def test_recording_metrics_events(tmp_path):
    g, gb = (100, 100, 40, 20), (-10, 100, 40, 20)  # gb: crossing the left border
    rows = (
        [_row(g)] * 2  # drone present, not yet locked
        + [_row(g, g, True, "correct")] * 4
        + [_row(None, g, True, "drone_absent")] * 3  # drone left, still pointing for 3 frames
        + [_row(None)] * 2
        + [_row(None, (500, 500, 40, 20), True, "drone_absent")] * 2  # locked on something, drone out
        + [_row(gb)] * 1 + [_row(gb, (5, 100, 40, 20), True, "correct")] * 2  # re-entry at the border, re-acquired after 1 frame
        + [_row(None)] * 1
        + [_row(g)] * 3  # re-entry never re-acquired
        + [_row(g, (400, 100, 40, 20), True, "wrong_target")] * 2
    )
    path = _write(tmp_path / "20261117_000000_X1_test_r1.csv", rows)
    m = R.recording_metrics(R.load_replay(path), 1000.0, 1000.0, 8.0, (1000, 800))
    assert m["present"] == 2 + 4 + 3 + 3 + 2 and m["border"] == 3
    assert m["correct"] == 6 and m["correct_inner"] == 4 and m["inner"] == 11
    assert m["drone_absent_after_exit"] == 3 and m["drone_absent_other"] == 2 and m["wrong_target"] == 2
    assert m["exit_delay_s"] == pytest.approx([0.3, 0.0])  # two exits: frames 6 and 16
    assert m["first_correct_s"] == pytest.approx(0.2)
    assert m["reacquisition_s"] == pytest.approx([0.1]) and m["missed_stretches"] == 1
    assert m["longest_wrong_or_other_absent_s"] == pytest.approx(0.2)
    assert m["longest_dangerous_s"] == pytest.approx(0.3)
    # clipped IoU at the border: box 5..45 vs visible part of the label -0.5..30
    assert min(m["iou_clipped"]) == pytest.approx(25 / 45.5)
    assert m["err_mrad"][4] == pytest.approx(15.0, rel=1e-3)  # 15 px at fx 1000
    d = R.digest(R.combine([m, m]))
    assert d["present"] == 2 * m["present"] and d["reacquisition_s"]["n"] == 2 and d["coverage"] == pytest.approx(6 / 14)


def test_main_writes_report(tmp_path, monkeypatch, capsys):
    g = (100, 100, 40, 20)
    _write(tmp_path / "20261117_000000_X1_test_r1.csv", [_row(g, g, True, "correct")] * 5)
    _write(tmp_path / "20261117_000001_X1_test_r2.csv", [_row(g)] + [_row(g, g, True, "correct")] * 4)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "A1.json").write_text(json.dumps({"frames": 626, "fps": 62.6, "reliable_frames": 0}))
    monkeypatch.setattr(R, "load_intrinsics", lambda path: (24000.0, 24000.0))
    out = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["v4_report.py", str(tmp_path), "--empty", str(empty), "--out", str(out)])
    assert R.main() == 0
    rep = json.loads(out.read_text())
    assert rep["per_scenario"]["X1_test"]["correct"] == 9 and rep["total"]["coverage"] == pytest.approx(0.9)
    assert rep["empty_scene"]["reliable_frames"] == 0 and rep["empty_scene"]["recordings"] == 1
    assert "TOTAL (2)" in capsys.readouterr().out
