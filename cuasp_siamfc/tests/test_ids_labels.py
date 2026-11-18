"""IDS labelling tools (V4): tools/ids_labels.py, tools/prelabel_ids.py, tools/review_labels.py.

No models, no windows: fake detector outputs and a fake tracker.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))

import ids_labels as L  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = L.PrelabelParams()


# ---------------------------------------------------------------------------- geometry and gain


def test_pixel_centre_and_edge_conversions_are_inverse():
    assert L.box_from_pixel_centre((9.5, -0.5, 10, 4)) == (10.0, 0.0, 10.0, 4.0)
    assert L.box_to_pixel_centre(L.box_from_pixel_centre((3.25, 7.0, 5, 6))) == (3.25, 7.0, 5.0, 6.0)


def test_iou_and_interpolation():
    assert L.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert L.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert L.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    mid = L.interpolate_box((0, 0, 10, 10), (100, 50, 30, 10), 0.5)
    assert L.box_centre(mid) == pytest.approx((60.0, 30.0)) and mid[2:] == pytest.approx((20.0, 10.0))


def test_auto_gain_brightens_dark_frames_only_and_never_saturates():
    dark = np.full((64, 64), 25, np.uint8)
    assert L.auto_gain(dark, target_mean=80, max_gain=3.0, step=1) == pytest.approx(3.0)  # 3.2 capped
    assert L.auto_gain(np.full((64, 64), 120, np.uint8), step=1) == 1.0  # never darkens
    hot = np.full((64, 64), 30, np.uint8)
    hot[:8, :8] = 200  # 1.6 % of pixels at 200: p99.5 = 200 -> gain <= 1.25
    assert L.auto_gain(hot, step=1) == pytest.approx(1.25)
    out = L.apply_gain(np.array([[0, 10, 100, 200]], np.uint8), 2.0)
    np.testing.assert_array_equal(out, [[0, 20, 200, 255]])


# ---------------------------------------------------------------------------- label track


def test_label_track_edits_and_undo():
    t = L.LabelTrack(10)
    t.set_box(2, (10, 20, 30, 40), "manual")
    assert t.present[2] == 1 and t.source[2] == "manual" and t.dirty
    with pytest.raises(ValueError):
        t.set_box(3, (0, 0, 0, 5), "manual")
    with pytest.raises(ValueError):
        t.set_box(3, (0, 0, 5, 5), "bogus")
    with pytest.raises(IndexError):
        t.set_absent(10)
    t.set_box(6, (40, 20, 30, 40), "manual")
    assert t.interpolate(2, 6) == 3
    assert t.source[4] == "interp" and L.box_centre(t.box[4]) == pytest.approx((40.0, 40.0))
    t.set_range_absent(7, 9)
    assert not t.present[7:10].any() and t.reviewed[7:10].all()
    assert t.undo() == (7, 9) and t.source[7] == "none"
    assert t.undo() == (3, 5) and t.source[4] == "none"
    with pytest.raises(ValueError):
        t.interpolate(0, 6)  # frame 0 has no box


def test_confirm_clears_flags_and_next_flagged_finds_segments():
    t = L.LabelTrack(20)
    t.flag[3:6] = 1
    t.flag[10:12] = 1
    assert t.next_flagged(-1) == 3
    assert t.next_flagged(3) == 10  # leaves the current segment first
    assert t.next_flagged(11) is None
    assert t.next_flagged(15, forward=False) == 10
    assert t.next_flagged(10, forward=False) == 3
    t.mark_reviewed(3, 5)
    assert t.next_flagged(-1, unreviewed_only=True) == 10
    t.confirm(10, 11)
    assert t.counts()["flagged"] == 3 and t.next_flagged(-1, unreviewed_only=True) is None


def test_write_run_is_one_undoable_edit():
    t = L.LabelTrack(10)
    t.write_run([4, 5, 6], [(1, 1, 5, 5), None, (2, 2, 5, 5)], "retrack", [0.9, None, 0.8], [0, 1, 0])
    assert t.source[4] == "retrack" and t.source[5] == "absent" and t.flag[5] == 1
    assert t.undo() == (4, 6) and t.source[4] == t.source[5] == "none"


def test_save_load_round_trip(tmp_path):
    t = L.LabelTrack(4)
    t.set_box(0, (1.5, 2.25, 30, 40), "det", 0.75, 0, undoable=False)
    t.set_absent(1, "absent", 1, undoable=False)
    t.set_box(2, (5, 6, 7, 8), "fwd", None, 1, undoable=False)
    t.reviewed[3] = 1
    path = tmp_path / "rec.csv"
    L.save_labels(t, path, {"tool_version": "x"})
    assert not t.dirty
    u = L.load_labels(path)
    np.testing.assert_allclose(u.box[0], (1.5, 2.25, 30, 40))
    assert u.source == ["det", "absent", "fwd", "none"]
    assert u.flag.tolist() == [0, 1, 1, 0] and u.reviewed.tolist() == [0, 0, 0, 1]
    assert u.score[0] == pytest.approx(0.75) and math.isnan(u.score[2])
    meta = L.load_meta(path)
    assert meta["tool_version"] == "x" and meta["counts"]["present"] == 2
    assert L.has_user_work(u)  # frame 3 reviewed
    assert not L.has_user_work(L.LabelTrack(3))
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        L.load_labels(path)


def test_detections_cache_round_trip(tmp_path):
    dets = [[((1, 2, 3, 4), 0.2), ((5, 6, 7, 8), 0.9)], [], [((9, 9, 9, 9), 0.5)]]
    L.save_detections(tmp_path / "d.csv", dets, [2.0, 2.0, 1.0])
    got, gains = L.load_detections(tmp_path / "d.csv", 3)
    assert gains == [2.0, 2.0, 1.0]
    assert got[0][0][1] == pytest.approx(0.9) and got[0][1][1] == pytest.approx(0.2)  # sorted by score
    assert got[1] == [] and got[2][0][0] == (9.0, 9.0, 9.0, 9.0)
    with pytest.raises(ValueError):
        L.load_detections(tmp_path / "d.csv", 2)


# ---------------------------------------------------------------------------- pre-label fusion


def _det(x, s=0.8, y=100.0, w=40.0, h=20.0):
    return [((x, y, w, h), s)]


def test_select_anchors_needs_temporal_confirmation():
    dets = [_det(100), _det(102), [], [], [], [], _det(500), [], [], [], _det(300, 0.3)]
    anchors = L.select_anchors(dets, P)
    assert sorted(anchors) == [0, 1]  # frame 6 is isolated, frame 10 is weak


def test_gap_ranges():
    assert L.gap_ranges([2, 3, 7], 10) == [(0, 1, None, 2), (4, 6, 3, 7), (8, 9, 7, None)]
    assert L.gap_ranges([], 5) == [(0, 4, None, None)]
    assert L.gap_ranges([0, 1, 2], 3) == []


def test_fuse_gap_rules():
    s = L.TrackStep
    nan = float("nan")
    b1, b2, far = (100, 100, 40, 20), (104, 100, 40, 20), (400, 100, 40, 20)
    dets = [[] for _ in range(10)]
    # both directions agree -> blend weighted by distance to the anchors (frames 1..3 between anchors 0 and 4)
    out = L.fuse_gap(1, 3, 0, 4, [s(b1, 0.9)] * 3, [s(b2, 0.8)] * 3, dets, P)
    assert [o[1] for o in out] == ["blend"] * 3 and all(o[3] == 0 for o in out)
    assert L.box_centre(out[0][0])[0] == pytest.approx(120 + 1.0)  # 1/4 of the way from b1 to b2
    # disagreement -> nearer anchor, flagged
    out = L.fuse_gap(1, 3, 0, 4, [s(b1, 0.9)] * 3, [s(far, 0.9)] * 3, dets, P)
    assert [o[1] for o in out] == ["fwd", "fwd", "bwd"] and all(o[3] == 1 for o in out)
    # one direction only: trusted near its anchor, flagged far from it
    p = L.PrelabelParams(trust_frames=2)
    out = L.fuse_gap(1, 4, 0, None, [s(b1, 0.9)] * 4, [], dets, p)
    assert [o[3] for o in out] == [0, 0, 1, 1]
    # no tracking: absent; flagged between two anchors or with a weak detection, else not
    out = L.fuse_gap(1, 2, 0, 3, [s(None, nan)] * 2, [s(None, nan)] * 2, dets, P)
    assert out == [(None, "absent", None, 1)] * 2
    out = L.fuse_gap(1, 2, 0, None, [s(None, nan)] * 2, [], dets, P)
    assert out == [(None, "absent", None, 0)] * 2
    dets_w = [[] for _ in range(10)]
    dets_w[2] = [((700, 700, 40, 20), 0.2)]
    out = L.fuse_gap(1, 2, 0, None, [s(None, nan)] * 2, [], dets_w, P)
    assert [o[3] for o in out] == [0, 1]
    # a weak detection that agrees with the tracker replaces the tracker box
    dets_w[2] = [((101, 100, 40, 20), 0.3)]
    out = L.fuse_gap(1, 2, 0, None, [s(b1, 0.9)] * 2, [], dets_w, P)
    assert out[1] == ((101, 100, 40, 20), "det_weak", 0.3, 0)


def test_tracker_confidence_matches_the_runtime_rule():
    c = L.TrackerConfidence(ref_frames=3)
    assert c(10.0, 100.0) == pytest.approx(1.0)
    assert c(10.0, 100.0) == pytest.approx(1.0)
    assert c(10.0, 100.0) == pytest.approx(1.0)
    assert c(5.0, 25.0) == pytest.approx(math.sqrt(0.5 * 0.25))  # references frozen after 3 updates
    assert c(50.0, 500.0) == pytest.approx(1.0)  # clipped
    assert math.isnan(c(float("nan"), 1.0))


def test_track_run_ends_on_loss_and_stops_consuming():
    consumed = []

    def steps(seq):
        for item in seq:
            consumed.append(item)
            yield item

    good, weak = ((100, 100, 40, 20), 10.0, 100.0), ((100, 100, 40, 20), 1.0, 1.0)
    p = L.PrelabelParams(lost_frames=2)
    out = L.track_run((0, 0, 1, 1), steps([good] * 5 + [weak, good, weak, weak, good, good]), (1000, 1000), p)
    assert [o.box is not None for o in out] == [True] * 5 + [False, True, False, False]
    assert len(consumed) == 9  # the generator was not consumed past the end of the run
    outside = ((2000, 100, 40, 20), 10.0, 100.0)
    out = L.track_run((0, 0, 1, 1), iter([good, outside, good]), (1000, 1000), p)
    assert len(out) == 2 and out[1].box is None


# ---------------------------------------------------------------------------- prelabel end to end


class _FakeTracker:
    """Moves the box 2 px right per update; peak drops when told to."""

    def __init__(self, lose_after=None):
        self.lose_after = lose_after

    def init(self, frame, center_xy, size_wh):
        self.c = list(center_xy)
        self.size = size_wh
        self.k = 0

    def update(self, frame):
        self.k += 1
        self.c[0] += 2.0
        peak = 0.0 if self.lose_after is not None and self.k > self.lose_after else 10.0
        return SimpleNamespace(center_xy=tuple(self.c), size_wh=self.size, peak=peak, apce=100.0)


def test_prelabel_fills_gaps_between_anchors(monkeypatch):
    pre = _load("prelabel_ids")
    n = 12
    frames = [Path(f"f{i}.jpg") for i in range(n)]
    monkeypatch.setattr(pre, "read_gray", lambda path: np.zeros((200, 1000), np.uint8))
    # anchors at 0-1 and 10-11, box moving 2 px per frame like the fake tracker
    dets = [_det(100 + 2 * f) if f in (0, 1, 10, 11) else [] for f in range(n)]
    track = pre.prelabel(frames, dets, _FakeTracker(), P, (1000, 200))
    assert track.source[0] == "det" and track.source[11] == "det"
    # backward tracking from frame 10 moves right as well (fake), so the two directions drift apart
    # slowly; near each anchor they agree with the truth within a few px
    assert all(track.present[2:10]) and track.counts()["absent"] == 0
    assert L.box_centre(track.box[2])[0] == pytest.approx(124.0, abs=10)


def test_prelabel_marks_absent_after_the_tracker_loses_the_target(monkeypatch):
    pre = _load("prelabel_ids")
    n = 20
    frames = [Path(f"f{i}.jpg") for i in range(n)]
    monkeypatch.setattr(pre, "read_gray", lambda path: np.zeros((200, 1000), np.uint8))
    dets = [_det(100) if f in (0, 1) else [] for f in range(n)]
    track = pre.prelabel(frames, dets, _FakeTracker(lose_after=3), L.PrelabelParams(lost_frames=2), (1000, 200))
    assert track.present[2:5].all()  # tracked for 3 frames
    assert not track.present[5:].any() and set(track.source[5:]) == {"absent"}
    assert not track.flag[5:].any()  # open-ended gap, no detection: absent, not flagged


def test_border_distance_and_detection_support():
    assert L.border_distance((10, 20, 30, 40), (100, 100)) == 10
    assert L.border_distance((80, 20, 30, 40), (100, 100)) == -10  # crosses the right edge
    dets = [[] for _ in range(40)]
    dets[30] = _det(100)
    box = (105, 100, 40, 20)  # centre 5 px from the detection's
    assert L.detection_support(box, dets, 20, 10, 0.75)
    assert not L.detection_support(box, dets, 19, 10, 0.75)  # frame 30 out of the window
    assert not L.detection_support((200, 100, 40, 20), dets, 30, 10, 0.75)  # too far


def test_flag_for_review_edges_and_unsupported_tracker_boxes():
    # B3/B4 (2026-11-17): after the drone left the frame SiamFC stayed on the border or on a stain
    t = L.LabelTrack(6)
    mid, edge = (400, 300, 40, 20), (400, 3, 40, 20)
    t.set_box(0, mid, "det", 0.9, 0, undoable=False)
    t.set_box(1, edge, "det", 0.9, 0, undoable=False)  # strong anchor at the edge: trusted
    t.set_box(2, edge, "det_weak", 0.3, 0, undoable=False)  # weak detection at the edge: flagged
    t.set_box(3, mid, "fwd", 0.6, 0, undoable=False)  # supported by the detection of frame 0
    t.set_box(4, (900, 300, 40, 20), "blend", 0.6, 0, undoable=False)  # nothing detected near it
    t.set_absent(5, "absent", 0, undoable=False)
    dets = [[(mid, 0.9)], [(edge, 0.9)], [(edge, 0.3)], [], [], []]
    added = L.flag_for_review(t, dets, (1000, 800), L.PrelabelParams())
    assert t.flag.tolist() == [0, 0, 1, 0, 1, 0] and added == 2
    assert L.flag_for_review(t, dets, (1000, 800), L.PrelabelParams()) == 0  # idempotent


# ---------------------------------------------------------------------------- review tool helpers


def test_review_tool_helpers():
    rv = _load("review_labels")
    t = L.LabelTrack(5)
    t.set_box(0, (1, 1, 5, 5), "det", undoable=False)
    t.set_box(1, (1, 1, 5, 5), "fwd", undoable=False)
    t.set_box(2, (1, 1, 5, 5), "manual", undoable=False)
    t.set_absent(3, undoable=False)
    t.set_absent(4, flag=1, undoable=False)
    assert rv.timeline_codes(t).tolist() == [0, 2, 3, 1, 4]
    assert rv.box_colour("det", 1) == rv.RED and rv.box_colour("retrack", 0) == rv.MAGENTA
    assert any("re-track" in line for line in rv.HELP_LINES)


def test_review_keys_ignore_caps_lock():
    # 2026-11-18: with Caps Lock on only the space bar worked
    rv = _load("review_labels")
    assert rv.key_char(ord("M"), caps_lock=True) == "m"
    assert rv.key_char(ord("T"), caps_lock=True) == "t"  # forward, as without Caps Lock
    assert rv.key_char(ord("t"), caps_lock=True) == "T"  # Shift + t with Caps Lock: backward
    assert rv.key_char(ord("N"), caps_lock=False) == "N"
    assert rv.key_char(ord(" "), caps_lock=True) == " " and rv.key_char(ord("+"), caps_lock=True) == "+"
    assert rv.key_char(rv.K_LEFT, caps_lock=True) == "\0"
    r = SimpleNamespace(quit=False, track=L.LabelTrack(3))
    rv.caps_lock_on = lambda: True  # module loaded fresh for this test
    rv.Reviewer.on_key(r, ord("Q"))
    assert r.quit


def test_replay_reads_the_label_files(tmp_path):
    rp = _load("replay_siamfc_source")
    t = L.LabelTrack(3)
    t.set_box(0, (10, 20, 30, 40), "det", 0.9, undoable=False)
    t.set_absent(1, undoable=False)
    t.set_box(2, (11, 21, 31, 41), "manual", undoable=False)
    L.save_labels(t, tmp_path / "rec.csv")
    gt = rp.read_gt(tmp_path / "rec.csv")
    np.testing.assert_allclose(gt[0], (10, 20, 30, 40))
    assert np.isnan(gt[1]).all()
    np.testing.assert_allclose(gt[2], (11, 21, 31, 41))
