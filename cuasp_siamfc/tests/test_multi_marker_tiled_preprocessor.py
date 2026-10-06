"""State-machine tests for camera.py's MultiMarkerTiledPreprocessor.

ROI/scout mode transitions are exactly the kind of logic that's easy to get
subtly wrong (an off-by-one in a miss counter, a wrong mode transition) and
hard to catch by reading the code alone.
"""

import cv2
import numpy as np

from suncubes import camera as cam

IMG_W, IMG_H = 1280, 960


def _preprocessor():
    fast_cfg = cam.load_fast_tracking_config()
    multi_cfg = cam.load_multi_marker_tracking_config()
    return cam.MultiMarkerTiledPreprocessor(cv2, fast_cfg, multi_cfg)


def test_starts_in_scout_mode():
    pre = _preprocessor()
    assert pre.mode == "scout"


def test_scout_mode_produces_at_least_one_tile():
    pre = _preprocessor()
    tiles = pre.prepare(np.zeros((IMG_H, IMG_W), dtype=np.uint8))
    assert len(tiles) >= 1


def test_both_markers_detected_switches_to_track_mode():
    pre = _preprocessor()
    quad0 = np.array([[600, 470], [640, 470], [640, 510], [600, 510]], dtype=np.float64)
    quad1 = np.array([[680, 470], [720, 470], [720, 510], [680, 510]], dtype=np.float64)
    pre.update((IMG_H, IMG_W), [quad0, quad1], detected_marker_count=2, expected_marker_count=2)
    assert pre.mode == "track"

    tiles = pre.prepare(np.zeros((IMG_H, IMG_W), dtype=np.uint8))
    assert len(tiles) == 1, "track mode adds no extra scout tiles by default"
    assert tiles[0].state == "track"


def test_one_of_two_markers_switches_to_partial_mode():
    pre = _preprocessor()
    quad0 = np.array([[600, 470], [640, 470], [640, 510], [600, 510]], dtype=np.float64)
    pre.update((IMG_H, IMG_W), [quad0], detected_marker_count=1, expected_marker_count=2)
    assert pre.mode == "partial"

    tiles = pre.prepare(np.zeros((IMG_H, IMG_W), dtype=np.uint8))
    assert len(tiles) >= 2, "partial mode must include the primary ROI plus extra scout tile(s)"
    assert tiles[0].state == "partial"


def test_repeated_total_miss_reverts_to_scout():
    pre = _preprocessor()
    quad0 = np.array([[600, 470], [640, 470], [640, 510], [600, 510]], dtype=np.float64)
    pre.update((IMG_H, IMG_W), [quad0], detected_marker_count=1, expected_marker_count=2)
    assert pre.mode == "partial"

    for _ in range(5):
        pre.update((IMG_H, IMG_W), [], detected_marker_count=0, expected_marker_count=2)
    assert pre.mode == "scout"


def test_brief_miss_expands_roi_instead_of_reverting_to_scout():
    pre = _preprocessor()
    quad0 = np.array([[600, 470], [640, 470], [640, 510], [600, 510]], dtype=np.float64)
    quad1 = np.array([[680, 470], [720, 470], [720, 510], [680, 510]], dtype=np.float64)
    pre.update((IMG_H, IMG_W), [quad0, quad1], detected_marker_count=2, expected_marker_count=2)
    assert pre.mode == "track"

    pre.update((IMG_H, IMG_W), [], detected_marker_count=0, expected_marker_count=2)
    assert pre.mode == "recover", "a single miss should expand the ROI, not jump straight to scout"


def test_partial_search_hint_is_used_for_scout_tile_ordering():
    pre = _preprocessor()
    pre.set_partial_search_hint((50.0, 50.0))
    quad0 = np.array([[600, 470], [640, 470], [640, 510], [600, 510]], dtype=np.float64)
    pre.update((IMG_H, IMG_W), [quad0], detected_marker_count=1, expected_marker_count=2)
    tiles = pre.prepare(np.zeros((IMG_H, IMG_W), dtype=np.uint8))
    scout_tiles = [t for t in tiles if t.state.endswith("_scout")]
    assert scout_tiles, "partial mode with a hint set should still schedule scout tiles"
    # The nearest scout tile to (50, 50) should be the top-left corner tile.
    x, y, _w, _h = scout_tiles[0].roi_xywh
    assert x < IMG_W / 2 and y < IMG_H / 2
