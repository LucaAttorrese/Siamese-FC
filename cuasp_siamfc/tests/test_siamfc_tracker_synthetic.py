"""End-to-end SiamFC tracking on a synthetic Mono8 sequence (real converted weights).

Skipped when torch or the converted weights are not available (CI): the weights are
not version-controlled, see README "Network weights".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

WEIGHTS = Path(__file__).resolve().parents[2] / "pretrained" / "networks" / "baseline-conv5_gray_e100.pth"
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason=f"converted weights not found: {WEIGHTS}")


def _make_sequence(n_frames=40, size=(480, 640), seed=0):
    rng = np.random.default_rng(seed)
    background = rng.normal(110, 12, size).clip(0, 255)
    target = rng.integers(0, 255, (40, 56)).astype(np.float64)
    target[::8, :] = 30  # some structure
    centres, frames = [], []
    for i in range(n_frames):
        cx, cy = 220.0 + 3.0 * i, 200.0 + 1.5 * i + 10.0 * np.sin(i / 6.0)
        img = background.copy()
        x0, y0 = int(round(cx - 28)), int(round(cy - 20))
        img[y0:y0 + 40, x0:x0 + 56] = target
        frames.append(img.astype(np.uint8))
        centres.append((x0 + 27.5, y0 + 19.5))  # 0-based pixel-centre of the patch
    return frames, centres


def test_tracks_translating_patch_on_mono8_frames():
    from suncubes.siamfc.net import SiamFCNet  # noqa: PLC0415
    from suncubes.siamfc.tracker import SiamFCTracker  # noqa: PLC0415

    frames, centres = _make_sequence()
    tracker = SiamFCTracker(SiamFCNet.from_file(WEIGHTS), device="cpu")
    tracker.init(frames[0], centres[0], (56.0, 40.0))
    errors = []
    for frame, (cx, cy) in zip(frames[1:], centres[1:]):
        res = tracker.update(frame)
        errors.append(np.hypot(res.center_xy[0] - cx, res.center_xy[1] - cy))
        assert np.isfinite(res.peak) and res.apce > 0.0
    # Measured max error is ~1 px (README "Known reference quirks").
    assert max(errors) < 2.0, errors
