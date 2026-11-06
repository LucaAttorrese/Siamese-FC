"""The device crop used by the tracker must equal the NumPy reference crop exactly.

``SiamFCTracker._crop_to_device`` copies only the in-image part of the frame in
its own dtype (uint8) and pads and converts it on the device. This must give the
same values as ``crop_with_padding`` (float32 crop built on the CPU), including
at the image borders and fully outside the image.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from suncubes.siamfc.tracker import SiamFCTracker, crop_window, crop_with_padding  # noqa: E402


def _bare_tracker(avg: np.ndarray, device: str = "cpu") -> SiamFCTracker:
    """Tracker instance with only what ``_crop_to_device`` needs (no network)."""
    tr = SiamFCTracker.__new__(SiamFCTracker)
    tr.device = torch.device(device)
    tr._fill = torch.as_tensor(avg, dtype=torch.float32, device=tr.device).view(-1, 1, 1)
    return tr


CASES = [
    ((60.0, 80.0), 31),  # fully inside
    ((60.4, 80.6), 32),  # even side, fractional position
    ((2.0, 3.0), 41),  # top-left border
    ((118.0, 158.0), 41),  # bottom-right border
    ((60.0, 80.0), 301),  # crop larger than the image
    ((-500.0, -500.0), 21),  # fully outside
]


@pytest.mark.parametrize("pos,side", CASES)
@pytest.mark.parametrize("channels", [1, 3])
def test_device_crop_equals_numpy_reference(pos, side, channels):
    rng = np.random.default_rng(side)
    frame = rng.integers(0, 256, (120, 160, channels), dtype=np.uint8)
    avg = frame.reshape(-1, channels).mean(axis=0).astype(np.float32)
    ref = crop_with_padding(frame, pos, side, avg)  # (H, W, C) float32
    out = _bare_tracker(avg)._crop_to_device(frame, pos, side)  # (C, H, W) float32
    assert out.dtype == torch.float32 and tuple(out.shape) == (channels, side, side)
    np.testing.assert_array_equal(out.permute(1, 2, 0).numpy(), ref)


def test_crop_window_reports_in_image_part():
    y0, x0, sy0, sy1, sx0, sx1 = crop_window((10, 10), (0.0, 0.0), 5)
    assert (y0, x0) == (-3, -3)
    assert (sy0, sy1, sx0, sx1) == (0, 2, 0, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_crop_on_cuda_matches_cpu():
    frame = np.random.default_rng(0).integers(0, 256, (200, 300, 1), dtype=np.uint8)
    avg = frame.reshape(-1, 1).mean(axis=0).astype(np.float32)
    cpu = _bare_tracker(avg, "cpu")._crop_to_device(frame, (10.0, 290.0), 101)
    gpu = _bare_tracker(avg, "cuda")._crop_to_device(frame, (10.0, 290.0), 101)
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=0, atol=0)
