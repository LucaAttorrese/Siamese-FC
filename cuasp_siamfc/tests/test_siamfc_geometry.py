"""Pure-numpy geometry of the SiamFC port (no torch, no weights).

Covers the pieces where a 1-based MATLAB -> 0-based Python port typically goes
wrong: rounding, crop placement and padding, MATLAB bicubic resize weights,
Hann window, response-peak -> frame displacement.
"""

from __future__ import annotations

import numpy as np
import pytest

from suncubes.siamfc.imresize import imresize_np, resize_matrix

tracker_mod = pytest.importorskip("suncubes.siamfc.tracker", reason="tracker module imports torch")


def test_matlab_round_is_half_away_from_zero():
    r = tracker_mod.matlab_round
    assert [r(0.5), r(1.5), r(2.5), r(-0.5), r(-1.5), r(0.49)] == [1, 2, 3, -1, -2, 0]


@pytest.mark.parametrize("n_in,n_out", [(33, 264), (255, 127), (1200, 280), (17, 17), (7, 3)])
def test_resize_rows_sum_to_one_and_preserve_constants(n_in, n_out):
    m = resize_matrix(n_in, n_out)
    assert m.shape == (n_out, n_in)
    np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(m @ np.full(n_in, 7.0), 7.0, atol=1e-10)


def test_resize_identity_when_sizes_match():
    np.testing.assert_allclose(resize_matrix(10, 10), np.eye(10), atol=1e-12)


def test_upscale_reproduces_linear_ramp_away_from_borders():
    # Keys cubic (a = -0.5) reproduces linear functions exactly; MATLAB sample
    # positions are u = x/s + 0.5(1 - 1/s) (1-based).
    n_in, scale = 33, 8
    out = resize_matrix(n_in, n_in * scale) @ np.arange(1, n_in + 1, dtype=np.float64)
    x = np.arange(1, n_in * scale + 1)
    u = x / scale + 0.5 * (1 - 1 / scale)
    interior = (u > 3) & (u < n_in - 2)
    np.testing.assert_allclose(out[interior], u[interior], atol=1e-10)


def test_downscale_is_antialiased():
    checker = np.indices((64, 64)).sum(axis=0) % 2 * 255.0
    small = imresize_np(checker, (16, 16))
    assert np.ptp(small[2:-2, 2:-2]) < 1.0  # high frequency removed, mean kept
    assert abs(small.mean() - 127.5) < 1.0


def test_crop_reproduces_reference_minus_one_pixel_centre():
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    # 1-based MATLAB pos (5, 5) == 0-based (4, 4); side 3 -> round(5 - 2) = 3
    # (1-based) -> rows/cols 2..4 (0-based), i.e. centred on 0-based 3.
    crop = tracker_mod.crop_with_padding(img, (4.0, 4.0), 3, np.array([0.0]))[:, :, 0]
    np.testing.assert_array_equal(crop, img[2:5, 2:5])


def test_crop_pads_outside_with_fill_value():
    img = np.ones((6, 8), np.uint8) * 10
    crop = tracker_mod.crop_with_padding(img, (0.0, 0.0), 5, np.array([99.0]))[:, :, 0]
    assert crop.shape == (5, 5)
    assert (crop[:3, :] == 99).all() and (crop[:, :3] == 99).all()
    assert (crop[3:, 3:] == 10).all()


def test_crop_multichannel_fill():
    img = np.zeros((4, 4, 3), np.uint8)
    crop = tracker_mod.crop_with_padding(img, (-10.0, -10.0), 2, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(crop[0, 0], [1.0, 2.0, 3.0])


def test_hann_window_matches_matlab_symmetric_hann():
    n = 264
    w = tracker_mod.hann_window(n)
    k = np.arange(n)
    hann = 0.5 * (1 - np.cos(2 * np.pi * k / (n - 1)))
    np.testing.assert_allclose(w, np.outer(hann, hann) / np.outer(hann, hann).sum())
    assert w[0, 0] == 0.0 and abs(w.sum() - 1.0) < 1e-12


def test_displacement_zero_at_response_centre_and_scales_with_s_x():
    p = tracker_mod.SiamFCParams()
    n_up = p.score_size * p.response_up  # 264 -> centre (1-based) 132.5
    assert tracker_mod.displacement_in_frame(131, 131, p, 255.0)[0] == pytest.approx(-0.5 * 4 / 8)
    dy, dx = tracker_mod.displacement_in_frame(n_up - 1, 0, p, 510.0)
    k = p.total_stride / p.response_up * 510.0 / p.instance_size
    assert dy == pytest.approx((n_up - 132.5) * k)
    assert dx == pytest.approx((1 - 132.5) * k)


def test_params_validate_rejects_even_scales():
    with pytest.raises(ValueError):
        tracker_mod.params_with(num_scale=2).validate()
