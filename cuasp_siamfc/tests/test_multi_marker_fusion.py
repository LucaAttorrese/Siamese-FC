"""Tests for camera.py's rigid multi-marker target-point fusion.

Includes a dedicated regression test for the target-offset values that were
silently reset to (0,0,0) during the multi-marker migration and then restored
from the original standalone script into settings.py -- this test is what
would have caught that regression the moment it happened.
"""

import math

import numpy as np
import pytest

from suncubes import camera as cam


_CORNERS = np.array([[100, 100], [140, 100], [140, 140], [100, 140]], dtype=np.float64)


def _make_observation(marker_id, position_camera_m, confidence=0.9, rvec=(0.0, 0.0, 0.0)):
    return cam.ArucoObservation(
        marker_id=marker_id,
        dict_name="DICT_5X5_100",
        position_camera_m=np.asarray(position_camera_m, dtype=np.float64),
        velocity_camera_mps=np.zeros(3),
        covariance_camera_m2=np.eye(3) * (0.01 ** 2),
        confidence=confidence,
        reproj_rmse_px=1.0,
        rvec_cm=np.asarray(rvec, dtype=np.float64),
        corners_px=_CORNERS,
        ts_capture_ns=0,
    )


@pytest.fixture
def cfg():
    return cam.MultiMarkerTrackingConfig()


def test_full_fusion_returns_geometric_midpoint_with_zero_offset(cfg):
    obs0 = _make_observation(0, (-0.09, 0.0, 2.0))
    obs1 = _make_observation(1, (0.09, 0.0, 2.0))
    target = cam.compute_multi_marker_target_point([obs0, obs1], expected_marker_count=2, cfg=cfg)
    assert target is not None
    assert np.allclose(target.position_camera_m, [0.0, 0.0, 2.0], atol=1e-9)
    assert target.mode.startswith("midpoint_target_frame_ref_")
    assert math.isclose(target.confidence, 0.9, rel_tol=1e-9)
    assert set(target.contributing_markers) == {0, 1}


def test_full_fusion_uses_configured_reference_marker_for_frame_orientation(cfg):
    cfg = cam.MultiMarkerTrackingConfig(target_frame_reference_marker_id=1)
    obs0 = _make_observation(0, (-0.09, 0.0, 2.0))
    obs1 = _make_observation(1, (0.09, 0.0, 2.0))
    target = cam.compute_multi_marker_target_point([obs0, obs1], expected_marker_count=2, cfg=cfg)
    assert target is not None
    assert target.mode == "midpoint_target_frame_ref_1"
    assert target.reference_marker_id == 1


def test_single_marker_fallback_when_one_of_two_missing(cfg):
    obs0 = _make_observation(0, (-0.09, 0.0, 2.0))
    target = cam.compute_multi_marker_target_point([obs0], expected_marker_count=2, cfg=cfg)
    assert target is not None
    assert target.mode == "single_marker_0_offset"
    assert target.reference_marker_id == 0
    assert target.confidence < obs0.confidence, "single-marker fallback must scale confidence down"
    assert np.all(
        np.diag(target.covariance_camera_m2) >= np.diag(obs0.covariance_camera_m2)
    ), "single-marker fallback must inflate covariance"


def test_single_marker_fallback_disabled_yields_no_target():
    cfg = cam.MultiMarkerTrackingConfig(single_marker_fallback_enable=False)
    obs0 = _make_observation(0, (-0.09, 0.0, 2.0))
    assert cam.compute_multi_marker_target_point([obs0], expected_marker_count=2, cfg=cfg) is None


def test_no_observations_yields_no_target(cfg):
    assert cam.compute_multi_marker_target_point([], expected_marker_count=2, cfg=cfg) is None


def test_non_finite_observation_is_excluded(cfg):
    obs_bad = _make_observation(0, (math.nan, 0.0, 2.0))
    assert cam.compute_multi_marker_target_point([obs_bad], expected_marker_count=2, cfg=cfg) is None


# ---------------------------------------------------------------------------
# Regression test: the (0,5,0) / (-18,5,0) / (18,5,0) cm rig offsets restored
# into settings.py after the multi-marker migration silently zeroed them out.
# ---------------------------------------------------------------------------


def test_settings_default_target_offsets_match_restored_rig_values():
    """Guards against the exact regression this session already hit once.

    yesterday's migration of the CUAS pipeline into camera.py used (0,0,0) as
    the code-level default for every target-offset field instead of the
    original standalone script's real rig measurements. settings.py now
    carries the restored values -- this test fails loudly if a future change
    reverts camera.py's fallback or settings.py's values back to zero.

    2026-09-21: real target marker ids renumbered from (0, 1) to (1, 2) (see
    settings.py's CV_ARUCO_TARGET_IDS) -- these per-id offset fields were
    renumbered to match (id 0 is no longer physically used and now correctly
    defaults to (0,0,0); id 1 keeps its old value as the new frame-reference
    marker; id 2 carries the value the old, retired id 0 had).
    """
    multi_cfg = cam.load_multi_marker_tracking_config()
    assert multi_cfg.target_midpoint_offset_cm == (0.0, 5.0, 0.0)
    assert cam._multi_marker_offset_cm(0) == (0.0, 0.0, 0.0)
    assert cam._multi_marker_offset_cm(1) == (18.0, 5.0, 0.0)
    assert cam._multi_marker_offset_cm(2) == (-18.0, 5.0, 0.0)


def test_full_fusion_with_restored_offsets_shifts_target_off_geometric_midpoint():
    """With the real (non-zero) offsets, the fused point must not sit exactly
    at the raw geometric midpoint of the two markers -- unlike the all-zero
    default this pipeline briefly and silently shipped with."""
    multi_cfg = cam.load_multi_marker_tracking_config()
    obs0 = _make_observation(0, (-0.18, 0.0, 2.0))
    obs1 = _make_observation(1, (0.18, 0.0, 2.0))
    target = cam.compute_multi_marker_target_point([obs0, obs1], expected_marker_count=2, cfg=multi_cfg)
    assert target is not None
    geometric_midpoint = np.array([0.0, 0.0, 2.0])
    assert not np.allclose(target.position_camera_m, geometric_midpoint, atol=1e-6)
    # The configured offset is +5 cm along the marker-frame Y axis with an
    # identity marker rotation (rvec=0) -> +0.05 m in camera Y.
    assert math.isclose(float(target.position_camera_m[1]), 0.05, abs_tol=1e-9)
