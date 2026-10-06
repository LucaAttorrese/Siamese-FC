"""Pure-math unit tests for camera.py's rotation and covariance helpers.

No I/O, no camera/motor hardware. These are exactly the functions where a
silent sign-flip or unit-conversion mistake would hide.
"""

import math

import numpy as np

from suncubes import camera as cam


def test_rotation_from_rotvec_zero_is_identity():
    assert np.allclose(cam._rotation_from_rotvec(np.zeros(3)), np.eye(3))


def test_rotation_from_rotvec_90deg_about_z_rotates_x_to_y():
    rotvec = np.array([0.0, 0.0, math.pi / 2.0])
    r = cam._rotation_from_rotvec(rotvec)
    rotated_x = r @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated_x, [0.0, 1.0, 0.0], atol=1e-9)


def test_rotation_from_rotvec_is_orthonormal():
    r = cam._rotation_from_rotvec(np.array([0.2, -0.4, 0.7]))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert math.isclose(float(np.linalg.det(r)), 1.0, abs_tol=1e-9)


def test_rotation_from_rpy_deg_zero_is_identity():
    assert np.allclose(cam._rotation_from_rpy_deg((0.0, 0.0, 0.0)), np.eye(3))


def test_rotation_from_rpy_deg_yaw_90_rotates_x_to_y():
    r = cam._rotation_from_rpy_deg((0.0, 0.0, 90.0))
    assert np.allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


_CAMERA_MATRIX = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 512.0], [0.0, 0.0, 1.0]])
_CORNERS = np.array([[100, 100], [140, 100], [140, 140], [100, 140]], dtype=np.float64)


def test_position_covariance_is_diagonal_and_positive():
    cfg = cam.MultiMarkerTrackingConfig()
    cov = cam._multi_marker_position_covariance_m2(
        _CORNERS, np.array([0.1, 0.05, 2.0]), _CAMERA_MATRIX,
        confidence=0.8, reproj_rmse_px=1.0, cfg=cfg,
    )
    assert cov.shape == (3, 3)
    assert np.all(np.diag(cov) > 0.0)
    assert np.allclose(cov - np.diag(np.diag(cov)), 0.0)


def test_position_covariance_grows_as_confidence_drops():
    cfg = cam.MultiMarkerTrackingConfig()
    p_c = np.array([0.1, 0.05, 2.0])
    cov_high_conf = cam._multi_marker_position_covariance_m2(
        _CORNERS, p_c, _CAMERA_MATRIX, confidence=0.8, reproj_rmse_px=1.0, cfg=cfg
    )
    cov_low_conf = cam._multi_marker_position_covariance_m2(
        _CORNERS, p_c, _CAMERA_MATRIX, confidence=0.02, reproj_rmse_px=1.0, cfg=cfg
    )
    assert np.all(np.diag(cov_low_conf) >= np.diag(cov_high_conf) - 1e-12)


def test_transform_covariance_to_laser_identity_is_noop():
    cfg = cam.MultiMarkerTrackingConfig()
    cov = cam._multi_marker_position_covariance_m2(
        _CORNERS, np.array([0.1, 0.05, 2.0]), _CAMERA_MATRIX,
        confidence=0.8, reproj_rmse_px=1.0, cfg=cfg,
    )
    cov_laser = cam._transform_covariance_to_laser(cov, np.eye(3))
    assert np.allclose(cov_laser, cov)


def test_angles_from_point_laser_boresight_is_zero():
    angles = cam._angles_from_point_laser(np.array([0.0, 0.0, 1.0]))
    assert angles is not None
    d_az, d_po = angles
    assert math.isclose(d_az, 0.0, abs_tol=1e-12)
    assert math.isclose(d_po, 0.0, abs_tol=1e-12)


def test_angles_from_point_laser_off_axis_x_gives_positive_azimuth():
    angles = cam._angles_from_point_laser(np.array([1.0, 0.0, 1.0]))
    assert angles is not None
    d_az, _d_po = angles
    assert math.isclose(d_az, math.pi / 4.0, abs_tol=1e-9)


def test_angles_from_point_laser_rejects_non_positive_z():
    assert cam._angles_from_point_laser(np.array([0.1, 0.1, 0.0])) is None
    assert cam._angles_from_point_laser(np.array([0.1, 0.1, -1.0])) is None


def test_point_c_to_laser_identity_transform():
    p_c = np.array([1.0, 2.0, 3.0])
    p_l = cam._point_c_to_laser(p_c, np.eye(3), np.zeros((3, 1)))
    assert np.allclose(p_l, p_c)


def test_point_c_to_laser_applies_translation():
    p_c = np.array([1.0, 2.0, 3.0])
    t_lc = np.array([[0.1], [0.2], [0.3]])
    p_l = cam._point_c_to_laser(p_c, np.eye(3), t_lc)
    assert np.allclose(p_l, [1.1, 2.2, 3.3])


def test_scale_intrinsics_to_runtime_pure_resize():
    k = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    k_runtime, sx, sy = cam._scale_intrinsics_to_runtime(k, 1000, 800, 500, 400)
    assert math.isclose(sx, 0.5)
    assert math.isclose(sy, 0.5)
    assert math.isclose(k_runtime[0, 0], 500.0)
    assert math.isclose(k_runtime[1, 1], 500.0)
    assert math.isclose(k_runtime[0, 2], 250.0)
    assert math.isclose(k_runtime[1, 2], 200.0)


def test_scale_intrinsics_to_runtime_same_size_is_noop():
    k = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    k_runtime, sx, sy = cam._scale_intrinsics_to_runtime(k, 1000, 800, 1000, 800)
    assert np.allclose(k_runtime, k)
    assert sx == 1.0 and sy == 1.0
