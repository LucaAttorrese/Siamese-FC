"""Tests for cuasp.py's compute_pbvs_correction_from_camera.

This is the supervisor's rad -> deg conversion, sign map and transport clamp
-- AGENTS.md explicitly calls out "visual rad->deg and sign mapping" as a
priority test, since a flipped sign here drives the mount the wrong way.
"""

import math

import pytest


@pytest.fixture
def cuasp(cuasp_module):
    return cuasp_module


def test_unreliable_camera_correction_passes_through_unreliable(cuasp):
    correction = cuasp.CameraCorrection(False, "confidence_low", 0.0, 0.0, 0.2, 0.01, 0, 1.0, 123, 456)
    out = cuasp.compute_pbvs_correction_from_camera(correction)
    assert out.reliable is False
    assert out.reason == "confidence_low"


def test_non_finite_angles_are_rejected(cuasp):
    correction = cuasp.CameraCorrection(True, "ok", math.nan, 0.0, 0.9, 0.01, 0, 1.5, 123, 456)
    out = cuasp.compute_pbvs_correction_from_camera(correction)
    assert out.reliable is False
    assert out.reason == "pbvs_angles_not_finite"


def test_sign_map_and_rad_to_deg_conversion(cuasp):
    correction = cuasp.CameraCorrection(
        True, "ok", math.radians(2.0), math.radians(-1.0), 0.9, 0.01, 0, 1.5, 123, 456
    )
    out = cuasp.compute_pbvs_correction_from_camera(correction)
    assert out.reliable is True
    assert out.reason == "ok"

    expected_az_deg = (
        cuasp.PBVS_RAW_TO_CMD_AZ_SIGN * cuasp.POINTING_AZ_DIR_CMD * math.degrees(math.radians(2.0)) * cuasp.PBVS_GAIN
    )
    expected_po_deg = (
        cuasp.PBVS_RAW_TO_CMD_PO_SIGN * cuasp.POINTING_PO_DIR_CMD * math.degrees(math.radians(-1.0)) * cuasp.PBVS_GAIN
    )
    assert math.isclose(out.d_az_motor_units, expected_az_deg, abs_tol=1e-9)
    assert math.isclose(out.d_po_motor_units, expected_po_deg, abs_tol=1e-9)
    assert math.isclose(out.d_az_cmd_rad, math.radians(out.d_az_motor_units), abs_tol=1e-12)
    assert math.isclose(out.d_po_cmd_rad, math.radians(out.d_po_motor_units), abs_tol=1e-12)
    assert out.az_clamped is False
    assert out.po_clamped is False


def test_large_error_is_clamped_not_rejected(cuasp):
    huge_rad = math.radians(500.0)
    correction = cuasp.CameraCorrection(True, "ok", huge_rad, 0.0, 0.9, 0.01, 0, 1.5, 123, 456)
    out = cuasp.compute_pbvs_correction_from_camera(correction)
    assert out.reliable is True
    assert out.az_clamped is True
    assert abs(out.d_az_motor_units) <= max(0.1, cuasp.PBVS_MAX_ERROR_DEG) + 1e-9


def test_zero_error_is_a_valid_reliable_correction(cuasp):
    """Zero angular error is meaningful (on-target), not the same as
    'no correction available' -- KAS deadband/stop logic needs to see it."""
    correction = cuasp.CameraCorrection(True, "ok", 0.0, 0.0, 0.95, 0.01, 0, 1.5, 123, 456)
    out = cuasp.compute_pbvs_correction_from_camera(correction)
    assert out.reliable is True
    assert out.d_az_motor_units == 0.0
    assert out.d_po_motor_units == 0.0
