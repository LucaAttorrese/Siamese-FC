"""Tests for suncubes.settings.validate().

One positive case (today's real defaults must pass) and one negative case per
class of mistake a manual edit is likely to introduce -- validate() reads its
own module globals(), so monkeypatching module attributes is the correct way
to simulate a bad edit without touching the file on disk.
"""

import pytest

from suncubes import settings


def test_default_settings_are_valid():
    settings.validate()  # must not raise


def test_negative_klt_window_size_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "CV_CUAS_KLT_WIN_SIZE", 14)  # even, invalid
    with pytest.raises(ValueError, match="CV_CUAS_KLT_WIN_SIZE"):
        settings.validate()


def test_out_of_range_port_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "PORT", 99999)
    with pytest.raises(ValueError, match="PORT"):
        settings.validate()


def test_empty_ip_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "IP", "")
    with pytest.raises(ValueError, match="IP"):
        settings.validate()


def test_zero_sign_constant_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "POINTING_AZ_DIR_CMD", 0.0)
    with pytest.raises(ValueError, match="POINTING_AZ_DIR_CMD"):
        settings.validate()


def test_negative_marker_size_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "CV_ARUCO_MARKER_SIZE_M", -0.1)
    with pytest.raises(ValueError, match="CV_ARUCO_MARKER_SIZE_M"):
        settings.validate()


def test_invalid_cuda_mode_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "CV_CUAS_CUDA_MODE", "sometimes")
    with pytest.raises(ValueError, match="CV_CUAS_CUDA_MODE"):
        settings.validate()


def test_malformed_target_offset_vector_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "CV_CUAS_TARGET_MIDPOINT_OFFSET_CM", (0.0, 5.0))  # only 2 elements
    with pytest.raises(ValueError, match="CV_CUAS_TARGET_MIDPOINT_OFFSET_CM"):
        settings.validate()


def test_duplicate_target_ids_are_rejected(monkeypatch):
    monkeypatch.setattr(settings, "CV_ARUCO_TARGET_IDS", (0, 0, 1))
    with pytest.raises(ValueError, match="CV_ARUCO_TARGET_IDS"):
        settings.validate()


def test_multiple_errors_are_all_reported_together(monkeypatch):
    monkeypatch.setattr(settings, "PORT", 99999)
    monkeypatch.setattr(settings, "CV_CUAS_KLT_WIN_SIZE", 14)
    with pytest.raises(ValueError) as excinfo:
        settings.validate()
    message = str(excinfo.value)
    assert "PORT" in message
    assert "CV_CUAS_KLT_WIN_SIZE" in message
