"""Tests for cuasp.py's CameraConfig construction and CameraAPI source routing.

The routing property (2+ configured marker ids -> MultiArucoCorrectionSourceAdapter,
1 id -> the original, untouched IdsOpenCvCorrectionSourceAdapter) is the
regression-safety guarantee the whole multi-marker migration depends on: it
proves single-marker deployments keep running through unmodified code.
"""

import dataclasses

import pytest

from suncubes import camera as cam


@pytest.fixture
def cuasp(cuasp_module):
    return cuasp_module


def test_build_camera_config_has_expected_shape(cuasp):
    cfg = cuasp._build_camera_config()
    assert cfg.enable is True
    assert cfg.mode == "ids_opencv"
    assert cfg.ids_opencv["aruco"]["target_marker_ids"] == list(cuasp.ARUCO_IDS)
    assert cfg.ids_opencv["aruco"]["dict_name"] == cuasp.ARUCO_DICT_NAME
    assert cfg.ids_opencv["aruco"]["marker_size_m"] == cuasp.ARUCO_MARKER_SIZE_M
    assert cfg.ids_opencv["calibration"]["path"] == str(cuasp.CALIBRATION_YAML)
    assert cfg.gating.conf_min == cuasp.CONFIDENCE_MIN
    assert cfg.gating.z_min_m == cuasp.Z_MIN_M
    assert cfg.gating.max_age_s == cuasp.MAX_AGE_S


def test_two_marker_config_routes_to_multi_marker_adapter(cuasp):
    cfg = cuasp._build_camera_config()
    api = cuasp.CameraAPI(cfg)
    assert type(api._app._source).__name__ == "MultiArucoCorrectionSourceAdapter"


def test_single_marker_config_routes_to_original_adapter(cuasp):
    cfg = cuasp._build_camera_config()
    single_marker_cfg = dataclasses.replace(
        cfg,
        ids_opencv={
            **cfg.ids_opencv,
            "aruco": {**cfg.ids_opencv["aruco"], "target_marker_ids": [0]},
        },
    )
    api = cam.CameraAPI(single_marker_cfg)
    assert type(api._app._source).__name__ == "IdsOpenCvCorrectionSourceAdapter"


def test_validate_configuration_passes_with_defaults(cuasp):
    cuasp._validate_configuration()  # must not raise


def test_validate_configuration_rejects_duplicate_aruco_ids(cuasp, monkeypatch):
    monkeypatch.setattr(cuasp, "ARUCO_IDS", (0, 0, 1))
    with pytest.raises(ValueError, match="ARUCO_IDS"):
        cuasp._validate_configuration()


def test_validate_configuration_folds_in_settings_validate_errors(cuasp, monkeypatch):
    from suncubes import settings
    monkeypatch.setattr(settings, "PORT", 99999)
    with pytest.raises(ValueError, match="PORT"):
        cuasp._validate_configuration()
