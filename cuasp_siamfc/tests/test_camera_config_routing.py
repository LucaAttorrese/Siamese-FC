"""Tests for cuasp.py's CameraConfig construction and CameraAPI source routing.

The routing property (2+ configured marker ids -> MultiArucoCorrectionSourceAdapter,
1 id -> the original, untouched IdsOpenCvCorrectionSourceAdapter) is the
regression-safety guarantee the whole multi-marker migration depends on: it
proves single-marker deployments keep running through unmodified code.

cuasp_siamfc: CV_TARGET_TRACKER = "siamfc" is the default here and routes to
IdsSiamFCCorrectionSourceAdapter; the ArUco tests run with TARGET_TRACKER
forced to "aruco" and must keep passing unchanged.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from suncubes import camera as cam


@pytest.fixture
def cuasp(cuasp_module):
    return cuasp_module


@pytest.fixture
def cuasp_aruco(cuasp_module, monkeypatch):
    monkeypatch.setattr(cuasp_module, "TARGET_TRACKER", "aruco")
    return cuasp_module


# ---------------------------------------------------------------------------- ArUco (unchanged behaviour)


def test_build_camera_config_has_expected_shape(cuasp_aruco):
    cuasp = cuasp_aruco
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
    assert "siamfc" not in cfg.ids_opencv


def test_two_marker_config_routes_to_multi_marker_adapter(cuasp_aruco):
    cfg = cuasp_aruco._build_camera_config()
    api = cuasp_aruco.CameraAPI(cfg)
    assert type(api._app._source).__name__ == "MultiArucoCorrectionSourceAdapter"


def test_single_marker_config_routes_to_original_adapter(cuasp_aruco):
    cfg = cuasp_aruco._build_camera_config()
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
    cuasp._validate_configuration()  # must not raise (siamfc default)


def test_validate_configuration_passes_in_aruco_mode(cuasp_aruco):
    cuasp_aruco._validate_configuration()


def test_validate_configuration_rejects_duplicate_aruco_ids(cuasp_aruco, monkeypatch):
    monkeypatch.setattr(cuasp_aruco, "ARUCO_IDS", (0, 0, 1))
    with pytest.raises(ValueError, match="ARUCO_IDS"):
        cuasp_aruco._validate_configuration()


def test_validate_configuration_folds_in_settings_validate_errors(cuasp, monkeypatch):
    from suncubes import settings
    monkeypatch.setattr(settings, "PORT", 99999)
    with pytest.raises(ValueError, match="PORT"):
        cuasp._validate_configuration()


# ---------------------------------------------------------------------------- SiamFC (cuasp_siamfc)


def test_siamfc_is_the_default_tracker_and_motors_are_disabled_by_default(cuasp):
    from suncubes import settings
    assert settings.CV_TARGET_TRACKER == "siamfc" and cuasp.TARGET_TRACKER == "siamfc"
    assert settings.CV_CUAS_ENABLE_MOTOR_COMMANDS == 0 and cuasp.ENABLE_MOTOR_COMMANDS is False


def test_siamfc_camera_config_shape(cuasp):
    from suncubes import settings
    cfg = cuasp._build_camera_config()
    assert cfg.mode == "ids_siamfc"
    assert cfg.gating.conf_min == cuasp.SIAMFC_CONF_MIN == settings.CV_SIAMFC_CONF_MIN
    assert cfg.gating.z_min_m == cuasp.Z_MIN_M and cfg.gating.max_age_s == cuasp.MAX_AGE_S
    src = cfg.ids_opencv
    assert "aruco" not in src and src["calibration"]["path"] == str(cuasp.CALIBRATION_YAML)
    s, d = src["siamfc"], src["detector"]
    # paths resolved against cuasp_siamfc/
    assert Path(s["weights_path"]) == (cuasp.REPO_ROOT_DIR / settings.CV_SIAMFC_WEIGHTS_PATH).resolve()
    assert Path(d["weights_path"]) == (cuasp.REPO_ROOT_DIR / settings.CV_DETECTOR_WEIGHTS_PATH).resolve()
    assert s["weights_sha256"] == settings.CV_SIAMFC_WEIGHTS_SHA256
    assert s["target_range_m"] == settings.CV_SIAMFC_TARGET_RANGE_M
    assert s["redetect_hz"] == settings.CV_SIAMFC_REDETECT_HZ and s["acquire_min_score"] == settings.CV_SIAMFC_ACQUIRE_MIN_SCORE
    assert s["device"] is None  # "" -> auto
    assert d["confirm_frames"] == settings.CV_DETECTOR_CONFIRM_FRAMES and d["fp16"] is True
    assert d["max_box_frac"] == settings.CV_DETECTOR_MAX_BOX_FRAC


def test_every_siamfc_setting_reaches_the_source_configuration(cuasp):
    """No CV_SIAMFC_* / CV_DETECTOR_* setting may be silently ignored."""
    from suncubes import settings
    from suncubes.siamfc_source import SiamFCSourceConfig
    names = {n for n in dir(settings) if n.startswith(("CV_SIAMFC_", "CV_DETECTOR_"))}
    mapped = set(cuasp._SIAMFC_SETTING_KEYS.values()) | set(cuasp._DETECTOR_SETTING_KEYS.values())
    mapped |= {"CV_SIAMFC_WEIGHTS_PATH", "CV_DETECTOR_WEIGHTS_PATH", "CV_SIAMFC_CONF_MIN"}  # handled separately
    assert names == mapped
    c = SiamFCSourceConfig.from_dicts(*cuasp._siamfc_source_dicts())
    c.validate()
    assert c.session.verify_miss_limit == settings.CV_SIAMFC_VERIFY_MISS_LIMIT
    assert c.confirmer.frames_required == settings.CV_DETECTOR_CONFIRM_FRAMES
    assert c.detector.imgsz == settings.CV_DETECTOR_IMGSZ


def test_siamfc_config_routes_to_the_siamfc_adapter(cuasp):
    api = cuasp.CameraAPI(cuasp._build_camera_config())
    assert type(api._app._source).__name__ == "IdsSiamFCCorrectionSourceAdapter"


def test_aruco_checks_are_skipped_in_siamfc_mode(cuasp, monkeypatch):
    monkeypatch.setattr(cuasp, "ARUCO_IDS", (0, 0, 1))  # would be an error in aruco mode
    cuasp._validate_configuration()


def test_validate_configuration_rejects_an_inconsistent_siamfc_setup(cuasp, monkeypatch):
    from suncubes import settings
    # Each value is valid on its own, but 2 Hz full-frame detection can never yield
    # detections 0.5 s apart: only the cross-field check of SiamFCSourceConfig sees it.
    monkeypatch.setattr(settings, "CV_SIAMFC_MISSING_HZ", 2.0)
    monkeypatch.setattr(settings, "CV_DETECTOR_CONFIRM_MAX_GAP_S", 0.5)
    with pytest.raises(ValueError, match="siamfc configuration"):
        cuasp._validate_configuration()


def test_validate_configuration_rejects_bad_siamfc_gating(cuasp, monkeypatch):
    monkeypatch.setattr(cuasp, "SIAMFC_CONF_MIN", 0.0)
    with pytest.raises(ValueError, match="SIAMFC_CONF_MIN"):
        cuasp._validate_configuration()


def test_unknown_tracker_is_rejected(cuasp, monkeypatch):
    monkeypatch.setattr(cuasp, "TARGET_TRACKER", "kcf")
    with pytest.raises(ValueError, match="TARGET_TRACKER"):
        cuasp._validate_configuration()


class _NoWindowCv2:
    """cv2 proxy for _draw_debug_frame: real drawing, imshow/waitKey captured."""

    def __init__(self):
        import cv2
        self._cv2 = cv2
        self.shown = None

    def __getattr__(self, name):
        return getattr(self._cv2, name)

    def imshow(self, name, img):
        self.shown = img

    def waitKey(self, delay):
        return -1


def test_debug_frame_draws_the_siamfc_box(cuasp):
    cv2p = _NoWindowCv2()
    correction = cam.CameraCorrection(True, "ok", 0.001, -0.002, 0.8, 0.01, None, 21.0, 1, 2)
    pbvs = cuasp.compute_pbvs_correction_from_camera(correction)
    snapshot = {
        "available": True,
        "frame_gray": np.zeros((240, 320), np.uint8),
        "markers": [], "expected_marker_ids": (), "target_point": None,
        "pbvs_target_pixel_px": np.array([160.0, 120.0]),
        "principal_point_px": np.array([160.0, 120.0]),
        "laser_axis_pixel_px": np.array([160.0, 130.0]), "laser_axis_range_m": 21.0, "laser_axis_range_source": "fixed",
        "tile_state": "tracking", "tracking_backend": "siamfc", "tracker_state": "tracking",
        "siamfc_bbox_xywh": (130.0, 100.0, 60.0, 40.0), "siamfc_score": 0.8,
        "detections": [{"bbox_xywh": (128.0, 98.0, 64.0, 44.0), "score": 0.9}],
    }
    stop = cuasp._draw_debug_frame(cv2p, correction, pbvs, snapshot, 5, 50.0, None)
    assert stop is False and cv2p.shown is not None
    # The SiamFC box is drawn in green (tracking). The status line is green too and the shown image is
    # downscaled, so compare the amount of green with and without the box.
    def green_pixels(img):
        a = np.asarray(img).astype(int)
        return int(((a[:, :, 0] < 60) & (a[:, :, 1] > 180) & (a[:, :, 2] < 60)).sum())

    with_box = green_pixels(cv2p.shown)
    cuasp._draw_debug_frame(cv2p, correction, pbvs, {**snapshot, "siamfc_bbox_xywh": None}, 5, 50.0, None)
    assert with_box > green_pixels(cv2p.shown) + 10
