"""tools/v7_compare.py on synthetic frames: two ArUco markers rendered at a known 3D position with
the calibration in use, so that the ArUco reference, the fixed-range geometry and the verdict can
be checked against values computed by hand."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import v7_compare  # noqa: E402
from suncubes import camera as cam  # noqa: E402
from suncubes import settings  # noqa: E402

CALIB = ROOT / "calibration" / "calibration_legacy_20260921_x_test.yaml"
W, H = 2040, 1536
MID = np.array([0.05, -0.03, 21.0])  # camera frame, m: 5 cm right, 3 cm up, 21 m
HALF_BAR = 0.20  # markers 20 cm either side of the midpoint


def _calib():
    c = cam._load_vs_calibration_for_camera(CALIB)
    return (np.asarray(c["camera_matrix"], float), np.asarray(c["dist_coeffs"], float),
            np.asarray(c["R_lc"], float).reshape(3, 3), np.asarray(c["t_lc_m"], float).reshape(3, 1))


def render_frame() -> np.ndarray:
    """Markers facing the camera (marker X = camera x, marker Y = -camera y)."""
    k, dist, _, _ = _calib()
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, settings.CV_ARUCO_DICT_NAME))
    frame = np.full((H, W), 190, np.uint8)
    h = float(settings.CV_ARUCO_MARKER_SIZE_M) / 2
    for marker_id, dx in zip(settings.CV_ARUCO_TARGET_IDS, (-HALF_BAR, HALF_BAR)):
        p = MID + np.array([dx, 0.0, 0.0])
        obj = np.array([p + [-h, -h, 0], p + [h, -h, 0], p + [h, h, 0], p + [-h, h, 0]], np.float64)  # TL TR BR BL
        img_pts, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, dist)
        img_pts = img_pts.reshape(4, 2)
        # Supersampled rendering (x8) of a window around the marker, then INTER_AREA down to the
        # frame: anti-aliased edges, no systematic growth of the marker.
        ss, side = 8, 240
        tag = cv2.aruco.generateImageMarker(adict, int(marker_id), side)
        x0, y0 = int(img_pts[:, 0].min()) - 8, int(img_pts[:, 1].min()) - 8
        x1, y1 = int(img_pts[:, 0].max()) + 9, int(img_pts[:, 1].max()) + 9
        src = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)  # pixel-edge corners of the tag
        dst = ((img_pts - [x0, y0] + 0.5) * ss).astype(np.float32)  # pixel-centre -> pixel-edge, supersampled
        hmat = cv2.getPerspectiveTransform(src, dst)
        big = np.full(((y1 - y0) * ss, (x1 - x0) * ss), 190, np.uint8)
        cv2.warpPerspective(tag, hmat, (big.shape[1], big.shape[0]), dst=big, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
        frame[y0:y1, x0:x1] = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return frame


def expected_angles(range_m: float | None = None):
    """(az, po) of MID through the laser transform; with ``range_m`` the point on MID's camera ray
    at that depth (the SiamFC fixed-range geometry)."""
    _, _, r_lc, t_lc = _calib()
    p = MID if range_m is None else MID * (range_m / MID[2])
    p_l = (r_lc @ p.reshape(3, 1) + t_lc).reshape(3)
    return math.atan2(p_l[0], p_l[2]), math.atan2(-p_l[1], math.hypot(p_l[0], p_l[2]))


@pytest.fixture(scope="module")
def frame():
    return render_frame()


def test_aruco_reference_recovers_the_known_midpoint(frame):
    ref = v7_compare.ArucoReference(CALIB)
    m = ref.measure(frame, 0)
    assert m["complete"] and m["observations"] == 2
    az, po = expected_angles()
    assert abs(m["az"] - az) < 0.05e-3 and abs(m["po"] - po) < 0.05e-3
    assert m["z_cam_m"] == pytest.approx(21.0, rel=0.01)
    # expected marker side: 5 cm at 21 m with fx ~24 119 -> ~57 px
    assert m["marker_side_px"] == pytest.approx(0.05 * 24119.4 / 21.0, rel=0.03)


def test_fixed_range_geometry_part(frame, monkeypatch):
    ref = v7_compare.ArucoReference(CALIB)
    m = ref.measure(frame, 0)
    assert abs(m["az_fixz"] - m["az"]) < 0.05e-3 and abs(m["po_fixz"] - m["po"]) < 0.05e-3  # fixed range = true range
    monkeypatch.setattr(settings, "CV_SIAMFC_TARGET_RANGE_M", 23.0)
    m23 = ref.measure(frame, 0)
    az23, po23 = expected_angles(23.0)
    az, po = expected_angles()
    assert (m23["az_fixz"] - m23["az"]) == pytest.approx(az23 - az, abs=0.03e-3)
    assert (m23["po_fixz"] - m23["po"]) == pytest.approx(po23 - po, abs=0.03e-3)
    assert abs(az23 - az) > 0.5e-3  # the lever arm makes a 2 m range error visible (t_lc ~ (-0.16, -0.20) m)


def _recording(tmp_path, frame, n=12):
    rec = tmp_path / "rec"
    (rec / "frames").mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(rec / "frames" / f"{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (rec / "meta.json").write_text(json.dumps({"distance_m": 21.0}))
    return rec


def _siamfc_csv(path, n, d_az, d_po, u, v, w=300.0, h=90.0):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "state", "reason", "reliable", "confidence", "bx", "by", "bw", "bh", "d_az_rad", "d_po_rad"])
        for i in range(n):
            wr.writerow([i, "tracking", "ok", 1, 0.9, u - w / 2, v - h / 2, w, h, d_az, d_po])


@pytest.mark.parametrize("bias_mrad, passes", [(0.3, True), (0.8, False)])
def test_end_to_end_verdict(tmp_path, frame, bias_mrad, passes):
    ref = v7_compare.ArucoReference(CALIB)
    m = ref.measure(frame, 0)
    rec = _recording(tmp_path, frame)
    csv_path = tmp_path / "siamfc.csv"
    _siamfc_csv(csv_path, 12, m["az"] + bias_mrad * 1e-3, m["po"], m["u"], m["v"])
    out_json = tmp_path / "v7.json"
    sys_argv = sys.argv
    try:
        sys.argv = ["v7_compare.py", str(rec), "--siamfc-csv", str(csv_path), "--json", str(out_json), "--out", str(tmp_path / "v7.csv")]
        assert v7_compare.main() == 0
    finally:
        sys.argv = sys_argv
    res = json.loads(out_json.read_text())
    assert res["both"] == 12
    assert res["diff_mrad"]["az_mean"] == pytest.approx(bias_mrad, abs=0.05)
    assert abs(res["diff_mrad"]["po_mean"]) < 0.05
    assert res["pass"] is passes
    assert abs(res["pixel_diff"]["du_median"]) < 0.5 and abs(res["geometry_part_mrad"]["az_mean"]) < 0.05


def test_siamfc_csv_aim_pixel_is_the_box_centre_plus_the_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CV_SIAMFC_BBOX_OFFSET_PX", (0.0, 12.0))
    p = tmp_path / "s.csv"
    _siamfc_csv(p, 2, 0.001, -0.002, 1000.0, 700.0)
    rows = v7_compare.read_siamfc_csv(p)
    assert rows[0]["u"] == pytest.approx(1000.0) and rows[0]["v"] == pytest.approx(712.0)
    assert rows[1]["az"] == pytest.approx(0.001) and rows[1]["po"] == pytest.approx(-0.002)
