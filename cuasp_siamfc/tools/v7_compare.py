"""V7: SiamFC pointing angles against the ArUco reference, on the same recorded IDS frames.

The drone carries the two ArUco markers of C-UASP (ids ``CV_ARUCO_TARGET_IDS``, 5 cm, on a bar
either side of the drone). On every frame (or every ``--step``-th) this tool:

1. computes the **ArUco reference** with the unchanged C-UASP code of ``suncubes.camera``: CLAHE
   preprocessing and detector tuning from ``settings.py``, detection on the whole frame,
   per-marker pose, confidence and reprojection checks (``MultiArucoCorrectionSourceAdapter.
   _make_observation``), then the plain **midpoint of the two markers** (user decision of
   2026-11-19: it marks the drone centre; frames with one marker only are not used), moved to the
   laser frame and converted to (d_az, d_po) with the same functions as the runtime. The C-UASP
   aim point (midpoint + ``CV_CUAS_TARGET_MIDPOINT_OFFSET_CM``) is reported for information;
2. reads the SiamFC angles of the same frame from a replay CSV of
   ``tools/replay_siamfc_source.py`` (the ``ids_siamfc`` source with the runtime settings);
3. compares them where both are valid: bias and spread per axis in mrad, the share inside the
   acceptance limits, and a split of the difference into
   - tracker part: SiamFC angle minus the angle of the ArUco midpoint pixel through the SiamFC
     fixed-range geometry (``pixel_to_laser_angles`` at ``CV_SIAMFC_TARGET_RANGE_M``),
   - geometry part: that angle minus the ArUco angle (fixed range against the range measured
     by the markers, through the camera-laser lever arm);
   plus a sign and scale check: least-squares slope and correlation of SiamFC against ArUco
   per axis (meaningful only when the drone moves).

Acceptance limits (user decision of 2026-11-19, to be revised with the laser spot size): mean
bias per axis <= 0.5 mrad and 95 % of the frames within 1.5 mrad (radial difference).

Usage (from cuasp_siamfc/):
  .venv/Scripts/python tools/replay_siamfc_source.py ../recordings/<rec>/frames \
      --calibration calibration/calibration_legacy_20260921_x_test.yaml --fps 62.6 --out ../runs/v7/<rec>_siamfc.csv
  .venv/Scripts/python tools/v7_compare.py ../recordings/<rec> --siamfc-csv ../runs/v7/<rec>_siamfc.csv \
      --out ../runs/v7/<rec>_v7.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))

from suncubes import camera as cam  # noqa: E402
from suncubes import settings  # noqa: E402
from suncubes.siamfc_source import pixel_to_laser_angles  # noqa: E402

DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "calibration" / "calibration_legacy_20260921_x_test.yaml"


class ArucoReference:
    """Per-frame ArUco measurement with the C-UASP functions (no KLT, no tiles: the whole frame is
    searched on every frame, which is the most complete measurement those functions can give)."""

    def __init__(self, calibration: Path):
        self.cv2 = cam._import_cv2_or_raise()
        calib = cam._load_vs_calibration_for_camera(Path(calibration))
        self.k_calib = np.asarray(calib["camera_matrix"], np.float64)
        self.dist = np.asarray(calib["dist_coeffs"], np.float64)
        self.r_lc = np.asarray(calib["R_lc"], np.float64).reshape(3, 3)
        self.t_lc = np.asarray(calib["t_lc_m"], np.float64).reshape(3, 1)
        self.calib_wh = (int(calib["image_width"]), int(calib["image_height"]))
        self.ids = [int(v) for v in settings.CV_ARUCO_TARGET_IDS]
        self.dict_name = str(settings.CV_ARUCO_DICT_NAME)
        self.marker_size_m = float(settings.CV_ARUCO_MARKER_SIZE_M)
        gating = cam.GatingConfig(conf_min=float(getattr(settings, "CV_ARUCO_CONFIDENCE_MIN", 0.005)),
                                  z_min_m=float(getattr(settings, "CV_ARUCO_Z_MIN_M", 0.25)))
        # The adapter is only used for _make_observation (pose, reprojection and confidence checks);
        # it is never started, so it never opens the camera.
        self.adapter = cam.MultiArucoCorrectionSourceAdapter(
            {"aruco": {"dict_name": self.dict_name, "marker_size_m": self.marker_size_m, "target_marker_ids": self.ids},
             "calibration": {"path": str(calibration)}}, gating)
        self.adapter._cv2 = self.cv2
        self.adapter._dist = self.dist
        self.track_cfg = cam.load_multi_marker_tracking_config()
        self.pre = cam.MultiMarkerTiledPreprocessor(self.cv2, cam.load_fast_tracking_config(), self.track_cfg)
        self.detector, self.aruco_dict, self.aruco_params = cam._init_aruco_detector(self.cv2, self.dict_name)
        self.specs = {i: cam.ArucoMarkerSpec(marker_id=i, dict_name=self.dict_name, marker_size_m=self.marker_size_m) for i in self.ids}

    def k_runtime(self, w: int, h: int) -> np.ndarray:
        return cam._scale_intrinsics_to_runtime(self.k_calib, self.calib_wh[0], self.calib_wh[1], w, h)[0]

    def measure(self, gray: np.ndarray, ts_ns: int) -> dict:
        h, w = gray.shape[:2]
        k = self.k_runtime(w, h)
        corners, ids, _ = cam._detect_markers(self.cv2, self.pre._preprocess(gray), self.detector, self.aruco_dict, self.aruco_params)
        found: dict[int, np.ndarray] = {}
        if ids is not None:
            for i, item in enumerate(corners):
                mid = int(ids[i].item())
                if mid not in self.specs:
                    continue
                quad = np.asarray(item, np.float32).reshape(4, 2)
                if mid not in found or abs(self.cv2.contourArea(quad)) > abs(self.cv2.contourArea(found[mid])):
                    found[mid] = quad
        obs = []
        for mid, quad in found.items():
            o = self.adapter._make_observation(self.specs[mid], quad, k, w, h, ts_ns)
            if o is not None:
                obs.append(o)
        out = {"markers": len(found), "observations": len(obs), "complete": len(obs) == len(self.ids)}
        side = [math.sqrt(abs(self.cv2.contourArea(q))) for q in found.values()]
        out["marker_side_px"] = float(np.mean(side)) if side else float("nan")
        if not out["complete"]:
            return out
        p_mid = np.mean([o.position_camera_m for o in obs], axis=0)
        p_l = cam._point_c_to_laser(p_mid, self.r_lc, self.t_lc)
        ang = cam._angles_from_point_laser(p_l)
        px = cam._project_camera_point_to_pixel(self.cv2, p_mid, k, self.dist)
        if ang is None or px is None:
            out["complete"] = False
            return out
        out.update({"az": ang[0], "po": ang[1], "z_cam_m": float(p_mid[2]), "u": float(px[0]), "v": float(px[1]),
                    "marker_dist_px": float(np.linalg.norm(np.mean(found[self.ids[0]], 0) - np.mean(found[self.ids[1]], 0)))})
        # C-UASP aim point (midpoint + configured rigid offset), for information only
        tgt = cam.compute_multi_marker_target_point(obs, expected_marker_count=len(self.ids), cfg=self.track_cfg)
        if tgt is not None:
            a2 = cam._angles_from_point_laser(cam._point_c_to_laser(tgt.position_camera_m, self.r_lc, self.t_lc))
            if a2 is not None:
                out["az_cuasp"], out["po_cuasp"] = a2
        # the ArUco midpoint pixel through the SiamFC fixed-range geometry
        g = pixel_to_laser_angles(self.cv2, (out["u"], out["v"]), k, self.dist, self.r_lc, self.t_lc,
                                  float(settings.CV_SIAMFC_TARGET_RANGE_M))
        if g is not None:
            out["az_fixz"], out["po_fixz"] = g[0], g[1]
        return out


def read_siamfc_csv(path: Path) -> dict[int, dict]:
    rows = {}
    for r in csv.DictReader(open(path)):
        rel = r["reliable"] in ("1", "True", "true")
        if not rel or not r["d_az_rad"]:
            rows[int(r["frame"])] = None
            continue
        bx, by, bw, bh = (float(r[k]) for k in ("bx", "by", "bw", "bh"))
        ox, oy = (float(v) for v in settings.CV_SIAMFC_BBOX_OFFSET_PX)
        rows[int(r["frame"])] = {"az": float(r["d_az_rad"]), "po": float(r["d_po_rad"]),
                                 "u": bx + bw / 2 + ox, "v": by + bh / 2 + oy}
    return rows


def fit(x, y):
    if len(x) < 10 or np.ptp(x) <= 0:
        return float("nan"), float("nan")
    slope = float(np.polyfit(x, y, 1)[0])
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", type=Path, help="recordings/<name> (with frames/ and meta.json)")
    ap.add_argument("--siamfc-csv", type=Path, required=True, help="replay CSV of tools/replay_siamfc_source.py for the same frames")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--step", type=int, default=1, help="use every N-th frame")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--bias-max-mrad", type=float, default=0.5)
    ap.add_argument("--p95-max-mrad", type=float, default=1.5)
    ap.add_argument("--out", type=Path, help="per-frame CSV")
    ap.add_argument("--json", type=Path, help="summary JSON")
    args = ap.parse_args()

    frames = sorted((args.recording / "frames").glob("*.jpg")) or sorted((args.recording / "frames").glob("*.png"))
    if args.max_frames:
        frames = frames[:args.max_frames]
    meta = json.loads((args.recording / "meta.json").read_text()) if (args.recording / "meta.json").exists() else {}
    fc = read_siamfc_csv(args.siamfc_csv)
    ref = ArucoReference(args.calibration)
    cv2 = ref.cv2

    rows = []
    t0 = time.perf_counter()
    for idx in range(0, len(frames), max(1, args.step)):
        gray = cv2.imread(str(frames[idx]), cv2.IMREAD_GRAYSCALE)
        a = ref.measure(gray, idx)
        s = fc.get(idx)
        row = {"frame": idx, "aruco_markers": a["markers"], "aruco_complete": int(a["complete"]),
               "marker_side_px": a["marker_side_px"], "siamfc_reliable": int(s is not None)}
        for k in ("az", "po", "z_cam_m", "u", "v", "az_cuasp", "po_cuasp", "az_fixz", "po_fixz", "marker_dist_px"):
            row[f"aruco_{k}"] = a.get(k, float("nan"))
        for k in ("az", "po", "u", "v"):
            row[f"siamfc_{k}"] = s[k] if s else float("nan")
        rows.append(row)
    elapsed = time.perf_counter() - t0

    def col(name, mask=None):
        v = np.array([r[name] for r in rows], float)
        return v if mask is None else v[mask]

    both = (col("aruco_complete") == 1) & (col("siamfc_reliable") == 1)
    n = len(rows)
    res: dict = {"recording": args.recording.name, "frames_used": n, "step": args.step,
                 "aruco_complete_frac": float(np.mean(col("aruco_complete") == 1)),
                 "siamfc_reliable_frac": float(np.mean(col("siamfc_reliable") == 1)),
                 "both": int(both.sum()), "distance_m_meta": meta.get("distance_m"),
                 "limits": {"bias_max_mrad": args.bias_max_mrad, "p95_max_mrad": args.p95_max_mrad}}
    print(f"{args.recording.name}: {n} frames (step {args.step}, {elapsed:.0f} s); ArUco both markers {res['aruco_complete_frac'] * 100:.1f} %, "
          f"SiamFC reliable {res['siamfc_reliable_frac'] * 100:.1f} %, compared {int(both.sum())}")
    if both.sum() < 10:
        print("  too few frames with both measurements")
    else:
        d_az = (col("siamfc_az", both) - col("aruco_az", both)) * 1e3
        d_po = (col("siamfc_po", both) - col("aruco_po", both)) * 1e3
        rad = np.hypot(d_az, d_po)
        t_az = (col("siamfc_az", both) - col("aruco_az_fixz", both)) * 1e3
        t_po = (col("siamfc_po", both) - col("aruco_po_fixz", both)) * 1e3
        g_az = (col("aruco_az_fixz", both) - col("aruco_az", both)) * 1e3
        g_po = (col("aruco_po_fixz", both) - col("aruco_po", both)) * 1e3
        du = col("siamfc_u", both) - col("aruco_u", both)
        dv = col("siamfc_v", both) - col("aruco_v", both)
        c_az = (col("aruco_az_cuasp", both) - col("aruco_az", both)) * 1e3
        c_po = (col("aruco_po_cuasp", both) - col("aruco_po", both)) * 1e3
        z = col("aruco_z_cam_m", both)
        sl_az, r_az = fit(col("aruco_az", both), col("siamfc_az", both))
        sl_po, r_po = fit(col("aruco_po", both), col("siamfc_po", both))
        within = float(np.mean(rad <= args.p95_max_mrad))
        p95 = float(np.percentile(rad, 95))
        bias_ok = abs(float(d_az.mean())) <= args.bias_max_mrad and abs(float(d_po.mean())) <= args.bias_max_mrad
        t_p95 = float(np.percentile(np.hypot(t_az, t_po), 95))
        tracker_ok = abs(float(t_az.mean())) <= args.bias_max_mrad and abs(float(t_po.mean())) <= args.bias_max_mrad and t_p95 <= args.p95_max_mrad
        res.update({
            "diff_mrad": {"az_mean": float(d_az.mean()), "po_mean": float(d_po.mean()), "az_median": float(np.median(d_az)),
                          "po_median": float(np.median(d_po)), "az_std": float(d_az.std()), "po_std": float(d_po.std()),
                          "radial_p50": float(np.median(rad)), "radial_p95": p95, "radial_max": float(rad.max()),
                          "within_p95_limit_frac": within, "within_2mrad_frac": float(np.mean(rad <= 2.0))},
            "tracker_part_mrad": {"az_mean": float(t_az.mean()), "po_mean": float(t_po.mean()), "az_std": float(t_az.std()), "po_std": float(t_po.std()),
                                  "radial_p95": t_p95},
            "pass_tracker_part": bool(tracker_ok),
            "geometry_part_mrad": {"az_mean": float(g_az.mean()), "po_mean": float(g_po.mean()), "az_std": float(g_az.std()), "po_std": float(g_po.std())},
            "pixel_diff": {"du_median": float(np.median(du)), "dv_median": float(np.median(dv)),
                           "radial_p95": float(np.percentile(np.hypot(du, dv), 95))},
            "aruco_range_m": {"median": float(np.median(z)), "p5": float(np.percentile(z, 5)), "p95": float(np.percentile(z, 95))},
            "marker_side_px_median": float(np.nanmedian(col("marker_side_px", both))),
            "cuasp_aim_minus_midpoint_mrad": {"az": float(np.nanmedian(c_az)), "po": float(np.nanmedian(c_po))},
            "sign_scale": {"az_slope": sl_az, "az_r": r_az, "po_slope": sl_po, "po_r": r_po,
                           "az_range_mrad": float(np.ptp(col("aruco_az", both)) * 1e3), "po_range_mrad": float(np.ptp(col("aruco_po", both)) * 1e3)},
            "pass": bool(bias_ok and p95 <= args.p95_max_mrad),
        })
        dm = res["diff_mrad"]
        print(f"  SiamFC - ArUco: bias az {dm['az_mean']:+.2f} po {dm['po_mean']:+.2f} mrad (median {dm['az_median']:+.2f} / {dm['po_median']:+.2f}), "
              f"std {dm['az_std']:.2f} / {dm['po_std']:.2f}; radial p50 {dm['radial_p50']:.2f}, p95 {p95:.2f}, max {dm['radial_max']:.2f} mrad; "
              f"within {args.p95_max_mrad} mrad {within * 100:.1f} %, within 2 mrad {dm['within_2mrad_frac'] * 100:.1f} %")
        tp, gp = res["tracker_part_mrad"], res["geometry_part_mrad"]
        print(f"  split: tracker {tp['az_mean']:+.2f} / {tp['po_mean']:+.2f} mrad (std {tp['az_std']:.2f} / {tp['po_std']:.2f}, radial p95 {tp['radial_p95']:.2f}); "
              f"geometry (fixed {settings.CV_SIAMFC_TARGET_RANGE_M:g} m vs ArUco range) {gp['az_mean']:+.2f} / {gp['po_mean']:+.2f} mrad")
        pdiff, rng = res["pixel_diff"], res["aruco_range_m"]
        print(f"  pixels: SiamFC aim - ArUco midpoint du {pdiff['du_median']:+.1f} dv {pdiff['dv_median']:+.1f} px (median), radial p95 {pdiff['radial_p95']:.1f} px")
        print(f"  ArUco range {rng['median']:.2f} m (p5-p95 {rng['p5']:.2f}-{rng['p95']:.2f}); meta.json distance {res['distance_m_meta']}; "
              f"marker side {res['marker_side_px_median']:.1f} px")
        ss = res["sign_scale"]
        print(f"  sign/scale: az slope {ss['az_slope']:+.3f} r {ss['az_r']:+.3f} (range {ss['az_range_mrad']:.1f} mrad); "
              f"po slope {ss['po_slope']:+.3f} r {ss['po_r']:+.3f} (range {ss['po_range_mrad']:.1f} mrad)")
        ca = res["cuasp_aim_minus_midpoint_mrad"]
        print(f"  for information: C-UASP aim point (midpoint + {tuple(settings.CV_CUAS_TARGET_MIDPOINT_OFFSET_CM)} cm) - midpoint: "
              f"az {ca['az']:+.2f} po {ca['po']:+.2f} mrad")
        print(f"  RESULT: {'PASS' if res['pass'] else 'FAIL'} (bias <= {args.bias_max_mrad} mrad per axis, radial p95 <= {args.p95_max_mrad} mrad); "
              f"tracker part alone: {'PASS' if res['pass_tracker_part'] else 'FAIL'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
