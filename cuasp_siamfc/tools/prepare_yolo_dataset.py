"""Build the grayscale YOLO detection dataset for the drone detector (Phase 2).

Sources (see ../../datasets/MANIFEST.md):
- DUT-Anti-UAV detection: VOC XML, class "UAV", official train/val/test split.
- Anti-UAV300 RGB videos (optional, ``--anti-uav300``): frames sampled every
  ``--anti-uav-stride`` frames from each visible-light video. Frames where the
  target is absent are kept as negatives (empty label file).

Every image is converted to **grayscale** (the IDS camera is Mono8) and written as
a single-channel JPEG. Ultralytics reads it back as 3 identical channels, which
matches the runtime input. Labels use YOLO format: ``0 cx cy w h``, normalised.

Usage (from cuasp_siamfc/)::

    .venv/Scripts/python tools/prepare_yolo_dataset.py --out ../datasets/yolo_uav_gray [--anti-uav300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

DATASETS = Path(__file__).resolve().parents[2] / "datasets"
JPEG_QUALITY = 95


def _write(out_root: Path, split: str, stem: str, gray: np.ndarray, boxes_xyxy: list[tuple[float, float, float, float]]) -> None:
    h, w = gray.shape[:2]
    img_dir, lbl_dir = out_root / "images" / split, out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), gray, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    lines = []
    for x0, y0, x1, y1 in boxes_xyxy:
        x0, x1 = max(0.0, min(x0, x1)), min(float(w), max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(float(h), max(y0, y1))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            continue
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")


def convert_dut_detection(out_root: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "val", "test"):
        src = DATASETS / "DUT-Anti-UAV" / "detection" / split
        n = 0
        for xml_path in sorted((src / "xml").glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            img = cv2.imread(str(src / "img" / f"{xml_path.stem}.jpg"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"missing image for {xml_path}")
            boxes = []
            for obj in root.findall("object"):
                b = obj.find("bndbox")
                boxes.append(tuple(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")))
            _write(out_root, split, f"dut_{split}_{xml_path.stem}", img, boxes)
            n += 1
        counts[f"dut_{split}"] = n
    return counts


def convert_anti_uav300(out_root: Path, stride: int) -> dict[str, int]:
    """Anti-UAV300 layout: <split>/<video>/visible.mp4 + visible.json {"exist": [...], "gt_rect": [[x, y, w, h], ...]}."""
    base = DATASETS / "Anti-UAV300" / "Anti-UAV-RGBT"
    split_map = {"train": "train", "val": "val", "test": "test"}
    counts = {}
    for src_split, dst_split in split_map.items():
        n = 0
        for video_dir in sorted((base / src_split).glob("*")):
            ann_path, vid_path = video_dir / "visible.json", video_dir / "visible.mp4"
            if not (ann_path.exists() and vid_path.exists()):
                continue
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            exist, rects = ann.get("exist", []), ann.get("gt_rect", [])
            cap = cv2.VideoCapture(str(vid_path))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0 and idx < len(rects):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    boxes = []
                    rect = rects[idx]
                    if (idx < len(exist) and exist[idx]) and rect and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                        x, y, w, h = (float(v) for v in rect)
                        boxes.append((x, y, x + w, y + h))
                    _write(out_root, dst_split, f"auav_{src_split}_{video_dir.name}_{idx:05d}", gray, boxes)
                    n += 1
                idx += 1
            cap.release()
        counts[f"anti_uav300_{src_split}"] = n
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anti-uav300", action="store_true")
    ap.add_argument("--anti-uav-stride", type=int, default=10)
    args = ap.parse_args()
    out = args.out.resolve()
    counts = convert_dut_detection(out)
    if args.anti_uav300:
        counts.update(convert_anti_uav300(out, max(1, args.anti_uav_stride)))
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: UAV\n",
        encoding="utf-8",
    )
    (out / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
