"""Train the grayscale drone detector (Phase 2) with Ultralytics YOLO.

Usage (from cuasp_siamfc/)::

    .venv/Scripts/python tools/train_detector.py --data ../datasets/yolo_uav_gray/data.yaml

Choices (documented here so the run is reproducible):
- model: YOLO11s pretrained on COCO (``yolo11s.pt``), fine-tuned on a single class "UAV";
- imgsz 1280: DUT boxes are small (median side ~2.3 % of the image side). At 640 the
  median drone would be ~15 px;
- colour augmentation off (hsv_h = hsv_s = 0): images are grayscale. Brightness
  augmentation (hsv_v) is kept;
- fixed seed with deterministic=True.
Outputs go to ../runs/detector/<name>/ (gitignored via the weights patterns + runs/).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto (60%% of GPU memory)")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="yolo11s_uav_gray_1280")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from ultralytics import YOLO  # noqa: PLC0415

    project = (Path(__file__).resolve().parents[2] / "runs" / "detector").as_posix()
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=project,
        name=args.name,
        exist_ok=False,
        seed=0,
        deterministic=True,
        single_cls=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        plots=True,
    )
    metrics = model.val(data=str(args.data.resolve()), split="test", imgsz=args.imgsz, project=project, name=f"{args.name}_test")
    print(f"test mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} P={metrics.box.mp:.4f} R={metrics.box.mr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
