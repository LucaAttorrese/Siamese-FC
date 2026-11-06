"""Train the grayscale drone detector (Phase 2) with Ultralytics YOLO.

Usage (from cuasp_siamfc/)::

    # new run
    .venv/Scripts/python tools/train_detector.py --data ../datasets/yolo_uav_gray/data.yaml --name yolo11s_uav_gray_1280_e40 --epochs 40 --workers 2
    # ask the running training to stop at the end of the current epoch (from another terminal)
    .venv/Scripts/python tools/train_detector.py --name yolo11s_uav_gray_1280_e40 --request-stop
    # resume it later from the last completed epoch
    .venv/Scripts/python tools/train_detector.py --data ../datasets/yolo_uav_gray/data.yaml --name yolo11s_uav_gray_1280_e40 --resume --workers 2

Choices (documented here so the run is reproducible):
- model: YOLO11s pretrained on COCO (``yolo11s.pt``), fine-tuned on a single class "UAV";
- imgsz 1280: DUT boxes are small (median side ~2.3 % of the image side). At 640 the
  median drone would be ~15 px;
- colour augmentation off (hsv_h = hsv_s = 0): images are grayscale. Brightness
  augmentation (hsv_v) is kept;
- fixed seed with deterministic=True.
Outputs go to ../runs/detector/<name>/ (gitignored via the weights patterns + runs/).

Stopping and resuming:
- a file named ``STOP`` in the run folder (``--request-stop`` creates it) makes the
  training stop cleanly at the end of the current epoch, after that epoch's validation
  and after ``weights/last.pt`` has been written. The test evaluation is skipped and the
  script exits with code 3. ``STOP`` is deleted when it is acted on;
- the request is ignored during the last epoch (the run then simply finishes), because
  Ultralytics cannot resume a checkpoint of the final epoch;
- the stop deliberately bypasses Ultralytics' normal end of training: that path calls
  ``strip_optimizer`` on ``last.pt``, which makes it impossible to resume;
- ``--resume`` continues from ``weights/last.pt`` with the checkpoint's own settings
  (epochs, imgsz, batch, seed...). Only ``--workers`` and ``--patience`` can be changed on resume;
- a hard interruption (Ctrl+C, crash, power loss) loses only the epoch in progress:
  ``--resume`` works the same way. On Windows, Ctrl+C also reaches the dataloader
  workers, so it is always a hard stop: use ``STOP`` for a clean one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

DEFAULT_PROJECT = Path(__file__).resolve().parents[2] / "runs" / "detector"
STOP_FILE_NAME = "STOP"
EXIT_STOPPED = 3


class StopRequested(Exception):
    """Raised from the epoch-end callback to leave Ultralytics' loop without final_eval()."""

    def __init__(self, epoch: int, epochs: int) -> None:
        super().__init__(f"stop requested after epoch {epoch}/{epochs}")
        self.epoch = epoch
        self.epochs = epochs


def should_stop(stop_file: Path, completed_epochs: int, total_epochs: int, already_stopping: bool) -> bool:
    """Return True when a requested stop must be honoured now.

    ``completed_epochs`` counts the epoch that has just ended (1-based). A stop is not
    honoured after the final epoch, nor when Ultralytics is already ending the run
    (early stopping), because then the run finishes normally and there is nothing to resume.
    """
    if not stop_file.exists():
        return False
    return completed_epochs < total_epochs and not already_stopping


def _add_callbacks(model, stop_file: Path) -> None:
    timing: dict[str, float] = {}

    def on_train_epoch_start(trainer) -> None:
        timing["t0"] = time.perf_counter()

    def on_train_epoch_end(trainer) -> None:
        timing["t1"] = time.perf_counter()

    def on_fit_epoch_end(trainer) -> None:
        if "t0" not in timing or "t1" not in timing:
            return  # final_eval() also fires this event; there is no epoch to report then
        done, total = trainer.epoch + 1, trainer.epochs
        t2 = time.perf_counter()
        print(
            "EPOCH_DONE "
            + json.dumps(
                {
                    "epoch": done,
                    "epochs": total,
                    "train_min": round((timing["t1"] - timing["t0"]) / 60, 2),
                    "val_min": round((t2 - timing["t1"]) / 60, 2),
                    "fitness": None if trainer.fitness is None else round(float(trainer.fitness), 5),
                }
            ),
            flush=True,
        )
        timing.clear()
        if should_stop(stop_file, done, total, bool(trainer.stop)):
            raise StopRequested(done, total)
        if stop_file.exists():
            print(f"STOP ignored: epoch {done}/{total} is the last one or the run is already ending", flush=True)

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _close_loaders(model) -> None:
    trainer = getattr(model, "trainer", None)
    for name in ("train_loader", "test_loader"):
        loader = getattr(trainer, name, None)
        if hasattr(loader, "close"):
            loader.close()  # do not leave persistent dataloader workers behind


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="data.yaml (required except with --request-stop)")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto (60%% of GPU memory)")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="yolo11s_uav_gray_1280")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help=argparse.SUPPRESS)
    ap.add_argument("--resume", action="store_true", help="continue <project>/<name>/weights/last.pt")
    ap.add_argument("--request-stop", action="store_true", help="ask the running <name> to stop after the current epoch")
    args = ap.parse_args()

    run_dir = args.project.resolve() / args.name
    stop_file = run_dir / STOP_FILE_NAME
    last_pt = run_dir / "weights" / "last.pt"

    if args.request_stop:
        if not run_dir.is_dir():
            print(f"no run folder {run_dir}", file=sys.stderr)
            return 2
        stop_file.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        print(f"stop requested: {stop_file}\nthe training stops at the end of the current epoch")
        return 0

    if args.data is None:
        ap.error("--data is required")
    if args.resume and not last_pt.exists():
        print(f"cannot resume: {last_pt} not found", file=sys.stderr)
        return 2
    if not args.resume and run_dir.exists():
        # Ultralytics would silently train into <name>2, ... and the STOP file would be looked for in the wrong place.
        print(f"{run_dir} already exists: use --resume to continue it, or choose another --name", file=sys.stderr)
        return 2
    if stop_file.exists():
        print(f"removing a stale {stop_file} left by an earlier run", flush=True)
        stop_file.unlink()

    from ultralytics import YOLO  # noqa: PLC0415

    data = str(args.data.resolve())
    try:
        if args.resume:
            model = YOLO(str(last_pt))
            _add_callbacks(model, stop_file)
            model.train(resume=True, data=data, workers=args.workers, patience=args.patience)
        else:
            model = YOLO(args.model)
            _add_callbacks(model, stop_file)
            model.train(
                data=data,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                patience=args.patience,
                workers=args.workers,
                project=str(args.project.resolve()),
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
    except StopRequested as stop:
        _close_loaders(model)
        if stop_file.exists():
            stop_file.unlink()
        print(f"TRAINING_STOPPED after epoch {stop.epoch}/{stop.epochs}; checkpoint: {last_pt}", flush=True)
        print(
            "resume with:\n  .venv\\Scripts\\python tools\\train_detector.py "
            f"--data {args.data} --name {args.name} --resume --workers {args.workers}",
            flush=True,
        )
        return EXIT_STOPPED

    imgsz = getattr(getattr(model, "trainer", None), "args", args).imgsz  # on resume, the checkpoint's value
    metrics = model.val(data=data, split="test", imgsz=imgsz, project=str(args.project.resolve()), name=f"{args.name}_test")
    print(f"test mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} P={metrics.box.mp:.4f} R={metrics.box.mr:.4f}")
    print("TRAINING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
