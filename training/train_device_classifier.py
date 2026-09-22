#!/usr/bin/env python3
"""
OPTIONAL: train a smartphone-vs-laptop image classifier to replace the
zero-training heuristic in src/device_recognition/device_classifier.py.

The heuristic baseline (aspect-ratio based) works reasonably well for
typical face-on photos with no training at all, so this script is only
needed if you collect a labeled dataset of phone/laptop photos and want
higher accuracy on unusual crops/angles.

Expected data layout (standard YOLO-classification / ImageFolder style):

    datasets/raw/device_classification/
        train/
            smartphone/*.jpg
            laptop/*.jpg
        val/
            smartphone/*.jpg
            laptop/*.jpg

Usage:
    python training/train_device_classifier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("train_device_classifier")

DATA_DIR = PROJECT_ROOT / "datasets" / "raw" / "device_classification"


def main() -> None:
    cfg = load_config()

    if not DATA_DIR.exists():
        logger.error(
            "%s not found. Create it with train/<class>/*.jpg and "
            "val/<class>/*.jpg subfolders (classes: smartphone, laptop) "
            "before running this script. Until then, "
            "device_recognition.mode stays 'heuristic' in config.yaml.",
            DATA_DIR,
        )
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO("yolov8n-cls.pt")  # small pretrained classification backbone

    results_dir = resolve_path(cfg["paths"]["results_dir"])
    model.train(
        data=str(DATA_DIR),
        epochs=30,
        imgsz=224,
        project=str(results_dir),
        name="device_cls_run",
        exist_ok=True,
    )

    best_ckpt = results_dir / "device_cls_run" / "weights" / "best.pt"
    model_dest = resolve_path(cfg["device_recognition"]["model_path"])
    model_dest.parent.mkdir(parents=True, exist_ok=True)
    if best_ckpt.exists():
        import shutil
        shutil.copy2(best_ckpt, model_dest)
        logger.info("Best device classifier copied to %s", model_dest)
        logger.info("Set device_recognition.mode: 'yolo_cls' in config/config.yaml to use it.")
    else:
        logger.warning("Training finished but best.pt was not found at %s", best_ckpt)


if __name__ == "__main__":
    main()
