#!/usr/bin/env python3
"""
Run validation (on the 'val' split) for the trained defect model and print
precision/recall/mAP metrics. Use this during/after training to check for
overfitting before running the final held-out test (training/test.py).

Usage:
    python training/validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("validate")


def main() -> None:
    cfg = load_config()
    data_yaml = resolve_path(cfg["paths"]["dataset_processed"]) / "data.yaml"
    model_path = resolve_path(cfg["damage_detection"]["model_path"])

    if not model_path.exists():
        logger.error("No trained model found at %s. Run training/train.py first.", model_path)
        sys.exit(1)
    if not data_yaml.exists():
        logger.error("%s not found. Run datasets/prepare_dataset.py first.", data_yaml)
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=cfg["damage_detection"]["image_size"],
        conf=cfg["damage_detection"]["confidence_threshold"],
        iou=cfg["damage_detection"]["iou_threshold"],
        project=str(results_dir),
        name="val_run",
        exist_ok=True,
    )

    logger.info("Validation results (see also %s/val_run/):", results_dir)
    logger.info("  mAP@50:    %.4f", metrics.box.map50)
    logger.info("  mAP@50:95: %.4f", metrics.box.map)
    logger.info("  Precision: %.4f", metrics.box.mp)
    logger.info("  Recall:    %.4f", metrics.box.mr)


if __name__ == "__main__":
    main()
