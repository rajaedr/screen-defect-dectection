#!/usr/bin/env python3
"""
Train the defect-detection YOLOv8 model on datasets/processed/data.yaml.

All hyperparameters come from config/config.yaml -> training (never edit
this file to change epochs/batch size/etc — edit the config instead).

Usage:
    python training/train.py

Requires: datasets/processed/data.yaml to exist (run
datasets/prepare_dataset.py first) and `ultralytics` to be installed
(pip install -r requirements.txt).

Why YOLOv8 (nano, by default)?
- Single-stage detector => fast enough for CPU/MPS inference on a MacBook,
  no NVIDIA GPU required (see README "Local computer requirements").
- Native support for training/val/test splits, mAP/precision/recall
  metrics, and ONNX/CoreML export if you want to optimize further later.
- Mature, well-documented Ultralytics Python API that keeps this codebase
  small instead of hand-rolling training loops.
- 'yolov8n' (nano) is the smallest/fastest variant — appropriate for an MVP
  and modest dataset sizes; config.yaml -> training.base_model can be
  changed to yolov8s/m/l/x if you have more data and compute later.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("train")


def main() -> None:
    cfg = load_config()
    train_cfg = cfg["training"]
    data_yaml = resolve_path(cfg["paths"]["dataset_processed"]) / "data.yaml"

    if not data_yaml.exists():
        logger.error(
            "%s not found. Run `python datasets/prepare_dataset.py` first "
            "(after downloading/adding a dataset — see DATASET_SETUP.md).",
            data_yaml,
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "The 'ultralytics' package is not installed. Run: "
            "pip install -r requirements.txt"
        )
        sys.exit(1)

    device = train_cfg["device"]
    if device == "auto":
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = 0
        else:
            device = "cpu"
    logger.info("Training on device: %s", device)

    model = YOLO(train_cfg["base_model"])  # auto-downloads pretrained COCO weights on first use

    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=train_cfg["image_size"],
        lr0=train_cfg["learning_rate"],
        patience=train_cfg["patience"],
        device=device,
        seed=train_cfg["seed"],
        project=str(results_dir),
        name="train_run",
        exist_ok=True,
    )

    # Ultralytics saves the best checkpoint at
    # <results_dir>/train_run/weights/best.pt — copy it to the path the rest
    # of this project expects (models/defect_yolo.pt, from config.yaml)
    best_ckpt = results_dir / "train_run" / "weights" / "best.pt"
    model_dest = resolve_path(cfg["damage_detection"]["model_path"])
    model_dest.parent.mkdir(parents=True, exist_ok=True)
    if best_ckpt.exists():
        import shutil
        shutil.copy2(best_ckpt, model_dest)
        logger.info("Best model copied to %s", model_dest)
        logger.info(
            "Set damage_detection.mode: 'yolo' in config/config.yaml to use "
            "this trained model instead of the heuristic baseline."
        )
    else:
        logger.warning("Training finished but best.pt was not found at %s", best_ckpt)


if __name__ == "__main__":
    main()
