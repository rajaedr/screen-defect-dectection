#!/usr/bin/env python3
"""
Comprehensive evaluation: computes precision/recall/F1/mAP@50/mAP@50:95,
per-class AP, and a confusion matrix on the test split, and saves everything
under results/ as both a JSON summary and Ultralytics' own plots.

This is a slightly more detailed/report-friendly wrapper around
training/test.py — run either one; test.py is the quick version.

Usage:
    python training/evaluation.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("evaluation")


def main() -> None:
    cfg = load_config()
    data_yaml = resolve_path(cfg["paths"]["dataset_processed"]) / "data.yaml"
    model_path = resolve_path(cfg["damage_detection"]["model_path"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])

    if not model_path.exists():
        logger.error("No trained model found at %s. Run training/train.py first.", model_path)
        sys.exit(1)
    if not data_yaml.exists():
        logger.error("%s not found. Run datasets/prepare_dataset.py first.", data_yaml)
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=cfg["damage_detection"]["image_size"],
        conf=cfg["damage_detection"]["confidence_threshold"],
        iou=cfg["damage_detection"]["iou_threshold"],
        project=str(results_dir),
        name="evaluation_run",
        exist_ok=True,
        plots=True,  # saves confusion_matrix.png, PR curves, etc.
    )

    class_names = model.names
    per_class = {}
    try:
        ap50_per_class = metrics.box.ap50  # array indexed like class_names
        for idx, name in class_names.items():
            if idx < len(ap50_per_class):
                per_class[name] = round(float(ap50_per_class[idx]), 4)
    except Exception:  # noqa: BLE001
        logger.warning("Per-class AP not available from this Ultralytics version.")

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "split": "test",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "per_class_ap50": per_class,
        "note": "All numbers above are produced by actual model evaluation on "
                "the held-out test split — none are estimated or invented.",
    }

    out_path = results_dir / "evaluation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation summary written to %s", out_path)
    logger.info("Confusion matrix / PR curves saved under %s/evaluation_run/", results_dir)
    for k, v in summary.items():
        if k not in ("per_class_ap50", "note"):
            logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
