"""
Device recognition: smartphone vs laptop, and optional model recognition.

Two modes (config.yaml -> device_recognition.mode):

  "heuristic" (default, works with zero training data)
      Uses simple, explainable visual cues from the whole-image aspect ratio
      and shape: laptops photographed normally are wide (hinge + keyboard +
      screen visible, aspect ratio well above 1.0), while smartphones
      photographed face-on are tall/narrow (aspect ratio well below 1.0).
      This is a *rough* baseline, not a trained classifier -- it will be
      wrong for unusual crops/angles. That's why it also returns a
      confidence and reasoning string, and the low-confidence branch reports
      "Unknown" rather than guessing.

  "yolo_cls" (optional, requires training/train_device_classifier.py first)
      Loads a fine-tuned Ultralytics YOLO classification model from
      models/device_classifier.pt.

We deliberately do NOT attempt exact phone/laptop *model* identification
(e.g. "iPhone 14 Pro") in the MVP -- see README/MODEL_CARD for why: there is
no reliable, legally-usable public dataset for this without brand
partnership, and a wrong guess is worse than "Unknown" for a QC tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceResult:
    device_type: str          # "smartphone" | "laptop" | "unknown"
    confidence: float
    model_name: str = "Unknown"   # specific model, almost always "Unknown" in MVP
    reasoning: str = ""


def _heuristic_predict(image_rgb: np.ndarray) -> DeviceResult:
    h, w = image_rgb.shape[:2]
    aspect = w / float(h)

    # Also look at how much of the frame is filled by a single large bright
    # rectangular region (a strong cue for a face-on screen photo) using the
    # same contour approach as the screen detector, cheaply, just for a
    # secondary signal.
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_rect_ratio = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        largest_rect_ratio = (cw * ch) / float(w * h)

    if aspect >= 1.35:
        # wide image: consistent with a laptop (screen + keyboard/base visible)
        confidence = min(0.5 + (aspect - 1.35) * 0.4, 0.85)
        return DeviceResult(
            device_type="laptop",
            confidence=round(confidence, 2),
            reasoning=(
                f"Wide image aspect ratio ({aspect:.2f}) is typical of a laptop "
                "photographed with screen and base visible."
            ),
        )
    elif aspect <= 0.75:
        confidence = min(0.5 + (0.75 - aspect) * 0.5, 0.85)
        return DeviceResult(
            device_type="smartphone",
            confidence=round(confidence, 2),
            reasoning=(
                f"Tall/narrow image aspect ratio ({aspect:.2f}) is typical of a "
                "smartphone photographed face-on."
            ),
        )
    else:
        # ambiguous aspect ratio (e.g. a laptop screen cropped tight, or a
        # phone photographed landscape) -- do not guess with false confidence
        return DeviceResult(
            device_type="unknown",
            confidence=0.3,
            reasoning=(
                f"Aspect ratio ({aspect:.2f}) is ambiguous between a "
                "close-cropped laptop screen and a landscape-oriented "
                "smartphone. Falling back to 'Unknown' rather than guessing."
            ),
        )


_yolo_cls_model = None


def _load_yolo_cls_model():
    global _yolo_cls_model
    if _yolo_cls_model is not None:
        return _yolo_cls_model
    cfg = load_config()["device_recognition"]
    model_path = resolve_path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Device classifier model not found at {model_path}. "
            "Train one with training/train_device_classifier.py, or set "
            "device_recognition.mode back to 'heuristic' in config.yaml."
        )
    from ultralytics import YOLO  # imported lazily: heavy optional dependency

    _yolo_cls_model = YOLO(str(model_path))
    return _yolo_cls_model


def _yolo_predict(image_rgb: np.ndarray) -> DeviceResult:
    cfg = load_config()["device_recognition"]
    model = _load_yolo_cls_model()
    results = model.predict(image_rgb, verbose=False)
    r = results[0]
    top1_idx = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    class_name = model.names.get(top1_idx, "unknown")

    if conf < cfg["confidence_threshold"]:
        return DeviceResult(
            device_type="unknown",
            confidence=round(conf, 2),
            reasoning=f"Classifier confidence {conf:.2f} below threshold "
                      f"{cfg['confidence_threshold']}; reporting Unknown.",
        )
    return DeviceResult(
        device_type=class_name,
        confidence=round(conf, 2),
        reasoning=f"Predicted by trained classifier ({model_path_name(cfg)}).",
    )


def model_path_name(cfg) -> str:
    return Path(cfg["model_path"]).name


def _recognize_device_base(image_rgb: np.ndarray) -> DeviceResult:
    """Main entry point. Picks heuristic or trained model per config.yaml."""
    cfg = load_config()["device_recognition"]
    mode = cfg.get("mode", "heuristic")

    if mode == "yolo_cls":
        try:
            return _yolo_predict(image_rgb)
        except FileNotFoundError as e:
            logger.warning("%s Falling back to heuristic mode.", e)
            return _heuristic_predict(image_rgb)
    return _heuristic_predict(image_rgb)


def recognize_device(image_rgb: np.ndarray) -> DeviceResult:
    """Device type from the classifier, plus the model name read from on-screen text."""
    result = _recognize_device_base(image_rgb)
    try:
        from src.device_recognition.model_reader import read_model_name

        name = read_model_name(image_rgb)
    except Exception:
        name = None
    if name:
        result.model_name = name
    return result
