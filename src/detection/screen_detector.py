"""
Screen localization.

"contour" mode (default, zero-training baseline):
    The screen is usually the single largest, roughly-rectangular,
    high-contrast region in a face-on device photo. We find edges, look for
    the largest contour whose bounding box fills a plausible fraction of the
    frame, and approximate it as a quadrilateral. This is intentionally
    simple and will fail on cluttered backgrounds or extreme angles -- in
    which case we report "not detected" rather than a wrong box, and the
    downstream damage-detection step falls back to analyzing the full frame
    with a lower confidence.

"yolo" mode (optional): a trained single-class "screen" detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScreenResult:
    detected: bool
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    confidence: float = 0.0
    method: str = "contour"


def _contour_detect(image_rgb: np.ndarray) -> ScreenResult:
    cfg = load_config()["screen_detection"]
    h, w = image_rgb.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ScreenResult(detected=False, method="contour")

    best = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        ratio = area / img_area
        if ratio < cfg["min_area_ratio"] or ratio > cfg["max_area_ratio"]:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        rect_area = cw * ch
        # how "rectangular" the contour is: filled-area / bounding-box-area
        rectangularity = area / rect_area if rect_area > 0 else 0
        score = ratio * rectangularity
        if score > best_score:
            best_score = score
            best = (x, y, x + cw, y + ch)

    if best is None:
        return ScreenResult(detected=False, method="contour")

    # crude confidence: scaled rectangularity/area score, capped
    confidence = float(min(0.4 + best_score * 1.5, 0.9))
    if confidence < cfg["confidence_threshold"]:
        return ScreenResult(detected=False, confidence=confidence, method="contour")

    return ScreenResult(detected=True, bbox=best, confidence=round(confidence, 2), method="contour")


_yolo_screen_model = None


def _load_yolo_model():
    global _yolo_screen_model
    if _yolo_screen_model is not None:
        return _yolo_screen_model
    cfg = load_config()["screen_detection"]
    model_path = resolve_path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Screen detector model not found at {model_path}. Train one, or "
            "set screen_detection.mode back to 'contour' in config.yaml."
        )
    from ultralytics import YOLO

    _yolo_screen_model = YOLO(str(model_path))
    return _yolo_screen_model


def _yolo_detect(image_rgb: np.ndarray) -> ScreenResult:
    cfg = load_config()["screen_detection"]
    model = _load_yolo_model()
    results = model.predict(
        image_rgb, conf=cfg["confidence_threshold"], verbose=False
    )
    r = results[0]
    if len(r.boxes) == 0:
        return ScreenResult(detected=False, method="yolo")
    # take highest-confidence box
    best_idx = int(r.boxes.conf.argmax())
    box = r.boxes.xyxy[best_idx].tolist()
    conf = float(r.boxes.conf[best_idx])
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    return ScreenResult(detected=True, bbox=(x1, y1, x2, y2), confidence=round(conf, 2), method="yolo")


def detect_screen(image_rgb: np.ndarray) -> ScreenResult:
    cfg = load_config()["screen_detection"]
    mode = cfg.get("mode", "contour")

    if mode == "yolo":
        try:
            return _yolo_detect(image_rgb)
        except FileNotFoundError as e:
            logger.warning("%s Falling back to contour mode.", e)
            return _contour_detect(image_rgb)
    return _contour_detect(image_rgb)
