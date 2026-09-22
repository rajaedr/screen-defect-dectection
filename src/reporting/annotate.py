"""Draws screen + defect bounding boxes and labels onto a copy of the image."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.damage_detection.defect_detector import Defect

SCREEN_COLOR = (0, 200, 255)     # cyan-ish, BGR for cv2 drawing
CONFIRMED_COLOR = (0, 0, 255)    # red
UNCERTAIN_COLOR = (0, 165, 255)  # orange


def _put_label(img_bgr: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(img_bgr.shape[1], img_bgr.shape[0]) / 1400)
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(0, y - th - 6)
    cv2.rectangle(img_bgr, (x, y_top), (x + tw + 6, y_top + th + 6), color, -1)
    cv2.putText(img_bgr, text, (x + 3, y_top + th + 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def annotate_image(
    image_rgb: np.ndarray,
    screen_bbox: Optional[Tuple[int, int, int, int]],
    defects: List[Defect],
) -> np.ndarray:
    """Return a new RGB uint8 image with screen + defect boxes drawn."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    thickness = max(1, min(img_bgr.shape[1], img_bgr.shape[0]) // 400)

    if screen_bbox is not None:
        x1, y1, x2, y2 = screen_bbox
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), SCREEN_COLOR, thickness)
        _put_label(img_bgr, "SCREEN", x1, y1, SCREEN_COLOR)

    for d in defects:
        x1, y1, x2, y2 = d.bbox
        color = UNCERTAIN_COLOR if d.uncertain else CONFIRMED_COLOR
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
        prefix = "POSSIBLE " if d.uncertain else ""
        label = f"{prefix}{d.class_name.upper()} {int(d.confidence * 100)}%"
        _put_label(img_bgr, label, x1, y1, color)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
