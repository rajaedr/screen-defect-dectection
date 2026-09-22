"""
Image preprocessing module.

Deliberately conservative: aggressive denoising/contrast normalization can
erase the very hairline scratches and faint cracks we are trying to detect,
so by default only resizing + basic quality checks are applied (see
config.yaml -> preprocessing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from src.utils.config_loader import load_config


@dataclass
class QualityReport:
    """Result of image-quality checks, used to warn the user rather than
    silently produce an unreliable inspection."""
    is_low_resolution: bool = False
    is_blurry: bool = False
    has_glare: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        # We still allow inspection to proceed (never hard-block the user),
        # but the report/UI should surface these warnings prominently.
        return True


def _sharpness_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — a standard, cheap blur estimator.
    Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_image_quality(image_rgb: np.ndarray) -> QualityReport:
    cfg = load_config()["preprocessing"]
    report = QualityReport()

    h, w = image_rgb.shape[:2]
    if min(h, w) < cfg["min_resolution_px"]:
        report.is_low_resolution = True
        report.warnings.append(
            f"Image resolution ({w}x{h}) is low; small defects may be missed."
        )

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    sharpness = _sharpness_score(gray)
    if sharpness < cfg["blur_variance_threshold"]:
        report.is_blurry = True
        report.warnings.append(
            "Image appears blurry; results may be unreliable. Try a steadier, "
            "closer, well-lit photo."
        )

    mean_brightness = float(gray.mean())
    bright_pixel_ratio = float((gray > 250).mean())
    if mean_brightness > cfg["glare_brightness_threshold"] or bright_pixel_ratio > 0.35:
        report.has_glare = True
        report.warnings.append(
            "Strong glare/reflection detected; this can be mistaken for screen "
            "damage. Try photographing at an angle or in more diffuse light."
        )

    return report


def resize_keep_aspect(image_rgb: np.ndarray, max_dim: int | None = None) -> np.ndarray:
    cfg = load_config()["preprocessing"]
    max_dim = max_dim or cfg["resize_width"]

    h, w = image_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_rgb

    scale = max_dim / float(longest)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_for_model(image_rgb: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] float32. Used only right before feeding a
    model — the *displayed*/annotated image should stay in the original
    0-255 uint8 range."""
    return image_rgb.astype(np.float32) / 255.0


def optional_denoise(image_rgb: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return image_rgb
    return cv2.fastNlMeansDenoisingColored(image_rgb, None, 5, 5, 7, 21)


def optional_auto_contrast(image_rgb: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return image_rgb
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_pipeline(image_rgb: np.ndarray) -> tuple[np.ndarray, QualityReport]:
    """Full preprocessing pipeline used by the app and inference scripts.

    Returns the (possibly resized) image plus a quality report. Quality
    checks run on the *original* image before any denoising so glare/blur
    detection isn't skewed by our own processing.
    """
    cfg = load_config()["preprocessing"]

    quality = check_image_quality(image_rgb)

    out = resize_keep_aspect(image_rgb, cfg["resize_width"])
    out = optional_denoise(out, cfg["denoise"])
    out = optional_auto_contrast(out, cfg["auto_contrast"])

    return out, quality
