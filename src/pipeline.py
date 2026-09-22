"""
Main inspection pipeline: the single place that wires together
preprocessing -> device recognition -> screen detection -> damage
detection -> severity -> report, so the Streamlit app, CLI scripts, and
tests all call the exact same logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.damage_detection.defect_detector import Defect, detect_defects
from src.detection.screen_detector import ScreenResult, detect_screen
from src.device_recognition.device_classifier import DeviceResult, recognize_device
from src.preprocessing.preprocess import QualityReport, preprocess_pipeline
from src.reporting.annotate import annotate_image
from src.reporting.report_generator import build_report_dict
from src.severity.severity import SeverityResult, estimate_severity
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InspectionResult:
    original_image: np.ndarray
    processed_image: np.ndarray
    annotated_image: np.ndarray
    quality: QualityReport
    device: DeviceResult
    screen: ScreenResult
    defects: List[Defect]
    severity: SeverityResult
    report: dict


def run_inspection(image_rgb: np.ndarray, image_filename: str = "uploaded_image") -> InspectionResult:
    """Run the full inspection pipeline on an already-loaded RGB image array.

    Never raises on a "normal" bad case (no screen found, no defects found,
    low quality image) — those are represented in the result. It can still
    raise if the input array itself is invalid; callers should load images
    via src.utils.image_io (which raises a friendly ImageLoadError earlier).
    """
    logger.info("Running inspection pipeline on '%s' (%dx%d)", image_filename,
                image_rgb.shape[1], image_rgb.shape[0])

    processed, quality = preprocess_pipeline(image_rgb)

    device = recognize_device(processed)
    logger.info("Device: %s (confidence=%.2f)", device.device_type, device.confidence)

    screen = detect_screen(processed)
    logger.info("Screen detected: %s (confidence=%.2f)", screen.detected, screen.confidence)

    defects = detect_defects(processed, screen.bbox if screen.detected else None)
    logger.info("Found %d candidate defect(s)", len(defects))

    severity = estimate_severity(defects, screen.bbox if screen.detected else None)
    logger.info("Result: %s / severity=%s", severity.inspection_result, severity.severity)

    annotated = annotate_image(processed, screen.bbox if screen.detected else None, defects)

    report = build_report_dict(device, screen, defects, severity, image_filename)
    report["quality_warnings"] = quality.warnings
    try:
        from src.device_recognition.model_reader import assess_settings, read_device_info

        _fields = read_device_info(image_rgb)
    except Exception:
        _fields = {}
    if _fields:
        report["device_settings"] = {"fields": _fields, **assess_settings(_fields)}

    return InspectionResult(
        original_image=image_rgb,
        processed_image=processed,
        annotated_image=annotated,
        quality=quality,
        device=device,
        screen=screen,
        defects=defects,
        severity=severity,
        report=report,
    )
