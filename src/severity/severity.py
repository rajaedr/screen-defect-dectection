"""
Severity estimation.

Implements the engineering-heuristic rules defined in
src/severity/severity_rules.yaml (kept separate/editable from the general
config, as required). These rules are NOT a scientifically validated damage
grading standard -- see the header comment in that file and MODEL_CARD.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.damage_detection.defect_detector import Defect, BBox
from src.utils.config_loader import load_severity_rules


@dataclass
class SeverityResult:
    severity: Optional[str]   # "Low" | "Medium" | "High" | None (no confirmed defects)
    score: float
    inspection_result: str    # "PASS" | "ATTENTION" | "FAIL"
    explanation: str


def _bbox_area(b: BBox) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def estimate_severity(defects: List[Defect], screen_bbox: Optional[BBox] = None) -> SeverityResult:
    rules = load_severity_rules()

    confirmed = [d for d in defects if not d.uncertain and d.class_name != "no_defect"]
    uncertain_only = [d for d in defects if d.uncertain]

    if not confirmed:
        if uncertain_only:
            return SeverityResult(
                severity=None,
                score=0.0,
                inspection_result="ATTENTION",
                explanation=(
                    f"{len(uncertain_only)} possible defect(s) detected below the "
                    "confidence threshold required for a confirmed finding. "
                    "Manual inspection recommended."
                ),
            )
        return SeverityResult(
            severity=None,
            score=0.0,
            inspection_result="PASS",
            explanation="No defects detected.",
        )

    screen_area = _bbox_area(screen_bbox) if screen_bbox else None
    large_thresh = rules["large_area_ratio_threshold"]
    large_bonus = rules["large_area_weight_bonus"]

    score = 0.0
    for d in confirmed:
        base = rules["defect_base_weight"].get(d.class_name, 1)
        weight = base
        if screen_area and screen_area > 0:
            area_ratio = _bbox_area(d.bbox) / screen_area
            if area_ratio >= large_thresh:
                weight += large_bonus
        score += weight

    if len(confirmed) > rules["multi_defect_count_threshold"]:
        score += rules["multi_defect_weight_bonus"]

    thresholds = rules["severity_thresholds"]
    if score <= thresholds["low_max"]:
        severity = "Low"
    elif score <= thresholds["medium_max"]:
        severity = "Medium"
    else:
        severity = "High"

    result_rules = rules["result_rules"]
    if severity in result_rules["fail_severity_levels"]:
        inspection_result = "FAIL"
    elif severity in result_rules["attention_severity_levels"]:
        inspection_result = "ATTENTION"
    else:
        inspection_result = "ATTENTION"

    explanation = (
        f"{len(confirmed)} confirmed defect(s) "
        f"({', '.join(sorted(set(d.class_name for d in confirmed)))}) "
        f"produced a severity score of {score:.1f} -> {severity}. "
        "Severity is an engineering heuristic, not a certified grading standard."
    )
    if uncertain_only:
        explanation += f" Additionally, {len(uncertain_only)} low-confidence possible defect(s) were noted."

    return SeverityResult(
        severity=severity,
        score=score,
        inspection_result=inspection_result,
        explanation=explanation,
    )
