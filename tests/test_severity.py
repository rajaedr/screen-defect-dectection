from src.damage_detection.defect_detector import Defect
from src.severity.severity import estimate_severity


def test_no_defects_gives_pass():
    result = estimate_severity([], screen_bbox=(0, 0, 100, 100))
    assert result.inspection_result == "PASS"
    assert result.severity is None


def test_only_uncertain_defects_gives_attention_not_fail():
    defects = [Defect(class_name="scratch", confidence=0.4, bbox=(0, 0, 10, 10), uncertain=True)]
    result = estimate_severity(defects, screen_bbox=(0, 0, 100, 100))
    assert result.inspection_result == "ATTENTION"
    assert result.severity is None


def test_single_small_scratch_is_low_or_medium_not_high():
    defects = [Defect(class_name="scratch", confidence=0.8, bbox=(10, 10, 15, 15), uncertain=False)]
    result = estimate_severity(defects, screen_bbox=(0, 0, 1000, 1000))
    assert result.severity in ("Low", "Medium")
    assert result.inspection_result in ("ATTENTION",)


def test_broken_glass_escalates_to_fail():
    defects = [Defect(class_name="broken_glass", confidence=0.9, bbox=(0, 0, 900, 900), uncertain=False)]
    result = estimate_severity(defects, screen_bbox=(0, 0, 1000, 1000))
    assert result.severity == "High"
    assert result.inspection_result == "FAIL"


def test_multiple_defects_increase_score_over_single():
    single = [Defect(class_name="scratch", confidence=0.8, bbox=(0, 0, 10, 10), uncertain=False)]
    multiple = single + [
        Defect(class_name="scratch", confidence=0.8, bbox=(20, 20, 30, 30), uncertain=False),
        Defect(class_name="scratch", confidence=0.8, bbox=(40, 40, 50, 50), uncertain=False),
    ]
    r_single = estimate_severity(single, screen_bbox=(0, 0, 1000, 1000))
    r_multi = estimate_severity(multiple, screen_bbox=(0, 0, 1000, 1000))
    assert r_multi.score > r_single.score
