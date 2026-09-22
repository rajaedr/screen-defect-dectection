from src.damage_detection.defect_detector import detect_defects
from src.detection.screen_detector import detect_screen


def test_clean_screen_has_no_confirmed_defects(clean_screen_image):
    screen = detect_screen(clean_screen_image)
    defects = detect_defects(clean_screen_image, screen.bbox)
    confirmed = [d for d in defects if not d.uncertain]
    assert confirmed == []


def test_scratched_screen_detects_a_line_defect(scratched_screen_image):
    screen = detect_screen(scratched_screen_image)
    defects = detect_defects(scratched_screen_image, screen.bbox)
    assert len(defects) >= 1
    class_names = {d.class_name for d in defects}
    assert class_names & {"scratch", "crack"}


def test_all_returned_defects_meet_minimum_confidence(scratched_screen_image):
    from src.utils.config_loader import load_config
    cfg = load_config()["damage_detection"]
    screen = detect_screen(scratched_screen_image)
    defects = detect_defects(scratched_screen_image, screen.bbox)
    for d in defects:
        assert d.confidence >= cfg["confidence_threshold"]


def test_low_confidence_defects_are_flagged_uncertain(scratched_screen_image):
    screen = detect_screen(scratched_screen_image)
    defects = detect_defects(scratched_screen_image, screen.bbox)
    from src.utils.config_loader import load_config
    cfg = load_config()["damage_detection"]
    for d in defects:
        expected_uncertain = d.confidence < cfg["uncertain_confidence_threshold"]
        assert d.uncertain == expected_uncertain


def test_works_without_a_screen_bbox(scratched_screen_image):
    # should not crash when screen localization failed (bbox=None)
    defects = detect_defects(scratched_screen_image, None)
    assert isinstance(defects, list)
