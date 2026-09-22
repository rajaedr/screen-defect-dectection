from src.pipeline import run_inspection
from src.utils.config_loader import load_config, load_severity_rules


def test_config_loads_with_expected_top_level_sections():
    cfg = load_config()
    for section in ("project", "paths", "device_recognition", "screen_detection",
                     "damage_detection", "preprocessing", "training", "severity", "reporting"):
        assert section in cfg


def test_severity_rules_load():
    rules = load_severity_rules()
    assert "defect_base_weight" in rules
    assert "severity_thresholds" in rules


def test_full_pipeline_runs_end_to_end_on_clean_image(clean_screen_image):
    result = run_inspection(clean_screen_image, "clean.jpg")
    assert result.severity.inspection_result in ("PASS", "ATTENTION", "FAIL")
    assert result.annotated_image.shape == result.processed_image.shape
    assert "device" in result.report
    assert "inspection_result" in result.report


def test_full_pipeline_runs_end_to_end_on_scratched_image(scratched_screen_image):
    result = run_inspection(scratched_screen_image, "scratched.jpg")
    assert result.severity.inspection_result in ("PASS", "ATTENTION", "FAIL")
    assert isinstance(result.defects, list)
