import json

from src.damage_detection.defect_detector import Defect
from src.detection.screen_detector import ScreenResult
from src.device_recognition.device_classifier import DeviceResult
from src.reporting.report_generator import (
    build_report_dict,
    report_to_json_bytes,
    report_to_pdf_bytes,
)
from src.severity.severity import estimate_severity


def _sample_state():
    device = DeviceResult(device_type="smartphone", confidence=0.7, reasoning="test")
    screen = ScreenResult(detected=True, bbox=(0, 0, 100, 200), confidence=0.8)
    defects = [Defect(class_name="scratch", confidence=0.9, bbox=(10, 10, 20, 20), uncertain=False)]
    severity = estimate_severity(defects, screen.bbox)
    return device, screen, defects, severity


def test_build_report_dict_has_expected_keys():
    device, screen, defects, severity = _sample_state()
    report = build_report_dict(device, screen, defects, severity, "test.jpg")
    for key in ("report_meta", "device", "screen", "inspection_result",
                "severity", "defects_confirmed", "defects_uncertain"):
        assert key in report


def test_report_to_json_bytes_round_trips():
    device, screen, defects, severity = _sample_state()
    report = build_report_dict(device, screen, defects, severity, "test.jpg")
    data = report_to_json_bytes(report)
    parsed = json.loads(data)
    assert parsed["device"]["device_type"] == "smartphone"
    assert parsed["defects_confirmed"][0]["type"] == "scratch"


def test_report_to_pdf_bytes_produces_valid_pdf_header():
    device, screen, defects, severity = _sample_state()
    report = build_report_dict(device, screen, defects, severity, "test.jpg")
    pdf_bytes = report_to_pdf_bytes(report)
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 500
