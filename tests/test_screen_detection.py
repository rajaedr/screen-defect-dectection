from src.detection.screen_detector import detect_screen


def test_detects_screen_in_synthetic_image(clean_screen_image):
    result = detect_screen(clean_screen_image)
    assert result.detected is True
    assert result.bbox is not None
    x1, y1, x2, y2 = result.bbox
    assert x2 > x1 and y2 > y1
    assert 0.0 <= result.confidence <= 1.0


def test_no_screen_in_blank_uniform_image():
    import numpy as np
    uniform = np.full((400, 400, 3), 128, dtype=np.uint8)  # no edges at all
    result = detect_screen(uniform)
    assert result.detected is False
    assert result.bbox is None
