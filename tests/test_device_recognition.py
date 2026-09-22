from src.device_recognition.device_classifier import recognize_device


def test_wide_image_predicted_as_laptop(clean_screen_image):
    # fixture image is 900x600 (wide) -> aspect ratio 1.5
    result = recognize_device(clean_screen_image)
    assert result.device_type == "laptop"
    assert 0.0 <= result.confidence <= 1.0
    assert result.model_name == "Unknown"


def test_tall_image_predicted_as_smartphone(phone_like_image):
    result = recognize_device(phone_like_image)
    assert result.device_type == "smartphone"
    assert result.model_name == "Unknown"


def test_ambiguous_aspect_ratio_reports_unknown_not_a_guess():
    import numpy as np
    square_ish = np.full((500, 480, 3), 100, dtype=np.uint8)  # aspect ~0.96
    result = recognize_device(square_ish)
    assert result.device_type == "unknown"
    assert result.confidence < 0.5
