import numpy as np

from src.preprocessing.preprocess import (
    check_image_quality,
    preprocess_pipeline,
    resize_keep_aspect,
)


def test_resize_keep_aspect_shrinks_large_image():
    img = np.zeros((2000, 3000, 3), dtype=np.uint8)
    out = resize_keep_aspect(img, max_dim=1000)
    assert max(out.shape[:2]) == 1000
    # aspect ratio preserved
    orig_ratio = 3000 / 2000
    new_ratio = out.shape[1] / out.shape[0]
    assert abs(orig_ratio - new_ratio) < 0.01


def test_resize_keep_aspect_leaves_small_image_untouched():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    out = resize_keep_aspect(img, max_dim=1000)
    assert out.shape == img.shape


def test_quality_check_flags_low_resolution():
    tiny = np.full((50, 50, 3), 128, dtype=np.uint8)
    report = check_image_quality(tiny)
    assert report.is_low_resolution is True
    assert any("resolution" in w.lower() for w in report.warnings)


def test_quality_check_flags_glare():
    bright = np.full((400, 400, 3), 253, dtype=np.uint8)
    report = check_image_quality(bright)
    assert report.has_glare is True


def test_quality_check_passes_normal_image(scratched_screen_image):
    report = check_image_quality(scratched_screen_image)
    # a reasonably sized, non-blown-out synthetic image shouldn't trip glare/low-res
    assert report.is_low_resolution is False
    assert report.has_glare is False


def test_preprocess_pipeline_returns_image_and_quality(clean_screen_image):
    out, quality = preprocess_pipeline(clean_screen_image)
    assert out.shape[2] == 3
    assert hasattr(quality, "warnings")
