import io

import numpy as np
import pytest
from PIL import Image

from src.utils.image_io import ImageLoadError, load_image_as_rgb_array


def _make_png_bytes(w=50, h=50) -> bytes:
    img = Image.new("RGB", (w, h), color=(120, 130, 140))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_load_valid_png_bytes_returns_rgb_array():
    data = _make_png_bytes()
    arr = load_image_as_rgb_array(data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (50, 50, 3)
    assert arr.dtype == np.uint8


def test_load_corrupted_bytes_raises_friendly_error():
    corrupted = b"this is not an image, just random bytes 12345"
    with pytest.raises(ImageLoadError):
        load_image_as_rgb_array(corrupted)


def test_load_missing_file_raises_friendly_error(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageLoadError):
        load_image_as_rgb_array(missing)


def test_unsupported_extension_raises_friendly_error(tmp_path):
    bad_file = tmp_path / "not_an_image.txt"
    bad_file.write_text("hello")
    with pytest.raises(ImageLoadError):
        load_image_as_rgb_array(bad_file)


def test_load_from_path_roundtrip(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(_make_png_bytes(30, 40))
    arr = load_image_as_rgb_array(path)
    assert arr.shape == (40, 30, 3)
