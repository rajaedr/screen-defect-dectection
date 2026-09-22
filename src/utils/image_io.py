"""
Image loading & validation helpers.

Centralizes error handling for:
- unsupported image formats
- corrupted images
- unreadable files

so every caller (Streamlit app, scripts, tests) gets the same friendly
exceptions instead of raw stack traces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or is invalid for inspection."""


def is_supported_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def load_image_as_rgb_array(source: Union[str, Path, bytes, "Image.Image"]) -> np.ndarray:
    """Load an image from a path, raw bytes, or a PIL Image and return an
    HxWx3 uint8 RGB numpy array.

    Raises
    ------
    ImageLoadError
        On unsupported format, corrupted file, or empty/degenerate image.
    """
    try:
        if isinstance(source, Image.Image):
            pil_img = source
        elif isinstance(source, (bytes, bytearray)):
            import io

            pil_img = Image.open(io.BytesIO(source))
            pil_img.load()  # force decode now, so corrupt data raises here
        else:
            path = Path(source)
            if not path.exists():
                raise ImageLoadError(f"File not found: {path}")
            if not is_supported_extension(path.name):
                raise ImageLoadError(
                    f"Unsupported image format '{path.suffix}'. "
                    f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            pil_img = Image.open(path)
            pil_img.load()

        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)

        if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise ImageLoadError("Image is empty or has degenerate dimensions.")

        return arr

    except UnidentifiedImageError as e:
        raise ImageLoadError(
            "The file could not be read as an image. It may be corrupted or "
            "not actually an image file."
        ) from e
    except ImageLoadError:
        raise
    except Exception as e:  # noqa: BLE001 - we deliberately convert everything
        raise ImageLoadError(f"Failed to load image: {e}") from e


def array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))
