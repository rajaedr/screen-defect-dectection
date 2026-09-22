"""Shared pytest fixtures: synthetic images so the test suite runs
deterministically without needing real device photos."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import clear_config_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_config():
    """Ensure each test sees config.yaml freshly (in case another test
    process or an earlier run mutated the on-disk file in a fixture)."""
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def clean_screen_image() -> np.ndarray:
    """A synthetic 'laptop photo' with a uniform, defect-free screen."""
    img = np.full((600, 900, 3), 40, dtype=np.uint8)
    img[80:520, 100:800] = 200
    return img


@pytest.fixture
def scratched_screen_image() -> np.ndarray:
    img = np.full((600, 900, 3), 40, dtype=np.uint8)
    img[80:520, 100:800] = 230
    cv2.line(img, (200, 150), (650, 480), (150, 150, 150), 2)
    return img


@pytest.fixture
def phone_like_image() -> np.ndarray:
    """Tall/narrow synthetic image, consistent with a face-on phone photo."""
    img = np.full((900, 500, 3), 30, dtype=np.uint8)
    img[100: 800, 60:440] = 210
    return img


@pytest.fixture
def tmp_uploads_dir(tmp_path) -> Path:
    d = tmp_path / "uploads"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _force_heuristic_modes(_fresh_config):
    """Tests use synthetic images, so run the classical (heuristic) code paths.

    The app itself uses the trained models (see config/config.yaml).
    """
    from src.utils.config_loader import load_config

    cfg = load_config()
    saved = {}
    for section in ("device_recognition", "damage_detection"):
        saved[section] = cfg[section].get("mode")
        cfg[section]["mode"] = "heuristic"
    yield
    for section, mode in saved.items():
        if mode is None:
            cfg[section].pop("mode", None)
        else:
            cfg[section]["mode"] = mode
