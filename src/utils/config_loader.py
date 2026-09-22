"""
Config loader utility.

Loads config/config.yaml (and the separate severity_rules.yaml it points to)
so nothing in the codebase hardcodes paths, thresholds, or hyperparameters.

Usage:
    from src.utils.config_loader import load_config
    cfg = load_config()
    cfg["damage_detection"]["confidence_threshold"]
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Return the project root directory (the one containing config/, src/, app/)."""
    # this file lives at <root>/src/utils/config_loader.py
    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml (which is written relative to project root)
    into an absolute path, regardless of the current working directory."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return get_project_root() / p


@functools.lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load and cache the main configuration file.

    Parameters
    ----------
    config_path: optional override path to a config yaml. Defaults to
        <project_root>/config/config.yaml
    """
    if config_path is None:
        config_path = str(get_project_root() / "config" / "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Did you run this from the project directory?"
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


@functools.lru_cache(maxsize=1)
def load_severity_rules() -> Dict[str, Any]:
    """Load the separate, independently editable severity rules file."""
    cfg = load_config()
    rules_path = resolve_path(cfg["severity"]["rules_file"])
    if not rules_path.exists():
        raise FileNotFoundError(f"Severity rules file not found at {rules_path}")
    with open(rules_path, "r") as f:
        return yaml.safe_load(f)


def clear_config_cache() -> None:
    """Useful in tests: forces config.yaml / severity_rules.yaml to be re-read."""
    load_config.cache_clear()
    load_severity_rules.cache_clear()
