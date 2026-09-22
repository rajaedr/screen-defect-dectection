#!/usr/bin/env python3
"""
Download public datasets listed in DATASET_SETUP.md, using only legitimate,
authenticated access (the official `roboflow` Python package + your own free
API key). This script will NEVER attempt to scrape, bypass login walls, or
work around a dataset's access controls.

Most datasets referenced in DATASET_SETUP.md are small, community-contributed
Roboflow Universe projects (see that file for exact names/URLs/licenses).
Roboflow requires a (free) account + API key to export a dataset
programmatically, even for public CC-BY-licensed projects — that is a
platform requirement, not something we can or should route around.

Usage
-----
1. Create a free account at https://roboflow.com and copy your private API
   key from https://app.roboflow.com/settings/api
2. Set it as an environment variable (never hardcode it in source):
       export ROBOFLOW_API_KEY="your_key_here"
3. Run:
       python datasets/download_datasets.py

If ROBOFLOW_API_KEY is not set, or the `roboflow` package is not installed,
this script prints exactly what to download manually and where to place it,
instead of failing silently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"

# Kept in sync with DATASET_SETUP.md — update both together.
DATASETS = [
    {
        "name": "crackedscreen",
        "workspace": "20701s-workspace",
        "project": "crackedscreen",
        "version": 1,
        "url": "https://universe.roboflow.com/20701s-workspace/crackedscreen",
        "license": "CC BY 4.0",
    },
    {
        "name": "mobile-damage-diagnosis",
        "workspace": "test2-cedxe",
        "project": "mobile-damage-diagnosis-14usw",
        "version": 1,
        "url": "https://universe.roboflow.com/test2-cedxe/mobile-damage-diagnosis-14usw",
        "license": "CC BY 4.0",
    },
    {
        "name": "lens_defect_scratch",
        "workspace": "lensdefect",
        "project": "lens_defect_scratch-vmydi",
        "version": 1,
        "url": "https://universe.roboflow.com/lensdefect/lens_defect_scratch-vmydi",
        "license": "CC BY 4.0",
    },
]


def print_manual_instructions() -> None:
    print("\n" + "=" * 78)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 78)
    print(
        "Automatic download needs a free Roboflow account + API key "
        "(ROBOFLOW_API_KEY env var) and the `roboflow` pip package.\n"
        "Without those, download each dataset by hand:\n"
    )
    for d in DATASETS:
        print(f"- {d['name']}  (license: {d['license']})")
        print(f"    1. Open {d['url']}")
        print(f"    2. Click 'Download Dataset' -> format 'YOLOv8' -> download zip")
        print(f"    3. Unzip into: datasets/raw/{d['name']}/")
        print()
    print("See DATASET_SETUP.md for the full list, licenses, and details.")
    print("=" * 78 + "\n")


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("The 'roboflow' package is not installed (it's optional — see "
              "requirements.txt 'dataset-download' extra).")
        print_manual_instructions()
        return

    if not api_key:
        print("ROBOFLOW_API_KEY environment variable is not set.")
        print_manual_instructions()
        return

    rf = Roboflow(api_key=api_key)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    any_failed = False
    for d in DATASETS:
        dest = RAW_DIR / d["name"]
        if dest.exists():
            print(f"[skip] {d['name']} already exists at {dest}")
            continue
        try:
            print(f"[download] {d['name']} from {d['url']} ...")
            project = rf.workspace(d["workspace"]).project(d["project"])
            dataset = project.version(d["version"]).download("yolov8", location=str(dest))
            print(f"[ok] {d['name']} -> {dataset.location}")
        except Exception as e:  # noqa: BLE001
            any_failed = True
            print(f"[failed] {d['name']}: {e}")
            print(f"         Manually download from {d['url']} instead.")

    if any_failed:
        print("\nSome datasets could not be downloaded automatically.")
        print_manual_instructions()

    print("\nNext step: python datasets/prepare_dataset.py")


if __name__ == "__main__":
    main()
