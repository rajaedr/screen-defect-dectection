#!/usr/bin/env python3
"""
Merge every dataset under datasets/raw/<name>/{images,labels} (YOLO-format:
one .txt per image with `class_id x_center y_center width height`, all
normalized 0-1) into a single datasets/processed/ tree with a train/val/test
split, remapped onto this project's unified class list
(config.yaml -> damage_detection.classes), and writes the Ultralytics
data.yaml that training/train.py consumes.

Each raw sub-dataset needs a `classes.txt` (one class name per line, in the
order matching its label files' class indices) OR a `data.yaml` with a
`names:` list — either is auto-detected. If a raw dataset's class name isn't
recognized, edit CLASS_NAME_MAP below to map it onto one of our classes
(this is expected: different public datasets name things differently, e.g.
"Cracked-screen" vs "crack" vs "screen_crack" all mean the same thing to us).

Usage:
    python datasets/prepare_dataset.py
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config  # noqa: E402

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -----------------------------------------------------------------------------
# EDIT ME: map a raw dataset's class-name strings (lowercased) onto one of
# this project's unified classes (config.yaml -> damage_detection.classes).
# Anything not listed here, and not already an exact match to a unified
# class name, is skipped (its label lines are dropped) with a warning so you
# can decide whether to extend this map.
# -----------------------------------------------------------------------------
CLASS_NAME_MAP: Dict[str, str] = {
    "cracked-screen": "crack",
    "cracked screen": "crack",
    "screen crack": "crack",
    "broken phone": "broken_glass",
    "not broken phone": "no_defect",
    "has_damage": "other_damage",
    "no_damage": "no_defect",
    "scratch": "scratch",
    "dead pixel": "dead_pixel",
    "dent": "other_damage",
    "pits": "other_damage",
    "collapse": "other_damage",
    "digs": "scratch",
    "internal crack": "crack",
    "glass stick": "other_damage",
}


def load_class_names(dataset_dir: Path) -> List[str]:
    classes_txt = dataset_dir / "classes.txt"
    if classes_txt.exists():
        return [l.strip() for l in classes_txt.read_text().splitlines() if l.strip()]

    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        with open(data_yaml) as f:
            d = yaml.safe_load(f)
        names = d.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return names

    raise FileNotFoundError(
        f"No classes.txt or data.yaml with 'names' found in {dataset_dir}. "
        "Add one (see datasets/README.md) before running this script."
    )


def remap_class_index(raw_name: str, unified_classes: List[str]) -> Optional[int]:
    key = raw_name.strip().lower()
    if key in unified_classes:
        target = key
    elif key in CLASS_NAME_MAP:
        target = CLASS_NAME_MAP[key]
    else:
        return None
    return unified_classes.index(target) if target in unified_classes else None


def find_image_label_pairs(dataset_dir: Path) -> List[tuple[Path, Path]]:
    pairs = []

    # Support datasets with images/ and labels/ directly
    split_dirs = [(dataset_dir / "images", dataset_dir / "labels")]

    # Also support standard YOLO train/valid/test structure
    for split in ["train", "valid", "val", "test"]:
        split_dirs.append(
            (
                dataset_dir / split / "images",
                dataset_dir / split / "labels",
            )
        )

    for images_dir, labels_dir in split_dirs:
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for img_path in sorted(images_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = labels_dir / (img_path.stem + ".txt")

            if label_path.exists():
                pairs.append((img_path, label_path))

    return pairs


def rewrite_label_file(src_label: Path, raw_classes: List[str], unified_classes: List[str]) -> Optional[str]:
    """Return new label-file content with remapped class indices.
    Empty label files (background images with no defects) are kept as an
    empty string. Returns None only if the file had labels but none of
    them mapped to a known class."""
    lines = [l for l in src_label.read_text().splitlines() if l.strip()]
    if not lines:
        return ""  # background image: keep it as a negative example

    out_lines = []
    for line in lines:
        parts = line.strip().split()
        try:
            old_idx = int(parts[0])
            raw_name = raw_classes[old_idx]
        except (ValueError, IndexError):
            continue
        new_idx = remap_class_index(raw_name, unified_classes)
        if new_idx is None:
            continue
        out_lines.append(" ".join([str(new_idx)] + parts[1:]))
    if not out_lines:
        return None
    return "\n".join(out_lines) + "\n"



def main() -> None:
    cfg = load_config()
    unified_classes = [c.lower() for c in cfg["damage_detection"]["classes"]]
    split_ratios = cfg["training"]["train_val_test_split"]
    seed = cfg["training"]["seed"]

    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"No raw datasets found in {RAW_DIR}.")
        print("Run datasets/download_datasets.py first, or see DATASET_SETUP.md "
              "for manual download instructions, or add your own photos "
              "(see datasets/README.md).")
        return

    all_pairs: List[tuple[Path, Path, List[str]]] = []
    for dataset_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        try:
            raw_classes = [c.lower() for c in load_class_names(dataset_dir)]
        except FileNotFoundError as e:
            print(f"[skip] {dataset_dir.name}: {e}")
            continue
        pairs = find_image_label_pairs(dataset_dir)
        print(f"[found] {dataset_dir.name}: {len(pairs)} labeled images, classes={raw_classes}")
        for img, lbl in pairs:
            all_pairs.append((img, lbl, raw_classes))

    if not all_pairs:
        print("No labeled image/label pairs found across any raw dataset. Nothing to prepare.")
        return

        # Group by source photo so augmented copies never land in different splits
    groups = defaultdict(list)
    for item in all_pairs:
        groups[item[0].stem.split(".rf.")[0]].append(item)

    keys = sorted(groups)
    random.seed(seed)
    random.shuffle(keys)

    n = len(all_pairs)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    splits = {"train": [], "val": [], "test": []}
    for k in keys:
        if len(splits["train"]) < n_train:
            splits["train"].extend(groups[k])
        elif len(splits["val"]) < n_val:
            splits["val"].extend(groups[k])
        else:
            splits["test"].extend(groups[k])

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    kept, dropped = 0, 0
    for split_name, items in splits.items():
        img_out = PROCESSED_DIR / "images" / split_name
        lbl_out = PROCESSED_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path, raw_classes in items:
            new_label_content = rewrite_label_file(lbl_path, raw_classes, unified_classes)
            if new_label_content is None:
                dropped += 1
                continue
            kept += 1
            unique_name = f"{img_path.parent.parent.name}_{img_path.stem}{img_path.suffix}"
            shutil.copy2(img_path, img_out / unique_name)
            (lbl_out / (Path(unique_name).stem + ".txt")).write_text(new_label_content)

    data_yaml = {
        "path": str(PROCESSED_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(unified_classes)},
    }
    with open(PROCESSED_DIR / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"\nDone. Kept {kept} image/label pairs, dropped {dropped} "
          "(unrecognized classes — extend CLASS_NAME_MAP in this script if needed).")
    print(f"Split sizes -> train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")
    print(f"Wrote {PROCESSED_DIR / 'data.yaml'}")
    print("\nNext step: python training/train.py")


if __name__ == "__main__":
    main()
