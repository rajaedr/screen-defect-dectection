"""
Defect detection & classification.

Two modes (config.yaml -> damage_detection.mode):

  "heuristic" (default, zero-training baseline)
      A classical computer-vision pipeline (edges, line segments, blob
      analysis) that gives the MVP something to actually show on day one,
      before any dataset has been collected/trained on. It is intentionally
      conservative and clearly labeled: it will miss subtle defects and can
      be fooled by fingerprints/dust/reflections/on-screen content. Every
      detection below `uncertain_confidence_threshold` is reported as
      "uncertain" (POSSIBLE DEFECT - MANUAL INSPECTION REQUIRED) rather than
      a confirmed defect -- see config.yaml section 15/false-positive
      control in the original spec.

  "yolo" (recommended once you have a trained model -- see training/train.py
      and DATASET_SETUP.md)
      Loads a fine-tuned Ultralytics YOLOv8 detection (or segmentation)
      model from models/defect_yolo.pt and returns its detections directly.

Both modes return a list of `Defect` objects with the SAME shape, so the
rest of the pipeline (severity, reporting, UI) never needs to know which
mode produced them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2 in full-image pixel coords


@dataclass
class Defect:
    class_name: str
    confidence: float
    bbox: BBox
    uncertain: bool = False
    method: str = "heuristic"
    notes: str = ""


# -----------------------------------------------------------------------------
# Heuristic (classical CV) baseline
# -----------------------------------------------------------------------------

def _glare_mask(hsv: np.ndarray, brightness_threshold: int) -> np.ndarray:
    """Pixels that are very bright and low-saturation are almost certainly
    glare/reflection/blown-out highlight, not physical damage. We exclude
    these from consideration to reduce false positives."""
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    return (v > brightness_threshold) & (s < 40)


def _line_segments(gray: np.ndarray, exclude_mask: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """Find long, mostly-straight line segments (candidate scratches/cracks)
    using probabilistic Hough transform. Returns (x1,y1,x2,y2,length)."""
    edges = cv2.Canny(gray, 60, 160)
    edges[exclude_mask] = 0

    min_dim = min(gray.shape[:2])
    min_line_len = max(15, int(min_dim * 0.04))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=25,
        minLineLength=min_line_len,
        maxLineGap=6,
    )
    segments = []
    if lines is not None:
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [int(v) for v in l]
            length = float(np.hypot(x2 - x1, y2 - y1))
            segments.append((x1, y1, x2, y2, length))
    return segments


def _cluster_segments_to_defects(
    segments: List[Tuple[int, int, int, int, float]],
    region_offset: Tuple[int, int],
    region_diag: float,
) -> List[Defect]:
    """Turn raw line segments into scratch/crack defects.

    Heuristic distinguishing rule:
      - A cluster with many segments converging/crossing near a common area
        (branching pattern) -> "crack" (cracks in glass typically spider out).
      - A small number of long, parallel/isolated segments -> "scratch".
    This is a simplification appropriate for an MVP baseline, not a
    physically rigorous fracture-mechanics model.
    """
    if not segments:
        return []

    ox, oy = region_offset
    # Sort by length, keep the more significant segments to avoid noise explosion
    segments = sorted(segments, key=lambda s: -s[4])[:60]

    # crude spatial clustering by bounding-box proximity
    clusters: List[List[Tuple[int, int, int, int, float]]] = []
    used = [False] * len(segments)

    def bbox_of(seg):
        x1, y1, x2, y2, _ = seg
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def overlaps_or_near(b1, b2, pad):
        return not (b1[2] + pad < b2[0] or b2[2] + pad < b1[0] or
                    b1[3] + pad < b2[1] or b2[3] + pad < b1[1])

    pad = max(6, int(region_diag * 0.01))
    for i, seg in enumerate(segments):
        if used[i]:
            continue
        cluster = [seg]
        used[i] = True
        changed = True
        while changed:
            changed = False
            cluster_bbox = None
            for cs in cluster:
                b = bbox_of(cs)
                cluster_bbox = b if cluster_bbox is None else (
                    min(cluster_bbox[0], b[0]), min(cluster_bbox[1], b[1]),
                    max(cluster_bbox[2], b[2]), max(cluster_bbox[3], b[3]),
                )
            for j, seg2 in enumerate(segments):
                if used[j]:
                    continue
                if overlaps_or_near(cluster_bbox, bbox_of(seg2), pad):
                    cluster.append(seg2)
                    used[j] = True
                    changed = True
        clusters.append(cluster)

    defects = []
    for cluster in clusters:
        xs1, ys1, xs2, ys2 = [], [], [], []
        total_len = 0.0
        for x1, y1, x2, y2, length in cluster:
            xs1 += [x1, x2]
            ys1 += [y1, y2]
            total_len += length
        x_min, x_max = min(xs1), max(xs1)
        y_min, y_max = min(ys1), max(ys1)

        n_segments = len(cluster)
        avg_len = total_len / n_segments

        # crack heuristic: several intersecting/branching segments in a
        # compact area -> spidery pattern
        is_crack_like = n_segments >= 4 and avg_len > region_diag * 0.02

        class_name = "crack" if is_crack_like else "scratch"
        # confidence grows with total evidence (segment count & length),
        # capped conservatively since this is a classical baseline
        raw_score = min(1.0, (total_len / region_diag) * 1.2 + 0.15 * min(n_segments, 4))
        confidence = round(0.25 + raw_score * 0.55, 2)  # keep in a believable ~0.25-0.8 band

        bbox = (x_min + ox, y_min + oy, x_max + ox, y_max + oy)
        defects.append(Defect(class_name=class_name, confidence=confidence, bbox=bbox))

    return defects


def _group_consecutive(idxs: np.ndarray) -> List[Tuple[int, int]]:
    """Group consecutive/near-consecutive indices into (start, end) spans so
    a single stuck line doesn't get reported as many 1px-apart duplicates."""
    if len(idxs) == 0:
        return []
    idxs = sorted(idxs.tolist())
    spans = []
    start = prev = idxs[0]
    for i in idxs[1:]:
        if i - prev <= 2:
            prev = i
            continue
        spans.append((start, prev))
        start = prev = i
    spans.append((start, prev))
    return spans


def _detect_display_lines(gray: np.ndarray, region_offset: Tuple[int, int]) -> List[Defect]:
    """Full-width/height anomalous rows/columns -> candidate stuck display
    lines (common flat-panel defect).

    The outermost few percent of rows/columns are skipped: when the screen
    region comes from an imperfect contour-based crop, the very edge often
    still contains a sliver of bezel, which would otherwise look like a
    "line" artifact and cause false positives.
    """
    ox, oy = region_offset
    h, w = gray.shape
    border_h = max(2, int(h * 0.03))
    border_w = max(2, int(w * 0.03))
    defects = []

    # Use the MEDIAN per row/column rather than the mean: a genuine stuck
    # line is a near-uniform strip of one color across the whole row/column,
    # while a single small dust speck / dead-pixel blob only shifts the
    # mean of an otherwise-uniform row, not its median. This keeps a lone
    # dark spot from being mistaken for a full-width line.
    row_profile = np.median(gray, axis=1)
    col_profile = np.median(gray, axis=0)

    def find_outliers(profile, lo, hi):
        interior = profile[lo:hi]
        if len(interior) < 5:
            return np.array([], dtype=int), profile
        mu, sigma = float(interior.mean()), float(interior.std() + 1e-6)
        z = (profile - mu) / sigma
        idxs = np.where(np.abs(z) > 4.0)[0]
        idxs = idxs[(idxs >= lo) & (idxs < hi)]
        return idxs, z

    row_idxs, row_z = find_outliers(row_profile, border_h, h - border_h)
    for start, end in _group_consecutive(row_idxs):
        peak_z = float(np.max(np.abs(row_z[start:end + 1])))
        conf = round(min(0.85, 0.35 + peak_z * 0.08), 2)
        defects.append(Defect(
            class_name="display_line",
            confidence=conf,
            bbox=(ox, oy + start, ox + w, oy + end + 2),
            notes="Full-width anomalous row band (possible stuck horizontal line).",
        ))

    col_idxs, col_z = find_outliers(col_profile, border_w, w - border_w)
    for start, end in _group_consecutive(col_idxs):
        peak_z = float(np.max(np.abs(col_z[start:end + 1])))
        conf = round(min(0.85, 0.35 + peak_z * 0.08), 2)
        defects.append(Defect(
            class_name="display_line",
            confidence=conf,
            bbox=(ox + start, oy, ox + end + 2, oy + h),
            notes="Full-height anomalous column band (possible stuck vertical line).",
        ))

    # cap: don't flood the report with dozens of near-duplicate line detections
    defects = sorted(defects, key=lambda d: -d.confidence)[:5]
    return defects


def _detect_spots(gray: np.ndarray, exclude_mask: np.ndarray, region_offset: Tuple[int, int],
                   region_area: float) -> List[Defect]:
    """Small, compact, very dark blobs -> candidate dead pixels / black spots."""
    ox, oy = region_offset
    _, dark_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
    dark_mask[exclude_mask] = 0
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    for c in contours:
        area = cv2.contourArea(c)
        area_ratio = area / region_area
        if area_ratio < 0.00005 or area_ratio > 0.01:
            # too small to be meaningful, or too big to be a "spot"
            # (large dark regions are handled as broken_glass/other elsewhere)
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch + 1e-6)
        compactness = area / float(cw * ch + 1e-6)
        if compactness < 0.4:
            continue  # not blob-like enough; likely noise/text
        class_name = "dead_pixel" if area_ratio < 0.0006 else "black_spot"
        confidence = round(0.3 + min(compactness, 1.0) * 0.35, 2)
        defects.append(Defect(
            class_name=class_name,
            confidence=confidence,
            bbox=(x + ox, y + oy, x + cw + ox, y + ch + oy),
        ))
    # avoid flooding results with dozens of dust-speck detections
    defects = sorted(defects, key=lambda d: -d.confidence)[:8]
    return defects


def _detect_heuristic(image_rgb: np.ndarray, screen_bbox: Optional[BBox]) -> List[Defect]:
    cfg = load_config()["damage_detection"]

    if screen_bbox is not None:
        x1, y1, x2, y2 = screen_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)
        region = image_rgb[y1:y2, x1:x2]
        offset = (x1, y1)
    else:
        region = image_rgb
        offset = (0, 0)

    if region.size == 0:
        return []

    # Inset the analysis region by a small margin. When the screen bbox
    # comes from the contour-based localizer, its own outer edge is often
    # included as part of the crop and is itself a strong, perfectly
    # straight edge -- which would otherwise be misread as a scratch/crack
    # running along all four sides of the screen. Insetting avoids that
    # without meaningfully shrinking the area we actually inspect.
    rh, rw = region.shape[:2]
    inset_y = max(2, int(rh * 0.02))
    inset_x = max(2, int(rw * 0.02))
    if rh > 2 * inset_y + 10 and rw > 2 * inset_x + 10:
        inset_region = region[inset_y:rh - inset_y, inset_x:rw - inset_x]
        offset = (offset[0] + inset_x, offset[1] + inset_y)
        region = inset_region

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    exclude = _glare_mask(hsv, brightness_threshold=245)

    region_diag = float(np.hypot(*region.shape[:2]))
    region_area = float(region.shape[0] * region.shape[1])

    defects: List[Defect] = []

    segments = _line_segments(gray, exclude)
    defects += _cluster_segments_to_defects(segments, offset, region_diag)
    defects += _detect_display_lines(gray, offset)
    defects += _detect_spots(gray, exclude, offset, region_area)

    # apply uncertain flag + confidence threshold filtering
    conf_thresh = cfg["confidence_threshold"]
    uncertain_thresh = cfg["uncertain_confidence_threshold"]
    filtered = []
    for d in defects:
        if d.confidence < conf_thresh:
            continue
        d.uncertain = d.confidence < uncertain_thresh
        d.method = "heuristic"
        filtered.append(d)

    filtered = filtered[: cfg["max_detections"]]
    return filtered


# -----------------------------------------------------------------------------
# YOLO (trained model) mode
# -----------------------------------------------------------------------------

_yolo_defect_model = None


def _load_yolo_model():
    global _yolo_defect_model
    if _yolo_defect_model is not None:
        return _yolo_defect_model
    cfg = load_config()["damage_detection"]
    model_path = resolve_path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Defect detection model not found at {model_path}. Train one "
            "with training/train.py (see DATASET_SETUP.md), or set "
            "damage_detection.mode back to 'heuristic' in config.yaml."
        )
    from ultralytics import YOLO

    _yolo_defect_model = YOLO(str(model_path))
    return _yolo_defect_model


def _detect_yolo(image_rgb: np.ndarray) -> List[Defect]:
    cfg = load_config()["damage_detection"]
    model = _load_yolo_model()
    results = model.predict(
        image_rgb,
        imgsz=cfg["image_size"],
        conf=cfg["confidence_threshold"],
        iou=cfg["iou_threshold"],
        max_det=cfg["max_detections"],
        verbose=False,
    )
    r = results[0]
    defects = []
    uncertain_thresh = cfg["uncertain_confidence_threshold"]
    for box in r.boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names.get(cls_idx, f"class_{cls_idx}")
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        defects.append(Defect(
            class_name=class_name,
            confidence=round(conf, 2),
            bbox=(x1, y1, x2, y2),
            uncertain=conf < uncertain_thresh,
            method="yolo",
        ))
    return defects


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def detect_defects(image_rgb: np.ndarray, screen_bbox: Optional[BBox] = None) -> List[Defect]:
    """Detect and classify visible screen defects.

    Parameters
    ----------
    image_rgb: full (preprocessed) image, RGB uint8.
    screen_bbox: optional screen region from screen_detector; when provided,
        the heuristic baseline restricts analysis to that region for better
        precision (YOLO mode currently runs on the full image, since a
        properly trained detector learns screen context itself).
    """
    cfg = load_config()["damage_detection"]
    mode = cfg.get("mode", "heuristic")

    if mode == "yolo":
        try:
            return _detect_yolo(image_rgb)
        except FileNotFoundError as e:
            logger.warning("%s Falling back to heuristic mode.", e)
            return _detect_heuristic(image_rgb, screen_bbox)
    return _detect_heuristic(image_rgb, screen_bbox)
