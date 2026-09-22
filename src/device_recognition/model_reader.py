"""Read the device model name from text visible in a screenshot (OCR)."""
from __future__ import annotations

import re

import cv2

_reader = None

_PATTERNS = [
    r"iPhone\s?(?:SE|X[SR]?|\d{1,2})(?:\s?(?:Pro\s?Max|Pro|Plus|mini|e))?",
    r"iPad(?:\s?(?:Pro|Air|mini))?(?:\s?\d+(?:\.\d)?)?",
    r"MacBook(?:\s?(?:Air|Pro))?",
    r"Galaxy\s?(?:S|A|Z|Note)\s?\d{1,2}(?:\s?(?:Ultra|Plus|FE|\+))?",
    r"Pixel\s?\d{1,2}(?:\s?(?:Pro|XL|a))?",
]


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def read_model_name(image_rgb, min_conf: float = 0.5):
    """Return e.g. "iPhone 12 Pro Max" if the text names the device, else None."""
    try:
        h, w = image_rgb.shape[:2]
        scale = 1600 / max(h, w)
        if scale < 1:
            image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
        results = _get_reader().readtext(image_rgb)
    except Exception:
        return None
    for _, text, conf in results:
        if conf < min_conf:
            continue
        for pattern in _PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return " ".join(match.group(0).split())
    return None


_LABELS = {
    "model name": "model_name",
    "model number": "model_number",
    "serial number": "serial_number",
    "capacity": "capacity",
    "available": "available",
    "ios version": "ios_version",
}


def _rows(results):
    items = []
    for box, text, conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((sum(ys) / len(ys), min(xs), max(ys) - min(ys), text.strip(), conf))
    items.sort()
    rows = []
    for yc, x, h, text, conf in items:
        if rows and abs(yc - rows[-1]["y"]) < 0.6 * max(h, rows[-1]["h"]):
            rows[-1]["cells"].append((x, text, conf))
        else:
            rows.append({"y": yc, "h": h, "cells": [(x, text, conf)]})
    for row in rows:
        row["cells"].sort()
    return rows


def _fix_model_number(value):
    value = value.replace(" ", "").upper()
    if len(value) == 9 and value[7] in "J1I|":
        value = value[:7] + "/" + value[8:]
    return value


def read_device_info(image_rgb, min_conf: float = 0.5):
    """Read fields from a Settings > About screen: {field: {value, conf, verify}}."""
    try:
        h, w = image_rgb.shape[:2]
        scale = 1600 / max(h, w)
        if scale < 1:
            image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
        results = _get_reader().readtext(image_rgb)
    except Exception:
        return {}
    info = {}
    for row in _rows(results):
        cells = row["cells"]
        for _, text, conf in cells:
            match = re.search(r"coverage\s+\w+", text, flags=re.IGNORECASE)
            if match and conf >= min_conf:
                info["coverage"] = {
                    "value": " ".join(match.group(0).split()).title(),
                    "conf": round(float(conf), 2),
                    "verify": False,
                }
        if len(cells) < 2:
            continue
        key = _LABELS.get(cells[0][1].lower())
        value, conf = cells[-1][1], cells[-1][2]
        if not key or conf < min_conf:
            continue
        if key == "model_number":
            value = _fix_model_number(value)
        info[key] = {
            "value": value,
            "conf": round(float(conf), 2),
            "verify": bool(key in ("model_number", "serial_number") or conf < 0.85),
        }
    return info


def _to_gb(text):
    match = re.search(r"([\d.,]+)\s*(GB|TB)", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return value * 1000 if match.group(2).upper() == "TB" else value


def assess_settings(info):
    """Turn fields read from a Settings > About screen into a short verdict."""
    warnings = []
    if "expired" in info.get("coverage", {}).get("value", "").lower():
        warnings.append("Coverage expired")
    cap = _to_gb(info.get("capacity", {}).get("value"))
    avail = _to_gb(info.get("available", {}).get("value"))
    if cap and avail is not None and avail / cap < 0.10:
        warnings.append(f"Low storage: {avail:g} GB free of {cap:g} GB")
    return {"status": "ATTENTION" if warnings else "OK", "warnings": warnings}
