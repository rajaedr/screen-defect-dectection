#!/usr/bin/env python3
"""
Command-line inference: run the full inspection pipeline on one image
without launching the Streamlit UI. Useful for quick checks, scripting, or
batch processing.

Usage:
    python training/predict.py path/to/photo.jpg
    python training/predict.py path/to/photo.jpg --out results/reports/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_inspection  # noqa: E402
from src.reporting.report_generator import report_to_json_bytes, report_to_pdf_bytes  # noqa: E402
from src.utils.config_loader import resolve_path  # noqa: E402
from src.utils.image_io import ImageLoadError, array_to_pil, load_image_as_rgb_array  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("predict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run screen inspection on a single image.")
    parser.add_argument("image", type=str, help="Path to the input image.")
    parser.add_argument("--out", type=str, default="results/reports", help="Output directory for reports.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF report generation (faster).")
    args = parser.parse_args()

    image_path = Path(args.image)
    try:
        image_rgb = load_image_as_rgb_array(image_path)
    except ImageLoadError as e:
        logger.error("%s", e)
        sys.exit(1)

    result = run_inspection(image_rgb, image_path.name)

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    annotated_path = out_dir / f"{stem}_annotated.png"
    array_to_pil(result.annotated_image).save(annotated_path)
    logger.info("Annotated image saved to %s", annotated_path)

    json_path = out_dir / f"{stem}_report.json"
    json_path.write_bytes(report_to_json_bytes(result.report))
    logger.info("JSON report saved to %s", json_path)

    if not args.no_pdf:
        pdf_path = out_dir / f"{stem}_report.pdf"
        pdf_path.write_bytes(report_to_pdf_bytes(result.report, result.annotated_image))
        logger.info("PDF report saved to %s", pdf_path)

    print("\n--- Inspection Summary ---")
    print(f"Device type:       {result.device.device_type}")
    print(f"Screen detected:   {result.screen.detected}")
    print(f"Overall result:    {result.severity.inspection_result}")
    print(f"Severity:          {result.severity.severity}")
    confirmed = [d for d in result.defects if not d.uncertain]
    print(f"Confirmed defects: {len(confirmed)}")
    for d in confirmed:
        print(f"  - {d.class_name} ({int(d.confidence*100)}%) at {d.bbox}")


if __name__ == "__main__":
    main()
