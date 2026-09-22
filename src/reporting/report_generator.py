"""
Inspection report generation — JSON and PDF.

JSON is the canonical machine-readable format (also handy for the pytest
suite). PDF is generated with reportlab (pure-Python, no system
dependencies) for a human-readable "Download Inspection Report" deliverable.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)

from src.damage_detection.defect_detector import Defect, BBox
from src.device_recognition.device_classifier import DeviceResult
from src.detection.screen_detector import ScreenResult
from src.severity.severity import SeverityResult
from src.utils.config_loader import load_config


def build_report_dict(
    device: DeviceResult,
    screen: ScreenResult,
    defects: List[Defect],
    severity: SeverityResult,
    image_filename: str = "uploaded_image",
) -> dict:
    cfg = load_config()
    confirmed = [d for d in defects if not d.uncertain]
    uncertain = [d for d in defects if d.uncertain]

    return {
        "report_meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "system": cfg["project"]["name"],
            "version": cfg["project"]["version"],
            "image_filename": image_filename,
            "disclaimer": (
                "Automated prototype result for a university Manufacturing "
                "Practice project. Not a certified quality-control decision. "
                "Manual verification is recommended, especially for "
                "ATTENTION/FAIL results and any 'possible' (uncertain) findings."
            ),
        },
        "device": {
            "device_type": device.device_type,
            "confidence": device.confidence,
            "model": device.model_name,
            "reasoning": device.reasoning,
        },
        "screen": {
            "detected": screen.detected,
            "bbox": list(screen.bbox) if screen.bbox else None,
            "confidence": screen.confidence,
            "method": screen.method,
        },
        "inspection_result": severity.inspection_result,
        "severity": severity.severity,
        "severity_score": severity.score,
        "severity_explanation": severity.explanation,
        "defects_confirmed": [
            {"type": d.class_name, "confidence": d.confidence, "bbox": list(d.bbox),
             "notes": d.notes, "method": d.method}
            for d in confirmed
        ],
        "defects_uncertain": [
            {"type": d.class_name, "confidence": d.confidence, "bbox": list(d.bbox),
             "notes": d.notes, "method": d.method}
            for d in uncertain
        ],
    }


def report_to_json_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2).encode("utf-8")


def report_to_pdf_bytes(report: dict, annotated_image_rgb=None) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18)
    h2 = styles["Heading2"]
    normal = styles["Normal"]
    small = ParagraphStyle("Small", parent=normal, fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph("AI Visual Screen Inspection Report", title_style))
    story.append(Paragraph(report["report_meta"]["system"] + " · v" + report["report_meta"]["version"], small))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(report["report_meta"]["disclaimer"], small))
    story.append(Spacer(1, 6 * mm))

    if annotated_image_rgb is not None:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(annotated_image_rgb)
        img_buf = BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)
        max_w = 150 * mm
        max_h = 200 * mm
        w, h = pil_img.size
        scale = min(1.0, max_w / w, max_h / h)
        story.append(RLImage(img_buf, width=w * scale, height=h * scale))
        story.append(Spacer(1, 6 * mm))

    from reportlab.platypus import CondPageBreak

    story.append(CondPageBreak(70 * mm))
    story.append(Paragraph("Summary", h2))
    result = report["inspection_result"]
    result_color = {"PASS": colors.green, "ATTENTION": colors.orange, "FAIL": colors.red}.get(result, colors.black)
    summary_data = [
        ["Device Type", report["device"]["device_type"]],
        ["Device Model", report["device"]["model"]],
        ["Screen Detected", "Yes" if report["screen"]["detected"] else "No"],
        ["Overall Result", result],
        ["Severity", report["severity"] or "N/A"],
    ]
    t = Table(summary_data, colWidths=[55 * mm, 100 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("TEXTCOLOR", (1, 3), (1, 3), result_color),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)

    _ds = report.get("device_settings")
    if _ds:
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Device Settings", h2))
        if _ds["status"] == "OK":
            story.append(Paragraph("Settings look fine.", small))
        else:
            for _w in _ds["warnings"]:
                story.append(Paragraph("Warning: " + _w, small))
        story.append(Spacer(1, 3 * mm))
        _labels = {
            "model_name": "Model name",
            "model_number": "Model number",
            "serial_number": "Serial number",
            "ios_version": "iOS version",
            "capacity": "Storage capacity",
            "available": "Storage available",
            "coverage": "Coverage",
        }
        _rows = [["Field", "Value", "Check"]]
        for _k, _lbl in _labels.items():
            _f = _ds["fields"].get(_k)
            if _f:
                _rows.append([_lbl, _f["value"], "verify manually" if _f["verify"] else "OK"])
        _t = Table(_rows, colWidths=[45 * mm, 65 * mm, 40 * mm])
        _t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))
        story.append(_t)
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("Read automatically from the screenshot text. Fields marked 'verify manually' may contain misread characters.", small))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Confirmed Defects", h2))
    confirmed = report["defects_confirmed"]
    if confirmed:
        rows = [["#", "Type", "Confidence", "Location (x1,y1,x2,y2)"]]
        for i, d in enumerate(confirmed, 1):
            rows.append([str(i), d["type"], f"{int(d['confidence']*100)}%", str(tuple(d["bbox"]))])
        t2 = Table(rows, colWidths=[10 * mm, 40 * mm, 25 * mm, 80 * mm])
        t2.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t2)
    else:
        story.append(Paragraph("None.", normal))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Possible (Uncertain) Defects — Manual Inspection Recommended", h2))
    uncertain = report["defects_uncertain"]
    if uncertain:
        rows = [["#", "Type", "Confidence", "Location (x1,y1,x2,y2)"]]
        for i, d in enumerate(uncertain, 1):
            rows.append([str(i), d["type"], f"{int(d['confidence']*100)}%", str(tuple(d["bbox"]))])
        t3 = Table(rows, colWidths=[10 * mm, 40 * mm, 25 * mm, 80 * mm])
        t3.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t3)
    else:
        story.append(Paragraph("None.", normal))

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Severity Explanation", h2))
    story.append(Paragraph(report["severity_explanation"] or "N/A", normal))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Device Reasoning", h2))
    story.append(Paragraph(report["device"]["reasoning"] or "N/A", normal))

    doc.build(story)
    return buf.getvalue()
