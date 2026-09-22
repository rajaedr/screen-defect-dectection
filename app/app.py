"""
AI Visual Screen Inspection — Streamlit application.

Run with:
    streamlit run app/app.py
(or simply ./run.sh from the project root)
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `python -m streamlit run app/app.py` or `streamlit run app/app.py`
# to find the `src` package regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.report_view import render_quality_warnings, render_report
from src.pipeline import run_inspection
from src.utils.config_loader import load_config
from src.utils.image_io import ImageLoadError, load_image_as_rgb_array
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="AI Visual Screen Inspection", page_icon="📱", layout="centered")


def _run_and_show(image_bytes: bytes, filename: str) -> None:
    try:
        image_rgb = load_image_as_rgb_array(image_bytes)
    except ImageLoadError as e:
        st.error(f"Could not process this image: {e}")
        return

    with st.spinner("Running inspection pipeline..."):
        try:
            result = run_inspection(image_rgb, filename)
        except Exception as e:  # noqa: BLE001
            logger.exception("Inspection pipeline failed")
            st.error(
                "Something went wrong while analyzing this image. "
                f"Technical detail: {e}"
            )
            return

    render_quality_warnings(result)

    st.markdown("### Original Image")
    st.image(result.original_image, use_container_width=True)

    st.markdown("### Annotated Detection Result")
    st.image(result.annotated_image, use_container_width=True)
    st.caption("Cyan = detected screen region. Red = confirmed defect. Orange = possible defect (uncertain).")

    render_report(result)

    st.markdown("### Download Inspection Report")
    col_a, col_b = st.columns(2)
    with col_a:
        from src.reporting.report_generator import report_to_json_bytes
        st.download_button(
            "⬇️ Download JSON report",
            data=report_to_json_bytes(result.report),
            file_name=f"inspection_report_{Path(filename).stem}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_b:
        from src.reporting.report_generator import report_to_pdf_bytes
        pdf_bytes = report_to_pdf_bytes(result.report, result.annotated_image)
        st.download_button(
            "⬇️ Download PDF report",
            data=pdf_bytes,
            file_name=f"inspection_report_{Path(filename).stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def main() -> None:
    cfg = load_config()

    st.title("📱 AI Visual Screen Inspection")
    st.caption(
        "Automated visual quality-control prototype for smartphone & laptop screens "
        "— Manufacturing Practice university project."
    )

    with st.expander("ℹ️ How to use this tool / limitations", expanded=False):
        st.markdown(
            """
- Upload a **photo of the physical device** (not a screenshot) for physical
  damage detection (scratches, cracks, broken glass, dead pixels, etc.).
  A screenshot only captures digital pixels and *cannot* show physical
  surface damage that wasn't in the capture.
- Use a **screenshot** only if you want to flag *digital/display* anomalies
  (e.g. stuck lines, discoloration) — see the "Screenshot mode" toggle below.
- This is a **university prototype**. It does not claim to catch every
  defect or work on every device. Low-confidence findings are shown as
  *"possible defect — manual inspection recommended"* rather than a
  confirmed result. See `MODEL_CARD.md` for full details.
- Images are processed **locally on this machine** — nothing is uploaded to
  an external server by this application.
            """
        )

    input_mode_label = "Screenshot mode (digital/display anomalies only)"
    screenshot_mode = st.checkbox(input_mode_label, value=False)
    if screenshot_mode:
        st.info(
            "Screenshot mode: physical damage detection (scratches/cracks/broken "
            "glass) is not applicable to screenshots and will be skipped in your "
            "own review of the results — only digital anomaly classes "
            "(stuck lines, discoloration, dead/stuck pixels) are meaningful here."
        )

    tab_upload, tab_camera, tab_pattern = st.tabs(["📁 Upload Image", "📷 Use Camera", "🎨 Test-pattern mode"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a photo of a smartphone or laptop",
            type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"],
        )
        if uploaded is not None:
            _run_and_show(uploaded.getvalue(), uploaded.name)

    with tab_camera:
        st.caption(
            "Camera capture uses your browser's camera via Streamlit's built-in "
            "widget. If your browser/OS blocks camera access, use the Upload "
            "tab instead — file upload always works."
        )
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            _run_and_show(camera_image.getvalue(), "camera_capture.jpg")

    with tab_pattern:
        st.markdown(
            """
**Optional screen inspection mode.** Display a solid test color full-screen
on the device (search "full screen color" or use a free "blank screen"
app/website), then photograph the screen and upload it here. Solid,
content-free colors make dead pixels, stuck lines, discoloration, and
backlight bleed much easier to see reliably than a normal wallpaper/app.
            """
        )
        colors = cfg["screen_test_pattern"]["colors"]
        st.write("Suggested test colors: " + ", ".join(c.capitalize() for c in colors))
        pattern_upload = st.file_uploader(
            "Upload a photo of the screen showing a solid test color",
            type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"],
            key="pattern_uploader",
        )
        if pattern_upload is not None:
            _run_and_show(pattern_upload.getvalue(), pattern_upload.name)

    st.divider()
    st.caption(
        f"{cfg['project']['name']} · v{cfg['project']['version']} · "
        f"Detection mode: device={cfg['device_recognition']['mode']}, "
        f"screen={cfg['screen_detection']['mode']}, "
        f"defects={cfg['damage_detection']['mode']}"
    )


if __name__ == "__main__":
    main()
