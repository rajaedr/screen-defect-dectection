"""Reusable Streamlit rendering components for the inspection report."""
from __future__ import annotations

import streamlit as st

from src.pipeline import InspectionResult

RESULT_COLORS = {"PASS": "#1e8e3e", "ATTENTION": "#e8710a", "FAIL": "#d93025"}


def render_quality_warnings(result: InspectionResult) -> None:
    if result.quality.warnings:
        for w in result.quality.warnings:
            st.warning(w)


def render_report(result: InspectionResult) -> None:
    r = result.report
    result_label = r["inspection_result"]
    color = RESULT_COLORS.get(result_label, "#333")

    st.markdown("## Inspection Report")

    col1, col2, col3 = st.columns(3)
    col1.metric("Device Type", r["device"]["device_type"].capitalize())
    col2.metric("Screen Detected", "Yes" if r["screen"]["detected"] else "No")
    col3.markdown(
        f"<div style='text-align:center'><span style='font-size:0.8rem;color:gray'>Overall Result</span><br>"
        f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>{result_label}</span></div>",
        unsafe_allow_html=True,
    )

    st.caption(f"Device model: **{r['device']['model']}** — {r['device']['reasoning']}")
    if r["screen"]["detected"]:
        st.caption(f"Screen localization confidence: {int(r['screen']['confidence']*100)}% ({r['screen']['method']})")
    else:
        st.caption("Screen region could not be confidently localized; analysis ran on the full image.")

    st.markdown("### Defects Found")
    confirmed = r["defects_confirmed"]
    uncertain = r["defects_uncertain"]

    if not confirmed and not uncertain:
        st.success("No defects detected.")
    else:
        if confirmed:
            for i, d in enumerate(confirmed, 1):
                st.markdown(f"**{i}. {d['type'].replace('_', ' ').title()}** — {int(d['confidence']*100)}% confidence "
                            f"&nbsp;·&nbsp; location `{tuple(d['bbox'])}`")
        if uncertain:
            st.markdown("**Possible defects (below confirmation threshold) — manual inspection recommended:**")
            for i, d in enumerate(uncertain, 1):
                st.markdown(f"- {d['type'].replace('_', ' ').title()} — {int(d['confidence']*100)}% confidence "
                            f"&nbsp;·&nbsp; location `{tuple(d['bbox'])}`")

    st.markdown("### Severity")
    st.write(r["severity"] or "N/A")
    st.caption(r["severity_explanation"])

    with st.expander("Full JSON report"):
        st.json(r)

    render_device_settings(r)
    st.caption(r["report_meta"]["disclaimer"])


def render_device_settings(r):
    """Show the fields read from a Settings > About screenshot, plus a verdict."""
    ds = r.get("device_settings")
    if not ds:
        return
    st.markdown("### Device Settings")
    if ds["status"] == "OK":
        st.success("Settings look fine.")
    else:
        for w in ds["warnings"]:
            st.warning(w)
    labels = {
        "model_name": "Model name",
        "model_number": "Model number",
        "serial_number": "Serial number",
        "ios_version": "iOS version",
        "capacity": "Storage capacity",
        "available": "Storage available",
        "coverage": "Coverage",
    }
    rows = []
    for key, label in labels.items():
        f = ds["fields"].get(key)
        if f:
            rows.append({
                "Field": label,
                "Value": f["value"],
                "Check": "verify manually" if f["verify"] else "OK",
            })
    if rows:
        st.table(rows)
    st.caption("Read automatically from the screenshot text. Fields marked 'verify manually' may contain misread characters.")
