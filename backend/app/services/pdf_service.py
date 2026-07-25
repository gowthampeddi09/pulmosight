"""
PDF report generation using reportlab.

Chosen over weasyprint because reportlab is pure Python — no system-level
dependencies (libcairo, pango, etc.), which makes Docker builds simpler
and faster. The trade-off is more manual layout code, but we get full
control over the hospital-style report format.
"""
import io
import os
import logging
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from app.models.prediction import Prediction

log = logging.getLogger(__name__)

PRIMARY_COLOR = HexColor("#0A2540")
ACCENT_COLOR = HexColor("#0EA5E9")
LIGHT_GRAY = HexColor("#F1F5F9")


def generate_pdf(prediction: Prediction) -> bytes:
    """Generate a hospital-style PDF report. Returns raw PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"],
        fontSize=20, textColor=PRIMARY_COLOR, alignment=TA_CENTER,
        spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=13, textColor=ACCENT_COLOR, spaceBefore=6 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        "ReportBody", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=8, leading=10, textColor=HexColor("#6B7280"), spaceAfter=2 * mm,
    ))

    elements = []

    # Header
    elements.append(Paragraph("PulmoSight — AI Chest X-Ray Analysis Report", styles["ReportTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=ACCENT_COLOR))
    elements.append(Spacer(1, 4 * mm))

    # Patient info table (if available)
    patient_data = [["Field", "Value"]]
    if prediction.patient_age:
        patient_data.append(["Age", str(prediction.patient_age)])
    if prediction.patient_gender:
        patient_data.append(["Gender", prediction.patient_gender])
    if prediction.patient_symptoms:
        patient_data.append(["Symptoms", prediction.patient_symptoms])
    patient_data.append(["Analysis Date", prediction.created_at.strftime("%Y-%m-%d %H:%M UTC")])
    patient_data.append(["Model Version", prediction.model_version])

    if len(patient_data) > 1:
        t = Table(patient_data, colWidths=[35 * mm, 130 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT_COLOR),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 4 * mm))

    # Prediction result
    label = prediction.prediction.value if hasattr(prediction.prediction, 'value') else str(prediction.prediction)
    confidence_pct = f"{prediction.confidence * 100:.1f}%"
    result_color = "#EF4444" if label == "PNEUMONIA" else "#22C55E"

    elements.append(Paragraph("AI Prediction Result", styles["SectionHead"]))
    result_data = [
        ["Prediction", "Confidence", "Processing Time"],
        [label, confidence_pct, f"{prediction.processing_time_ms:.0f}ms"],
    ]
    rt = Table(result_data, colWidths=[55 * mm, 55 * mm, 55 * mm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(rt)
    elements.append(Spacer(1, 4 * mm))

    # Embed heatmap if available
    if prediction.heatmap_path and os.path.exists(prediction.heatmap_path):
        elements.append(Paragraph("Grad-CAM Activation Heatmap", styles["SectionHead"]))
        try:
            img = RLImage(prediction.heatmap_path, width=120 * mm, height=120 * mm, kind="proportional")
            elements.append(img)
            elements.append(Spacer(1, 3 * mm))
        except Exception as e:
            log.warning("Could not embed heatmap in PDF: %s", e)

    # LLM Report sections
    if prediction.report_text:
        elements.append(Paragraph("Clinical Analysis Report", styles["SectionHead"]))
        # Parse markdown-ish sections into paragraphs
        for line in prediction.report_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("## "):
                elements.append(Paragraph(line[3:], styles["SectionHead"]))
            elif line.startswith("- "):
                elements.append(Paragraph(f"• {line[2:]}", styles["ReportBody"]))
            else:
                elements.append(Paragraph(line, styles["ReportBody"]))
    else:
        elements.append(Paragraph("Clinical report has not been generated yet.", styles["ReportBody"]))

    # Footer disclaimer
    elements.append(Spacer(1, 8 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CBD5E1")))
    elements.append(Paragraph(
        "<b>IMPORTANT DISCLAIMER:</b> This report is generated by an AI decision-support system "
        "for educational and portfolio demonstration purposes only. It is NOT a medical diagnosis. "
        "All findings must be reviewed and validated by a qualified healthcare professional. "
        "Do not make clinical decisions based solely on this analysis.",
        styles["Disclaimer"],
    ))
    elements.append(Paragraph(
        f"Generated by PulmoSight v{prediction.model_version} on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        styles["Disclaimer"],
    ))

    doc.build(elements)
    return buf.getvalue()
