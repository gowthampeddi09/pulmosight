"""
Hospital-grade PDF report generation using ReportLab.
Produces a rich, executive diagnostic report suitable for leading clinical networks.
"""
import io
import os
import logging
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from app.models.prediction import Prediction

log = logging.getLogger(__name__)

PRIMARY_NAVY = HexColor("#0F172A")
ACCENT_SKY = HexColor("#0284C7")
BG_SLATE = HexColor("#F8FAFC")
BORDER_COLOR = HexColor("#E2E8F0")
ALERT_RED = HexColor("#DC2626")
SUCCESS_GREEN = HexColor("#16A34A")


def generate_pdf(prediction: Prediction) -> bytes:
    """Generate an executive hospital-style PDF report. Returns raw PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "DocTitle", parent=styles["Heading1"],
        fontSize=18, textColor=PRIMARY_NAVY, fontName="Helvetica-Bold",
        spaceAfter=1 * mm,
    ))
    styles.add(ParagraphStyle(
        "DocSubTitle", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#64748B"), fontName="Helvetica-Bold",
        spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=11, textColor=ACCENT_SKY, fontName="Helvetica-Bold",
        spaceBefore=4 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        "ReportBody", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=HexColor("#334155"),
        spaceAfter=1.5 * mm,
    ))
    styles.add(ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=7.5, leading=10, textColor=HexColor("#64748B"),
        spaceAfter=1 * mm,
    ))

    elements = []

    # Hospital Header Banner Table
    header_data = [
        [
            Paragraph("<b>PULMOSIGHT MEDICAL INTELLIGENCE</b><br/><font size=8 color='#64748B'>Advanced Diagnostic Chest Radiography Platform</font>", styles["ReportBody"]),
            Paragraph(f"<b>REPORT REF:</b> {str(prediction.id)[:8].upper()}<br/><font size=8 color='#64748B'>Date: {prediction.created_at.strftime('%d %b %Y, %H:%M UTC')}</font>", ParagraphStyle("RAlign", parent=styles["ReportBody"], alignment=TA_RIGHT)),
        ]
    ]
    header_table = Table(header_data, colWidths=[110 * mm, 70 * mm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_SKY, spaceAfter=4 * mm))

    # Patient & Scan Metadata Grid
    meta_rows = [["PATIENT DEMOGRAPHICS & SCAN CONTEXT", ""]]
    if prediction.patient_age:
        meta_rows.append(["Patient Age", f"{prediction.patient_age} Years"])
    if prediction.patient_gender:
        meta_rows.append(["Gender", prediction.patient_gender])
    if prediction.patient_symptoms:
        meta_rows.append(["Presenting Symptoms", prediction.patient_symptoms])
    meta_rows.append(["Modality / View", "Chest Radiograph (AP/PA View)"])
    meta_rows.append(["AI Vision Engine", f"EfficientNet-B0 Medical Vision (v{prediction.model_version})"])

    meta_table = Table(meta_rows, colWidths=[55 * mm, 125 * mm])
    meta_table.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("BACKGROUND", (0, 0), (1, 0), PRIMARY_NAVY),
        ("TEXTCOLOR", (0, 0), (1, 0), HexColor("#FFFFFF")),
        ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (1, 0), 9),
        ("BACKGROUND", (0, 1), (-1, -1), BG_SLATE),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 4 * mm))

    # Diagnostic Verdict Banner
    label = prediction.prediction.value if hasattr(prediction.prediction, 'value') else str(prediction.prediction)
    conf_pct = f"{prediction.confidence * 100:.1f}%"
    v_color = ALERT_RED if label == "PNEUMONIA" else SUCCESS_GREEN

    verdict_data = [
        [
            Paragraph("AUTOMATED VERDICT", ParagraphStyle("VHead", parent=styles["Normal"], fontSize=8, fontName="Helvetica-Bold", textColor=HexColor("#64748B"))),
            Paragraph("NEURAL CONFIDENCE", ParagraphStyle("VHead", parent=styles["Normal"], fontSize=8, fontName="Helvetica-Bold", textColor=HexColor("#64748B"))),
            Paragraph("INFERENCE TIME", ParagraphStyle("VHead", parent=styles["Normal"], fontSize=8, fontName="Helvetica-Bold", textColor=HexColor("#64748B"))),
        ],
        [
            Paragraph(f"<font color='{v_color.hexval()}'><b>{label}</b></font>", ParagraphStyle("VVal", parent=styles["Normal"], fontSize=14, fontName="Helvetica-Bold")),
            Paragraph(f"<b>{conf_pct}</b>", ParagraphStyle("VVal", parent=styles["Normal"], fontSize=14, fontName="Helvetica-Bold")),
            Paragraph(f"<b>{prediction.processing_time_ms:.0f} ms</b>", ParagraphStyle("VVal", parent=styles["Normal"], fontSize=14, fontName="Helvetica-Bold")),
        ]
    ]
    verdict_table = Table(verdict_data, colWidths=[60 * mm, 60 * mm, 60 * mm])
    verdict_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, -1), BG_SLATE),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 4 * mm))

    # Grad-CAM Heatmap Image
    if prediction.heatmap_path and os.path.exists(prediction.heatmap_path):
        elements.append(Paragraph("GRAD-CAM NEURAL ACTIVATION MAP", styles["SectionHead"]))
        try:
            img = RLImage(prediction.heatmap_path, width=100 * mm, height=100 * mm, kind="proportional")
            elements.append(img)
            if prediction.gradcam_observation:
                elements.append(Paragraph(f"<b>Spatial Observation:</b> {prediction.gradcam_observation}", styles["ReportBody"]))
            elements.append(Spacer(1, 3 * mm))
        except Exception as e:
            log.warning("Failed to embed heatmap in PDF: %s", e)

    # LLM Structured Clinical Report
    if prediction.report_text:
        elements.append(Paragraph("EXECUTIVE CLINICAL DIAGNOSTIC REPORT", styles["SectionHead"]))
        for line in prediction.report_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("## "):
                elements.append(Paragraph(line[3:].upper(), styles["SectionHead"]))
            elif line.startswith("• ") or line.startswith("- "):
                elements.append(Paragraph(f"• {line[2:]}", styles["ReportBody"]))
            else:
                elements.append(Paragraph(line, styles["ReportBody"]))

    # Footer Sign-off & Disclaimer
    elements.append(Spacer(1, 6 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceAfter=3 * mm))

    footer_table = Table([
        [
            Paragraph("<b>PULMOSIGHT AI DECISION SUPPORT PLATFORM</b><br/><font size=7 color='#64748B'>Verified Digital Record · Educational & Clinical Research Demonstration</font>", styles["ReportBody"]),
            Paragraph("____________________________<br/><b>Reviewing Radiologist Signature</b>", ParagraphStyle("Sign", parent=styles["ReportBody"], alignment=TA_RIGHT)),
        ]
    ], colWidths=[110 * mm, 70 * mm])
    footer_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    elements.append(footer_table)

    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>NOTICE:</b> This report is generated by an artificial intelligence decision-support platform. "
        "It does not constitute a certified medical diagnosis and must be evaluated by a licensed healthcare professional "
        "prior to clinical intervention.",
        styles["Disclaimer"],
    ))

    doc.build(elements)
    return buf.getvalue()
