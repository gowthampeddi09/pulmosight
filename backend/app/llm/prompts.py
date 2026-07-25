"""Structured prompt templates for clinical report generation."""
from typing import Optional


def build_report_prompt(
    prediction: str,
    confidence: float,
    model_version: str,
    gradcam_observation: str,
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    patient_symptoms: Optional[str] = None,
) -> str:
    patient_section = ""
    if any([patient_age, patient_gender, patient_symptoms]):
        parts = []
        if patient_age is not None:
            parts.append(f"Age: {patient_age}")
        if patient_gender:
            parts.append(f"Gender: {patient_gender}")
        if patient_symptoms:
            parts.append(f"Presenting symptoms: {patient_symptoms}")
        patient_section = f"""
PATIENT CONTEXT (provided for clinical relevance, not verified):
{chr(10).join(parts)}
"""

    return f"""You are a radiological AI assistant generating a structured clinical-style report
for a chest X-ray analysis. This is a DECISION-SUPPORT tool for educational and portfolio
demonstration purposes only — it is NOT a diagnostic device.

ANALYSIS RESULTS:
- AI Prediction: {prediction}
- Confidence: {confidence:.1%}
- Model: EfficientNet-B0 (version {model_version})
- Grad-CAM Observation: {gradcam_observation}
{patient_section}
Generate a structured report with EXACTLY these sections, using these exact headings:

## Clinical Summary
Summarize the AI findings in 2-3 sentences.

## Possible Differential Considerations
List 3-5 conditions to consider given the findings.

## Recommended Next Steps
Suggest 3-4 appropriate follow-up actions.

## Limitations of AI Analysis
State 2-3 specific limitations of this analysis.

## Urgency Level
Classify as: ROUTINE / SEMI-URGENT / URGENT with a brief justification.

## Disclaimer
State clearly that this is not a medical diagnosis, requires physician review,
and should not be used as the sole basis for clinical decisions.

Be concise and clinically precise. Do not use speculative language beyond what
the confidence score supports. Do not fabricate patient history."""


FALLBACK_REPORT = """## Clinical Summary
AI analysis was performed on the submitted chest X-ray. Due to a temporary service
interruption, the detailed clinical report could not be generated at this time.
Please refer to the prediction label and confidence score for preliminary results.

## Possible Differential Considerations
- Detailed differential analysis unavailable due to service interruption
- Manual radiological review is recommended

## Recommended Next Steps
- Retry report generation after a few minutes
- Consult with a qualified radiologist for definitive interpretation
- Consider additional imaging if clinically indicated

## Limitations of AI Analysis
- This report was generated using a fallback template due to LLM service unavailability
- AI-based analysis should always be correlated with clinical findings

## Urgency Level
ROUTINE — AI prediction available; detailed analysis pending service restoration.

## Disclaimer
This is NOT a medical diagnosis. This tool is for educational and decision-support
purposes only. All findings must be reviewed by a qualified healthcare professional.
Do not make clinical decisions based solely on this AI analysis."""
