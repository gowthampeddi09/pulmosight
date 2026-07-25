"""Validate that LLM-generated reports contain all required sections."""
import logging

log = logging.getLogger(__name__)

REQUIRED_SECTIONS = [
    "Clinical Summary",
    "Possible Differential Considerations",
    "Recommended Next Steps",
    "Limitations of AI Analysis",
    "Urgency Level",
    "Disclaimer",
]


def validate_report(text: str) -> tuple[bool, list[str]]:
    """
    Check that the report contains all required section headings.
    Returns (is_valid, list_of_missing_sections).
    """
    missing = []
    text_lower = text.lower()
    for section in REQUIRED_SECTIONS:
        if section.lower() not in text_lower:
            missing.append(section)

    if missing:
        log.warning("Report missing sections: %s", missing)
        return False, missing

    return True, []
