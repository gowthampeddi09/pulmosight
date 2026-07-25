import uuid
from pydantic import BaseModel, Field
from typing import Optional


class GenerateReportRequest(BaseModel):
    prediction_id: uuid.UUID
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    patient_gender: Optional[str] = Field(None, max_length=20)
    patient_symptoms: Optional[str] = Field(None, max_length=1000)


class ReportResponse(BaseModel):
    prediction_id: uuid.UUID
    report_text: str
    generated_by: str  # "gemini" | "groq" | "fallback"

    model_config = {"from_attributes": True}
