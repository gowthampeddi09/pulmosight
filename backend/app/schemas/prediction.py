import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    id: uuid.UUID
    filename: str
    prediction: str
    confidence: float
    model_version: str
    processing_time_ms: float
    heatmap_url: Optional[str] = None
    gradcam_observation: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionDetailResponse(PredictionResponse):
    original_image_url: Optional[str] = None
    report_text: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    patient_symptoms: Optional[str] = None


class ModelInfoResponse(BaseModel):
    architecture: str
    version: str
    training_date: Optional[str] = None
    input_size: str = "224x224"
    num_classes: int = 2
    labels: list[str] = ["NORMAL", "PNEUMONIA"]
    model_loaded: bool
    metrics: Optional[dict] = None
