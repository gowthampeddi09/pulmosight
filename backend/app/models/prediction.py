import enum
from sqlalchemy import String, Float, Text, Enum, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class PredictionLabel(str, enum.Enum):
    PNEUMONIA = "PNEUMONIA"
    NORMAL = "NORMAL"


class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    prediction: Mapped[PredictionLabel] = mapped_column(
        Enum(PredictionLabel, name="prediction_label"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    heatmap_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    gradcam_observation: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Optional patient context for LLM — NOT real PHI
    patient_age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    patient_gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    patient_symptoms: Mapped[str | None] = mapped_column(Text, nullable=True)

    user = relationship("User", backref="predictions", lazy="selectin")
