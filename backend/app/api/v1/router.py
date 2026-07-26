import uuid
import time
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Response
from fastapi.responses import FileResponse, Response
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.config import get_settings
from app.schemas.health import HealthResponse
from app.schemas.common import make_error, ErrorResponse
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, UserResponse
from app.schemas.prediction import PredictionResponse, PredictionDetailResponse, ModelInfoResponse
from app.schemas.history import PaginatedResponse
from app.schemas.report import GenerateReportRequest, ReportResponse
from app.models.user import User
from app.models.prediction import Prediction
from app.auth.dependencies import get_current_user
from app.auth.security import decode_token
from app.services.auth_service import create_user, authenticate_user, get_user_by_email, create_tokens
from app.services.prediction_service import run_prediction
from app.services.history_service import (
    get_predictions_paginated,
    get_prediction_by_id,
    delete_prediction,
)
from app.services.report_service import generate_clinical_report
from app.services.pdf_service import generate_pdf
from app.utils.image_validation import validate_upload
from app.inference.model_loader import get_model_metadata, is_model_loaded

log = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# ---------------------------------------------------------------------------
# Runtime metrics collector — lightweight in-memory counters for /metrics
# ---------------------------------------------------------------------------
_metrics = {
    "request_count": 0,
    "total_latency_ms": 0.0,
    "prediction_count": 0,
    "report_count": 0,
}


# ---------------------------------------------------------------------------
# System & Health Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Check database connectivity and model initialization state."""
    db_status = "connected"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        log.error("Health check database failure: %s", e)
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        database=db_status,
        model_loaded=is_model_loaded(),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Retrieve metadata about the loaded PyTorch model."""
    metadata = get_model_metadata()
    return ModelInfoResponse(
        architecture="EfficientNet-B0",
        version=metadata.get("version", settings.model_version),
        training_date=metadata.get("training_date"),
        input_size="224x224",
        num_classes=2,
        labels=["NORMAL", "PNEUMONIA"],
        model_loaded=is_model_loaded(),
        metrics=metadata.get("metrics"),
    )


@router.get("/metrics")
async def get_runtime_metrics():
    """Basic runtime metrics: request count, average latency, prediction/report counts."""
    avg_latency = (
        round(_metrics["total_latency_ms"] / _metrics["request_count"], 2)
        if _metrics["request_count"] > 0
        else 0.0
    )
    return {
        "request_count": _metrics["request_count"],
        "average_latency_ms": avg_latency,
        "prediction_count": _metrics["prediction_count"],
        "report_count": _metrics["report_count"],
    }


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user account and return an access/refresh token pair."""
    existing = await get_user_by_email(db, req.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=make_error("EMAIL_EXISTS", "A user with this email address already exists"),
        )
    user = await create_user(db, req.email, req.password, req.full_name)
    return create_tokens(user.id)


@router.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate credentials and return an access/refresh token pair."""
    user = await authenticate_user(db, req.email, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=make_error("INVALID_CREDENTIALS", "Invalid email or password"),
        )
    return create_tokens(user.id)


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest):
    """Exchange a valid refresh token for a new access/refresh token pair."""
    payload = decode_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=make_error("INVALID_TOKEN", "Invalid or expired refresh token"),
        )
    user_id = uuid.UUID(payload["sub"])
    return create_tokens(user_id)


@router.get("/auth/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    return user

# ---------------------------------------------------------------------------
# Prediction & Inference Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a Chest X-ray image for pneumonia prediction and Grad-CAM generation."""
    start = time.perf_counter()
    image_bytes = await validate_upload(file)
    prediction = await run_prediction(
        db=db,
        user_id=user.id,
        image_bytes=image_bytes,
        original_filename=file.filename or "xray.png",
    )

    elapsed = (time.perf_counter() - start) * 1000
    _metrics["request_count"] += 1
    _metrics["total_latency_ms"] += elapsed
    _metrics["prediction_count"] += 1

    heatmap_url = f"/api/v1/prediction/{prediction.id}/heatmap" if prediction.heatmap_path else None

    return PredictionResponse(
        id=prediction.id,
        filename=prediction.filename,
        prediction=prediction.prediction.value if hasattr(prediction.prediction, 'value') else str(prediction.prediction),
        confidence=prediction.confidence,
        model_version=prediction.model_version,
        processing_time_ms=prediction.processing_time_ms,
        heatmap_url=heatmap_url,
        gradcam_observation=prediction.gradcam_observation,
        created_at=prediction.created_at,
    )


@router.post("/generate-report", response_model=ReportResponse)
async def generate_report_endpoint(
    req: GenerateReportRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Generate a structured LLM clinical report for a completed prediction."""
    start = time.perf_counter()
    try:
        report_text, provider = await generate_clinical_report(
            db=db,
            prediction_id=req.prediction_id,
            user_id=user.id,
            patient_age=req.patient_age,
            patient_gender=req.patient_gender,
            patient_symptoms=req.patient_symptoms,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=make_error("NOT_FOUND", str(e)),
        )

    elapsed = (time.perf_counter() - start) * 1000
    _metrics["request_count"] += 1
    _metrics["total_latency_ms"] += elapsed
    _metrics["report_count"] += 1

    return ReportResponse(
        prediction_id=req.prediction_id,
        report_text=report_text,
        generated_by=provider,
    )

# ---------------------------------------------------------------------------
# History & Detail Endpoints
# ---------------------------------------------------------------------------

@router.get("/history", response_model=PaginatedResponse)
async def history_endpoint(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    label: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Retrieve paginated prediction history with filtering and sorting."""
    items, total = await get_predictions_paginated(
        db=db,
        user_id=user.id,
        page=page,
        per_page=per_page,
        label=label,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    formatted_items = [
        PredictionResponse(
            id=item.id,
            filename=item.filename,
            prediction=item.prediction.value if hasattr(item.prediction, 'value') else str(item.prediction),
            confidence=item.confidence,
            model_version=item.model_version,
            processing_time_ms=item.processing_time_ms,
            heatmap_url=f"/api/v1/prediction/{item.id}/heatmap" if item.heatmap_path else None,
            gradcam_observation=item.gradcam_observation,
            created_at=item.created_at,
        )
        for item in items
    ]

    total_pages = (total + per_page - 1) // per_page if total > 0 else 0

    return PaginatedResponse(
        items=formatted_items,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )


@router.get("/prediction/{id}", response_model=PredictionDetailResponse)
async def get_prediction_detail(
    id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Retrieve full details of a specific prediction record."""
    try:
        prediction_id = uuid.UUID(id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error("INVALID_ID", "Invalid UUID format"),
        )

    item = await get_prediction_by_id(db, prediction_id, user.id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=make_error("NOT_FOUND", "Prediction record not found"),
        )

    return PredictionDetailResponse(
        id=item.id,
        filename=item.filename,
        prediction=item.prediction.value if hasattr(item.prediction, 'value') else str(item.prediction),
        confidence=item.confidence,
        model_version=item.model_version,
        processing_time_ms=item.processing_time_ms,
        heatmap_url=f"/api/v1/prediction/{item.id}/heatmap" if item.heatmap_path else None,
        original_image_url=f"/api/v1/prediction/{item.id}/image",
        gradcam_observation=item.gradcam_observation,
        report_text=item.report_text,
        patient_age=item.patient_age,
        patient_gender=item.patient_gender,
        patient_symptoms=item.patient_symptoms,
        created_at=item.created_at,
    )


@router.delete("/prediction/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prediction_endpoint(
    id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a prediction record and its associated images from disk."""
    try:
        prediction_id = uuid.UUID(id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error("INVALID_ID", "Invalid UUID format"),
        )

    deleted = await delete_prediction(db, prediction_id, user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=make_error("NOT_FOUND", "Prediction record not found"),
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# ---------------------------------------------------------------------------
# File & Artifact Serving Endpoints (Publicly accessible by Prediction UUID)
# ---------------------------------------------------------------------------

@router.get("/prediction/{id}/heatmap")
async def get_heatmap_file(
    id: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve the generated Grad-CAM heatmap image directly for HTML img tags."""
    try:
        prediction_id = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=make_error("INVALID_ID", "Invalid UUID format"))

    stmt = select(Prediction).where(Prediction.id == prediction_id)
    res = await db.execute(stmt)
    item = res.scalar_one_or_none()

    if not item or not item.heatmap_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=make_error("NOT_FOUND", "Heatmap not found"))

    return FileResponse(item.heatmap_path, media_type="image/jpeg")


@router.get("/prediction/{id}/image")
async def get_original_image_file(
    id: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve the original uploaded X-ray image directly for HTML img tags."""
    try:
        prediction_id = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=make_error("INVALID_ID", "Invalid UUID format"))

    stmt = select(Prediction).where(Prediction.id == prediction_id)
    res = await db.execute(stmt)
    item = res.scalar_one_or_none()

    if not item or not item.original_image_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=make_error("NOT_FOUND", "Original image not found"))

    return FileResponse(item.original_image_path)


@router.get("/prediction/{id}/pdf")
async def download_pdf_report(
    id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Generate and stream a PDF clinical report."""
    try:
        prediction_id = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=make_error("INVALID_ID", "Invalid UUID format"))

    item = await get_prediction_by_id(db, prediction_id, user.id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=make_error("NOT_FOUND", "Prediction not found"))

    pdf_bytes = generate_pdf(item)
    filename = f"pulmosight_report_{item.id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
