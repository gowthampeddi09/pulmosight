import uuid
import logging
from typing import Optional
from datetime import datetime

from sqlalchemy import select, func, desc, asc, delete, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prediction import Prediction
from app.utils.file_handling import delete_prediction_files

log = logging.getLogger(__name__)


async def get_predictions_paginated(
    db: AsyncSession,
    user_id: uuid.UUID,
    page: int = 1,
    per_page: int = 10,
    label: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> tuple[list[Prediction], int]:
    """
    Fetch paginated prediction history with optional filters.
    Returns (items, total_count).
    """
    query = select(Prediction).where(Prediction.user_id == user_id)
    count_query = select(func.count(Prediction.id)).where(Prediction.user_id == user_id)

    # Apply filters
    if label:
        query = query.where(Prediction.prediction == label.upper())
        count_query = count_query.where(Prediction.prediction == label.upper())

    if date_from:
        query = query.where(Prediction.created_at >= date_from)
        count_query = count_query.where(Prediction.created_at >= date_from)

    if date_to:
        query = query.where(Prediction.created_at <= date_to)
        count_query = count_query.where(Prediction.created_at <= date_to)

    if search:
        search_filter = or_(
            Prediction.filename.ilike(f"%{search}%"),
            Prediction.prediction.cast(str).ilike(f"%{search}%"),
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)

    # Total count for pagination
    total = (await db.execute(count_query)).scalar() or 0

    # Sorting
    sort_column = getattr(Prediction, sort_by, Prediction.created_at)
    order_fn = desc if sort_order == "desc" else asc
    query = query.order_by(order_fn(sort_column))

    # Pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    result = await db.execute(query)
    items = list(result.scalars().all())

    return items, total


async def get_prediction_by_id(
    db: AsyncSession, prediction_id: uuid.UUID, user_id: uuid.UUID
) -> Optional[Prediction]:
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_prediction(
    db: AsyncSession, prediction_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    """Delete a prediction and its associated files. Returns True if found and deleted."""
    prediction = await get_prediction_by_id(db, prediction_id, user_id)
    if prediction is None:
        return False

    delete_prediction_files(prediction.original_image_path, prediction.heatmap_path)
    await db.delete(prediction)
    await db.flush()

    log.info("Deleted prediction %s", prediction_id)
    return True
