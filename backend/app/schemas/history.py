from pydantic import BaseModel
from typing import Optional

from app.schemas.prediction import PredictionResponse


class HistoryFilters(BaseModel):
    label: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    search: Optional[str] = None


class PaginatedResponse(BaseModel):
    items: list[PredictionResponse]
    total: int
    page: int
    per_page: int
    total_pages: int
