import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class HistoryFilters(BaseModel):
    label: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search: Optional[str] = None


class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    per_page: int
    total_pages: int
