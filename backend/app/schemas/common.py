from pydantic import BaseModel
from typing import Any


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Consistent error shape across all endpoints."""
    error: ErrorDetail


def make_error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}
