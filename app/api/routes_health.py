from __future__ import annotations

from fastapi import APIRouter
from app.config import settings
from app.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", index_ready=settings.vector_db_path.exists())
