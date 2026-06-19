from __future__ import annotations

from fastapi import APIRouter
from src.config import settings


router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "background_backend": settings.background_backend,
        "vlm_backend": settings.vlm_backend,
        "vlm_model_id": settings.vlm_model_id,
    }
