from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.domain.models import ScreenshotSummary
from src.preprocessing.image_io import load_image
from src.preprocessing.palette import extract_palette

router = APIRouter(prefix="/screenshots", tags=["screenshots"])


@router.post("/inspect", response_model=ScreenshotSummary)
async def inspect_screenshot(file: Annotated[UploadFile, File()]) -> ScreenshotSummary:
    data = await file.read()
    try:
        image = load_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ScreenshotSummary(
        filename=file.filename or "upload",
        width=image.width,
        height=image.height,
        palette=extract_palette(image),
    )
