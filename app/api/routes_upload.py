from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from vlm.image_captioner import caption_image_bytes


router = APIRouter(tags=["upload"])


@router.post("/upload-screenshot")
async def upload_screenshot(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    return {"filename": file.filename, "caption": caption_image_bytes(data)}
