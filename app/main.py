from __future__ import annotations

from fastapi import FastAPI
from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.api.routes_upload import router as upload_router


app = FastAPI(title="FFXIV Multimodal RAG Assistant", version="0.1.0")
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
