from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool = False


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
