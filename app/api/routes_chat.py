from __future__ import annotations

from fastapi import APIRouter
from app.schemas import ChatRequest, ChatResponse
from rag.answer_generator import answer_question


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = answer_question(request.question)
    return ChatResponse(**result)
