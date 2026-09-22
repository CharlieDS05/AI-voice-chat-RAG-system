"""FastAPI application: JSON API + mounted Gradio UI.

The API is key-authed and rate-limited because it fronts a paid LLM
provider; an unprotected public endpoint is a quota and billing risk,
not just a security one.
"""

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(key_func=get_remote_address)

api = FastAPI(
    title="rag-voice-chat",
    description="Grounded question answering over PDFs, with citations and refusal.",
    version="1.1.0",
)
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_rag = None


def get_rag():
    """Lazy singleton: build the graph once, reuse across requests."""
    global _rag
    if _rag is None:
        from app.graph import make_graph
        from app.retrieval.bm25 import build_bm25_index

        _rag = make_graph(build_bm25_index())
    return _rag


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Constant-time key check. No key configured -> auth disabled."""
    if not settings.api_key:
        return
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    source: str
    page: int
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    top_score: float


@api.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "demo_mode": settings.demo_mode,
    }


@api.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a question from the indexed corpus, with citations.

    Returns refused=true when retrieved evidence scores below the
    answerability threshold — the LLM is not called in that case.
    """
    try:
        state = get_rag().invoke({"question": body.question})
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Service not ready: {exc}") from exc

    return AskResponse(
        answer=state["answer"],
        refused=state["refused"],
        sources=state["sources"],
        top_score=state["top_score"],
    )
