"""Cross-encoder reranking: the precision stage of the retrieval funnel.

The hybrid retriever optimizes recall (get the right evidence into a
small pool, cheaply). This module optimizes precision: a cross-encoder
reads (query, chunk) pairs jointly and re-scores the pool, which is
far more accurate than any comparison of independently computed
vectors, and affordable because the pool is small.
"""

from typing import TYPE_CHECKING

from langchain_core.documents import Document

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# Same lazy-singleton pattern as the embedder: load once per process.
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # deferred

        _reranker = TextCrossEncoder(model_name=settings.reranker_model)
    return _reranker


def rerank(
    query: str, docs: list[Document], top_k: int | None = None
) -> list[tuple[Document, float]]:
    """Re-score docs against the query with the ONNX cross-encoder.

    Scores are raw logits (roughly -11..+10) — the same scale the
    refusal threshold is calibrated against.
    """
    if not docs:
        return []
    top_k = top_k or settings.rerank_top_k

    scores = list(get_reranker().rerank(query, [d.page_content for d in docs]))
    ranked = sorted(zip(docs, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]
