"""Cross-encoder reranking: the precision stage of the retrieval funnel.

The hybrid retriever optimizes recall (get the right evidence into a
small pool, cheaply). This module optimizes precision: a cross-encoder
reads (query, chunk) pairs jointly and re-scores the pool, which is
far more accurate than any comparison of independently computed
vectors, and affordable because the pool is small.
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings

# Same lazy-singleton pattern as the embedder: load once per process.
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(
    query: str, docs: list[Document], top_k: int | None = None
) -> list[tuple[Document, float]]:
    """Re-score docs against the query with the cross-encoder.

    Returns the top_k documents ordered by descending relevance score.
    Scores are raw logits (roughly -11..+10): only their ORDER is
    meaningful, not their absolute scale.
    """
    if not docs:
        return []
    top_k = top_k or settings.rerank_top_k

    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)

    ranked = sorted(zip(docs, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]
