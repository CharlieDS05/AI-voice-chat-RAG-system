"""The complete retrieval pipeline: hybrid recall -> cross-encoder precision.

This is the one function downstream layers should call. It hides the
funnel's internals and returns citation-ready, precision-ranked chunks.
"""

from langchain_core.documents import Document

from app.config import settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank


def retrieve(
    query: str,
    bm25_index: BM25Index,
    top_k: int | None = None,
    collection: str | None = None,
) -> list[tuple[Document, float]]:
    """Hybrid-retrieve a candidate pool, then rerank it. Returns
    (chunk, cross-encoder score) pairs, best first."""
    candidates = [
        doc
        for doc, _ in hybrid_search(
            query, bm25_index, top_n=settings.hybrid_top_n, collection=collection
        )
    ]
    return rerank(query, candidates, top_k=top_k)
