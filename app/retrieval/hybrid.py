"""Hybrid retrieval: dense (Chroma) + lexical (BM25), fused with RRF.

Reciprocal Rank Fusion combines the two rankings using only ranks,
never raw scores. Cosine relevance and BM25 scores live on
incompatible scales, but ranks are always comparable.
"""

from langchain_core.documents import Document

from app.config import settings
from app.retrieval.bm25 import BM25Index


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int | None = None
) -> list[tuple[Document, float]]:
    """Fuse multiple ranked lists into one, scored by summed 1/(k + rank).

    Chunks are identified by chunk_id, so the same chunk appearing in
    several lists accumulates score -- agreement between independent
    retrievers is the strongest relevance signal.
    """
    k = k or settings.rrf_k
    fused_scores: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            chunk_id = doc.metadata["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            doc_by_id.setdefault(chunk_id, doc)

    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(doc_by_id[chunk_id], score) for chunk_id, score in ordered]


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    top_n: int | None = None,
    collection: str | None = None,
) -> list[tuple[Document, float]]:
    """Run dense + BM25 retrieval and return the RRF-fused top_n."""

    from app.retrieval.vectorstore import semantic_search

    top_n = top_n or settings.hybrid_top_n

    dense_docs = [
        doc for doc, _ in semantic_search(query, k=settings.retrieval_k, collection=collection)
    ]
    bm25_docs = [doc for doc, _ in bm25_index.search(query, k=settings.retrieval_k)]

    return reciprocal_rank_fusion([dense_docs, bm25_docs])[:top_n]
