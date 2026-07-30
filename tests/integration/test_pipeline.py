"""Integration: ingest -> index -> hybrid retrieve -> rerank, with real
embedding and reranker models, against a temp corpus. No LLM involved."""

import pytest
from langchain_core.documents import Document

from app.ingestion.loader import chunk_documents
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import rerank

pytestmark = pytest.mark.integration

CORPUS = [
    ("The refund policy allows returns within 30 days with a receipt. " * 5, 0),
    ("Our espresso machines are serviced every 90 days by technicians. " * 5, 1),
    ("Employee training lasts six weeks and covers safety procedures. " * 5, 2),
]


def build_chunks():
    pages = [
        Document(page_content=text, metadata={"source": "/tmp/handbook.pdf", "page": p})
        for text, p in CORPUS
    ]
    return chunk_documents(pages)


def test_funnel_surfaces_correct_chunk_first():
    chunks = build_chunks()
    bm25 = BM25Index(chunks)
    # Hybrid over BM25 only here would need Chroma; test rerank precision
    # on the BM25 pool, the reranker is the precision stage under test.
    pool = [doc for doc, _ in bm25.search("how long do I have to return a purchase?", k=3)]
    ranked = rerank("how long do I have to return a purchase?", pool, top_k=1)
    assert ranked[0][0].metadata["page"] == 1  # the refund page (1-indexed)
