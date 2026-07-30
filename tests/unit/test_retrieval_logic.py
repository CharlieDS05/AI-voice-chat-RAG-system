"""The retrieval-logic contract: RRF rewards agreement, tokenization
normalizes case and punctuation."""

from langchain_core.documents import Document

from app.retrieval.bm25 import tokenize
from app.retrieval.hybrid import reciprocal_rank_fusion


def doc(cid: str) -> Document:
    return Document(page_content=f"text {cid}", metadata={"chunk_id": cid})


def test_tokenize_normalizes():
    assert tokenize("Clutter, clutter! CLUTTER?") == ["clutter", "clutter", "clutter"]


def test_rrf_agreement_beats_single_top_rank():
    """A chunk both retrievers rank mid-list must outrank a chunk only
    one retriever ranked #1. The mathematical heart of RRF."""
    agreed, solo = doc("agreed"), doc("solo")
    fillers = [doc(f"f{i}") for i in range(4)]
    dense = [solo, fillers[0], agreed, fillers[1]]  # agreed at rank 3
    lexical = [fillers[2], agreed, fillers[3]]  # agreed at rank 2
    ranked = reciprocal_rank_fusion([dense, lexical])
    assert ranked[0][0].metadata["chunk_id"] == "agreed"


def test_rrf_deduplicates_by_chunk_id():
    d = doc("same")
    ranked = reciprocal_rank_fusion([[d], [d]])
    assert len(ranked) == 1
