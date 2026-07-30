"""The ingestion contract: chunks carry citation metadata, pages are
1-indexed, noise is filtered, and IDs are stable in format."""

from langchain_core.documents import Document

from app.ingestion.loader import chunk_documents


def make_page(text: str, page0: int) -> Document:
    """Simulate a PyPDFLoader page (0-indexed, full path as source)."""
    return Document(
        page_content=text,
        metadata={"source": "/tmp/somewhere/report.pdf", "page": page0},
    )


def test_chunks_carry_citation_metadata():
    chunks = chunk_documents([make_page("A meaningful paragraph. " * 20, page0=0)])
    meta = chunks[0].metadata
    assert meta["source"] == "report.pdf"  # bare filename, no local path
    assert meta["page"] == 1  # 1-indexed for humans
    assert meta["chunk_id"] == "report.pdf:p1:c0"  # stable schema


def test_noise_chunks_are_filtered():
    pages = [
        make_page("Ch. 3", page0=0),  # 5 chars: noise
        make_page("Real content sentence here. " * 10, 1),
    ]  # real
    chunks = chunk_documents(pages)
    assert all(len(c.page_content.strip()) >= 50 for c in chunks)


def test_chunk_ids_are_contiguous_after_filtering():
    pages = [make_page("x", 0), make_page("Substantial text. " * 15, 1)]
    chunks = chunk_documents(pages)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
