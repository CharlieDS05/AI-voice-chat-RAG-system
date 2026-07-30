"""Document ingestion: PDF -> per-page text -> metadata-stamped chunks.

This module is the foundation of citation support:
every chunk it produces carries the source filename, page number,
and a stable chunk_id. Downstream layers (retrieval, generation,
evaluation) rely on this metadata and must never have to guess it.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def load_pdf(path: str | Path) -> list[Document]:
    """Extract a PDF into one Document per page, with page metadata.

    Raises FileNotFoundError for a missing file and ValueError for
    a PDF that yields no text (e.g. a scanned/image-only PDF).
    """
    from langchain_community.document_loaders import PyPDFLoader

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.name}")

    pages = PyPDFLoader(str(path)).load()

    if not any(page.page_content.strip() for page in pages):
        raise ValueError(
            f"'{path.name}' produced no extractable text. "
            "It may be a scanned/image-only PDF, which needs OCR (out of scope)."
        )
    return pages


def chunk_documents(pages: list[Document]) -> list[Document]:
    """Split page-Documents into overlapping chunks with citation metadata.

    Each chunk inherits its page's metadata, then gets normalized fields:
      source      -> bare filename (not the full local path)
      page        -> 1-indexed page number (human-style, for citations)
      chunk_index -> position of the chunk within the whole document
      chunk_id    -> stable unique ID, e.g. "report.pdf:p4:c12"
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= settings.min_chunk_chars]

    for index, chunk in enumerate(chunks):
        filename = Path(chunk.metadata.get("source", "unknown")).name
        page_number = int(chunk.metadata.get("page", 0)) + 1  # pypdf pages are 0-indexed

        chunk.metadata["source"] = filename
        chunk.metadata["page"] = page_number
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{filename}:p{page_number}:c{index}"

    return chunks


def ingest_pdf(path: str | Path) -> list[Document]:
    """Full pipeline: load a PDF and return citation-ready chunks."""
    return chunk_documents(load_pdf(path))
