"""Vector store layer: embeddings + ChromaDB persistence + semantic search.

Chunks arrive here already carrying citation metadata (source, page,
chunk_id) from ingestion; Chroma stores that metadata alongside each
vector, so every search result remains fully citable.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

# Module-level cache: load the embedding model once per process, not per call.
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the (lazily loaded, cached) local embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


# cosine similarity is the default for sentence-transformers


def get_vectorstore(collection: str | None = None) -> Chroma:
    """Open (or create) a persistent Chroma collection."""
    return Chroma(
        collection_name=collection or settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Document], collection: str | None = None) -> int:
    """Embed and store chunks; returns the number indexed.

    Uses chunk_id as the Chroma document ID, which makes indexing
    idempotent: re-indexing the same file overwrites rather than
    duplicates entries.
    """
    store = get_vectorstore(collection)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    store.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def semantic_search(
    query: str, k: int = 4, collection: str | None = None
) -> list[tuple[Document, float]]:
    """Return the top-k most similar chunks with their relevance scores."""
    store = get_vectorstore(collection)
    return store.similarity_search_with_relevance_scores(query, k=k)
