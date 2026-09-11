"""Embedding backend: fastembed (ONNX) behind LangChain's Embeddings interface.

Same all-MiniLM-L6-v2 weights as sentence-transformers, executed by
onnxruntime instead of torch. Verified numerically identical
(cosine similarity 1.000000), so existing Chroma indexes stay valid.
"""

from langchain_core.embeddings import Embeddings

from app.config import settings

_model = None


def _get_model():
    """Lazy singleton: load the ONNX model once per process."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # deferred: pulls onnxruntime

        _model = TextEmbedding(model_name=settings.embedding_model)
    return _model


class FastEmbedEmbeddings(Embeddings):
    """Minimal LangChain-compatible wrapper over fastembed."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in _get_model().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(_get_model().embed([text]))).tolist()