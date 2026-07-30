"""BM25 lexical retrieval over the chunks stored in Chroma.

The BM25 index is in-memory statistics, rebuilt from the Chroma
collection at startup. Chroma remains the single source of truth
for the corpus; this module never ingests documents itself.
"""

import re  # Library for regular expressions

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")  # Rule for compiling tokens: lowercase alphanumeric sequences


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens.

    Deliberately simple: lowercasing means 'Clutter' matches 'clutter',
    and the regex strips punctuation so 'clutter,' matches 'clutter'.
    No stemming/lemmatization -- BM25's IDF weighting does most of the
    heavy lifting without them at our corpus size.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over a list of chunks."""

    def __init__(self, chunks: list[Document]):  # Reads the chunks and builds the BM25 index
        if not chunks:
            raise ValueError("Cannot build a BM25 index over an empty corpus.")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    # searches the BM25 index for the top-k chunks matching the query
    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Return top-k chunks by BM25 score (higher = better)."""
        k = k or settings.retrieval_k
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], float(scores[i])) for i in ranked[:k]]


# loads the corpus from the Chroma collection, returning a list of Document objects
def load_corpus_from_chroma(collection: str | None = None) -> list[Document]:
    """Read every chunk (text + metadata) back out of the Chroma collection."""
    from app.retrieval.vectorstore import get_vectorstore

    store = get_vectorstore(collection)
    payload = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(payload["documents"], payload["metadatas"], strict=True)
    ]


# Builds a BM25 index over the chunks in the Chroma collection, returning a BM25Index object
def build_bm25_index(collection: str | None = None) -> BM25Index:
    """Convenience: BM25 index over the current Chroma contents."""
    return BM25Index(load_corpus_from_chroma(collection))
