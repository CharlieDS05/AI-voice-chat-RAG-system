"""Compare dense, BM25, and hybrid retrieval side by side.

Usage: python -m scripts.compare_retrieval "why is too much information on a graph bad?"
"""

import sys

from app.retrieval.bm25 import build_bm25_index
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vectorstore import semantic_search


def show(title: str, results: list, score_fmt: str = "{:.3f}") -> None:
    print(f"\n=== {title} ===")
    for doc, score in results:
        meta = doc.metadata
        snippet = doc.page_content[:110].replace("\n", " ")
        print(f"[{score_fmt.format(score)}] p.{meta['page']:>3} ({meta['chunk_id']})  {snippet!r}")


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python -m scripts.compare_retrieval "your question"')
        sys.exit(1)
    query = sys.argv[1]

    print("Building BM25 index from Chroma corpus...")
    bm25 = build_bm25_index()
    print(f"BM25 corpus: {len(bm25.chunks)} chunks")

    show("DENSE (semantic) top 5", semantic_search(query, k=5))
    show("BM25 (lexical) top 5", bm25.search(query, k=5), score_fmt="{:.2f}")
    show("HYBRID (RRF-fused) top 5", hybrid_search(query, bm25)[:5], score_fmt="{:.4f}")


if __name__ == "__main__":
    main()
