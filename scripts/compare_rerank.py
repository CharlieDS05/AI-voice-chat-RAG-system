"""Show the funnel's precision stage: hybrid pool vs. reranked top-k.

Usage: python -m scripts.compare_rerank "why is too much information on a graph bad?"
"""

import sys

from app.retrieval.bm25 import build_bm25_index
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python -m scripts.compare_rerank "your question"')
        sys.exit(1)
    query = sys.argv[1]

    bm25 = build_bm25_index()
    pool = hybrid_search(query, bm25)  # top_n candidates, RRF order

    print(f"\n=== HYBRID POOL (RRF order, {len(pool)} candidates) ===")
    for doc, score in pool:
        meta = doc.metadata
        print(
            f"[{score:.4f}] p.{meta['page']:>3} ({meta['chunk_id']})  "
            f"{doc.page_content[:100].replace(chr(10), ' ')!r}"
        )

    reranked = rerank(query, [doc for doc, _ in pool])

    print("\n=== AFTER CROSS-ENCODER RERANK (final top-k for the LLM) ===")
    for doc, score in reranked:
        meta = doc.metadata
        print(
            f"[{score:+7.2f}] p.{meta['page']:>3} ({meta['chunk_id']})  "
            f"{doc.page_content[:100].replace(chr(10), ' ')!r}"
        )


if __name__ == "__main__":
    main()
