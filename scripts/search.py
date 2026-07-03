"""Run a semantic search against the indexed documents.

Usage: python -m scripts.search "how do I choose the right chart type?"
"""

import sys

from app.retrieval.vectorstore import semantic_search


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python -m scripts.search "your question here"')
        sys.exit(1)

    results = semantic_search(sys.argv[1], k=4)

    print(f'🔎 Top {len(results)} results for: "{sys.argv[1]}"\n')
    for doc, score in results:
        meta = doc.metadata
        print(f"[{score:.3f}] {meta['source']} — page {meta['page']} ({meta['chunk_id']})")
        print(f"    {doc.page_content[:180]!r}\n")


if __name__ == "__main__":
    main()
