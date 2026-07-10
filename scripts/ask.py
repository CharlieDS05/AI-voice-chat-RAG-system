"""End-to-end RAG: retrieve -> rerank -> generate a cited answer.

Usage: python -m scripts.ask "what makes a pie chart hard to read?"
"""

import sys

from app.generation.answer import generate_answer
from app.retrieval.bm25 import build_bm25_index
from app.retrieval.pipeline import retrieve


def main() -> None:
    if len(sys.argv) != 2:  # Format validation
        print('Usage: python -m scripts.ask "your question"')
        sys.exit(1)
    question = sys.argv[1]

    print("Retrieving...")  # Retrieve candidate chunks from the hybrid retriever
    bm25 = build_bm25_index()
    results = retrieve(question, bm25)
    docs = [doc for doc, _ in results]

    print(
        f"Top evidence (reranker scores): "  # Evidence including reranked scores
        f"{[f'{score:+.1f}' for _, score in results]}"
    )
    print("Generating...\n")
    # Answer generation: format context, render prompt, query LLM
    result = generate_answer(question, docs)

    print("=" * 70)
    print(result.answer)
    print("=" * 70)
    print("\nEvidence shown to the model:")  # Evidence including reranked scores
    for src in result.sources:
        print(f"  - {src['source']}, p.{src['page']}  ({src['chunk_id']})")


if __name__ == "__main__":
    main()
