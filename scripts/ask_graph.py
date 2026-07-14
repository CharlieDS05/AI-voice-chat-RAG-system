"""Ask a question through the full LangGraph flow (with refusal gate).

Usage: python -m scripts.ask_graph "what makes a pie chart hard to read?"
"""

import sys

from app.graph import make_graph
from app.retrieval.bm25 import build_bm25_index


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python -m scripts.ask_graph "your question"')
        sys.exit(1)

    graph = make_graph(build_bm25_index())
    state = graph.invoke({"question": sys.argv[1]})

    print(f"\nTop evidence score : {state['top_score']:+.2f} (threshold: refuse below 0.0)")
    print(
        f"Route taken        : {
            'REFUSED at gate — LLM never called' if state['refused'] else 'generated'
        }"
    )
    print("=" * 70)
    print(state["answer"])
    print("=" * 70)
    if state["sources"]:
        print("Sources:")
        for src in state["sources"]:
            print(f"  - {src['source']}, p.{src['page']}  ({src['chunk_id']})")


if __name__ == "__main__":
    main()
