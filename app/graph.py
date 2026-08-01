"""The RAG flow as an explicit LangGraph state machine.

    retrieve ──► gate ──► generate ──► END
                   │
                   └────► refuse  ──► END

The gate makes refusal STRUCTURAL: if the best
reranked evidence scores below the threshold, the LLM is never
invoked, so no tokens spent, no document content placed before a
model for a question the corpus cannot answer.
"""

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from app.config import settings
from app.generation.answer import generate_answer
from app.retrieval.bm25 import BM25Index
from app.retrieval.pipeline import retrieve


class RAGState(TypedDict, total=False):
    question: str
    docs: list[Document]
    scores: list[float]
    top_score: float
    refused: bool
    answer: str
    sources: list[dict]
    provider: str


def make_graph(bm25_index: BM25Index):
    """Build and compile the RAG graph around a ready BM25 index."""

    def retrieve_node(state: RAGState) -> RAGState:
        results = retrieve(state["question"], bm25_index)
        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]
        return {
            "docs": docs,
            "scores": scores,
            "top_score": max(scores) if scores else float("-inf"),
        }

    def gate(state: RAGState) -> str:
        if state["top_score"] < settings.refusal_threshold:
            return "refuse"
        return "generate"

    def generate_node(state: RAGState) -> RAGState:
        result = generate_answer(state["question"], state["docs"], provider=state.get("provider"))
        return {"answer": result.answer, "sources": result.sources, "refused": False}

    def refuse_node(state: RAGState) -> RAGState:
        return {"answer": settings.refusal_message, "sources": [], "refused": True}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", gate, {"generate": "generate", "refuse": "refuse"})
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph.compile()
