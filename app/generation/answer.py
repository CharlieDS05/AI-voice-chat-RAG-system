"""Grounded answer generation: retrieved chunks -> formatted context ->
versioned prompt -> LLM -> cited answer.

The context format is a contract with the prompt template: each chunk
is labeled with its citation metadata so the model can cite by copying
the label adjacent to the text it used.
"""

from dataclasses import dataclass

from langchain_core.documents import Document

from app.generation.llm import complete
from app.generation.prompts import load_prompt


@dataclass
class GroundedAnswer:
    answer: str
    sources: list[dict]  # citation metadata of the chunks shown to the LLM


def format_context(docs: list[Document]) -> str:
    """Render chunks as labeled excerpts the model can cite."""
    blocks = []
    for doc in docs:
        meta = doc.metadata
        blocks.append(f"[source: {meta['source']}, p.{meta['page']}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(
    question: str, docs: list[Document], provider: str | None = None
) -> GroundedAnswer:
    """Build the grounded prompt and query the configured LLM."""
    prompt = load_prompt()
    rendered = prompt.render(context=format_context(docs), question=question)
    answer = complete(rendered, provider=provider)

    sources = [
        {
            "source": d.metadata["source"],
            "page": d.metadata["page"],
            "chunk_id": d.metadata["chunk_id"],
        }
        for d in docs
    ]
    return GroundedAnswer(answer=answer, sources=sources)
