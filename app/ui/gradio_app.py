"""Gradio UI: upload PDFs, chat with them through the RAG graph.

Architecture note: this module owns the LONG-LIVED state a server
needs, the compiled graph and the BM25 index. Both are rebuilt
whenever a new document is indexed, because BM25 is in-memory
statistics over the corpus and goes stale the moment Chroma grows.
"""

import gradio as gr

from app.config import settings
from app.graph import make_graph
from app.ingestion.loader import ingest_pdf
from app.retrieval.bm25 import build_bm25_index
from app.retrieval.vectorstore import index_chunks


class RagApp:
    """Holds the compiled graph; rebuilds it when the corpus changes."""

    def __init__(self) -> None:
        self.graph = None
        try:
            self.rebuild()  # pick up any previously indexed corpus
        except ValueError:
            pass  # empty Chroma on first launch: fine, wait for an upload

    def rebuild(self) -> None:
        self.graph = make_graph(build_bm25_index())

    # Event handlers

    def upload(self, file_path: str | None) -> str:
        if not file_path:
            return "No file received."
        try:
            chunks = ingest_pdf(file_path)
            count = index_chunks(chunks)
            self.rebuild()
            return f"✅ Indexed {count} chunks. Ready to chat."
        except (FileNotFoundError, ValueError) as exc:
            return f"⚠️ {exc}"

    def chat(self, message: str, history: list[dict]) -> str:
        if self.graph is None:
            return "Please upload a PDF first — I have no documents indexed yet."

        state = self.graph.invoke({"question": message})

        if state["refused"]:
            return state["answer"]

        sources = {f"{s['source']}, p.{s['page']}" for s in state["sources"]}
        cited = "\n".join(f"- {s}" for s in sorted(sources))
        return f"{state['answer']}\n\n**Evidence shown to the model:**\n{cited}"


def build_demo() -> gr.Blocks:
    app = RagApp()

    with gr.Blocks(title="Local RAG Chat") as demo:
        gr.Markdown(
            "# 📚 Local RAG Chat\n"
            f"Provider: **{settings.llm_provider}** · "
            "All processing stays on this machine in local mode."
        )
        with gr.Row():
            file_box = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
            status = gr.Textbox(label="Index status", interactive=False)

        file_box.upload(app.upload, inputs=file_box, outputs=status)

        gr.ChatInterface(fn=app.chat)

    return demo


def main() -> None:
    # localhost only, no public share link: privacy by default.
    build_demo().launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
