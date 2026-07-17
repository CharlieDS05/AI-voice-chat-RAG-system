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

# Graceful voice degradation: if the user doesn't have the optional dependencies installed,
# the voice panel will be hidden and the app will still work for text-only chat.
try:
    from app.voice.stt import transcribe
    from app.voice.tts import synthesize

    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


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

    def voice_ask(self, audio_path: str | None):
        """Mic audio in -> (transcribed question, answer text, spoken answer)."""
        if self.graph is None:
            return "", "Please upload a PDF first.", None

        question = transcribe(audio_path)
        if not question:
            return "", "I couldn't hear anything — try recording again.", None

        state = self.graph.invoke({"question": question})
        answer = state["answer"]

        if not state["refused"]:
            sources = {f"{s['source']}, p.{s['page']}" for s in state["sources"]}
            answer_display = f"{answer}\n\n**Evidence:** " + "; ".join(sorted(sources))
        else:
            answer_display = answer

        # Speak ONLY the answer text, citations are for the screen.
        audio_out = synthesize(answer)
        return question, answer_display, audio_out


# Theme

INK = gr.themes.Color(
    name="ink",
    c50="#EAF0FB",
    c100="#D3DCF0",
    c200="#9AA8C7",
    c300="#6B7A9E",
    c400="#3D4A6B",
    c500="#232D47",
    c600="#182036",
    c700="#121A2C",
    c800="#0D1424",
    c900="#0A0E1A",
    c950="#070B14",
)

CYAN = gr.themes.Color(
    name="icecyan",
    c50="#EFFDFF",
    c100="#D5F8FC",
    c200="#AEEFF6",
    c300="#8AE6F0",
    c400="#6EE7F0",
    c500="#3ECBDA",
    c600="#22A9B8",
    c700="#178491",
    c800="#12626C",
    c900="#0E4A52",
    c950="#082E33",
)

THEME = gr.themes.Base(
    primary_hue=CYAN,
    neutral_hue=INK,
    font=[gr.themes.GoogleFont("Space Grotesk"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#070B14",
    body_text_color="#E6EAF2",
    body_text_color_subdued="#8B94A7",
    background_fill_primary="#0D1424",
    background_fill_secondary="#121A2C",
    border_color_primary="#1F2A44",
    block_background_fill="#0D1424",
    block_border_color="#1F2A44",
    block_label_background_fill="#121A2C",
    block_label_text_color="#8B94A7",
    block_title_text_color="#E6EAF2",
    input_background_fill="#121A2C",
    input_border_color="#1F2A44",
    input_border_color_focus="#3ECBDA",
    button_primary_background_fill="#22A9B8",
    button_primary_background_fill_hover="#3ECBDA",
    button_primary_text_color="#070B14",
    button_secondary_background_fill="#182036",
    button_secondary_text_color="#E6EAF2",
    color_accent_soft="#10333B",
    shadow_drop="0 1px 2px rgba(0,0,0,0.4)",
    shadow_drop_lg="0 8px 30px rgba(0,0,0,0.45)",
)

CSS = """
/* ---- header ------------------------------------------------------------ */
#hdr {
    padding: 28px 8px 18px 8px;
    border-bottom: 1px solid #1F2A44;
    margin-bottom: 4px;
}
#hdr .wordmark {
    font-family: 'Space Grotesk', system-ui, sans-serif;
    font-size: 1.9rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #E6EAF2;
    margin: 0;
}
#hdr .wordmark .tick { color: #6EE7F0; }
#hdr .tagline {
    color: #8B94A7;
    font-size: 0.95rem;
    margin: 6px 0 0 0;
}
 
/* ---- status rail: real system state as monospace badges ---------------- */
#rail {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}
#rail .badge {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    color: #8B94A7;
    background: #121A2C;
    border: 1px solid #1F2A44;
    border-radius: 6px;
    padding: 5px 10px;
}
#rail .badge b {
    color: #E6EAF2;
    font-weight: 600;
}
#rail .badge.live::before {
    content: "●";
    color: #6EE7F0;
    margin-right: 7px;
    font-size: 0.6rem;
    vertical-align: 1px;
}
 
/* ---- corpus panel -------------------------------------------------------- */
#corpus-panel {
    border: 1px solid #1F2A44;
    border-radius: 12px;
    background: #0D1424;
    padding: 6px;
}
#status-box textarea {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.82rem !important;
}

/* Uploaded-file preview: the filename is a download link, and Gradio derives
   link + row-highlight colors from pale primary tints. Scoped to #file-box
   only, so nothing else is affected. */
#file-box .file-preview,
#file-box .file-preview td,
#file-box .file-preview span,
#file-box .file-preview p {
    color: #E6EAF2 !important;
    background: transparent !important;
}
#file-box .file-preview a,
#file-box a {
    color: #8AE6F0 !important;          /* filename link: readable ice cyan */
    text-decoration: none;
}
#file-box .file-preview a:hover,
#file-box a:hover {
    text-decoration: underline;
}
#file-box .file-preview tr,
#file-box .file-preview tbody tr {
    background: #121A2C !important;      /* dark row instead of pale highlight */
    border-color: #1F2A44 !important;
}
#file-box .wrap, #file-box .or {
    color: #8B94A7 !important;           /* "drop file here / or" helper text */
}
 
/* ---- chat ---------------------------------------------------------------- */
#chat-panel { margin-top: 10px; }
.message-wrap .bot, .message-wrap .user { border-radius: 12px !important; }
 
/* Contrast guarantee: user bubbles get a dark teal surface with light text,
   bot bubbles a neutral dark surface, regardless of theme-derived tints. */
.message.user, .message-wrap .user, div[class*="user"] .message {
    background: #10333B !important;
    color: #E6EAF2 !important;
    border: 1px solid #178491 !important;
}
.message.bot, .message-wrap .bot {
    background: #121A2C !important;
    color: #E6EAF2 !important;
    border: 1px solid #1F2A44 !important;
}
.message.user *, .message.bot * { color: #E6EAF2; }
 
/* focus visibility + reduced motion, quietly ------------------------------ */
:focus-visible { outline: 2px solid #3ECBDA !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
 
footer { display: none !important; }

#voice-panel {
    border: 1px solid #1F2A44 !important;
    border-radius: 12px;
    background: #0D1424;
    margin-top: 10px;
}
#heard-box textarea {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.82rem !important;
}
"""


def build_demo() -> gr.Blocks:
    app = RagApp()

    corpus_state = "corpus loaded" if app.graph is not None else "awaiting first upload"

    header_html = f"""
    <div id="hdr">
      <p class="wordmark">Local RAG<span class="tick">_</span></p>
      <p class="tagline">Ask questions of your own documents. Every answer cites its evidence.</p>
      <div id="rail">
        <span class="badge live">LOCAL &nbsp;<b>all processing stays on this machine</b></span>
        <span class="badge">PROVIDER &nbsp;<b>{settings.llm_provider}</b></span>
        <span class="badge">INDEX &nbsp;<b>{corpus_state}</b></span>
      </div>
    </div>
    """

    with gr.Blocks(title="Local RAG Chat") as demo:
        gr.HTML(header_html)

        with gr.Group(elem_id="corpus-panel"):
            with gr.Row():
                file_box = gr.File(
                    label="Upload a PDF",
                    file_types=[".pdf"],
                    type="filepath",
                    elem_id="file-box",
                )

                status = gr.Textbox(
                    label="Index status",
                    interactive=False,
                    elem_id="status-box",
                    placeholder="Nothing indexed this session yet. Upload a PDF to start.",
                )

        file_box.upload(app.upload, inputs=file_box, outputs=status)

        with gr.Column(elem_id="chat-panel"):
            gr.ChatInterface(
                fn=app.chat,
                chatbot=gr.Chatbot(
                    height=520,
                    show_label=False,
                    placeholder=(
                        "<div style='color:#8B94A7;font-size:0.95rem;'>"
                        "Upload a PDF above, then ask it anything."
                        "</div>"
                    ),
                ),
                textbox=gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    show_label=False,
                ),
            )

        if VOICE_AVAILABLE:
            with gr.Accordion("🎙️ Voice mode", open=True, elem_id="voice-panel"):
                with gr.Row():
                    mic = gr.Audio(sources=["microphone"], type="filepath", label="Ask by voice")
                    heard = gr.Textbox(label="What I heard", interactive=False, elem_id="heard-box")
                voice_answer = gr.Markdown(label="Answer")
                speaker = gr.Audio(label="Spoken answer", autoplay=True)

                mic.stop_recording(
                    app.voice_ask, inputs=mic, outputs=[heard, voice_answer, speaker]
                )
        else:
            gr.Markdown(
                "*🎙️ Voice mode requires a native Apple Silicon run "
                "(MLX Whisper). Running in text-only mode.*"
            )

    return demo


def main() -> None:
    # localhost only, no public share link: privacy by default.
    build_demo().launch(
        server_name=settings.server_name,
        share=False,
        theme=THEME,
        css=CSS,
        show_error=True,
    )


if __name__ == "__main__":
    main()
