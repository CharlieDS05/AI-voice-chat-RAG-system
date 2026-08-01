# rag-voice-chat

Local-first RAG chatbot with voice: ask questions of your own PDFs by text or
speech, and get grounded answers that cite the exact page they came from.
Private by default in local mode, nothing leaves your machine.

The system is built to refuse rather than guess: if the retrieved evidence
doesn't support an answer, it says so instead of inventing one.

---

## Features

- **Hybrid retrieval** — BM25 keyword search fused with dense vector search via
  Reciprocal Rank Fusion, so both exact terms and paraphrased meaning are found
- **Cross-encoder reranking** — an SBERT reranker re-scores candidates by reading
  query and passage together, which is far more accurate than vector similarity alone
- **Citations with page numbers** — every chunk carries `source` and `page`
  metadata from ingestion through to the answer
- **Structural refusal gate** — questions the corpus can't answer are refused
  *before* the LLM is called, making anti-hallucination a property of the
  architecture rather than of prompt compliance
- **Swappable LLM provider** — local Ollama or any OpenAI-compatible hosted API,
  switchable from the UI or a single environment variable
- **Voice mode** — MLX Whisper for speech-to-text, Piper for text-to-speech,
  both running locally on Apple Silicon
- **Versioned prompts** — prompts live in `prompts/*.yaml` under version control,
  not buried in code
- **Measured quality** — Ragas faithfulness scoring against a hand-reviewed golden
  dataset, enforced by a CI gate that fails the build when quality drops

## Architecture

```
Upload → chunk (+ source/page metadata) → embed → Chroma

Question
   ├── BM25 (lexical)  ─┐
   │                     ├── RRF fusion → cross-encoder rerank → top-k
   └── Vector (dense)  ─┘
                                    │
                            answerability gate
                            ├── below threshold → refuse (LLM never called)
                            └── above threshold → generate with citations
```

The retrieval funnel is **cheap-and-broad first, expensive-and-precise second**:
hybrid search optimizes recall over the whole corpus, then the cross-encoder
optimizes precision over a small candidate pool.

The flow is a LangGraph state machine (`app/graph.py`), so the refusal branch is
an explicit edge rather than an `if` buried in application code.

**Stack:** LangGraph · ChromaDB · sentence-transformers · rank-bm25 · Ollama ·
Gradio · MLX Whisper · Piper · Ragas · Pydantic Settings · GitHub Actions

---

## Prerequisites

- **macOS Apple Silicon** for the full experience (voice), or **any Docker host**
  for text mode
- **[Ollama](https://ollama.com)** running natively (local mode):
  ```bash
  ollama pull llama3.2:3b
  ```
  Ollama must run natively rather than in Docker — Docker Desktop on macOS has no
  GPU access, which forces CPU-only inference.
- **Python 3.10+** (native mode)
- **ffmpeg** for voice mode: `brew install ffmpeg` (macOS) or
  `apt-get install ffmpeg` (Linux)

At least 8 GB of RAM and ~5 GB of free disk space.

---

## Quickstart — native (full voice)

```bash
git clone https://github.com/CharlieDS05/AI-voice-chat-RAG-system.git
cd AI-voice-chat-RAG-system

python3 -m venv .venv
source .venv/bin/activate          # run this in every new terminal
pip install -r requirements.txt

cp .env.example .env               # defaults work as-is: local and private

python -m piper.download_voices en_US-lessac-medium --data-dir models/voices

python -m scripts.run_ui
```

Open **http://127.0.0.1:7860**, upload a PDF, and ask it something.

The first run downloads the embedding and reranker models (~160 MB total) and can
take several minutes; see [first-run speed](#note-on-first-run-speed) below.

## Quickstart — Docker (text mode)

```bash
git clone https://github.com/CharlieDS05/AI-voice-chat-RAG-system.git
cd AI-voice-chat-RAG-system
docker compose up --build
```

Open **http://127.0.0.1:7860**. Ollama still runs natively on the host; the
container reaches it via `host.docker.internal`.

The first build takes several minutes, and the first question inside a fresh
container downloads the embedding and reranker models again — the container has
its own cache. Voice mode is unavailable in Docker (see
[Limitations](#limitations)).

To stop: `Ctrl+C`. To start again later: `docker compose up` (no `--build`).

---

## Verifying your setup

Work through these in order. Each step's failure tells you exactly where the
problem is.

**1. Confirm config is loading**

```bash
python -m scripts.check_config
```

Shows the effective provider, URLs, and model. If the values don't match your
`.env`, check `env | grep HOSTED` — exported shell variables override the file.

**2. Confirm the local engine (local mode only)**

```bash
curl http://localhost:11434
```

Expect `Ollama is running`. If not, launch the Ollama app.

**3. Run the unit tests**

```bash
pytest tests/unit
```

10 tests, no network, no models. All green means the core logic is intact.

**4. Index a document and ask it something**

```bash
python -m scripts.build_index path/to/your.pdf
python -m scripts.ask_graph "a question your document can answer"
```

You should get an answer with `[source: file.pdf, p.N]` citations. Then test the
refusal gate with something the document cannot answer:

```bash
python -m scripts.ask_graph "how do I configure a Kubernetes cluster?"
```

Expect `REFUSED at gate — LLM never called`.

**5. Launch the UI**

```bash
python -m scripts.run_ui
```

Upload a PDF, chat, and switch the answer engine between local and hosted to
compare them on the same question.

### Note on first-run speed

The first launch downloads the embedding (~80 MB) and reranker (~80 MB) models.
ML libraries can also take several minutes to load *cold* on memory-constrained
machines (measured: ~6s warm vs ~410s cold on an 8 GB M1). Subsequent runs are
much faster.

Once the models are cached, set `HF_OFFLINE=1` in `.env`. This skips Hugging Face
Hub network calls at import time — measured at **346s → 5.7s** on a throttled
connection. Leave it at `0` in Docker and CI, where no cache exists.

---

## LLM providers

The system talks to any OpenAI-compatible endpoint, so the choice of engine is
pure configuration. Switch it in the UI at runtime, or set the default in `.env`:

| Mode | Model | Trade-off |
|---|---|---|
| `local` | Ollama (default `llama3.2:3b`) | Free and fully private; weaker on multi-step synthesis |
| `hosted` | Any OpenAI-compatible API | More capable; retrieved excerpts leave your machine |

Local mode needs no API key and is the default.

### Using a hosted model (optional)

1. Get a free API key at <https://console.groq.com>
2. In `.env`:
   ```bash
   LLM_PROVIDER=hosted
   HOSTED_BASE_URL=https://api.groq.com/openai/v1
   HOSTED_MODEL=llama-3.3-70b-versatile
   HOSTED_API_KEY=your_key
   ```
3. Verify and launch:
   - **Native:** `python -m scripts.check_config`, then `python -m scripts.run_ui`
   - **Docker:** `docker compose up` reads the same `.env` for these variables

Restart after any change — configuration is read once at startup.

> ⚠️ Hosted mode sends retrieved document excerpts to the provider's API. Local
> mode keeps everything on your machine. The UI states which mode is active.

**Troubleshooting:** a 401 mentioning `platform.openai.com` means your values
aren't reaching the app — you're getting the code defaults. Check
`env | grep HOSTED`, since exported shell variables override `.env`.

---

## Evaluation & CI

Quality is measured, not assumed. Two corpora serve two purposes:

- `evaluation/golden.yaml` — hand-reviewed questions against a real document, for
  measuring answer quality
- `evaluation/ci_golden.yaml` — questions against a synthetic fixture
  (`scripts/make_ci_corpus.py`), for regression detection in CI

The split exists because CI can only test against artifacts the repo is allowed
to contain — hence a fictional handbook generated from committed code.

### Metrics

| Metric | What it measures |
|---|---|
| `retrieval_hit_rate` | Did retrieval surface the expected pages? |
| `refusal_accuracy` | Refused the unanswerable *and* answered the answerable |
| `false_refusals` | Answerable questions incorrectly declined |
| `citation_presence` | Do answers carry `[source: ...]` markers? |
| `faithfulness` (Ragas) | Fraction of answer claims supported by retrieved context |
| `faithfulness_coverage` | Fraction of cases the judge scored successfully |

Faithfulness needs a judge model. Set `JUDGE_BASE_URL`, `JUDGE_MODEL`, and
`JUDGE_API_KEY` in `.env`. A 70B-class judge is required — smaller models fail
Ragas's structured-output format (measured: 25% coverage vs 100%).

### Running evaluation locally

```bash
python -m scripts.build_index path/to/document.pdf   # index the eval corpus
python -m evaluation.run_eval                        # full run, with judge
python -m evaluation.run_eval --no-ragas             # deterministic only, fast
```

Reports land in `evaluation/results/` as JSON, with every answer and its retrieved
pages. Read these when a score moves, before changing anything.

**Baseline** (12 cases): faithfulness 0.92 at full coverage, retrieval hit rate
1.0, refusal accuracy 1.0, false refusals 0.

### The quality gate

`evaluation/gate.py` runs the same evaluation and exits non-zero if any metric
falls below its threshold (see `THRESHOLDS` at the top of the file):

```bash
python -m scripts.make_ci_corpus
CHROMA_DIR=chroma_ci_db python -m scripts.build_index data/ci/ci_corpus.pdf
CHROMA_DIR=chroma_ci_db python -m evaluation.gate
```

### In CI

`.github/workflows/quality-gate.yml` runs on every pull request, cheapest checks
first: **lint → unit tests → quality gate**. A failing metric fails the build and
blocks the merge.

CI uses hosted generation (there is no Ollama on GitHub runners) via the provider
switch. Set a repository secret named `GROQ_API_KEY` under
*Settings → Secrets and variables → Actions*.

### When the gate fails

Open the JSON report and audit the failing cases individually **before** adjusting
thresholds. Wrong ground truth in the golden set and over-strict metric
definitions are as common as genuine regressions — check all three.

---

## Limitations

- **Scanned PDFs are not supported** (no OCR). Ingestion fails with a clear error.
- **Voice requires a native Apple Silicon run** (MLX Whisper). Docker runs
  text-only and shows a notice in place of the voice panel.
- **Voice is turn-based, not streaming** — record, wait, then hear the answer.
- **Small local models struggle with synthesis.** Llama 3.2 3B handles fact
  lookup well but can mis-answer multi-step questions and "does X ever happen?"
  questions. Switch to hosted mode for those.
- **No conversational memory** — each question is answered independently.
- **Chunking is tuned for expository documents.** Narrative text may benefit from
  a larger `CHUNK_SIZE`.

---

## Privacy

- **Local mode keeps everything on the machine.** Documents, embeddings, the
  vector store, speech recognition, and speech synthesis all run locally; no
  document content is sent anywhere.
- **Hosted mode sends retrieved excerpts** to the configured API provider at query
  time. The UI states the active mode so the trade-off is always visible.
- **The UI binds to localhost only** and never creates a public share link.
- **Uploaded documents and the vector store are git-ignored.** Deleting
  `chroma_db/` removes the derived embeddings along with the documents.
- Note that ChromaDB ships anonymized telemetry enabled by default; set
  `ANONYMIZED_TELEMETRY=False` if you want to disable it.

---

## Licenses

This project is MIT licensed. Notable dependencies:

- **Piper TTS** (`piper1-gpl`) — GPL-3.0
- **Whisper** weights — MIT
- **Llama models** — Llama Community License, via Ollama