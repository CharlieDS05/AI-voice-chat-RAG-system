# rag-voice-chat
Local-first RAG chatbot with voice: ask questions of your PDFs by text or
speech, get grounded answers with citations. Private by default.

## Features
(hybrid retrieval BM25+vector with RRF, cross-encoder reranking, structural
refusal gate, citations with page numbers, swappable LLM provider, voice via
MLX Whisper + Piper, Ragas evaluation, CI quality gate)

## Architecture
(your retrieve → gate → generate/refuse diagram; the funnel; the provider switch)

## Prerequisites
- macOS Apple Silicon (full experience) or any Docker host (text mode)
- [Ollama](https://ollama.com) running natively: `ollama pull llama3.2:3b`
- ffmpeg for voice mode: `brew install ffmpeg`
- Python 3.10+ (native mode)

## Quickstart — native (full voice)
(clone, venv, pip install -r requirements.txt, cp .env.example .env,
download Piper voice, python -m scripts.run_ui)

## Quickstart — Docker (text mode)
(docker compose up --build → http://127.0.0.1:7860)

## LLM providers
(local vs hosted; the three .env variables; the privacy trade-off stated plainly)

## Evaluation & CI
(golden dataset, run_eval, the gate, thresholds, GROQ_API_KEY secret)

## Privacy
(local mode: nothing leaves the machine — documents, embeddings, voice all local;
hosted mode sends retrieved chunks to the provider; localhost-only binding;
deleting chroma_db/ removes derived embeddings)

## Licenses
This project: MIT. Notable dependencies: Piper TTS (piper1-gpl) is GPL-3.0;
Whisper weights MIT; Llama models under the Llama license via Ollama.