# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-core.txt .
RUN pip install -r requirements-core.txt

# runtime
# Platform pinned deliberately: App Runner runs x86_64 images.
FROM --platform=linux/amd64 python:3.11-slim

# Non-secret config = the validated .env values. Secrets are injected at runtime.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_OFFLINE=true \
    DEMO_MODE=true \
    CHROMA_DIR=/app/chroma_demo \
    SERVER_NAME=0.0.0.0 \
    API_PORT=8000 \
    LLM_PROVIDER=hosted \
    HOSTED_BASE_URL=https://api.groq.com/openai/v1 \
    HOSTED_MODEL=openai/gpt-oss-120b

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app

COPY models/ /app/models/
COPY chroma_demo/ /app/chroma_demo/
COPY demo/corpus.pdf /app/demo/corpus.pdf
COPY prompts/ /app/prompts/

COPY app/ /app/app/
COPY scripts/ /app/scripts/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "-m", "scripts.run_api"]