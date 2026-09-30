# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-core.txt .
RUN pip install -r requirements-core.txt

# runtime 
# Platform is chosen at build time: --platform linux/arm64 for EC2 t4g (Graviton).
FROM python:3.11-slim

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

# Create the runtime user first, so the COPY steps below can give it ownership.
# /app starts empty and owned by appuser. Changing the owner of an empty
# folder costs nothing, unlike chown -R over 180 MB of files.
RUN useradd -m -u 1000 appuser && mkdir /app && chown appuser:appuser /app
WORKDIR /app

# --chown sets the owner during the copy, so each file is stored once.
# Large, rarely-changing files first:
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser chroma_demo/ /app/chroma_demo/
COPY --chown=appuser:appuser demo/corpus.pdf /app/demo/corpus.pdf
COPY --chown=appuser:appuser prompts/ /app/prompts/

# App code last: the layer that changes on every build
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser scripts/ /app/scripts/

USER appuser

EXPOSE 8000
CMD ["python", "-m", "scripts.run_api"]