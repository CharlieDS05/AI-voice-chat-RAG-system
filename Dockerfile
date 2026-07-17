FROM python:3.12-slim

# System deps: ffmpeg for Gradio's audio handling (harmless in text mode,
# required if a CPU voice backend is ever added).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first: Docker caches layers, so code edits won't
# re-trigger the slow pip install unless requirements change.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Application code
COPY app/ app/
COPY prompts/ prompts/
COPY scripts/ scripts/

# The UI must bind 0.0.0.0 inside a container to be reachable;
# privacy is enforced by compose publishing only to the host's localhost.
ENV SERVER_NAME=0.0.0.0
EXPOSE 7860

CMD ["python", "-m", "scripts.run_ui"]