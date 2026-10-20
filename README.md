# rag-voice-chat

Local-first RAG chatbot with voice: ask questions of your own PDFs by text or
speech, and get grounded answers that cite the exact page they came from.
Private by default in local mode, nothing leaves your machine.

The system is built to refuse rather than guess: if the retrieved evidence
doesn't support an answer, it says so instead of inventing one.

**Live demo: <https://52-44-32-240.sslip.io>** answers questions about the
NIST AI Risk Management Framework, cites the page behind every claim, and
declines what the document doesn't cover. Ask *"What does the framework say
about training a transformer model?"* to watch it refuse.

## Results at a glance

| | |
|---|---|
| Demo golden set (7 cases, hosted `gpt-oss-120b`) | retrieval hit rate 1.0 · refusal accuracy 1.0 · citation presence 1.0 · 0 false refusals |
| Live smoke test against the deployed URL | 7/7 |
| Faithfulness (Ragas, 12-case real-document set) | 0.92 at full judge coverage |
| Refusal margin on the demo corpus | hardest answerable question scores −1.20, closest unanswerable −10.25, threshold −5.0 |
| Container | ≈500 MB compressed (arm64) · 525 MiB peak memory under a 1 GB limit · ~5 s to healthy |
| Hosting | AWS EC2 `t4g.small` (Graviton) · about $17/month |

---

## Features

- **Hybrid retrieval** — BM25 keyword search fused with dense vector search via
  Reciprocal Rank Fusion, so both exact terms and paraphrased meaning are found
- **Cross-encoder reranking** — a cross-encoder re-scores candidates by reading
  query and passage together, which is far more accurate than vector similarity
  alone; it runs on ONNX Runtime via fastembed
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
- **HTTP API** — FastAPI service with `/ask` (API-key auth that fails closed,
  plus rate limiting), `/health`, OpenAPI docs at `/docs`, and the Gradio UI
  mounted at `/ui`
- **Demo mode** — a read-only public configuration: a fixed corpus with
  attribution, uploads and the provider switch hidden, and a notice that
  answers come from a hosted model
- **Production deployment** — one Docker image on AWS Graviton behind HTTPS,
  with no SSH port and no stored credentials on the server

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

### Deployed demo

```
Browser ──HTTPS──▶ Caddy (automatic certificates) ──▶ app container :8000
                                                      ├── FastAPI   /ask  /health  /docs
                                                      └── Gradio    /ui
                   EC2 t4g.small · Amazon Linux 2023 · Docker Compose

Image    ◀── ECR (arm64, tagged with the git commit it was built from)
Secrets  ◀── SSM Parameter Store (read with the server's IAM role)
Answers  ──▶ Groq (openai/gpt-oss-120b)
```

The image bakes in the NIST corpus (48 pages, 144 chunks), its index, the
embedding and reranker models, and the validated non-secret configuration.
Secrets are injected only when the container starts.

**Stack:** LangGraph · ChromaDB · fastembed (ONNX Runtime) · rank-bm25 · Ollama ·
Groq · FastAPI · Gradio · MLX Whisper · Piper · Ragas · Pydantic Settings ·
Docker · AWS (EC2 Graviton, ECR, SSM) · Caddy · GitHub Actions

---

## Prerequisites

- **macOS Apple Silicon** for the full experience (voice), or **any Docker host**
  for the demo image
- **[Ollama](https://ollama.com)** running natively (local mode):
  ```bash
  ollama pull llama3.2:3b
  ```
  Ollama must run natively rather than in Docker — Docker Desktop on macOS has no
  GPU access, which forces CPU-only inference.
- **Python 3.10+** (native mode)
- **ffmpeg** for voice mode: `brew install ffmpeg` (macOS) or
  `apt-get install ffmpeg` (Linux)
- **AWS CLI v2**, only if you deploy (see [Deployment](#deployment-aws))

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

To run the API server with the UI mounted inside it instead:

```bash
python -m scripts.run_api
```

Then open **http://127.0.0.1:8000/ui** for the UI, or **/docs** for the API.

The first run downloads the embedding and reranker models (~160 MB total) and can
take several minutes; see [first-run speed](#note-on-first-run-speed) below.

## Quickstart — the demo image (Docker)

The Docker image is the public demo: it bakes in the NIST corpus, its index, and
the model files, and answers with a hosted model. It's text-only (see
[Limitations](#limitations)).

```bash
# 1. Create what the image bakes in, using the native setup above.
#    Skip the first command if chroma_demo/ already exists.
CHROMA_DIR=chroma_demo python -m scripts.build_index demo/corpus.pdf
CHROMA_DIR=chroma_demo python -m scripts.compare_rerank "What are the four core functions of the AI RMF?"   # caches the reranker

# 2. Build. On an x86 machine, use --platform linux/amd64.
docker build --platform linux/arm64 -t rag-demo:local .

# 3. Secrets are passed at run time, never baked into the image.
printf 'API_KEY=%s\nHOSTED_API_KEY=%s\n' \
  "$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" "your_groq_key" > .env.secrets

# 4. Run with the same memory limit as production.
docker run --rm -p 8000:8000 --memory=1g --env-file .env.secrets rag-demo:local
```

Open **http://localhost:8000/ui**. `.env.secrets` is git-ignored. Write its
values without quotes: Docker's `--env-file` keeps quote characters as part of
the value.

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

**3. Run the tests**

```bash
pytest tests/unit
pytest tests/test_auth.py
```

The unit tests (10, no network, no models) cover the core logic. The auth tests
cover the API key check, including the case of a server started without a key.

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
The current retrieval stack uses fastembed with ONNX Runtime rather than
sentence-transformers/PyTorch, which substantially reduces the dependency
footprint and warm import time; see [Runtime optimization](#runtime-optimization).

Once the models are cached, set `HF_OFFLINE=1` in `.env`. This skips Hugging Face
Hub network calls at import time. Tests force offline mode as well, because the
test suite must never depend on the network.

---

## Runtime optimization

The original stack loaded PyTorch through sentence-transformers for both embedding and
reranking. Replacing it with fastembed (ONNX Runtime, using the same model weights)
cut the dependency footprint by **3×** with no change in retrieval quality:

|                     | sentence-transformers | fastembed (ONNX) |
| ------------------- | --------------------- | ---------------- |
| Installed packages  | 199                   | 133              |
| Installed size      | 1,963 MB              | 636 MB           |
| Graph import (warm) | ~6s                   | 1.9s             |
| Retrieval hit rate  | 1.0                   | 1.0              |
| Refusal accuracy    | 1.0                   | 1.0              |
| False refusals      | 0                     | 0                |

Equivalence was verified before migrating: embeddings matched at cosine similarity
**1.000000** and cross-encoder scores differed by **0.0000**, so existing Chroma
indexes remained valid without re-embedding. The golden-set evaluation confirmed
retrieval behaviour end-to-end.

### The container

| | |
|---|---|
| Image size | ≈500 MB compressed (arm64) |
| Memory | 525 MiB after a full answer, under a 1 GB limit |
| Startup | ~5 s until `/health` responds |
| Cross-platform parity | reranker scores agree to six decimal places on Linux amd64 and arm64, so the refusal threshold carries over unchanged |

Two image fixes mattered:

- **`COPY --chown` instead of a later `chown -R`.** Changing ownership in a later
  layer makes Docker store a second copy of every file. The duplicate layer was
  exactly the size of `/app`, models included: 251 MB.
- **No voice models in the image.** The container can't run voice, so
  `models/voices/` (60 MiB) is excluded via `.dockerignore`.

---

## LLM providers

The system talks to any OpenAI-compatible endpoint, so the choice of engine is
pure configuration. Switch it in the UI at runtime, or set the default in `.env`:

| Mode | Model | Trade-off |
|---|---|---|
| `local` | Ollama (default `llama3.2:3b`) | Free and fully private; weaker on multi-step synthesis |
| `hosted` | Any OpenAI-compatible API (the demo uses Groq with `openai/gpt-oss-120b`) | More capable; your question and the retrieved excerpts leave your machine |

Local mode needs no API key and is the default.

### Using a hosted model (optional)

1. Get a free API key at <https://console.groq.com>
2. In `.env`:
   ```bash
   LLM_PROVIDER=hosted
   HOSTED_BASE_URL=https://api.groq.com/openai/v1
   HOSTED_MODEL=openai/gpt-oss-120b
   HOSTED_API_KEY=your_key
   ```
3. Verify and launch: `python -m scripts.check_config`, then
   `python -m scripts.run_ui`

Restart after any change — configuration is read once at startup.

> ⚠️ Hosted mode sends your question and the retrieved document excerpts to the
> provider's API. Local mode keeps everything on your machine. The UI states
> which mode is active.

Groq has retired the Llama 3.3 70B model this project first used, which is why
the demo moved to `openai/gpt-oss-120b`. The gpt-oss models write citations with
`【 】` brackets, so the answer step normalizes them to `[source: …]`: every
model produces the same citation format for the UI, the API, and the metrics.

**Troubleshooting:** a 401 mentioning `platform.openai.com` means your values
aren't reaching the app — you're getting the code defaults. Check
`env | grep HOSTED`, since exported shell variables override `.env`.

---

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check; reports the provider and whether demo mode is on |
| `POST /ask` | Answer a question; requires the `X-API-Key` header; 10 requests/minute |
| `GET /docs` | Interactive OpenAPI documentation |
| `/ui` | The Gradio interface |

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"question": "What are the four core functions of the AI RMF?"}'
```

The response carries the answer, whether it was refused, the sources (file,
page, chunk), and the top reranker score. A refused question returns a fixed
message with no sources, and the LLM is never called.

Authentication fails closed: a missing or wrong key gets **401**, and a server
started without a key configured refuses every request with **503** instead of
silently running unprotected. The key comparison is constant-time.

The public demo's API key is private; run the demo image locally to use the API
yourself.

---

## Evaluation & CI

Quality is measured, not assumed. Three golden sets serve three purposes:

- `evaluation/golden.yaml` — hand-reviewed questions against a real document, for
  measuring answer quality
- `evaluation/ci_golden.yaml` — questions against a synthetic fixture
  (`scripts/make_ci_corpus.py`), for regression detection in CI
- `evaluation/golden_demo.yaml` — questions against the public demo corpus, the
  NIST AI RMF

The CI split exists because CI can only test against artifacts the repo is
allowed to contain — hence a fictional handbook generated from committed code.

### Metrics

| Metric | What it measures |
|---|---|
| `retrieval_hit_rate` | Did retrieval surface at least one expected page? |
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

### The demo corpus

`golden_demo.yaml` covers each kind of question the demo has to handle: direct
lookups, multi-hop synthesis, a paraphrase, a hard paraphrase that shares almost
no vocabulary with the document, and two unanswerable questions — one off-topic,
and one adjacent (*training a transformer* is on-theme, but the document doesn't
cover it).

```bash
LLM_PROVIDER=hosted CHROMA_DIR=chroma_demo python -m evaluation.run_eval \
  --golden evaluation/golden_demo.yaml --no-ragas
```

**Scorecard** (7 cases, hosted `openai/gpt-oss-120b`): retrieval hit rate 1.0,
refusal accuracy 1.0, citation presence 1.0, false refusals 0.

The refusal threshold of −5.0 was re-validated on this corpus, because
cross-encoder scores compress on small corpora:

| Question type | Top rerank score | Distance from −5.0 |
|---|---|---|
| Direct (four functions) | +9.37 | +14.4 |
| Hard paraphrase (who is accountable) | −1.20 | +3.8 |
| Unanswerable, adjacent (transformer training) | −10.25 | −5.3 |
| Unanswerable, off-topic (Kubernetes) | −11.15 | −6.2 |

The deployed system is checked separately, by sending every golden question to
the live URL:

```bash
API_KEY=... python evaluation/smoke_live.py https://52-44-32-240.sslip.io
```

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

Both quality drops seen while preparing the demo were measurement problems, not
regressions:

- **Citation presence fell to 0.2** after switching to `gpt-oss-120b`. The answers
  did cite, but in `【 】` brackets the metric's pattern didn't match. The fix
  normalizes the output and keeps the metric strict.
- **Retrieval hit rate fell to 0.8** after a new case was added. Retrieval hadn't
  changed; one existing case's expected pages had been overwritten. The original
  values were recovered from an earlier evaluation report.

---

## Deployment (AWS)

The live demo runs on a single EC2 `t4g.small` (2 Graviton vCPUs, 2 GiB of
memory) for about $17 a month, including the disk and a fixed public address.

**Why EC2.** The original target, AWS App Runner, stopped accepting new customers
on April 30, 2026. Its recommended successor, ECS Express Mode, needs an
Application Load Balancer, which comes to roughly $35–40 a month at this size.
One small instance with Caddy serves the same container for less than half that.

### What runs where

- **ECR** stores the arm64 image, tagged with the git commit it was built from,
  with vulnerability scanning on push.
- **EC2** runs Amazon Linux 2023. `deploy/bootstrap.sh` installs Docker and the
  Compose plugin on first boot.
- **Docker Compose** runs two containers: the app, with a 1 GB memory limit,
  and Caddy, which obtains and renews the HTTPS certificate automatically.
  Both restart on reboot.
- **An Elastic IP** keeps the address fixed, and
  [sslip.io](https://sslip.io) turns it into a hostname, since certificates are
  issued for names rather than raw IPs.
- **`deploy/deploy.sh`** deploys or updates: it finds the newest image in ECR,
  fetches the secrets, writes the Compose and Caddy configuration, and restarts
  only what changed. It's safe to re-run.

### Security

- **No SSH port.** Administrative access goes through Systems Manager Session
  Manager.
- **No stored credentials on the server.** Its IAM role pulls images from ECR
  and reads the secrets.
- **Secrets live in Parameter Store as SecureStrings.** They're written to a
  root-only file at deploy time, and never enter the image.
- **Only ports 80 and 443 are open.** The app's port 8000 isn't reachable from
  the internet.
- **CPU credits run in standard mode,** so a traffic spike slows the server down
  instead of adding charges.

### Deploying a new version

```bash
# On your machine: build and push, tagged with the current commit
docker build --platform linux/arm64 -t rag-demo:local .
REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com
TAG=$(git rev-parse --short HEAD)
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY
docker tag rag-demo:local $REGISTRY/rag-demo:$TAG
docker push $REGISTRY/rag-demo:$TAG

# On the server, in a Session Manager terminal
sudo curl -fsSL https://raw.githubusercontent.com/CharlieDS05/AI-voice-chat-RAG-system/main/deploy/deploy.sh -o /opt/rag-demo/deploy.sh
sudo bash /opt/rag-demo/deploy.sh

# Back on your machine: check the live system
API_KEY=$(aws ssm get-parameter --name /rag-demo/API_KEY --with-decryption \
  --query Parameter.Value --output text) python evaluation/smoke_live.py https://52-44-32-240.sslip.io
```

<details>
<summary><strong>First-time setup</strong></summary>

```bash
# Secrets: a new random API key, and your Groq key
aws ssm put-parameter --name /rag-demo/API_KEY --type SecureString \
  --value="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
aws ssm put-parameter --name /rag-demo/HOSTED_API_KEY --type SecureString --value="your_groq_key"

# Image registry
aws ecr create-repository --repository-name rag-demo --image-scanning-configuration scanOnPush=true

# Server identity: pull from ECR, Session Manager, read parameters
aws iam create-role --role-name rag-demo-ec2 --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name rag-demo-ec2 --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam attach-role-policy --role-name rag-demo-ec2 --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name rag-demo-ec2
aws iam add-role-to-instance-profile --instance-profile-name rag-demo-ec2 --role-name rag-demo-ec2

# Firewall: web traffic only, no SSH
SG=$(aws ec2 create-security-group --group-name rag-demo-web \
  --description "rag-demo: HTTP and HTTPS only" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 443 --cidr 0.0.0.0/0

# Server: Graviton, Amazon Linux 2023, Docker installed on first boot
INSTANCE=$(aws ec2 run-instances \
  --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
  --instance-type t4g.small --credit-specification CpuCredits=standard \
  --iam-instance-profile Name=rag-demo-ec2 --security-group-ids $SG \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":16,"VolumeType":"gp3"}}]' \
  --user-data file://deploy/bootstrap.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rag-demo}]' \
  --query 'Instances[0].InstanceId' --output text)

# Fixed public address
ALLOC=$(aws ec2 allocate-address --domain vpc --query AllocationId --output text)
aws ec2 associate-address --instance-id $INSTANCE --allocation-id $ALLOC
```

Then push an image and run `deploy.sh` on the server, as in
[Deploying a new version](#deploying-a-new-version).

</details>

### Tearing it down

```bash
aws ec2 terminate-instances --instance-ids <instance-id>
aws ec2 release-address --allocation-id <allocation-id>    # an unattached Elastic IP is still billed
aws ecr delete-repository --repository-name rag-demo --force
aws ssm delete-parameters --names /rag-demo/API_KEY /rag-demo/HOSTED_API_KEY
```

---

## Limitations

- **Scanned PDFs are not supported** (no OCR). Ingestion fails with a clear error.
- **Voice requires a native Apple Silicon run** (MLX Whisper). Docker runs
  text-only.
- **Voice is turn-based, not streaming** — record, wait, then hear the answer.
- **Small local models struggle with synthesis.** Llama 3.2 3B handles fact
  lookup well but can mis-answer multi-step questions and "does X ever happen?"
  questions. Switch to hosted mode for those.
- **No conversational memory** — each question is answered independently.
- **Chunking is tuned for expository documents.** Narrative text may benefit from
  a larger `CHUNK_SIZE`.
- **The public demo is a single small instance** with no autoscaling, and its
  answers depend on the hosted provider's availability and rate limits.
- **The demo URL is derived from its IP address.** Recreating the server with a
  new address changes the URL.

---

## Privacy

- **Local mode keeps everything on the machine.** Documents, embeddings, the
  vector store, speech recognition, and speech synthesis all run locally; no
  document content is sent anywhere.
- **Hosted mode sends your question and the retrieved excerpts** to the
  configured API provider at query time. The UI states the active mode so the
  trade-off is always visible.
- **Local runs bind to localhost only** and never create a public share link.
- **The public demo runs in hosted mode with uploads disabled.** Questions typed
  into it are sent to the model provider, along with excerpts of the public NIST
  document; the page says so.
- **Uploaded documents and the vector store are git-ignored.** Deleting
  `chroma_db/` removes the derived embeddings along with the documents.
- Note that ChromaDB ships anonymized telemetry enabled by default; set
  `ANONYMIZED_TELEMETRY=False` if you want to disable it.

---

## Engineering notes

What broke on the way from a local app to a public deployment, and what each
problem taught:

- **The container wasn't running the evaluated configuration.** `.env` is
  correctly kept out of the image, so the container fell back to code defaults,
  including a different provider URL and model. A side-by-side dump of the
  effective settings, local versus container, found it. Validated non-secret
  settings are now baked into the image, and secrets are injected at runtime.
- **Authentication failed open.** "No key configured" meant "auth disabled",
  and `compare_digest("", "")` is `True`, so deleting the early return wasn't
  enough. An unconfigured server now returns 503, and tests cover every case.
- **A stale server hid every change.** A process from an earlier test still held
  port 8000, so the new one couldn't start and the browser kept showing the old
  code. Check the port before restarting.
- **`chown -R` stored the models twice.** The duplicate layer was exactly the
  size of `/app` (251 MB). `COPY --chown` stores each file once.
- **A TLS error turned out to be a clock error.** "Certificate is not yet valid"
  inside Docker Desktop came from the Linux VM's clock lagging behind the Mac's.
  The problem was the environment, not the code.
- **Both quality drops were measurement problems.** A citation-format mismatch
  and a damaged answer key — see [When the gate fails](#when-the-gate-fails).
- **The deployment target closed.** App Runner stopped accepting new customers
  during the project, so the same container moved to EC2.

## Changelog

- **v1.2** — Public demo on AWS EC2 (Graviton): NIST AI RMF corpus, FastAPI
  service with fail-closed auth, HTTPS via Caddy, 7/7 on the demo golden set and
  the live smoke test.
- **v1.1** — fastembed (ONNX Runtime) replaces sentence-transformers: 3× smaller
  footprint, identical retrieval.

---

## Licenses

This project is MIT licensed. Notable dependencies:

- **Piper TTS** (`piper1-gpl`) — GPL-3.0
- **Whisper** weights — MIT
- **Llama models** — Llama Community License, via Ollama

The demo corpus, NIST AI 100-1 (AI RMF 1.0), is a U.S. Government work, not
subject to copyright in the United States.
