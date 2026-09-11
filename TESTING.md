# Testing & Quality

Four testing levels, ordered by cost. Run the cheapest level that answers your question.

| Level       | Command                                                                                | Time      | Needs                | Answers                                  |
| ----------- | -------------------------------------------------------------------------------------- | --------- | -------------------- | ---------------------------------------- |
| Unit        | `pytest tests/unit`                                                                    | ~11s      | nothing              | "Did I break the machinery?"             |
| Integration | `pytest -m integration`                                                                | ~16s warm | cached models        | "Do the ML components wire together?"    |
| Evaluation  | `CHROMA_DIR=chroma_eval python -m evaluation.run_eval`                                 | ~90s      | LLM + judge key      | "Is answer quality still good?"          |
| CI gate     | `CHROMA_DIR=chroma_ci_db python -m evaluation.gate --golden evaluation/ci_golden.yaml` | ~90s      | same + fixture index | "Would this change be allowed to merge?" |

## Workflows

* **While coding:** `pytest tests/unit`
* **Before committing:** `ruff check . --fix && ruff format . && pytest`
* **After prompt or retrieval changes:** rebuild the isolated eval index, then run
  the evaluation and compare to baseline (retrieval hit rate 1.0, refusal accuracy 1.0,
  false refusals 0, faithfulness 0.92). See [Corpus isolation](#corpus-isolation).
* **In CI, automatic on every PR:** lint → unit tests → quality gate

## Performance notes (measured on an 8 GB M1)

* Tests force `HF_HUB_OFFLINE=1` via `tests/conftest.py`. This mattered enormously
  under the old sentence-transformers stack (346s vs 5.7s to import). Since the
  fastembed/ONNX migration the runtime no longer loads torch at all, but the flag
  stays: tests must never depend on the network.
* ML libraries import in ~6s **warm** but can take several minutes **cold**
  (410s measured post-reboot). Loading the LLM evicts them from the file cache,
  so prefer running tests *before* the app, not after. CI runners are unaffected.
* Keep free disk space above ~20 GB; below that, macOS swap thrashing makes
  every import pathologically slow.

## Reading a failure

* **Unit red** → a code contract broke; the test name identifies the layer.
* **Integration red** → model wiring or a dependency version changed.
* **Gate red** → behavioral regression. Open the JSON in `evaluation/results/`
  and audit failing cases individually *before* touching thresholds — ground-truth
  errors and metric bugs are as common as real regressions.

## Baselines

Baselines (12-case golden set, isolated corpus):

* **Deterministic:** hit rate 1.0 · refusal accuracy 1.0 · false refusals 0
* **LLM-dependent:** citation presence 1.0 · faithfulness 0.92 at coverage 1.0 (hosted judge)
  — these vary with the generating model; local 3B typically scores lower on citations.

## The two corpora

* `evaluation/golden.yaml` — real-corpus quality (local; grows toward 100+ cases)
* `evaluation/ci_golden.yaml` — synthetic fixture for CI regression detection

## Corpus isolation

Evaluation must run against its own vector store, never the one the app uses.

```bash
CHROMA_DIR=chroma_eval python -m scripts.build_index data/samples/sample.pdf
CHROMA_DIR=chroma_eval python -m evaluation.run_eval --no-ragas

# CI fixture
CHROMA_DIR=chroma_ci_db python -m scripts.build_index data/ci/ci_corpus.pdf
CHROMA_DIR=chroma_ci_db python -m evaluation.gate --golden evaluation/ci_golden.yaml
```

Why: golden-set expectations (`expected_pages`, `answerable`) are only valid for
the corpus they were written against. Uploading a document through the UI writes
into the default collection — enough to invalidate every baseline.

Observed: a session that indexed two extra PDFs into the shared collection moved
retrieval hit rate 1.0 → 0.875 and refusal accuracy 1.0 → 0.917, because an
"unanswerable" question became answerable and extra chunks displaced correct hits.
Rebuilding an isolated collection restored both to 1.0.

Rebuild the eval index whenever the corpus, chunking parameters, or embedding
model change. `chroma_*/` is git-ignored, so these are disposable by design.
