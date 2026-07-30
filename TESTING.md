# Testing & Quality

Three tiers, ordered by cost. Run the cheapest tier that answers your question.

| Tier | Command | Time | Needs | Answers |
|---|---|---|---|---|
| Unit | `pytest tests/unit` | ~11s | nothing | "Did I break the machinery?" |
| Integration | `pytest -m integration` | ~16s warm | cached models | "Do the ML components wire together?" |
| Evaluation | `python -m evaluation.run_eval` | ~90s | LLM + judge key | "Is answer quality still good?" |
| CI gate | `python -m evaluation.gate --golden evaluation/ci_golden.yaml` | ~90s | same + fixture index | "Would this change be allowed to merge?" |

## Workflows
- **While coding:** `pytest tests/unit`
- **Before committing:** `ruff check . --fix && ruff format . && pytest`
- **After prompt or retrieval changes:** run the evaluation and compare to baseline
  (faithfulness 0.92, retrieval hit rate 1.0, refusal accuracy 1.0, false refusals 0)
- **In CI, automatic on every PR:** lint → unit tests → quality gate

## Performance notes (measured on an 8 GB M1)
- Tests force `HF_HUB_OFFLINE=1` via `tests/conftest.py`. Without it,
  `import sentence_transformers` makes Hugging Face Hub calls at import time:
  **346s vs 5.7s** measured. Tests must never depend on the network.
- ML libraries import in ~6s **warm** but can take several minutes **cold**
  (410s measured post-reboot). Loading the LLM evicts them from the file cache,
  so prefer running tests *before* the app, not after. CI runners are unaffected.
- Keep free disk space above ~20 GB; below that, macOS swap thrashing makes
  every import pathologically slow.

## Reading a failure
- **Unit red** → a code contract broke; the test name identifies the layer.
- **Integration red** → model wiring or a dependency version changed.
- **Gate red** → behavioral regression. Open the JSON in `evaluation/results/`
  and audit failing cases individually *before* touching thresholds — ground-truth
  errors and metric bugs are as common as real regressions.

## The two corpora
- `evaluation/golden.yaml` — real-corpus quality (local; grows toward 100+ cases)
- `evaluation/ci_golden.yaml` — synthetic fixture for CI regression detection