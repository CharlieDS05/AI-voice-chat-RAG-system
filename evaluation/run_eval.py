"""Offline evaluation: run the graph over the golden dataset and score it.

Metrics:
  Deterministic (always run, no LLM):
    - retrieval_hit_rate : expected page present in retrieved chunks
    - refusal_accuracy   : refused unanswerables AND answered answerables
    - citation_presence  : answered responses contain [source: ...] markers
  LLM-judged (slow; skip with: --no-ragas):
    - faithfulness (Ragas): fraction of answer claims supported by context

Usage (from project root):
    python -m evaluation.run_eval
    python -m evaluation.run_eval --no-ragas
"""

import argparse
import json
import math
import re

# compat shim
# ragas (<=0.4.3) imports ChatVertexAI from langchain_community.chat_models,
# but langchain-community 0.4.x removed that module (it moved to the separate
# langchain-google-vertexai package). We never use VertexAI; this registers a
# stub so ragas can import. Delete once ragas fixes the import.
# Ref: github.com/vibrantlabsai/ragas/issues/2745
import sys
import time
import types
from pathlib import Path

import yaml

from app.config import settings
from app.graph import make_graph
from app.retrieval.bm25 import build_bm25_index

try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class _StubChatVertexAI:
        """Placeholder; ragas only references this class, never instantiates it here."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("VertexAI is not available in this environment.")

    _shim.ChatVertexAI = _StubChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

CITATION_RE = re.compile(r"[\[\(]source:.*?[\]\)]")


# Loads the golden dataset from evaluation/golden.yaml, runs the RAG graph on each case,
# and computes metrics.
def load_golden(path: str = "evaluation/golden.yaml") -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["cases"]


# Runs the RAG graph on each case and collects the results for metric computation.
def run_system(cases: list[dict]) -> list[dict]:
    """Invoke the graph once per case; collect everything metrics need."""
    graph = make_graph(build_bm25_index())
    records = []
    for case in cases:
        state = graph.invoke({"question": case["question"]})
        records.append(
            {
                **case,
                "answer": state["answer"],
                "refused": state["refused"],
                "retrieved_pages": [s["page"] for s in state["sources"]],
                "contexts": [f"[source: {s['source']}, p.{s['page']}]" for s in state["sources"]],
                "docs_text": [d.page_content for d in state.get("docs", [])],
            }
        )
        print(f"  ran: {case['id']:<28} refused={state['refused']}")
    return records


def deterministic_metrics(records: list[dict]) -> dict:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    hits = [
        any(p in r["retrieved_pages"] for p in r["expected_pages"])
        for r in answerable
        if not r["refused"]
    ]
    refusal_ok = [r["refused"] for r in unanswerable] + [not r["refused"] for r in answerable]
    cited = [bool(CITATION_RE.search(r["answer"])) for r in answerable if not r["refused"]]

    return {
        "retrieval_hit_rate": sum(hits) / len(hits) if hits else 0.0,
        "refusal_accuracy": sum(refusal_ok) / len(refusal_ok),
        "citation_presence": sum(cited) / len(cited) if cited else 0.0,
        "false_refusals": sum(r["refused"] for r in answerable),
    }


# Calculates the faithfulness metric using Ragas, which requires an LLM judge.
# Returns None if no answerable cases were answered.
def ragas_faithfulness(records: list[dict]) -> float | None:
    """Judge answered answerables with Ragas faithfulness."""
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    from app.generation.llm import get_client  # reuse the provider switch

    _, model = get_client()
    base_url = settings.judge_base_url or (
        settings.ollama_base_url if settings.llm_provider == "local" else settings.hosted_base_url
    )
    model = settings.judge_model or (
        settings.ollama_model if settings.llm_provider == "local" else settings.hosted_model
    )
    api_key = settings.judge_api_key or (
        "ollama" if settings.llm_provider == "local" else settings.hosted_api_key
    )
    # TODO: migrate to ragas llm_factory when convenient
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0.0)
    )

    rows = [
        {
            "user_input": r["question"],
            "response": CITATION_RE.sub("", r["answer"]).strip(),
            "retrieved_contexts": r["docs_text"],
        }
        for r in records
        if r["answerable"] and not r["refused"]
    ]
    if not rows:
        return None

    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[Faithfulness(llm=judge)],
        run_config=RunConfig(timeout=600, max_retries=1, max_workers=1),
    )

    per_case = [float(s) for s in result["faithfulness"] if not math.isnan(s)]
    coverage = len(per_case) / len(rows)
    if not per_case:
        return {"faithfulness": None, "faithfulness_coverage": 0.0}
    return {
        "faithfulness": round(sum(per_case) / len(per_case), 3),
        "faithfulness_coverage": round(coverage, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ragas", action="store_true", help="skip the LLM judge")
    parser.add_argument("--golden", default="evaluation/golden.yaml")
    args = parser.parse_args()

    cases = load_golden(args.golden)
    print(f"Golden dataset: {len(cases)} cases\n\nRunning system...")
    start = time.time()
    records = run_system(cases)

    scores = deterministic_metrics(records)
    if not args.no_ragas:
        print("\nJudging faithfulness with Ragas...")
        scores.update(ragas_faithfulness(records))

    scores["eval_seconds"] = round(time.time() - start, 1)

    print("\n" + "=" * 50)
    print("EVALUATION SCORECARD")
    print("=" * 50)
    for key, value in scores.items():
        print(f"  {key:<22}: {value}")

    out = Path("evaluation/results")
    out.mkdir(exist_ok=True)
    report = out / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report.write_text(json.dumps({"scores": scores, "records": records}, indent=2, default=str))
    print(f"\nFull report: {report}")


if __name__ == "__main__":
    main()
