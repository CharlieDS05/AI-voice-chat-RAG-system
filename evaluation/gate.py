"""The CI quality gate: run the eval and FAIL the build
if any metric drops below its threshold.

Deterministic metrics gate strictly; the LLM-judged faithfulness gates
with margin below baseline, because judge scores carry noise and a
flaky gate is worse than no gate.

Usage: python -m evaluation.gate --golden evaluation/ci_golden.yaml
"""

import argparse
import sys

from evaluation.run_eval import (
    deterministic_metrics,
    load_golden,
    ragas_faithfulness,
    run_system,
)

# Thresholds are baselined on the CI fixture.
THRESHOLDS = [
    ("refusal_accuracy", 1.0, True),      # deterministic: no slack
    ("retrieval_hit_rate", 0.80, True),   # 4 of 5 answerable must hit
    ("citation_presence", 0.80, True),
    ("faithfulness", 0.75, True),         # margin below baseline for judge noise
    ("faithfulness_coverage", 0.99, True) # a judge that can't judge fails the gate
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="evaluation/ci_golden.yaml")
    args = parser.parse_args()

    cases = load_golden(args.golden)
    print(f"Quality gate: {len(cases)} cases from {args.golden}\n")
    records = run_system(cases)

    scores = deterministic_metrics(records)
    scores.update(ragas_faithfulness(records))

    print("\n" + "=" * 56)
    print(f"{'METRIC':<24}{'SCORE':>8}{'MIN':>8}   VERDICT")
    print("=" * 56)

    failed = False
    for metric, minimum, gating in THRESHOLDS:
        value = scores.get(metric)
        ok = value is not None and value >= minimum
        verdict = "PASS" if ok else ("FAIL" if gating else "warn")
        if gating and not ok:
            failed = True
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"{metric:<24}{shown:>8}{minimum:>8}   {verdict}")

    print("=" * 56)
    if failed:
        print("❌ Quality gate FAILED — build must not merge.")
        sys.exit(1)
    print("✅ Quality gate passed.")


if __name__ == "__main__":
    main()