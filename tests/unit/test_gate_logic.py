"""The gate contract: the refusal decision is a pure threshold
comparison on the top score -- tested here without models or graphs."""

from app.config import settings


def route(top_score: float) -> str:
    """Mirror of the gate's decision (kept in sync with app/graph.py)."""
    return "refuse" if top_score < settings.refusal_threshold else "generate"


def test_measured_landmarks_route_correctly():
    # Empirical score landmarks from the project's own benchmarks:
    assert route(+7.3) == "generate"  # strong evidence
    assert route(-3.16) == "generate"  # hard paraphrase
    assert route(-11.2) == "refuse"  # unanswerable


def test_empty_retrieval_refuses():
    assert route(float("-inf")) == "refuse"
