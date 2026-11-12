"""The gate contract: the refusal decision is a pure threshold comparison
on the top score. These tests call the same function the graph uses."""

from app.graph import should_refuse


def test_measured_landmarks_route_correctly():
    # Empirical score landmarks from the project's own benchmarks:
    assert not should_refuse(+7.3)  # strong evidence
    assert not should_refuse(-3.16)  # hard paraphrase
    assert should_refuse(-11.2)  # unanswerable


def test_empty_retrieval_refuses():
    assert should_refuse(float("-inf"))
