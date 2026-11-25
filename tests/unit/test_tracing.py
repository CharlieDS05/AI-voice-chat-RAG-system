"""Tracing stays off unless configured, and every graph run carries its labels.

No network and no models: retrieval and generation are replaced with stubs,
and a callback records each step in memory instead of sending anything.
"""

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

import app.graph as graph_module
from app.config import Settings
from app.tracing import run_config

DOC = Document(
    page_content="stub", metadata={"source": "a.pdf", "page": 1, "chunk_id": "a.pdf:p1:c0"}
)


@pytest.fixture(autouse=True)
def no_tracing_env(monkeypatch):
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_tracing_is_off_by_default():
    assert not Settings(_env_file=None).tracing_enabled


def test_tracing_needs_a_key():
    assert not Settings(_env_file=None, langsmith_tracing=True).tracing_enabled


def test_tracing_is_on_with_switch_and_key():
    settings = Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="k")
    assert settings.tracing_enabled


def test_run_config_labels():
    assert run_config("api")["tags"] == ["api"]
    assert run_config("api", synthetic=True)["tags"] == ["api", "smoke-test"]
    assert run_config("ui")["metadata"]["channel"] == "ui"


class StepRecorder(BaseCallbackHandler):
    """Records the run's name, tags and metadata, and the name of every step."""

    def __init__(self):
        self.root = None
        self.steps = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name")
        if parent_run_id is None:
            self.root = {"name": name, "tags": kwargs["tags"], "metadata": kwargs["metadata"]}
        else:
            self.steps.append(name)


def _traced_run(monkeypatch, top_score, config):
    monkeypatch.setattr(graph_module, "retrieve", lambda question, index: [(DOC, top_score)])
    recorder = StepRecorder()
    graph = graph_module.make_graph(bm25_index=None)
    state = graph.invoke({"question": "q"}, config={**config, "callbacks": [recorder]})
    return state, recorder


def test_refusal_trace_has_no_generation_step(monkeypatch):
    def model_must_not_run(*args, **kwargs):
        raise AssertionError("the model must not be called for a refused question")

    monkeypatch.setattr(graph_module, "generate_answer", model_must_not_run)
    state, run = _traced_run(monkeypatch, -11.0, run_config("api", synthetic=True))

    assert state["refused"]
    assert run.root["name"] == "rag-question"
    assert {"api", "smoke-test"} <= set(run.root["tags"])
    assert run.steps == ["retrieve", "gate", "refuse"]


def test_answer_trace_has_generation_step(monkeypatch):
    class Stub:
        answer = "stub answer [source: a.pdf, p.1]"
        sources = [DOC.metadata]

    monkeypatch.setattr(graph_module, "generate_answer", lambda *a, **k: Stub())
    state, run = _traced_run(monkeypatch, 9.4, run_config("ui"))

    assert not state["refused"]
    assert run.root["metadata"]["channel"] == "ui"
    assert run.steps == ["retrieve", "gate", "generate"]
