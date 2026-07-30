"""The prompt contract: the YAML loads, is versioned, and its template
contains the placeholders generation depends on."""

import pytest

from app.generation.prompts import load_prompt


def test_prompt_loads_and_is_versioned():
    prompt = load_prompt()
    assert prompt.version >= 1
    assert "{context}" in prompt.template and "{question}" in prompt.template


def test_render_fails_loudly_on_missing_placeholder():
    with pytest.raises(KeyError):
        load_prompt().render(context="only context, no question")
