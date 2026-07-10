"""Loader for versioned prompt configs (prompts/*.yaml).

Prompts are architecture, not string literals: they live in git,
carry a version number, and are loaded **never hard-coded** so the
evaluation layer can tie quality scores to specific prompt versions.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

from app.config import settings


class PromptConfig(BaseModel):
    name: str
    version: int
    description: str
    template: str

    def render(self, **kwargs: str) -> str:
        """Fill the template's placeholders (e.g. {context}, {question})."""
        return self.template.format(**kwargs)


def load_prompt(path: str | Path | None = None) -> PromptConfig:
    path = Path(path or settings.prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptConfig(**data)
