"""Labels for traced requests.

Tracing itself is switched on in app/config.py (off unless configured).
Each question becomes one trace; these labels say where it came from
(API, web chat, or voice), whether it was synthetic smoke-test traffic,
and which build served it, so live metrics can count real visitors only.
"""

from app.config import settings


def run_config(channel: str, synthetic: bool = False) -> dict:
    """Config for one graph run: its trace name, tags, and metadata."""
    tags = [channel, "smoke-test"] if synthetic else [channel]
    return {
        "run_name": "rag-question",
        "tags": tags,
        "metadata": {"channel": channel, "app_version": settings.app_version},
    }
