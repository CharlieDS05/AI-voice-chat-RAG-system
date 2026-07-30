"""Pytest configuration, applied before any test module is imported.

sentence-transformers contacts the Hugging Face Hub at import time.
On a throttled connection this measured 346s vs 5.7s offline, and it
made the unit tier unusable. Tests must never depend on the network:
model files are already cached locally, and importing the library
needs no downloads at all.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
