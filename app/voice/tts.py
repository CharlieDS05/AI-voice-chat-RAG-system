"""Text-to-speech via Piper (local, CPU, ONNX).

Synthesizes the answer text to a WAV file and returns its path,
which is exactly what Gradio's audio output component wants. The
voice model loads once per process (same lazy-singleton pattern as
the embedder and reranker).
"""

import tempfile
import wave
from pathlib import Path

from piper import PiperVoice

from app.config import settings

_voice: PiperVoice | None = None


def get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        voice_path = Path(settings.piper_voice)
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found at '{voice_path}'. "
                "Download one with: python -m piper.download_voices en_US-lessac-medium"
            )
        _voice = PiperVoice.load(str(voice_path))
    return _voice


def synthesize(text: str) -> str:
    """Render text to a temporary WAV file; returns the file path."""
    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(out_path, "wb") as wav_file:
        get_voice().synthesize_wav(text, wav_file)
    return out_path
