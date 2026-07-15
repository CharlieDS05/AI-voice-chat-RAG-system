"""Speech-to-text via mlx-whisper (Metal-accelerated on Apple Silicon).

Takes a path to an audio file (Gradio's mic component hands us exactly
that) and returns the transcribed text. The model downloads once from
the Hugging Face MLX community and is cached thereafter.
"""

import mlx_whisper

from app.config import settings


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file to text. Returns '' for empty/failed audio."""
    if not audio_path:
        return ""
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=settings.whisper_model,
    )
    return (result.get("text") or "").strip()
