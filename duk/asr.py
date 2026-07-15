from __future__ import annotations

from pathlib import Path
from typing import Optional


_LANGUAGE_MAP = {
    "chinese": "zh",
    "zh": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "ja": "ja",
    "english": "en",
    "en": "en",
    "korean": "ko",
    "ko": "ko",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "spanish": "es",
    "es": "es",
}

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MANDARIN_PROMPT = "以下是普通话的句子。"

# Short aliases so callers can say "turbo"/"small" without the full HF path.
_MODEL_ALIASES = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny",
}


def normalize_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    return _LANGUAGE_MAP.get(lowered, lowered)


def resolve_model_name(model_name: Optional[str]) -> str:
    cleaned = str(model_name or "").strip()
    if not cleaned:
        return DEFAULT_WHISPER_MODEL
    return _MODEL_ALIASES.get(cleaned.lower(), cleaned)


def transcribe_audio(
    audio_path: Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
    device: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> str:
    """Transcribe a clip with mlx-whisper; returns the raw transcript text."""
    _ = device  # MLX always runs on the unified-memory GPU.
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    import mlx_whisper

    kwargs: dict = {}
    normalized_lang = normalize_language(language)
    if normalized_lang:
        kwargs["language"] = normalized_lang
    if initial_prompt is None and normalized_lang == "zh":
        initial_prompt = DEFAULT_MANDARIN_PROMPT
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=resolve_model_name(model_name),
        **kwargs,
    )
    text = str(result.get("text") or "").strip()
    return text
