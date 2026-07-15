from pathlib import Path

import pytest

from duk import asr


def test_normalize_language() -> None:
    assert asr.normalize_language(None) is None
    assert asr.normalize_language("") is None
    assert asr.normalize_language("Chinese") == "zh"
    assert asr.normalize_language("Mandarin") == "zh"
    assert asr.normalize_language("EN") == "en"
    assert asr.normalize_language("xx") == "xx"


def test_resolve_model_name_aliases() -> None:
    assert asr.resolve_model_name(None) == asr.DEFAULT_WHISPER_MODEL
    assert asr.resolve_model_name("") == asr.DEFAULT_WHISPER_MODEL
    assert asr.resolve_model_name("turbo") == "mlx-community/whisper-large-v3-turbo"
    assert asr.resolve_model_name("small") == "mlx-community/whisper-small-mlx"
    assert asr.resolve_model_name("custom/model") == "custom/model"


def test_transcribe_audio_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        asr.transcribe_audio(missing)
