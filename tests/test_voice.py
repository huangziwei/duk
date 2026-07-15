import json
from pathlib import Path

import pytest

from duk import voice as voice_util


def test_resolve_voice_config_requires_text(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    with pytest.raises(ValueError):
        voice_util.resolve_voice_config(str(audio_path), base_dir=tmp_path)


def test_resolve_voice_config_from_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    config = voice_util.resolve_voice_config(
        str(audio_path), base_dir=tmp_path, voice_text="参考文本"
    )
    assert config.ref_audio == str(audio_path)
    assert config.ref_text == "参考文本"


def test_load_voice_config_resolves_relative_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "man.json"
    config_path.write_text(
        json.dumps({"name": "man", "ref_audio": "clip.wav", "ref_text": "你好"}),
        encoding="utf-8",
    )
    config = voice_util.load_voice_config(config_path)
    assert config.ref_audio == str(audio_path.resolve())
