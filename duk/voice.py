from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_LANGUAGE = "Chinese"
VOICE_DIRNAME = "voices"


@dataclass(frozen=True)
class VoiceConfig:
    name: str
    ref_audio: str
    ref_text: Optional[str]
    language: str = DEFAULT_LANGUAGE
    gender: Optional[str] = None

    @property
    def audio_path(self) -> Path:
        return Path(self.ref_audio)


def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Voice config must be a JSON object: {path}")
    return data


def load_voice_config(path: Path) -> VoiceConfig:
    data = _load_json(path)
    name = str(data.get("name") or path.stem).strip() or path.stem
    ref_audio = str(data.get("ref_audio") or data.get("audio") or "").strip()
    if not ref_audio:
        raise ValueError(f"Voice config missing ref_audio: {path}")
    audio_path = Path(ref_audio)
    if not audio_path.is_absolute():
        audio_path = (path.parent / audio_path).resolve()
        ref_audio = str(audio_path)
    ref_text = data.get("ref_text")
    if ref_text is not None:
        ref_text = str(ref_text).strip()
        if not ref_text:
            ref_text = None
    language = str(data.get("language") or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    gender_value = data.get("gender")
    gender = None
    if gender_value is not None:
        cleaned = str(gender_value).strip().lower()
        if cleaned in {"female", "male"}:
            gender = cleaned
    return VoiceConfig(
        name=name,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        gender=gender,
    )


def write_voice_config(config: VoiceConfig, path: Path) -> None:
    payload = {
        "name": config.name,
        "ref_audio": config.ref_audio,
        "ref_text": config.ref_text,
        "language": config.language,
    }
    if config.gender:
        payload["gender"] = config.gender
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def coerce_voice_name(raw: Optional[str], source_name: str) -> str:
    value = raw or source_name
    base = Path(value).name.strip()
    if not base:
        return "voice"
    suffix = Path(base).suffix.lower()
    if suffix in {".mp3", ".wav", ".json"}:
        base = Path(base).stem
    base = base.strip()
    return base or "voice"


def _read_voice_text(
    voice_text: Optional[str],
    voice_text_path: Optional[Path],
) -> Optional[str]:
    if voice_text_path:
        text = voice_text_path.read_text(encoding="utf-8").strip()
        return text or None
    if voice_text is not None:
        text = str(voice_text).strip()
        return text or None
    return None


def resolve_voice_config(
    voice: str,
    base_dir: Path,
    voice_text: Optional[str] = None,
    voice_text_path: Optional[Path] = None,
    language: Optional[str] = None,
) -> VoiceConfig:
    if not voice:
        raise ValueError("Voice is required for synthesis.")

    candidate = Path(voice)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()

    if candidate.exists():
        if candidate.suffix.lower() == ".json":
            return load_voice_config(candidate)
        ref_text = _read_voice_text(voice_text, voice_text_path)
        if not ref_text:
            raise ValueError(
                "Voice text is required when providing a raw audio file. "
                "Pass --voice-text/--voice-text-file."
            )
        return VoiceConfig(
            name=candidate.stem,
            ref_audio=str(candidate),
            ref_text=ref_text,
            language=(language or DEFAULT_LANGUAGE),
        )

    repo_root = find_repo_root(base_dir)
    voices_dir = repo_root / VOICE_DIRNAME
    for path in (
        voices_dir / f"{voice}.json",
        voices_dir / voice / "voice.json",
    ):
        if path.exists():
            return load_voice_config(path)

    raise FileNotFoundError(f"Voice config not found: {voice}")
