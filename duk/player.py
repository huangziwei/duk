from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import audio_norm as audio_norm_util
from . import asr as asr_util
from . import epub as epub_util
from . import sanitize
from . import tts as tts_util
from . import voice as voice_util
from .text import read_clean_text

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _no_store(data: dict) -> JSONResponse:
    return JSONResponse(data, headers={"Cache-Control": "no-store"})

def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


_CLONE_TIME_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?$")
_VOICE_GENDERS = {"female", "male"}


def _normalize_voice_gender(value: object) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw not in _VOICE_GENDERS:
        raise ValueError("Gender must be 'female' or 'male'.")
    return raw


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_clone_source(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "duk voice clone"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _coerce_clone_voice_name(raw: Optional[str], source_name: str) -> str:
    return voice_util.coerce_voice_name(raw, source_name)


def _format_clone_seconds(seconds: float) -> str:
    rounded = round(seconds, 3)
    text = f"{rounded:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _parse_clone_time(value: object, field_name: str, *, allow_zero: bool) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        if allow_zero:
            return "0"
        raise ValueError(f"{field_name} is required.")

    if ":" in raw:
        parts = raw.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(
                f"{field_name} must be seconds or timecode (MM:SS or HH:MM:SS)."
            )
        if any(not _CLONE_TIME_TOKEN_RE.fullmatch(part) for part in parts):
            raise ValueError(
                f"{field_name} must be seconds or timecode (MM:SS or HH:MM:SS)."
            )
        sec_part = float(parts[-1])
        min_part = float(parts[-2])
        hour_part = float(parts[-3]) if len(parts) == 3 else 0.0
        if sec_part >= 60 or min_part >= 60:
            raise ValueError(f"{field_name} timecode must keep MM/SS values below 60.")
        seconds = hour_part * 3600 + min_part * 60 + sec_part
    else:
        try:
            seconds = float(raw)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a number of seconds.") from exc

    if seconds < 0:
        raise ValueError(f"{field_name} must be >= 0 seconds.")
    if not allow_zero and seconds <= 0:
        raise ValueError(f"{field_name} must be > 0 seconds.")
    return _format_clone_seconds(seconds)


def _build_clone_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    start: str,
    duration: str,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        start,
        "-t",
        duration,
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def _clone_cache_dir(repo_root: Path) -> Path:
    return repo_root / ".cache" / "voice-clone"


def _clone_preview_path(repo_root: Path) -> Path:
    return _clone_cache_dir(repo_root) / "preview.wav"


def _clone_source_cache_path(source: str, repo_root: Path) -> Path:
    parsed = urllib.parse.urlparse(source)
    suffix = Path(parsed.path).suffix.lower() or ".audio"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    return _clone_cache_dir(repo_root) / "sources" / f"{digest}{suffix}"


def _resolve_clone_source(source: str, repo_root: Path) -> tuple[Path, str, str]:
    raw = str(source or "").strip()
    if not raw:
        raise ValueError("Source is required.")
    if _is_http_url(raw):
        cache_path = _clone_source_cache_path(raw, repo_root)
        try:
            if not cache_path.exists() or cache_path.stat().st_size <= 0:
                _download_clone_source(raw, cache_path)
        except Exception as exc:
            raise ValueError(f"Download failed: {exc}") from exc
        parsed = urllib.parse.urlparse(raw)
        source_name = Path(parsed.path).stem or "voice"
        return cache_path, source_name, raw

    input_path = Path(raw).expanduser()
    if not input_path.is_absolute():
        raise ValueError("Local source path must be absolute (or start with '~/').")
    input_path = input_path.resolve()
    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f"Input file not found: {input_path}")
    return input_path, input_path.stem, str(input_path)


def _run_clone_ffmpeg(
    input_path: Path,
    output_path: Path,
    start: str,
    duration: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_clone_ffmpeg_cmd(input_path, output_path, start, duration)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            tail = detail.splitlines()[-1]
            raise RuntimeError(f"ffmpeg failed to process the audio: {tail}")
        raise RuntimeError("ffmpeg failed to process the audio.")
    try:
        audio_norm_util.normalize_clone_wav(output_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to normalize clone audio: {exc}") from exc


def _normalize_clone_text(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _payload_fields_set(payload: BaseModel) -> set[str]:
    fields = getattr(payload, "model_fields_set", None)
    if isinstance(fields, set):
        return {str(item) for item in fields}
    legacy = getattr(payload, "__fields_set__", None)
    if isinstance(legacy, set):
        return {str(item) for item in legacy}
    return set()


def _resolve_voice_config_path(value: str, repo_root: Path) -> tuple[str, Path]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Voice is required.")
    voices_dir = (repo_root / voice_util.VOICE_DIRNAME).resolve()
    candidate = Path(raw)
    resolved: Optional[Path] = None
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        repo_candidate = (repo_root / candidate).resolve()
        if repo_candidate.exists():
            resolved = repo_candidate

    config_path: Optional[Path] = None
    voice_id = ""
    if resolved is not None:
        if resolved.is_dir():
            config_path = resolved / "voice.json"
            voice_id = resolved.name
        elif resolved.suffix.lower() == ".json":
            config_path = resolved
            voice_id = resolved.stem
        else:
            config_path = resolved.with_suffix(".json")
            voice_id = resolved.stem
    else:
        voice_id = candidate.stem
        config_path = voices_dir / f"{voice_id}.json"
        if not config_path.exists():
            alt = voices_dir / voice_id / "voice.json"
            if alt.exists():
                config_path = alt
            else:
                config_path = None

    if not config_path or not config_path.exists() or not config_path.is_file():
        raise ValueError(f"Voice config not found: {raw}")
    config_path = config_path.resolve()
    if voices_dir not in config_path.parents and config_path != voices_dir:
        raise ValueError("Voice must be inside the voices directory.")
    return voice_id, config_path


def _resolve_voice_audio_path(config_path: Path, data: dict) -> tuple[str, Path]:
    ref_audio = str(data.get("ref_audio") or data.get("audio") or "").strip()
    if not ref_audio:
        return "", config_path.with_suffix(".wav")
    audio_path = Path(ref_audio)
    if not audio_path.is_absolute():
        audio_path = (config_path.parent / audio_path).resolve()
    return ref_audio, audio_path


def _resolve_clone_text(
    payload: "VoiceCloneTextPayload",
    audio_path: Path,
) -> Optional[str]:
    text = _normalize_clone_text(payload.text)
    if not text and not payload.auto_text:
        raise ValueError(
            "Reference text is required. Provide a transcript or enable Whisper "
            "auto-text."
        )
    if not text and payload.auto_text:
        whisper_language = payload.whisper_language or payload.language
        try:
            text = asr_util.transcribe_audio(
                audio_path,
                model_name=payload.whisper_model,
                language=whisper_language,
                device=payload.whisper_device,
                initial_prompt=payload.whisper_prompt,
            )
        except Exception as exc:
            raise ValueError(f"Whisper transcription failed: {exc}") from exc
        text = _normalize_clone_text(text)
        if not text:
            raise ValueError(
                "Whisper returned empty text; provide a transcript instead."
            )
    if not text:
        raise ValueError(
            "Reference text is required. Provide a transcript or enable Whisper "
            "auto-text."
        )
    return text


def _template_rules_path() -> Path:
    return sanitize.template_rules_path()


def _book_rules_path(book_dir: Path) -> Path:
    return sanitize.book_rules_path(book_dir)


def _select_rules_path(book_dir: Path) -> Optional[Path]:
    book_rules = _book_rules_path(book_dir)
    if book_rules.exists():
        return book_rules
    template_rules = _template_rules_path()
    if template_rules.exists():
        return template_rules
    return None


def _rules_payload_from_ruleset(rules: sanitize.Ruleset) -> dict:
    return {
        "replace_defaults": rules.replace_defaults,
        "drop_chapter_title_patterns": list(rules.drop_chapter_title_patterns),
        "section_cutoff_patterns": list(rules.section_cutoff_patterns),
        "remove_patterns": list(rules.remove_patterns),
    }


def _effective_rules_payload(book_dir: Path) -> dict:
    rules_path = _select_rules_path(book_dir)
    rules = sanitize.load_rules(rules_path)
    return _rules_payload_from_ruleset(rules)


def _write_rules_payload(rules_path: Path, payload: dict) -> None:
    data = {
        "replace_defaults": bool(payload.get("replace_defaults", False)),
        "drop_chapter_title_patterns": list(
            payload.get("drop_chapter_title_patterns", [])
        ),
        "section_cutoff_patterns": list(payload.get("section_cutoff_patterns", [])),
        "remove_patterns": list(payload.get("remove_patterns", [])),
    }
    if not data["replace_defaults"]:
        for key, defaults in sanitize.DEFAULT_RULES.items():
            data[key] = [entry for entry in data[key] if entry not in defaults]
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(rules_path, data)


def _highlight_ranges(text: str, patterns: list[re.Pattern]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            if match.start() == match.end():
                continue
            ranges.append((match.start(), match.end()))
    if not ranges:
        return []
    ranges.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _render_highlight(text: str, ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return html.escape(text)
    parts: list[str] = []
    last = 0
    for start, end in ranges:
        parts.append(html.escape(text[last:start]))
        parts.append(f"<mark>{html.escape(text[start:end])}</mark>")
        last = end
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def _load_chapter_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _pick_preview_chapter(clean_toc: dict, chapter_index: Optional[int]) -> dict:
    chapters = clean_toc.get("chapters", []) if isinstance(clean_toc, dict) else []
    if not chapters:
        return {}
    if chapter_index is not None:
        for entry in chapters:
            if entry.get("index") == chapter_index:
                return entry
    return chapters[0]


def _find_clean_chapter(clean_toc: dict, chapter_index: int) -> Optional[dict]:
    chapters = clean_toc.get("chapters", []) if isinstance(clean_toc, dict) else []
    for entry in chapters:
        if entry.get("index") == chapter_index:
            return entry
    return None


def _resolve_raw_path(book_dir: Path, raw_toc: dict, clean_entry: dict) -> Optional[Path]:
    source_index = clean_entry.get("source_index")
    if source_index is None:
        return None
    for entry in raw_toc.get("chapters", []):
        if entry.get("index") == source_index:
            rel = entry.get("path")
            if rel:
                return book_dir / rel
    return None


def _slug_from_title(title: str, fallback: str) -> str:
    base = title.strip() if title else fallback
    base = base or fallback or "book"
    slug = _sanitize_title_for_path(base)
    return slug or "book"


def _sanitize_title_for_path(value: str, max_len: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("/", " ").replace("\\", " ")
    text = text.translate({ord(ch): " " for ch in ':*?"<>|'})
    text = "".join(ch for ch in text if ch.isprintable() and ch != "\x00")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(".")
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _ensure_unique_slug(root_dir: Path, slug: str) -> str:
    if not (root_dir / slug).exists():
        return slug
    for idx in range(2, 1000):
        candidate = f"{slug}-{idx}"
        if not (root_dir / candidate).exists():
            return candidate
    return f"{slug}-{int(time.time())}"


def _source_type_from_toc(toc: dict) -> str:
    source = str(toc.get("source_epub") or "")
    suffix = Path(source).suffix.lower()
    if suffix == ".txt":
        return "txt"
    if suffix == ".epub":
        return "epub"
    return "unknown"


def _source_origin_from_metadata(metadata: dict) -> str:
    if not isinstance(metadata, dict):
        return "books"
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    if isinstance(authors, list):
        for author in authors:
            if str(author or "").strip().lower() == "tsundoku":
                return "tsundoku"
    return "books"


def _resolve_book_dir(root_dir: Path, book_id: str) -> Path:
    candidate = (root_dir / book_id).resolve()
    if root_dir not in candidate.parents and candidate != root_dir:
        raise HTTPException(status_code=404, detail="Book not found.")
    if not (candidate / "clean" / "toc.json").exists():
        raise HTTPException(status_code=404, detail="Book not found.")
    return candidate


def _sample_chapter_info(book_dir: Path) -> tuple[str, str]:
    manifest = _load_json(book_dir / "tts" / "manifest.json")
    chapters = manifest.get("chapters") if isinstance(manifest, dict) else []
    if isinstance(chapters, list) and chapters:
        entry = chapters[0]
        return str(entry.get("id") or ""), str(entry.get("title") or "")

    clean_toc = _load_json(book_dir / "clean" / "toc.json")
    entries = clean_toc.get("chapters") if isinstance(clean_toc, dict) else []
    if isinstance(entries, list) and entries:
        entry = entries[0]
        title = str(entry.get("title") or "")
        rel_path = entry.get("path") or ""
        if rel_path:
            return Path(rel_path).stem, title
        idx = entry.get("index") or 1
        slug = epub_util.slugify(title or "chapter")
        return f"{int(idx):04d}-{slug}", title

    return "", ""


def _total_chunks_from_manifest(manifest: dict) -> int:
    if not isinstance(manifest, dict):
        return 0
    chapters = manifest.get("chapters", [])
    total = 0
    if not isinstance(chapters, list):
        return total
    for entry in chapters:
        if not isinstance(entry, dict):
            continue
        chunks = entry.get("chunks")
        spans = entry.get("chunk_spans")
        if isinstance(chunks, list):
            total += len(chunks)
        elif isinstance(spans, list):
            total += len(spans)
    return total


def _book_summary(book_dir: Path) -> dict:
    toc = _load_json(book_dir / "clean" / "toc.json")
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    cover = metadata.get("cover") or {}
    cover_path = cover.get("path") or ""
    cover_url = f"/audio/{book_dir.name}/{cover_path}" if cover_path else ""
    chapters = toc.get("chapters", []) if isinstance(toc, dict) else []
    source_type = _source_type_from_toc(toc) if isinstance(toc, dict) else "unknown"
    source_origin = _source_origin_from_metadata(metadata)
    manifest_path = book_dir / "tts" / "manifest.json"
    has_audio = manifest_path.exists()
    manifest = _load_json(manifest_path)
    audio_total, audio_done = _audio_progress_summary(book_dir, manifest)
    total_chunks = _total_chunks_from_manifest(manifest) if has_audio else 0
    raw_playback = _load_json(_playback_path(book_dir))
    if not isinstance(raw_playback, dict):
        raw_playback = {}
    playback = _sanitize_playback(raw_playback)
    furthest = playback.get("furthest_played")
    if furthest is None:
        furthest = playback.get("last_played")
    playback_complete = (
        has_audio
        and total_chunks > 0
        and isinstance(furthest, int)
        and furthest >= total_chunks - 1
    )
    return {
        "id": book_dir.name,
        "title": metadata.get("title") or book_dir.name,
        "authors": metadata.get("authors") or [],
        "year": metadata.get("year") or "",
        "cover_url": cover_url,
        "has_audio": has_audio,
        "audio_total": audio_total,
        "audio_done": audio_done,
        "playback_complete": playback_complete,
        "chapter_count": len(chapters) if isinstance(chapters, list) else 0,
        "source_type": source_type,
        "source_origin": source_origin,
    }


def _list_voice_ids(repo_root: Path) -> List[str]:
    voices_dir = repo_root / "voices"
    ids: List[str] = []
    if voices_dir.exists():
        for path in sorted(voices_dir.glob("*.json")):
            if path.stem == "metadata":
                continue
            ids.append(path.stem)
        for path in sorted(voices_dir.glob("*/voice.json")):
            ids.append(path.parent.name)
    return sorted(set(ids))


def _normalize_voice_value(value: object, repo_root: Path) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned.lower() == "default":
        return ""
    candidate = Path(cleaned)
    if candidate.is_absolute():
        try:
            rel = candidate.relative_to(repo_root)
        except ValueError:
            return cleaned
        return rel.as_posix()
    return cleaned


def _normalize_metadata_text(value: Optional[str]) -> str:
    return str(value or "").strip()


def _normalize_authors(raw: Union[str, List[str], None]) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[,\n]+", str(raw))
    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _default_book_voice(book_dir: Path, repo_root: Path) -> str:
    manifest = _load_json(book_dir / "tts" / "manifest.json")
    fallback = _normalize_voice_value(manifest.get("voice"), repo_root)
    if fallback:
        return fallback
    voices = _list_voice_ids(repo_root)
    return voices[0] if voices else ""


def _voice_map_path(book_dir: Path) -> Path:
    return book_dir / "voice-map.json"


def _sanitize_voice_map(payload: dict, repo_root: Path, fallback_default: str) -> dict:
    default_voice = _normalize_voice_value(payload.get("default"), repo_root)
    if not default_voice:
        default_voice = fallback_default or ""
    chapters: dict[str, str] = {}
    raw_chapters = payload.get("chapters", {})
    if isinstance(raw_chapters, dict):
        for key, value in raw_chapters.items():
            voice = _normalize_voice_value(value, repo_root)
            if not voice or voice == default_voice:
                continue
            chapters[str(key)] = voice
    return {"default": default_voice, "chapters": chapters}


def _load_voice_map(book_dir: Path, repo_root: Path) -> dict:
    path = _voice_map_path(book_dir)
    if not path.exists():
        return {"default": _default_book_voice(book_dir, repo_root), "chapters": {}}
    data = _load_json(path)
    return _sanitize_voice_map(
        data if isinstance(data, dict) else {},
        repo_root,
        fallback_default=_default_book_voice(book_dir, repo_root),
    )


def _book_details(book_dir: Path, repo_root: Path) -> dict:
    toc = _load_json(book_dir / "clean" / "toc.json")
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    cover = metadata.get("cover") or {}
    cover_path = cover.get("path") or ""
    cover_url = f"/audio/{book_dir.name}/{cover_path}" if cover_path else ""
    source_type = _source_type_from_toc(toc) if isinstance(toc, dict) else "unknown"
    source_origin = _source_origin_from_metadata(metadata)

    manifest = _load_json(book_dir / "tts" / "manifest.json")
    chapters: List[dict] = []
    pad_ms = 0
    last_voice = ""
    audio_total, audio_done = _audio_progress_summary(book_dir, manifest)
    if manifest and isinstance(manifest.get("chapters"), list):
        try:
            pad_ms = int(manifest.get("pad_ms") or 0)
        except (TypeError, ValueError):
            pad_ms = 0
        last_voice = _normalize_voice_value(manifest.get("voice"), repo_root)
        chapter_total = len(manifest["chapters"])
        for chapter_idx, entry in enumerate(manifest["chapters"]):
            chunk_spans = entry.get("chunk_spans", [])
            if not isinstance(chunk_spans, list):
                chunk_spans = []
            pause_multipliers = entry.get("pause_multipliers", [])
            if not isinstance(pause_multipliers, list):
                pause_multipliers = []
            if len(pause_multipliers) != len(chunk_spans):
                pause_multipliers = tts_util._normalize_pause_multipliers(
                    pause_multipliers,
                    len(chunk_spans),
                    fallback=tts_util._legacy_pause_multipliers(
                        entry.get("chunk_section_breaks"),
                        len(chunk_spans),
                        add_chapter_boundary=chapter_idx < chapter_total - 1,
                    ),
                )
            chapters.append(
                {
                    "id": entry.get("id") or "",
                    "title": entry.get("title") or entry.get("id") or "Chapter",
                    "chunk_spans": chunk_spans,
                    "pause_multipliers": pause_multipliers,
                    "chunk_count": len(chunk_spans),
                }
            )

    return {
        "book": {
            "id": book_dir.name,
            "title": metadata.get("title") or book_dir.name,
            "authors": metadata.get("authors") or [],
            "year": metadata.get("year") or "",
            "cover_url": cover_url,
            "has_audio": bool(chapters),
            "audio_total": audio_total,
            "audio_done": audio_done,
            "pad_ms": pad_ms,
            "last_voice": last_voice,
            "source_type": source_type,
            "source_origin": source_origin,
        },
        "chapters": chapters,
        "audio_base": f"/audio/{book_dir.name}/tts/segments",
    }

def _playback_path(book_dir: Path) -> Path:
    return book_dir / "playback.json"


def _boundary_latency_log_path(book_dir: Path) -> Path:
    return book_dir / "tts" / "player-boundary-latency.jsonl"


def _append_jsonl(path: Path, entries: List[dict]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _m4b_library_dir(root_dir: Path) -> Path:
    return root_dir / "_m4b"


def _merge_output_path(book_dir: Path) -> Path:
    library_dir = _m4b_library_dir(book_dir.parent)
    return library_dir / f"{book_dir.name}.m4b"


def _list_m4b_parts(book_dir: Path) -> List[Path]:
    library_dir = _m4b_library_dir(book_dir.parent)
    pattern = f"{book_dir.name}.part*.m4b"
    return sorted(library_dir.glob(pattern))


def _delete_m4b_outputs(book_dir: Path) -> bool:
    removed = False
    base = _merge_output_path(book_dir)
    if base.exists():
        base.unlink()
        removed = True
    for path in _list_m4b_parts(book_dir):
        if path.exists():
            path.unlink()
            removed = True
    return removed


def _merge_progress_data(tts_dir: Path) -> dict:
    progress_path = tts_dir / "merge.progress.json"
    if progress_path.exists():
        return _load_json(progress_path)
    part_paths = sorted(tts_dir.glob("merge.progress.part*.json"))
    if part_paths:
        return _load_json(part_paths[-1])
    return {}


def _clear_merge_progress(tts_dir: Path) -> None:
    progress_path = tts_dir / "merge.progress.json"
    if progress_path.exists():
        progress_path.unlink()
    for part_path in tts_dir.glob("merge.progress.part*.json"):
        try:
            part_path.unlink()
        except OSError:
            continue


def _merge_ready(book_dir: Path) -> bool:
    manifest = _load_json(book_dir / "tts" / "manifest.json")
    if not manifest:
        return False
    progress = _compute_progress(manifest)
    if not progress or not progress.get("total"):
        return False
    return progress.get("done", 0) > 0


def _ffmpeg_install_command() -> str:
    installer = shutil.which("brew")
    if installer is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install Homebrew (https://brew.sh/) "
            "or install ffmpeg manually, then retry."
        )
    return f"{installer} install ffmpeg"


def _build_merge_command(
    book_dir: Path,
    output_path: Path,
    overwrite: bool,
    install_ffmpeg: bool,
    progress_path: Path,
) -> list[str]:
    merge_cmd = [
        "uv",
        "run",
        "duk",
        "merge",
        "--book",
        str(book_dir),
        "--output",
        str(output_path),
        "--progress-file",
        str(progress_path),
    ]
    if overwrite:
        merge_cmd.append("--overwrite")
    if not install_ffmpeg:
        return merge_cmd

    install_cmd = _ffmpeg_install_command()
    merge_cmd_str = " ".join(shlex.quote(part) for part in merge_cmd)
    combined = f"{install_cmd} && {merge_cmd_str}"
    return ["bash", "-lc", combined]


def _sanitize_playback(data: dict) -> dict:
    last = data.get("last_played")
    if not isinstance(last, int) or last < 0:
        last = None
    furthest = data.get("furthest_played")
    if not isinstance(furthest, int) or furthest < 0:
        furthest = None
    bookmarks: List[dict] = []
    raw_marks = data.get("bookmarks")
    if isinstance(raw_marks, list):
        for entry in raw_marks:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            if not isinstance(idx, int) or idx < 0:
                continue
            cleaned = {"index": idx}
            label = entry.get("label")
            created_at = entry.get("created_at")
            if isinstance(label, str):
                cleaned["label"] = label
            if isinstance(created_at, int):
                cleaned["created_at"] = created_at
            bookmarks.append(cleaned)
    return {
        "last_played": last,
        "furthest_played": furthest,
        "bookmarks": bookmarks,
    }


def _sanitize_boundary_log_entries(entries: list[dict]) -> list[dict]:
    def _optional_ms(value: object, upper_bound: float = 5000.0) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= parsed <= upper_bound):
            return None
        return round(parsed, 3)

    cleaned: list[dict] = []
    for raw in entries[:500]:
        if not isinstance(raw, dict):
            continue
        try:
            from_index = int(raw.get("from_index"))
            to_index = int(raw.get("to_index"))
            delta_ms = float(raw.get("delta_ms"))
        except (TypeError, ValueError):
            continue
        if from_index < 0 or to_index < 0:
            continue
        if not (-5000.0 <= delta_ms <= 5000.0):
            continue
        try:
            from_chunk_index = (
                int(raw.get("from_chunk_index"))
                if raw.get("from_chunk_index") is not None
                else None
            )
        except (TypeError, ValueError):
            from_chunk_index = None
        try:
            to_chunk_index = (
                int(raw.get("to_chunk_index"))
                if raw.get("to_chunk_index") is not None
                else None
            )
        except (TypeError, ValueError):
            to_chunk_index = None
        try:
            remaining_ms = float(raw.get("remaining_ms") or 0.0)
        except (TypeError, ValueError):
            remaining_ms = 0.0
        try:
            pad_ms = int(raw.get("pad_ms") or 0)
        except (TypeError, ValueError):
            pad_ms = 0
        try:
            playback_rate = float(raw.get("playback_rate") or 1.0)
        except (TypeError, ValueError):
            playback_rate = 1.0
        try:
            captured_unix = int(raw.get("captured_unix") or time.time())
        except (TypeError, ValueError):
            captured_unix = int(time.time())
        entry = {
            "logged_unix": int(time.time()),
            "captured_unix": captured_unix,
            "from_index": from_index,
            "to_index": to_index,
            "from_chapter_id": str(raw.get("from_chapter_id") or ""),
            "to_chapter_id": str(raw.get("to_chapter_id") or ""),
            "from_chunk_index": from_chunk_index,
            "to_chunk_index": to_chunk_index,
            "chapter_switch": bool(raw.get("chapter_switch", False)),
            "trigger": str(raw.get("trigger") or "auto"),
            "remaining_ms": round(max(0.0, remaining_ms), 3),
            "delta_ms": round(delta_ms, 3),
            "pad_ms": max(0, pad_ms),
            "playback_rate": round(max(0.25, min(4.0, playback_rate)), 3),
            "preloaded": bool(raw.get("preloaded", False)),
        }
        play_call_delay_ms = _optional_ms(raw.get("play_call_delay_ms"))
        if play_call_delay_ms is not None:
            entry["play_call_delay_ms"] = play_call_delay_ms
        play_promise_ms = _optional_ms(raw.get("play_promise_ms"))
        if play_promise_ms is not None:
            entry["play_promise_ms"] = play_promise_ms
        base_latency_ms = _optional_ms(raw.get("base_latency_ms"), upper_bound=2000.0)
        if base_latency_ms is not None:
            entry["base_latency_ms"] = base_latency_ms
        output_latency_ms = _optional_ms(
            raw.get("output_latency_ms"), upper_bound=5000.0
        )
        if output_latency_ms is not None:
            entry["output_latency_ms"] = output_latency_ms
        audio_context_state = str(raw.get("audio_context_state") or "").strip()
        if audio_context_state:
            entry["audio_context_state"] = audio_context_state[:32]
        cleaned.append(entry)
    return cleaned



def _compute_progress(manifest: dict) -> dict:
    chapters = manifest.get("chapters", [])
    total = 0
    done = 0
    current = None
    global_index = 0

    for entry in chapters if isinstance(chapters, list) else []:
        chunks = entry.get("chunks", [])
        durations = entry.get("durations_ms", [])
        if not isinstance(chunks, list):
            chunks = []
        total += len(chunks)
        for idx in range(len(chunks)):
            global_index += 1
            if idx < len(durations) and durations[idx] is not None:
                done += 1
                continue
            if current is None:
                current = {
                    "chapter_id": entry.get("id") or "",
                    "chapter_title": entry.get("title") or "",
                    "chunk_index": idx + 1,
                    "chunk_total": len(chunks),
                    "global_index": global_index,
                }

    percent = (done / total * 100.0) if total else 0.0
    return {
        "total": total,
        "done": done,
        "percent": round(percent, 2),
        "current": current,
    }


def _audio_total_from_manifest(manifest: dict) -> int:
    if not isinstance(manifest, dict):
        return 0
    total = _total_chunks_from_manifest(manifest)
    if total:
        return total
    chapters = manifest.get("chapters", [])
    if not isinstance(chapters, list):
        return 0
    total = 0
    for entry in chapters:
        durations = entry.get("durations_ms", [])
        if isinstance(durations, list):
            total += len(durations)
    return total


def _audio_done_from_manifest(manifest: dict) -> int:
    if not isinstance(manifest, dict):
        return 0
    chapters = manifest.get("chapters", [])
    if not isinstance(chapters, list):
        return 0
    done = 0
    for entry in chapters:
        durations = entry.get("durations_ms", [])
        if not isinstance(durations, list):
            continue
        done += sum(1 for value in durations if value is not None)
    return done


def _count_segment_wavs(book_dir: Path) -> int:
    segments_dir = book_dir / "tts" / "segments"
    if not segments_dir.exists():
        return 0
    total = 0
    for chapter_dir in segments_dir.iterdir():
        if not chapter_dir.is_dir():
            continue
        for wav in chapter_dir.glob("*.wav"):
            try:
                if wav.is_file() and wav.stat().st_size > 0:
                    total += 1
            except OSError:
                continue
    return total


def _audio_progress_summary(book_dir: Path, manifest: dict) -> tuple[int, int]:
    total = _audio_total_from_manifest(manifest)
    done = _audio_done_from_manifest(manifest)
    if total:
        done = min(done, total)
    if total and done < total:
        m4b_path = _merge_output_path(book_dir)
        if m4b_path.exists():
            done = total
        elif done == 0:
            wav_done = _count_segment_wavs(book_dir)
            if wav_done:
                done = min(wav_done, total)
    return total, done


def _compute_chapter_progress(manifest: dict, chapter_id: str) -> dict:
    chapters = manifest.get("chapters", [])
    entry = None
    for item in chapters if isinstance(chapters, list) else []:
        if (item.get("id") or "") == chapter_id:
            entry = item
            break
    if not entry:
        return {}
    chunks = entry.get("chunks", [])
    durations = entry.get("durations_ms", [])
    if not isinstance(chunks, list):
        chunks = []
    total = len(chunks)
    done = 0
    current = None
    for idx in range(total):
        if idx < len(durations) and durations[idx] is not None:
            done += 1
            continue
        if current is None:
            current = {
                "chapter_id": entry.get("id") or "",
                "chapter_title": entry.get("title") or "",
                "chunk_index": idx + 1,
                "chunk_total": total,
                "global_index": idx + 1,
            }
    percent = (done / total * 100.0) if total else 0.0
    return {
        "total": total,
        "done": done,
        "percent": round(percent, 2),
        "current": current,
    }


def _ffmpeg_log_path(repo_root: Path) -> Path:
    return repo_root / ".cache" / "ffmpeg-install.log"


def _spawn_ffmpeg_install(
    repo_root: Path,
) -> tuple[Optional["FfmpegJob"], Optional[str]]:
    if shutil.which("ffmpeg") is not None:
        return None, None
    try:
        install_cmd = _ffmpeg_install_command()
    except RuntimeError as exc:
        return None, str(exc)

    log_path = _ffmpeg_log_path(repo_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-lc", install_cmd],
        cwd=str(repo_root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return (
        FfmpegJob(
            process=process,
            started_at=time.time(),
            log_path=log_path,
            log_handle=log_handle,
        ),
        None,
    )


def _load_tts_status(book_dir: Path) -> dict:
    path = book_dir / "tts" / "status.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class SynthJob:
    book_id: str
    book_dir: Path
    process: subprocess.Popen
    started_at: float
    log_path: Path
    voice: str
    max_chars: int
    pad_ms: int
    chunk_mode: str
    rechunk: bool
    mode: str = "tts"
    sample_chapter_id: Optional[str] = None
    sample_chapter_title: Optional[str] = None
    log_handle: Optional[IO[str]] = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None
    min_chars: int = 0


@dataclass
class MergeJob:
    book_id: str
    book_dir: Path
    process: subprocess.Popen
    started_at: float
    log_path: Path
    output_path: Path
    install_ffmpeg: bool
    progress_path: Path
    log_handle: Optional[IO[str]] = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None


@dataclass
class FfmpegJob:
    process: subprocess.Popen
    started_at: float
    log_path: Path
    log_handle: Optional[IO[str]] = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None


@dataclass
class VoiceClonePreview:
    source_key: str
    start: str
    duration: str
    path: Path


class SynthRequest(BaseModel):
    book_id: str
    voice: Optional[str] = None
    max_chars: int = 100
    min_chars: int = 15
    pad_ms: int = 350
    chunk_mode: str = "chinese"
    rechunk: bool = False
    use_voice_map: bool = False


class ChunkSynthRequest(BaseModel):
    book_id: str
    chapter_id: str
    chunk_index: int
    voice: Optional[str] = None
    use_voice_map: bool = False


class MergeRequest(BaseModel):
    book_id: str
    overwrite: bool = False


class StopRequest(BaseModel):
    book_id: str


class ClearRequest(BaseModel):
    book_id: str


class RulesPayload(BaseModel):
    book_id: Optional[str] = None
    drop_chapter_title_patterns: List[str] = []
    section_cutoff_patterns: List[str] = []
    remove_patterns: List[str] = []
    replace_defaults: bool = False


class VoiceMapPayload(BaseModel):
    default: Optional[str] = None
    chapters: dict = {}


class ChapterAction(BaseModel):
    book_id: str
    title: str
    chapter_index: Optional[int] = None


class SanitizeRequest(BaseModel):
    book_id: str


class CleanEditPayload(BaseModel):
    book_id: str
    chapter_index: int
    text: str


class PlaybackPayload(BaseModel):
    last_played: Optional[int] = None
    furthest_played: Optional[int] = None
    bookmarks: List[dict] = []


class BoundaryLogPayload(BaseModel):
    book_id: str
    entries: List[dict] = []


class DeleteBookRequest(BaseModel):
    book_id: str


class DeleteM4bRequest(BaseModel):
    book_id: str


class MetadataPayload(BaseModel):
    book_id: str
    title: Optional[str] = None
    authors: Union[str, List[str], None] = None
    year: Optional[str] = None


class VoiceCloneTextPayload(BaseModel):
    text: Optional[str] = None
    auto_text: bool = True
    whisper_model: str = asr_util.DEFAULT_WHISPER_MODEL
    whisper_language: Optional[str] = None
    whisper_device: Optional[str] = None
    whisper_prompt: Optional[str] = None
    language: Optional[str] = None
    gender: Optional[str] = None


class VoiceClonePreviewPayload(VoiceCloneTextPayload):
    source: str
    start: Optional[str] = None
    duration: Union[str, float, int] = 12


class VoiceCloneSavePayload(VoiceCloneTextPayload):
    source: str
    start: Optional[str] = None
    duration: Union[str, float, int] = 12
    name: Optional[str] = None
    overwrite: bool = False


class VoiceMetadataPayload(BaseModel):
    voice: str
    gender: Optional[str] = None
    name: Optional[str] = None


class VoiceDeletePayload(BaseModel):
    voice: str


def create_app(root_dir: Path) -> FastAPI:
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app = FastAPI()
    root_dir = root_dir.resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _find_repo_root(root_dir)
    jobs: dict[str, SynthJob] = {}
    merge_jobs: dict[str, MergeJob] = {}
    ffmpeg_job: Optional[FfmpegJob] = None
    ffmpeg_error: Optional[str] = None
    clone_preview: Optional[VoiceClonePreview] = None

    app.mount("/audio", StaticFiles(directory=str(root_dir)), name="audio")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        context = {"root_dir": str(root_dir)}
        return templates.TemplateResponse(request, "player.html", context)

    @app.get("/api/books")
    def list_books() -> JSONResponse:
        books: List[dict] = []
        for child in sorted(root_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            if child.name == "_m4b":
                continue
            toc_path = child / "clean" / "toc.json"
            if not toc_path.exists():
                continue
            books.append(_book_summary(child))
        def sort_key(entry: dict) -> str:
            title = str(entry.get("title") or "").strip().lower()
            if title.startswith(("the ", "a ", "an ")):
                parts = title.split(" ", 1)
                if len(parts) == 2:
                    title = parts[1]
            return title or str(entry.get("id") or "").lower()

        books.sort(key=sort_key)
        return _no_store({"root": str(root_dir), "books": books})

    @app.get("/api/books/{book_id}")
    def get_book(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        return _no_store(_book_details(book_dir, repo_root))

    @app.get("/api/books/{book_id}/chapter-text")
    def get_chapter_text(book_id: str, chapter_id: str) -> JSONResponse:
        if not chapter_id:
            raise HTTPException(status_code=400, detail="Missing chapter_id.")
        book_dir = _resolve_book_dir(root_dir, book_id)
        manifest = _load_json(book_dir / "tts" / "manifest.json")
        chapters = manifest.get("chapters") if isinstance(manifest, dict) else []
        if not isinstance(chapters, list):
            chapters = []
        target = None
        for entry in chapters:
            if not isinstance(entry, dict):
                continue
            if (entry.get("id") or "") == chapter_id:
                target = entry
                break
        if not target:
            raise HTTPException(status_code=404, detail="Chapter not found.")
        rel_path = target.get("path")
        if not rel_path:
            raise HTTPException(status_code=404, detail="Chapter text not found.")
        clean_path = (book_dir / rel_path).resolve()
        if book_dir not in clean_path.parents and clean_path != book_dir:
            raise HTTPException(status_code=404, detail="Chapter not found.")
        if not clean_path.exists():
            raise HTTPException(status_code=404, detail="Chapter text not found.")
        clean_text = tts_util._normalize_text(read_clean_text(clean_path))
        return _no_store(
            {
                "book_id": book_id,
                "chapter_id": chapter_id,
                "clean_text": clean_text,
            }
        )

    @app.get("/api/books/{book_id}/voices")
    def get_book_voices(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        payload = _load_voice_map(book_dir, repo_root)
        return _no_store(payload)

    @app.post("/api/books/{book_id}/voices")
    def set_book_voices(book_id: str, payload: VoiceMapPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        data = _sanitize_voice_map(
            payload.dict(),
            repo_root,
            fallback_default=_default_book_voice(book_dir, repo_root),
        )
        path = _voice_map_path(book_dir)
        _atomic_write_json(path, data)
        return _no_store(data)

    @app.post("/api/books/delete")
    def delete_book(payload: DeleteBookRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before deleting.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before deleting.")
        _delete_m4b_outputs(book_dir)
        if book_dir.exists():
            shutil.rmtree(book_dir)
        return _no_store({"status": "deleted", "book_id": payload.book_id})

    @app.post("/api/m4b/delete")
    def delete_m4b(payload: DeleteM4bRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before deleting.")
        removed = _delete_m4b_outputs(book_dir)
        return _no_store(
            {
                "status": "deleted" if removed else "missing",
                "book_id": payload.book_id,
            }
        )

    @app.post("/api/books/metadata")
    def update_metadata(payload: MetadataPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        updates: dict[str, object] = {}
        if payload.title is not None:
            updates["title"] = _normalize_metadata_text(payload.title)
        if payload.authors is not None:
            updates["authors"] = _normalize_authors(payload.authors)
        if payload.year is not None:
            updates["year"] = _normalize_metadata_text(payload.year)

        for path in (book_dir / "toc.json", book_dir / "clean" / "toc.json"):
            if not path.exists():
                continue
            data = _load_json(path)
            if not isinstance(data, dict):
                continue
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            for key, value in updates.items():
                metadata[key] = value
            data["metadata"] = metadata
            _atomic_write_json(path, data)

        metadata_payload: dict = {}
        clean_toc = _load_json(book_dir / "clean" / "toc.json")
        if isinstance(clean_toc, dict):
            meta = clean_toc.get("metadata")
            if isinstance(meta, dict):
                metadata_payload = meta
        if not metadata_payload:
            raw_toc = _load_json(book_dir / "toc.json")
            if isinstance(raw_toc, dict):
                meta = raw_toc.get("metadata")
                if isinstance(meta, dict):
                    metadata_payload = meta

        return _no_store(
            {
                "status": "ok",
                "metadata": metadata_payload,
                "book": _book_summary(book_dir),
            }
        )

    @app.get("/api/voices")
    def list_voices() -> JSONResponse:
        local: List[dict] = []
        for voice_id in _list_voice_ids(repo_root):
            entry = {"label": voice_id, "value": voice_id}
            try:
                _, config_path = _resolve_voice_config_path(voice_id, repo_root)
                data = _load_json(config_path)
                if isinstance(data, dict):
                    try:
                        gender = _normalize_voice_gender(data.get("gender"))
                    except ValueError:
                        gender = None
                    if gender:
                        entry["gender"] = gender
            except Exception:
                pass
            local.append(entry)
        return _no_store({"local": local, "builtin": [], "default": ""})

    @app.post("/api/books/cover")
    def update_cover(
        book_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> JSONResponse:
        allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        if file.content_type not in allowed:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG, PNG, WebP, or GIF images are accepted.",
            )
        book_dir = _resolve_book_dir(root_dir, book_id)
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        ext = ext_map.get(file.content_type, ".jpg")
        # Remove old cover files
        for old in book_dir.glob("cover.*"):
            if old.is_file():
                old.unlink()
        cover_filename = f"cover{ext}"
        dest = book_dir / cover_filename
        with open(dest, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        # Update metadata.cover in toc.json files
        for path in (book_dir / "toc.json", book_dir / "clean" / "toc.json"):
            if not path.exists():
                continue
            data = _load_json(path)
            if not isinstance(data, dict):
                continue
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["cover"] = {"path": cover_filename}
            data["metadata"] = metadata
            _atomic_write_json(path, data)
        cover_url = f"/audio/{book_dir.name}/{cover_filename}"
        return _no_store(
            {
                "status": "ok",
                "cover_url": cover_url,
                "book": _book_summary(book_dir),
            }
        )

    @app.post("/api/voices/metadata")
    def set_voice_metadata(payload: VoiceMetadataPayload) -> JSONResponse:
        try:
            voice_id, config_path = _resolve_voice_config_path(payload.voice, repo_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        data = _load_json(config_path)
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="Voice config must be a JSON object.",
            )

        fields_set = _payload_fields_set(payload)
        has_gender = "gender" in fields_set
        has_name = "name" in fields_set
        voices_dir = (repo_root / voice_util.VOICE_DIRNAME).resolve()

        old_voice_id = voice_id
        next_voice_id = voice_id
        next_config_path = config_path
        if has_name:
            requested_name = str(payload.name or "").strip()
            if not requested_name:
                raise HTTPException(status_code=400, detail="Voice name is required.")
            next_stem = _coerce_clone_voice_name(requested_name, voice_id)
            next_voice_id = next_stem
            if config_path.name == "voice.json" and config_path.parent != voices_dir:
                next_dir = config_path.parent.parent / next_stem
                if next_dir.resolve() != config_path.parent.resolve():
                    if next_dir.exists():
                        raise HTTPException(
                            status_code=409,
                            detail=f"Voice '{next_stem}' already exists.",
                        )
                    try:
                        config_path.parent.rename(next_dir)
                    except OSError as exc:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Unable to rename voice: {exc}",
                    ) from exc
                    next_config_path = next_dir / "voice.json"
            else:
                next_config_path = config_path.with_name(f"{next_stem}.json")
                if next_config_path.resolve() != config_path.resolve():
                    if next_config_path.exists():
                        raise HTTPException(
                            status_code=409,
                            detail=f"Voice '{next_stem}' already exists.",
                        )
                    try:
                        config_path.rename(next_config_path)
                    except OSError as exc:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Unable to rename voice: {exc}",
                        ) from exc

            data["name"] = next_stem
        else:
            next_voice_id = voice_id

        ref_audio_value, audio_path = _resolve_voice_audio_path(next_config_path, data)
        if has_name:
            if (
                audio_path.exists()
                and audio_path.is_file()
                and voices_dir in audio_path.resolve().parents
                and audio_path.stem == old_voice_id
            ):
                next_audio_path = audio_path.with_name(
                    f"{next_voice_id}{audio_path.suffix}"
                )
                if next_audio_path.resolve() != audio_path.resolve():
                    if next_audio_path.exists():
                        raise HTTPException(
                            status_code=409,
                            detail=f"Voice audio '{next_voice_id}' already exists.",
                        )
                    try:
                        audio_path.rename(next_audio_path)
                    except OSError as exc:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Unable to rename voice audio: {exc}",
                        ) from exc
                    audio_path = next_audio_path
                    if ref_audio_value and not Path(ref_audio_value).is_absolute():
                        data["ref_audio"] = next_audio_path.relative_to(
                            next_config_path.parent
                        ).as_posix()
                    else:
                        data["ref_audio"] = str(next_audio_path)

        if has_gender:
            try:
                gender = _normalize_voice_gender(payload.gender)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if gender:
                data["gender"] = gender
            else:
                data.pop("gender", None)
        else:
            try:
                gender = _normalize_voice_gender(data.get("gender"))
            except ValueError:
                gender = None

        _atomic_write_json(next_config_path, data)
        return _no_store(
            {
                "status": "saved",
                "voice": next_voice_id,
                "gender": gender,
                "name": next_voice_id,
            }
        )

    @app.post("/api/voices/delete")
    def delete_voice(payload: VoiceDeletePayload) -> JSONResponse:
        try:
            voice_id, config_path = _resolve_voice_config_path(payload.voice, repo_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        data = _load_json(config_path)
        if not isinstance(data, dict):
            data = {}
        _, audio_path = _resolve_voice_audio_path(config_path, data)
        voices_dir = (repo_root / voice_util.VOICE_DIRNAME).resolve()

        if audio_path.exists() and audio_path.is_file():
            if voices_dir in audio_path.resolve().parents:
                try:
                    audio_path.unlink()
                except OSError as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unable to delete voice audio: {exc}",
                    ) from exc

        try:
            config_path.unlink()
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to delete voice config: {exc}",
            ) from exc

        if config_path.parent != voices_dir and config_path.name == "voice.json":
            try:
                if not any(config_path.parent.iterdir()):
                    config_path.parent.rmdir()
            except OSError:
                pass

        return _no_store({"status": "deleted", "voice": voice_id})

    @app.post("/api/voices/clone/preview")
    def preview_clone_voice(payload: VoiceClonePreviewPayload) -> JSONResponse:
        nonlocal clone_preview
        if shutil.which("ffmpeg") is None:
            raise HTTPException(status_code=400, detail="ffmpeg not found on PATH.")

        try:
            input_path, source_name, source_key = _resolve_clone_source(
                payload.source, repo_root
            )
            start = _parse_clone_time(payload.start, "start", allow_zero=True)
            duration = _parse_clone_time(payload.duration, "duration", allow_zero=False)
            gender = _normalize_voice_gender(payload.gender)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _ = gender

        preview_path = _clone_preview_path(repo_root)
        try:
            _run_clone_ffmpeg(input_path, preview_path, start, duration)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            text = _resolve_clone_text(payload, preview_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        clone_preview = VoiceClonePreview(
            source_key=source_key,
            start=start,
            duration=duration,
            path=preview_path,
        )
        return _no_store(
            {
                "status": "ready",
                "preview_url": f"/api/voices/clone/preview-audio?ts={int(time.time() * 1000)}",
                "suggested_name": _coerce_clone_voice_name(None, source_name),
                "start": start,
                "duration": duration,
                "text": text or "",
            }
        )

    @app.get("/api/voices/clone/preview-audio")
    def preview_clone_voice_audio() -> FileResponse:
        preview_path = _clone_preview_path(repo_root)
        if not preview_path.exists() or preview_path.stat().st_size <= 0:
            raise HTTPException(status_code=404, detail="No clone preview available.")
        return FileResponse(
            str(preview_path),
            media_type="audio/wav",
            filename="voice-preview.wav",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/voices/clone/save")
    def save_clone_voice(payload: VoiceCloneSavePayload) -> JSONResponse:
        nonlocal clone_preview
        if shutil.which("ffmpeg") is None:
            raise HTTPException(status_code=400, detail="ffmpeg not found on PATH.")

        try:
            input_path, source_name, source_key = _resolve_clone_source(
                payload.source, repo_root
            )
            start = _parse_clone_time(payload.start, "start", allow_zero=True)
            duration = _parse_clone_time(payload.duration, "duration", allow_zero=False)
            gender = _normalize_voice_gender(payload.gender)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        output_name = _coerce_clone_voice_name(payload.name, source_name)
        voices_dir = repo_root / voice_util.VOICE_DIRNAME
        voices_dir.mkdir(parents=True, exist_ok=True)
        output_path = voices_dir / f"{output_name}.wav"
        config_path = voices_dir / f"{output_name}.json"
        replaced = output_path.exists()
        if replaced and not payload.overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"Voice '{output_name}' already exists.",
            )

        can_reuse_preview = (
            clone_preview is not None
            and clone_preview.path.exists()
            and clone_preview.source_key == source_key
            and clone_preview.start == start
            and clone_preview.duration == duration
        )
        try:
            if can_reuse_preview:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(clone_preview.path, output_path)
            else:
                _run_clone_ffmpeg(input_path, output_path, start, duration)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Unable to write voice: {exc}") from exc

        try:
            text = _resolve_clone_text(payload, output_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        language = str(payload.language or voice_util.DEFAULT_LANGUAGE).strip()
        if not language:
            language = voice_util.DEFAULT_LANGUAGE
        config = voice_util.VoiceConfig(
            name=output_name,
            ref_audio=output_path.name,
            ref_text=text,
            language=language,
            gender=gender,
        )
        voice_util.write_voice_config(config, config_path)

        return _no_store(
            {
                "status": "saved",
                "voice": {"label": output_name, "value": output_name},
                "overwrote": replaced,
                "used_preview": can_reuse_preview,
            }
        )

    @app.get("/api/chunk-status")
    def chunk_status(
        book_id: str,
        chapter_id: str,
        chunk: int,
    ) -> JSONResponse:
        if chunk < 1:
            return _no_store({"exists": False})
        book_dir = _resolve_book_dir(root_dir, book_id)
        wav_path = (
            book_dir
            / "tts"
            / "segments"
            / chapter_id
            / f"{chunk:06d}.wav"
        )
        exists = wav_path.exists() and wav_path.is_file() and wav_path.stat().st_size > 0
        return _no_store({"exists": exists})

    @app.get("/api/playback")
    def playback_get(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        path = _playback_path(book_dir)
        exists = path.exists()
        data = _load_json(path) if exists else {}
        payload = _sanitize_playback(data)
        payload["exists"] = exists
        return _no_store(payload)

    @app.post("/api/playback")
    def playback_set(book_id: str, payload: PlaybackPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        cleaned = _sanitize_playback(payload.dict())
        cleaned["updated_unix"] = int(time.time())
        _atomic_write_json(_playback_path(book_dir), cleaned)
        cleaned["exists"] = True
        return _no_store(cleaned)

    @app.post("/api/playback/boundary-log")
    def playback_boundary_log(payload: BoundaryLogPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        cleaned = _sanitize_boundary_log_entries(payload.entries)
        if not cleaned:
            return _no_store({"logged": 0})
        log_path = _boundary_latency_log_path(book_dir)
        _append_jsonl(log_path, cleaned)
        try:
            rel = log_path.relative_to(book_dir)
            log_ref = rel.as_posix()
        except ValueError:
            log_ref = str(log_path)
        return _no_store({"logged": len(cleaned), "path": log_ref})

    @app.post("/api/ingest")
    def ingest_file(file: UploadFile = File(...), override: bool = False) -> JSONResponse:
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in {".epub", ".txt"}:
            raise HTTPException(
                status_code=400,
                detail="Only .epub or .txt files are supported.",
            )

        tmp_path = None
        title = Path(filename).stem
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                shutil.copyfileobj(file.file, handle)
                tmp_path = Path(handle.name)

            metadata = {}
            if suffix == ".epub":
                try:
                    book = epub_util.read_epub(tmp_path)
                    metadata = epub_util.extract_metadata(book)
                    if metadata.get("title"):
                        title = str(metadata.get("title") or "").strip() or title
                except Exception:
                    metadata = {}

            slug = _slug_from_title(title, Path(filename).stem)
            out_dir = root_dir / slug
            if out_dir.exists():
                if not override:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "detail": f"Book already exists: {slug}",
                            "book_id": slug,
                        },
                    )
                synth_job = jobs.get(slug)
                if synth_job and synth_job.process.poll() is None:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "detail": "Stop TTS before overwriting.",
                            "book_id": slug,
                        },
                    )
                merge_job = merge_jobs.get(slug)
                if merge_job and merge_job.process.poll() is None:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "detail": "Stop merge before overwriting.",
                            "book_id": slug,
                        },
                    )
                resolved = out_dir.resolve()
                if root_dir not in resolved.parents and resolved != root_dir:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid book path.",
                    )
                m4b_path = _merge_output_path(resolved)
                if m4b_path.exists():
                    m4b_path.unlink()
                if resolved.is_dir():
                    shutil.rmtree(resolved)
                else:
                    resolved.unlink()
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "ingest.log"
            stage = "ingest"
            with log_path.open("w", encoding="utf-8") as log_handle:
                cmd = [
                    "uv",
                    "run",
                    "duk",
                    "ingest",
                    "--input",
                    str(tmp_path),
                    "--out",
                    str(out_dir),
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                if proc.returncode == 0:
                    stage = "sanitize"
                    log_handle.write("\n--- sanitize ---\n")
                    cmd = [
                        "uv",
                        "run",
                        "duk",
                        "sanitize",
                        "--book",
                        str(out_dir),
                        "--overwrite",
                    ]
                    proc = subprocess.run(
                        cmd,
                        cwd=str(repo_root),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
            if proc.returncode != 0:
                detail = (
                    f"Sanitize failed. Check {log_path}."
                    if stage == "sanitize"
                    else f"Ingest failed. Check {log_path}."
                )
                raise HTTPException(
                    status_code=400,
                    detail=detail,
                )
            return _no_store(
                {
                    "status": "ok",
                    "book_id": slug,
                    "title": title,
                    "sanitized": True,
                }
            )
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    @app.get("/api/sanitize/preview")
    def sanitize_preview(
        book_id: str, chapter: Optional[int] = None
    ) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        clean_toc = _load_json(book_dir / "clean" / "toc.json")
        raw_toc = _load_json(book_dir / "toc.json")
        report = _load_json(book_dir / "clean" / "report.json")
        if not clean_toc:
            raise HTTPException(status_code=404, detail="Missing clean/toc.json.")
        rules_path = _select_rules_path(book_dir)
        rules = sanitize.load_rules(rules_path)
        selected = _pick_preview_chapter(clean_toc, chapter)

        clean_path = None
        clean_rel = selected.get("path") if selected else None
        if clean_rel:
            clean_path = book_dir / clean_rel
        raw_path = _resolve_raw_path(book_dir, raw_toc, selected)

        raw_text = _load_chapter_text(raw_path)
        clean_text = _load_chapter_text(clean_path)
        patterns = sanitize.compile_patterns(rules.remove_patterns)
        raw_ranges = _highlight_ranges(raw_text, patterns)

        dropped = []
        for entry in report.get("chapters", []):
            if entry.get("dropped"):
                dropped.append(
                    {
                        "index": entry.get("index"),
                        "title": entry.get("title") or "",
                    }
                )

        payload = {
            "book_id": book_id,
            "chapters": clean_toc.get("chapters", []),
            "selected": selected,
            "raw_text": _render_highlight(raw_text, raw_ranges),
            "clean_text": clean_text,
            "dropped": dropped,
            "rules": {
                "replace_defaults": rules.replace_defaults,
                "drop_chapter_title_patterns": rules.drop_chapter_title_patterns,
                "section_cutoff_patterns": rules.section_cutoff_patterns,
                "remove_patterns": rules.remove_patterns,
            },
        }
        return _no_store(payload)

    @app.post("/api/sanitize/rules")
    def sanitize_rules(payload: RulesPayload) -> JSONResponse:
        data = payload.dict()
        book_id = data.pop("book_id", None)
        if not book_id:
            raise HTTPException(
                status_code=400,
                detail="book_id is required to save per-book rules.",
            )
        book_dir = _resolve_book_dir(root_dir, book_id)
        _write_rules_payload(_book_rules_path(book_dir), data)
        return _no_store(payload.dict())

    @app.post("/api/sanitize/drop")
    def sanitize_drop(payload: ChapterAction) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before editing.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before editing.")
        rules = _effective_rules_payload(book_dir)
        pattern = f"^{re.escape(payload.title)}$"
        patterns = list(rules.get("drop_chapter_title_patterns", []))
        if pattern not in patterns:
            patterns.append(pattern)
        rules["drop_chapter_title_patterns"] = patterns
        _write_rules_payload(_book_rules_path(book_dir), rules)
        try:
            dropped = sanitize.drop_chapter(
                book_dir=book_dir,
                title=payload.title,
                chapter_index=payload.chapter_index,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _no_store({"status": "ok", "pattern": pattern, "dropped": dropped})

    @app.post("/api/sanitize/restore")
    def sanitize_restore(payload: ChapterAction) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before editing.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before editing.")
        rules = _effective_rules_payload(book_dir)
        pattern = f"^{re.escape(payload.title)}$"
        patterns = list(rules.get("drop_chapter_title_patterns", []))
        if pattern in patterns:
            patterns = [p for p in patterns if p != pattern]
        rules["drop_chapter_title_patterns"] = patterns
        _write_rules_payload(_book_rules_path(book_dir), rules)
        try:
            rules_path = _select_rules_path(book_dir)
            restored = sanitize.restore_chapter(
                book_dir=book_dir,
                title=payload.title,
                chapter_index=payload.chapter_index,
                rules_path=rules_path,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _no_store({"status": "ok", "pattern": pattern, "restored": restored})

    @app.post("/api/sanitize/run")
    def sanitize_run(payload: SanitizeRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before sanitizing.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before sanitizing.")
        tts_cleared = (book_dir / "tts").exists()
        log_path = book_dir / "sanitize.log"
        cmd = [
            "uv",
            "run",
            "duk",
            "sanitize",
            "--book",
            str(book_dir),
            "--overwrite",
        ]
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"Sanitize failed. Check {log_path}.",
            )
        return _no_store({"status": "ok", "tts_cleared": tts_cleared})

    @app.post("/api/sanitize/clean")
    def sanitize_clean(payload: CleanEditPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before editing text.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before editing text.")

        clean_toc = _load_json(book_dir / "clean" / "toc.json")
        if not clean_toc:
            raise HTTPException(status_code=404, detail="Missing clean/toc.json.")
        entry = _find_clean_chapter(clean_toc, payload.chapter_index)
        rel_path = entry.get("path") if entry else None
        if not rel_path:
            raise HTTPException(status_code=404, detail="Chapter not found.")
        clean_path = book_dir / rel_path
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.write_text(payload.text.rstrip() + "\n", encoding="utf-8")

        try:
            tts_cleared = sanitize.refresh_chunks(book_dir=book_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _no_store(
            {"status": "ok", "chapter_index": payload.chapter_index, "tts_cleared": tts_cleared}
        )

    @app.get("/api/synth/status")
    def synth_status(book_id: str) -> JSONResponse:
        nonlocal ffmpeg_job, ffmpeg_error
        book_dir = _resolve_book_dir(root_dir, book_id)
        manifest_path = book_dir / "tts" / "manifest.json"
        manifest = _load_json(manifest_path)
        progress = _compute_progress(manifest) if manifest else None
        overall_progress = progress
        tts_status = _load_tts_status(book_dir)
        manifest_created = 0
        if manifest:
            try:
                manifest_created = int(manifest.get("created_unix") or 0)
            except (TypeError, ValueError):
                manifest_created = 0

        job = jobs.get(book_id)
        running = False
        exit_code = None
        mode = "tts"
        sample_chapter = None
        if job:
            exit_code = job.process.poll()
            mode = job.mode or "tts"
            sample_chapter = job.sample_chapter_id
            if exit_code is None:
                running = True
            else:
                job.exit_code = exit_code
                job.ended_at = job.ended_at or time.time()
                if job.log_handle:
                    job.log_handle.close()
                    job.log_handle = None
        if running and job and job.rechunk:
            started_at = int(job.started_at)
            if manifest_created and manifest_created < started_at:
                progress = None

        if manifest and mode == "sample" and sample_chapter:
            progress = _compute_chapter_progress(manifest, sample_chapter) or None

        ffmpeg_status = (
            "installed" if shutil.which("ffmpeg") is not None else "missing"
        )
        ffmpeg_log = ""
        if ffmpeg_job:
            ffmpeg_log = str(ffmpeg_job.log_path.relative_to(repo_root))
            ffmpeg_exit = ffmpeg_job.process.poll()
            if ffmpeg_exit is None:
                ffmpeg_status = "installing"
            else:
                ffmpeg_job.exit_code = ffmpeg_exit
                ffmpeg_job.ended_at = ffmpeg_job.ended_at or time.time()
                if ffmpeg_job.log_handle:
                    ffmpeg_job.log_handle.close()
                    ffmpeg_job.log_handle = None
                if ffmpeg_exit != 0:
                    ffmpeg_status = "error"
                    if not ffmpeg_error:
                        ffmpeg_error = (
                            f"ffmpeg install failed. Check {ffmpeg_log}."
                        )
        if ffmpeg_status == "installed":
            ffmpeg_error = None

        log_path = ""
        if job:
            try:
                log_path = str(job.log_path.relative_to(book_dir))
            except ValueError:
                log_path = str(job.log_path)

        payload = {
            "book_id": book_id,
            "running": running,
            "exit_code": exit_code,
            "progress": progress,
            "log_path": log_path,
            "stage": "idle",
            "ffmpeg_status": ffmpeg_status,
            "ffmpeg_error": ffmpeg_error,
            "ffmpeg_log_path": ffmpeg_log,
            "mode": mode,
            "tts_status": tts_status,
            "overall_progress": overall_progress,
        }
        if running:
            status_stage = str(tts_status.get("stage") or "")
            if status_stage == "cloning":
                payload["stage"] = "cloning"
            elif status_stage == "loading_model":
                payload["stage"] = "loading_model"
            elif not progress or not progress.get("total"):
                payload["stage"] = "chunking"
            else:
                payload["stage"] = "sampling" if mode == "sample" else "synthesizing"
        elif mode == "sample" and job and job.exit_code == 0:
            payload["stage"] = "sampled"
        elif progress and progress.get("total") and progress.get("done") >= progress.get("total"):
            payload["stage"] = "done"
        return _no_store(payload)

    @app.post("/api/synth/start")
    def synth_start(payload: SynthRequest) -> JSONResponse:
        nonlocal ffmpeg_job, ffmpeg_error
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        if payload.chunk_mode == "packed":
            payload.chunk_mode = "chinese"

        existing = jobs.get(payload.book_id)
        if existing and existing.process.poll() is None:
            raise HTTPException(status_code=409, detail="TTS is already running.")

        use_voice_map = bool(payload.use_voice_map)
        voice_value = payload.voice
        voice_map_path = None
        if use_voice_map:
            voice_map_path = _voice_map_path(book_dir)
            voice_map = _load_voice_map(book_dir, repo_root)
            voice_value = voice_map.get("default") or voice_value
        if not voice_value:
            raise HTTPException(status_code=400, detail="Select a voice first.")

        try:
            voice_util.resolve_voice_config(voice=voice_value, base_dir=repo_root)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if shutil.which("ffmpeg") is None:
            if ffmpeg_job and ffmpeg_job.process.poll() is None:
                pass
            else:
                ffmpeg_error = None
                ffmpeg_job, error = _spawn_ffmpeg_install(repo_root)
                if error:
                    ffmpeg_error = error

        tts_dir = book_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        if payload.rechunk:
            seg_dir = tts_dir / "segments"
            if seg_dir.exists():
                shutil.rmtree(seg_dir)
        log_path = tts_dir / "synth.log"
        log_handle = log_path.open("w", encoding="utf-8")

        cmd = [
            "uv",
            "run",
            "duk",
            "synth",
            "--book",
            str(book_dir),
            "--voice",
            voice_value,
            "--max-chars",
            str(payload.max_chars),
            "--min-chars",
            str(payload.min_chars),
            "--pad-ms",
            str(payload.pad_ms),
        ]
        if use_voice_map and voice_map_path and voice_map_path.exists():
            cmd += ["--voice-map", str(voice_map_path)]
        if payload.rechunk:
            cmd.append("--rechunk")

        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        jobs[payload.book_id] = SynthJob(
            book_id=payload.book_id,
            book_dir=book_dir,
            process=process,
            started_at=time.time(),
            log_path=log_path,
            voice=voice_value,
            max_chars=payload.max_chars,
            pad_ms=payload.pad_ms,
            chunk_mode=payload.chunk_mode,
            rechunk=payload.rechunk,
            mode="tts",
            log_handle=log_handle,
            min_chars=payload.min_chars,
        )

        return _no_store({"status": "started", "book_id": payload.book_id})

    @app.post("/api/synth/sample")
    def synth_sample(payload: SynthRequest) -> JSONResponse:
        nonlocal ffmpeg_job, ffmpeg_error
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        if payload.chunk_mode == "packed":
            payload.chunk_mode = "chinese"
        if payload.use_voice_map:
            raise HTTPException(
                status_code=400,
                detail="Sample is disabled in Advanced Mode.",
            )
        if not payload.voice:
            raise HTTPException(status_code=400, detail="Select a voice first.")

        existing = jobs.get(payload.book_id)
        if existing and existing.process.poll() is None:
            raise HTTPException(status_code=409, detail="TTS is already running.")

        try:
            voice_util.resolve_voice_config(voice=payload.voice, base_dir=repo_root)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if shutil.which("ffmpeg") is None:
            if ffmpeg_job and ffmpeg_job.process.poll() is None:
                pass
            else:
                ffmpeg_error = None
                ffmpeg_job, error = _spawn_ffmpeg_install(repo_root)
                if error:
                    ffmpeg_error = error

        tts_dir = book_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        log_path = tts_dir / "sample.log"
        log_handle = log_path.open("w", encoding="utf-8")
        sample_id, sample_title = _sample_chapter_info(book_dir)
        if not sample_id:
            log_handle.close()
            raise HTTPException(status_code=404, detail="No chapters available to sample.")

        sample_seg_dir = tts_dir / "segments" / sample_id
        if sample_seg_dir.exists():
            shutil.rmtree(sample_seg_dir)

        manifest_path = tts_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
            chapters_meta = manifest.get("chapters")
            if isinstance(chapters_meta, list):
                for entry in chapters_meta:
                    if entry.get("id") == sample_id:
                        chunks = entry.get("chunks")
                        if isinstance(chunks, list):
                            entry["durations_ms"] = [None] * len(chunks)
                        break
                _atomic_write_json(manifest_path, manifest)

        cmd = [
            "uv",
            "run",
            "duk",
            "sample",
            "--book",
            str(book_dir),
            "--voice",
            payload.voice,
            "--max-chars",
            str(payload.max_chars),
            "--min-chars",
            str(payload.min_chars),
            "--pad-ms",
            str(payload.pad_ms),
        ]
        if payload.rechunk:
            cmd.append("--rechunk")

        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        jobs[payload.book_id] = SynthJob(
            book_id=payload.book_id,
            book_dir=book_dir,
            process=process,
            started_at=time.time(),
            log_path=log_path,
            voice=payload.voice,
            max_chars=payload.max_chars,
            pad_ms=payload.pad_ms,
            chunk_mode=payload.chunk_mode,
            rechunk=payload.rechunk,
            mode="sample",
            sample_chapter_id=sample_id,
            sample_chapter_title=sample_title or None,
            log_handle=log_handle,
            min_chars=payload.min_chars,
        )

        return _no_store({"status": "started", "book_id": payload.book_id})

    @app.post("/api/synth/chunk")
    def synth_chunk(payload: ChunkSynthRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Merge is running.")
        if payload.chunk_index < 0:
            raise HTTPException(status_code=400, detail="chunk_index must be >= 0")

        tts_dir = book_dir / "tts"
        manifest_path = tts_dir / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Missing TTS manifest.")

        voice_map_path = _voice_map_path(book_dir) if payload.use_voice_map else None

        try:
            result = tts_util.synthesize_chunk(
                out_dir=tts_dir,
                chapter_id=payload.chapter_id,
                chunk_index=payload.chunk_index,
                voice=payload.voice,
                voice_map_path=voice_map_path,
                base_dir=repo_root,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return _no_store({"status": "ok", **result})

    @app.post("/api/synth/stop")
    def synth_stop(payload: StopRequest) -> JSONResponse:
        job = jobs.get(payload.book_id)
        if not job or job.process.poll() is not None:
            return _no_store({"status": "not_running", "book_id": payload.book_id})

        job.process.terminate()
        try:
            job.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            job.process.kill()
            job.process.wait(timeout=2)

        job.exit_code = job.process.returncode
        job.ended_at = time.time()
        if job.log_handle:
            job.log_handle.close()
            job.log_handle = None

        return _no_store(
            {"status": "stopped", "book_id": payload.book_id, "exit_code": job.exit_code}
        )

    @app.post("/api/tts/clear")
    def clear_tts(payload: ClearRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before clearing cache.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before clearing cache.")
        tts_dir = book_dir / "tts"
        manifest_path = tts_dir / "manifest.json"
        max_chars = 0
        min_chars = 15
        pad_ms = 350
        chunk_mode = "chinese"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            try:
                max_chars = int(manifest.get("max_chars") or max_chars)
            except (TypeError, ValueError):
                max_chars = 0
            try:
                min_chars = int(manifest.get("min_chars") or min_chars)
            except (TypeError, ValueError):
                min_chars = 15
            try:
                pad_ms = int(manifest.get("pad_ms") or pad_ms)
            except (TypeError, ValueError):
                pad_ms = 350
            raw_chunk_mode = str(manifest.get("chunk_mode") or chunk_mode).strip()
            if raw_chunk_mode:
                chunk_mode = raw_chunk_mode
        try:
            tts_cleared = sanitize.refresh_chunks(
                book_dir=book_dir,
                max_chars=max_chars,
                pad_ms=pad_ms,
                chunk_mode=chunk_mode,
                min_chars=min_chars,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Rechunk failed: {exc}") from exc
        return _no_store(
            {"status": "cleared", "book_id": payload.book_id, "tts_cleared": tts_cleared}
        )

    @app.get("/api/merge/status")
    def merge_status(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        output_path = _merge_output_path(book_dir)
        output_parts = _list_m4b_parts(book_dir)
        tts_dir = book_dir / "tts"
        progress_path = tts_dir / "merge.progress.json"
        _ = progress_path
        job = merge_jobs.get(book_id)
        running = False
        exit_code = None
        stage = "idle"
        if job:
            exit_code = job.process.poll()
            if exit_code is None:
                running = True
            else:
                job.exit_code = exit_code
                job.ended_at = job.ended_at or time.time()
                if job.log_handle:
                    job.log_handle.close()
                    job.log_handle = None
        log_path = ""
        if job:
            try:
                log_path = str(job.log_path.relative_to(book_dir))
            except ValueError:
                log_path = str(job.log_path)
        if running:
            if job and job.install_ffmpeg and shutil.which("ffmpeg") is None:
                stage = "installing"
            else:
                stage = "merging"
        elif output_path.exists() or output_parts:
            stage = "done"
        elif exit_code is not None and exit_code != 0:
            stage = "failed"
        payload = {
            "book_id": book_id,
            "running": running,
            "exit_code": exit_code,
            "output_path": str(output_path),
            "output_exists": output_path.exists() or bool(output_parts),
            "output_parts": [path.name for path in output_parts],
            "log_path": log_path,
            "stage": stage,
            "progress": _merge_progress_data(tts_dir),
        }
        return _no_store(payload)

    @app.post("/api/merge/start")
    def merge_start(payload: MergeRequest) -> JSONResponse:
        nonlocal ffmpeg_job
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        output_path = _merge_output_path(book_dir)

        existing = merge_jobs.get(payload.book_id)
        if existing and existing.process.poll() is None:
            raise HTTPException(status_code=409, detail="Merge is already running.")

        if not _merge_ready(book_dir):
            raise HTTPException(
                status_code=409,
                detail="No synthesized chunks yet. Generate at least one chunk first.",
            )

        if (output_path.exists() or _list_m4b_parts(book_dir)) and not payload.overwrite:
            raise HTTPException(status_code=409, detail="Output file already exists.")

        if ffmpeg_job and ffmpeg_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="ffmpeg install in progress.")

        tts_dir = book_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        log_path = tts_dir / "merge.log"
        progress_path = tts_dir / "merge.progress.json"
        _clear_merge_progress(tts_dir)
        log_handle = log_path.open("w", encoding="utf-8")

        install_ffmpeg = shutil.which("ffmpeg") is None
        try:
            if payload.overwrite:
                _delete_m4b_outputs(book_dir)
            cmd = _build_merge_command(
                book_dir=book_dir,
                output_path=output_path,
                overwrite=payload.overwrite,
                install_ffmpeg=install_ffmpeg,
                progress_path=progress_path,
            )
        except RuntimeError as exc:
            log_handle.write(f"{exc}\n")
            log_handle.close()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        merge_jobs[payload.book_id] = MergeJob(
            book_id=payload.book_id,
            book_dir=book_dir,
            process=process,
            started_at=time.time(),
            log_path=log_path,
            output_path=output_path,
            install_ffmpeg=install_ffmpeg,
            progress_path=progress_path,
            log_handle=log_handle,
        )

        return _no_store(
            {
                "status": "started",
                "book_id": payload.book_id,
                "output_path": str(output_path),
            }
        )

    @app.get("/api/m4b/download")
    def download_m4b(book_id: str, part: Optional[int] = None) -> FileResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        output_path = _merge_output_path(book_dir)
        if part is not None:
            output_parts = _list_m4b_parts(book_dir)
            if part < 1 or part > len(output_parts):
                raise HTTPException(status_code=404, detail="M4B part not found.")
            output_path = output_parts[part - 1]
        elif not output_path.exists():
            if _list_m4b_parts(book_dir):
                raise HTTPException(
                    status_code=409, detail="M4B is split; request a part."
                )
            raise HTTPException(status_code=404, detail="M4B not found.")
        return FileResponse(
            path=str(output_path),
            media_type="audio/x-m4b",
            filename=output_path.name,
        )

    return app


def run(root_dir: Path, host: str, port: int) -> None:
    import uvicorn

    app = create_app(root_dir=root_dir)
    uvicorn.run(app, host=host, port=port)
