from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from . import audio_norm as audio_norm_util
from . import asr as asr_util
from . import epub as epub_util
from . import merge as merge_util
from . import sanitize as sanitize_util
from . import tts as tts_util
from . import voice as voice_util
from .text import guess_title_from_path


def _ingest_epub(input_path: Path, out_dir: Path, raw_dir: Path) -> int:
    book = epub_util.read_epub(input_path)
    metadata = epub_util.extract_metadata(book)
    cover = epub_util.extract_cover_image(book)
    if cover:
        cover_path = _write_cover_image(cover, out_dir)
        if cover_path:
            cover_info = metadata.get("cover") or {}
            cover_info.setdefault("id", cover.get("id", ""))
            cover_info.setdefault("href", cover.get("href", ""))
            cover_info["path"] = cover_path.relative_to(out_dir).as_posix()
            cover_info["media_type"] = cover.get("media_type") or cover_info.get(
                "media_type", ""
            )
            metadata["cover"] = cover_info
    chapters = epub_util.extract_chapters(book, prefer_toc=True)

    if not chapters:
        sys.stderr.write("No chapters found in EPUB.\n")
        return 2

    toc_items = []
    for idx, chapter in enumerate(chapters, start=1):
        title = chapter.title or f"Chapter {idx}"
        slug = epub_util.slugify(title)
        filename = f"{idx:04d}-{slug}.txt"
        out_path = raw_dir / filename

        out_path.write_text(chapter.text.rstrip() + "\n", encoding="utf-8")
        toc_items.append(
            {
                "index": idx,
                "title": title,
                "href": chapter.href,
                "source": chapter.source,
                "headings": chapter.headings,
                "heading_categories": chapter.heading_categories,
                "path": out_path.relative_to(out_dir).as_posix(),
            }
        )

    toc_data = {
        "created_unix": int(time.time()),
        "source_epub": str(input_path),
        "metadata": metadata,
        "chapters": toc_items,
    }

    toc_path = out_dir / "toc.json"
    toc_path.write_text(
        json.dumps(toc_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(toc_items)} chapters to {raw_dir}")
    print(f"TOC metadata saved to {toc_path}")
    return 0


def _ingest_txt(input_path: Path, out_dir: Path, raw_dir: Path) -> int:
    try:
        text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = input_path.read_text(encoding="utf-8", errors="replace")
    title = guess_title_from_path(input_path)
    chapter_title = title or "Chapter 1"
    slug = epub_util.slugify(chapter_title)
    filename = f"0001-{slug}.txt"
    out_path = raw_dir / filename
    out_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    toc_items = [
        {
            "index": 1,
            "title": chapter_title,
            "href": "",
            "source": "",
            "path": out_path.relative_to(out_dir).as_posix(),
        }
    ]
    toc_data = {
        "created_unix": int(time.time()),
        "source_epub": str(input_path),
        "metadata": {
            "title": title,
            "authors": [],
            "language": "",
            "dates": [],
            "year": "",
            "cover": None,
        },
        "chapters": toc_items,
    }
    toc_path = out_dir / "toc.json"
    toc_path.write_text(
        json.dumps(toc_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(toc_items)} chapters to {raw_dir}")
    print(f"TOC metadata saved to {toc_path}")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Input file not found: {input_path}\n")
        return 2

    suffix = input_path.suffix.lower()
    if suffix not in {".epub", ".txt"}:
        sys.stderr.write("Only .epub or .txt files are supported.\n")
        return 2

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw" / "chapters"

    if raw_dir.exists():
        existing = [p for p in raw_dir.iterdir() if p.is_file()]
        if existing and not args.overwrite:
            sys.stderr.write(
                "Raw chapters already exist. Use --overwrite to regenerate.\n"
            )
            return 2

    raw_dir.mkdir(parents=True, exist_ok=True)
    if suffix == ".txt":
        return _ingest_txt(input_path, out_dir, raw_dir)
    return _ingest_epub(input_path, out_dir, raw_dir)


def _cover_extension(media_type: str, href: str) -> str:
    media_type = (media_type or "").lower()
    if media_type == "image/jpeg":
        return ".jpg"
    if media_type == "image/png":
        return ".png"
    if media_type == "image/webp":
        return ".webp"
    if media_type == "image/gif":
        return ".gif"
    if href:
        suffix = Path(href).suffix.lower()
        if suffix:
            return suffix
    return ".jpg"


def _write_cover_image(cover: dict, out_dir: Path) -> Optional[Path]:
    data = cover.get("data")
    if not data:
        return None
    ext = _cover_extension(cover.get("media_type", ""), cover.get("href", ""))
    path = out_dir / f"cover{ext}"
    path.write_bytes(data)
    return path


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_to(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "duk voice clone"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


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


def _clone(args: argparse.Namespace) -> int:
    source = str(args.source)
    repo_root = voice_util.find_repo_root(Path.cwd())
    voices_dir = repo_root / voice_util.VOICE_DIRNAME
    voices_dir.mkdir(parents=True, exist_ok=True)
    start = args.start or "00:00:00"
    duration_value = float(args.duration or 0)
    if duration_value <= 0:
        sys.stderr.write("--duration must be a positive number of seconds.\n")
        return 2
    duration = str(duration_value)

    if shutil.which("ffmpeg") is None:
        sys.stderr.write("ffmpeg not found on PATH.\n")
        return 2

    voice_text = None
    if args.text_file:
        voice_text = Path(args.text_file).read_text(encoding="utf-8").strip()
    elif args.text:
        voice_text = str(args.text).strip()

    if not voice_text and not args.auto_text:
        sys.stderr.write(
            "Reference text is required. Provide --text/--text-file or enable "
            "Whisper auto-text.\n"
        )
        return 2

    output_name = voice_util.coerce_voice_name(args.name, Path(source).stem)
    output_path = voices_dir / f"{output_name}.wav"
    config_path = voices_dir / f"{output_name}.json"

    if _is_http_url(source):
        parsed = urllib.parse.urlparse(source)
        filename = Path(parsed.path).name or "voice.mp3"
        if not Path(filename).suffix:
            filename = f"{filename}.mp3"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / filename
            try:
                _download_to(source, input_path)
            except Exception as exc:
                sys.stderr.write(f"Download failed: {exc}\n")
                return 2
            cmd = _build_clone_ffmpeg_cmd(input_path, output_path, start, duration)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.stderr.write("ffmpeg failed to process the audio.\n")
                return 2
    else:
        input_path = Path(source).expanduser()
        if not input_path.exists():
            sys.stderr.write(f"Input file not found: {input_path}\n")
            return 2
        cmd = _build_clone_ffmpeg_cmd(input_path, output_path, start, duration)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.stderr.write("ffmpeg failed to process the audio.\n")
            return 2

    try:
        audio_norm_util.normalize_clone_wav(output_path)
    except Exception as exc:
        sys.stderr.write(f"Failed to normalize cloned audio: {exc}\n")
        return 2

    if not voice_text and args.auto_text:
        whisper_language = args.whisper_language or args.language
        try:
            voice_text = asr_util.transcribe_audio(
                output_path,
                model_name=args.whisper_model,
                language=whisper_language,
                device=args.whisper_device,
                initial_prompt=args.whisper_prompt,
            )
        except Exception as exc:
            sys.stderr.write(f"Whisper transcription failed: {exc}\n")
            return 2
        if not voice_text:
            sys.stderr.write(
                "Whisper returned empty text; provide --text/--text-file instead.\n"
            )
            return 2

    if not voice_text:
        sys.stderr.write(
            "Reference text is required. Provide --text/--text-file or enable "
            "Whisper auto-text.\n"
        )
        return 2

    config = voice_util.VoiceConfig(
        name=output_name,
        ref_audio=output_path.name,
        ref_text=voice_text,
        language=args.language or voice_util.DEFAULT_LANGUAGE,
    )
    voice_util.write_voice_config(config, config_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {config_path}")
    return 0


def _sanitize(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    rules_path = Path(args.rules) if args.rules else None
    try:
        written = sanitize_util.sanitize_book(
            book_dir=book_dir,
            rules_path=rules_path,
            overwrite=args.overwrite,
        )
        tts_cleared = sanitize_util.refresh_chunks(book_dir=book_dir)
    except Exception as exc:
        sys.stderr.write(f"Sanitize failed: {exc}\n")
        return 2

    print(f"Wrote {written} cleaned chapters to {book_dir / 'clean' / 'chapters'}")
    print(f"Report saved to {book_dir / 'clean' / 'report.json'}")
    if tts_cleared:
        print("Cleared TTS cache and prepared chunks.")
    else:
        print("Prepared chunks for TTS.")
    return 0


def _rechunk(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    manifest_path = book_dir / "tts" / "manifest.json"
    max_chars = args.max_chars
    min_chars = args.min_chars
    pad_ms = args.pad_ms
    chunk_mode = args.chunk_mode

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if max_chars is None:
            max_chars = manifest.get("max_chars")
        if min_chars is None:
            min_chars = manifest.get("min_chars")
        if pad_ms is None:
            pad_ms = manifest.get("pad_ms")
        if chunk_mode is None:
            chunk_mode = manifest.get("chunk_mode")

    if max_chars is None:
        max_chars = 100
    if min_chars is None:
        min_chars = 15
    if pad_ms is None:
        pad_ms = 300
    if chunk_mode is None:
        chunk_mode = "chinese"

    try:
        tts_cleared = sanitize_util.refresh_chunks(
            book_dir=book_dir,
            max_chars=int(max_chars),
            pad_ms=int(pad_ms),
            chunk_mode=str(chunk_mode),
            min_chars=int(min_chars),
        )
    except Exception as exc:
        sys.stderr.write(f"Rechunk failed: {exc}\n")
        return 2

    if tts_cleared:
        print("Cleared TTS cache and prepared chunks.")
    else:
        print("Prepared chunks for TTS.")
    return 0


def _merge(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    output_path = Path(args.output)
    try:
        return merge_util.merge_book(
            book_dir=book_dir,
            output_path=output_path,
            bitrate=args.bitrate,
            overwrite=args.overwrite,
            progress_path=Path(args.progress_file) if args.progress_file else None,
            split_hours=args.split_hours,
            split_count=args.split_count,
        )
    except Exception as exc:
        sys.stderr.write(f"Merge failed: {exc}\n")
        return 2


def _play(args: argparse.Namespace) -> int:
    player_util = importlib.import_module("duk.player")
    player_util.run(Path(args.root), host=args.host, port=args.port)
    return 0


def _synth(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    out_dir = Path(args.out) if args.out else None
    voice_text_path = Path(args.voice_text_file) if args.voice_text_file else None
    voice_map_path = Path(args.voice_map) if args.voice_map else None

    try:
        voice_config = voice_util.resolve_voice_config(
            voice=args.voice,
            base_dir=Path.cwd(),
            voice_text=args.voice_text,
            voice_text_path=voice_text_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    return int(tts_util.synthesize_book(
        book_dir=book_dir,
        voice=voice_config,
        out_dir=out_dir,
        max_chars=args.max_chars,
        pad_ms=args.pad_ms,
        rechunk=args.rechunk,
        voice_map_path=voice_map_path,
        base_dir=Path.cwd(),
        min_chars=args.min_chars,
    ))


def _sample(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    out_dir = Path(args.out) if args.out else None
    voice_text_path = Path(args.voice_text_file) if args.voice_text_file else None
    voice_map_path = Path(args.voice_map) if args.voice_map else None

    try:
        voice_config = voice_util.resolve_voice_config(
            voice=args.voice,
            base_dir=Path.cwd(),
            voice_text=args.voice_text,
            voice_text_path=voice_text_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    return int(tts_util.synthesize_book_sample(
        book_dir=book_dir,
        voice=voice_config,
        out_dir=out_dir,
        max_chars=args.max_chars,
        pad_ms=args.pad_ms,
        rechunk=args.rechunk,
        voice_map_path=voice_map_path,
        base_dir=Path.cwd(),
        min_chars=args.min_chars,
    ))


def _jsonl_entries(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def _as_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_values: list[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def _series_stats(values: list[float]) -> Optional[dict]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "min": round(ordered[0], 3),
        "p10": round(_percentile(ordered, 0.10) or ordered[0], 3),
        "p25": round(_percentile(ordered, 0.25) or ordered[0], 3),
        "p50": round(_percentile(ordered, 0.50) or ordered[0], 3),
        "p75": round(_percentile(ordered, 0.75) or ordered[-1], 3),
        "p90": round(_percentile(ordered, 0.90) or ordered[-1], 3),
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / n, 3),
    }


def _boundary_report_payload(entries: list[dict], min_samples: int = 200) -> dict:
    filtered: list[dict] = []
    delta_values: list[float] = []
    output_latency_values: list[float] = []
    corrected_values: list[float] = []

    for raw in entries:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("trigger") or "") != "gapless":
            continue
        if raw.get("preloaded") is not True:
            continue
        playback_rate = _as_float(raw.get("playback_rate"))
        if playback_rate is None or not (0.95 <= playback_rate <= 1.05):
            continue
        delta_ms = _as_float(raw.get("delta_ms"))
        if delta_ms is None or not (-5000.0 <= delta_ms <= 5000.0):
            continue
        filtered.append(raw)
        delta_values.append(delta_ms)
        output_latency_ms = _as_float(raw.get("output_latency_ms"))
        if output_latency_ms is None or not (0.0 <= output_latency_ms <= 5000.0):
            continue
        output_latency_values.append(output_latency_ms)
        corrected_values.append(delta_ms + output_latency_ms)

    filtered_count = len(filtered)
    corrected_count = len(corrected_values)
    coverage = corrected_count / filtered_count if filtered_count > 0 else 0.0
    pad_values = sorted(
        {
            value
            for value in (_as_int(entry.get("pad_ms")) for entry in filtered)
            if value is not None
        }
    )

    recommendation = {
        "recommended_pad_adjust_ms": None,
        "basis": "insufficient_data",
        "reason": "",
    }
    if filtered_count < min_samples:
        recommendation["reason"] = (
            f"Need at least {min_samples} filtered boundaries; found {filtered_count}."
        )
    elif coverage < 0.8:
        recommendation["reason"] = (
            "Need output latency on >=80% of filtered boundaries; "
            f"current coverage is {round(coverage * 100.0, 1)}%."
        )
    else:
        corrected_stats = _series_stats(corrected_values)
        if corrected_stats:
            recommendation["recommended_pad_adjust_ms"] = int(
                round(corrected_stats["p50"])
            )
            recommendation["basis"] = "p50(delta_ms + output_latency_ms)"
            recommendation["reason"] = (
                "Positive means increase pad_ms to make merged M4B pauses closer "
                "to player pauses."
            )

    return {
        "samples_total": len(entries),
        "samples_filtered": filtered_count,
        "samples_with_output_latency": corrected_count,
        "output_latency_coverage": round(coverage, 4),
        "pad_ms_values": pad_values,
        "delta_ms": _series_stats(delta_values),
        "output_latency_ms": _series_stats(output_latency_values),
        "corrected_delta_ms": _series_stats(corrected_values),
        "recommendation": recommendation,
    }


def _boundary_report(args: argparse.Namespace) -> int:
    path = Path(args.log)
    if not path.exists():
        sys.stderr.write(f"Log file not found: {path}\n")
        return 2
    try:
        entries = _jsonl_entries(path)
    except OSError as exc:
        sys.stderr.write(f"Failed to read log file: {exc}\n")
        return 2
    payload = _boundary_report_payload(entries, min_samples=max(1, args.min_samples))
    payload["log_path"] = str(path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="duk")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser(
        "ingest", help="Extract chapters from an EPUB or TXT (Chinese only)"
    )
    ingest.add_argument("--input", required=True, help="Path to input .epub or .txt")
    ingest.add_argument(
        "--out",
        "--output",
        required=True,
        dest="out",
        help="Output book directory (e.g., out/book)",
    )
    ingest.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing raw chapters"
    )
    ingest.set_defaults(func=_ingest)

    sanitize = subparsers.add_parser("sanitize", help="Clean chapter text")
    sanitize.add_argument("--book", required=True, help="Book output directory")
    sanitize.add_argument(
        "--rules",
        help=(
            "Path to JSON rules file (defaults to sanitize-rules.json in the book "
            "directory if present, otherwise duk/templates/sanitize-rules.json)"
        ),
    )
    sanitize.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing cleaned output"
    )
    sanitize.set_defaults(func=_sanitize)

    rechunk = subparsers.add_parser(
        "rechunk",
        help="Rebuild TTS chunks without re-sanitizing",
    )
    rechunk.add_argument("--book", required=True, help="Book output directory")
    rechunk.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Max characters per chunk (default: reuse manifest or 100)",
    )
    rechunk.add_argument(
        "--min-chars",
        type=int,
        default=None,
        help="Min characters per chunk (hard floor the TTS model needs); chunks flush at the sentence level once they reach it, and any shorter chunk merges into a neighbor (default: reuse manifest or 15)",
    )
    rechunk.add_argument(
        "--pad-ms",
        type=int,
        default=None,
        help="Silence padding per chunk in ms (default: reuse manifest or 300)",
    )
    rechunk.add_argument(
        "--chunk-mode",
        choices=("chinese", "sentence"),
        default=None,
        help="Chunking mode (default: reuse manifest or chinese)",
    )
    rechunk.set_defaults(func=_rechunk)

    clone = subparsers.add_parser(
        "clone", help="Create a voice sample from audio + transcript"
    )
    clone.add_argument("source", help="URL or local path to an audio file")
    clone.add_argument(
        "--name",
        help="Output voice name (default: input filename without extension)",
    )
    clone.add_argument(
        "--start",
        default="00:00:00",
        help="Start timestamp (HH:MM:SS or seconds)",
    )
    clone.add_argument(
        "--duration",
        type=float,
        default=10,
        help="Duration in seconds",
    )
    clone.add_argument("--text", help="Reference transcript for the audio clip")
    clone.add_argument("--text-file", dest="text_file", help="Path to transcript file")
    clone.add_argument(
        "--auto-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-transcribe the clip with Whisper when no text is provided",
    )
    clone.add_argument(
        "--whisper-model",
        default=asr_util.DEFAULT_WHISPER_MODEL,
        help="mlx-whisper model repo or alias (default: large-v3-turbo)",
    )
    clone.add_argument(
        "--whisper-language",
        help="Whisper language code/name (default: derived from --language)",
    )
    clone.add_argument(
        "--whisper-device",
        help="Ignored (MLX runs on the unified-memory GPU)",
    )
    clone.add_argument(
        "--whisper-prompt",
        help="Initial prompt to guide Whisper transcription",
    )
    clone.add_argument(
        "--language",
        default=voice_util.DEFAULT_LANGUAGE,
        help="Language label for the voice config (default: Chinese)",
    )
    clone.set_defaults(func=_clone)

    synth = subparsers.add_parser("synth", help="Synthesize audio")
    synth.add_argument("--book", required=True, help="Book output directory")
    synth.add_argument(
        "--out",
        help="Output directory (default: <book>/tts)",
    )
    synth.add_argument(
        "--voice",
        required=True,
        help="Voice config name/path, or a raw audio path with --voice-text",
    )
    synth.add_argument(
        "--voice-map",
        help="Path to voice map JSON for per-chapter voices",
    )
    synth.add_argument(
        "--voice-text",
        help="Reference transcript when providing a raw audio file",
    )
    synth.add_argument(
        "--voice-text-file",
        help="Reference transcript file when providing a raw audio file",
    )
    synth.add_argument("--max-chars", type=int, default=100)
    synth.add_argument("--min-chars", type=int, default=15)
    synth.add_argument("--pad-ms", type=int, default=350)
    synth.add_argument("--rechunk", action="store_true")
    synth.set_defaults(func=_synth)

    sample = subparsers.add_parser(
        "sample", help="Generate a sample (first chapter)"
    )
    sample.add_argument("--book", required=True, help="Book output directory")
    sample.add_argument(
        "--out",
        help="Output directory (default: <book>/tts)",
    )
    sample.add_argument(
        "--voice",
        required=True,
        help="Voice config name/path, or a raw audio path with --voice-text",
    )
    sample.add_argument(
        "--voice-map",
        help="Path to voice map JSON for per-chapter voices",
    )
    sample.add_argument(
        "--voice-text",
        help="Reference transcript when providing a raw audio file",
    )
    sample.add_argument(
        "--voice-text-file",
        help="Reference transcript file when providing a raw audio file",
    )
    sample.add_argument("--max-chars", type=int, default=100)
    sample.add_argument("--min-chars", type=int, default=15)
    sample.add_argument("--pad-ms", type=int, default=350)
    sample.add_argument("--rechunk", action="store_true")
    sample.set_defaults(func=_sample)

    merge = subparsers.add_parser("merge", help="Merge audio into M4B")
    merge.add_argument("--book", required=True, help="Book output directory")
    merge.add_argument("--output", required=True, help="Path to output .m4b")
    merge.add_argument("--bitrate", default="64k", help="Audio bitrate (default: 64k)")
    merge.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if it exists"
    )
    merge.add_argument(
        "--progress-file",
        help="Write merge progress to JSON file",
    )
    merge.add_argument(
        "--split-hours",
        type=float,
        help="Target hours per part; splits at chapter boundaries",
    )
    merge.add_argument(
        "--split-count",
        type=int,
        help="Number of parts to split into; splits at chapter boundaries",
    )
    merge.set_defaults(func=_merge)

    boundary = subparsers.add_parser(
        "boundary-report", help="Analyze player boundary latency logs"
    )
    boundary.add_argument(
        "--log",
        required=True,
        help="Path to player-boundary-latency.jsonl",
    )
    boundary.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum filtered boundary count required for recommendation (default: 200)",
    )
    boundary.set_defaults(func=_boundary_report)

    play = subparsers.add_parser("play", help="Play generated audio in a web UI")
    play.add_argument(
        "--root",
        default="out",
        help="Root folder containing book outputs (default: out)",
    )
    play.add_argument("--host", default="0.0.0.0")
    play.add_argument("--port", type=int, default=1912)
    play.set_defaults(func=_play)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return int(args.func(args))
