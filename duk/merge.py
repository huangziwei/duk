from __future__ import annotations

import json
import os
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import List, Sequence, Tuple


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH.")


def _cover_average_rgb(cover_path: Path) -> tuple[int, int, int] | None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(cover_path),
        "-vf",
        "scale=1:1:flags=area",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    data = proc.stdout
    if len(data) < 3:
        return None
    r, g, b = data[0], data[1], data[2]
    return int(r), int(g), int(b)


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _mix_channel(base: int, target: int, amount: float) -> int:
    return _clamp_channel(base + (target - base) * amount)


def _mix_color(
    base: tuple[int, int, int],
    target: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    return (
        _mix_channel(base[0], target[0], amount),
        _mix_channel(base[1], target[1], amount),
        _mix_channel(base[2], target[2], amount),
    )


def _cover_gradient_colors(
    cover_path: Path,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    avg = _cover_average_rgb(cover_path)
    if not avg:
        return None
    light = _mix_color(avg, (255, 255, 255), 0.22)
    dark = _mix_color(avg, (0, 0, 0), 0.18)
    return light, dark


def _build_ffmpeg_cmd(
    concat_path: Path,
    chapters_path: Path,
    output_path: Path,
    bitrate: str,
    overwrite: bool,
    cover_path: Path | None = None,
    cover_gradient: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    progress: bool = False,
) -> list[str]:
    cmd = ["ffmpeg"]
    cmd.append("-y" if overwrite else "-n")
    cmd += ["-f", "concat", "-safe", "0", "-i", str(concat_path)]
    cmd += ["-i", str(chapters_path)]
    if cover_path is not None:
        cmd += ["-i", str(cover_path)]
    cmd += ["-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1"]
    if cover_path is not None:
        if cover_gradient:
            light, dark = cover_gradient
            r_expr = f"{light[0]}+({dark[0]}-{light[0]})*(X+Y)/(W+H)"
            g_expr = f"{light[1]}+({dark[1]}-{light[1]})*(X+Y)/(W+H)"
            b_expr = f"{light[2]}+({dark[2]}-{light[2]})*(X+Y)/(W+H)"
            filter_graph = (
                "[2:v]split=2[cover][bg0];"
                "[bg0]scale=if(gt(iw\\,ih)\\,iw\\,ih):"
                "if(gt(iw\\,ih)\\,iw\\,ih),"
                "format=rgb24,"
                f"geq=r={r_expr}:g={g_expr}:b={b_expr}[bg];"
                "[cover]scale=if(gt(iw\\,ih)\\,iw\\,ih):"
                "if(gt(iw\\,ih)\\,iw\\,ih):"
                "force_original_aspect_ratio=decrease[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
            )
        else:
            filter_graph = (
                "[2:v]split=2[cover][bg0];"
                "[bg0]scale=if(gt(iw\\,ih)\\,iw\\,ih):"
                "if(gt(iw\\,ih)\\,iw\\,ih),"
                "boxblur=20:1[bg];"
                "[cover]scale=if(gt(iw\\,ih)\\,iw\\,ih):"
                "if(gt(iw\\,ih)\\,iw\\,ih):"
                "force_original_aspect_ratio=decrease[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
            )
        cmd += [
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-c:v",
            "mjpeg",
            "-disposition:v:0",
            "attached_pic",
            "-metadata:s:v",
            "title=Cover",
            "-metadata:s:v",
            "comment=Cover (front)",
        ]
    if progress:
        cmd += ["-progress", "pipe:1", "-nostats"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-ac", "1", str(output_path)]
    return cmd


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _ffmeta_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\\n")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
    )


def _load_book_metadata(book_dir: Path) -> dict:
    for path in (book_dir / "clean" / "toc.json", book_dir / "toc.json"):
        if path.exists():
            data = _load_json(path)
            meta = data.get("metadata")
            if isinstance(meta, dict):
                return meta
    return {}


def _resolve_cover_path(book_dir: Path, metadata: dict) -> Path | None:
    cover = metadata.get("cover")
    if not isinstance(cover, dict):
        return None
    path = cover.get("path")
    if not path:
        return None
    cover_path = (book_dir / path).resolve()
    if cover_path.exists():
        return cover_path
    return None


def _metadata_tags(metadata: dict) -> dict:
    tags: dict[str, str] = {}
    title = str(metadata.get("title") or "").strip()
    if title:
        tags["title"] = title
        tags["album"] = title
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    if isinstance(authors, list):
        cleaned = [str(a).strip() for a in authors if str(a).strip()]
        if cleaned:
            author_text = ", ".join(cleaned)
            tags["artist"] = author_text
            tags["album_artist"] = author_text
    year = str(metadata.get("year") or "").strip()
    if year:
        tags["date"] = year
    language = str(metadata.get("language") or "").strip()
    if language:
        tags["language"] = language
    return tags


def _parse_timecode(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3:
        return 0
    hours, minutes, seconds = parts
    try:
        secs = float(seconds)
        total = (int(hours) * 3600.0) + (int(minutes) * 60.0) + secs
    except ValueError:
        return 0
    return int(round(total * 1000.0))


def _write_progress(
    path: Path,
    stage: str,
    out_time_ms: int,
    total_ms: int,
    part_index: int | None = None,
    part_total: int | None = None,
) -> None:
    percent = 0.0
    if total_ms > 0:
        percent = min(100.0, (out_time_ms / total_ms) * 100.0)
    payload = {
        "stage": stage,
        "out_time_ms": int(out_time_ms),
        "total_ms": int(total_ms),
        "percent": round(percent, 2),
    }
    if part_index is not None:
        payload["part_index"] = int(part_index)
    if part_total is not None:
        payload["part_total"] = int(part_total)
    _atomic_write_json(path, payload)


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate <= 0:
        return 0
    return int(round(frames * 1000.0 / rate))


def _resolve_chunk_count(entry: dict, chapter_dir: Path) -> int:
    for key in ("durations_ms", "chunks", "chunk_spans"):
        value = entry.get(key)
        if isinstance(value, list) and value:
            return len(value)
    if chapter_dir.exists():
        return len(sorted(chapter_dir.glob("*.wav")))
    return 0


def _duration_hint_ms(durations: Sequence[object], idx: int) -> int | None:
    pos = idx - 1
    if pos < 0 or pos >= len(durations):
        return None
    candidate = durations[pos]
    if isinstance(candidate, (int, float)):
        return int(candidate)
    return None


def _build_concat_file(
    segment_paths: List[Path], concat_path: Path, base_dir: Path
) -> None:
    lines = []
    for path in segment_paths:
        rel = Path(os.path.relpath(path, start=base_dir)).as_posix()
        lines.append(f"file '{rel}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_chapters_ffmeta(
    chapters: Sequence[Tuple[str, int]],
    ffmeta_path: Path,
    metadata: dict | None = None,
) -> None:
    out = [";FFMETADATA1"]
    if metadata:
        for key, value in _metadata_tags(metadata).items():
            out.append(f"{key}={_ffmeta_escape(value)}")
    t = 0
    for title, duration in chapters:
        start = t
        end = t + max(int(duration), 1)
        out.append("")
        out.append("[CHAPTER]")
        out.append("TIMEBASE=1/1000")
        out.append(f"START={start}")
        out.append(f"END={end}")
        out.append(f"title={title}")
        t = end
    ffmeta_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _ensure_merge_inputs(tts_dir: Path, metadata: dict) -> tuple[Path, Path, int]:
    concat_path = tts_dir / "concat.txt"
    chapters_path = tts_dir / "chapters.ffmeta"
    manifest_path = tts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    chapters = manifest.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("manifest.json contains no chapters.")

    seg_root = tts_dir / "segments"
    segment_paths: List[Path] = []
    chapter_meta: List[Tuple[str, int]] = []
    missing: List[Path] = []
    total_ms = 0

    for entry in chapters:
        if not isinstance(entry, dict):
            continue
        chapter_id = entry.get("id") or "chapter"
        title = entry.get("title") or chapter_id
        chapter_dir = seg_root / chapter_id
        chunk_count = _resolve_chunk_count(entry, chapter_dir)
        if chunk_count <= 0:
            raise ValueError(f"No chunks found for chapter: {chapter_id}")

        durations = entry.get("durations_ms")
        durations_list = durations if isinstance(durations, list) else []
        chapter_total_ms = 0

        for idx in range(1, chunk_count + 1):
            seg_path = chapter_dir / f"{idx:06d}.wav"
            if not seg_path.exists():
                duration_hint = _duration_hint_ms(durations_list, idx)
                if duration_hint is not None and duration_hint <= 0:
                    # Some normalization paths intentionally emit zero-duration
                    # chunks without producing a WAV segment file.
                    continue
                missing.append(seg_path)
                continue
            segment_paths.append(seg_path)
            duration_ms = None
            duration_hint = _duration_hint_ms(durations_list, idx)
            if duration_hint is not None and duration_hint > 0:
                duration_ms = duration_hint
            if duration_ms is None:
                duration_ms = _wav_duration_ms(seg_path)
            chapter_total_ms += duration_ms

        chapter_meta.append((title, chapter_total_ms))
        total_ms += chapter_total_ms

    if missing:
        sample = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} segment(s). Sample:\n{sample}"
        )

    _build_concat_file(segment_paths, concat_path, base_dir=tts_dir)
    _build_chapters_ffmeta(chapter_meta, chapters_path, metadata=metadata)
    return concat_path, chapters_path, total_ms


def _load_chapter_segments(tts_dir: Path) -> tuple[List[dict], int]:
    manifest_path = tts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    chapters = manifest.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("manifest.json contains no chapters.")

    seg_root = tts_dir / "segments"
    chapter_segments: List[dict] = []
    first_missing: Path | None = None
    total_ms = 0

    for entry in chapters:
        if not isinstance(entry, dict):
            continue
        chapter_id = entry.get("id") or "chapter"
        title = entry.get("title") or chapter_id
        chapter_dir = seg_root / chapter_id
        chunk_count = _resolve_chunk_count(entry, chapter_dir)
        if chunk_count <= 0:
            raise ValueError(f"No chunks found for chapter: {chapter_id}")

        durations = entry.get("durations_ms")
        durations_list = durations if isinstance(durations, list) else []
        chapter_total_ms = 0
        segments: List[Path] = []

        for idx in range(1, chunk_count + 1):
            seg_path = chapter_dir / f"{idx:06d}.wav"
            if not seg_path.exists():
                duration_hint = _duration_hint_ms(durations_list, idx)
                if duration_hint is not None and duration_hint <= 0:
                    # Skip intentionally silent/empty chunks.
                    continue
                first_missing = seg_path
                break
            segments.append(seg_path)
            duration_ms = None
            duration_hint = _duration_hint_ms(durations_list, idx)
            if duration_hint is not None and duration_hint > 0:
                duration_ms = duration_hint
            if duration_ms is None:
                duration_ms = _wav_duration_ms(seg_path)
            chapter_total_ms += duration_ms

        if segments:
            chapter_segments.append(
                {
                    "title": str(title),
                    "duration_ms": int(chapter_total_ms),
                    "segments": segments,
                }
            )
            total_ms += chapter_total_ms
        if first_missing is not None:
            break

    if total_ms <= 0:
        hint = f" First missing segment: {first_missing}." if first_missing else ""
        raise FileNotFoundError(f"No synthesized segments available for merge.{hint}")
    if first_missing is not None:
        print(
            f"warning: partial merge mode; stopped at first missing segment: {first_missing}",
            file=sys.stderr,
        )
    return chapter_segments, int(total_ms)


def _plan_chapter_splits(chapters: Sequence[dict], parts: int) -> List[List[dict]]:
    if parts <= 1 or not chapters:
        return [list(chapters)]
    total_ms = sum(int(entry.get("duration_ms") or 0) for entry in chapters)
    remaining_ms = total_ms
    remaining_parts = parts
    planned: List[List[dict]] = []
    current: List[dict] = []
    current_ms = 0

    for idx, entry in enumerate(chapters):
        if remaining_parts == 1:
            current.append(entry)
            current_ms += int(entry.get("duration_ms") or 0)
            continue

        remaining_chapters = len(chapters) - idx
        target_ms = remaining_ms / remaining_parts if remaining_parts else remaining_ms

        if current and remaining_chapters == remaining_parts:
            planned.append(current)
            remaining_ms -= current_ms
            remaining_parts -= 1
            current = []
            current_ms = 0
            target_ms = remaining_ms / remaining_parts if remaining_parts else remaining_ms

        duration_ms = int(entry.get("duration_ms") or 0)
        if (
            current
            and current_ms + duration_ms > target_ms
            and remaining_chapters >= remaining_parts
        ):
            planned.append(current)
            remaining_ms -= current_ms
            remaining_parts -= 1
            current = []
            current_ms = 0

        current.append(entry)
        current_ms += duration_ms

    if current:
        planned.append(current)
    return planned


def _part_output_path(output_path: Path, idx: int, total: int) -> Path:
    stem = output_path.stem
    suffix = output_path.suffix or ".m4b"
    width = max(2, len(str(total)))
    return output_path.with_name(f"{stem}.part{idx:0{width}d}{suffix}")


def _part_progress_path(progress_path: Path, idx: int, total: int) -> Path:
    stem = progress_path.stem
    suffix = progress_path.suffix or ".json"
    width = max(2, len(str(total)))
    return progress_path.with_name(f"{stem}.part{idx:0{width}d}{suffix}")


def _auto_split_count(total_ms: int, hours: float) -> int:
    if total_ms <= 0:
        return 1
    target_ms = hours * 3600 * 1000
    if target_ms <= 0:
        return 1
    if total_ms <= target_ms:
        return 1
    return int(math.ceil(total_ms / target_ms))


def _build_part_inputs(
    chapters: Sequence[dict],
    out_dir: Path,
    tts_dir: Path,
    metadata: dict,
) -> tuple[Path, Path, int]:
    segment_paths: List[Path] = []
    chapter_meta: List[Tuple[str, int]] = []
    total_ms = 0
    for entry in chapters:
        segment_paths.extend(entry.get("segments") or [])
        duration_ms = int(entry.get("duration_ms") or 0)
        chapter_meta.append((str(entry.get("title") or "Chapter"), duration_ms))
        total_ms += duration_ms

    out_dir.mkdir(parents=True, exist_ok=True)
    concat_path = out_dir / "concat.txt"
    chapters_path = out_dir / "chapters.ffmeta"
    _build_concat_file(segment_paths, concat_path, base_dir=out_dir)
    _build_chapters_ffmeta(chapter_meta, chapters_path, metadata=metadata)
    return concat_path, chapters_path, total_ms


def _run_ffmpeg_with_progress(
    cmd: list[str],
    progress_path: Path,
    total_ms: int,
    part_index: int | None = None,
    part_total: int | None = None,
) -> int:
    _write_progress(
        progress_path,
        "merging",
        0,
        total_ms,
        part_index=part_index,
        part_total=part_total,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    out_time_ms = 0
    pending_time_ms = None
    if proc.stdout:
        for line in proc.stdout:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "out_time":
                pending_time_ms = _parse_timecode(value)
            elif key == "out_time_ms":
                if pending_time_ms is None:
                    try:
                        pending_time_ms = int(value)
                    except ValueError:
                        pending_time_ms = None
            elif key == "progress":
                if pending_time_ms is not None:
                    if total_ms > 0 and pending_time_ms > total_ms * 100:
                        out_time_ms = int(pending_time_ms / 1000)
                    else:
                        out_time_ms = pending_time_ms
                    pending_time_ms = None
                if value == "end":
                    out_time_ms = total_ms or out_time_ms
                    _write_progress(
                        progress_path,
                        "done",
                        out_time_ms,
                        total_ms,
                        part_index=part_index,
                        part_total=part_total,
                    )
                elif value == "continue":
                    _write_progress(
                        progress_path,
                        "merging",
                        out_time_ms,
                        total_ms,
                        part_index=part_index,
                        part_total=part_total,
                    )
    proc.wait()
    if proc.returncode != 0:
        _write_progress(
            progress_path,
            "failed",
            out_time_ms,
            total_ms,
            part_index=part_index,
            part_total=part_total,
        )
    return int(proc.returncode)


def merge_book(
    book_dir: Path,
    output_path: Path,
    bitrate: str = "64k",
    overwrite: bool = False,
    progress_path: Path | None = None,
    split_hours: float | None = None,
    split_count: int | None = None,
) -> int:
    book_dir = book_dir.resolve()
    output_path = output_path.resolve()
    tts_dir = (book_dir / "tts").resolve()
    metadata = _load_book_metadata(book_dir)
    cover_path = _resolve_cover_path(book_dir, metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_ffmpeg()
    cover_gradient = _cover_gradient_colors(cover_path) if cover_path else None

    if split_hours is not None and split_hours <= 0:
        raise ValueError("split_hours must be positive.")
    if split_count is not None and split_count <= 0:
        raise ValueError("split_count must be positive.")
    if split_hours is not None and split_count is not None:
        raise ValueError("Use split_hours or split_count, not both.")

    chapters, total_ms = _load_chapter_segments(tts_dir)
    if split_hours is None and split_count is None:
        split_count = _auto_split_count(total_ms, AUTO_SPLIT_HOURS)
    elif split_hours is not None:
        split_count = _auto_split_count(total_ms, split_hours)

    if not split_count or split_count <= 1:
        concat_path, chapters_path, total_ms = _build_part_inputs(
            chapters, out_dir=tts_dir, tts_dir=tts_dir, metadata=metadata
        )
        cmd = _build_ffmpeg_cmd(
            concat_path=concat_path,
            chapters_path=chapters_path,
            output_path=output_path,
            bitrate=bitrate,
            overwrite=overwrite,
            cover_path=cover_path,
            cover_gradient=cover_gradient,
            progress=progress_path is not None,
        )
        if progress_path is not None:
            return _run_ffmpeg_with_progress(cmd, progress_path, total_ms)
        proc = subprocess.run(cmd, cwd=str(tts_dir))
        return int(proc.returncode)

    parts = _plan_chapter_splits(chapters, split_count)
    parts_dir = tts_dir / "parts"

    for idx, part in enumerate(parts, start=1):
        concat_path, chapters_path, part_ms = _build_part_inputs(
            part, out_dir=parts_dir / f"part{idx:02d}", tts_dir=tts_dir, metadata=metadata
        )
        part_output = _part_output_path(output_path, idx, len(parts))
        part_progress = (
            _part_progress_path(progress_path, idx, len(parts))
            if progress_path is not None
            else None
        )
        cmd = _build_ffmpeg_cmd(
            concat_path=concat_path,
            chapters_path=chapters_path,
            output_path=part_output,
            bitrate=bitrate,
            overwrite=overwrite,
            cover_path=cover_path,
            cover_gradient=cover_gradient,
            progress=part_progress is not None,
        )
        if part_progress is not None:
            code = _run_ffmpeg_with_progress(
                cmd,
                part_progress,
                part_ms,
                part_index=idx,
                part_total=len(parts),
            )
        else:
            proc = subprocess.run(cmd, cwd=str(tts_dir))
            code = int(proc.returncode)
        if code != 0:
            return code
    return 0
AUTO_SPLIT_HOURS = 10.0
