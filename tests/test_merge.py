import json
import wave
from pathlib import Path

import pytest

from duk import merge


def test_parse_timecode_ms() -> None:
    assert merge._parse_timecode("01:02:03.5") == 3723500


def test_metadata_tags_from_metadata() -> None:
    tags = merge._metadata_tags(
        {
            "title": "Book",
            "authors": ["Alice", "Bob"],
            "year": "2020",
            "language": "zh",
        }
    )
    assert tags["title"] == "Book"
    assert tags["album"] == "Book"
    assert tags["artist"] == "Alice, Bob"
    assert tags["album_artist"] == "Alice, Bob"
    assert tags["date"] == "2020"
    assert tags["language"] == "zh"


def test_plan_chapter_splits_balances_parts() -> None:
    chapters = [
        {"title": "a", "duration_ms": 1000, "segments": []},
        {"title": "b", "duration_ms": 2000, "segments": []},
        {"title": "c", "duration_ms": 3000, "segments": []},
        {"title": "d", "duration_ms": 4000, "segments": []},
    ]
    parts = merge._plan_chapter_splits(chapters, 2)
    assert len(parts) == 2
    assert [p["title"] for p in parts[0]] == ["a", "b"]
    assert [p["title"] for p in parts[1]] == ["c", "d"]


def test_auto_split_count() -> None:
    eight_hours_ms = int(8 * 3600 * 1000)
    assert merge._auto_split_count(eight_hours_ms, 8.0) == 1
    assert merge._auto_split_count(eight_hours_ms + 1, 8.0) == 2


def _write_wav(path: Path, duration_ms: int = 120, rate: int = 24_000) -> None:
    frames = int(rate * duration_ms / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)


def _write_manifest(tts_dir: Path, chapters: list[dict]) -> None:
    payload = {"chapters": chapters}
    tts_dir.mkdir(parents=True, exist_ok=True)
    (tts_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_chapter_segments_stops_at_first_missing_chunk(tmp_path: Path) -> None:
    tts_dir = tmp_path / "tts"
    seg_root = tts_dir / "segments"
    _write_manifest(
        tts_dir,
        chapters=[
            {"id": "c1", "title": "Chapter 1", "chunks": ["a", "b", "c"]},
            {"id": "c2", "title": "Chapter 2", "chunks": ["d"]},
        ],
    )
    _write_wav(seg_root / "c1" / "000001.wav", duration_ms=120)
    _write_wav(seg_root / "c1" / "000003.wav", duration_ms=120)
    _write_wav(seg_root / "c2" / "000001.wav", duration_ms=120)

    chapters, total_ms = merge._load_chapter_segments(tts_dir)

    assert len(chapters) == 1
    assert chapters[0]["title"] == "Chapter 1"
    assert [path.name for path in chapters[0]["segments"]] == ["000001.wav"]
    assert total_ms == 120


def test_load_chapter_segments_raises_when_no_prefix_is_available(
    tmp_path: Path,
) -> None:
    tts_dir = tmp_path / "tts"
    seg_root = tts_dir / "segments"
    _write_manifest(
        tts_dir,
        chapters=[{"id": "c1", "title": "Chapter 1", "chunks": ["a", "b"]}],
    )
    _write_wav(seg_root / "c1" / "000002.wav", duration_ms=120)

    with pytest.raises(FileNotFoundError, match="No synthesized segments available"):
        merge._load_chapter_segments(tts_dir)


def test_load_chapter_segments_skips_missing_zero_duration_chunks(
    tmp_path: Path,
) -> None:
    tts_dir = tmp_path / "tts"
    seg_root = tts_dir / "segments"
    _write_manifest(
        tts_dir,
        chapters=[
            {
                "id": "c1",
                "title": "Chapter 1",
                "chunks": ["a", "b", "c"],
                "durations_ms": [120, 0, 120],
            },
            {"id": "c2", "title": "Chapter 2", "chunks": ["d"], "durations_ms": [120]},
        ],
    )
    _write_wav(seg_root / "c1" / "000001.wav", duration_ms=120)
    _write_wav(seg_root / "c1" / "000003.wav", duration_ms=120)
    _write_wav(seg_root / "c2" / "000001.wav", duration_ms=120)

    chapters, total_ms = merge._load_chapter_segments(tts_dir)

    assert [entry["title"] for entry in chapters] == ["Chapter 1", "Chapter 2"]
    assert [path.name for path in chapters[0]["segments"]] == ["000001.wav", "000003.wav"]
    assert chapters[0]["duration_ms"] == 240
    assert chapters[1]["duration_ms"] == 120
    assert total_ms == 360
