from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable


SECTION_BREAK = ""


def _is_symbolic_line_candidate(stripped: str) -> bool:
    if not stripped or SECTION_BREAK in stripped:
        return False
    has_visible = False
    for ch in stripped:
        if ch.isspace():
            continue
        has_visible = True
        category = unicodedata.category(ch)
        if category.startswith(("L", "N", "M")):
            return False
    return has_visible


def _contains_quote_or_bracket(stripped: str) -> bool:
    for ch in stripped:
        category = unicodedata.category(ch)
        if category in {"Ps", "Pe", "Pi", "Pf"}:
            return True
    return False


def _infer_section_break_markers(text: str) -> set[str]:
    lines = text.split("\n")
    if not lines:
        return set()

    candidate_indices: list[int] = []
    candidate_stats: dict[str, list[int]] = {}
    non_candidate_char_counts: dict[str, int] = {}
    stripped_lines = [line.strip() for line in lines]

    for idx, stripped in enumerate(stripped_lines):
        if not stripped:
            continue
        if _is_symbolic_line_candidate(stripped):
            candidate_indices.append(idx)
            stats = candidate_stats.setdefault(stripped, [0, 0])
            stats[0] += 1
            continue
        for ch in stripped:
            if ch.isspace():
                continue
            non_candidate_char_counts[ch] = non_candidate_char_counts.get(ch, 0) + 1

    candidate_index_set = set(candidate_indices)
    idx = 0
    while idx < len(lines):
        if idx not in candidate_index_set:
            idx += 1
            continue
        marker = stripped_lines[idx]
        stats = candidate_stats.get(marker)
        if not stats:
            idx += 1
            continue
        run_end = idx
        while (
            run_end + 1 < len(lines)
            and run_end + 1 in candidate_index_set
            and stripped_lines[run_end + 1] == marker
        ):
            run_end += 1
        prev_blank = idx <= 0 or not stripped_lines[idx - 1]
        next_blank = run_end >= len(lines) - 1 or not stripped_lines[run_end + 1]
        if prev_blank and next_blank:
            stats[1] += run_end - idx + 1
        idx = run_end + 1

    markers: set[str] = set()
    for marker, (count, isolated_count) in candidate_stats.items():
        if count <= 0:
            continue
        if _contains_quote_or_bracket(marker):
            continue
        chars = [ch for ch in marker if not ch.isspace()]
        if not chars:
            continue
        embedded = sum(non_candidate_char_counts.get(ch, 0) for ch in chars)
        marker_size = max(1, len(chars))
        embedded_density = embedded / float(max(1, count * marker_size))
        if embedded_density > 0.35:
            continue
        isolated_ratio = isolated_count / float(count)
        if count >= 2:
            if isolated_ratio < 0.8:
                continue
            markers.add(marker)
            continue
        # Single-occurrence markers are allowed only when they are a single
        # symbol-like glyph and do not appear in body lines.
        if count == 1 and marker_size == 1 and embedded == 0:
            markers.add(marker)
    return markers


def normalize_section_breaks(text: str) -> str:
    if not text:
        return text
    markers = _infer_section_break_markers(text)
    if markers:
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            if line.strip() in markers:
                lines[idx] = SECTION_BREAK
        text = "\n".join(lines)
    if SECTION_BREAK not in text:
        return text
    text = re.sub(
        rf"[ \t]*{SECTION_BREAK}[ \t]*",
        f"\n\n{SECTION_BREAK}\n\n",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(
        rf"(?:\n\n{SECTION_BREAK}){{2,}}",
        f"\n\n{SECTION_BREAK}",
        text,
    )
    text = re.sub(
        rf"(?:{SECTION_BREAK}\n\n){{2,}}",
        f"{SECTION_BREAK}\n\n",
        text,
    )
    text = re.sub(rf"^(?:\s*{SECTION_BREAK}\s*)+", "", text)
    text = re.sub(rf"(?:\s*{SECTION_BREAK}\s*)+$", "", text)
    return text


def strip_section_breaks(text: str) -> str:
    if not text or SECTION_BREAK not in text:
        return text
    return text.replace(SECTION_BREAK, "")


def read_clean_text(path: Path) -> str:
    s = path.read_text(encoding="utf-8", errors="strict")
    s = s.replace("ʼ", "'").replace("‘", "'").replace("’", "'")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip() + "\n"


def _extract_markdown_title(lines: Iterable[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            return heading
        return ""
    return ""


def title_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    if not stem:
        return ""
    stem = re.sub(r"^[0-9]+[-_ ]+", "", stem)
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem or path.stem


def guess_title_from_path(path: Path) -> str:
    heading = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            heading = _extract_markdown_title(handle)
    except (OSError, UnicodeDecodeError):
        heading = ""
    if heading:
        return heading
    fallback = title_from_filename(path)
    return fallback or path.stem or "text"
