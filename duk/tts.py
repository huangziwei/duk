from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import synth_qwen
from . import voice as voice_util
from .text import (
    SECTION_BREAK,
    normalize_section_breaks,
    read_clean_text,
    strip_section_breaks,
)
from .voice import VoiceConfig

_END_PUNCT = set("。！？?!…")
_MID_PUNCT = set("、，,;；:：")
_CLOSE_PUNCT = set("」』）】］〉》」』”’\"'")
_DOUBLE_QUOTE_CHARS = {'"', "“", "”", "«", "»", "„", "‟", "❝", "❞", "＂"}
_SINGLE_QUOTE_CHARS = {"'", "‘", "’", "‚", "‛", "＇"}
_LEADING_ELISIONS = {
    "tis",
    "twas",
    "twere",
    "twill",
    "til",
    "em",
    "cause",
    "bout",
    "round",
}
_CJK_QUOTE_CHARS = {
    "「",
    "」",
    "『",
    "』",
    "《",
    "》",
    "〈",
    "〉",
    "【",
    "】",
    "〔",
    "〕",
    "［",
    "］",
    "｢",
    "｣",
    "〝",
    "〞",
    "〟",
}
# Book-title marks: dropped without injecting a pause (《红楼梦》 reads through).
_TITLE_MARK_CHARS = {"《", "》", "〈", "〉"}

_SECTION_PAD_MULT = 2
_CHAPTER_BREAK_PAD_MULTIPLIER = 4
_CHAPTER_PAD_MULT = _CHAPTER_BREAK_PAD_MULTIPLIER
_SECTION_BREAK_PAUSE_MULTIPLIER = 1 + _SECTION_PAD_MULT
_TITLE_BREAK_PAUSE_MULTIPLIER = 1 + _CHAPTER_PAD_MULT
_HEADING_CATEGORY_SECTION = "section"
_HEADING_CATEGORY_TITLE = "title"
_HEADING_CATEGORY_TO_PAUSE_MULTIPLIER = {
    _HEADING_CATEGORY_SECTION: _SECTION_BREAK_PAUSE_MULTIPLIER,
    _HEADING_CATEGORY_TITLE: _TITLE_BREAK_PAUSE_MULTIPLIER,
}
_DEFAULT_OUTPUT_SAMPLE_RATE = 24_000
_ELLIPSIS_ONLY_RE = re.compile(r"[…⋯]+")
_ELLIPSIS_PAUSE_BASE_MS = 380
_ELLIPSIS_PAUSE_STEP_MS = 120
_ELLIPSIS_PAUSE_MAX_MS = 1200
_HEADING_TOKEN_RE = re.compile(r"[^\W_]+(?:['’.\-][^\W_]+)*")
_ROMAN_TOKEN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(
    r"[.!?。！？](?:[\"')\]\}”’»」』）】])*(?=\s|$)"
)
_CJK_OPEN_QUOTES = {"「", "『", "《", "〈", "【", "〔", "［", "｢", "〝"}
_CJK_CLOSE_QUOTES = {"」", "』", "》", "〉", "】", "〕", "］", "｣", "〞", "〟"}
_OPEN_QUOTE_TO_CLOSES: dict[str, set[str]] = {
    "「": {"」"},
    "『": {"』"},
    "《": {"》"},
    "〈": {"〉"},
    "【": {"】"},
    "〔": {"〕"},
    "［": {"］"},
    "｢": {"｣"},
    "〝": {"〞", "〟"},
    "“": {"”"},
    "«": {"»"},
    "❝": {"❞"},
}
_DASH_RUN_RE = re.compile(r"[‐‑‒–—―─━]{2,}")
_HANZI_SAFE_MARKS = {"々"}
# Speech-attribution verbs that justify protecting a short inline quote.
_SPEECH_VERB_CHARS = "说說道问問答喊叫嚷唱念嘆叹"
_SHORT_TAIL_PUNCT = "。"


@dataclass(frozen=True)
class ChapterInput:
    index: int
    id: str
    title: str
    text: str
    path: Optional[str] = None
    headings: Tuple[str, ...] = ()
    heading_categories: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TtsPipeline:
    simplified: str
    after_numbers: str
    prepared: str


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def chapter_id_from_path(index: int, title: str, rel_path: Optional[str]) -> str:
    if rel_path:
        stem = Path(rel_path).stem
        if stem:
            return stem
    return f"{index:04d}-chapter"


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n\n\n" in text:
        text = re.sub(r"\n{3,}", f"\n\n{SECTION_BREAK}\n\n", text)
    text = normalize_section_breaks(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# character classes


def _is_hanzi_char(ch: str) -> bool:
    if not ch or len(ch) != 1:
        return False
    code = ord(ch)
    ranges = (
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F),  # Extension C
        (0x2B740, 0x2B81F),  # Extension D
        (0x2B820, 0x2CEAF),  # Extension E
        (0x2CEB0, 0x2EBEF),  # Extension F
        (0x30000, 0x3134F),  # Extension G
    )
    for start, end in ranges:
        if start <= code <= end:
            return True
    return False


def _has_hanzi(text: str) -> bool:
    if not text:
        return False
    return any(_is_hanzi_char(ch) for ch in text)


def _is_chinese_char(ch: str) -> bool:
    if not ch:
        return False
    if ch in _HANZI_SAFE_MARKS:
        return True
    return _is_hanzi_char(ch)


def _has_chinese(text: str) -> bool:
    if not text:
        return False
    return any(_is_chinese_char(ch) for ch in text)


# traditional -> simplified


@lru_cache(maxsize=1)
def _get_t2s_converter():
    import opencc

    return opencc.OpenCC("t2s")


def _normalize_traditional(text: str) -> str:
    """Fold traditional characters to simplified for the TTS front end.

    jieba segmentation and pypinyin dictionaries are strongest on simplified
    text; the clean chapter text keeps the book's original script.
    """
    if not text or not _has_hanzi(text):
        return text
    return _get_t2s_converter().convert(text)


# chunking


def _is_ws(ch: str) -> bool:
    return ch.isspace() or ch == SECTION_BREAK


def _trim_span(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    while start < end and _is_ws(text[start]):
        start += 1
    while end > start and _is_ws(text[end - 1]):
        end -= 1
    if start >= end:
        return None
    return start, end


def _span_has_content(text: str, start: int, end: int) -> bool:
    for ch in text[start:end]:
        if _is_ws(ch):
            continue
        if _is_chinese_char(ch):
            return True
        if ch.isdigit():
            return True
        if ch.isascii() and ch.isalnum():
            return True
    return False


def _span_has_line_content(text: str, start: int, end: int) -> bool:
    if _span_has_content(text, start, end):
        return True
    for ch in text[start:end]:
        if _is_ws(ch):
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            return True
    return False


def _advance_ws(text: str, pos: int) -> int:
    while pos < len(text) and _is_ws(text[pos]):
        pos += 1
    return pos


def _space_pause_split_indices(text: str) -> set[int]:
    if not text:
        return set()
    splits: set[int] = set()
    start = 0
    length = len(text)
    while start < length:
        end = text.find("\n", start)
        if end == -1:
            end = length
        line = text[start:end]
        if line:
            candidates: List[int] = []
            for idx, ch in enumerate(line):
                if ch not in (" ", "　"):
                    continue
                prev = line[idx - 1] if idx > 0 else ""
                next_ch = line[idx + 1] if idx + 1 < len(line) else ""
                if _is_chinese_char(prev) and _is_chinese_char(next_ch):
                    candidates.append(idx)
            if len(candidates) == 1:
                has_punct = any(ch in _END_PUNCT or ch in _MID_PUNCT for ch in line)
                has_ascii_letters = any(ch.isascii() and ch.isalpha() for ch in line)
                if not has_punct and not has_ascii_letters:
                    splits.add(start + candidates[0])
        start = end + 1
    return splits


def split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    space_splits = _space_pause_split_indices(text)
    start = 0
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if i in space_splits:
            if _span_has_content(text, start, i):
                span = _trim_span(text, start, i)
                if span:
                    spans.append(span)
                i += 1
                i = _advance_ws(text, i)
                start = i
                continue
            i += 1
            i = _advance_ws(text, i)
            continue
        if ch == "\n":
            if _span_has_line_content(text, start, i):
                span = _trim_span(text, start, i)
                if span:
                    spans.append(span)
                i += 1
                i = _advance_ws(text, i)
                start = i
                continue
            i += 1
            i = _advance_ws(text, i)
            continue
        if ch == SECTION_BREAK:
            if _span_has_line_content(text, start, i):
                span = _trim_span(text, start, i)
                if span:
                    spans.append(span)
                i += 1
                i = _advance_ws(text, i)
                start = i
                continue
            i += 1
            i = _advance_ws(text, i)
            continue
        if ch in _END_PUNCT:
            j = i + 1
            while j < length and text[j] in _END_PUNCT:
                j += 1
            while j < length and text[j] in _CLOSE_PUNCT:
                j += 1
            if _span_has_content(text, start, j):
                span = _trim_span(text, start, j)
                if span:
                    spans.append(span)
                j = _advance_ws(text, j)
                start = j
                i = j
                continue
            if j < length and text[j] in {"\n", SECTION_BREAK}:
                span = _trim_span(text, start, j)
                if span:
                    spans.append(span)
                j = _advance_ws(text, j)
                start = j
                i = j
                continue
            i = _advance_ws(text, j)
            continue
        i += 1
    span = _trim_span(text, start, length)
    if span:
        spans.append(span)
    return spans


def _split_long_span(
    text: str, start: int, end: int, max_chars: int
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    i = start
    while i < end:
        j = min(end, i + max_chars)
        if j < end:
            split_at = -1
            for punct in _MID_PUNCT:
                idx = text.rfind(punct, i, j)
                if idx > split_at:
                    split_at = idx
            if split_at > i:
                j = split_at + 1
        span = _trim_span(text, i, j)
        if span:
            spans.append(span)
        i = _advance_ws(text, j)
    return spans


def _collect_quote_pair_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    stack: List[Tuple[str, int]] = []
    for idx, ch in enumerate(text):
        close_quotes = _OPEN_QUOTE_TO_CLOSES.get(ch)
        if close_quotes:
            stack.append((ch, idx))
            continue
        if not stack:
            continue
        open_quote, open_idx = stack[-1]
        expected_close = _OPEN_QUOTE_TO_CLOSES.get(open_quote)
        if expected_close and ch in expected_close:
            stack.pop()
            spans.append((open_idx, idx + 1))
    spans.sort()
    return spans


def _is_protectable_quote_span(text: str, start: int, end: int, max_chars: int) -> bool:
    if end <= start:
        return False
    if max_chars > 0 and end - start > max_chars:
        return False
    segment = text[start:end]
    if "\n" in segment or SECTION_BREAK in segment:
        return False
    inner = segment[1:-1].strip() if len(segment) > 2 else ""
    if not inner:
        return False
    prev_ch = text[start - 1] if start > 0 else ""
    next_ch = text[end] if end < len(text) else ""

    inline_before = (prev_ch.isalnum() or _is_chinese_char(prev_ch)) and not prev_ch.isspace()
    if not inline_before and prev_ch in {"、", "，", ",", ":", "："}:
        idx = start - 2
        while idx >= 0 and text[idx].isspace():
            idx -= 1
        if idx >= 0:
            prev_prev = text[idx]
            inline_before = (prev_prev.isalnum() or _is_chinese_char(prev_prev)) and not prev_prev.isspace()

    inline_after = (next_ch.isalnum() or _is_chinese_char(next_ch)) and not next_ch.isspace()
    has_sentence_punct = any(ch in "。！？!?…⋯" for ch in inner)
    if inline_before and inline_after and not has_sentence_punct:
        # Inline short quotes are often emphasis/term markers, not dialogue.
        # Keep protection only for explicit speech-attribution patterns.
        next_tail = text[end : min(len(text), end + 4)].lstrip()
        if not any(ch in _SPEECH_VERB_CHARS for ch in next_tail[:3]):
            return False
    return True


def _split_spans_on_cut_points(
    text: str, spans: Sequence[Tuple[int, int]], cut_points: Sequence[int]
) -> List[Tuple[int, int]]:
    if not spans:
        return []
    cuts = sorted(set(cut_points))
    segmented: List[Tuple[int, int]] = []
    for start, end in spans:
        points = [start]
        for cut in cuts:
            if start < cut < end:
                points.append(cut)
        points.append(end)
        points = sorted(set(points))
        local: List[Tuple[int, int]] = []
        for idx in range(len(points) - 1):
            span = _trim_span(text, points[idx], points[idx + 1])
            if span:
                if local and not _span_has_content(text, span[0], span[1]):
                    prev_start, _ = local[-1]
                    local[-1] = (prev_start, span[1])
                else:
                    local.append(span)
        segmented.extend(local)
    return segmented


def _enclosing_quote_span_index(
    start: int, end: int, quote_spans: Sequence[Tuple[int, int]]
) -> Optional[int]:
    best_idx: Optional[int] = None
    best_len = -1
    for idx, (quote_start, quote_end) in enumerate(quote_spans):
        if quote_start <= start and end <= quote_end:
            span_len = quote_end - quote_start
            if span_len > best_len:
                best_idx = idx
                best_len = span_len
    return best_idx


def _is_hard_chunk_boundary(
    text: str, prev_span: Tuple[int, int], next_span: Tuple[int, int]
) -> bool:
    prev_start, prev_end = prev_span
    next_start, _next_end = next_span
    gap = text[prev_end:next_start]
    if "\n" in gap or SECTION_BREAK in gap:
        return True
    if gap and gap.strip() == "":
        prev_text = text[prev_start:prev_end]
        if not _ends_with_sentence_punct(prev_text):
            return True
    return False


def _is_same_paragraph_gap(text: str, left_end: int, right_start: int) -> bool:
    gap = text[left_end:right_start]
    if "\n" in gap or SECTION_BREAK in gap:
        return False
    return True


def _build_chunk_units(
    text: str,
    sentence_spans: Sequence[Tuple[int, int]],
    max_chars: int,
) -> List[Tuple[int, int, bool, bool]]:
    if not sentence_spans:
        return []

    quote_spans = [
        span
        for span in _collect_quote_pair_spans(text)
        if _is_protectable_quote_span(text, span[0], span[1], max_chars)
    ]
    if not quote_spans:
        return [(start, end, False, False) for start, end in sentence_spans]

    cut_points = [idx for span in quote_spans for idx in span]
    segmented = _split_spans_on_cut_points(text, sentence_spans, cut_points)
    grouped: List[List[Optional[int]]] = []
    for start, end in segmented:
        quote_idx = _enclosing_quote_span_index(start, end, quote_spans)
        if grouped and quote_idx is not None and grouped[-1][2] == quote_idx:
            grouped[-1][1] = end
            continue
        grouped.append([start, end, quote_idx])

    units: List[Tuple[int, int, bool, bool]] = []
    for idx, (start, end, quote_idx) in enumerate(grouped):
        is_quote = quote_idx is not None
        adjacent_quote = False
        if not is_quote:
            if idx > 0 and grouped[idx - 1][2] is not None:
                prev_end = int(grouped[idx - 1][1])
                if _is_same_paragraph_gap(text, prev_end, int(start)):
                    adjacent_quote = True
            if idx + 1 < len(grouped) and grouped[idx + 1][2] is not None:
                next_start = int(grouped[idx + 1][0])
                if _is_same_paragraph_gap(text, int(end), next_start):
                    adjacent_quote = True
        units.append((int(start), int(end), is_quote, adjacent_quote))
    return units


def _pack_chunk_units(
    text: str,
    units: Sequence[Tuple[int, int, bool, bool]],
    max_chars: int,
    min_chars: int = 0,
) -> List[Tuple[int, int]]:
    if not units:
        return []
    if max_chars <= 0:
        # No length cap: keep every unit as its own span. Any min_chars merging
        # is left to _merge_short_chunks.
        return [(start, end) for start, end, _is_quote, _adj in units]

    # With a minimum set we chunk at the sentence level: a chunk is emitted as
    # soon as it reaches min_chars, so a sentence already in [min, max] stays on
    # its own instead of being packed toward max_chars. With min_chars <= 0 we
    # keep the historical behavior of filling each chunk toward max_chars.
    flush_threshold = min_chars if min_chars > 0 else max_chars

    packed: List[Tuple[int, int]] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    def flush() -> None:
        nonlocal current_start, current_end
        if current_start is None or current_end is None:
            return
        # A trailing run that is too short to stand on its own folds back into
        # the previous chunk, but only within the same paragraph (never across a
        # blank line or section break) and only while it stays under max_chars.
        if (
            min_chars > 0
            and current_end - current_start < min_chars
            and packed
            and _is_same_paragraph_gap(text, packed[-1][1], current_start)
            and current_end - packed[-1][0] <= max_chars
        ):
            packed[-1] = (packed[-1][0], current_end)
        else:
            packed.append((current_start, current_end))
        current_start = None
        current_end = None

    for idx, (start, end, is_quote, adjacent_quote) in enumerate(units):
        if is_quote or adjacent_quote:
            flush()
            packed.append((start, end))
            continue

        hard_boundary = False
        if idx > 0:
            prev_start, prev_end, prev_is_quote, prev_adjacent_quote = units[idx - 1]
            if prev_is_quote or prev_adjacent_quote:
                hard_boundary = True
            else:
                hard_boundary = _is_hard_chunk_boundary(
                    text, (prev_start, prev_end), (start, end)
                )
        if hard_boundary:
            flush()

        if current_start is None or current_end is None:
            current_start = start
            current_end = end
        elif end - current_start <= max_chars:
            current_end = end
        else:
            flush()
            current_start = start
            current_end = end

        if current_end - current_start >= flush_threshold:
            flush()

    flush()
    return packed


def _merge_short_chunks(
    spans: Sequence[Tuple[int, int]],
    min_chars: int,
    max_chars: int = 0,
) -> List[Tuple[int, int]]:
    # min_chars is a hard floor: chunks shorter than this synthesize poorly, so
    # every below-min chunk must be merged into a neighbor. A chunk that already
    # reaches min_chars is left at the sentence level (it is never grown toward
    # max_chars); only short chunks accumulate, and they flush as soon as they
    # reach the floor. Reaching min_chars takes priority over max_chars in the
    # rare case the two conflict.
    if min_chars <= 0:
        return list(spans)
    merged: List[Tuple[int, int]] = []
    pending: Optional[Tuple[int, int]] = None
    for start, end in spans:
        if pending is None:
            pending = (start, end)
        elif pending[1] - pending[0] >= min_chars:
            merged.append(pending)
            pending = (start, end)
        elif (
            max_chars > 0
            and end - pending[0] > max_chars
            and end - start >= min_chars
            and merged
        ):
            # Absorbing this span would overflow max_chars and it is long enough
            # to stand on its own, so fold the short pending back into the
            # previous chunk instead of bloating this one.
            merged[-1] = (merged[-1][0], pending[1])
            pending = (start, end)
        else:
            pending = (pending[0], end)
    if pending is not None:
        if pending[1] - pending[0] >= min_chars or not merged:
            merged.append(pending)
        else:
            merged[-1] = (merged[-1][0], pending[1])
    return merged


def make_chunk_spans(
    text: str,
    max_chars: int,
    chunk_mode: str = "chinese",
    min_chars: int = 0,
) -> List[Tuple[int, int]]:
    _ = chunk_mode
    sentence_spans: List[Tuple[int, int]] = []
    for sent_start, sent_end in split_sentence_spans(text):
        if max_chars > 0 and sent_end - sent_start > max_chars:
            sentence_spans.extend(
                _split_long_span(text, sent_start, sent_end, max_chars)
            )
        else:
            sentence_spans.append((sent_start, sent_end))
    units = _build_chunk_units(text, sentence_spans, max_chars)
    spans = _pack_chunk_units(text, units, max_chars, min_chars)
    return _merge_short_chunks(spans, min_chars, max_chars)


def make_chunks(
    text: str,
    max_chars: int,
    chunk_mode: str = "chinese",
    min_chars: int = 0,
) -> List[str]:
    spans = make_chunk_spans(
        text, max_chars=max_chars, chunk_mode=chunk_mode, min_chars=min_chars
    )
    return [text[start:end] for start, end in spans]


def _coerce_span_pairs(spans: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        pairs.append((start, end))
    return pairs


# pause multipliers


def _pause_multiplier_from_gap(gap: str) -> int:
    if not gap:
        return 1
    if SECTION_BREAK in gap:
        return _SECTION_BREAK_PAUSE_MULTIPLIER
    max_run = 0
    for match in re.finditer(r"\n+", gap):
        max_run = max(max_run, len(match.group(0)))
    if max_run >= 5:
        return _TITLE_BREAK_PAUSE_MULTIPLIER
    if max_run >= 3:
        return _SECTION_BREAK_PAUSE_MULTIPLIER
    return 1


def _heading_word_count(text: str) -> int:
    return len(_HEADING_TOKEN_RE.findall(text))


def _clean_heading_token(token: str) -> str:
    cleaned = token.strip().strip("\"'“”‘’()[]{}<>「」『』【】")
    return cleaned.rstrip(".,:;!?)]")


def _is_heading_number_token(token: str) -> bool:
    cleaned = _clean_heading_token(token)
    if not cleaned:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)*", cleaned):
        return True
    if _ROMAN_TOKEN_RE.fullmatch(cleaned):
        return True
    return False


def _looks_like_numeric_heading(text: str) -> bool:
    tokens = _HEADING_TOKEN_RE.findall(text)
    if not tokens or len(tokens) > 6:
        return False
    return all(_is_heading_number_token(token) for token in tokens)


def _looks_like_paragraph_chunk(stripped: str) -> bool:
    words = _heading_word_count(stripped)
    if words <= 0:
        return False
    sentence_endings = len(_SENTENCE_END_RE.findall(stripped))
    clause_breaks = (
        stripped.count(",")
        + stripped.count(";")
        + stripped.count("、")
        + stripped.count("，")
        + stripped.count("；")
        + stripped.count("：")
    )
    if sentence_endings >= 2:
        return True
    if sentence_endings >= 1 and words >= 6:
        return True
    if clause_breaks >= 2 and words >= 12:
        return True
    return False


def _looks_like_dialogue_chunk(stripped: str) -> bool:
    text = stripped.strip()
    if not text:
        return False

    # Dialogue lines are often isolated and wrapped by quote pairs.
    trimmed = text.rstrip("。！？!?…．，、,;；:：")
    if (
        len(trimmed) >= 2
        and trimmed[0] in _CJK_OPEN_QUOTES
        and trimmed[-1] in _CJK_CLOSE_QUOTES
    ):
        inner = trimmed[1:-1].strip()
        if inner:
            return True

    # Chunking can split quoted dialogue at sentence punctuation, leaving only one
    # quote edge in a fragment (e.g., “不。 / 谢谢你们”).
    if len(trimmed) >= 2 and trimmed[0] in _CJK_OPEN_QUOTES:
        inner = trimmed[1:].strip()
        if inner:
            return True
    if len(trimmed) >= 2 and trimmed[-1] in _CJK_CLOSE_QUOTES:
        inner = trimmed[:-1].strip()
        if inner:
            return True

    if (
        len(trimmed) >= 2
        and trimmed[0] in _DOUBLE_QUOTE_CHARS
        and trimmed[-1] in _DOUBLE_QUOTE_CHARS
    ):
        inner = trimmed[1:-1].strip()
        if inner:
            return True

    if (
        len(trimmed) >= 2
        and trimmed[0] in _SINGLE_QUOTE_CHARS
        and trimmed[-1] in _SINGLE_QUOTE_CHARS
    ):
        inner = trimmed[1:-1].strip()
        if inner:
            return True

    return False


def _ends_with_sentence_punct(text: str) -> bool:
    trailing = "".join(_CLOSE_PUNCT) + "」』）】"
    stripped = text.rstrip(trailing)
    if not stripped:
        return False
    return stripped[-1] in (_END_PUNCT | {"。", "！", "？"})


def _ends_with_continuation_punct(text: str) -> bool:
    trailing = "".join(_CLOSE_PUNCT) + "」』）】"
    stripped = text.rstrip(trailing)
    if not stripped:
        return False
    return stripped[-1] in (_MID_PUNCT | {"—", "―", "─", "━", "…", "⋯"})


def _looks_like_heading_chunk(chunk: str) -> bool:
    stripped = " ".join(part.strip() for part in chunk.splitlines() if part.strip())
    if not stripped:
        return False
    if _looks_like_dialogue_chunk(stripped):
        return False
    if _looks_like_numeric_heading(stripped):
        return True
    tokens = _HEADING_TOKEN_RE.findall(stripped)
    if not tokens:
        return False
    if _ends_with_sentence_punct(stripped):
        return False
    if _ends_with_continuation_punct(stripped):
        return False
    if _looks_like_paragraph_chunk(stripped):
        return False
    alpha_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    if not alpha_tokens:
        return False
    return True


def _gap_has_symbolic_separator(gap: str) -> bool:
    for line in gap.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_BREAK in stripped:
            return True
        if any(ch.isalnum() for ch in stripped):
            continue
        return True
    return False


def _is_continuation_linebreak_gap(prev_chunk: str, gap: str) -> bool:
    if not gap or "\n" not in gap:
        return False
    if SECTION_BREAK in gap:
        return False
    if not _ends_with_continuation_punct(prev_chunk):
        return False
    if _gap_has_symbolic_separator(gap):
        return False
    return True


def _normalize_heading_line_key(text: str) -> str:
    normalized = (
        str(text or "")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("«", '"')
        .replace("»", '"')
    )
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def _build_heading_line_keys(heading_lines: Optional[Sequence[str]]) -> set[str]:
    keys: set[str] = set()
    if not heading_lines:
        return keys
    for line in heading_lines:
        key = _normalize_heading_line_key(str(line))
        if key:
            keys.add(key)
    return keys


def _normalize_heading_lines(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        heading = str(value).strip()
        if not heading:
            continue
        key = _normalize_heading_line_key(heading)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(heading)
    return out


def _normalize_heading_category(value: object) -> str:
    cleaned = str(value or "").strip().lower()
    if cleaned == _HEADING_CATEGORY_TITLE:
        return _HEADING_CATEGORY_TITLE
    return _HEADING_CATEGORY_SECTION


def _normalize_heading_categories(values: object) -> Dict[str, str]:
    if not isinstance(values, dict):
        return {}
    out: Dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = _normalize_heading_line_key(str(raw_key))
        if not key:
            continue
        category = _normalize_heading_category(raw_value)
        prev = out.get(key)
        if prev == _HEADING_CATEGORY_TITLE:
            continue
        out[key] = category
    return out


def _with_chapter_title_heading_category(
    chapter_title: str,
    heading_lines: Sequence[str],
    heading_categories: Dict[str, str],
) -> Dict[str, str]:
    title_key = _normalize_heading_line_key(chapter_title)
    if not title_key:
        return dict(heading_categories)
    heading_keys = {_normalize_heading_line_key(line) for line in heading_lines}
    if title_key not in heading_keys:
        return dict(heading_categories)
    out = dict(heading_categories)
    out[title_key] = _HEADING_CATEGORY_TITLE
    return out


def _heading_pause_multiplier_from_category(category: object) -> int:
    normalized = _normalize_heading_category(category)
    return _HEADING_CATEGORY_TO_PAUSE_MULTIPLIER.get(
        normalized, _SECTION_BREAK_PAUSE_MULTIPLIER
    )


def _explicit_heading_chunk_pause_multiplier(
    chunk: str,
    heading_line_keys: set[str],
    heading_categories: Dict[str, str],
) -> Optional[int]:
    if not heading_line_keys:
        return None
    lines = [line for line in chunk.splitlines() if line.strip()]
    if not lines:
        return None
    pause = _SECTION_BREAK_PAUSE_MULTIPLIER
    for line in lines:
        key = _normalize_heading_line_key(line)
        if not key or key not in heading_line_keys:
            return None
        category = heading_categories.get(key, _HEADING_CATEGORY_SECTION)
        pause = max(pause, _heading_pause_multiplier_from_category(category))
    return pause


def compute_chunk_pause_multipliers(
    text: str,
    spans: Sequence[Tuple[int, int]],
    heading_lines: Optional[Sequence[str]] = None,
    heading_categories: Optional[Dict[str, str]] = None,
) -> List[int]:
    if not spans:
        return []
    multipliers = [1] * len(spans)
    chunk_texts = [text[int(start) : int(end)] for start, end in spans]
    heading_line_keys = _build_heading_line_keys(heading_lines)
    heading_category_map = _normalize_heading_categories(heading_categories)
    heading_pause_by_chunk: List[Optional[int]] = [
        _explicit_heading_chunk_pause_multiplier(
            chunk,
            heading_line_keys,
            heading_category_map,
        )
        for chunk in chunk_texts
    ]
    heading_like = [pause is not None for pause in heading_pause_by_chunk]
    for idx in range(len(spans) - 1):
        end = int(spans[idx][1])
        next_start = int(spans[idx + 1][0])
        if next_start < end:
            continue
        gap = text[end:next_start]
        pause = _pause_multiplier_from_gap(gap)
        if _is_continuation_linebreak_gap(chunk_texts[idx], gap):
            multipliers[idx] = 1
            continue
        if "\n" in gap or SECTION_BREAK in gap:
            if _gap_has_symbolic_separator(gap):
                pause = max(pause, _SECTION_BREAK_PAUSE_MULTIPLIER)
            if heading_like[idx] or heading_like[idx + 1]:
                heading_pause = _SECTION_BREAK_PAUSE_MULTIPLIER
                if heading_pause_by_chunk[idx] is not None:
                    heading_pause = max(heading_pause, heading_pause_by_chunk[idx])
                if heading_pause_by_chunk[idx + 1] is not None:
                    heading_pause = max(heading_pause, heading_pause_by_chunk[idx + 1])
                pause = max(pause, heading_pause)
        elif not _ends_with_sentence_punct(chunk_texts[idx]):
            # Forced mid-sentence split (long sentence cut to fit the model).
            # The next chunk continues the same sentence with no paragraph
            # boundary between them — adding the normal inter-sentence pad
            # produces an unnatural beat. Drop the pause entirely.
            pause = 0
        multipliers[idx] = pause
    return multipliers


def _legacy_pause_multipliers(
    chunk_section_breaks: object,
    chunk_count: int,
    add_chapter_boundary: bool,
) -> List[int]:
    legacy = [1] * max(0, chunk_count)
    if (
        isinstance(chunk_section_breaks, list)
        and chunk_section_breaks
        and chunk_count > 0
    ):
        for idx in range(min(chunk_count, len(chunk_section_breaks))):
            if chunk_section_breaks[idx]:
                legacy[idx] = max(legacy[idx], _SECTION_BREAK_PAUSE_MULTIPLIER)
    if add_chapter_boundary and legacy:
        legacy[-1] = max(legacy[-1], _TITLE_BREAK_PAUSE_MULTIPLIER)
    return legacy


def _normalize_pause_multipliers(
    pause_multipliers: object,
    chunk_count: int,
    fallback: Optional[Sequence[int]] = None,
) -> List[int]:
    if chunk_count <= 0:
        return []
    normalized = [1] * chunk_count
    if isinstance(fallback, Sequence):
        for idx in range(min(chunk_count, len(fallback))):
            try:
                parsed = int(fallback[idx])
            except (TypeError, ValueError):
                parsed = 1
            normalized[idx] = parsed if parsed >= 0 else 1
    if isinstance(pause_multipliers, list) and len(pause_multipliers) == chunk_count:
        for idx, value in enumerate(pause_multipliers):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                normalized[idx] = parsed
    return normalized


def _apply_chapter_boundary_pause_multipliers(
    manifest_chapters: Sequence[dict],
) -> None:
    for idx, entry in enumerate(manifest_chapters):
        if idx >= len(manifest_chapters) - 1:
            break
        if not isinstance(entry, dict):
            continue
        chunks = entry.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            continue
        fallback = _legacy_pause_multipliers(
            entry.get("chunk_section_breaks"),
            len(chunks),
            add_chapter_boundary=False,
        )
        normalized = _normalize_pause_multipliers(
            entry.get("pause_multipliers"),
            len(chunks),
            fallback=fallback,
        )
        normalized[-1] = max(normalized[-1], _TITLE_BREAK_PAUSE_MULTIPLIER)
        entry["pause_multipliers"] = normalized


# TTS text preparation


def _strip_double_quotes(text: str) -> str:
    if not text:
        return text
    return "".join(ch for ch in text if ch not in _DOUBLE_QUOTE_CHARS)


def _strip_single_quotes(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch not in _SINGLE_QUOTE_CHARS:
            out.append(ch)
            continue
        prev = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < len(text) else ""
        if (
            prev
            and next_ch
            and prev.isascii()
            and next_ch.isascii()
            and prev.isalnum()
            and next_ch.isalnum()
        ):
            out.append(ch)
            continue
        if prev and next_ch and _is_chinese_char(prev) and _is_chinese_char(next_ch):
            out.append(ch)
            continue
        if (
            (not prev or not prev.isalnum())
            and next_ch
            and next_ch.isascii()
            and next_ch.isalpha()
        ):
            end = idx + 1
            while end < len(text) and text[end].isascii() and text[end].isalpha():
                end += 1
            word = text[idx + 1 : end].lower()
            if word in _LEADING_ELISIONS:
                out.append(ch)
                continue
        continue
    return "".join(out)


def _strip_format_chars(text: str) -> str:
    if not text:
        return text
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _strip_cjk_quotes(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    length = len(text)
    for idx, ch in enumerate(text):
        if ch not in _CJK_QUOTE_CHARS:
            out.append(ch)
            continue
        if ch in _TITLE_MARK_CHARS:
            # Book-title marks read straight through: 《红楼梦》 -> 红楼梦.
            continue
        if ch in _CJK_CLOSE_QUOTES:
            prev = text[idx - 1] if idx > 0 else ""
            j = idx + 1
            while j < length and text[j] in _CJK_QUOTE_CHARS:
                j += 1
            next_ch = text[j] if j < length else ""
            if (
                prev
                and next_ch
                and _is_chinese_char(prev)
                and _is_chinese_char(next_ch)
                and prev not in _END_PUNCT
                and prev not in _MID_PUNCT
                and next_ch not in _END_PUNCT
                and next_ch not in _MID_PUNCT
            ):
                out.append("、")
        continue
    return "".join(out)


def _line_needs_tail_punct(line: str) -> bool:
    """True if `line` (already trimmed of trailing spaces) lacks pause punct.

    A line ending in `」』` etc. counts as needing punct unless an end/mid-punct
    sits immediately before the closing quote(s).
    """
    if not line:
        return False
    last = line[-1]
    if last == "·":
        return False
    if last in _END_PUNCT or last in _MID_PUNCT:
        return False
    if last in _CLOSE_PUNCT:
        j = len(line) - 2
        while j >= 0 and line[j] in _CLOSE_PUNCT:
            j -= 1
        if j >= 0 and (line[j] in _END_PUNCT or line[j] in _MID_PUNCT):
            return False
    return True


def _ensure_line_tail_punct(text: str) -> str:
    """Per-line: append `。` to non-empty Chinese lines lacking pause punct.

    Runs BEFORE `_strip_cjk_quotes` so that lines ending in `」』` get a trailing
    `。` injected (which the quote stripper then leaves in place when the close
    quote is dropped). Idempotent.
    """
    if not text or not _has_chinese(text):
        return text
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        stripped_right = line.rstrip()
        trailing = line[len(stripped_right):]
        if not stripped_right or not _line_needs_tail_punct(stripped_right):
            out.append(line)
            continue
        out.append(stripped_right + _SHORT_TAIL_PUNCT + trailing)
    return "\n".join(out)


def _append_tail_punct_if_missing(text: str) -> str:
    if not text or not _has_chinese(text):
        return text
    idx = len(text) - 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    if idx < 0:
        return text
    if not _line_needs_tail_punct(text[: idx + 1]):
        return text
    return text + _SHORT_TAIL_PUNCT


def _chinese_space_to_pause(text: str, *, allow_full_stop: bool = True) -> str:
    """Convert stylistic spaces between Chinese characters to pause punct.

    Books use full/half-width spaces as beats (e.g. `他　停了下来`); F5 gets a
    real pause mark instead. Single spaces become `、`; runs of 2+ become `。`
    (unless a dash-run replacement already spent the full stop budget).
    """
    if not text:
        return text
    out: List[str] = []
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch not in (" ", "　"):
            out.append(ch)
            idx += 1
            continue
        run_start = idx
        while idx < length and text[idx] in (" ", "　"):
            idx += 1
        prev = text[run_start - 1] if run_start > 0 else ""
        next_ch = text[idx] if idx < length else ""
        if _is_chinese_char(prev) and _is_chinese_char(next_ch):
            run_len = idx - run_start
            if run_len >= 2 and allow_full_stop:
                out.append("。")
            else:
                out.append("、")
        else:
            out.append(" ")
    return "".join(out)


# number normalization (Mandarin)

_ZH_DIGITS = "零一二三四五六七八九"
_ZH_UNITS = ["", "十", "百", "千"]
_ZH_BIG_UNITS = ["", "万", "亿", "万亿"]


def _digit_seq_to_zh(seq: str) -> str:
    return "".join(_ZH_DIGITS[int(ch)] for ch in seq)


def _group_to_zh(value: int, *, is_leading_group: bool) -> str:
    """Read a 1..9999 group; 10-19 in the leading group reads 十X not 一十X."""
    if value == 0:
        return ""
    parts: List[str] = []
    digits = [int(ch) for ch in str(value)]
    length = len(digits)
    pending_zero = False
    for pos, digit in enumerate(digits):
        unit = _ZH_UNITS[length - pos - 1]
        if digit == 0:
            if parts:
                pending_zero = True
            continue
        if pending_zero:
            parts.append("零")
            pending_zero = False
        if (
            digit == 1
            and unit == "十"
            and pos == 0
            and length == 2
            and is_leading_group
        ):
            parts.append(unit)
            continue
        parts.append(_ZH_DIGITS[digit] + unit)
    return "".join(parts)


def _int_to_zh(value: int) -> str:
    if value < 0:
        return "负" + _int_to_zh(-value)
    if value == 0:
        return "零"
    groups: List[int] = []
    while value > 0:
        groups.append(value % 10_000)
        value //= 10_000
    parts: List[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        group = groups[idx]
        if group == 0:
            continue
        text = _group_to_zh(group, is_leading_group=idx == len(groups) - 1)
        unit = _ZH_BIG_UNITS[idx] if idx < len(_ZH_BIG_UNITS) else ""
        # A group under 1000 after a bigger group needs a spoken 零
        # (100500 -> 十万零五百).
        higher_nonzero = any(groups[j] for j in range(idx + 1, len(groups)))
        if higher_nonzero and group < 1000:
            parts.append("零" + text + unit)
        else:
            parts.append(text + unit)
    return "".join(parts)


_NFKC_ELLIPSIS_SENTINEL = "\uE001"


def _nfkc_preserve_ellipsis(text: str) -> str:
    if not text:
        return text
    guarded = text.replace("…", _NFKC_ELLIPSIS_SENTINEL)
    normalized = unicodedata.normalize("NFKC", guarded)
    return normalized.replace(_NFKC_ELLIPSIS_SENTINEL, "…")


# Digit runs longer than this are codes/phone numbers, not quantities.
_MAX_VALUE_DIGITS = 8


def _number_value_to_zh(raw: str) -> str:
    """Value reading with digit-by-digit fallback for long/zero-led strings."""
    if not raw:
        return raw
    if raw[0] == "0" and len(raw) > 1:
        return _digit_seq_to_zh(raw)
    if len(raw) > _MAX_VALUE_DIGITS:
        return _digit_seq_to_zh(raw)
    return _int_to_zh(int(raw))


# Range separators. NFKC has already folded full-width "－" to ASCII "-", so
# only the ASCII/CJK dashes plus the spelled-out 至/到 need matching here.
_RANGE_SEP = r"(?:[-–—~～〜]|至|到)"
# A range endpoint must not touch another digit or dash: that shape is an
# archive code (123-25-2), not a range.
_RANGE_EDGE_L = r"(?<![\d\-–—.])"
_RANGE_EDGE_R = r"(?![\d\-–—.])"


def _range_joiner(sep: str) -> str:
    """Dashes become a spoken 到; 至/到 are already words and stay put."""
    return sep if sep in ("至", "到") else "到"


def _year_to_zh(raw: str) -> str:
    """Calendar years read digit-wise (1949 -> 一九四九).

    One- and two-digit runs before 年 are durations in practice (凡39年,
    租期99年), so those keep the value reading.
    """
    return _digit_seq_to_zh(raw) if len(raw) >= 3 else _int_to_zh(int(raw))


def _normalize_numbers(text: str) -> str:
    """Rewrite Arabic-digit constructs as Chinese numerals for the TTS input.

    Runs on NFKC + simplified text, before quote/punct preparation. Regex order
    matters: date/time/percent/fraction shapes must fire before the general
    per-number fallback.
    """
    if not text or not re.search(r"\d", text):
        return text

    # thousands separators: 1,050 -> 1050 (read as one value, not "一,零五零")
    text = re.sub(
        r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?!\d)",
        lambda m: m.group(1).replace(",", ""),
        text,
    )

    # 2024/05/01 or 2024-05-01 -> 二零二四年五月一日
    def _replace_slash_date(match: re.Match) -> str:
        year, month, day = match.group(1), match.group(2), match.group(3)
        if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
            return match.group(0)
        return (
            f"{_year_to_zh(year)}年{_int_to_zh(int(month))}月{_int_to_zh(int(day))}日"
        )

    text = re.sub(
        r"(?<!\d)(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})(?!\d)",
        _replace_slash_date,
        text,
    )

    # percent ranges: 10-15% -> 百分之十到十五 (one 百分之, not two)
    def _replace_percent_range(match: re.Match) -> str:
        left, sep, right = match.group(1), match.group(2), match.group(3)
        return (
            f"百分之{_number_value_to_zh(left)}"
            f"{_range_joiner(sep)}{_number_value_to_zh(right)}"
        )

    text = re.sub(
        rf"{_RANGE_EDGE_L}(\d+)\s*({_RANGE_SEP})\s*(\d+)\s*%",
        _replace_percent_range,
        text,
    )

    # era-marked ranges: 公元前221－公元前206年 -> 公元前二二一到公元前二零六年.
    # The left endpoint carries no 年 of its own, so the rule below cannot see it.
    def _replace_era_range(match: re.Match) -> str:
        era_l, left, year_l, sep, era_r, right = match.groups()
        return (
            f"{era_l}{_year_to_zh(left)}{year_l}"
            f"{_range_joiner(sep)}{era_r or ''}{_year_to_zh(right)}"
        )

    text = re.sub(
        rf"(公元前|公元后|公元|前)(\d{{2,4}})(年?)\s*({_RANGE_SEP})\s*"
        rf"(公元前|公元后|公元|前)?(\d{{2,4}})",
        _replace_era_range,
        text,
    )

    # year ranges: 1368-1643年 -> 一三六八到一六四三年; 1937至1939年 keeps 至.
    # Without this the left endpoint never sees the 年 lookahead below and
    # falls through to a value reading (一千三百六十八).
    def _replace_year_range(match: re.Match) -> str:
        left, sep, right = match.group(1), match.group(2), match.group(3)
        # 1977-78年: a two-digit tail abbreviates the left year rather than
        # naming a duration, so it stays digit-wise.
        right_zh = (
            _digit_seq_to_zh(right)
            if len(right) == 2 and len(left) == 4
            else _year_to_zh(right)
        )
        return f"{_year_to_zh(left)}{_range_joiner(sep)}{right_zh}"

    text = re.sub(
        rf"{_RANGE_EDGE_L}(\d{{2,4}})\s*({_RANGE_SEP})\s*(\d{{2,4}})(?=\s*年)",
        _replace_year_range,
        text,
    )

    # bare year ranges in citations: 《年譜（1898-1969）》 -> 一八九八到一九六九.
    # Both endpoints must look like calendar years and carry no counter after
    # them, so quantity ranges (2000-3000人) keep their value reading.
    text = re.sub(
        rf"{_RANGE_EDGE_L}(1\d{{3}}|20\d{{2}})\s*({_RANGE_SEP})\s*"
        rf"(1\d{{3}}|20\d{{2}}){_RANGE_EDGE_R}(?![一-鿿])",
        _replace_year_range,
        text,
    )

    # 1984年 -> 一九八四年 (digit-wise year reading)
    text = re.sub(
        r"(?<!\d)(\d{2,4})(?=年)",
        lambda m: _year_to_zh(m.group(1)),
        text,
    )

    # 公元前1122 -> 公元前一一二二 (era years with no 年 of their own)
    text = re.sub(
        r"(?<=公元)(前)?(\d{2,4})(?!\d)",
        lambda m: (m.group(1) or "") + _year_to_zh(m.group(2)),
        text,
    )

    # 14:30(:05) -> 十四点三十分(零五秒)
    def _replace_clock(match: re.Match) -> str:
        hour, minute, second = match.group(1), match.group(2), match.group(3)
        if int(hour) > 24 or int(minute) > 59:
            return match.group(0)
        out = f"{_int_to_zh(int(hour))}点"
        if minute == "00" and not second:
            out += "整"
        elif minute[0] == "0" and minute != "00":
            out += f"零{_int_to_zh(int(minute))}分"
        else:
            out += f"{_int_to_zh(int(minute))}分"
        if second:
            if int(second) > 59:
                return match.group(0)
            if second[0] == "0" and second != "00":
                out += f"零{_int_to_zh(int(second))}秒"
            else:
                out += f"{_int_to_zh(int(second))}秒"
        return out

    text = re.sub(
        r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])",
        _replace_clock,
        text,
    )

    # percentages: 50% -> 百分之五十, 3.5% -> 百分之三点五
    def _replace_percent(match: re.Match) -> str:
        whole, frac = match.group(1), match.group(2)
        out = "百分之" + _number_value_to_zh(whole)
        if frac:
            out += "点" + _digit_seq_to_zh(frac)
        return out

    text = re.sub(r"(?<!\d)(\d+)(?:\.(\d+))?%", _replace_percent, text)

    # decimals: 3.14 -> 三点一四
    def _replace_decimal(match: re.Match) -> str:
        whole, frac = match.group(1), match.group(2)
        return _number_value_to_zh(whole) + "点" + _digit_seq_to_zh(frac)

    text = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", _replace_decimal, text)

    # fractions: 1/2 -> 二分之一 (only small values; larger slashes are codes)
    def _replace_fraction(match: re.Match) -> str:
        numer, denom = match.group(1), match.group(2)
        if int(denom) == 0 or len(numer) > 4 or len(denom) > 4:
            return match.group(0)
        return f"{_int_to_zh(int(denom))}分之{_int_to_zh(int(numer))}"

    text = re.sub(r"(?<![\d/.])(\d+)/(\d+)(?![\d/.])", _replace_fraction, text)

    # unit-suffixed ranges: 400亿-430亿 -> 四百亿到四百三十亿 (the repeated unit
    # sits between the endpoints, so the bare-number rule below cannot pair them)
    def _replace_unit_range(match: re.Match) -> str:
        left, unit_l, sep, right, unit_r = match.groups()
        return (
            f"{_number_value_to_zh(left)}{unit_l}"
            f"{_range_joiner(sep)}{_number_value_to_zh(right)}{unit_r}"
        )

    text = re.sub(
        rf"{_RANGE_EDGE_L}(\d+)([万亿])\s*({_RANGE_SEP})\s*(\d+)([万亿])",
        _replace_unit_range,
        text,
    )

    # numeric ranges: 3~5 -> 三到五, 8-10月 -> 八到十月, 1000至2000 keeps 至
    def _replace_range(match: re.Match) -> str:
        left, sep, right = match.group(1), match.group(2), match.group(3)
        return (
            f"{_number_value_to_zh(left)}{_range_joiner(sep)}"
            f"{_number_value_to_zh(right)}"
        )

    text = re.sub(
        rf"{_RANGE_EDGE_L}(\d+)\s*({_RANGE_SEP})\s*(\d+){_RANGE_EDGE_R}",
        _replace_range,
        text,
    )

    # digits glued to latin letters read digit-by-digit: A380 -> A三八零
    def _replace_alnum(match: re.Match) -> str:
        return _digit_seq_to_zh(match.group(1))

    text = re.sub(r"(?<=[A-Za-z])(\d+)", _replace_alnum, text)
    text = re.sub(r"(\d+)(?=[A-Za-z])", _replace_alnum, text)

    # everything left: value reading (digit-by-digit for long/zero-led runs)
    text = re.sub(r"\d+", lambda m: _number_value_to_zh(m.group(0)), text)

    return text


# wave dashes / ellipsis


def _is_numeric_range_char(ch: str) -> bool:
    if not ch:
        return False
    if ch.isdigit():
        return True
    return ch in "零〇一二三四五六七八九十百千万亿兆两"


def _normalize_wave_dashes_for_tts(text: str) -> str:
    if not text:
        return text
    text = text.replace("～", "~").replace("〜", "~")
    if "~" not in text:
        return text
    text = re.sub(r"~{2,}", "~", text)
    out: List[str] = []
    length = len(text)
    for idx, ch in enumerate(text):
        if ch != "~":
            out.append(ch)
            continue
        prev = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < length else ""
        has_left_space = bool(prev) and prev.isspace()
        has_right_space = bool(next_ch) and next_ch.isspace()
        left_idx = idx - 1
        while left_idx >= 0 and text[left_idx].isspace():
            left_idx -= 1
        left = text[left_idx] if left_idx >= 0 else ""
        right_idx = idx + 1
        while right_idx < length and text[right_idx].isspace():
            right_idx += 1
        right = text[right_idx] if right_idx < length else ""

        if has_left_space or has_right_space:
            if _is_chinese_char(left) and _is_chinese_char(right):
                out.append("、")
            continue
        if _is_numeric_range_char(left) and _is_numeric_range_char(right):
            out.append("到")
            continue
        # Tone-elongation tildes (哇~) and everything else drop silently.
    return "".join(out)


def _normalize_ellipsis_for_tts(text: str) -> str:
    if not text:
        return text
    # Keep short ellipsis as-is, but collapse long runs that can trigger cutoff bugs.
    # Canonicalize collapsed runs to a single midline ellipsis.
    return re.sub(r"[…⋯]{3,}", "⋯", text)


def prepare_tts_text(text: str, *, add_short_punct: bool = False) -> str:
    """Punctuation/quote/space preparation for a chunk.

    Expects NFKC-normalized, simplified, number-normalized input (the pipeline
    handles those first).
    """
    if not text:
        return ""
    has_dash_run = bool(_DASH_RUN_RE.search(text))
    text = _DASH_RUN_RE.sub(" ", text)
    text = _strip_format_chars(text)
    if add_short_punct:
        text = _ensure_line_tail_punct(text)
    text = _strip_cjk_quotes(text)
    text = _strip_double_quotes(text)
    text = _strip_single_quotes(text)
    text = _normalize_ellipsis_for_tts(text)
    text = _normalize_wave_dashes_for_tts(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _chinese_space_to_pause(text, allow_full_stop=not has_dash_run)
    if add_short_punct:
        text = _append_tail_punct_if_missing(text)
    return text


def _compact_ellipsis_candidate(text: str) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return ""
    quote_chars = _CJK_QUOTE_CHARS | _DOUBLE_QUOTE_CHARS | _SINGLE_QUOTE_CHARS
    if quote_chars:
        compact = "".join(ch for ch in compact if ch not in quote_chars)
    return compact


def _ellipsis_only_run_length(text: str) -> int:
    compact = _compact_ellipsis_candidate(text)
    if not compact:
        return 0
    if _ELLIPSIS_ONLY_RE.fullmatch(compact):
        return len(compact)
    return 0


def _ellipsis_pause_ms(run_length: int) -> int:
    if run_length <= 0:
        return 0
    total = _ELLIPSIS_PAUSE_BASE_MS + (run_length - 1) * _ELLIPSIS_PAUSE_STEP_MS
    return max(_ELLIPSIS_PAUSE_BASE_MS, min(_ELLIPSIS_PAUSE_MAX_MS, total))


def _resolve_output_sample_rate(
    *,
    model: Any,
    manifest: Optional[dict] = None,
) -> int:
    if isinstance(manifest, dict):
        try:
            sample_rate = int(manifest.get("sample_rate") or 0)
            if sample_rate > 0:
                return sample_rate
        except (TypeError, ValueError):
            pass
    try:
        sample_rate = int(getattr(model, "sample_rate", 0) or 0)
        if sample_rate > 0:
            return sample_rate
    except (TypeError, ValueError):
        pass
    return _DEFAULT_OUTPUT_SAMPLE_RATE


def _silence_audio(duration_ms: int, sample_rate: int) -> np.ndarray:
    if duration_ms <= 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    sample_count = int(round(sample_rate * (duration_ms / 1000.0)))
    if sample_count <= 0:
        sample_count = 1
    return np.zeros(sample_count, dtype=np.float32)


def _prepare_tts_pipeline(
    chunk_text: str,
    *,
    add_short_punct: bool = True,
) -> TtsPipeline:
    simplified = _normalize_traditional(_nfkc_preserve_ellipsis(chunk_text))
    after_numbers = _normalize_numbers(simplified)
    prepared = prepare_tts_text(after_numbers, add_short_punct=add_short_punct)
    return TtsPipeline(
        simplified=simplified,
        after_numbers=after_numbers,
        prepared=prepared,
    )


def load_book_chapters(book_dir: Path) -> List[ChapterInput]:
    toc_path = book_dir / "clean" / "toc.json"
    if not toc_path.exists():
        toc_path = book_dir / "toc.json"
    if not toc_path.exists():
        raise FileNotFoundError(f"Missing toc.json at {toc_path}")

    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    entries = toc.get("chapters", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("toc.json contains no chapters.")

    headings_by_source_index: Dict[int, List[str]] = {}
    headings_by_filename: Dict[str, List[str]] = {}
    heading_categories_by_source_index: Dict[int, Dict[str, str]] = {}
    heading_categories_by_filename: Dict[str, Dict[str, str]] = {}
    raw_toc_path = book_dir / "toc.json"
    if raw_toc_path.exists():
        raw_toc = json.loads(raw_toc_path.read_text(encoding="utf-8"))
        raw_entries = raw_toc.get("chapters", [])
        if isinstance(raw_entries, list):
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, dict):
                    continue
                normalized_headings = _normalize_heading_lines(raw_entry.get("headings"))
                if not normalized_headings:
                    normalized_headings = []
                normalized_categories = _normalize_heading_categories(
                    raw_entry.get("heading_categories")
                )
                try:
                    source_index = int(raw_entry.get("index"))
                except (TypeError, ValueError):
                    source_index = -1
                if source_index > 0:
                    if (
                        normalized_headings
                        and source_index not in headings_by_source_index
                    ):
                        headings_by_source_index[source_index] = normalized_headings
                    if (
                        normalized_categories
                        and source_index not in heading_categories_by_source_index
                    ):
                        heading_categories_by_source_index[source_index] = (
                            normalized_categories
                        )
                raw_rel = str(raw_entry.get("path") or "")
                raw_name = Path(raw_rel).name
                if raw_name:
                    if normalized_headings and raw_name not in headings_by_filename:
                        headings_by_filename[raw_name] = normalized_headings
                    if (
                        normalized_categories
                        and raw_name not in heading_categories_by_filename
                    ):
                        heading_categories_by_filename[raw_name] = normalized_categories

    chapters: List[ChapterInput] = []
    for fallback_idx, entry in enumerate(entries, start=1):
        rel = entry.get("path")
        if not rel:
            continue
        path = book_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing chapter file: {path}")

        text = read_clean_text(path)
        if not text.strip():
            continue

        index = int(entry.get("index") or fallback_idx)
        title = str(entry.get("title") or f"Chapter {index}")
        chapter_id = chapter_id_from_path(index, title, rel)
        heading_items = _normalize_heading_lines(entry.get("headings"))
        heading_category_map = _normalize_heading_categories(
            entry.get("heading_categories")
        )
        try:
            source_index = int(entry.get("source_index"))
        except (TypeError, ValueError):
            source_index = -1
        if not heading_items and source_index > 0:
            heading_items = headings_by_source_index.get(source_index, [])
        if not heading_items:
            heading_items = headings_by_filename.get(Path(rel).name, [])
        if not heading_category_map and source_index > 0:
            heading_category_map = heading_categories_by_source_index.get(
                source_index, {}
            )
        if not heading_category_map:
            heading_category_map = heading_categories_by_filename.get(
                Path(rel).name, {}
            )
        heading_category_map = _with_chapter_title_heading_category(
            title,
            heading_items,
            heading_category_map,
        )

        chapters.append(
            ChapterInput(
                index=index,
                id=chapter_id,
                title=title,
                text=_normalize_text(text),
                path=rel,
                headings=tuple(heading_items),
                heading_categories=dict(heading_category_map),
            )
        )

    if not chapters:
        raise ValueError("No chapter text found.")

    return chapters


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_status(out_dir: Path, stage: str, detail: Optional[str] = None) -> None:
    payload = {"stage": stage, "updated_unix": int(time.time())}
    if detail:
        payload["detail"] = detail
    atomic_write_json(out_dir / "status.json", payload)


def _load_voice_map(path: Optional[Path]) -> dict:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Voice map not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Voice map must be a JSON object: {path}")
    chapters = data.get("chapters")
    if chapters is not None and not isinstance(chapters, dict):
        raise ValueError(f"Voice map chapters must be an object: {path}")
    return data


def _normalize_voice_id(value: Optional[str], default_voice: str) -> str:
    if value is None:
        return default_voice
    cleaned = str(value).strip()
    if not cleaned:
        return default_voice
    if cleaned.lower() == "default":
        return default_voice
    return cleaned


def write_chunk_files(
    chunks: Sequence[str], chunk_dir: Path, overwrite: bool = False
) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in chunk_dir.glob("*.txt"):
            path.unlink()

    for idx, chunk in enumerate(chunks, start=1):
        path = chunk_dir / f"{idx:06d}.txt"
        if overwrite or not path.exists():
            path.write_text(chunk.rstrip() + "\n", encoding="utf-8")

    if overwrite:
        for path in chunk_dir.glob("*.txt"):
            stem = path.stem
            if stem.isdigit() and int(stem) > len(chunks):
                path.unlink()


def _prepare_manifest(
    chapters: Sequence[ChapterInput],
    out_dir: Path,
    voice: str,
    max_chars: int,
    pad_ms: int,
    rechunk: bool,
    min_chars: int = 0,
) -> Tuple[dict, List[List[str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = out_dir / "chunks"
    manifest_path = out_dir / "manifest.json"

    if rechunk and chunk_root.exists():
        shutil.rmtree(chunk_root)

    if manifest_path.exists() and not rechunk:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("chunk_mode") != "chinese":
            raise ValueError(
                "manifest.json chunk_mode differs from requested. "
                "Run with --rechunk to regenerate."
            )
        if int(manifest.get("max_chars") or 0) != int(max_chars):
            raise ValueError(
                "manifest.json max_chars differs from requested. "
                "Run with --rechunk to regenerate."
            )
        if int(manifest.get("min_chars") or 0) != int(min_chars):
            raise ValueError(
                "manifest.json min_chars differs from requested. "
                "Run with --rechunk to regenerate."
            )
        manifest_chapters = manifest.get("chapters", [])
        if not isinstance(manifest_chapters, list) or not manifest_chapters:
            raise ValueError("manifest.json contains no chapters.")
        if len(manifest_chapters) != len(chapters):
            raise ValueError(
                "manifest.json chapter count differs from input. "
                "Run with --rechunk to regenerate."
            )
        chapter_chunks: List[List[str]] = []
        for chapter_idx, (ch_manifest, chapter) in enumerate(
            zip(manifest_chapters, chapters)
        ):
            if ch_manifest.get("text_sha256") != sha256_str(chapter.text):
                raise ValueError(
                    "manifest.json text hash differs from input. "
                    "Run with --rechunk to regenerate."
                )
            chunks = ch_manifest.get("chunks", [])
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("manifest.json missing chunks. Run with --rechunk.")
            chunk_spans = _coerce_span_pairs(ch_manifest.get("chunk_spans") or [])
            computed_authoritative = len(chunk_spans) == len(chunks)
            computed_pause = (
                compute_chunk_pause_multipliers(
                    chapter.text,
                    chunk_spans,
                    heading_lines=chapter.headings,
                    heading_categories=chapter.heading_categories,
                )
                if computed_authoritative
                else [1] * len(chunks)
            )
            ch_manifest["headings"] = list(chapter.headings)
            ch_manifest["heading_categories"] = dict(chapter.heading_categories)
            legacy_pause = _legacy_pause_multipliers(
                ch_manifest.get("chunk_section_breaks"),
                len(chunks),
                add_chapter_boundary=chapter_idx < len(chapters) - 1,
            )
            # When computed_pause comes from authoritative chunk spans, trust
            # it directly — otherwise max() with the legacy default of 1 would
            # silently re-pad mid-sentence chunks that we want set to 0.
            fallback_pause = (
                list(computed_pause)
                if computed_authoritative
                else [
                    max(computed_pause[idx], legacy_pause[idx])
                    for idx in range(len(chunks))
                ]
            )
            ch_manifest["pause_multipliers"] = _normalize_pause_multipliers(
                fallback_pause,
                len(chunks),
                fallback=fallback_pause,
            )
            chapter_chunks.append([str(c) for c in chunks])
        _apply_chapter_boundary_pause_multipliers(manifest_chapters)
        manifest["voice"] = voice
        manifest["pad_ms"] = int(pad_ms)
        atomic_write_json(manifest_path, manifest)
        return manifest, chapter_chunks

    chapter_chunks: List[List[str]] = []
    manifest_chapters: List[dict] = []
    for ch in chapters:
        spans = make_chunk_spans(
            ch.text,
            max_chars=max_chars,
            chunk_mode="chinese",
            min_chars=min_chars,
        )
        pause_multipliers = compute_chunk_pause_multipliers(
            ch.text,
            spans,
            heading_lines=ch.headings,
            heading_categories=ch.heading_categories,
        )
        section_breaks = [
            idx for idx, chv in enumerate(ch.text) if chv == SECTION_BREAK
        ]
        chunk_section_breaks = [False] * len(spans)
        if section_breaks and spans:
            break_idx = 0
            for idx, (_start, end) in enumerate(spans):
                next_start = spans[idx + 1][0] if idx + 1 < len(spans) else len(ch.text)
                while (
                    break_idx < len(section_breaks) and section_breaks[break_idx] < end
                ):
                    break_idx += 1
                if (
                    break_idx < len(section_breaks)
                    and section_breaks[break_idx] < next_start
                ):
                    chunk_section_breaks[idx] = True
        chunks = [strip_section_breaks(ch.text[start:end]) for start, end in spans]
        if not chunks:
            raise ValueError(f"No chunks generated for chapter: {ch.id}")
        chapter_chunks.append(chunks)
        manifest_chapters.append(
            {
                "index": ch.index,
                "id": ch.id,
                "title": ch.title,
                "path": ch.path,
                "text_sha256": sha256_str(ch.text),
                "chunks": chunks,
                "chunk_spans": [[start, end] for start, end in spans],
                "headings": list(ch.headings),
                "heading_categories": dict(ch.heading_categories),
                "chunk_section_breaks": chunk_section_breaks,
                "pause_multipliers": pause_multipliers,
                "durations_ms": [None] * len(chunks),
            }
        )
    _apply_chapter_boundary_pause_multipliers(manifest_chapters)

    manifest = {
        "created_unix": int(time.time()),
        "voice": voice,
        "max_chars": int(max_chars),
        "min_chars": int(min_chars),
        "pad_ms": int(pad_ms),
        "section_pad_ms": int(pad_ms) * _SECTION_PAD_MULT,
        "chapter_pad_ms": int(pad_ms) * _CHAPTER_PAD_MULT,
        "chunk_mode": "chinese",
        "chapters": manifest_chapters,
    }
    atomic_write_json(manifest_path, manifest)

    for ch_entry, chunks in zip(manifest["chapters"], chapter_chunks):
        chunk_dir = chunk_root / ch_entry["id"]
        write_chunk_files(chunks, chunk_dir, overwrite=True)

    return manifest, chapter_chunks


def _is_valid_wav(path: Path) -> bool:
    try:
        with sf.SoundFile(path) as handle:
            return handle.channels == 1 and handle.frames > 0
    except Exception:
        return False


def _wav_duration_ms(path: Path) -> int:
    with sf.SoundFile(path) as handle:
        frames = handle.frames
        rate = handle.samplerate
    if rate <= 0:
        return 0
    return int(round(frames * 1000.0 / rate))


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def build_concat_file(
    segment_paths: List[Path], concat_path: Path, base_dir: Path
) -> None:
    lines = []
    for p in segment_paths:
        rel = p.relative_to(base_dir).as_posix()
        lines.append(f"file '{rel}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chapters_ffmeta(
    chapters: Sequence[Tuple[str, int]], ffmeta_path: Path
) -> None:
    def _ffmeta_escape(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("\n", "\\\n")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#")
        )

    out = [";FFMETADATA1"]
    t = 0
    for title, d in chapters:
        start = t
        end = t + max(int(d), 1)
        out.append("")
        out.append("[CHAPTER]")
        out.append("TIMEBASE=1/1000")
        out.append(f"START={start}")
        out.append(f"END={end}")
        out.append(f"title={_ffmeta_escape(str(title))}")
        t = end
    ffmeta_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def chunk_book(
    book_dir: Path,
    out_dir: Optional[Path] = None,
    voice: str = "voice",
    max_chars: int = 0,
    pad_ms: int = 350,
    chunk_mode: str = "chinese",
    rechunk: bool = True,
    min_chars: int = 0,
) -> dict:
    _ = chunk_mode
    if out_dir is None:
        out_dir = book_dir / "tts"
    chapters = load_book_chapters(book_dir)
    manifest, _chapter_chunks = _prepare_manifest(
        chapters=chapters,
        out_dir=out_dir,
        voice=voice,
        max_chars=max_chars,
        pad_ms=pad_ms,
        rechunk=rechunk,
        min_chars=min_chars,
    )
    return manifest


def synthesize_book(
    book_dir: Path,
    voice: VoiceConfig,
    out_dir: Optional[Path] = None,
    max_chars: int = 0,
    pad_ms: int = 350,
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    only_chapter_ids: Optional[set[str]] = None,
    min_chars: int = 0,
) -> int:
    chapters = load_book_chapters(book_dir)
    if out_dir is None:
        out_dir = book_dir / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    if base_dir is None:
        base_dir = Path.cwd()

    try:
        voice_map = _load_voice_map(voice_map_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        manifest, chapter_chunks = _prepare_manifest(
            chapters=chapters,
            out_dir=out_dir,
            voice=voice.name,
            max_chars=max_chars,
            pad_ms=pad_ms,
            rechunk=rechunk,
            min_chars=min_chars,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    seg_dir = out_dir / "segments"
    manifest_path = out_dir / "manifest.json"
    concat_path = out_dir / "concat.txt"
    chapters_path = out_dir / "chapters.ffmeta"
    if rechunk and seg_dir.exists():
        shutil.rmtree(seg_dir)

    default_voice = voice.name
    if voice_map:
        default_voice = _normalize_voice_id(voice_map.get("default"), default_voice)

    chapter_voice_map: Dict[str, str] = {}
    voice_overrides: Dict[str, str] = {}
    raw_overrides = voice_map.get("chapters", {}) if voice_map else {}
    for entry in manifest.get("chapters", []):
        chapter_id = entry.get("id") or "chapter"
        raw_value = (
            raw_overrides.get(chapter_id) if isinstance(raw_overrides, dict) else None
        )
        selected = _normalize_voice_id(raw_value, default_voice)
        chapter_voice_map[chapter_id] = selected
        entry["voice"] = selected
        if voice_map and selected != default_voice:
            voice_overrides[chapter_id] = selected
    manifest["voice"] = default_voice
    if voice_map:
        manifest["voice_overrides"] = voice_overrides
    manifest["pad_ms"] = int(pad_ms)
    manifest["section_pad_ms"] = int(pad_ms) * _SECTION_PAD_MULT
    manifest["chapter_pad_ms"] = int(pad_ms) * _CHAPTER_PAD_MULT
    atomic_write_json(manifest_path, manifest)

    voice_configs: Dict[str, VoiceConfig] = {}
    try:
        for voice_id in sorted(set(chapter_voice_map.values())):
            if voice_id == voice.name:
                voice_configs[voice_id] = voice
            else:
                voice_configs[voice_id] = voice_util.resolve_voice_config(
                    voice=voice_id,
                    base_dir=base_dir,
                )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    write_status(out_dir, "loading_model", "Loading TTS model")

    model = synth_qwen.get_runtime()

    write_status(out_dir, "synthesizing")

    selected_ids = set(only_chapter_ids) if only_chapter_ids else None
    selected_indices = [
        idx
        for idx, entry in enumerate(manifest["chapters"])
        if not selected_ids or (entry.get("id") or "chapter") in selected_ids
    ]
    if selected_ids and not selected_indices:
        sys.stderr.write("No matching chapters found for synthesis.\n")
        return 2

    total_chunks = sum(len(chapter_chunks[idx]) for idx in selected_indices)
    if total_chunks <= 0:
        sys.stderr.write("No chunks selected for synthesis.\n")
        return 2

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    segment_paths: List[Path] = []

    with progress:
        overall_task = progress.add_task("Total", total=total_chunks)
        chapter_task = progress.add_task("Chapter", total=0)

        chapter_entries = manifest.get("chapters", [])
        chapter_total_count = (
            len(chapter_entries) if isinstance(chapter_entries, list) else 0
        )
        for chapter_idx, (ch_entry, chunks) in enumerate(
            zip(manifest["chapters"], chapter_chunks)
        ):
            chapter_id = ch_entry.get("id") or "chapter"
            chapter_title = ch_entry.get("title") or chapter_id
            chapter_total = len(chunks)
            chunk_section_breaks = ch_entry.get("chunk_section_breaks") or []
            chapter_pause_multipliers = _normalize_pause_multipliers(
                ch_entry.get("pause_multipliers"),
                chapter_total,
                fallback=_legacy_pause_multipliers(
                    chunk_section_breaks,
                    chapter_total,
                    add_chapter_boundary=chapter_idx < chapter_total_count - 1,
                ),
            )
            if chapter_idx < chapter_total_count - 1 and chapter_pause_multipliers:
                chapter_pause_multipliers[-1] = max(
                    chapter_pause_multipliers[-1], _TITLE_BREAK_PAUSE_MULTIPLIER
                )
            ch_entry["pause_multipliers"] = chapter_pause_multipliers
            if selected_ids and chapter_id not in selected_ids:
                continue
            progress.update(
                chapter_task,
                total=chapter_total,
                completed=0,
                description=f"{chapter_id}: {chapter_title}",
            )

            chapter_seg_dir = seg_dir / chapter_id
            voice_id = chapter_voice_map.get(chapter_id, default_voice)
            voice_config = voice_configs.get(voice_id)

            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                seg_path = chapter_seg_dir / f"{chunk_idx:06d}.wav"
                progress.update(
                    chapter_task,
                    description=f"{chapter_id}: {chapter_title} ({chunk_idx}/{chapter_total})",
                )
                pause_multiplier = 1
                if chunk_idx - 1 < len(chapter_pause_multipliers):
                    try:
                        pause_multiplier = max(
                            0, int(chapter_pause_multipliers[chunk_idx - 1])
                        )
                    except (TypeError, ValueError):
                        pause_multiplier = 1
                pad_ms_total = int(pad_ms) * pause_multiplier
                raw_ellipsis_run = _ellipsis_only_run_length(str(chunk_text))

                if (
                    seg_path.exists()
                    and _is_valid_wav(seg_path)
                    and raw_ellipsis_run <= 0
                ):
                    segment_paths.append(seg_path)
                    dms = _wav_duration_ms(seg_path)
                    if ch_entry["durations_ms"][chunk_idx - 1] != dms:
                        ch_entry["durations_ms"][chunk_idx - 1] = dms
                        atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                if raw_ellipsis_run > 0:
                    sample_rate = _resolve_output_sample_rate(
                        model=model,
                        manifest=manifest,
                    )
                    silence_ms = _ellipsis_pause_ms(raw_ellipsis_run) + pad_ms_total
                    audio = _silence_audio(silence_ms, sample_rate)
                    _write_wav(seg_path, audio, sample_rate)
                    segment_paths.append(seg_path)

                    dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
                    ch_entry["durations_ms"][chunk_idx - 1] = dms
                    if manifest.get("sample_rate") != sample_rate:
                        manifest["sample_rate"] = sample_rate
                    atomic_write_json(manifest_path, manifest)

                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                pipeline = _prepare_tts_pipeline(chunk_text, add_short_punct=True)
                tts_text = pipeline.prepared
                if not tts_text:
                    ch_entry["durations_ms"][chunk_idx - 1] = 0
                    atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue
                ellipsis_run = _ellipsis_only_run_length(tts_text)
                if ellipsis_run > 0:
                    sample_rate = _resolve_output_sample_rate(
                        model=model,
                        manifest=manifest,
                    )
                    silence_ms = _ellipsis_pause_ms(ellipsis_run) + pad_ms_total
                    audio = _silence_audio(silence_ms, sample_rate)
                    _write_wav(seg_path, audio, sample_rate)
                    segment_paths.append(seg_path)

                    dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
                    ch_entry["durations_ms"][chunk_idx - 1] = dms
                    if manifest.get("sample_rate") != sample_rate:
                        manifest["sample_rate"] = sample_rate
                    atomic_write_json(manifest_path, manifest)

                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                if voice_config is None:
                    sys.stderr.write(
                        f"Missing voice config for {chapter_id} ({chunk_idx}/{chapter_total}).\n"
                    )
                    ch_entry["durations_ms"][chunk_idx - 1] = 0
                    atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue
                audio, sample_rate = synth_qwen.generate_chunk(
                    model, tts_text, voice_config
                )
                if audio.size == 0:
                    sys.stderr.write(
                        f"Skipping empty audio {chapter_id} ({chunk_idx}/{chapter_total}).\n"
                    )
                    ch_entry["durations_ms"][chunk_idx - 1] = 0
                    atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                audio = audio.flatten()
                if pad_ms_total > 0:
                    pad_samples = int(round(sample_rate * (pad_ms_total / 1000.0)))
                    if pad_samples > 0:
                        pad = np.zeros(pad_samples, dtype=audio.dtype)
                        audio = np.concatenate([audio, pad])

                _write_wav(seg_path, audio, sample_rate)
                segment_paths.append(seg_path)

                dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
                ch_entry["durations_ms"][chunk_idx - 1] = dms
                if manifest.get("sample_rate") != sample_rate:
                    manifest["sample_rate"] = sample_rate
                atomic_write_json(manifest_path, manifest)

                progress.advance(chapter_task, 1)
                progress.advance(overall_task, 1)

    build_concat_file(segment_paths, concat_path, base_dir=out_dir)

    chapter_meta: List[Tuple[str, int]] = []
    for ch_entry in manifest["chapters"]:
        title = ch_entry.get("title") or ch_entry.get("id") or "Chapter"
        durations = ch_entry.get("durations_ms", [])
        total_ms = sum(int(d or 0) for d in durations)
        chapter_meta.append((title, total_ms))

    build_chapters_ffmeta(chapter_meta, chapters_path)
    write_status(out_dir, "done")
    return 0


def synthesize_book_sample(
    book_dir: Path,
    voice: VoiceConfig,
    out_dir: Optional[Path] = None,
    max_chars: int = 0,
    pad_ms: int = 350,
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    min_chars: int = 0,
) -> int:
    chapters = load_book_chapters(book_dir)
    if not chapters:
        sys.stderr.write("No chapters found for sampling.\n")
        return 2
    if out_dir is None:
        out_dir = book_dir / "tts"

    sample_id = chapters[0].id
    sample_dir = out_dir / "segments" / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    manifest_path = out_dir / "manifest.json"
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
            atomic_write_json(manifest_path, manifest)

    return synthesize_book(
        book_dir=book_dir,
        voice=voice,
        out_dir=out_dir,
        max_chars=max_chars,
        pad_ms=pad_ms,
        rechunk=rechunk,
        voice_map_path=voice_map_path,
        base_dir=base_dir,
        only_chapter_ids={sample_id},
        min_chars=min_chars,
    )


def synthesize_chunk(
    out_dir: Path,
    chapter_id: str,
    chunk_index: int,
    voice: Optional[str] = None,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> dict:
    if base_dir is None:
        base_dir = Path.cwd()
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chapters = manifest.get("chapters", [])
    if not isinstance(chapters, list):
        raise ValueError("manifest.json chapters missing or invalid")

    entry = None
    entry_index = -1
    for idx, item in enumerate(chapters):
        if isinstance(item, dict) and item.get("id") == chapter_id:
            entry = item
            entry_index = idx
            break
    if entry is None:
        raise ValueError(f"Unknown chapter_id: {chapter_id}")

    chunks = entry.get("chunks")
    if not isinstance(chunks, list):
        chunks = []
    spans = entry.get("chunk_spans")
    if not isinstance(spans, list):
        spans = []
    chunk_count = len(chunks) or len(spans)
    if chunk_count <= 0:
        chunk_dir = out_dir / "chunks" / chapter_id
        if chunk_dir.exists():
            chunk_count = len([p for p in chunk_dir.glob("*.txt") if p.stem.isdigit()])
    if chunk_count <= 0:
        raise ValueError(f"No chunks available for chapter: {chapter_id}")
    if chunk_index >= chunk_count:
        raise ValueError(f"chunk_index out of range for {chapter_id}")

    chunk_text: Optional[str] = None
    if chunks and chunk_index < len(chunks):
        chunk_text = str(chunks[chunk_index])
    if chunk_text is None:
        chunk_path = out_dir / "chunks" / chapter_id / f"{chunk_index + 1:06d}.txt"
        if chunk_path.exists():
            chunk_text = chunk_path.read_text(encoding="utf-8").rstrip("\n")
    if chunk_text is None:
        raise ValueError(f"Chunk text missing for {chapter_id} #{chunk_index + 1}")

    durations = entry.get("durations_ms")
    if not isinstance(durations, list) or len(durations) != chunk_count:
        durations = [None] * chunk_count
        entry["durations_ms"] = durations
    chunk_section_breaks = entry.get("chunk_section_breaks") or []
    span_pairs = _coerce_span_pairs(spans)
    pause_multipliers = entry.get("pause_multipliers")
    computed_pause = [1] * chunk_count
    if len(span_pairs) == chunk_count:
        rel_path = entry.get("path")
        if isinstance(rel_path, str) and rel_path:
            clean_path = (out_dir.parent / rel_path).resolve()
            if clean_path.exists():
                chapter_text = _normalize_text(read_clean_text(clean_path))
                heading_lines = entry.get("headings")
                if not isinstance(heading_lines, list):
                    heading_lines = []
                heading_categories = entry.get("heading_categories")
                if not isinstance(heading_categories, dict):
                    heading_categories = {}
                computed_pause = compute_chunk_pause_multipliers(
                    chapter_text,
                    span_pairs,
                    heading_lines=heading_lines,
                    heading_categories=heading_categories,
                )
    legacy_pause = _legacy_pause_multipliers(
        chunk_section_breaks,
        chunk_count,
        add_chapter_boundary=entry_index >= 0 and entry_index < len(chapters) - 1,
    )
    fallback_pause = [
        max(computed_pause[idx], legacy_pause[idx]) for idx in range(chunk_count)
    ]
    pause_multipliers = _normalize_pause_multipliers(
        fallback_pause,
        chunk_count,
        fallback=fallback_pause,
    )
    if entry_index >= 0 and entry_index < len(chapters) - 1 and pause_multipliers:
        pause_multipliers[-1] = max(
            pause_multipliers[-1], _TITLE_BREAK_PAUSE_MULTIPLIER
        )
    entry["pause_multipliers"] = pause_multipliers

    default_voice = voice or entry.get("voice") or manifest.get("voice") or ""
    if not default_voice:
        raise ValueError("Voice is required for synthesis.")

    voice_id = default_voice
    if voice_map_path:
        voice_map = _load_voice_map(voice_map_path)
        if voice_map:
            default_voice = _normalize_voice_id(voice_map.get("default"), default_voice)
            raw_voice = (
                voice_map.get("chapters", {}).get(chapter_id)
                if isinstance(voice_map.get("chapters"), dict)
                else None
            )
            voice_id = _normalize_voice_id(raw_voice, default_voice)
    else:
        # Explicit voice overrides chapter-level manifest voice for manual regen.
        if voice:
            voice_id = _normalize_voice_id(voice, default_voice)
        else:
            voice_id = _normalize_voice_id(entry.get("voice"), default_voice)

    config = voice_util.resolve_voice_config(voice=voice_id, base_dir=base_dir)
    model = synth_qwen.get_runtime()

    raw_ellipsis_run = _ellipsis_only_run_length(chunk_text)
    pipeline = _prepare_tts_pipeline(chunk_text, add_short_punct=True)
    tts_text = pipeline.prepared
    seg_path = out_dir / "segments" / chapter_id / f"{chunk_index + 1:06d}.wav"
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    pad_ms = int(manifest.get("pad_ms") or 0)
    pause_multiplier = 1
    try:
        pause_multiplier = max(0, int(pause_multipliers[chunk_index]))
    except (TypeError, ValueError, IndexError):
        pause_multiplier = 1
    pad_ms_total = pad_ms * pause_multiplier

    if not tts_text:
        if seg_path.exists():
            seg_path.unlink()
        durations[chunk_index] = 0
        atomic_write_json(manifest_path, manifest)
        return {
            "status": "skipped",
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "duration_ms": 0,
        }
    ellipsis_run = raw_ellipsis_run or _ellipsis_only_run_length(tts_text)
    if ellipsis_run > 0:
        sample_rate = _resolve_output_sample_rate(model=model, manifest=manifest)
        silence_ms = _ellipsis_pause_ms(ellipsis_run) + pad_ms_total
        audio = _silence_audio(silence_ms, sample_rate)
        _write_wav(seg_path, audio, sample_rate)
        dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
        durations[chunk_index] = dms
        if manifest.get("sample_rate") != sample_rate:
            manifest["sample_rate"] = sample_rate
        atomic_write_json(manifest_path, manifest)
        return {
            "status": "ok",
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "duration_ms": dms,
        }

    audio, sample_rate = synth_qwen.generate_chunk(model, tts_text, config)
    if audio.size == 0:
        durations[chunk_index] = 0
        atomic_write_json(manifest_path, manifest)
        return {
            "status": "skipped",
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "duration_ms": 0,
        }
    audio = audio.flatten()

    if pad_ms_total > 0:
        pad_samples = int(round(sample_rate * (pad_ms_total / 1000.0)))
        if pad_samples > 0:
            pad = np.zeros(pad_samples, dtype=audio.dtype)
            audio = np.concatenate([audio, pad])

    _write_wav(seg_path, audio, sample_rate)
    dms = int(round(audio.shape[0] * 1000.0 / sample_rate))
    durations[chunk_index] = dms
    if manifest.get("sample_rate") != sample_rate:
        manifest["sample_rate"] = sample_rate
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "ok",
        "chapter_id": chapter_id,
        "chunk_index": chunk_index,
        "duration_ms": dms,
    }
