"""Adapter for duk's F5-TTS MLX engine.

Mirrors the synthesis surface the pipeline consumes:
  - `get_runtime()` returns an opaque runtime object with a `sample_rate`
    attribute (consumed by `_resolve_output_sample_rate`).
  - `generate_chunk(runtime, text, voice)` returns
    `(np.ndarray float32 1d, sample_rate)`.

F5 is an infill model: it fills a fixed duration window sized up front. The
bundled duration predictor is unreliable cross-lingual, so the window comes
from an explicit Mandarin pacing heuristic (chars/sec + pause allowance).
Quality config defaults to the listening-approved rk4 @ 8 steps (32 NFE) at
fp32; env knobs exist for experiments:

  DUK_STEPS      ODE steps (default 8)
  DUK_METHOD     euler|midpoint|rk4 (default rk4)
  DUK_PRECISION  fp32|fp16 (default fp32)
  DUK_CFG        classifier-free guidance strength (default 2.0)
  DUK_CHARS_PER_SEC  Mandarin pacing for the duration window (default 4.5)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from duk.f5 import F5Engine, Voice

from .voice import VoiceConfig

ENV_STEPS = "DUK_STEPS"
ENV_METHOD = "DUK_METHOD"
ENV_PRECISION = "DUK_PRECISION"
ENV_CFG = "DUK_CFG"
ENV_CHARS_PER_SEC = "DUK_CHARS_PER_SEC"

DEFAULT_STEPS = 8
DEFAULT_METHOD = "rk4"
DEFAULT_PRECISION = "fp32"
DEFAULT_CFG = 2.0

SAMPLE_RATE = 24_000

ZH_CHARS_PER_SEC = 4.5
ZH_PAUSE_SECONDS = 0.3
ZH_TAIL_SLACK_SECONDS = 0.4
# NFKC folds fullwidth ，；：？！ to ASCII, so count both forms.
_ZH_PAUSE_RE = re.compile(r"[。，、；：？！,;:?!]")

# Trailing-window silence trim: F5 fills the fixed window, so an overshoot
# window ends in silence that must not reach the audiobook.
_TRIM_FRAME_MS = 20
_TRIM_HOP_MS = 10
_TRIM_THRESHOLD_DBFS = -45.0
_TRIM_KEEP_HEAD_MS = 60
_TRIM_KEEP_TAIL_MS = 140


def zh_target_seconds(text: str) -> float:
    """Duration window for a Mandarin chunk: pacing + pause allowance."""
    chars = len(re.sub(r"\s", "", text))
    pauses = len(_ZH_PAUSE_RE.findall(text))
    return chars / _chars_per_sec() + pauses * ZH_PAUSE_SECONDS + ZH_TAIL_SLACK_SECONDS


def _chars_per_sec() -> float:
    raw = os.environ.get(ENV_CHARS_PER_SEC)
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return ZH_CHARS_PER_SEC


def _default_steps() -> int:
    raw = os.environ.get(ENV_STEPS)
    if raw:
        try:
            return max(2, int(raw))
        except ValueError:
            pass
    return DEFAULT_STEPS


def _default_method() -> str:
    raw = (os.environ.get(ENV_METHOD) or "").strip().lower()
    if raw in {"euler", "midpoint", "rk4"}:
        return raw
    return DEFAULT_METHOD


def _default_precision() -> str:
    raw = (os.environ.get(ENV_PRECISION) or "").strip().lower()
    if raw in {"fp32", "fp16"}:
        return raw
    return DEFAULT_PRECISION


def _default_cfg() -> float:
    raw = os.environ.get(ENV_CFG)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_CFG


@dataclass
class F5Runtime:
    engine: F5Engine
    sample_rate: int = SAMPLE_RATE
    _voices: dict[str, Voice] = field(default_factory=dict)

    def voice_for(self, config: VoiceConfig) -> Voice:
        key = str(Path(config.ref_audio).resolve())
        cached = self._voices.get(key)
        if cached is not None:
            return cached
        if not config.ref_text:
            raise ValueError(
                f"Voice '{config.name}' has no ref_text; F5 requires the exact "
                "transcript of the reference audio."
            )
        audio_path = Path(config.ref_audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Voice audio not found: {audio_path}")
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Voice audio must be {SAMPLE_RATE} Hz mono (got {sr} Hz): "
                f"{audio_path}. Re-clone the voice through duk."
            )
        mono = audio.mean(axis=1)
        voice = self.engine.prepare_voice(mono, config.ref_text)
        self._voices[key] = voice
        return voice


_runtime_cache: dict[str, F5Runtime] = {}


def get_runtime() -> F5Runtime:
    """Load (or fetch from cache) the F5 engine at the configured precision."""
    precision = _default_precision()
    cached = _runtime_cache.get(precision)
    if cached is not None:
        return cached
    engine = F5Engine.from_pretrained(precision=precision)
    runtime = F5Runtime(engine=engine)
    _runtime_cache[precision] = runtime
    return runtime


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = _TRIM_THRESHOLD_DBFS,
    keep_head_ms: int = _TRIM_KEEP_HEAD_MS,
    keep_tail_ms: int = _TRIM_KEEP_TAIL_MS,
) -> np.ndarray:
    """Trim leading/trailing silence via a moving-RMS envelope.

    Cuts are made only into frames below an absolute dBFS floor, with a keep
    margin on both sides so soft onsets and trailing fricatives survive.
    """
    if audio.size == 0:
        return audio
    frame = max(1, int(sample_rate * _TRIM_FRAME_MS / 1000))
    hop = max(1, int(sample_rate * _TRIM_HOP_MS / 1000))
    if audio.size <= frame:
        return audio
    starts = np.arange(0, audio.size - frame + 1, hop)
    frames = np.stack([audio[s : s + frame] for s in starts])
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1) + 1e-12)
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    active = np.flatnonzero(db > threshold_dbfs)
    if active.size == 0:
        return audio
    head_margin = int(sample_rate * keep_head_ms / 1000)
    tail_margin = int(sample_rate * keep_tail_ms / 1000)
    start = max(0, int(starts[active[0]]) - head_margin)
    end = min(audio.size, int(starts[active[-1]]) + frame + tail_margin)
    if end <= start:
        return audio
    return audio[start:end]


def generate_chunk(
    runtime: F5Runtime,
    text: str,
    voice: VoiceConfig,
    *,
    steps: Optional[int] = None,
    method: Optional[str] = None,
    cfg_strength: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Synthesize one chunk; returns (audio_float32_1d, sample_rate)."""
    cleaned = text.strip()
    if not cleaned:
        return np.zeros(0, dtype=np.float32), runtime.sample_rate

    prepared_voice = runtime.voice_for(voice)
    audio = runtime.engine.synthesize(
        prepared_voice,
        cleaned,
        zh_target_seconds(cleaned),
        steps=steps if steps is not None else _default_steps(),
        method=method if method is not None else _default_method(),
        cfg_strength=cfg_strength if cfg_strength is not None else _default_cfg(),
        seed=seed,
    )
    audio = trim_silence(audio, runtime.sample_rate)
    return audio, runtime.sample_rate
