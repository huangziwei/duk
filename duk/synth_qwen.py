"""Adapter for duk's Qwen3-TTS MLX engine (mlx-audio).

Mirrors the synthesis surface the pipeline consumes:
  - `get_runtime()` returns an opaque runtime object with a `sample_rate`
    attribute (consumed by `_resolve_output_sample_rate`).
  - `generate_chunk(runtime, text, voice)` returns
    `(np.ndarray float32 1d, sample_rate)`.

Qwen3-TTS-0.6B-Base is an autoregressive multi-codebook LM (24 kHz audio,
12.5 Hz acoustic tokens). It clones in-context: the reference audio and its
transcript are prefilled, then the target text is spoken in the reference
voice. Unlike F5 there is no fixed-duration window (the talker emits EOS, so
each chunk self-lengths) and no ODE/CFG/step knobs. The reference must be a
same-language (Chinese) voice — Qwen is not a cross-lingual cloner.

Quantized 6-bit weights are the default (listening-approved: ~3.6x real time
on M4 Pro, quality indistinguishable from bf16 by ear and ASR). Env knobs:

  DUK_QWEN_MODEL        HF repo (default mlx-community/...-0.6B-Base-6bit)
  DUK_QWEN_TEMPERATURE  sampling temperature (default 0.9)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
import soundfile as sf

from .voice import VoiceConfig

ENV_MODEL = "DUK_QWEN_MODEL"
ENV_TEMPERATURE = "DUK_QWEN_TEMPERATURE"

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit"
DEFAULT_TEMPERATURE = 0.9

SAMPLE_RATE = 24_000

# Trailing/leading silence trim: keeps concatenated chunks tight in merge.
# Cuts only into frames below an absolute dBFS floor, with a keep margin on
# both sides so soft onsets and trailing fricatives survive.
_TRIM_FRAME_MS = 20
_TRIM_HOP_MS = 10
_TRIM_THRESHOLD_DBFS = -45.0
_TRIM_KEEP_HEAD_MS = 60
_TRIM_KEEP_TAIL_MS = 140


def _model_repo() -> str:
    return (os.environ.get(ENV_MODEL) or "").strip() or DEFAULT_MODEL


def _temperature() -> float:
    raw = os.environ.get(ENV_TEMPERATURE)
    if raw:
        try:
            value = float(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
    return DEFAULT_TEMPERATURE


@dataclass
class QwenRuntime:
    model: object
    sample_rate: int = SAMPLE_RATE
    temperature: float = DEFAULT_TEMPERATURE
    _refs: dict[str, mx.array] = field(default_factory=dict)

    def ref_for(self, config: VoiceConfig) -> mx.array:
        """Decoded reference waveform for a voice, cached per audio path.

        The model's internal ICL cache keys on the array's (size, sum), so
        reusing the same cached array across chunks reuses the reference
        encode instead of recomputing it every chunk.
        """
        key = str(Path(config.ref_audio).resolve())
        cached = self._refs.get(key)
        if cached is not None:
            return cached
        audio_path = Path(config.ref_audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Voice audio not found: {audio_path}")
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Voice audio must be {SAMPLE_RATE} Hz mono (got {sr} Hz): "
                f"{audio_path}. Re-clone the voice through duk."
            )
        ref = mx.array(audio.mean(axis=1))
        self._refs[key] = ref
        return ref


_runtime_cache: dict[str, QwenRuntime] = {}


def get_runtime() -> QwenRuntime:
    """Load (or fetch from cache) the Qwen3-TTS engine at the configured repo."""
    from mlx_audio.tts.utils import load_model

    repo = _model_repo()
    cached = _runtime_cache.get(repo)
    if cached is not None:
        return cached
    model = load_model(repo)
    runtime = QwenRuntime(
        model=model,
        sample_rate=int(getattr(model, "sample_rate", SAMPLE_RATE)),
        temperature=_temperature(),
    )
    _runtime_cache[repo] = runtime
    return runtime


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = _TRIM_THRESHOLD_DBFS,
    keep_head_ms: int = _TRIM_KEEP_HEAD_MS,
    keep_tail_ms: int = _TRIM_KEEP_TAIL_MS,
) -> np.ndarray:
    """Trim leading/trailing silence via a moving-RMS envelope."""
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
    runtime: QwenRuntime,
    text: str,
    voice: VoiceConfig,
    *,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Synthesize one chunk; returns (audio_float32_1d, sample_rate)."""
    cleaned = text.strip()
    if not cleaned:
        return np.zeros(0, dtype=np.float32), runtime.sample_rate
    if not voice.ref_text:
        raise ValueError(
            f"Voice '{voice.name}' has no ref_text; Qwen3-TTS in-context "
            "cloning requires the exact transcript of the reference audio."
        )

    ref_audio = runtime.ref_for(voice)
    if seed is not None:
        mx.random.seed(seed)
    results = list(
        runtime.model.generate(
            text=cleaned,
            ref_audio=ref_audio,
            ref_text=voice.ref_text,
            lang_code="auto",
            temperature=runtime.temperature,
            verbose=False,
        )
    )
    if not results:
        return np.zeros(0, dtype=np.float32), runtime.sample_rate
    audio = mx.concatenate([r.audio for r in results], axis=0)
    mx.eval(audio)
    trimmed = trim_silence(np.asarray(audio, dtype=np.float32), runtime.sample_rate)
    return trimmed, runtime.sample_rate
