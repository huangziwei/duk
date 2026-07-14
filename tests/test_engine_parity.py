"""Parity: duk's F5 engine must reproduce upstream f5-tts-mlx output.

Same checkpoint, same voice/text/seed/duration/solver -> near-identical
waveforms. The engine's restructurings (precomputed text embed and rope,
fused batch=2 CFG, persistent compile) are mathematically no-ops; only
kernel tiling/fusion differences may introduce tiny float drift.

Run: uv run pytest tests/test_engine_parity.py -v -s
(needs the HF snapshot cached; the feasibility runs already fetched it)
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = REPO_ROOT / "voices"

TEXT = "我冒了严寒，回到相隔二千余里，别了二十余年的故乡去。"
GEN_SECONDS = 6.0
SEED = 7
STEPS = 3
METHOD = "euler"
CFG = 2.0
SWAY = -1.0


def load_ref_audio() -> tuple[np.ndarray, str]:
    config = json.loads((VOICES_DIR / "kate.json").read_text(encoding="utf-8"))
    audio, sr = sf.read(str(VOICES_DIR / config["ref_audio"]))
    assert sr == 24_000
    return audio, config["ref_text"]


@pytest.fixture(scope="module")
def waves() -> tuple[np.ndarray, np.ndarray]:
    from f5_tts_mlx.cfm import F5TTS
    from f5_tts_mlx.utils import convert_char_to_pinyin as upstream_pinyin

    from duk.f5 import F5Engine

    audio, ref_text = load_ref_audio()

    # duk engine
    engine = F5Engine.from_pretrained()
    voice = engine.prepare_voice(audio, ref_text)
    wave_duk = engine.synthesize(
        voice,
        TEXT,
        GEN_SECONDS,
        steps=STEPS,
        method=METHOD,
        cfg_strength=CFG,
        sway_sampling_coef=SWAY,
        seed=SEED,
    )

    # upstream reference
    upstream = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")
    norm = mx.array(audio)
    rms = mx.sqrt(mx.mean(mx.square(norm)))
    if rms < 0.1:
        norm = norm * 0.1 / rms
    duration_frames = voice.ref_frames + int(GEN_SECONDS * (24_000 / 256))

    pinyin = upstream_pinyin([ref_text + " " + TEXT])
    wave_up, _ = upstream.sample(
        mx.expand_dims(norm, axis=0),
        text=pinyin,
        duration=duration_frames,
        steps=STEPS,
        method=METHOD,
        cfg_strength=CFG,
        sway_sampling_coef=SWAY,
        seed=SEED,
    )
    wave_up = np.asarray(wave_up[norm.shape[0] :], dtype=np.float32)
    return wave_duk, wave_up


def test_same_length(waves):
    wave_duk, wave_up = waves
    assert wave_duk.shape == wave_up.shape


def test_waveform_parity(waves):
    wave_duk, wave_up = waves
    corr = float(
        np.corrcoef(wave_duk.astype(np.float64), wave_up.astype(np.float64))[0, 1]
    )
    max_diff = float(np.max(np.abs(wave_duk - wave_up)))
    rms = float(np.sqrt(np.mean(wave_up**2)))
    print(f"\ncorr={corr:.6f} max_diff={max_diff:.5f} ref_rms={rms:.4f}")
    assert corr > 0.999, f"waveform correlation too low: {corr}"
    assert max_diff < 0.05 * max(rms, 0.01) * 10, f"max sample diff too high: {max_diff}"
