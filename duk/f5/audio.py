"""Mel spectrogram, ported from f5-tts-mlx (lucasnewman) audio.py.

Torch-compatible HTK mel filterbank + STFT. Kept mechanically identical to
the upstream implementation so reference mels match bit-for-bit; the engine
caches the result per voice instead of recomputing per chunk.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

SAMPLE_RATE = 24_000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 100


@lru_cache(maxsize=None)
def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0,
    f_max: float | None = None,
) -> mx.array:
    """HTK-scale mel filterbank, shape (n_mels, n_fft // 2 + 1)."""

    def hz_to_mel(freq: float) -> float:
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def mel_to_hz(mels: mx.array) -> mx.array:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_max = f_max or sample_rate / 2

    n_freqs = n_fft // 2 + 1
    all_freqs = mx.linspace(0, sample_rate // 2, n_freqs)

    m_pts = mx.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    f_pts = mel_to_hz(m_pts)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = mx.maximum(
        mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes)
    )
    return filterbank.moveaxis(0, 1)


@lru_cache(maxsize=None)
def hanning(size: int) -> mx.array:
    return mx.array(np.hanning(size + 1)[:-1])


def stft(x: mx.array, window: mx.array, nperseg: int, noverlap: int) -> mx.array:
    padding = nperseg // 2
    x = mx.pad(x, [(padding, padding)])

    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nperseg]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(audio: mx.array) -> mx.array:
    """(t,) or (1, t) float audio -> (1, frames, N_MELS) log-mel."""
    if audio.ndim == 2:
        audio = audio.squeeze(0)

    freqs = stft(audio, hanning(N_FFT), nperseg=N_FFT, noverlap=HOP_LENGTH)
    magnitudes = mx.abs(freqs[:-1, :])

    filters = mel_filters(SAMPLE_RATE, N_FFT, N_MELS)
    mel_spec = mx.matmul(magnitudes, filters.T)
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return mx.expand_dims(log_spec, axis=0)
