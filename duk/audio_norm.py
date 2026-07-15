from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

CLONE_TARGET_ACTIVE_DBFS = -18.71
CLONE_PEAK_CEILING_DBFS = -1.5


def _otsu_threshold(values: np.ndarray, *, bins: int = 256) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return -40.0
    low = float(np.min(data))
    high = float(np.max(data))
    if high - low < 1e-9:
        return low

    hist, edges = np.histogram(data, bins=bins, range=(low, high))
    probs = hist.astype(np.float64)
    probs /= np.sum(probs)
    centers = (edges[:-1] + edges[1:]) / 2.0

    omega = np.cumsum(probs)
    mu = np.cumsum(probs * centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0.0] = np.nan
    sigma_between = (mu_total * omega - mu) ** 2 / denom
    best = int(np.nanargmax(sigma_between))
    return float(centers[best])


def measure_speech_active_level_dbfs(path: Path) -> tuple[float, float]:
    samples, sample_rate = sf.read(path, always_2d=True)
    mono = samples.mean(axis=1).astype(np.float64)
    if mono.size == 0:
        raise ValueError("Audio clip is empty.")

    frame = max(1, int(sample_rate * 0.02))
    hop = max(1, int(sample_rate * 0.01))
    if mono.size <= frame:
        starts = np.array([0], dtype=np.int64)
    else:
        starts = np.arange(0, mono.size - frame + 1, hop)
    frames = np.stack([mono[start : start + frame] for start in starts], axis=0)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    frame_db = np.clip(frame_db, -90.0, 0.0)

    threshold = _otsu_threshold(frame_db)
    active_mask = frame_db >= threshold
    if not np.any(active_mask):
        active_mask = frame_db > -40.0
    if not np.any(active_mask):
        active_mask = np.ones_like(frame_db, dtype=bool)

    active_rms = float(np.sqrt(np.mean(np.square(frames[active_mask])) + 1e-12))
    active_db = 20.0 * np.log10(max(active_rms, 1e-12))
    peak = float(np.max(np.abs(mono)))
    peak_db = 20.0 * np.log10(max(peak, 1e-12))
    return active_db, peak_db


def recommended_clone_gain_db(
    *,
    active_db: float,
    peak_db: float,
    target_active_db: float = CLONE_TARGET_ACTIVE_DBFS,
    peak_ceiling_db: float = CLONE_PEAK_CEILING_DBFS,
) -> float:
    desired_gain = target_active_db - active_db
    peak_limited_gain = peak_ceiling_db - peak_db
    return min(desired_gain, peak_limited_gain)


def normalize_clone_wav(
    path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    target_active_db: float = CLONE_TARGET_ACTIVE_DBFS,
    peak_ceiling_db: float = CLONE_PEAK_CEILING_DBFS,
) -> float:
    active_db, peak_db = measure_speech_active_level_dbfs(path)
    gain_db = recommended_clone_gain_db(
        active_db=active_db,
        peak_db=peak_db,
        target_active_db=target_active_db,
        peak_ceiling_db=peak_ceiling_db,
    )
    if abs(gain_db) < 0.05:
        return gain_db

    tmp_path = path.with_name(f"{path.stem}.norm-tmp{path.suffix}")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-af",
        f"volume={gain_db:.6f}dB",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            raise RuntimeError(
                f"ffmpeg failed during clone normalization: {detail.splitlines()[-1]}"
            )
        raise RuntimeError("ffmpeg failed during clone normalization.")

    tmp_path.replace(path)
    return gain_db
