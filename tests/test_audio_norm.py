from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from duk import audio_norm as audio_norm_util


def test_recommended_clone_gain_respects_peak_ceiling() -> None:
    gain = audio_norm_util.recommended_clone_gain_db(
        active_db=-20.0,
        peak_db=-1.0,
        target_active_db=-18.71,
        peak_ceiling_db=-1.5,
    )
    assert gain == -0.5


def test_measure_speech_active_level_dbfs_uses_active_frames(tmp_path: Path) -> None:
    sample_rate = 24_000
    silence = np.zeros(sample_rate, dtype=np.float32)
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    tone = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    audio = np.concatenate([silence, tone]).astype(np.float32)

    path = tmp_path / "sample.wav"
    sf.write(path, audio, sample_rate, subtype="PCM_16")

    active_db, peak_db = audio_norm_util.measure_speech_active_level_dbfs(path)
    assert -25.0 < active_db < -21.0
    assert -20.5 < peak_db < -19.5


def test_normalize_clone_wav_runs_ffmpeg_volume_filter(
    tmp_path: Path, monkeypatch
) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFFIN")
    commands: list[list[str]] = []

    def fake_measure(_: Path) -> tuple[float, float]:
        return -20.0, -4.0

    def fake_run(cmd: list[str], capture_output: bool, text: bool) -> SimpleNamespace:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"RIFFOUT")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        audio_norm_util, "measure_speech_active_level_dbfs", fake_measure
    )
    monkeypatch.setattr(audio_norm_util.subprocess, "run", fake_run)

    gain = audio_norm_util.normalize_clone_wav(clip)

    assert abs(gain - 1.29) < 0.01
    assert len(commands) == 1
    assert "volume=1.290000dB" in commands[0]
    assert clip.read_bytes() == b"RIFFOUT"


def test_normalize_clone_wav_skips_when_gain_is_tiny(tmp_path: Path, monkeypatch) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")

    def fake_measure(_: Path) -> tuple[float, float]:
        return -18.70, -6.0

    def fail_run(*args, **kwargs) -> SimpleNamespace:
        raise AssertionError("ffmpeg should not run when gain is tiny")

    monkeypatch.setattr(
        audio_norm_util, "measure_speech_active_level_dbfs", fake_measure
    )
    monkeypatch.setattr(audio_norm_util.subprocess, "run", fail_run)

    gain = audio_norm_util.normalize_clone_wav(clip)
    assert abs(gain) < 0.05
