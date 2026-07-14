"""duk's F5-TTS inference engine for MLX.

A slim reimplementation of f5-tts-mlx's F5TTS.sample() with the same math
(user-approved rk4@8 fp32 output) but restructured for throughput:

  - per-voice reference mel cache (Voice),
  - text embedding computed once per chunk (both CFG variants),
  - rope cos/sin computed once per chunk,
  - CFG as a single batch=2 forward per NFE,
  - one persistent mx.compile'd step function (re-traces only on new
    sequence lengths),
  - async_eval per ODE step to overlap graph build with GPU work.

No training code, no duration predictor (duk sizes the window from its own
Chinese chars/sec heuristic), no quantization (measured useless on M4 Pro).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from vocos_mlx import Vocos

from duk.f5.audio import HOP_LENGTH, SAMPLE_RATE, log_mel_spectrogram
from duk.f5.dit import DiT, rope_tables
from duk.f5.text import convert_char_to_pinyin, tokens_to_ids

FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH
MAX_DURATION_FRAMES = 4096
TARGET_RMS = 0.1

DEFAULT_REPO = "lucasnewman/f5-tts-mlx"
VOCODER_REPO = "lucasnewman/vocos-mel-24khz"
MODEL_FILENAME = "model_v1.safetensors"

N_MELS = 100


@dataclass(frozen=True)
class Voice:
    """A prepared reference voice: cached mel + transcript."""

    ref_mel: mx.array  # (1, ref_frames, N_MELS)
    ref_frames: int
    ref_samples: int
    ref_text: str


def _convert_checkpoint_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Rename torch-checkpoint keys to the MLX module tree.

    Ported from f5-tts-mlx F5TTS.from_pretrained; our DiT reproduces the
    upstream parameter tree exactly, so the mapping is identical except for
    the removed duration predictor and mel-spec entries.
    """
    new_weights = {}
    for k, v in weights.items():
        k = k.replace("ema_model.", "")

        if len(k) < 1 or "mel_spec." in k or k in ("initted", "step"):
            continue
        # upstream nests the DiT under F5TTS.transformer; our engine loads
        # the DiT directly
        if not k.startswith("transformer."):
            continue
        k = k.removeprefix("transformer.")

        if ".to_out" in k:
            k = k.replace(".to_out", ".to_out.layers")
        elif ".text_blocks" in k:
            k = k.replace(".text_blocks", ".text_blocks.layers")
        elif ".ff.ff.0.0" in k:
            k = k.replace(".ff.ff.0.0", ".ff.ff.layers.0.layers.0")
        elif ".ff.ff.2" in k:
            k = k.replace(".ff.ff.2", ".ff.ff.layers.2")
        elif ".time_mlp" in k:
            k = k.replace(".time_mlp", ".time_mlp.layers")
        elif ".conv1d" in k:
            k = k.replace(".conv1d", ".conv1d.layers")

        if ".dwconv.weight" in k:
            v = v.swapaxes(1, 2)
        elif ".conv1d.layers.0.weight" in k:
            v = v.swapaxes(1, 2)
        elif ".conv1d.layers.2.weight" in k:
            v = v.swapaxes(1, 2)

        new_weights[k] = v
    return new_weights


class F5Engine:
    def __init__(
        self,
        transformer: DiT,
        vocab: dict[str, int],
        vocoder: Vocos,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        self.transformer = transformer
        self.vocab = vocab
        self.vocoder = vocoder
        self.compute_dtype = compute_dtype
        self._compiled_step = mx.compile(self._cfg_step)

    # loading

    @classmethod
    def from_pretrained(
        cls, hf_repo: str = DEFAULT_REPO, precision: str = "fp32"
    ) -> "F5Engine":
        path = Path(
            snapshot_download(
                repo_id=hf_repo, allow_patterns=[MODEL_FILENAME, "vocab.txt"]
            )
        )

        vocab = {
            v: i for i, v in enumerate((path / "vocab.txt").read_text().split("\n"))
        }
        if not vocab:
            raise ValueError(f"Could not load vocab from {path / 'vocab.txt'}")

        transformer = DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            text_num_embeds=len(vocab) - 1,
            text_mask_padding=True,
        )
        weights = mx.load(str(path / MODEL_FILENAME), format="safetensors")
        weights = {
            k: v
            for k, v in _convert_checkpoint_keys(weights).items()
            if not k.startswith("duration")
        }
        transformer.load_weights(list(weights.items()))
        transformer.eval()

        compute_dtype = {"fp32": mx.float32, "fp16": mx.float16}[precision]
        if compute_dtype != mx.float32:
            # set_dtype only touches parameters (underscore attrs like
            # text_embed._freqs_cis stay fp32); the fp32 time path and
            # per-chunk fp32 precomputes are cast at the boundaries.
            # inv_freq must stay fp32 — rope phase errors compound with
            # position — so the trig tables are built fp32 and only their
            # values are cast in synthesize().
            inv_freq = transformer.rotary_embed.inv_freq
            transformer.set_dtype(compute_dtype)
            transformer.rotary_embed.inv_freq = inv_freq

        vocoder = Vocos.from_pretrained(VOCODER_REPO)

        engine = cls(transformer, vocab, vocoder, compute_dtype=compute_dtype)
        mx.eval(transformer.parameters())
        return engine

    # voice preparation (cached per voice, not per chunk)

    def prepare_voice(self, audio: np.ndarray | mx.array, ref_text: str) -> Voice:
        """audio: 1-d float wave at 24 kHz, RMS-normalized to >= TARGET_RMS."""
        audio = mx.array(audio) if not isinstance(audio, mx.array) else audio
        rms = mx.sqrt(mx.mean(mx.square(audio)))
        if rms < TARGET_RMS:
            audio = audio * TARGET_RMS / rms

        ref_mel = log_mel_spectrogram(audio)
        mx.eval(ref_mel)
        return Voice(
            ref_mel=ref_mel,
            ref_frames=ref_mel.shape[1],
            ref_samples=audio.shape[0],
            ref_text=ref_text.strip(),
        )

    # sampling

    def _cfg_step(
        self,
        time: mx.array,  # scalar
        x: mx.array,  # (1, n, N_MELS)
        cond2: mx.array,  # (2, n, N_MELS)
        text_embed2: mx.array,  # (2, n, text_dim)
        rope_cos: mx.array,
        rope_sin: mx.array,
        cfg_strength: mx.array,  # scalar
    ) -> mx.array:
        x2 = mx.concatenate([x, x], axis=0).astype(self.compute_dtype)
        pred2 = self.transformer.fused_forward(
            x2, cond2, text_embed2, time, rope_cos, rope_sin
        ).astype(mx.float32)
        pred, null_pred = pred2[0:1], pred2[1:2]
        return pred + (pred - null_pred) * cfg_strength

    def synthesize(
        self,
        voice: Voice,
        text: str,
        gen_seconds: float,
        *,
        steps: int = 8,
        method: str = "rk4",
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """One chunk; returns float32 1-d wave (reference trimmed)."""
        gen_frames = int(gen_seconds * FRAMES_PER_SEC)
        duration = min(voice.ref_frames + max(gen_frames, 1), MAX_DURATION_FRAMES)

        # tokens for ref + generation text, pinyin-converted
        tokens = convert_char_to_pinyin([voice.ref_text + " " + text])[0]
        text_ids = tokens_to_ids(tokens, self.vocab)

        # per-chunk constants
        cond = mx.pad(
            voice.ref_mel, [(0, 0), (0, duration - voice.ref_frames), (0, 0)]
        )
        cond2 = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)
        text_embed2 = self.transformer.text_embed.compute(text_ids, duration)
        rope_cos, rope_sin = rope_tables(
            self.transformer.rotary_embed.inv_freq, duration
        )
        if self.compute_dtype != mx.float32:
            cond2 = cond2.astype(self.compute_dtype)
            text_embed2 = text_embed2.astype(self.compute_dtype)
            rope_cos = rope_cos.astype(self.compute_dtype)
            rope_sin = rope_sin.astype(self.compute_dtype)
        cfg = mx.array(cfg_strength)
        mx.eval(cond2, text_embed2, rope_cos, rope_sin)

        # noise; replicates upstream construction exactly (normal((mel, n))
        # then transpose) so seeds are comparable across engines
        if seed is not None:
            mx.random.seed(seed)
        y = mx.random.normal((N_MELS, duration)).T[None, :, :]

        # ODE over sway-warped time grid
        t_grid = mx.linspace(0, 1, steps)
        t_grid = t_grid + sway_sampling_coef * (
            mx.cos(mx.pi / 2 * t_grid) - 1 + t_grid
        )

        def f(t: mx.array, x: mx.array) -> mx.array:
            return self._compiled_step(
                t, x, cond2, text_embed2, rope_cos, rope_sin, cfg
            )

        for i in range(steps - 1):
            t_cur = t_grid[i]
            dt = t_grid[i + 1] - t_cur
            if method == "euler":
                y = y + dt * f(t_cur, y)
            elif method == "midpoint":
                k1 = f(t_cur, y)
                y = y + dt * f(t_cur + 0.5 * dt, y + 0.5 * dt * k1)
            elif method == "rk4":
                k1 = f(t_cur, y)
                k2 = f(t_cur + 0.5 * dt, y + 0.5 * dt * k1)
                k3 = f(t_cur + 0.5 * dt, y + 0.5 * dt * k2)
                k4 = f(t_cur + dt, y + dt * k3)
                y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown method: {method}")
            mx.async_eval(y)

        # restore exact reference mel over the ref region, vocode, trim
        out = mx.concatenate([voice.ref_mel, y[:, voice.ref_frames :, :]], axis=1)
        wave = self.vocoder.decode(out)
        wave = wave.squeeze(0) if wave.ndim == 2 else wave
        wave = wave[voice.ref_samples :]
        mx.eval(wave)
        return np.asarray(wave, dtype=np.float32)
