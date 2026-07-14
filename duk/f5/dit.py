"""DiT backbone, ported from f5-tts-mlx (lucasnewman) dit.py.

Mechanically identical parameter tree (weights load unchanged from the
converted checkpoint) with a restructured inference path:

  - text embedding is computed once per chunk via ``TextEmbedding.compute``
    (upstream re-ran embedding + 4 ConvNeXt blocks inside all ~64 CFG x NFE
    transformer calls per chunk),
  - rotary cos/sin tables are precomputed once per duration and passed in
    (upstream re-evaluated ``freqs.cos()``/``.sin()`` inside every attention
    call — ~2,800 times per chunk),
  - CFG conditional/unconditional passes run as one batch=2 forward via
    ``DiT.fused_forward`` (upstream ran two sequential batch=1 forwards; its
    batched mask path called torch's ``.expand`` and could never run),
  - no einops, no drop flags (conditioning variants are prepared by the
    caller), no training paths.

Math is unchanged w.r.t. upstream: every op is row-independent along the
batch axis, and the precomputed embeddings/tables are pure functions of
inputs that are constant across ODE steps.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


# rotary embedding tables


class RotaryEmbedding(nn.Module):
    """Holds inv_freq (present in the checkpoint); tables built via rope_tables."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))


def rope_tables(inv_freq: mx.array, seq_len: int) -> tuple[mx.array, mx.array]:
    """(cos, sin), each (seq_len, dim_head), matching upstream's interleaved
    pair layout: freqs = stack((f, f), -1).reshape(n, d)."""
    t = mx.arange(seq_len).astype(inv_freq.dtype)
    freqs = mx.einsum("i,j->ij", t, inv_freq)
    freqs = mx.stack([freqs, freqs], axis=-1).reshape(seq_len, -1)
    return mx.cos(freqs), mx.sin(freqs)


def rotate_half(x: mx.array) -> mx.array:
    shape = x.shape
    x = x.reshape(*shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return mx.stack([-x2, x1], axis=-1).reshape(shape)


def apply_rope(t: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """t: (b, h, n, dh); cos/sin: (n, dh) broadcast over batch and heads."""
    return t * cos + rotate_half(t) * sin


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000) -> mx.array:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim) * -emb)
        emb = scale * mx.expand_dims(x, axis=1) * mx.expand_dims(emb, axis=0)
        return mx.concatenate([emb.sin(), emb.cos()], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def __call__(self, timestep: mx.array) -> mx.array:
        return self.time_mlp(self.time_embed(timestep))


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv1d(x)


# ConvNeXt V2 (text refinement blocks)


class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.zeros((1, 1, dim))
        self.beta = mx.zeros((1, 1, dim))

    def __call__(self, x: mx.array) -> mx.array:
        Gx = mx.linalg.norm(x, ord=2, axis=1, keepdims=True)
        Nx = Gx / (Gx.mean(axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# text embedding (computed once per chunk, both CFG variants)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> mx.array:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: dim // 2].astype(mx.float32) / dim))
    t = mx.arange(end)
    freqs = mx.outer(t, freqs).astype(mx.float32)
    return mx.concatenate([freqs.cos(), freqs.sin()], axis=-1)


class TextEmbedding(nn.Module):
    def __init__(
        self,
        text_num_embeds: int,
        text_dim: int,
        mask_padding: bool = True,
        conv_layers: int = 0,
        conv_mult: int = 2,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        self.precompute_max_pos = 4096
        self._freqs_cis = precompute_freqs_cis(text_dim, self.precompute_max_pos)
        self.text_blocks = nn.Sequential(
            *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
        )

    def _embed_variant(self, text: mx.array, text_mask: mx.array, seq_len: int) -> mx.array:
        embed = self.text_embed(text)

        pos = mx.minimum(mx.arange(seq_len), self.precompute_max_pos - 1)
        embed = embed + self._freqs_cis[pos][None, :, :]

        if self.mask_padding:
            embed = mx.where(text_mask, mx.zeros_like(embed), embed)
            for block in self.text_blocks.layers:
                embed = block(embed)
                embed = mx.where(text_mask, mx.zeros_like(embed), embed)
        else:
            embed = self.text_blocks(embed)
        return embed

    def compute(self, text: mx.array, seq_len: int) -> mx.array:
        """(1, nt) padded-with--1 token ids -> (2, seq_len, text_dim).

        Row 0 is the conditional embedding, row 1 the unconditional
        (drop_text) variant. The padding mask comes from the real text in
        both cases, mirroring upstream where the mask is computed before the
        drop_text zeroing.
        """
        text = text + 1  # 0 is the filler token; input pads with -1
        text = text[:, :seq_len]
        text = mx.pad(text, [(0, 0), (0, seq_len - text.shape[1])], constant_values=0)
        text_mask = (text == 0)[..., None]

        cond = self._embed_variant(text, text_mask, seq_len)
        uncond = self._embed_variant(mx.zeros_like(text), text_mask, seq_len)
        return mx.concatenate([cond, uncond], axis=0)


# input embedding (x + cond + text, conv position)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def __call__(self, x: mx.array, cond: mx.array, text_embed: mx.array) -> mx.array:
        x = self.proj(mx.concatenate((x, cond, text_embed), axis=-1))
        return self.conv_pos_embed(x) + x


# adaptive layer norms


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            emb, 6, axis=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        emb = self.linear(self.silu(emb))
        scale, shift = mx.split(emb, 2, axis=1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


# attention


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self._scale_factor = 1 / math.sqrt(dim_head)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def __call__(self, x: mx.array, rope_cos: mx.array, rope_sin: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape

        query = self.to_q(x).reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)
        key = self.to_k(x).reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)
        value = self.to_v(x).reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)

        query = apply_rope(query, rope_cos, rope_sin)
        key = apply_rope(key, rope_cos, rope_sin)

        x = mx.fast.scaled_dot_product_attention(
            q=query, k=key, v=value, scale=self._scale_factor, mask=None
        )
        x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.to_out(x)


# feed forward


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        activation = nn.GELU(approx=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim))

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff(x)


# DiT block and backbone


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def __call__(
        self, x: mx.array, t: mx.array, rope_cos: mx.array, rope_sin: mx.array
    ) -> mx.array:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(norm, rope_cos, rope_sin)
        x = x + gate_msa[:, None, :] * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.ff(norm)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        ff_mult: int = 4,
        mel_dim: int = 100,
        text_num_embeds: int = 256,
        text_dim: int | None = None,
        text_mask_padding: bool = True,
        conv_layers: int = 0,
    ):
        super().__init__()
        if text_dim is None:
            text_dim = mel_dim

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            conv_layers=conv_layers,
        )
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = [
            DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout)
            for _ in range(depth)
        ]
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def fused_forward(
        self,
        x: mx.array,  # (2, n, mel_dim) — noised input, duplicated rows
        cond: mx.array,  # (2, n, mel_dim) — row 0 masked ref mel, row 1 zeros
        text_embed: mx.array,  # (2, n, text_dim) — cond and uncond variants
        time: mx.array,  # scalar
        rope_cos: mx.array,  # (n, dim_head)
        rope_sin: mx.array,  # (n, dim_head)
    ) -> mx.array:
        # the time path computes in fp32; cast its output to the compute
        # dtype (set by the caller via x/cond/text_embed) so the AdaLN gates
        # don't promote every block back to fp32
        t = self.time_embed(mx.broadcast_to(time, (x.shape[0],))).astype(x.dtype)
        x = self.input_embed(x, cond, text_embed)

        for block in self.transformer_blocks:
            x = block(x, t, rope_cos, rope_sin)

        x = self.norm_out(x, t)
        return self.proj_out(x)
