"""Image classifier for GQA gate, FFN, and metadata comparisons with common QK norm."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from core.layers import (
    AdaRMSNorm,
    AdaRMSScaleProjection,
    GatedFFN,
    RMSNorm2d,
    RotaryEmbedding2D,
    ScaleOnlyAdaRMSNorm,
)


VARIANTS = (
    # Active 2x2 comparison: AdaRMS scale on both branches and the paired
    # SiLU attention gate + SwiGLU FFN are independent experiment factors.
    "naive_gqa",
    "ada_naive_gqa",
    "gated_gqa_silu_gated_ffn",
    "ada_gated_gqa_silu_gated_ffn",
    "ada_1plus_silu_gated_gqa_silu_gated_ffn",
    "ada_1plus_softplus_halfnorm_gated_gqa_silu_gated_ffn",
    "ada_2sigmoid_gated_gqa_silu_gated_ffn",
    "ada_silu1_norm_gated_gqa_silu_gated_ffn",
    "ada_silu1_norm_shift_gated_gqa_silu_gated_ffn",
    "ada_softplus1_norm_gated_gqa_silu_gated_ffn",
    "ada_silu1_norm_q_shift_gated_gqa_silu_gated_ffn",
    "ada_shift_gated_gqa_silu_gated_ffn",
    "ada_1plus_silu_shift_gated_gqa_silu_gated_ffn",
    "ada_2sigmoid_shift_gated_gqa_silu_gated_ffn",
    "ada_softplus1_norm_shift_gated_gqa_silu_gated_ffn",
)

# Kept available for direct model diagnostics, but excluded from the train CLI:
# with zero Ada initialization it permanently disables the bias-free SwiGLU FFN.
_DIAGNOSTIC_VARIANTS = ("ada_silu_scale_gated_gqa_silu_gated_ffn",)

# Retained for reproducing older experiments/checkpoints. These are omitted
# from VARIANTS, so the training CLI exposes only the current controlled matrix.
_LEGACY_VARIANTS = (
    "naive_gqa_gated_ffn",
    "gated_gqa_sigmoid",
    "gated_gqa_silu",
    "gated_gqa_sigmoid_gated_ffn",
    "concat_meta_gated_gqa_silu_gated_ffn",
    "branch_qkv_meta_gated_gqa_silu_gated_ffn",
    "branch_ffn_meta_gated_gqa_silu_gated_ffn",
    "branch_qkv_ffn_meta_gated_gqa_silu_gated_ffn",
    "branch_qkv_meta_ffn_ada_gated_gqa_silu_gated_ffn",
    "branch_qkv_meta_ada_attn_ffn_gated_gqa_silu_gated_ffn",
)

_VARIANT_SETTINGS = {
    "naive_gqa": ("none", "plain"),
    "ada_naive_gqa": ("none", "plain"),
    "naive_gqa_gated_ffn": ("none", "gated"),
    "gated_gqa_sigmoid": ("sigmoid", "plain"),
    "gated_gqa_silu": ("silu", "plain"),
    "gated_gqa_sigmoid_gated_ffn": ("sigmoid", "gated"),
    "gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_1plus_silu_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_1plus_softplus_halfnorm_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_silu_scale_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_2sigmoid_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_silu1_norm_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_silu1_norm_shift_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_softplus1_norm_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_silu1_norm_q_shift_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_shift_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_1plus_silu_shift_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_2sigmoid_shift_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "ada_softplus1_norm_shift_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "concat_meta_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "branch_qkv_meta_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "branch_ffn_meta_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "branch_qkv_ffn_meta_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "branch_qkv_meta_ffn_ada_gated_gqa_silu_gated_ffn": ("silu", "gated"),
    "branch_qkv_meta_ada_attn_ffn_gated_gqa_silu_gated_ffn": ("silu", "gated"),
}

LEGACY_LINEAR_ADA_VARIANT = "ada_gated_gqa_silu_gated_ffn"
DEFAULT_ADA_SCALE_VARIANT = "ada_softplus1_norm_gated_gqa_silu_gated_ffn"
ADA_VARIANT = DEFAULT_ADA_SCALE_VARIANT
ADA_SCALE_MODES = {
    LEGACY_LINEAR_ADA_VARIANT: "linear",
    "ada_1plus_silu_gated_gqa_silu_gated_ffn": "one_sided_shifted_silu",
    "ada_1plus_softplus_halfnorm_gated_gqa_silu_gated_ffn": "one_plus_softplus_halfnorm",
    "ada_silu_scale_gated_gqa_silu_gated_ffn": "one_sided_silu",
    "ada_2sigmoid_gated_gqa_silu_gated_ffn": "two_sided_sigmoid",
    "ada_silu1_norm_gated_gqa_silu_gated_ffn": "normalized_shifted_silu",
    "ada_silu1_norm_shift_gated_gqa_silu_gated_ffn": "normalized_shifted_silu",
    "ada_softplus1_norm_gated_gqa_silu_gated_ffn": "normalized_shifted_softplus",
    "ada_silu1_norm_q_shift_gated_gqa_silu_gated_ffn": "normalized_shifted_silu",
    "ada_shift_gated_gqa_silu_gated_ffn": "linear",
    "ada_1plus_silu_shift_gated_gqa_silu_gated_ffn": "one_sided_shifted_silu",
    "ada_2sigmoid_shift_gated_gqa_silu_gated_ffn": "two_sided_sigmoid",
    "ada_softplus1_norm_shift_gated_gqa_silu_gated_ffn": "normalized_shifted_softplus",
}
Q_ONLY_SHIFT_VARIANTS = {
    "ada_silu1_norm_q_shift_gated_gqa_silu_gated_ffn",
}
ADA_SHIFT_VARIANTS = {
    "ada_shift_gated_gqa_silu_gated_ffn",
    "ada_1plus_silu_shift_gated_gqa_silu_gated_ffn",
    "ada_2sigmoid_shift_gated_gqa_silu_gated_ffn",
    "ada_silu1_norm_shift_gated_gqa_silu_gated_ffn",
    "ada_softplus1_norm_shift_gated_gqa_silu_gated_ffn",
}
ADA_NAIVE_VARIANT = "ada_naive_gqa"
CONCAT_META_VARIANT = "concat_meta_gated_gqa_silu_gated_ffn"
BRANCH_QKV_META_VARIANT = "branch_qkv_meta_gated_gqa_silu_gated_ffn"
BRANCH_FFN_META_VARIANT = "branch_ffn_meta_gated_gqa_silu_gated_ffn"
BRANCH_BOTH_META_VARIANT = "branch_qkv_ffn_meta_gated_gqa_silu_gated_ffn"
BRANCH_QKV_ADA_FFN_VARIANT = "branch_qkv_meta_ffn_ada_gated_gqa_silu_gated_ffn"
BRANCH_QKV_ADA_BOTH_VARIANT = "branch_qkv_meta_ada_attn_ffn_gated_gqa_silu_gated_ffn"
BRANCH_QKV_VARIANTS = {
    BRANCH_QKV_META_VARIANT, BRANCH_BOTH_META_VARIANT, BRANCH_QKV_ADA_FFN_VARIANT,
    BRANCH_QKV_ADA_BOTH_VARIANT,
}
BRANCH_FFN_VARIANTS = {BRANCH_FFN_META_VARIANT, BRANCH_BOTH_META_VARIANT}
BRANCH_META_VARIANTS = BRANCH_QKV_VARIANTS | BRANCH_FFN_VARIANTS
ADAPTIVE_VARIANTS = {ADA_NAIVE_VARIANT, *ADA_SCALE_MODES}
METADATA_VARIANTS = ADAPTIVE_VARIANTS | {CONCAT_META_VARIANT} | BRANCH_META_VARIANTS
ADAPTIVE_ATTENTION_VARIANTS = ADAPTIVE_VARIANTS | {BRANCH_QKV_ADA_BOTH_VARIANT}
ADAPTIVE_FFN_VARIANTS = {
    *ADAPTIVE_VARIANTS, BRANCH_QKV_ADA_FFN_VARIANT, BRANCH_QKV_ADA_BOTH_VARIANT,
}
META_FEATURES = 2
def _zero_initialized_shift_projection(condition_dim: int, width: int) -> AdaRMSScaleProjection:
    """Create zero Ada shift weights without changing later RNG draws.

    Scale-only and scale-plus-shift runs should start with identical shared
    and non-Ada parameters. Linear initialization consumes random values even
    though Ada projection weights are then zeroed, so isolate those draws.
    """
    with torch.random.fork_rng(devices=[]):
        return AdaRMSScaleProjection(condition_dim, width)


def _ada_scale_offset(raw_scale: torch.Tensor, mode: str) -> torch.Tensor:
    """Map learned s(m) to the offset consumed by ``1 + offset``.

    The modes implement the exact multipliers 1+s, 1+SiLU(s),
    1+Softplus(s-2.5),
    SiLU(s), 2*sigmoid(s),
    SiLU(1+s)/SiLU(1), and softplus(1+s)/softplus(1).
    """
    if mode == "linear":
        return raw_scale
    if mode == "one_sided_shifted_silu":
        return F.silu(raw_scale)
    if mode == "one_plus_softplus_halfnorm":
        # alpha=2.5 gives an initial effective scale of about 1.079.
        return F.softplus(raw_scale - 2.5)
    if mode == "one_sided_silu":
        return F.silu(raw_scale) - 1.0
    if mode == "two_sided_sigmoid":
        return 2.0 * torch.sigmoid(raw_scale) - 1.0
    if mode == "normalized_shifted_silu":
        silu_at_one = 1.0 / (1.0 + math.exp(-1.0))
        return F.silu(1.0 + raw_scale) / silu_at_one - 1.0
    if mode == "normalized_shifted_softplus":
        softplus_at_one = math.log1p(math.exp(1.0))
        offset = F.softplus(1.0 + raw_scale) / softplus_at_one - 1.0
        # Keep the effective multiplier ``1 + offset`` strictly positive even
        # when finite precision rounds a very negative input to exactly -1.
        minimum_offset = torch.nextafter(
            torch.full_like(raw_scale, -1.0), torch.zeros_like(raw_scale),
        )
        return torch.maximum(offset, minimum_offset)
    raise ValueError(f"unknown Ada scale mode: {mode}")


def _conv_with_broadcast_condition(
    features: torch.Tensor,
    condition: torch.Tensor,
    projection: nn.Conv2d,
) -> torch.Tensor:
    """Apply a 1x1 conv to [features, spatially broadcast condition] implicitly.

    Splitting the projection weights gives
    ``W_x x_i + W_m m + b``. The metadata term is computed once per sample,
    then broadcast over H and W without allocating a concatenated feature map.
    Under autocast, separate terms can round differently from one fused conv.
    """
    if (
        projection.kernel_size != (1, 1)
        or projection.stride != (1, 1)
        or projection.padding != (0, 0)
        or projection.dilation != (1, 1)
        or projection.groups != 1
    ):
        raise ValueError("broadcast-condition projection requires a plain 1x1 convolution")
    batch, feature_channels, _, _ = features.shape
    if condition.ndim != 2 or condition.shape[0] != batch:
        raise ValueError("condition must have shape (B, D) matching features")
    if projection.in_channels != feature_channels + condition.shape[1]:
        raise ValueError("projection input channels must equal feature plus condition channels")

    feature_weight = projection.weight[:, :feature_channels]
    condition_weight = projection.weight[:, feature_channels:, 0, 0]
    feature_term = F.conv2d(features, feature_weight, projection.bias)
    condition_term = F.linear(condition.to(dtype=features.dtype), condition_weight)
    return feature_term + condition_term.to(dtype=feature_term.dtype)[:, :, None, None]


class GQATransformerBlock2D(nn.Module):
    """Pre-norm GQA block with common QK RMSNorm and optional gate/Ada scales."""

    def __init__(
        self,
        width: int,
        heads: int,
        kv_heads: int,
        *,
        grid_size: int,
        variant: str,
        dropout: float = 0.1,
        ff_mult: float = 4.0,
        condition_dim: int | None = None,
    ) -> None:
        super().__init__()
        if variant not in _VARIANT_SETTINGS or variant not in (
            *VARIANTS, *_LEGACY_VARIANTS, *_DIAGNOSTIC_VARIANTS,
        ):
            raise ValueError(f"unknown block variant: {variant}")
        if width <= 0 or heads <= 0 or kv_heads <= 0:
            raise ValueError("width, heads and kv_heads must be positive")
        if width % heads or heads % kv_heads:
            raise ValueError("width must divide heads and heads must divide into kv_heads")
        head_dim = width // heads
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        if grid_size <= 0 or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("grid_size must be positive and dropout in [0, 1)")
        if not math.isfinite(ff_mult) or ff_mult <= 0:
            raise ValueError("ff_mult must be finite and positive")

        gate_mode, ffn_mode = _VARIANT_SETTINGS[variant]
        self.width = width
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.variant = variant
        self.ada_scale_mode = ADA_SCALE_MODES.get(variant, "linear")
        self.uses_adaptive_shift = variant in ADA_SHIFT_VARIANTS
        self.gate_mode = gate_mode
        self.uses_adaptive_attention_norm = variant in ADAPTIVE_ATTENTION_VARIANTS
        self.uses_adaptive_ffn_norm = variant in ADAPTIVE_FFN_VARIANTS
        self.uses_qkv_metadata = variant in BRANCH_QKV_VARIANTS
        self.uses_ffn_metadata = variant in BRANCH_FFN_VARIANTS
        self.requires_condition = (
            self.uses_adaptive_attention_norm or self.uses_adaptive_ffn_norm
            or self.uses_qkv_metadata or self.uses_ffn_metadata
        )
        self.condition_dim = condition_dim if self.requires_condition else None
        if self.requires_condition and (condition_dim is None or condition_dim <= 0):
            raise ValueError("metadata-conditioned blocks require a positive condition_dim")
        if self.uses_adaptive_attention_norm:
            assert condition_dim is not None
            self.norm1 = AdaRMSNorm(width) if self.uses_adaptive_shift else ScaleOnlyAdaRMSNorm(width)
            self.norm1_scale = AdaRMSScaleProjection(condition_dim, width)
            self.norm1_shift = (
                _zero_initialized_shift_projection(condition_dim, width)
                if self.uses_adaptive_shift else None
            )
        else:
            self.norm1 = nn.RMSNorm(width)
            self.norm1_scale = None
            self.norm1_shift = None
        self.q_proj = nn.Linear(width, width, bias=False)
        self.kv_proj = nn.Linear(width, 2 * kv_heads * head_dim, bias=False)
        self.q_meta_proj = (
            nn.Linear(condition_dim, width, bias=False)
            if self.uses_qkv_metadata else None
        )
        self.kv_meta_proj = (
            nn.Linear(condition_dim, 2 * kv_heads * head_dim, bias=False)
            if self.uses_qkv_metadata else None
        )
        for projection in (self.q_meta_proj, self.kv_meta_proj):
            if projection is not None:
                nn.init.zeros_(projection.weight)
        self.q_only_meta_shift = (
            _zero_initialized_shift_projection(condition_dim, width)
            if variant in Q_ONLY_SHIFT_VARIANTS and condition_dim is not None else None
        )
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.rope = RotaryEmbedding2D(head_dim, grid_size, grid_size)
        self.head_gate = nn.Linear(width, heads) if gate_mode != "none" else None
        if self.head_gate is not None:
            nn.init.zeros_(self.head_gate.weight)
            nn.init.zeros_(self.head_gate.bias)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        if self.uses_adaptive_ffn_norm:
            assert condition_dim is not None
            self.norm2 = AdaRMSNorm(width) if self.uses_adaptive_shift else ScaleOnlyAdaRMSNorm(width)
            self.norm2_scale = AdaRMSScaleProjection(condition_dim, width)
            self.norm2_shift = (
                _zero_initialized_shift_projection(condition_dim, width)
                if self.uses_adaptive_shift else None
            )
        else:
            self.norm2 = nn.RMSNorm(width)
            self.norm2_scale = None
            self.norm2_shift = None
        if ffn_mode == "plain":
            hidden = max(1, round(width * ff_mult))
            self.ffn = nn.Sequential(
                nn.Linear(width, hidden, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, width, bias=False),
            )
        else:
            # A 2/3 width multiplier approximately matches a 4x GELU FFN's
            # parameter/FLOP budget: 3 * (8/3)d^2 ~= 8d^2.
            hidden = max(1, round(width * ff_mult * 2.0 / 3.0))
            self.ffn = nn.Sequential(
                GatedFFN(width, hidden, backend="torch"),
                nn.Dropout(dropout),
            )
        self.ffn_meta_proj = None
        if self.uses_ffn_metadata:
            assert condition_dim is not None
            self.ffn_meta_proj = nn.Linear(condition_dim, 2 * hidden, bias=False)
            nn.init.zeros_(self.ffn_meta_proj.weight)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_shape: tuple[int, int],
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(f"tokens must have shape (B, T, {self.width})")
        batch, token_count, _ = tokens.shape
        height, width = grid_shape
        if height <= 0 or width <= 0 or height * width != token_count:
            raise ValueError("grid_shape must match the spatial token count")

        if self.requires_condition:
            if condition is None:
                raise ValueError("metadata-conditioned blocks require a condition")
            if condition.shape != (batch, self.condition_dim):
                raise ValueError(
                    f"condition must have shape (B, {self.condition_dim})"
                )
        if self.uses_adaptive_attention_norm:
            assert condition is not None
            assert self.norm1_scale is not None
            scale = _ada_scale_offset(
                self.norm1_scale(condition), self.ada_scale_mode,
            )
            if self.uses_adaptive_shift:
                assert isinstance(self.norm1, AdaRMSNorm) and self.norm1_shift is not None
                normalized = self.norm1(tokens, scale, self.norm1_shift(condition))
            else:
                assert isinstance(self.norm1, ScaleOnlyAdaRMSNorm)
                normalized = self.norm1(tokens, scale)
        else:
            normalized = self.norm1(tokens)
        query_features = self.q_proj(normalized)
        kv_features = self.kv_proj(normalized)
        if self.uses_qkv_metadata:
            assert condition is not None
            assert self.q_meta_proj is not None and self.kv_meta_proj is not None
            # Algebraically equivalent to projecting cat((normalized, condition))
            # per token, without materializing the broadcast condition tensor.
            query_features = query_features + self.q_meta_proj(condition).unsqueeze(1)
            kv_features = kv_features + self.kv_meta_proj(condition).unsqueeze(1)
        query = query_features.reshape(
            batch, token_count, self.heads, self.head_dim,
        ).transpose(1, 2)
        key_value = kv_features.reshape(
            batch, token_count, 2, self.kv_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        key, value = key_value.unbind(0)
        query, key = self.rope(self.q_norm(query), self.k_norm(key), grid_shape=grid_shape)
        if self.q_only_meta_shift is not None:
            assert condition is not None
            query_shift = self.q_only_meta_shift(condition).reshape(
                batch, self.heads, 1, self.head_dim,
            )
            # One metadata-derived query offset per head is shared by all target
            # positions and is added after QK norm/RoPE so it remains a direct
            # query-side logit modulator.
            query = query + query_shift.to(dtype=query.dtype)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            enable_gqa=self.heads != self.kv_heads,
        )
        if self.head_gate is not None:
            logits = self.head_gate(normalized)
            if self.gate_mode == "sigmoid":
                gate = 2.0 * torch.sigmoid(logits)
            else:
                # Residualized SiLU starts at exactly 1x and can learn to
                # suppress or amplify each token/head independently.
                gate = 1.0 + F.silu(logits)
            attended = attended * gate.transpose(1, 2).unsqueeze(-1)
        attended = attended.transpose(1, 2).reshape(batch, token_count, self.width)
        tokens = tokens + self.attn_dropout(self.attn_out(attended))
        if self.uses_adaptive_ffn_norm:
            assert condition is not None and self.norm2_scale is not None
            scale = _ada_scale_offset(
                self.norm2_scale(condition), self.ada_scale_mode,
            )
            if self.uses_adaptive_shift:
                assert isinstance(self.norm2, AdaRMSNorm) and self.norm2_shift is not None
                normalized = self.norm2(tokens, scale, self.norm2_shift(condition))
            else:
                assert isinstance(self.norm2, ScaleOnlyAdaRMSNorm)
                normalized = self.norm2(tokens, scale)
        else:
            normalized = self.norm2(tokens)
        if self.ffn_meta_proj is not None:
            assert condition is not None
            gated_ffn = self.ffn[0]
            projected = gated_ffn.gated.proj(normalized)
            projected = projected + self.ffn_meta_proj(condition).unsqueeze(1)
            gate, value = projected.chunk(2, dim=-1)
            ffn_output = gated_ffn.output(F.silu(gate) * value)
            ffn_output = self.ffn[1](ffn_output)
        else:
            ffn_output = self.ffn(normalized)
        return tokens + self.ffn_dropout(ffn_output)


class AttentionPoolingGQA(nn.Module):
    """Learned-query GQA pooling without an attention gate or positional RoPE."""

    def __init__(self, width: int, heads: int, kv_heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads or heads % kv_heads:
            raise ValueError("pool width/heads must satisfy GQA divisibility")
        self.width = width
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.query_token = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.query_token, std=width ** -0.5)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.kv_proj = nn.Linear(width, 2 * kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.dropout = dropout

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, _ = tokens.shape
        query = self.q_proj(self.query_token.expand(batch, -1, -1)).reshape(
            batch, 1, self.heads, self.head_dim,
        ).transpose(1, 2)
        key_value = self.kv_proj(tokens).reshape(
            batch, count, 2, self.kv_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        key, value = key_value.unbind(0)
        pooled = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.heads != self.kv_heads,
        )
        return self.out_proj(pooled.transpose(1, 2).reshape(batch, 1, self.width)[:, 0])


class SpatialGQAStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        *,
        grid_size: int,
        heads: int,
        kv_heads: int,
        blocks: int,
        variant: str,
        dropout: float,
        ff_mult: float,
        condition_dim: int | None,
        concat_condition_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1, bias=False),
            RMSNorm2d(width),
            nn.SiLU(),
        )
        self.meta_projection = None
        if concat_condition_dim is not None:
            if concat_condition_dim <= 0:
                raise ValueError("concat_condition_dim must be positive")
            self.meta_projection = nn.Conv2d(
                width + concat_condition_dim, width, kernel_size=1, bias=False,
            )
            # Preserve the unconditioned stem at initialization while allowing
            # the projection to learn how to use the appended metadata planes.
            with torch.no_grad():
                nn.init.zeros_(self.meta_projection.weight)
                identity = torch.eye(width, dtype=self.meta_projection.weight.dtype)
                self.meta_projection.weight[:, :width, 0, 0].copy_(identity)
        self.blocks = nn.ModuleList(
            GQATransformerBlock2D(
                width, heads, kv_heads, grid_size=grid_size, variant=variant,
                dropout=dropout, ff_mult=ff_mult, condition_dim=condition_dim,
            )
            for _ in range(blocks)
        )


class MiniImageNetGQAModel(nn.Module):
    """(conv downsample + M GQA blocks) x N, attention pool, gated head."""

    def __init__(
        self,
        *,
        num_classes: int,
        widths: Sequence[int] = (96, 192, 256),
        heads: int = 4,
        kv_heads: int = 2,
        blocks_per_stage: int = 1,
        image_size: int = 64,
        variant: str = "naive_gqa",
        dropout: float = 0.1,
        ff_mult: float = 4.0,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        widths = tuple(int(width) for width in widths)
        if num_classes <= 1 or not widths or min(widths) <= 0:
            raise ValueError("num_classes and all stage widths must be valid")
        if blocks_per_stage <= 0 or image_size <= 0 or image_size % (2 ** len(widths)):
            raise ValueError("blocks_per_stage must be positive and image_size divisible by 2**stages")
        self.num_classes = num_classes
        self.widths = widths
        self.image_size = image_size
        self.input_divisor = 2 ** len(widths)
        self.variant = variant
        if variant not in _VARIANT_SETTINGS or variant not in (
            *VARIANTS, *_LEGACY_VARIANTS, *_DIAGNOSTIC_VARIANTS,
        ):
            raise ValueError(f"unknown model variant: {variant}")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        self.meta_embedding = (
            nn.Sequential(
                nn.Linear(META_FEATURES, condition_dim),
                nn.SiLU(),
                nn.Linear(condition_dim, condition_dim),
            )
            if variant in METADATA_VARIANTS else None
        )
        in_channels = 3
        stages = []
        grid_size = image_size
        for stage_width in widths:
            grid_size //= 2
            stages.append(SpatialGQAStage(
                in_channels, stage_width, grid_size=grid_size, heads=heads,
                kv_heads=kv_heads, blocks=blocks_per_stage, variant=variant,
                dropout=dropout, ff_mult=ff_mult,
                condition_dim=(
                    condition_dim
                    if variant in ADAPTIVE_VARIANTS or variant in BRANCH_META_VARIANTS
                    else None
                ),
                concat_condition_dim=(
                    condition_dim
                    if variant == CONCAT_META_VARIANT and not stages
                    else None
                ),
            ))
            in_channels = stage_width
        self.stages = nn.ModuleList(stages)
        final_width = widths[-1]
        self.pool = AttentionPoolingGQA(final_width, heads, kv_heads, dropout)
        self.output_norm = nn.RMSNorm(final_width)
        self.output_ffn = GatedFFN(
            final_width, max(1, final_width * 2), backend="torch",
        )
        self.classifier = nn.Linear(final_width, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (B, 3, H, W)")
        image_height, image_width = images.shape[-2:]
        if (
            image_height <= 0 or image_width <= 0
            or image_height % self.input_divisor
            or image_width % self.input_divisor
        ):
            raise ValueError(
                f"image H/W must be positive multiples of {self.input_divisor}; "
                f"got {(image_height, image_width)}"
            )
        condition = None
        if self.meta_embedding is not None:
            if metadata is None:
                raise ValueError("metadata variant requires bucket resolution/aspect metadata")
            if metadata.ndim != 2 or metadata.shape != (images.shape[0], META_FEATURES):
                raise ValueError(
                    f"metadata must have shape (B, {META_FEATURES}); got {tuple(metadata.shape)}"
                )
            if metadata.device != images.device:
                raise ValueError("metadata and images must share a device")
            condition = self.meta_embedding(metadata.to(dtype=images.dtype))
        features = images
        for stage in self.stages:
            features = stage.downsample(features)
            if stage.meta_projection is not None:
                if condition is None:
                    raise ValueError("metadata channel-concat stage requires an embedding")
                features = _conv_with_broadcast_condition(
                    features, condition, stage.meta_projection,
                )
            batch, channels, height, width = features.shape
            tokens = features.flatten(2).transpose(1, 2)
            for block in stage.blocks:
                block_condition = (
                    condition
                    if self.variant in ADAPTIVE_VARIANTS or self.variant in BRANCH_META_VARIANTS
                    else None
                )
                tokens = block(tokens, (height, width), block_condition)
            features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        tokens = features.flatten(2).transpose(1, 2)
        pooled = self.pool(tokens)
        return self.classifier(self.output_ffn(self.output_norm(pooled)))



def copy_common_initialization_(
    target: MiniImageNetGQAModel,
    reference: MiniImageNetGQAModel,
) -> int:
    """Copy shared stem, attention, normalization, pooling, and classifier weights.

    Variant-specific head gates, block FFNs, and metadata modules retain the
    target model's seeded initialization. This is for controlled comparisons,
    not checkpoint loading.
    """
    target_state = target.state_dict()
    reference_state = reference.state_dict()

    def is_shared(name: str) -> bool:
        if name.startswith(("pool.", "output_norm.", "output_ffn.", "classifier.")):
            return True
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "stages":
            return False
        if parts[2] == "downsample":
            return True
        if parts[2] != "blocks":
            return False
        branch_name = parts[4] if len(parts) > 4 else ""
        if branch_name in {"q_proj", "kv_proj", "q_norm", "k_norm", "attn_out"}:
            return True
        return branch_name in {"norm1", "norm2"} and parts[5:] == ["weight"]

    copied = 0
    with torch.no_grad():
        for name, target_tensor in target_state.items():
            if not is_shared(name):
                continue
            reference_tensor = reference_state.get(name)
            if reference_tensor is None:
                raise ValueError(f"common initialization key missing from reference: {name}")
            if target_tensor.shape != reference_tensor.shape:
                raise ValueError(
                    f"common initialization shape mismatch for {name}: "
                    f"{tuple(target_tensor.shape)} != {tuple(reference_tensor.shape)}"
                )
            target_tensor.copy_(reference_tensor)
            copied += target_tensor.numel()
    if copied == 0:
        raise ValueError("no common initialization weights were copied")
    return copied
