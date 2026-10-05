"""Naive, reusable DiT transformer block with GQA and a condition head gate.

The Q, K, V and two SwiGLU input projections intentionally remain separate
Linear calls. Their inputs currently match, but concatenating them into a
single GEMM is a later optimization that must preserve this implementation's
outputs and gradients.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUFeedForward(nn.Module):
    """Reference SwiGLU MLP using separate gate and value projections."""

    def __init__(
        self,
        width: int,
        hidden_width: int,
        *,
        output_width: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if width <= 0 or hidden_width <= 0:
            raise ValueError("width and hidden_width must be positive")
        if output_width is None:
            output_width = width
        if output_width <= 0:
            raise ValueError("output_width must be positive")
        self.gate_proj = nn.Linear(width, hidden_width, bias=bias)
        self.value_proj = nn.Linear(width, hidden_width, bias=bias)
        self.out_proj = nn.Linear(hidden_width, output_width, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class DiTTransformerBlock(nn.Module):
    """Pre-norm DiT block for token sequences shaped (B, T, width).

    The block applies q/k RMSNorm, grouped-query attention through SDPA, an
    optional per-sample head gate projected from z with shape (B, condition_dim),
    and a SwiGLU MLP. qk_transform optionally applies caller-owned positional
    transforms to Q=(B, heads, T, head_dim) and
    K=(B, kv_heads, T, head_dim); this keeps 1D, 2D, and modality-aware RoPE
    coordinate policies outside this general block.

    Boolean attention_mask values follow SDPA semantics: True means the key is
    valid. Additive floating masks are also accepted. Supported masks must
    broadcast to (B, heads, T, T).
    """

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        kv_heads: int | None = None,
        ff_mult: float = 4.0,
        condition_dim: int | None = None,
        qk_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        if width <= 0 or heads <= 0 or kv_heads <= 0:
            raise ValueError("width, heads, and kv_heads must be positive")
        if width % heads or heads % kv_heads:
            raise ValueError("width must divide heads and heads must divide into kv_heads")
        if not math.isfinite(ff_mult) or ff_mult <= 0:
            raise ValueError("ff_mult must be finite and positive")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if condition_dim is not None and condition_dim <= 0:
            raise ValueError("condition_dim must be positive when provided")

        self.width = width
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.condition_dim = condition_dim
        self.dropout = float(dropout)

        self.norm1 = nn.RMSNorm(width, eps=eps)
        self.q_proj = nn.Linear(width, width, bias=bias)
        self.k_proj = nn.Linear(width, kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(width, kv_heads * self.head_dim, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.head_gate = (
            nn.Linear(condition_dim, heads) if condition_dim is not None else None
        )
        if self.head_gate is not None:
            nn.init.zeros_(self.head_gate.weight)
            nn.init.zeros_(self.head_gate.bias)
        self.attn_out = nn.Linear(width, width, bias=bias)

        hidden_width = max(1, int(width * ff_mult))
        self.norm2 = nn.RMSNorm(width, eps=eps)
        self.mlp = SwiGLUFeedForward(
            width, hidden_width, output_width=width, bias=bias,
        )

    def _validate_mask(
        self,
        mask: torch.Tensor | None,
        *,
        batch: int,
        tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.device != device:
            raise ValueError("attention_mask and tokens must be on the same device")
        if mask.dtype != torch.bool and not mask.is_floating_point():
            raise TypeError("attention_mask must be boolean or floating point")
        if mask.ndim == 3 and mask.shape == (batch, tokens, tokens):
            # Batch-major masks need an explicit singleton head axis for SDPA.
            mask = mask[:, None, :, :]
        if mask.ndim > 4:
            raise ValueError("attention_mask must broadcast to (B, heads, T, T)")
        try:
            torch.broadcast_shapes(mask.shape, (batch, self.heads, tokens, tokens))
        except RuntimeError as exc:
            raise ValueError(
                f"attention_mask shape {tuple(mask.shape)} cannot broadcast to "
                f"({batch}, {self.heads}, {tokens}, {tokens})"
            ) from exc
        return mask if mask.dtype == torch.bool else mask.to(dtype=dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        qk_transform: Callable[
            [torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ] | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(f"tokens must have shape (B, T, {self.width})")
        batch, token_count, _ = tokens.shape
        if batch <= 0 or token_count <= 0:
            raise ValueError("batch and token dimensions must be non-empty")
        if not tokens.is_floating_point():
            raise TypeError("tokens must use a floating-point dtype")
        if self.condition_dim is None:
            if z is not None:
                raise ValueError("z was provided but this block has no condition_dim")
            head_gate = None
        else:
            if z is None or z.shape != (batch, self.condition_dim):
                raise ValueError(
                    f"z must have shape ({batch}, {self.condition_dim})"
                )
            if z.device != tokens.device:
                raise ValueError("z and tokens must be on the same device")
            assert self.head_gate is not None
            head_gate = 2.0 * torch.sigmoid(
                self.head_gate(z.to(dtype=self.head_gate.weight.dtype))
            )

        normalized = self.norm1(tokens)
        # Keep Q/K/V projections explicit as the numerical reference path.
        query = self.q_proj(normalized).reshape(
            batch, token_count, self.heads, self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(normalized).reshape(
            batch, token_count, self.kv_heads, self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(normalized).reshape(
            batch, token_count, self.kv_heads, self.head_dim,
        ).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if qk_transform is not None:
            query, key = qk_transform(query, key)
        if query.shape != (batch, self.heads, token_count, self.head_dim):
            raise ValueError("qk_transform returned a query with an invalid shape")
        if key.shape != (batch, self.kv_heads, token_count, self.head_dim):
            raise ValueError("qk_transform returned a key with an invalid shape")
        if query.device != tokens.device or key.device != tokens.device:
            raise ValueError("qk_transform must preserve the query/key device")
        if query.dtype != value.dtype or key.dtype != value.dtype:
            raise TypeError("qk_transform must preserve the Q/K/V dtype")

        mask = self._validate_mask(
            attention_mask,
            batch=batch,
            tokens=token_count,
            device=tokens.device,
            dtype=query.dtype,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.heads != self.kv_heads,
        )
        if head_gate is not None:
            attended = attended * head_gate[:, :, None, None].to(attended.dtype)
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch, token_count, self.width,
        )
        hidden = tokens + self.attn_out(attended)
        return hidden + self.mlp(self.norm2(hidden))


__all__ = ["DiTTransformerBlock", "SwiGLUFeedForward"]
