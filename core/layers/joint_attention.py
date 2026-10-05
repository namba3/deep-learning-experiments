"""Joint attention over a primary memory and a masked context memory.

The layer owns shared Q/K/V/output projections and delegates the ordinary
attention path to PyTorch SDPA, allowing PyTorch to select an eligible fused
backend. Normalization and positional transforms stay outside the layer so
modality-specific AdaRMSNorm and RoPE policies can be supplied by callers.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class JointKVSDPAAttention(nn.Module):
    """Attend from queries over concatenated primary and context K/V memories.

    Args:
        width: Input and output feature width.
        heads: Number of query/key/value heads.
        dropout_p: Attention dropout used only while training.
        bias: Whether Q/K/V/output projections use bias.
        diagnostic_query_chunk: Query chunk size for optional mass diagnostics.

    ``query`` has shape ``(B, T, D)``. ``primary_memory`` and
    ``context_memory`` have shapes ``(B, P, D)`` and ``(B, C, D)``. Primary
    memory keys are always valid; ``context_mask`` is boolean ``(B, C)`` with
    ``True`` for valid context keys. All inputs must share device and floating
    dtype. Output shape is ``(B, T, D)`` and preserves query order.

    With ``return_mass=True``, this method requires eval mode and
    ``torch.no_grad()``. It returns per-sample primary/context probability-mass
    means and population standard deviations. The diagnostic path computes
    query chunks explicitly and does not retain a full attention map.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        dropout_p: float = 0.0,
        bias: bool = True,
        diagnostic_query_chunk: int = 128,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width and heads must be positive, and width divisible by heads")
        if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be finite and in [0, 1)")
        if diagnostic_query_chunk <= 0:
            raise ValueError("diagnostic_query_chunk must be positive")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.dropout_p = float(dropout_p)
        self.diagnostic_query_chunk = int(diagnostic_query_chunk)
        self.q_proj = nn.Linear(width, width, bias=bias)
        self.k_proj = nn.Linear(width, width, bias=bias)
        self.v_proj = nn.Linear(width, width, bias=bias)
        self.out_proj = nn.Linear(width, width, bias=bias)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        # (B, S, D) -> (B, H, S, Dh); H*Dh == D by constructor contract.
        return value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        primary_memory: torch.Tensor,
        context_memory: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        *,
        return_mass: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if query.ndim != 3 or primary_memory.ndim != 3 or context_memory.ndim != 3:
            raise ValueError("query and memories must have shapes (B, T/P/C, D)")
        batch, query_tokens, query_width = query.shape
        primary_tokens = primary_memory.shape[1]
        context_tokens = context_memory.shape[1]
        if batch <= 0:
            raise ValueError("batch size must be positive")
        if query_tokens <= 0 or primary_tokens <= 0 or context_tokens <= 0:
            raise ValueError("query and memory sequences must be non-empty")
        if (
            query_width != self.width
            or primary_memory.shape[0] != batch
            or primary_memory.shape[2] != self.width
            or context_memory.shape[0] != batch
            or context_memory.shape[2] != self.width
        ):
            raise ValueError("query/memory batch or width does not match JointKVSDPAAttention")
        if not query.is_floating_point() or not primary_memory.is_floating_point() or not context_memory.is_floating_point():
            raise TypeError("query and memory tensors must use floating-point dtypes")
        if query.device != primary_memory.device or query.device != context_memory.device:
            raise ValueError("query and memories must be on the same device")
        if query.dtype != primary_memory.dtype or query.dtype != context_memory.dtype:
            raise TypeError("query and memories must use the same dtype")
        if return_mass and (self.training or torch.is_grad_enabled()):
            raise ValueError("attention-mass diagnostics require eval mode under torch.no_grad()")

        if context_mask is None:
            context_mask = torch.ones(
                context_memory.shape[:2], device=query.device, dtype=torch.bool,
            )
        else:
            if context_mask.shape != context_memory.shape[:2]:
                raise ValueError("context_mask must have shape (B, C)")
            if context_mask.dtype != torch.bool:
                raise TypeError("context_mask must use boolean validity values")
            if context_mask.device != query.device:
                raise ValueError("context_mask and query must be on the same device")
        if not context_mask.any(dim=-1).all():
            raise ValueError("Every sample must contain at least one valid context token")

        query_heads = self._split_heads(self.q_proj(query))
        # One shared K/V projection is applied to both memory sources.
        memory = torch.cat((primary_memory, context_memory), dim=1)
        key_heads = self._split_heads(self.k_proj(memory))
        value_heads = self._split_heads(self.v_proj(memory))
        primary_valid = torch.ones(
            (batch, primary_tokens), device=query.device, dtype=torch.bool,
        )
        key_valid = torch.cat((primary_valid, context_mask), dim=1)
        attention_mask = key_valid[:, None, None, :]

        if return_mass:
            attended, mass = self._diagnostic_attention(
                query_heads, key_heads, value_heads, attention_mask, primary_tokens,
            )
        else:
            attended = F.scaled_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                attn_mask=attention_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
            mass = None

        # (B, H, T, Dh) -> (B, T, D), preserving query order.
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch, query_tokens, self.width,
        )
        return self.out_proj(attended), mass

    def _diagnostic_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        primary_tokens: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute output and source-mass statistics without a full map."""
        _batch, heads, query_tokens, _head_dim = query.shape
        scale = 1.0 / math.sqrt(self.head_dim)
        output_chunks = []
        primary_sum = query.new_zeros((query.shape[0],), dtype=torch.float64)
        primary_square_sum = torch.zeros_like(primary_sum)
        context_sum = torch.zeros_like(primary_sum)
        context_square_sum = torch.zeros_like(primary_sum)
        count = 0

        for start in range(0, query_tokens, self.diagnostic_query_chunk):
            stop = min(start + self.diagnostic_query_chunk, query_tokens)
            # FP32 logits/softmax stabilize diagnostic mass under BF16 inference.
            logits = torch.matmul(
                query[:, :, start:stop].float(), key.float().transpose(-2, -1),
            ) * scale
            logits = logits.masked_fill(~attention_mask, float("-inf"))
            probabilities = logits.softmax(dim=-1)
            output_chunks.append(
                torch.matmul(probabilities, value.float()).to(value.dtype),
            )

            primary_mass = probabilities[..., :primary_tokens].sum(dim=-1)
            context_mass = probabilities[..., primary_tokens:].sum(dim=-1)
            primary_sum += primary_mass.sum(dim=(1, 2), dtype=torch.float64)
            primary_square_sum += primary_mass.square().sum(dim=(1, 2), dtype=torch.float64)
            context_sum += context_mass.sum(dim=(1, 2), dtype=torch.float64)
            context_square_sum += context_mass.square().sum(dim=(1, 2), dtype=torch.float64)
            count += heads * (stop - start)

        primary_mean = primary_sum / count
        context_mean = context_sum / count
        mass = {
            "primary_mean": primary_mean.float(),
            "primary_std": (primary_square_sum / count - primary_mean.square()).clamp_min(0).sqrt().float(),
            "context_mean": context_mean.float(),
            "context_std": (context_square_sum / count - context_mean.square()).clamp_min(0).sqrt().float(),
        }
        return torch.cat(output_chunks, dim=2), mass


__all__ = ["JointKVSDPAAttention"]
