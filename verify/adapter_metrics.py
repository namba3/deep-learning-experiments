"""Shared numerical checks for low-rank adapter verification probes."""

from __future__ import annotations

import torch


def merge_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return ``(atol, rtol)`` for wrapped-vs-merged Linear outputs.

    BF16 evaluates the unmerged ``base(input) + adapter(input)`` path and the
    merged single-Linear path with different accumulation and rounding order.
    The tolerance therefore needs to cover a few BF16 ulps while remaining
    strict for FP32.
    """
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 3e-2, 1e-2
    raise ValueError(f"unsupported adapter comparison dtype: {dtype}")


def merge_is_equivalent(
    before: torch.Tensor,
    after: torch.Tensor,
    dtype: torch.dtype,
) -> bool:
    """Check numerical equivalence after adapter weight materialization."""
    atol, rtol = merge_tolerances(dtype)
    return torch.allclose(before, after, atol=atol, rtol=rtol)
