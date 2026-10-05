"""Adaptive RMS normalization components.

Conditioning projections can be shared across a stack while each block owns
its normalization parameters. Keeping those responsibilities separate matches
VFP-DiT's AdaRMS contract without coupling it to a specific backbone.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaRMSScaleProjection(nn.Module):
    """Project a condition vector to a scale, initialized to the identity."""

    def __init__(self, condition_dim: int, width: int, *, zero_init: bool = True) -> None:
        super().__init__()
        if condition_dim <= 0 or width <= 0:
            raise ValueError("condition_dim and width must be positive")
        self.condition_dim = condition_dim
        self.width = width
        self.proj = nn.Linear(condition_dim, width)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape (B, {self.condition_dim}); got {tuple(condition.shape)}"
            )
        return self.proj(condition)


class ScaleOnlyAdaRMSNorm(nn.Module):
    """Apply RMSNorm followed by ``(1 + scale)`` with batch broadcasting.

    Inputs use ``(..., width)`` layout and scales use ``(B, width)``. The scale
    is cast to the normalized activation dtype before broadcasting, matching
    the usual autocast boundary while keeping projection gradients intact.
    """

    def __init__(
        self,
        width: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.width = width
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width)) if elementwise_affine else None

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-1] != self.width:
            raise ValueError(f"x must have shape (B, ..., {self.width}); got {tuple(x.shape)}")
        if scale.ndim != 2 or scale.shape != (x.shape[0], self.width):
            raise ValueError(
                f"scale must have shape ({x.shape[0]}, {self.width}); got {tuple(scale.shape)}"
            )
        if x.device != scale.device:
            raise ValueError(f"x and scale must share a device; got {x.device} and {scale.device}")

        # Accumulate the RMS in FP32, then return to the activation dtype.
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        normalized = normalized.to(dtype=x.dtype)

        broadcast_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (self.width,)
        scale = scale.to(dtype=x.dtype).reshape(broadcast_shape)
        return normalized * (1 + scale)


class AdaRMSNorm(ScaleOnlyAdaRMSNorm):
    """RMSNorm with per-sample channel scale and shift conditioning.

    Applies ``(1 + scale) * RMSNorm(x) + shift``. Scale and shift have shape
    ``(B, width)`` and broadcast across any token or spatial axes.
    """

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if shift.ndim != 2 or shift.shape != (x.shape[0], self.width):
            raise ValueError(
                f"shift must have shape ({x.shape[0]}, {self.width}); "
                f"got {tuple(shift.shape)}"
            )
        if x.device != shift.device:
            raise ValueError(f"x and shift must share a device; got {x.device} and {shift.device}")
        normalized = super().forward(x, scale)
        broadcast_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (self.width,)
        return normalized + shift.to(dtype=x.dtype).reshape(broadcast_shape)
