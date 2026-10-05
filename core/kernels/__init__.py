"""Optional reusable fused kernels with portable PyTorch fallbacks."""

from .ffn import (
    gated_ffn,
    gated_ffn_naive,
    gated_ffn_triton,
    gated_silu,
    gated_silu_naive,
    gated_silu_triton,
)
from .rms_norm import rms_norm, rms_norm_naive, rms_norm_triton
from .rope import apply_rope, apply_rope_naive, apply_rope_triton

__all__ = [
    "gated_silu",
    "gated_silu_naive",
    "gated_silu_triton",
    "gated_ffn",
    "gated_ffn_naive",
    "gated_ffn_triton",
    "apply_rope",
    "apply_rope_naive",
    "apply_rope_triton",
    "rms_norm",
    "rms_norm_naive",
    "rms_norm_triton",
]
