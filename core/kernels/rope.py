"""Reusable pair-wise RoPE implementations with an optional Triton forward."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # Triton is optional.
    triton = None
    tl = None


def _normalize_tables(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 2:
        raise ValueError("RoPE input must have at least token and feature dimensions")
    if x.shape[-1] % 2:
        raise ValueError("RoPE feature dimension must be even")
    tokens = x.shape[-2]
    half_dim = x.shape[-1] // 2
    if cos.numel() != tokens * half_dim or sin.numel() != tokens * half_dim:
        raise ValueError(
            "RoPE tables must contain one compact cos/sin pair per token and "
            f"half-dimension; expected {tokens * half_dim} elements"
        )
    cos = cos.to(device=x.device, dtype=x.dtype).contiguous().reshape(tokens, half_dim)
    sin = sin.to(device=x.device, dtype=x.dtype).contiguous().reshape(tokens, half_dim)
    if torch.is_grad_enabled():
        # Cached RoPE tables may have been created under inference_mode.
        # Autograd cannot save those tensors to compute gradients for x.
        cos = _normal_autograd_tensor(cos)
        sin = _normal_autograd_tensor(sin)
    return cos, sin


def _normal_autograd_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Clone inference-mode tables before saving them for a backward pass."""
    is_inference = getattr(tensor, "is_inference", None)
    if is_inference is not None and is_inference():
        with torch.inference_mode(False):
            return tensor.clone()
    return tensor


def apply_rope_naive(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reference pair-wise RoPE for ``[..., tokens, head_dim]`` tensors."""
    cos, sin = _normalize_tables(x, cos, sin)
    broadcast_shape = (1,) * (x.ndim - 2) + cos.shape
    cos = cos.reshape(broadcast_shape)
    sin = sin.reshape(broadcast_shape)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2)


def _triton_is_available(x: torch.Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[-1] % 2 == 0
        and x.is_contiguous()
    )


def triton_available(x: torch.Tensor) -> bool:
    """Return whether the RoPE Triton implementation can consume ``x``."""
    return _triton_is_available(x)


if triton is not None:

    @triton.jit
    def _rope_forward_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        n_rows,
        tokens,
        head_dim,
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        half_dim = head_dim // 2
        mask = offsets < half_dim
        token = row % tokens
        even_offset = row * head_dim + offsets * 2
        odd_offset = even_offset + 1
        table_offset = token * half_dim + offsets

        even = tl.load(x_ptr + even_offset, mask=mask, other=0.0)
        odd = tl.load(x_ptr + odd_offset, mask=mask, other=0.0)
        cos_value = tl.load(cos_ptr + table_offset, mask=mask, other=0.0)
        sin_value = tl.load(sin_ptr + table_offset, mask=mask, other=0.0)

        tl.store(
            output_ptr + even_offset,
            even * cos_value - odd * sin_value,
            mask=mask,
        )
        tl.store(
            output_ptr + odd_offset,
            even * sin_value + odd * cos_value,
            mask=mask,
        )


class _TritonRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        tokens = x.shape[-2]
        head_dim = x.shape[-1]
        n_rows = x.numel() // head_dim
        output = torch.empty_like(x)
        block_size = 1 << ((head_dim // 2) - 1).bit_length()
        _rope_forward_kernel[(n_rows,)](  # pyright: ignore[reportIndexIssue]
            x,
            cos,
            sin,
            output,
            n_rows,
            tokens,
            head_dim,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(cos, sin)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        # The inverse rotation is the same operation with -sin.  Keeping this
        # path in PyTorch makes the shared kernel usable before a dedicated
        # Triton backward kernel is introduced.
        return apply_rope_naive(grad_output, cos, -sin), None, None


def apply_rope_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Run Triton RoPE forward with a differentiable PyTorch backward."""
    if (
        triton is None
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or x.shape[-1] % 2
    ):
        raise RuntimeError(
            "RoPE Triton backend requires CUDA, Triton, a supported dtype, "
            "and an even head dimension"
        )
    # Q/K commonly come from a transposed grouped-query projection.  The
    # kernel uses flat contiguous row addressing, so materialize that layout
    # here rather than failing when the caller explicitly selected Triton.
    # ``contiguous`` remains differentiable and therefore preserves the
    # gradient path to the original non-contiguous tensor.
    if not x.is_contiguous():
        x = x.contiguous()
    cos, sin = _normalize_tables(x, cos, sin)
    cos = _normal_autograd_tensor(cos)
    sin = _normal_autograd_tensor(sin)
    return _TritonRoPEFunction.apply(x, cos, sin)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Dispatch pair-wise RoPE to a portable or Triton implementation."""
    if backend not in {"auto", "naive", "triton"}:
        raise ValueError(f"unknown RoPE backend: {backend}")
    if backend == "naive":
        return apply_rope_naive(x, cos, sin)
    if backend == "triton":
        return apply_rope_triton(x, cos, sin)
    if _triton_is_available(x):
        return apply_rope_triton(x, cos, sin)
    return apply_rope_naive(x, cos, sin)


__all__ = [
    "apply_rope",
    "apply_rope_naive",
    "apply_rope_triton",
    "triton_available",
]
