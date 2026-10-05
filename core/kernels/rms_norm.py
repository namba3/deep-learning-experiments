"""Reusable RMSNorm implementations.

The public :func:`rms_norm` dispatcher keeps the reference PyTorch path
available on CPU and on installations without Triton.  The explicit Triton
path currently fuses the forward reduction and normalization; its backward is
implemented with ordinary PyTorch operations so it remains differentiable
without requiring a Triton backward kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # Triton is optional.
    triton = None
    tl = None


def rms_norm_naive(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference RMSNorm with FP32 accumulation and input-dtype output."""
    x_float = x.float()
    inverse_rms = torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + eps,
    )
    output = (x_float * inverse_rms).to(dtype=x.dtype)
    if weight is not None:
        output = output * weight.to(dtype=x.dtype)
    return output


def _triton_is_available(x: torch.Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[-1] > 0
        and x.shape[-1] <= 4096
        and x.is_contiguous()
    )


def triton_available(x: torch.Tensor) -> bool:
    """Return whether the RMSNorm Triton implementation can consume ``x``."""
    return _triton_is_available(x)


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_triton_weight(
    x: torch.Tensor,
    weight: torch.Tensor | None,
) -> None:
    """Validate the flat weight contract required by the Triton kernel."""
    if weight is None:
        return
    if tuple(weight.shape) != (x.shape[-1],):
        raise ValueError(
            f"RMSNorm weight shape must be {(x.shape[-1],)}, got {tuple(weight.shape)}"
        )
    if weight.device != x.device:
        raise ValueError(
            "RMSNorm weight and input must be on the same device; "
            f"got weight={weight.device}, input={x.device}"
        )
    if weight.dtype != x.dtype:
        raise ValueError(
            "RMSNorm Triton weight and input must have the same dtype; "
            f"got weight={weight.dtype}, input={x.dtype}"
        )
    if not weight.is_contiguous():
        raise ValueError("RMSNorm Triton weight must be contiguous")


if triton is not None:

    @triton.jit
    def _rms_norm_forward_kernel(
        x_ptr,
        weight_ptr,
        inverse_rms_ptr,
        output_ptr,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        HAS_WEIGHT: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_offsets = row * n_cols + offsets
        x_value = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        x_float = x_value.to(tl.float32)
        mean_square = tl.sum(x_float * x_float, axis=0) / n_cols
        inverse_rms = tl.rsqrt(mean_square + eps)

        output = x_float * inverse_rms
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
            output *= weight.to(tl.float32)

        tl.store(output_ptr + x_offsets, output, mask=mask)
        tl.store(inverse_rms_ptr + row, inverse_rms)


class _TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        x_2d = x.reshape(n_rows, n_cols)
        output = torch.empty_like(x_2d)
        inverse_rms = torch.empty(
            n_rows,
            1,
            device=x.device,
            dtype=torch.float32,
        )
        weight_arg = x_2d if weight is None else weight
        block_size = _next_power_of_two(n_cols)
        _rms_norm_forward_kernel[(n_rows,)](  # pyright: ignore[reportIndexIssue]
            x_2d,
            weight_arg,
            inverse_rms,
            output,
            n_rows,
            n_cols,
            float(eps),
            BLOCK_SIZE=block_size,
            HAS_WEIGHT=weight is not None,
        )
        ctx.has_weight = weight is not None
        ctx.n_cols = n_cols
        if weight is None:
            ctx.save_for_backward(x, inverse_rms)
        else:
            ctx.save_for_backward(x, inverse_rms, weight)
        return output.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x, inverse_rms = saved[:2]
        weight = saved[2] if ctx.has_weight else None
        x_float = x.float()
        grad_float = grad_output.float()
        if weight is not None:
            weighted_grad = grad_float * weight.float()
        else:
            weighted_grad = grad_float

        # The Triton forward stores one inverse RMS per flattened row. Restore
        # the original leading dimensions before broadcasting in backward.
        inverse_rms = inverse_rms.reshape(*x.shape[:-1], 1)
        dot = (weighted_grad * x_float).sum(dim=-1, keepdim=True)
        inverse_cubed = inverse_rms * inverse_rms * inverse_rms
        grad_x = (
            weighted_grad * inverse_rms - x_float * inverse_cubed * (dot / ctx.n_cols)
        ).to(dtype=x.dtype)

        if weight is None:
            return grad_x, None, None
        reduce_dims = tuple(range(x.ndim - 1))
        grad_weight = (
            (grad_float * x_float * inverse_rms)
            .sum(dim=reduce_dims)
            .to(dtype=weight.dtype)
        )
        return grad_x, grad_weight, None


def rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run the Triton RMSNorm forward with a differentiable PyTorch backward."""
    _validate_triton_weight(x, weight)
    if not _triton_is_available(x):
        raise RuntimeError(
            "RMSNorm Triton backend requires CUDA, Triton, a supported dtype, "
            "a contiguous tensor, and a last dimension no larger than 4096"
        )
    return _TritonRMSNormFunction.apply(x, weight, float(eps))


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Dispatch RMSNorm to ``torch``, ``triton``, or an automatic backend."""
    if backend not in {"auto", "torch", "naive", "triton"}:
        raise ValueError(f"unknown RMSNorm backend: {backend}")
    if backend in {"torch", "naive"}:
        if backend == "torch" and hasattr(F, "rms_norm"):
            native_weight = None if weight is None else weight.to(dtype=x.dtype)
            return F.rms_norm(x, (x.shape[-1],), weight=native_weight, eps=eps)
        return rms_norm_naive(x, weight, eps)
    if backend == "triton":
        return rms_norm_triton(x, weight, eps)

    # Keep auto conservative: native PyTorch has a mature backward and may
    # select a better architecture-specific kernel than this optional bridge.
    if hasattr(F, "rms_norm"):
        native_weight = None if weight is None else weight.to(dtype=x.dtype)
        return F.rms_norm(x, (x.shape[-1],), weight=native_weight, eps=eps)
    if _triton_is_available(x):
        return rms_norm_triton(x, weight, eps)
    return rms_norm_naive(x, weight, eps)


__all__ = [
    "rms_norm",
    "rms_norm_naive",
    "rms_norm_triton",
    "triton_available",
]
