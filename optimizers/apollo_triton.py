"""Optional Triton kernels for APOLLO-CAME low-rank updates.

The kernels operate on the projected FP32 gradient and CAME's low-rank
statistics.  They intentionally leave the global RMS reduction in PyTorch;
this keeps the kernel small while fusing the repeated elementwise operations.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # Triton is optional.
    triton = None
    tl = None


def is_available(*tensors: torch.Tensor) -> bool:
    """Return whether the fused CAME update can run for these tensors."""
    return (
        triton is not None
        and bool(tensors)
        and all(
            tensor.is_cuda
            and tensor.dtype == torch.float32
            and tensor.is_contiguous()
            for tensor in tensors
        )
        and all(tensor.device == tensors[0].device for tensor in tensors)
    )


if triton is not None:

    @triton.jit
    def _came_raw_update_kernel(
        gradient_ptr,
        row_stats_ptr,
        col_stats_ptr,
        row_mean_ptr,
        output_ptr,
        n_elements,
        n_cols,
        eps,
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        rows = offsets // n_cols
        cols = offsets - rows * n_cols

        gradient = tl.load(gradient_ptr + offsets, mask=mask, other=0.0)
        row_stat = tl.load(row_stats_ptr + rows, mask=mask, other=0.0)
        col_stat = tl.load(col_stats_ptr + cols, mask=mask, other=0.0)
        row_mean = tl.load(row_mean_ptr)
        row_factor = tl.rsqrt(tl.maximum(row_stat / tl.maximum(row_mean, eps), eps))
        col_factor = tl.rsqrt(tl.maximum(col_stat, eps))
        tl.store(
            output_ptr + offsets,
            gradient * row_factor * col_factor,
            mask=mask,
        )

    @triton.jit
    def _came_apply_update_kernel(
        output_ptr,
        exp_avg_ptr,
        clip_scale_ptr,
        n_elements,
        beta1,
        one_minus_beta1,
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        update = tl.load(output_ptr + offsets, mask=mask, other=0.0)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
        clip_scale = tl.load(clip_scale_ptr)
        update = update * clip_scale
        exp_avg = exp_avg * beta1 + update * one_minus_beta1
        tl.store(output_ptr + offsets, update, mask=mask)
        tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)


def fused_came_adaptive_update(
    gradient: torch.Tensor,
    exp_avg_sq_row: torch.Tensor,
    exp_avg_sq_col: torch.Tensor,
    exp_avg: torch.Tensor,
    output: torch.Tensor,
    *,
    beta1: float,
    clip_threshold: float,
    eps: float,
    backend: str = "auto",
) -> bool:
    """Fuse CAME factorization, clipping, and first-moment update.

    ``output`` receives the clipped adaptive update and ``exp_avg`` is updated
    in-place.  The caller is responsible for computing and passing the RMS
    clip scale through the ordinary PyTorch fallback when this function
    returns ``False``; keeping that reduction outside the kernel avoids an
    atomic accumulation path for every optimizer tensor.
    """
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    tensors = (
        gradient,
        exp_avg_sq_row,
        exp_avg_sq_col,
        exp_avg,
        output,
    )
    available = is_available(*tensors)
    if backend == "triton" and not available:
        raise RuntimeError(
            "APOLLO-CAME Triton backend requires CUDA, Triton, contiguous "
            "FP32 tensors, and matching devices"
        )
    if backend == "torch" or not available:
        return False
    if backend == "triton" or backend == "auto":
        if gradient.ndim != 2:
            return False
        if output.shape != gradient.shape or exp_avg.shape != gradient.shape:
            return False
        if exp_avg_sq_row.shape != (gradient.shape[0],):
            return False
        if exp_avg_sq_col.shape != (gradient.shape[1],):
            return False

        rows, cols = gradient.shape
        block = 1
        while block < 256:
            block *= 2
        row_mean = exp_avg_sq_row.mean().clamp_min_(eps)
        _came_raw_update_kernel[(triton.cdiv(gradient.numel(), block),)](  # pyright: ignore[reportIndexIssue]
            gradient,
            exp_avg_sq_row,
            exp_avg_sq_col,
            row_mean,
            output,
            gradient.numel(),
            cols,
            eps,
            BLOCK=block,
        )

        rms = output.float().square().mean().sqrt()
        clip_scale = (rms / clip_threshold).clamp_min(1.0).reciprocal()
        _came_apply_update_kernel[(triton.cdiv(gradient.numel(), block),)](  # pyright: ignore[reportIndexIssue]
            output,
            exp_avg,
            clip_scale,
            gradient.numel(),
            beta1,
            1.0 - beta1,
            BLOCK=block,
        )
        return True
    return False


__all__ = ["fused_came_adaptive_update", "is_available"]
