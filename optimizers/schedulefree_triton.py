"""Optional exact Triton kernel for Schedule-Free AdamW."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - runtime dependent
    triton = None
    tl = None


def is_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _adamw_schedulefree_kernel(
        parameter_ptr, grad_ptr, exp_avg_sq_ptr, z_ptr, n_elements,
        beta2, bias_correction2, eps, decay, ckp1, gradient_scale, z_scale,
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        # ``grid`` may contain multiple programs for tensors larger than one
        # block.  Include the program offset so every element is updated once.
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        parameter = tl.load(parameter_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
        z = tl.load(z_ptr + offsets, mask=mask)
        exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
        denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps
        y = parameter.to(tl.float32)
        grad_normalized = grad / denom + y * decay
        y = y + ckp1 * (z - y) + grad_normalized * gradient_scale
        z = z - grad_normalized * z_scale
        tl.store(parameter_ptr + offsets, y, mask=mask)
        tl.store(z_ptr + offsets, z, mask=mask)


def apply(
    parameter, grad, exp_avg_sq, z, *, beta2, bias_correction2, eps,
    decay, ckp1, gradient_scale, z_scale,
) -> bool:
    if not is_available():
        return False
    tensors = (parameter, grad, exp_avg_sq, z)
    if any(t.device.type != "cuda" or not t.is_contiguous() for t in tensors):
        return False
    if any(t.device != parameter.device or t.numel() != parameter.numel() for t in tensors):
        return False
    block = 1
    while block < parameter.numel() and block < 1024:
        block *= 2
    grid = (triton.cdiv(parameter.numel(), block),)
    _adamw_schedulefree_kernel[grid](  # pyright: ignore[reportIndexIssue]
        parameter, grad, exp_avg_sq, z, parameter.numel(), beta2,
        bias_correction2, eps, decay, ckp1, gradient_scale, z_scale,
        BLOCK=block,
    )
    return True


if triton is not None:

    @triton.jit
    def _adamw_lrsf_precondition_kernel(
        parameter_ptr, grad_ptr, exp_avg_sq_ptr, update_ptr, n_elements,
        beta2, bias_correction2, eps, decay,
        APPLY_DECAY: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        """Fuse AdamW second-moment update and LRSF preconditioning.

        The low-rank path still performs its two matrix multiplications in
        PyTorch/cuBLAS.  This kernel owns only the element-wise part and
        writes the FP32 effective update consumed by those GEMMs.
        """
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        parameter = tl.load(parameter_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
        exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
        denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps
        update = grad / denom
        if APPLY_DECAY:
            update += parameter.to(tl.float32) * decay
        tl.store(update_ptr + offsets, update, mask=mask)


def apply_lrsf_preconditioner(
    parameter, grad, exp_avg_sq, *, beta2, bias_correction2, eps, decay,
):
    """Return an FP32 preconditioned update, or ``None`` if unsupported."""

    if not is_available():
        return None
    tensors = (parameter, grad, exp_avg_sq)
    if any(t.device.type != "cuda" or not t.is_contiguous() for t in tensors):
        return None
    if any(t.device != parameter.device or t.numel() != parameter.numel() for t in tensors):
        return None
    update = torch.empty(
        parameter.shape, device=parameter.device, dtype=torch.float32,
    )
    block = 1
    while block < parameter.numel() and block < 1024:
        block *= 2
    grid = (triton.cdiv(parameter.numel(), block),)
    _adamw_lrsf_precondition_kernel[grid](  # pyright: ignore[reportIndexIssue]
        parameter, grad, exp_avg_sq, update, parameter.numel(), beta2,
        bias_correction2, eps, decay, APPLY_DECAY=decay != 0.0, BLOCK=block,
    )
    return update


if triton is not None:

    @triton.jit
    def _radam_schedulefree_kernel(
        parameter_ptr, grad_ptr, exp_avg_sq_ptr, z_ptr, n_elements,
        beta2, bias_correction2, eps, decay, ckp1, adaptive_y_lr, lr,
        NORMALIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        parameter = tl.load(parameter_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
        z = tl.load(z_ptr + offsets, mask=mask)
        exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
        if NORMALIZE:
            grad = grad / (tl.sqrt(exp_avg_sq / bias_correction2) + eps)
        y = parameter.to(tl.float32)
        grad = grad + y * decay
        y = y + ckp1 * (z - y) + grad * adaptive_y_lr
        z = z - grad * lr
        tl.store(parameter_ptr + offsets, y, mask=mask)
        tl.store(z_ptr + offsets, z, mask=mask)


def apply_radam(
    parameter, grad, exp_avg_sq, z, *, beta2, bias_correction2, eps,
    decay, ckp1, adaptive_y_lr, lr, normalize,
) -> bool:
    """Run the fused RAdam Schedule-Free recurrence if supported."""

    if not is_available():
        return False
    tensors = (parameter, grad, exp_avg_sq, z)
    if any(t.device.type != "cuda" or not t.is_contiguous() for t in tensors):
        return False
    if any(t.device != parameter.device or t.numel() != parameter.numel() for t in tensors):
        return False
    block = 1
    while block < parameter.numel() and block < 1024:
        block *= 2
    grid = (triton.cdiv(parameter.numel(), block),)
    _radam_schedulefree_kernel[grid](  # pyright: ignore[reportIndexIssue]
        parameter, grad, exp_avg_sq, z, parameter.numel(), beta2,
        bias_correction2, eps, decay, ckp1, adaptive_y_lr, lr,
        NORMALIZE=normalize, BLOCK=block,
    )
    return True
