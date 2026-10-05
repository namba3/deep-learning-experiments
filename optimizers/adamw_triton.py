"""Optional exact AdamW Triton kernels with a safe PyTorch fallback."""

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
    def _adamw_kernel(
        parameter_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        max_exp_avg_sq_ptr,
        n_elements,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
        lr,
        weight_decay,
        HAS_AMSGRAD: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        parameter = tl.load(parameter_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

        exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
        exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
        tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)

        if HAS_AMSGRAD:
            max_exp_avg_sq = tl.load(max_exp_avg_sq_ptr + offsets, mask=mask)
            max_exp_avg_sq = tl.maximum(exp_avg_sq, max_exp_avg_sq)
            tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)
            second_moment = max_exp_avg_sq
        else:
            second_moment = exp_avg_sq

        denominator = tl.sqrt(second_moment) / tl.sqrt(bias_correction2) + eps
        update = exp_avg / denominator * (lr / bias_correction1)
        parameter = parameter * (1.0 - lr * weight_decay) - update
        tl.store(parameter_ptr + offsets, parameter, mask=mask)


def apply(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    max_exp_avg_sq: torch.Tensor | None,
    beta1: float,
    beta2: float,
    bias_correction1: float,
    bias_correction2: float,
    eps: float,
    lr: float,
    weight_decay: float,
) -> bool:
    """Run the fused exact recurrence, or return False for the fallback."""

    if not is_available():
        return False
    tensors = (parameter, grad, exp_avg, exp_avg_sq)
    if any(t.device.type != "cuda" or not t.is_contiguous() for t in tensors):
        return False
    if max_exp_avg_sq is not None and (
        max_exp_avg_sq.device != parameter.device or not max_exp_avg_sq.is_contiguous()
    ):
        return False
    if any(t.numel() != parameter.numel() for t in tensors):
        return False

    block = 1
    while block < parameter.numel() and block < 1024:
        block *= 2
    grid = (triton.cdiv(parameter.numel(), block),)
    _adamw_kernel[grid](  # pyright: ignore[reportIndexIssue]
        parameter,
        grad,
        exp_avg,
        exp_avg_sq,
        exp_avg_sq if max_exp_avg_sq is None else max_exp_avg_sq,
        parameter.numel(),
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
        lr,
        weight_decay,
        HAS_AMSGRAD=max_exp_avg_sq is not None,
        BLOCK=block,
    )
    return True


def apply_parameter_update(
    parameter: torch.Tensor,
    update: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
) -> bool:
    """Fuse only the final update used by AdamWAutoSchedule."""

    if not is_available() or parameter.device.type != "cuda":
        return False
    if (
        not parameter.is_contiguous()
        or not update.is_contiguous()
        or parameter.device != update.device
        or parameter.numel() != update.numel()
    ):
        return False
    block = 1
    while block < parameter.numel() and block < 1024:
        block *= 2
    grid = (triton.cdiv(parameter.numel(), block),)

    @triton.jit
    def _kernel(  # pyright: ignore[reportInvalidTypeForm]
        parameter_ptr, update_ptr, n_elements, lr, decay,
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        parameter_value = tl.load(parameter_ptr + offsets, mask=mask)
        update_value = tl.load(update_ptr + offsets, mask=mask, other=0.0)
        tl.store(
            parameter_ptr + offsets,
            parameter_value * decay - update_value * lr,
            mask=mask,
        )

    _kernel[grid](  # pyright: ignore[reportIndexIssue]
        parameter, update, parameter.numel(), lr, 1.0 - lr * weight_decay, BLOCK=block
    )
    return True
