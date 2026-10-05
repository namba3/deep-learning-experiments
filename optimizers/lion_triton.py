"""Optional exact Lion Triton kernel with a PyTorch fallback."""

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
    def _lion_kernel(
        parameter_ptr,
        grad_ptr,
        exp_avg_ptr,
        n_elements,
        beta1,
        beta2,
        lr,
        weight_decay,
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        parameter = tl.load(parameter_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)

        # Direction is computed from the old momentum, before the state update.
        direction_input = exp_avg * beta1 + grad * (1.0 - beta1)
        direction = tl.where(direction_input > 0, 1.0, tl.where(direction_input < 0, -1.0, 0.0))
        exp_avg = exp_avg * beta2 + grad * (1.0 - beta2)
        tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

        result = parameter * (1.0 - lr * weight_decay) - lr * direction
        tl.store(parameter_ptr + offsets, result, mask=mask)


def apply(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
) -> bool:
    """Run the fused Lion recurrence, or return False for the fallback."""

    if not is_available():
        return False
    tensors = (parameter, grad, exp_avg)
    if any(t.device.type != "cuda" or not t.is_contiguous() for t in tensors):
        return False
    if any(t.numel() != parameter.numel() for t in tensors):
        return False

    block = 1
    while block < parameter.numel() and block < 1024:
        block *= 2
    grid = (triton.cdiv(parameter.numel(), block),)
    _lion_kernel[grid](  # pyright: ignore[reportIndexIssue]
        parameter,
        grad,
        exp_avg,
        parameter.numel(),
        beta1,
        beta2,
        lr,
        weight_decay,
        BLOCK=block,
    )
    return True
