"""Optional exact final-update kernel for Muon optimizers."""

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
    def _apply_kernel(  # pyright: ignore[reportInvalidTypeForm]
        parameter_ptr, update_ptr, n_elements, lr, decay,
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        parameter = tl.load(parameter_ptr + offsets, mask=mask)
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0)
        tl.store(parameter_ptr + offsets, parameter * decay - update * lr, mask=mask)


def apply(parameter, update, *, lr: float, weight_decay: float) -> bool:
    if not is_available():
        return False
    if (
        parameter.device.type != "cuda"
        or update.device != parameter.device
        or not parameter.is_contiguous()
        or not update.is_contiguous()
        or parameter.numel() != update.numel()
    ):
        return False
    block = 1
    while block < parameter.numel() and block < 1024:
        block *= 2
    _apply_kernel[(triton.cdiv(parameter.numel(), block),)](  # pyright: ignore[reportIndexIssue]
        parameter, update, parameter.numel(), lr,
        1.0 - lr * weight_decay, BLOCK=block,
    )
    return True
