"""Optional exact CAME update kernels.

The module is deliberately import-safe when Triton or CUDA is unavailable.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on the runtime environment
    triton = None
    tl = None


def is_available() -> bool:
    """Return whether the optional Triton path can be used in this process."""

    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _came_apply_update_kernel(
        parameter_ptr,
        update_ptr,
        n_elements,
        lr,
        decay,
        BLOCK: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        parameter = tl.load(parameter_ptr + offsets, mask=mask)
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0)
        # Same recurrence as p.add_(p, alpha=-wd*lr), then p.add_(-lr*update).
        result = parameter * decay - update * lr
        tl.store(parameter_ptr + offsets, result, mask=mask)


def apply_update(
    parameter: torch.Tensor,
    update: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
) -> bool:
    """Apply CAME's final update; return False when the safe fallback is needed."""

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
    grid = (triton.cdiv(parameter.numel(), block),)
    _came_apply_update_kernel[grid](  # pyright: ignore[reportIndexIssue]
        parameter,
        update,
        parameter.numel(),
        lr,
        1.0 - weight_decay * lr,
        BLOCK=block,
    )
    return True
