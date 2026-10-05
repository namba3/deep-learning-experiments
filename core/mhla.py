"""Optional Triton bridge for the 2D grid MHLA layer.

The image-generation model already owns the production MHLA Triton kernels.
This small bridge lets the image-only CIFAR model reuse those kernels without
making Triton a hard dependency of :mod:`core.layers` or importing the large
training module during normal CPU/model construction.
"""

from __future__ import annotations

import torch

try:
    import triton as _triton  # type: ignore[import-not-found]
    _TRITON_IMPORT_ERROR = None
except Exception as error:  # Triton is optional.
    _triton = None
    _TRITON_IMPORT_ERROR = repr(error)


def triton_available(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    """Return whether the existing MHLA Triton path can consume these tensors."""
    if _triton is None:
        return False
    return bool(
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
    )


def triton_unavailable_reason(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
) -> str:
    """Explain why the optional Triton backend cannot be used."""
    reasons = []
    if _triton is None:
        reasons.append(f"triton import failed: {_TRITON_IMPORT_ERROR}")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        reasons.append(
            f"cuda flags query={query.is_cuda} key={key.is_cuda} value={value.is_cuda}"
        )
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        reasons.append(
            "contiguous flags "
            f"query={query.is_contiguous()} key={key.is_contiguous()} "
            f"value={value.is_contiguous()}"
        )
    if not (
        query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
    ):
        reasons.append(
            f"dtypes query={query.dtype} key={key.dtype} value={value.dtype}"
        )
    return "; ".join(reasons) or "unknown Triton availability failure"


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gather_index: torch.Tensor,
    block_token_mask: torch.Tensor,
    heads: int,
    kv_heads: int,
) -> torch.Tensor:
    """Run image-only MHLA through the production Triton implementation.

    The shared implementation supports multiple modalities.  An image-only
    call is represented by one modality for every block and a zero bias
    matrix, so its block routing is exactly the single-stream formulation.
    """
    if not triton_available(query, key, value):
        raise RuntimeError(
            "MHLA Triton bridge requires CUDA, matching supported dtypes, and "
            "contiguous Q/K/V tensors: "
            + triton_unavailable_reason(query, key, value)
        )
    from image_gen.train import joint_mhla_attention

    block_indices = [
        row[mask]
        for row, mask in zip(gather_index, block_token_mask, strict=True)
    ]
    valid_mask = torch.ones(
        query.shape[0], query.shape[2], device=query.device, dtype=torch.bool,
    )
    block_modalities = torch.zeros(
        gather_index.shape[0], device=query.device, dtype=torch.int32,
    )
    modality_bias = torch.zeros(
        3, 3, device=query.device, dtype=query.dtype,
    )
    return joint_mhla_attention(
        query,
        key,
        value,
        block_indices,
        block_modalities,
        valid_mask,
        heads,
        kv_heads,
        modality_bias,
        backend="triton",
        padded_layout=(gather_index, block_token_mask),
    )
