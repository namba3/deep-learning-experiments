"""Classifier-free guidance transforms, independent of sampling solvers."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

GuidanceMethod = Literal["cfg", "tcfg"]


def available_guidance_methods() -> tuple[str, ...]:
    """Return the guidance transforms implemented by this module."""
    return ("cfg", "tcfg")


def _validate_pair(unconditional: Tensor, conditional: Tensor, scale: float) -> None:
    if unconditional.ndim < 2:
        raise ValueError("guidance predictions must have a batch axis and data axes")
    if unconditional.shape[0] <= 0:
        raise ValueError("guidance predictions must contain at least one sample")
    if unconditional.shape != conditional.shape:
        raise ValueError(
            "conditional and unconditional predictions must have the same shape; "
            f"got {tuple(conditional.shape)} and {tuple(unconditional.shape)}"
        )
    if unconditional.device != conditional.device:
        raise ValueError("conditional and unconditional predictions must share a device")
    if unconditional.dtype != conditional.dtype:
        raise ValueError("conditional and unconditional predictions must share a dtype")
    if not unconditional.is_floating_point():
        raise ValueError("guidance predictions must be floating point")
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("guidance scale must be finite and non-negative")


def classifier_free_guidance(
    unconditional: Tensor,
    conditional: Tensor,
    scale: float,
) -> Tensor:
    """Apply the standard velocity/score CFG interpolation or extrapolation."""
    _validate_pair(unconditional, conditional, scale)
    return unconditional + scale * (conditional - unconditional)


def tangential_damping_cfg(
    unconditional: Tensor,
    conditional: Tensor,
    scale: float,
) -> Tensor:
    """Apply TCFG by retaining the leading joint singular direction of CFG.

    The two predictions are flattened per sample into the rows of a 2 x D
    matrix. Reduced SVD gives right singular directions in feature space. The
    unconditional prediction is projected onto the leading direction before
    the ordinary CFG combination. SVD is computed independently for each batch
    item in FP32 (FP64 for FP64 inputs); the result is returned in input dtype.

    This transform operates on model outputs directly. For flow-matching
    models those are velocities, as in the paper's rectified-flow application.
    """
    _validate_pair(unconditional, conditional, scale)
    batch_size = unconditional.shape[0]
    feature_count = unconditional[0].numel()
    if feature_count == 0:
        raise ValueError("guidance predictions cannot have empty data dimensions")

    accumulation_dtype = (
        torch.float64 if unconditional.dtype == torch.float64 else torch.float32
    )
    unconditional_flat = unconditional.reshape(batch_size, feature_count).to(
        dtype=accumulation_dtype,
    )
    conditional_flat = conditional.reshape(batch_size, feature_count).to(
        dtype=accumulation_dtype,
    )
    # Row order is conditional, unconditional, matching the paper's reference
    # implementation. Each batch item has its own 2 x D score matrix.
    score_matrix = torch.stack((conditional_flat, unconditional_flat), dim=1)
    _, _, right_singular_vectors = torch.linalg.svd(
        score_matrix, full_matrices=False,
    )
    leading_direction = right_singular_vectors[:, 0, :]
    projection_coefficient = (
        unconditional_flat * leading_direction
    ).sum(dim=-1, keepdim=True)
    projected_unconditional = projection_coefficient * leading_direction
    guided = projected_unconditional + scale * (
        conditional_flat - projected_unconditional
    )
    return guided.reshape_as(unconditional).to(dtype=unconditional.dtype)


def apply_guidance(
    unconditional: Tensor,
    conditional: Tensor,
    scale: float,
    *,
    method: GuidanceMethod = "cfg",
) -> Tensor:
    """Apply a named guidance transform without coupling it to a solver."""
    if method == "cfg":
        return classifier_free_guidance(unconditional, conditional, scale)
    if method == "tcfg":
        return tangential_damping_cfg(unconditional, conditional, scale)
    raise ValueError(f"unknown guidance method {method!r}; choose from {available_guidance_methods()}")
