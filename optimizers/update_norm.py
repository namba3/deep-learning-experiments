"""Optional update-norm diagnostics and upper-tail variance limiting."""

from __future__ import annotations

import math

import torch


def _observe_update_norm(
    norm: torch.Tensor,
    state: dict,
    max_variance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update Welford state and return the accepted norm and cap flag."""
    if not math.isfinite(max_variance) or max_variance < 0.0:
        raise ValueError("max update norm variance must be finite and non-negative")

    count = int(state.get("update_norm_variance_count", 0))
    mean = state.get("update_norm_variance_mean")
    m2 = state.get("update_norm_variance_m2")
    if count > 0 and not isinstance(mean, torch.Tensor):
        raise TypeError("update norm variance mean state must be a tensor")
    if count > 0 and not isinstance(m2, torch.Tensor):
        raise TypeError("update norm variance M2 state must be a tensor")

    capped = torch.zeros((), dtype=torch.bool, device=norm.device)
    observed_norm = norm
    if count > 0:
        count_tensor = norm.new_tensor(float(count))
        available = norm.new_tensor(float(max_variance)) * (count_tensor + 1.0)
        available = available - m2
        max_delta = torch.sqrt(
            (available * (count_tensor + 1.0) / count_tensor).clamp_min(0.0)
        )
        max_norm = mean + max_delta
        capped = (
            (available >= 0.0) & (norm > mean) & (norm > max_norm)
        )
        observed_norm = torch.where(capped, max_norm, norm)

    if count == 0:
        state["update_norm_variance_mean"] = observed_norm.detach().clone()
        state["update_norm_variance_m2"] = observed_norm.new_zeros(())
    else:
        delta = observed_norm - mean
        new_count = count + 1
        new_mean = mean + delta / float(new_count)
        state["update_norm_variance_m2"] = (
            m2 + delta * (observed_norm - new_mean)
        ).detach()
        state["update_norm_variance_mean"] = new_mean.detach()
    state["update_norm_variance_count"] = count + 1
    capped_count = state.get("update_norm_variance_capped_count")
    if not isinstance(capped_count, torch.Tensor):
        capped_count = norm.new_zeros((), dtype=torch.int32)
    state["update_norm_variance_capped_count"] = (
        capped_count + capped.to(dtype=capped_count.dtype)
    ).detach()
    return observed_norm, capped


def cap_update_norm_variance_scale(
    update: torch.Tensor,
    state: dict,
    max_variance: float,
    *,
    scale: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a scalar effective-update scale under a variance cap.

    ``update`` is assumed to be multiplied by the scalar ``scale`` at apply
    time.  Keeping that multiplication scalar avoids materializing a full
    effective-update tensor on every capped-optimizer step.
    """
    scale_tensor = (
        scale if isinstance(scale, torch.Tensor) else update.new_tensor(scale)
    )
    if scale_tensor.numel() != 1:
        raise ValueError("update norm variance scale must be scalar")
    norm = torch.linalg.vector_norm(update, dtype=torch.float32)
    effective_norm = norm * scale_tensor.to(dtype=norm.dtype).abs()
    observed_norm, capped = _observe_update_norm(
        effective_norm, state, max_variance,
    )
    ratio = observed_norm / effective_norm.clamp_min(torch.finfo(norm.dtype).tiny)
    adjusted_scale = torch.where(
        capped,
        scale_tensor.to(dtype=norm.dtype) * ratio,
        scale_tensor.to(dtype=norm.dtype),
    )
    return adjusted_scale.to(dtype=update.dtype), capped


def cap_update_norm_variance(
    update: torch.Tensor,
    state: dict,
    max_variance: float,
) -> tuple[torch.Tensor, bool]:
    """Limit a new upper-tail update norm using cumulative Welford statistics.

    The cap applies to the optimizer update before decoupled weight decay.  It
    does not rewrite historical variance or optimizer moments.  If the
    historical variance already exceeds the cap, the current update is left
    unchanged; this policy only prevents a new large-norm sample from making
    the variance worse.
    """
    scale, capped = cap_update_norm_variance_scale(
        update, state, max_variance,
    )
    if bool(capped):
        update = update * scale
    return update, bool(capped)
