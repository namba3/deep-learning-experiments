"""Trajectory PCA metrics for optimizer convergence experiments."""

from __future__ import annotations

import torch
from verify._text_lm_optimizer_trajectory_pca import (
    _causal_trajectory_pca_metrics as _causal_trajectory_pca_metrics,
    _rolling_trajectory_pca_metrics as _rolling_trajectory_pca_metrics,
    _trajectory_pca_metrics as _trajectory_pca_metrics,
)

def _trajectory_curvature_metrics(
    values: list[torch.Tensor],
    *,
    max_elements: int,
) -> dict[str, object] | None:
    """Measure direction changes and roughness of sampled update vectors.

    The input values are effective parameter updates, not parameter positions.
    Consequently, this reports the curvature of the sampled update trajectory
    and deliberately does not claim to be a parameterization-invariant
    geodesic curvature.
    """
    if len(values) < 3 or values[0].numel() > max_elements:
        return None
    if any(value.numel() != values[0].numel() for value in values):
        return None
    matrix = torch.stack([value.detach().float().reshape(-1) for value in values])
    if not torch.isfinite(matrix).all():
        return None

    norms = torch.linalg.vector_norm(matrix, dim=1)
    differences = matrix[1:] - matrix[:-1]
    epsilon = 1e-30
    normalized_direction_change = torch.linalg.vector_norm(
        differences, dim=1,
    ) / (norms[:-1] + epsilon)
    valid = (norms[:-1] > epsilon) & (norms[1:] > epsilon)
    if not valid.any():
        return {
            "metric_type": "trajectory_curvature",
            "samples": len(values),
            "features": int(matrix.shape[1]),
            "valid_turns": 0,
            "path_length": float(norms.sum()),
            "update_norm_mean": float(norms.mean()),
            "roughness": float(differences.square().sum()),
            "normalized_roughness": float(
                (differences.square().sum(dim=1) / (norms[:-1].square() + epsilon)).mean()
            ),
            "turning_angle_mean": None,
            "turning_angle_p95": None,
            "direction_change_mean": float(normalized_direction_change.mean()),
            "direction_change_variance": float(normalized_direction_change.var(unbiased=False)),
            "curvature_proxy_mean": None,
        }

    dot = (matrix[:-1] * matrix[1:]).sum(dim=1)
    cosine = dot / (norms[:-1] * norms[1:]).clamp_min(epsilon)
    angle = torch.acos(cosine[valid].clamp(-1.0, 1.0))
    sine = torch.sqrt((1.0 - cosine[valid].square()).clamp_min(0.0))
    curvature_proxy = sine / (norms[:-1][valid] + norms[1:][valid] + epsilon)

    return {
        "metric_type": "trajectory_curvature",
        "samples": len(values),
        "features": int(matrix.shape[1]),
        "valid_turns": int(valid.sum()),
        "path_length": float(norms.sum()),
        "update_norm_mean": float(norms.mean()),
        "roughness": float(differences.square().sum()),
        "normalized_roughness": float(
            (differences.square().sum(dim=1) / (norms[:-1].square() + epsilon)).mean()
        ),
        "turning_angle_mean": float(angle.mean()),
        "turning_angle_p95": float(torch.quantile(angle, 0.95)),
        "direction_change_mean": float(normalized_direction_change.mean()),
        "direction_change_variance": float(normalized_direction_change.var(unbiased=False)),
        "curvature_proxy_mean": float(curvature_proxy.mean()),
    }

def _loss_second_difference_metrics(
    values: list[float],
) -> dict[str, object] | None:
    """Measure discrete second differences of a scalar loss sequence.

    The primary sequence is the per-step training loss.  This is a local
    smoothness diagnostic, not a validation metric: with shuffled minibatches
    it also contains minibatch noise and must be compared under the same data
    order and token budget.
    """
    if len(values) < 3:
        return None
    losses = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(losses).all():
        return None
    second_difference = losses[2:] - 2.0 * losses[1:-1] + losses[:-2]
    absolute = second_difference.abs()
    return {
        "metric_type": "loss_second_difference",
        "samples": len(values),
        "second_difference_samples": int(second_difference.numel()),
        "loss_mean": float(losses.mean()),
        "loss_std": float(losses.std(unbiased=False)),
        "second_difference_mean": float(second_difference.mean()),
        "second_difference_abs_mean": float(absolute.mean()),
        "second_difference_variance": float(
            second_difference.var(unbiased=False)
        ),
        "second_difference_abs_p95": float(torch.quantile(absolute, 0.95)),
    }

def _trajectory_position_curvature_metrics(
    values: list[torch.Tensor],
    *,
    max_elements: int,
) -> dict[str, object] | None:
    """Measure curvature of positions by first differencing them in time."""
    if len(values) < 4:
        return None
    if any(value.numel() != values[0].numel() for value in values):
        return None
    steps = [
        values[index + 1].detach().float() - values[index].detach().float()
        for index in range(len(values) - 1)
    ]
    metrics = _trajectory_curvature_metrics(steps, max_elements=max_elements)
    if metrics is None:
        return None
    metrics["metric_type"] = "trajectory_position_curvature"
    metrics["position_samples"] = len(values)
    return metrics

def _rank_analysis_role(source: str) -> str:
    """Classify whether a rank metric is a compression signal or a diagnostic."""
    roles = {
        "parameter": "full_state_baseline",
        "gradient": "instantaneous_signal",
        "exp_avg": "compression_candidate",
        "exp_avg_sq": "compression_candidate",
        "z": "full_state_baseline",
        "sf_delta": "compression_candidate",
        "sf_delta_ema": "trajectory_diagnostic",
        "lrsf_delta_latent": "compressed_state",
        "decoded_lrsf_delta": "projection_constrained_reference",
        "apollo_R_update": "projection_baseline",
        "lrsf_R_delta": "projection_baseline",
        "apollo_exp_avg_latent": "latent_utilization",
        "apollo_exp_avg_sq_latent": "latent_utilization",
        "apollo_low_rank_grad_latent": "latent_utilization",
        "apollo_exp_avg_decoded": "projection_constrained_reference",
        "apollo_exp_avg_sq_decoded": "projection_constrained_reference",
        "apollo_low_rank_grad_decoded": "projection_constrained_reference",
        "lr_ema_grad": "compressed_state",
        "lr_ema_residual_sq": "compressed_state",
        "lr_ema_projection": "projection_baseline",
        "lr_ema_confidence_projection": "projection_baseline",
    }
    return roles.get(source, "state_summary")

def _update_trajectory_before_snapshot(
    model: torch.nn.Module,
    *,
    max_elements: int,
    max_tensors: int,
    parameter_substring: str | None,
) -> dict[str, torch.Tensor]:
    """Copy selected matrix parameters before an optimizer step.

    The copy is deliberately kept outside the timed optimizer region. It is
    used only to recover the realized update ``parameter_after - parameter_before``
    for the temporal-PCA diagnostic.
    """
    snapshot: dict[str, torch.Tensor] = {}
    selected = 0
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or parameter.numel() > max_elements:
            continue
        if parameter_substring is not None and parameter_substring not in name:
            continue
        if selected >= max_tensors:
            break
        selected += 1
        snapshot[name] = parameter.detach().float().cpu().clone()
    return snapshot

def _update_trajectory_snapshot(
    model: torch.nn.Module,
    before: dict[str, torch.Tensor],
) -> list[tuple[str, str, torch.Tensor]]:
    """Return realized parameter updates matching a before-step snapshot."""
    snapshots: list[tuple[str, str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        old = before.get(name)
        if old is None:
            continue
        update = parameter.detach().float().cpu() - old
        snapshots.append((name, "effective_update", update.reshape(-1)))
    return snapshots
