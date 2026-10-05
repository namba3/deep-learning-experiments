"""Schedule-Free and optimizer-state reconstruction diagnostics."""

from __future__ import annotations

import torch


def _schedulefree_step_context(
    optimizer: torch.optim.Optimizer,
) -> dict[int, dict[str, float]]:
    """Predict the Schedule-Free scalar coefficients before one step."""
    if any("k" not in group or "weight_sum" not in group
           for group in optimizer.param_groups):
        return {}
    contexts: dict[int, dict[str, float]] = {}
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        k = group["k"]
        warmup_steps = group["warmup_steps"]
        sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
        lr = group["lr"] * sched
        lr_max = max(lr, group["lr_max"])
        weight = ((k + 1) ** group["r"]) * (
            lr_max ** group["weight_lr_power"]
        )
        weight_sum = group["weight_sum"] + weight
        ckp1 = weight / weight_sum if weight_sum else 0.0
        for parameter in group["params"]:
            contexts[id(parameter)] = {
                "update_scale": lr * (beta1 * (1.0 - ckp1) - 1.0),
                "carry_scale": ckp1,
                "bias_correction2": 1.0 - beta2 ** (k + 1),
                "eps": group["eps"],
            }
    return contexts

def _update_reconstruction_before_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_elements: int,
    max_tensors: int,
    parameter_substring: str | None,
) -> dict[str, dict[str, object]]:
    """Capture inputs needed for diagnostic low-rank update reconstruction."""
    contexts = _schedulefree_step_context(optimizer)
    snapshot: dict[str, dict[str, object]] = {}
    selected = 0
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or parameter.numel() > max_elements:
            continue
        if parameter_substring is not None and parameter_substring not in name:
            continue
        if selected >= max_tensors:
            break
        selected += 1
        state = optimizer.state.get(parameter, {})
        z = state.get("z")
        delta = None
        if torch.is_tensor(z) and z.shape == parameter.shape:
            delta = z.detach().float().cpu() - parameter.detach().float().cpu()
        snapshot[name] = {
            "parameter": parameter.detach().float().cpu().clone(),
            "gradient": (
                None
                if parameter.grad is None
                else parameter.grad.detach().float().cpu().clone()
            ),
            "sf_delta": delta,
            "context": contexts.get(id(parameter)),
        }
    return snapshot

def _truncated_svd_reconstructions(
    tensor: torch.Tensor,
    ranks: tuple[int, ...],
    *,
    max_elements: int,
) -> list[tuple[int, torch.Tensor, float]]:
    """Return spatial low-rank reconstructions and relative errors."""
    if tensor.ndim < 2 or tensor.numel() > max_elements:
        return []
    matrix = tensor.detach().float().cpu().reshape(tensor.shape[0], -1)
    if min(matrix.shape) < 2 or not torch.isfinite(matrix).all():
        return []
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    denominator = matrix.norm().clamp_min(1e-30)
    result = []
    for rank in ranks:
        effective_rank = min(rank, singular_values.numel())
        approximation = (
            (u[:, :effective_rank] * singular_values[:effective_rank])
            @ vh[:effective_rank, :]
        ).reshape(tensor.shape)
        error = float((matrix - approximation.reshape(matrix.shape)).norm() / denominator)
        result.append((rank, approximation, error))
    return result

def _update_reconstruction_metrics(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    before: dict[str, dict[str, object]],
    sf_delta_ema: dict[str, torch.Tensor],
    *,
    ranks: tuple[int, ...],
    max_elements: int,
    use_adamw_schedulefree_formula: bool,
) -> list[dict[str, object]]:
    """Measure how low-rank state replacements change the realized update."""
    if not use_adamw_schedulefree_formula:
        return []
    metrics: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        record = before.get(name)
        if record is None:
            continue
        old_parameter = record["parameter"]
        actual_update = parameter.detach().float().cpu() - old_parameter
        actual_flat = actual_update.reshape(-1)
        actual_norm = actual_flat.norm().clamp_min(1e-30)
        gradient = record["gradient"]
        context = record["context"]
        if not torch.is_tensor(gradient) or not isinstance(context, dict):
            continue
        state = optimizer.state.get(parameter, {})
        exp_avg_sq = state.get("exp_avg_sq")
        if torch.is_tensor(exp_avg_sq) and exp_avg_sq.shape == parameter.shape:
            bias_correction2 = context["bias_correction2"]
            eps = context["eps"]
            preconditioner = exp_avg_sq.detach().float().cpu().div(
                bias_correction2,
            ).clamp_min(0.0).sqrt()
            full_denom = preconditioner + eps
            full_normalized = gradient.div(full_denom)
            state_reconstructions = {
                rank: (approximation, state_error)
                for rank, approximation, state_error in _truncated_svd_reconstructions(
                    exp_avg_sq.detach().float().cpu(), ranks,
                    max_elements=max_elements,
                )
            }
            log_preconditioner = preconditioner.clamp_min(1e-12).log()
            for rank, log_approximation, _ in _truncated_svd_reconstructions(
                log_preconditioner, ranks, max_elements=max_elements,
            ):
                _, state_error = state_reconstructions[rank]
                approximation = log_approximation.exp()
                preconditioner_error = float(
                    (preconditioner - approximation).norm()
                    / preconditioner.norm().clamp_min(1e-30)
                )
                approx_denom = approximation.add(eps)
                approx_normalized = gradient.div(approx_denom)
                approximate_update = actual_update + context["update_scale"] * (
                    approx_normalized - full_normalized
                )
                approx_flat = approximate_update.reshape(-1)
                metrics.append({
                    "parameter": name,
                    "source": "exp_avg_sq",
                    "rank": rank,
                    "state_relative_error": state_error,
                    "preconditioner_relative_error": preconditioner_error,
                    "update_relative_error": float(
                        (approx_flat - actual_flat).norm() / actual_norm
                    ),
                    "update_cosine": float(
                        torch.dot(approx_flat, actual_flat)
                        / (approx_flat.norm().clamp_min(1e-30) * actual_norm)
                    ),
                    "update_norm_ratio": float(
                        approx_flat.norm() / actual_norm
                    ),
                })
        for source, candidate in (
            ("sf_delta", record["sf_delta"]),
            ("sf_delta_ema", sf_delta_ema.get(name)),
        ):
            if not torch.is_tensor(candidate):
                continue
            actual_delta = record["sf_delta"]
            if not torch.is_tensor(actual_delta):
                continue
            for rank, approximation, state_error in _truncated_svd_reconstructions(
                candidate, ranks, max_elements=max_elements,
            ):
                approximate_update = actual_update + context["carry_scale"] * (
                    approximation - actual_delta
                )
                approx_flat = approximate_update.reshape(-1)
                metrics.append({
                    "parameter": name,
                    "source": source,
                    "rank": rank,
                    "state_relative_error": state_error,
                    "update_relative_error": float(
                        (approx_flat - actual_flat).norm() / actual_norm
                    ),
                    "update_cosine": float(
                        torch.dot(approx_flat, actual_flat)
                        / (approx_flat.norm().clamp_min(1e-30) * actual_norm)
                    ),
                    "update_norm_ratio": float(
                        approx_flat.norm() / actual_norm
                    ),
                })
    return metrics

def _update_sf_delta_ema(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    previous: dict[str, torch.Tensor],
    *,
    max_elements: int,
    max_tensors: int,
    parameter_substring: str | None,
    decay: float,
) -> dict[str, torch.Tensor]:
    """Update diagnostic-only FP32 EMA tensors for full Schedule-Free drift."""
    selected = 0
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or parameter.numel() > max_elements:
            continue
        if parameter_substring is not None and parameter_substring not in name:
            continue
        if selected >= max_tensors:
            break
        state = optimizer.state.get(parameter, {})
        z = state.get("z")
        if not torch.is_tensor(z):
            continue
        selected += 1
        delta = z.detach().float() - parameter.detach().float()
        if name not in previous:
            previous[name] = delta
        else:
            previous[name].mul_(decay).add_(delta, alpha=1.0 - decay)
    return previous
