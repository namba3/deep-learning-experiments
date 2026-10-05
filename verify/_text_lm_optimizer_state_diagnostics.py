"""Optimizer state and confidence diagnostics for the Text-LM probe."""

from __future__ import annotations

import math

import torch
from verify._text_lm_optimizer_residuals import _matrix_rank_metrics
from verify._text_lm_optimizer_trajectory_diagnostics import (
    _record_update_trajectory_snapshots as _record_update_trajectory_snapshots,
    _schedulefree_beta as _schedulefree_beta,
    _schedulefree_hidden_state as _schedulefree_hidden_state,
    _schedulefree_refresh_policy as _schedulefree_refresh_policy,
    _schedulefree_trajectory_gap_snapshot as _schedulefree_trajectory_gap_snapshot,
    _schedulefree_trajectory_snapshot as _schedulefree_trajectory_snapshot,
    _state_trajectory_snapshot as _state_trajectory_snapshot,
)
from verify._text_lm_optimizer_trajectory import (
    _rank_analysis_role,
    _update_trajectory_before_snapshot,
)
from verify._text_lm_optimizer_state_reconstruction import (
    _update_reconstruction_before_snapshot,
    _update_reconstruction_metrics,
    _update_sf_delta_ema,
)

def _capture_pre_step_diagnostics(
    args,
    model,
    optimizer: torch.optim.Optimizer,
    update_trajectory_shapes: dict[str, tuple[int, ...]],
) -> tuple[
    dict[str, torch.Tensor] | None,
    dict[str, dict[str, object]] | None,
]:
    update_before = None
    if args.record_update_trajectory_pca or args.record_trajectory_curvature:
        update_before = _update_trajectory_before_snapshot(
            model,
            max_elements=args.state_rank_max_elements,
            max_tensors=args.state_rank_max_tensors,
            parameter_substring=args.state_rank_parameter,
        )
        for parameter_name, value in update_before.items():
            update_trajectory_shapes.setdefault(parameter_name, tuple(value.shape))

    reconstruction_before = None
    if args.record_update_reconstruction:
        reconstruction_before = _update_reconstruction_before_snapshot(
            model,
            optimizer,
            max_elements=args.state_rank_max_elements,
            max_tensors=args.state_rank_max_tensors,
            parameter_substring=args.state_rank_parameter,
        )
    return update_before, reconstruction_before

def _confidence_diagnostic_snapshot(
    optimizer: torch.optim.Optimizer,
) -> dict[str, object] | None:
    """Summarize confidence normalization without copying latent state.

    The reported confidence is the bounded signal-to-signal-plus-innovation
    ratio ``m_hat^2 / (m_hat^2 + c_hat)``.  ``normalized_update_rms`` uses the
    actual denominator floor from the optimizer and is useful for detecting
    an overly aggressive ``alpha`` setting.  Scalar extraction is deliberately
    outside the timed optimizer step in :func:`_run_case`.
    """
    confidence_sum = 0.0
    confidence_square_sum = 0.0
    normalized_rms_sum = 0.0
    innovation_rms_sum = 0.0
    elements = 0
    states = 0
    last_step = 0
    for group in optimizer.param_groups:
        default_confidence_beta = float(
            group.get("lr_ema_confidence_beta", group.get(
                "apollo_confidence_beta", 0.99,
            ))
        )
        default_alpha = float(
            group.get("lr_ema_confidence_alpha", group.get(
                "apollo_confidence_alpha", 1e-3,
            ))
        )
        eps = float(group.get("lr_ema_eps", group.get("eps", 1e-8)))
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            has_confidence_state = (
                state.get("backend") == "lr_ema_confidence"
                or torch.is_tensor(state.get("lr_ema_confidence_projection"))
                or bool(state.get("apollo_confidence", False))
            )
            if not has_confidence_state:
                continue
            is_apollo_confidence = bool(state.get("apollo_confidence", False))
            mean_state = state.get(
                "exp_avg" if is_apollo_confidence else "lr_ema_grad"
            )
            variance_state = state.get(
                "exp_avg_sq" if is_apollo_confidence else "lr_ema_residual_sq"
            )
            if (
                not torch.is_tensor(mean_state)
                or not torch.is_tensor(variance_state)
            ):
                continue
            # The integrated confidence+LRSF prototype uses a separate
            # counter because its inherited Schedule-Free state does not
            # own the standalone LR-EMA ``step`` key.
            step = int(state.get("step", state.get("lr_ema_step", 0)))
            if step <= 0:
                continue
            confidence_beta = float(
                group.get(
                    "apollo_confidence_beta" if is_apollo_confidence
                    else "lr_ema_confidence_beta",
                    default_confidence_beta,
                )
            )
            alpha = float(
                group.get(
                    "apollo_confidence_alpha" if is_apollo_confidence
                    else "lr_ema_confidence_alpha",
                    default_alpha,
                )
            )
            mean_beta = float(group["betas"][0]) if is_apollo_confidence else float(
                group["lr_ema_beta"]
            )
            mean = mean_state.detach().float().div(
                1.0 - mean_beta**step
            )
            variance = variance_state.detach().float().div(
                1.0 - confidence_beta**step
            ).clamp_min(0.0)
            signal_square = mean.square()
            confidence = signal_square / (
                signal_square + variance + eps
            )
            normalized = mean / (
                variance.addcmul(mean, mean, value=alpha).sqrt_().add_(eps)
            )
            count = int(mean.numel())
            elements += count
            states += 1
            last_step = max(last_step, step)
            confidence_sum += float(confidence.sum())
            confidence_square_sum += float(confidence.square().sum())
            normalized_rms_sum += float(normalized.square().mean().sqrt())
            innovation_rms_sum += float(variance.mean().sqrt())
    if elements == 0 or states == 0:
        return None
    confidence_mean = confidence_sum / elements
    confidence_variance = max(
        0.0, confidence_square_sum / elements - confidence_mean**2,
    )
    return {
        "metric_type": "confidence_normalization",
        "step": last_step,
        "states": states,
        "elements": elements,
        "confidence_mean": confidence_mean,
        "confidence_std": math.sqrt(confidence_variance),
        "normalized_update_rms_mean": normalized_rms_sum / states,
        "innovation_rms_mean": innovation_rms_sum / states,
    }

def _decode_projected_matrix(
    latent: torch.Tensor,
    projection: torch.Tensor,
    matrix_shape: tuple[int, int],
) -> torch.Tensor | None:
    """Decode one APOLLO/LRSF latent matrix to the parameter matrix shape."""
    rows, cols = matrix_shape
    latent = latent.detach().float()
    projection = projection.detach().float()
    if rows >= cols:
        expected = (rows, projection.shape[1])
        if tuple(latent.shape) != expected or projection.shape[0] != cols:
            return None
        return latent.matmul(projection.transpose(0, 1))
    expected = (projection.shape[0], cols)
    if tuple(latent.shape) != expected or projection.shape[1] != rows:
        return None
    return projection.transpose(0, 1).matmul(latent)

def _projection_geometry(
    projection: torch.Tensor,
    matrix_shape: tuple[int, int],
) -> dict[str, object]:
    """Summarize basis geometry without treating random APOLLO bases as orthogonal."""
    basis = projection.detach().float()
    rows, cols = matrix_shape
    gram = (
        basis.transpose(0, 1).matmul(basis)
        if rows >= cols
        else basis.matmul(basis.transpose(0, 1))
    )
    diagonal = torch.diagonal(gram)
    off_diagonal = gram - torch.diag_embed(diagonal)
    off_diagonal_count = max(1, gram.numel() - diagonal.numel())
    normalized_off_diagonal = off_diagonal.square().sum().sqrt() / math.sqrt(
        off_diagonal_count
    )
    return {
        "projection_shape": list(basis.shape),
        "projection_dtype": str(projection.dtype),
        "projection_frobenius_norm": float(basis.norm()),
        "projection_gram_diagonal_mean": float(diagonal.mean()),
        "projection_gram_diagonal_std": float(diagonal.std(unbiased=False)),
        "projection_gram_off_diagonal_rms": float(normalized_off_diagonal),
    }

def _state_summary(tensor: torch.Tensor) -> dict[str, object]:
    """Summarize vector/factor states that have no spatial SVD interpretation."""
    value = tensor.detach().float()
    return {
        "metric_type": "summary",
        "shape": list(value.shape),
        "numel": int(value.numel()),
        "dtype": str(tensor.dtype),
        "rms": float(value.square().mean().sqrt()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
    }

def _state_rank_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_elements: int,
    max_tensors: int,
    parameter_substring: str | None,
    sf_delta_ema: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, object]]:
    """Measure full and low-rank-relevant optimizer tensors at one step."""
    snapshots: list[dict[str, object]] = []
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
        matrix_shape = (parameter.shape[0], parameter.numel() // parameter.shape[0])
        apollo_projection = state.get("projection")
        is_apollo_matrix = (
            parameter.ndim >= 2
            and state.get("backend") == "apollo"
            and torch.is_tensor(apollo_projection)
        )
        tensors: list[tuple[str, torch.Tensor]] = [("parameter", parameter)]
        if parameter.grad is not None:
            tensors.append(("gradient", parameter.grad))
        if is_apollo_matrix:
            tensors.extend(
                (source, state[key])
                for key, source in (
                    ("exp_avg", "apollo_exp_avg_latent"),
                    ("exp_avg_sq", "apollo_exp_avg_sq_latent"),
                    ("came_low_rank_grad", "apollo_low_rank_grad_latent"),
                )
                if torch.is_tensor(state.get(key))
            )
            for key, source in (
                ("exp_avg_sq_row", "apollo_came_exp_avg_sq_row"),
                ("exp_avg_sq_col", "apollo_came_exp_avg_sq_col"),
                ("exp_avg_res_row", "apollo_came_exp_avg_res_row"),
                ("exp_avg_res_col", "apollo_came_exp_avg_res_col"),
            ):
                value = state.get(key)
                if torch.is_tensor(value):
                    summary = _state_summary(value)
                    snapshots.append({
                        "parameter": name,
                        "source": source,
                        "analysis_role": _rank_analysis_role(source),
                        **summary,
                    })
        elif (
            state.get("backend") in {"lr_ema", "lr_ema_confidence"}
            or torch.is_tensor(state.get("lr_ema_grad"))
        ):
            for key in ("lr_ema_grad", "lr_ema_residual_sq"):
                value = state.get(key)
                if torch.is_tensor(value):
                    tensors.append((key, value))
            projection_key = (
                "lr_ema_projection"
                if torch.is_tensor(state.get("lr_ema_projection"))
                else "lr_ema_confidence_projection"
            )
            projection = state.get(projection_key)
            if torch.is_tensor(projection):
                projection_metrics = _projection_geometry(
                    projection, matrix_shape,
                )
                projection_rank = _matrix_rank_metrics(
                    projection, max_elements=max_elements,
                )
                if projection_rank is not None:
                    snapshots.append({
                        "parameter": name,
                        "source": projection_key,
                        "analysis_role": _rank_analysis_role(
                            projection_key
                        ),
                        **projection_metrics,
                        **projection_rank,
                    })
        else:
            for key in ("exp_avg", "exp_avg_sq", "z"):
                value = state.get(key)
                if torch.is_tensor(value):
                    tensors.append((key, value))
        if torch.is_tensor(state.get("z")):
            tensors.append((
                "sf_delta",
                state["z"].detach().float() - parameter.detach().float(),
            ))
        if sf_delta_ema is not None and name in sf_delta_ema:
            tensors.append(("sf_delta_ema", sf_delta_ema[name]))
        if is_apollo_matrix:
            projection = apollo_projection
            projection_metrics = _projection_geometry(projection, matrix_shape)
            projection_rank = _matrix_rank_metrics(
                projection, max_elements=max_elements,
            )
            if projection_rank is not None:
                snapshots.append({
                    "parameter": name,
                    "source": "apollo_R_update",
                    "analysis_role": _rank_analysis_role("apollo_R_update"),
                    **projection_metrics,
                    **projection_rank,
                })
            for key, source, decoded_source in (
                ("exp_avg", "apollo_exp_avg_latent", "apollo_exp_avg_decoded"),
                (
                    "exp_avg_sq",
                    "apollo_exp_avg_sq_latent",
                    "apollo_exp_avg_sq_decoded",
                ),
                (
                    "came_low_rank_grad",
                    "apollo_low_rank_grad_latent",
                    "apollo_low_rank_grad_decoded",
                ),
            ):
                value = state.get(key)
                if not torch.is_tensor(value):
                    continue
                decoded = _decode_projected_matrix(
                    value, projection, matrix_shape,
                )
                if decoded is not None:
                    tensors.append((decoded_source, decoded))
            scaled_grad_norm = state.get("scaled_grad_norm")
            if torch.is_tensor(scaled_grad_norm):
                snapshots.append({
                    "parameter": name,
                    "source": "apollo_scaled_grad_norm",
                    "analysis_role": _rank_analysis_role(
                        "apollo_scaled_grad_norm"
                    ),
                    **_state_summary(scaled_grad_norm),
                })
        if torch.is_tensor(state.get("lrsf_delta")):
            delta = state["lrsf_delta"]
            projection = state.get("lrsf_projection")
            tensors.append(("lrsf_delta_latent", delta))
            if torch.is_tensor(projection):
                decoded = _decode_projected_matrix(
                    delta, projection, matrix_shape,
                )
                projection_metrics = _projection_geometry(
                    projection, matrix_shape,
                )
                projection_rank = _matrix_rank_metrics(
                    projection, max_elements=max_elements,
                )
                if projection_rank is not None:
                    snapshots.append({
                        "parameter": name,
                        "source": "lrsf_R_delta",
                        "analysis_role": _rank_analysis_role("lrsf_R_delta"),
                        **projection_metrics,
                        **projection_rank,
                    })
                if decoded is not None:
                    tensors.append(("decoded_lrsf_delta", decoded))
        for source, tensor in tensors:
            metrics = _matrix_rank_metrics(tensor, max_elements=max_elements)
            if metrics is None:
                continue
            snapshots.append({
                "parameter": name,
                "source": source,
                "analysis_role": _rank_analysis_role(source),
                **metrics,
            })
    return snapshots

def _record_state_diagnostics(
    args,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    loss: torch.Tensor,
    sf_delta_ema: dict[str, torch.Tensor],
    state_rank_history: list[dict[str, object]],
    state_trajectory_values: dict[tuple[str, str], list[torch.Tensor]],
    reconstruction_before: dict[str, object] | None,
    update_reconstruction_values: list[dict[str, object]],
    confidence_diagnostics: list[dict[str, object]],
) -> None:
    record_state_diagnostics = (
        args.record_state_rank
        or args.record_state_trajectory_pca
        or args.record_update_reconstruction
    )
    if record_state_diagnostics:
        _update_sf_delta_ema(
            model,
            optimizer,
            sf_delta_ema,
            max_elements=args.state_rank_max_elements,
            max_tensors=args.state_rank_max_tensors,
            parameter_substring=args.state_rank_parameter,
            decay=args.state_rank_delta_ema_decay,
        )
    if (
        record_state_diagnostics
        and total_steps % args.state_rank_interval == 0
    ):
        if args.record_state_rank:
            state_rank_history.append({
                "step": total_steps,
                "tensors": _state_rank_snapshot(
                    model,
                    optimizer,
                    max_elements=args.state_rank_max_elements,
                    max_tensors=args.state_rank_max_tensors,
                    parameter_substring=args.state_rank_parameter,
                    sf_delta_ema=sf_delta_ema,
                ),
            })
        if args.record_state_trajectory_pca:
            for parameter_name, source, value in _state_trajectory_snapshot(
                model,
                optimizer,
                max_elements=args.state_rank_max_elements,
                max_tensors=args.state_rank_max_tensors,
                parameter_substring=args.state_rank_parameter,
                sf_delta_ema=sf_delta_ema,
            ):
                key = (parameter_name, source)
                samples = state_trajectory_values.setdefault(key, [])
                if len(samples) < args.state_trajectory_max_snapshots:
                    samples.append(value)
        if reconstruction_before is not None:
            update_reconstruction_values.extend(
                _update_reconstruction_metrics(
                    model,
                    optimizer,
                    reconstruction_before,
                    sf_delta_ema,
                    ranks=args.update_reconstruction_ranks,
                    max_elements=args.state_rank_max_elements,
                    use_adamw_schedulefree_formula=name in (
                        "AdamW-SF", "AdamW-LRSF",
                    ),
                )
            )
    if (
        args.record_confidence_diagnostics
        and total_steps % args.state_rank_interval == 0
    ):
        confidence_snapshot = _confidence_diagnostic_snapshot(optimizer)
        if confidence_snapshot is not None:
            confidence_diagnostics.append(confidence_snapshot)

def _lrsf_latent_moment_snapshot(
    optimizer: torch.optim.Optimizer,
    step: int,
    previous_refresh_counts: dict[int, int],
) -> dict[str, object] | None:
    """Summarize integrated LRSF latent second moments for one step.

    This deliberately reduces only scalar diagnostics. It does not decode a
    full matrix and is called only when refresh diagnostics are requested.
    ``moment_step`` makes reset visible: after a hard refresh it starts again
    at one, while transport continues the previous age.
    """
    moment_squared_sum = 0.0
    moment_elements = 0
    moment_steps: list[int] = []
    refresh_counts: list[int] = []
    parameter_count = 0
    refresh_event = False
    for state in optimizer.state.values():
        moment = state.get("lrsf_exp_avg_sq")
        if not torch.is_tensor(moment):
            continue
        parameter_count += 1
        moment_squared_sum += float(moment.float().square().sum())
        moment_elements += moment.numel()
        moment_steps.append(int(state.get("lrsf_moment_step", 0)))
        refresh_count = int(state.get("refresh_count", 0))
        refresh_counts.append(refresh_count)
        state_key = id(state)
        if refresh_count > previous_refresh_counts.get(state_key, 0):
            refresh_event = True
        previous_refresh_counts[state_key] = refresh_count
    if not parameter_count:
        return None
    moment_norm = math.sqrt(max(moment_squared_sum, 0.0))
    return {
        "step": int(step),
        "parameter_count": parameter_count,
        "latent_elements": moment_elements,
        "latent_moment_norm": moment_norm,
        "latent_moment_rms": (
            moment_norm / math.sqrt(moment_elements)
            if moment_elements
            else 0.0
        ),
        "moment_step_min": min(moment_steps),
        "moment_step_max": max(moment_steps),
        "refresh_count_max": max(refresh_counts),
        "refresh_event": refresh_event,
    }
