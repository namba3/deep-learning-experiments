"""State and Schedule-Free trajectory snapshots for the Text-LM probe."""

from __future__ import annotations

import torch
from optimizers.factory import is_schedule_free_optimizer
from optimizers.projection_refresh import (
    ProjectionRefreshPolicy,
    add_mixed_delta,
)
from verify._text_lm_optimizer_trajectory import _update_trajectory_snapshot


def _state_trajectory_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_elements: int,
    max_tensors: int,
    parameter_substring: str | None,
    sf_delta_ema: dict[str, torch.Tensor] | None = None,
) -> list[tuple[str, str, torch.Tensor]]:
    """Collect full-rank candidate states for temporal PCA.

    APOLLO matrix states are intentionally excluded because their latent
    tensors are already constrained by the configured projection rank.
    """
    snapshots: list[tuple[str, str, torch.Tensor]] = []
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
        if state.get("backend") == "apollo":
            continue
        candidates: list[tuple[str, torch.Tensor]] = []
        if parameter.grad is not None:
            candidates.append(("gradient", parameter.grad))
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == parameter.shape:
                candidates.append((key, value))
        z = state.get("z")
        if torch.is_tensor(z) and z.shape == parameter.shape:
            candidates.append(("sf_delta", z.detach().float() - parameter.detach().float()))
        if sf_delta_ema is not None and name in sf_delta_ema:
            candidates.append(("sf_delta_ema", sf_delta_ema[name]))
        for source, value in candidates:
            snapshots.append((name, source, value.detach().float().reshape(-1).cpu()))
    return snapshots


def _schedulefree_beta(group: dict) -> float | None:
    """Return the Schedule-Free interpolation beta for one parameter group."""
    value = group.get("sf_beta1")
    if value is not None:
        return float(value)
    betas = group.get("betas")
    if isinstance(betas, (tuple, list)) and betas:
        return float(betas[0])
    return None


def _schedulefree_refresh_policy(group: dict) -> ProjectionRefreshPolicy:
    """Resolve the refresh key used by AdamW-, CAME-, and APOLLO-LRSF."""
    value = group.get("projection_refresh")
    if value is None:
        value = group.get("delta_refresh")
    return ProjectionRefreshPolicy.from_value(value)


def _schedulefree_hidden_state(
    parameter: torch.Tensor,
    state: dict,
    group: dict,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor | None:
    """Reconstruct the hidden Schedule-Free state without mutating optimizer state.

    Full Schedule-Free variants store either ``z`` or a full ``sf_delta``.
    LRSF variants store ``h = hidden_state - train_parameter`` in a projected
    coefficient tensor.  The latter is decoded using the same active/next
    refresh mixture as the optimizer's train/eval transition.
    """
    train_value = parameter.detach().float()
    z = state.get("z")
    if torch.is_tensor(z) and z.shape == parameter.shape:
        return z.detach().float().to(device=train_value.device).clone()

    full_delta = state.get("sf_delta")
    if torch.is_tensor(full_delta) and full_delta.shape == parameter.shape:
        return train_value + full_delta.detach().float().to(
            device=train_value.device,
        )

    projection = state.get("lrsf_projection")
    delta = state.get("lrsf_delta")
    add_low_rank = getattr(optimizer, "_add_low_rank", None)
    if (
        parameter.ndim < 2
        or not torch.is_tensor(projection)
        or not torch.is_tensor(delta)
        or not callable(add_low_rank)
    ):
        return None
    if projection.device != train_value.device or delta.device != train_value.device:
        return None
    decoded_delta = torch.zeros_like(train_value)
    add_mixed_delta(
        decoded_delta.reshape(decoded_delta.shape[0], -1),
        state,
        1.0,
        add_low_rank,
        _schedulefree_refresh_policy(group),
    )
    return train_value + decoded_delta


def _schedulefree_trajectory_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_elements: int,
    max_tensors: int,
    parameter_substring: str | None,
) -> list[tuple[str, str, torch.Tensor]]:
    """Capture train/eval/hidden positions for Schedule-Free trajectories.

    ``train_parameter`` is the parameter visible while ``optimizer.train()``
    is active (the local implementations call it ``y``).  ``hidden_state`` is
    the full or decoded hidden state, and ``eval_parameter`` is the parameter
    that the optimizer would expose after its non-mutating eval transform.
    All returned values are CPU FP32 diagnostic copies.
    """
    groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    snapshots: list[tuple[str, str, torch.Tensor]] = []
    selected = 0
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or parameter.numel() > max_elements:
            continue
        if parameter_substring is not None and parameter_substring not in name:
            continue
        if selected >= max_tensors:
            break
        selected += 1
        group = groups.get(id(parameter))
        if group is None:
            continue
        beta = _schedulefree_beta(group)
        if beta is None or not 0.0 < beta < 1.0:
            continue
        state = optimizer.state.get(parameter, {})
        train_value = parameter.detach().float().clone()
        hidden_value = _schedulefree_hidden_state(
            parameter, state, group, optimizer,
        )
        if hidden_value is None or hidden_value.shape != parameter.shape:
            continue
        eval_value = train_value + (hidden_value - train_value) * (
            1.0 - 1.0 / beta
        )
        snapshots.extend((
            (name, "train_parameter", train_value.reshape(-1).cpu()),
            (name, "hidden_state", hidden_value.reshape(-1).cpu()),
            (name, "eval_parameter", eval_value.reshape(-1).cpu()),
        ))
    return snapshots


def _schedulefree_trajectory_gap_snapshot(
    snapshots: list[tuple[str, str, torch.Tensor]],
    *,
    step: int,
    training_loss: float,
) -> list[dict[str, object]]:
    """Summarize train/eval gaps from one Schedule-Free position snapshot."""
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for parameter_name, source, value in snapshots:
        grouped.setdefault(parameter_name, {})[source] = value

    records: list[dict[str, object]] = []
    for parameter_name, values in grouped.items():
        train = values.get("train_parameter")
        hidden = values.get("hidden_state")
        evaluation = values.get("eval_parameter")
        if train is None or hidden is None or evaluation is None:
            continue
        hidden_train_gap = hidden - train
        eval_hidden_gap = evaluation - hidden
        eval_train_gap = evaluation - train
        hidden_train_norm = float(hidden_train_gap.norm())
        eval_hidden_norm = float(eval_hidden_gap.norm())
        eval_train_norm = float(eval_train_gap.norm())
        records.append({
            "step": step,
            "parameter": parameter_name,
            "training_loss": training_loss,
            "hidden_train_gap_norm": hidden_train_norm,
            "eval_hidden_gap_norm": eval_hidden_norm,
            "eval_train_gap_norm": eval_train_norm,
            "eval_hidden_gap_ratio": eval_hidden_norm / max(
                hidden_train_norm, 1e-30,
            ),
        })
    return records


def _record_update_trajectory_snapshots(
    args,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    loss: torch.Tensor,
    update_before: dict[str, torch.Tensor] | None,
    update_trajectory_values: dict[tuple[str, str], list[torch.Tensor]],
    schedulefree_trajectory_values: dict[tuple[str, str], list[torch.Tensor]],
    schedulefree_gap_history: list[dict[str, object]],
) -> None:
    if (
        update_before is not None
        and total_steps % args.state_rank_interval == 0
    ):
        for parameter_name, source, value in _update_trajectory_snapshot(
            model, update_before,
        ):
            key = (parameter_name, source)
            samples = update_trajectory_values.setdefault(key, [])
            if len(samples) < args.state_trajectory_max_snapshots:
                samples.append(value)
        if is_schedule_free_optimizer(name):
            schedulefree_snapshots = _schedulefree_trajectory_snapshot(
                model,
                optimizer,
                max_elements=args.state_rank_max_elements,
                max_tensors=args.state_rank_max_tensors,
                parameter_substring=args.state_rank_parameter,
            )
            schedulefree_gap_history.extend(
                _schedulefree_trajectory_gap_snapshot(
                    schedulefree_snapshots,
                    step=total_steps,
                    training_loss=float(loss.detach()),
                )
            )
            for parameter_name, source, value in schedulefree_snapshots:
                key = (parameter_name, source)
                samples = schedulefree_trajectory_values.setdefault(key, [])
                if len(samples) < args.state_trajectory_max_snapshots:
                    samples.append(value)
