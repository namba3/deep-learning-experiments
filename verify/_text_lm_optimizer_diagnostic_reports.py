"""Assemble diagnostic result sections for optimizer convergence probes."""

from __future__ import annotations

import math
import statistics

import torch
from optimizers.factory import is_schedule_free_optimizer
from verify._text_lm_optimizer_trajectory import (
    _causal_trajectory_pca_metrics,
    _loss_second_difference_metrics,
    _rank_analysis_role,
    _rolling_trajectory_pca_metrics,
    _trajectory_curvature_metrics,
    _trajectory_pca_metrics,
    _trajectory_position_curvature_metrics,
)


def _base_case_loss_metrics(
    args,
    name: str,
    seed: int,
    model: torch.nn.Module,
    total_steps: int,
    train_history: list[float],
    validation_history: list[float],
) -> dict[str, object]:
    return {
        "status": "passed",
        "optimizer": name,
        "seed": seed,
        "rank": args.rank if name in (
            "AdamW-LRSF", "CAME-LRSF", "APOLLO", "APOLLO-Conf",
            "APOLLO-CAME",
            "APOLLO-CAME-LRSF", "AdamW-LR-EMA-Conf-LRSF",
        ) else None,
        "parameter_numel": sum(parameter.numel() for parameter in model.parameters()),
        "total_steps": total_steps,
        "train_loss_history": train_history,
        "validation_loss_history": validation_history,
        "final_train_loss": train_history[-1],
        "final_validation_loss": validation_history[-1],
        "final_perplexity": math.exp(validation_history[-1]),
    }


def _case_resource_metrics(
    device: torch.device,
    baseline_allocated: int,
    state_bytes: int,
    state_elements: int,
    peak_state_bytes: int,
    peak_state_elements: int,
    step_seconds: float,
    total_steps: int,
) -> dict[str, int | float | None]:
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak_delta_allocated = peak_allocated - baseline_allocated
    else:
        peak_allocated = None
        peak_reserved = None
        peak_delta_allocated = None
    return {
        "persistent_state_bytes": state_bytes,
        "persistent_state_elements": state_elements,
        "peak_persistent_state_bytes": peak_state_bytes,
        "peak_persistent_state_elements": peak_state_elements,
        "host_seconds_per_optimizer_step": step_seconds / max(1, total_steps),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_delta_allocated_bytes": peak_delta_allocated,
    }


def _trajectory_result_metrics(
    args,
    name: str,
    state_trajectory_values: dict[tuple[str, str], list[torch.Tensor]],
    update_trajectory_values: dict[tuple[str, str], list[torch.Tensor]],
    update_trajectory_shapes: dict[str, tuple[int, ...]],
    schedulefree_trajectory_values: dict[tuple[str, str], list[torch.Tensor]],
    schedulefree_gap_history: list[dict[str, object]],
    train_step_loss_history: list[float],
    validation_history: list[float],
    validation_step_loss_history: list[float],
) -> dict[str, object]:
    result: dict[str, object] = {}
    if args.record_state_trajectory_pca:
        trajectory_metrics = []
        for (parameter_name, source), values in state_trajectory_values.items():
            metrics = _trajectory_pca_metrics(
                values, max_elements=args.state_rank_max_elements,
            )
            if metrics is None:
                continue
            trajectory_metrics.append({
                "parameter": parameter_name,
                "source": source,
                "analysis_role": _rank_analysis_role(source),
                **metrics,
            })
        result["state_trajectory_pca_config"] = {
            "interval": args.state_rank_interval,
            "max_elements": args.state_rank_max_elements,
            "max_tensors": args.state_rank_max_tensors,
            "max_snapshots": args.state_trajectory_max_snapshots,
            "parameter_substring": args.state_rank_parameter,
        }
        result["state_trajectory_pca"] = trajectory_metrics
    if args.record_update_trajectory_pca:
        update_trajectory_metrics = []
        for (parameter_name, source), values in update_trajectory_values.items():
            metrics = _trajectory_pca_metrics(
                values, max_elements=args.state_rank_max_elements,
            )
            if metrics is None:
                continue
            update_trajectory_metrics.append({
                "parameter": parameter_name,
                "source": source,
                "analysis_role": "effective_update",
                **metrics,
                "causal": _causal_trajectory_pca_metrics(
                    values,
                    max_elements=args.state_rank_max_elements,
                    calibration_fraction=args.trajectory_pca_calibration_fraction,
                ),
                "rolling": (
                    _rolling_trajectory_pca_metrics(
                        values,
                        max_elements=args.state_rank_max_elements,
                        window=args.trajectory_pca_rolling_window,
                        matrix_shape=update_trajectory_shapes.get(parameter_name),
                        residual_compression_block_size=(
                            args.residual_compression_block_size
                        ),
                        residual_compression_scale_mode=(
                            args.residual_compression_scale_mode
                        ),
                    )
                    if args.trajectory_pca_rolling_window > 0
                    else None
                ),
                "fixed_basis": (
                    _rolling_trajectory_pca_metrics(
                        values,
                        max_elements=args.state_rank_max_elements,
                        window=args.trajectory_pca_rolling_window,
                        matrix_shape=update_trajectory_shapes.get(parameter_name),
                        residual_compression_block_size=(
                            args.residual_compression_block_size
                        ),
                        residual_compression_scale_mode=(
                            args.residual_compression_scale_mode
                        ),
                        basis_mode="fixed",
                    )
                    if (
                        args.record_fixed_basis_residual_approximation
                        and args.trajectory_pca_rolling_window > 0
                    )
                    else None
                ),
            })
        result["update_trajectory_pca_config"] = {
            "interval": args.state_rank_interval,
            "max_elements": args.state_rank_max_elements,
            "max_tensors": args.state_rank_max_tensors,
            "max_snapshots": args.state_trajectory_max_snapshots,
            "parameter_substring": args.state_rank_parameter,
            "calibration_fraction": args.trajectory_pca_calibration_fraction,
            "rolling_window": args.trajectory_pca_rolling_window,
            "record_fixed_basis_residual_approximation": (
                args.record_fixed_basis_residual_approximation
            ),
            "residual_compression_block_size": (
                args.residual_compression_block_size
            ),
            "residual_compression_scale_mode": (
                args.residual_compression_scale_mode
            ),
        }
        result["update_trajectory_pca"] = update_trajectory_metrics
    if args.record_trajectory_curvature:
        result["loss_curvature_config"] = {
            "train_sequence": "per_step_training_loss_before_update",
            "validation_sequence": "per_epoch_validation_loss",
            "validation_step_sequence": (
                "periodic_validation_loss_after_optimizer_step"
            ),
            "eval_interval": args.eval_interval,
            "note": (
                "The train sequence includes minibatch noise; compare with "
                "the same batch order and token budget."
            ),
        }
        result["loss_curvature"] = {
            "train_step": _loss_second_difference_metrics(
                train_step_loss_history
            ),
            "validation_epoch": _loss_second_difference_metrics(
                validation_history
            ),
            "validation_step": _loss_second_difference_metrics(
                validation_step_loss_history
            ),
        }
        curvature_metrics = []
        for (parameter_name, source), values in update_trajectory_values.items():
            metrics = _trajectory_curvature_metrics(
                values, max_elements=args.state_rank_max_elements,
            )
            if metrics is None:
                continue
            curvature_metrics.append({
                "parameter": parameter_name,
                "source": source,
                "analysis_role": "effective_update",
                **metrics,
            })
        result["trajectory_curvature_config"] = {
            "interval": args.state_rank_interval,
            "max_elements": args.state_rank_max_elements,
            "max_tensors": args.state_rank_max_tensors,
            "max_snapshots": args.state_trajectory_max_snapshots,
            "parameter_substring": args.state_rank_parameter,
        }
        result["trajectory_curvature"] = curvature_metrics
        if is_schedule_free_optimizer(name):
            schedulefree_curvature_metrics = []
            for (parameter_name, source), values in schedulefree_trajectory_values.items():
                metrics = _trajectory_position_curvature_metrics(
                    values, max_elements=args.state_rank_max_elements,
                )
                if metrics is None:
                    continue
                schedulefree_curvature_metrics.append({
                    "parameter": parameter_name,
                    "source": source,
                    "analysis_role": "schedulefree_trajectory",
                    **metrics,
                })
            result["schedulefree_trajectory_curvature_config"] = {
                "interval": args.state_rank_interval,
                "max_elements": args.state_rank_max_elements,
                "max_tensors": args.state_rank_max_tensors,
                "max_snapshots": args.state_trajectory_max_snapshots,
                "minimum_position_samples": 4,
                "maximum_position_samples_recorded": max(
                    (
                        len(values)
                        for values in schedulefree_trajectory_values.values()
                    ),
                    default=0,
                ),
                "parameter_substring": args.state_rank_parameter,
                "sources": [
                    "train_parameter",
                    "hidden_state",
                    "eval_parameter",
                ],
                "position_semantics": (
                    "train_parameter is the train-mode y; hidden_state is z/s; "
                    "eval_parameter is x reconstructed with the optimizer's "
                    "non-mutating eval transform."
                ),
            }
            result["schedulefree_trajectory_curvature"] = (
                schedulefree_curvature_metrics
            )
            result["schedulefree_trajectory_curvature_status"] = (
                "passed"
                if schedulefree_curvature_metrics
                else "insufficient_samples"
            )
            result["schedulefree_trajectory_gap"] = schedulefree_gap_history
    return result


def _update_reconstruction_result_metrics(
    args,
    update_reconstruction_values: list[dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    if args.record_update_reconstruction:
        grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
        for item in update_reconstruction_values:
            key = (item["parameter"], item["source"], item["rank"])
            grouped.setdefault(key, []).append(item)
        reconstruction_metrics = []
        for (parameter_name, source, rank), items in grouped.items():
            item_metrics = {
                "parameter": parameter_name,
                "source": source,
                "rank": rank,
                "samples": len(items),
                "state_relative_error_mean": statistics.fmean(
                    item["state_relative_error"] for item in items
                ),
                "update_relative_error_mean": statistics.fmean(
                    item["update_relative_error"] for item in items
                ),
                "update_cosine_mean": statistics.fmean(
                    item["update_cosine"] for item in items
                ),
                "update_norm_ratio_mean": statistics.fmean(
                    item["update_norm_ratio"] for item in items
                ),
            }
            preconditioner_errors = [
                item["preconditioner_relative_error"]
                for item in items
                if "preconditioner_relative_error" in item
            ]
            if preconditioner_errors:
                item_metrics["preconditioner_relative_error_mean"] = (
                    statistics.fmean(preconditioner_errors)
                )
            reconstruction_metrics.append(item_metrics)
        result["update_reconstruction_config"] = {
            "interval": args.state_rank_interval,
            "ranks": list(args.update_reconstruction_ranks),
            "max_elements": args.state_rank_max_elements,
            "max_tensors": args.state_rank_max_tensors,
            "parameter_substring": args.state_rank_parameter,
        }
        result["update_reconstruction"] = reconstruction_metrics
    return result


def _refresh_transport_metrics(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    count = 0
    error_sum = 0.0
    error_max = 0.0
    norm_ratio_sum = 0.0
    cosine_sum = 0.0
    last_step = None
    shadow_count = 0
    shadow_error_sum = 0.0
    shadow_error_max = 0.0
    shadow_norm_ratio_sum = 0.0
    shadow_cosine_sum = 0.0
    shadow_last_step = None
    for state in optimizer.state.values():
        local_count = int(state.get("refresh_transport_diagnostic_count", 0))
        if local_count:
            count += local_count
            error_sum += float(state.get("refresh_transport_error_sum", 0.0))
            error_max = max(
                error_max,
                float(state.get("refresh_transport_error_max", 0.0)),
            )
            norm_ratio_sum += float(
                state.get("refresh_transport_norm_ratio_sum", 0.0)
            )
            cosine_sum += float(state.get("refresh_transport_cosine_sum", 0.0))
            candidate_step = state.get("refresh_transport_last_step")
            if candidate_step is not None:
                last_step = max(int(candidate_step), last_step or 0)
        local_shadow_count = int(state.get("shadow_gap_count", 0))
        if local_shadow_count:
            shadow_count += local_shadow_count
            shadow_error_sum += float(state.get("shadow_gap_error_sum", 0.0))
            shadow_error_max = max(
                shadow_error_max,
                float(state.get("shadow_gap_error_max", 0.0)),
            )
            shadow_norm_ratio_sum += float(
                state.get("shadow_gap_norm_ratio_sum", 0.0)
            )
            shadow_cosine_sum += float(
                state.get("shadow_gap_cosine_sum", 0.0)
            )
            candidate_step = state.get("shadow_gap_last_step")
            if candidate_step is not None:
                shadow_last_step = max(
                    int(candidate_step), shadow_last_step or 0,
                )
    result = {
        "shadow_gap_events": shadow_count,
        "shadow_gap_error_mean": (
            None if shadow_count == 0 else shadow_error_sum / shadow_count
        ),
        "shadow_gap_error_max": None if shadow_count == 0 else shadow_error_max,
        "shadow_gap_norm_ratio_mean": (
            None
            if shadow_count == 0
            else shadow_norm_ratio_sum / shadow_count
        ),
        "shadow_gap_cosine_mean": (
            None if shadow_count == 0 else shadow_cosine_sum / shadow_count
        ),
        "shadow_gap_last_step": shadow_last_step,
    }
    if count == 0:
        result.update({
            "refresh_transport_diagnostic_events": 0,
            "refresh_transport_error_mean": None,
            "refresh_transport_error_max": None,
            "refresh_transport_norm_ratio_mean": None,
            "refresh_transport_cosine_mean": None,
            "refresh_transport_last_step": None,
        })
        return result
    result.update({
        "refresh_transport_diagnostic_events": count,
        "refresh_transport_error_mean": error_sum / count,
        "refresh_transport_error_max": error_max,
        "refresh_transport_norm_ratio_mean": norm_ratio_sum / count,
        "refresh_transport_cosine_mean": cosine_sum / count,
        "refresh_transport_last_step": last_step,
    })
    return result


def _optimizer_state_result_metrics(
    args,
    optimizer: torch.optim.Optimizer,
    confidence_diagnostics: list[dict[str, object]],
    state_rank_history: list[dict[str, object]],
    lrsf_latent_moment_history: list[dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    if args.record_confidence_diagnostics:
        result["confidence_diagnostics_config"] = {
            "interval": args.state_rank_interval,
            "definition": "m_hat^2 / (m_hat^2 + c_hat)",
            "alpha": args.adamw_lr_ema_confidence_alpha,
            "confidence_beta": args.adamw_lr_ema_confidence_beta,
        }
        result["confidence_diagnostics"] = confidence_diagnostics
    if args.record_state_rank:
        result["state_rank_config"] = {
            "interval": args.state_rank_interval,
            "max_elements": args.state_rank_max_elements,
            "max_tensors": args.state_rank_max_tensors,
            "parameter_substring": args.state_rank_parameter,
            "delta_ema_decay": args.state_rank_delta_ema_decay,
        }
        result["state_rank_history"] = state_rank_history
    if args.record_refresh_diagnostics:
        result.update(_refresh_transport_metrics(optimizer))
        result["lrsf_latent_moment_history"] = lrsf_latent_moment_history
    return result
