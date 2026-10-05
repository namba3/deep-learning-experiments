"""Compare optimizers on the repository's TinyStories language task.

This probe reuses :mod:`text_lm.train` for the tokenizer, dataset formatting,
collator, and ``TinyTextLM`` model, but keeps the experiment deliberately
small and deterministic.  It is intended for optimizer comparisons rather
than production checkpoint training.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from itertools import islice
from time import perf_counter

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from text_lm.train import (  # noqa: E402
    load_text_datasets,
)
from optimizers.factory import build_optimizer, is_schedule_free_optimizer  # noqa: E402
from optimizers.projection_refresh import (  # noqa: E402
    ProjectionRefreshPolicy as ProjectionRefreshPolicy,
    add_mixed_delta as add_mixed_delta,
)
from verify._text_lm_optimizer_evaluation import _evaluate_validation  # noqa: E402
from verify._text_lm_optimizer_runtime import (  # noqa: E402
    _build_case_loaders as _build_case_loaders,
    _build_model as _build_model,
    _load_initial_state as _load_initial_state,
    _loss as _loss,
    _optimizer_args as _optimizer_args,
    _resolve_device as _resolve_device,
    _resolve_dtype as _resolve_dtype,
    _state_metrics as _state_metrics,
)
from verify._text_lm_optimizer_diagnostic_reports import (  # noqa: E402
    _base_case_loss_metrics as _base_case_loss_metrics,
    _case_resource_metrics as _case_resource_metrics,
    _optimizer_state_result_metrics as _optimizer_state_result_metrics,
    _refresh_transport_metrics as _refresh_transport_metrics,
    _trajectory_result_metrics,
    _update_reconstruction_result_metrics,
)
from verify._text_lm_optimizer_residuals import (  # noqa: E402
    _blockwise_dequantize as _blockwise_dequantize,
    _matrix_rank_metrics as _matrix_rank_metrics,
    _quantization_threshold as _quantization_threshold,
    _residual_approximation_metrics as _residual_approximation_metrics,
    _residual_compression_metrics as _residual_compression_metrics,
    _residual_error_feedback_metrics as _residual_error_feedback_metrics,
)
from verify._text_lm_optimizer_state_reconstruction import (  # noqa: E402
    _schedulefree_step_context as _schedulefree_step_context,
    _truncated_svd_reconstructions as _truncated_svd_reconstructions,
    _update_reconstruction_before_snapshot as _update_reconstruction_before_snapshot,
    _update_reconstruction_metrics as _update_reconstruction_metrics,
    _update_sf_delta_ema as _update_sf_delta_ema,
)
from verify._text_lm_optimizer_state_diagnostics import (  # noqa: E402
    _capture_pre_step_diagnostics as _capture_pre_step_diagnostics,
    _confidence_diagnostic_snapshot as _confidence_diagnostic_snapshot,
    _lrsf_latent_moment_snapshot as _lrsf_latent_moment_snapshot,
    _decode_projected_matrix as _decode_projected_matrix,
    _projection_geometry as _projection_geometry,
    _record_state_diagnostics,
    _record_update_trajectory_snapshots,
    _schedulefree_beta as _schedulefree_beta,
    _schedulefree_hidden_state as _schedulefree_hidden_state,
    _schedulefree_refresh_policy as _schedulefree_refresh_policy,
    _schedulefree_trajectory_gap_snapshot as _schedulefree_trajectory_gap_snapshot,
    _schedulefree_trajectory_snapshot as _schedulefree_trajectory_snapshot,
    _state_rank_snapshot as _state_rank_snapshot,
    _state_summary as _state_summary,
    _state_trajectory_snapshot as _state_trajectory_snapshot,
)
from verify._text_lm_optimizer_trajectory import (  # noqa: E402
    _causal_trajectory_pca_metrics as _causal_trajectory_pca_metrics,
    _loss_second_difference_metrics as _loss_second_difference_metrics,
    _rank_analysis_role as _rank_analysis_role,
    _rolling_trajectory_pca_metrics as _rolling_trajectory_pca_metrics,
    _trajectory_curvature_metrics as _trajectory_curvature_metrics,
    _trajectory_pca_metrics as _trajectory_pca_metrics,
    _trajectory_position_curvature_metrics as _trajectory_position_curvature_metrics,
    _update_trajectory_before_snapshot as _update_trajectory_before_snapshot,
    _update_trajectory_snapshot as _update_trajectory_snapshot,
)

from verify._text_lm_optimizer_cli import (  # noqa: E402
    OPTIMIZERS as OPTIMIZERS,
    SUPPORTED_OPTIMIZERS as SUPPORTED_OPTIMIZERS,
    _parse_csv as _parse_csv,
    _parse_positive_int_csv as _parse_positive_int_csv,
    _parse_seeds as _parse_seeds,
    parse_args as _parse_args,
    _validate_args as _validate_args,
)


def parse_args(argv=None):
    """Parse probe options while keeping this module's CLI description."""
    return _parse_args(argv, description=__doc__)


def _run_case(args, name: str, seed: int, tokenizer, train_dataset, eval_dataset,
              initial_state, device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    torch.manual_seed(seed)
    model = _build_model(args, len(tokenizer), device, dtype)
    model.load_state_dict(initial_state, strict=True)
    optimizer = build_optimizer(name, model.parameters(), args=_optimizer_args(args, seed))
    if is_schedule_free_optimizer(name):
        optimizer.train()

    sampler, loader, eval_loader = _build_case_loaders(
        args, tokenizer, train_dataset, eval_dataset, device, seed,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
    else:
        baseline_allocated = 0
    train_history: list[float] = []
    validation_history: list[float] = []
    validation_step_loss_history: list[float] = []
    validation_step_loss_records: list[dict[str, object]] = []
    train_step_loss_history: list[float] = []
    update_norms: list[float] = []
    step_seconds = 0.0
    total_steps = 0
    peak_state_bytes = 0
    peak_state_elements = 0
    state_rank_history: list[dict[str, object]] = []
    state_trajectory_values: dict[tuple[str, str], list[torch.Tensor]] = {}
    update_trajectory_values: dict[tuple[str, str], list[torch.Tensor]] = {}
    update_trajectory_shapes: dict[str, tuple[int, ...]] = {}
    schedulefree_trajectory_values: dict[tuple[str, str], list[torch.Tensor]] = {}
    schedulefree_gap_history: list[dict[str, object]] = []
    update_reconstruction_values: list[dict[str, object]] = []
    sf_delta_ema: dict[str, torch.Tensor] = {}
    confidence_diagnostics: list[dict[str, object]] = []
    lrsf_latent_moment_history: list[dict[str, object]] = []
    previous_refresh_counts: dict[int, int] = {}

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        if is_schedule_free_optimizer(name):
            optimizer.train()
        epoch_loss = 0.0
        train_steps = 0
        epoch_loader = loader if args.steps_per_epoch <= 0 else islice(loader, args.steps_per_epoch)
        for batch in epoch_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            before = None
            if args.record_update_norms:
                before = [parameter.detach().float().cpu().clone() for parameter in model.parameters()]
            update_before, reconstruction_before = _capture_pre_step_diagnostics(
                args, model, optimizer, update_trajectory_shapes,
            )
            started = perf_counter()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds += perf_counter() - started
            if before is not None:
                squared = sum(
                    (parameter.detach().float().cpu() - old).square().sum().item()
                    for parameter, old in zip(model.parameters(), before)
                )
                update_norms.append(squared ** 0.5)
            total_steps += 1
            if (
                args.eval_interval > 0
                and total_steps % args.eval_interval == 0
            ):
                periodic_validation_loss = _evaluate_validation(
                    model,
                    optimizer,
                    eval_loader,
                    device,
                    name,
                    _loss,
                    restore_training=True,
                )
                validation_step_loss_history.append(periodic_validation_loss)
                validation_step_loss_records.append({
                    "step": total_steps,
                    "validation_loss": periodic_validation_loss,
                })
            if args.record_refresh_diagnostics:
                moment_snapshot = _lrsf_latent_moment_snapshot(
                    optimizer, total_steps, previous_refresh_counts,
                )
                if moment_snapshot is not None:
                    lrsf_latent_moment_history.append(moment_snapshot)
            _record_update_trajectory_snapshots(
                args,
                name,
                model,
                optimizer,
                total_steps,
                loss,
                update_before,
                update_trajectory_values,
                schedulefree_trajectory_values,
                schedulefree_gap_history,
            )
            current_bytes, current_elements = _state_metrics(optimizer)
            peak_state_bytes = max(peak_state_bytes, current_bytes)
            peak_state_elements = max(peak_state_elements, current_elements)
            _record_state_diagnostics(
                args,
                name,
                model,
                optimizer,
                total_steps,
                loss,
                sf_delta_ema,
                state_rank_history,
                state_trajectory_values,
                reconstruction_before,
                update_reconstruction_values,
                confidence_diagnostics,
            )
            epoch_loss += float(loss.detach())
            if args.record_trajectory_curvature:
                train_step_loss_history.append(float(loss.detach()))
            train_steps += 1
        train_history.append(epoch_loss / max(1, train_steps))

        eval_loss = _evaluate_validation(
            model, optimizer, eval_loader, device, name, _loss
        )
        validation_history.append(eval_loss)

    state_bytes, state_elements = _state_metrics(optimizer)
    result: dict[str, object] = _base_case_loss_metrics(
        args,
        name,
        seed,
        model,
        total_steps,
        train_history,
        validation_history,
    )
    result.update(_case_resource_metrics(
        device,
        baseline_allocated,
        state_bytes,
        state_elements,
        peak_state_bytes,
        peak_state_elements,
        step_seconds,
        total_steps,
    ))
    if args.record_update_norms:
        result.update({
            "update_norm_mean": statistics.fmean(update_norms),
            "update_norm_variance": statistics.pvariance(update_norms),
            "update_norm_history": update_norms,
        })
    result.update(_optimizer_state_result_metrics(
        args,
        optimizer,
        confidence_diagnostics,
        state_rank_history,
        lrsf_latent_moment_history,
    ))
    result.update(
        _trajectory_result_metrics(
            args,
            name,
            state_trajectory_values,
            update_trajectory_values,
            update_trajectory_shapes,
            schedulefree_trajectory_values,
            schedulefree_gap_history,
            train_step_loss_history,
            validation_history,
            validation_step_loss_history,
        )
    )
    result.update(
        _update_reconstruction_result_metrics(
            args,
            update_reconstruction_values,
        )
    )
    if args.eval_interval > 0:
        result["validation_step_loss_records"] = validation_step_loss_records
    return result


def run(args) -> dict[str, object]:
    _validate_args(args)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    seeds = args.seeds or (args.seed,)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset, eval_dataset = load_text_datasets(
        args.dataset_name,
        args.dataset_config,
        args.dataset_path,
        args.text_column,
        tokenizer,
        args.max_seq_len,
        args.train_tokens,
        args.eval_tokens,
        split=args.dataset_split,
    )
    cases = []
    for seed in seeds:
        initial_state = _load_initial_state(
            args, tokenizer, device, dtype, seed,
        )
        for name in args.optimizers:
            cases.append(_run_case(
                args, name, seed, tokenizer, train_dataset, eval_dataset,
                initial_state, device, dtype,
            ))
    return {
        "status": "passed",
        "device": str(device),
        "dtype": args.dtype,
        "data_mode": "text",
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "text_column": args.text_column,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "train_tokens": args.train_tokens,
        "eval_tokens": args.eval_tokens,
        "loss_mode": "all_packed_tokens",
        "model": {
            "architecture": "naive",
            "max_seq_len": args.max_seq_len,
            "embed_dim": args.embed_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "kv_heads": args.kv_heads,
        },
        "optimizer_config": {
            "rank": args.rank,
            "lr_ema_beta": args.adamw_lr_ema_beta,
            "lr_ema_projection_scale": args.adamw_lr_ema_projection_scale,
            "lr_ema_confidence_beta": args.adamw_lr_ema_confidence_beta,
            "lr_ema_confidence_alpha": args.adamw_lr_ema_confidence_alpha,
            "adamw_sf_backend": args.adamw_sf_backend,
            "refresh_mode": args.refresh_mode,
            "refresh_interval": args.refresh_interval,
            "refresh_window": args.refresh_window,
            "refresh_mix": args.refresh_mix,
            "lrsf_refresh_state": args.lrsf_refresh_state,
            "refresh_ema_decay": args.refresh_ema_decay,
            "record_refresh_diagnostics": args.record_refresh_diagnostics,
            "refresh_transport_overlap": args.refresh_transport_overlap,
            "orthogonal_rate": args.orthogonal_rate,
            "orthogonal_direction": args.orthogonal_direction,
            "orthogonal_signal": args.orthogonal_signal,
            "record_state_rank": args.record_state_rank,
            "record_confidence_diagnostics": args.record_confidence_diagnostics,
            "record_update_trajectory_pca": args.record_update_trajectory_pca,
            "trajectory_pca_calibration_fraction": (
                args.trajectory_pca_calibration_fraction
            ),
            "trajectory_pca_rolling_window": args.trajectory_pca_rolling_window,
            "record_trajectory_curvature": args.record_trajectory_curvature,
            "eval_interval": args.eval_interval,
            "record_update_reconstruction": args.record_update_reconstruction,
            "update_reconstruction_ranks": list(args.update_reconstruction_ranks),
            "state_rank_interval": args.state_rank_interval,
            "state_rank_max_elements": args.state_rank_max_elements,
            "state_rank_max_tensors": args.state_rank_max_tensors,
            "state_rank_parameter": args.state_rank_parameter,
            "state_rank_delta_ema_decay": args.state_rank_delta_ema_decay,
            "record_state_trajectory_pca": args.record_state_trajectory_pca,
            "state_trajectory_max_snapshots": args.state_trajectory_max_snapshots,
        "apollo_norm_growth_rate": args.apollo_norm_growth_rate,
        "apollo_scale": args.apollo_scale,
        "apollo_confidence_beta": args.adamw_lr_ema_confidence_beta,
        "apollo_confidence_alpha": args.adamw_lr_ema_confidence_alpha,
            "apollo_disable_norm_growth_limiter": (
                args.apollo_disable_norm_growth_limiter
            ),
        },
        "cases": cases,
    }

def main(argv=None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as error:
        print(json.dumps({
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
