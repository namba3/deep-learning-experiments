"""Temporal PCA metrics for optimizer update trajectories."""

from __future__ import annotations

import statistics
from typing import cast

import torch

from verify._text_lm_optimizer_residuals import (
    _residual_approximation_metrics,
    _residual_compression_metrics,
    _residual_error_feedback_metrics,
)


def _trajectory_pca_metrics(
    values: list[torch.Tensor],
    *,
    max_elements: int,
) -> dict[str, object] | None:
    """Measure temporal PCA rank without constructing an element covariance."""
    if not values or values[0].numel() > max_elements or len(values) < 2:
        return None
    matrix = torch.stack([value.detach().float().reshape(-1) for value in values])
    if not torch.isfinite(matrix).all():
        return None
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    gram = centered.matmul(centered.transpose(0, 1))
    eigenvalues = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0.0)
    total_energy = eigenvalues.sum()
    if not torch.isfinite(total_energy) or total_energy <= 0:
        return {
            "metric_type": "trajectory_pca",
            "samples": len(values),
            "features": int(matrix.shape[1]),
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "rank_90": 0,
            "rank_95": 0,
            "rank_99": 0,
            "retained_energy": {str(rank): 0.0 for rank in (1, 2, 4, 8, 16)},
        }
    probabilities = eigenvalues / total_energy
    nonzero = probabilities > 1e-30
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    cumulative = eigenvalues.cumsum(0) / total_energy

    def threshold_rank(value: float) -> int:
        return int(torch.searchsorted(cumulative, value).item()) + 1

    return {
        "metric_type": "trajectory_pca",
        "samples": len(values),
        "features": int(matrix.shape[1]),
        "effective_rank": float(torch.exp(entropy)),
        "participation_ratio": float(
            1.0 / probabilities.square().sum().clamp_min(1e-30)
        ),
        "rank_90": threshold_rank(0.90),
        "rank_95": threshold_rank(0.95),
        "rank_99": threshold_rank(0.99),
        "retained_energy": {
            str(rank): float(cumulative[min(rank, cumulative.numel()) - 1])
            for rank in (1, 2, 4, 8, 16)
        },
    }


def _causal_trajectory_pca_metrics(
    values: list[torch.Tensor],
    *,
    max_elements: int,
    calibration_fraction: float = 0.5,
) -> dict[str, object] | None:
    """Fit a temporal PCA basis on the past and test it on the future.

    Ordinary trajectory PCA uses all samples and therefore measures
    compressibility retrospectively. This diagnostic avoids that leakage:
    the calibration mean and basis are fitted only on the first part of the
    sampled trajectory, then the remaining updates are reconstructed without
    refitting.
    """
    if (
        len(values) < 4
        or not 0.0 < calibration_fraction < 1.0
        or values[0].numel() > max_elements
    ):
        return None
    if any(value.numel() != values[0].numel() for value in values):
        return None

    matrix = torch.stack([value.detach().float().reshape(-1) for value in values])
    if not torch.isfinite(matrix).all():
        return None
    calibration_samples = min(
        max(2, int(round(len(values) * calibration_fraction))),
        len(values) - 2,
    )
    future_samples = len(values) - calibration_samples
    calibration = matrix[:calibration_samples]
    future = matrix[calibration_samples:]
    calibration_mean = calibration.mean(dim=0, keepdim=True)
    centered_calibration = calibration - calibration_mean
    gram = centered_calibration.matmul(centered_calibration.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
    eigenvectors = eigenvectors.flip(1)
    total_calibration_energy = eigenvalues.sum()
    ranks = (1, 2, 4, 8, 16)
    if (
        not torch.isfinite(total_calibration_energy)
        or total_calibration_energy <= 0
    ):
        return {
            "metric_type": "causal_trajectory_pca",
            "samples": len(values),
            "features": int(matrix.shape[1]),
            "calibration_samples": calibration_samples,
            "future_samples": future_samples,
            "calibration_fraction": calibration_fraction,
            "calibration_rank_limit": 0,
            "calibration_effective_rank": 0.0,
            "calibration_participation_ratio": 0.0,
            "calibration_retained_energy": {str(rank): 0.0 for rank in ranks},
            "future_reconstruction_error": {str(rank): None for rank in ranks},
            "future_explained_variance": {str(rank): None for rank in ranks},
        }

    probabilities = eigenvalues / total_calibration_energy
    nonzero = probabilities > 1e-30
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    cumulative = eigenvalues.cumsum(0) / total_calibration_energy
    singular_values = eigenvalues.sqrt()
    valid = singular_values > 1e-15
    components = (
        eigenvectors[:, valid].transpose(0, 1).matmul(centered_calibration)
        / singular_values[valid].unsqueeze(1)
        if valid.any()
        else eigenvalues.new_zeros((0, matrix.shape[1]))
    )

    centered_future = future - calibration_mean
    future_energy = centered_future.square().sum()
    calibration_retained_energy = {
        str(rank): float(cumulative[min(rank, cumulative.numel()) - 1])
        for rank in ranks
    }
    future_reconstruction_error: dict[str, float | None] = {}
    future_explained_variance: dict[str, float | None] = {}
    for rank in ranks:
        component_count = min(rank, components.shape[0])
        if component_count == 0 or future_energy <= 0:
            future_reconstruction_error[str(rank)] = None
            future_explained_variance[str(rank)] = None
            continue
        basis = components[:component_count]
        reconstruction = centered_future.matmul(basis.transpose(0, 1)).matmul(basis)
        residual = centered_future - reconstruction
        relative_error = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(
            centered_future
        ).clamp_min(1e-30)
        explained = 1.0 - residual.square().sum() / future_energy.clamp_min(1e-30)
        future_reconstruction_error[str(rank)] = float(relative_error)
        future_explained_variance[str(rank)] = float(explained.clamp(-1.0, 1.0))

    return {
        "metric_type": "causal_trajectory_pca",
        "samples": len(values),
        "features": int(matrix.shape[1]),
        "calibration_samples": calibration_samples,
        "future_samples": future_samples,
        "calibration_fraction": calibration_fraction,
        "calibration_rank_limit": min(
            int(matrix.shape[1]), calibration_samples - 1,
        ),
        "calibration_effective_rank": float(torch.exp(entropy)),
        "calibration_participation_ratio": float(
            1.0 / probabilities.square().sum().clamp_min(1e-30)
        ),
        "calibration_retained_energy": calibration_retained_energy,
        "future_reconstruction_error": future_reconstruction_error,
        "future_explained_variance": future_explained_variance,
    }


def _rolling_residual_summaries(
    ranks: tuple[int, ...],
    residual_compression: dict[int, list[dict[str, object]]],
    residual_approximations: dict[int, list[dict[str, object]]],
    residual_sequences: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    *,
    max_elements: int,
    block_size: int,
    scale_mode: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Aggregate residual compression and approximation diagnostics."""
    def mean_or_none(values_for_rank: list[float]) -> float | None:
        return statistics.fmean(values_for_rank) if values_for_rank else None

    def compression_mean(rank: int, key: str) -> float | None:
        values_for_rank = [
            float(item[key]) for item in residual_compression[rank]
        ]
        return mean_or_none(values_for_rank)

    def compression_energy_mean(rank: int, energy_rank: str) -> float | None:
        values_for_rank = [
            float(cast(dict[str, object], item["spatial_retained_energy"])[energy_rank])
            for item in residual_compression[rank]
        ]
        return mean_or_none(values_for_rank)

    def approximation_mean(
        temporal_rank: int,
        method: str,
        spatial_rank: str | None,
        key: str,
    ) -> float | None:
        values_for_rank: list[float] = []
        for item in residual_approximations[temporal_rank]:
            method_metrics = cast(dict[str, object], item[method])
            if spatial_rank is not None:
                method_metrics = cast(
                    dict[str, object], method_metrics[spatial_rank]
                )
            value = method_metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values_for_rank.append(float(value))
        return mean_or_none(values_for_rank)

    def approximation_samples(temporal_rank: int) -> int:
        return len(residual_approximations[temporal_rank])

    residual_error_feedback: dict[int, dict[str, object]] = {}
    for temporal_rank in ranks:
        sequence = residual_sequences[temporal_rank]
        if not sequence:
            continue
        feedback_metrics = _residual_error_feedback_metrics(
            [target for target, _ in sequence],
            [residual for _, residual in sequence],
            max_elements=max_elements,
            block_size=block_size,
            scale_mode=scale_mode,
            quantization_bits=4,
        )
        if feedback_metrics is not None:
            residual_error_feedback[temporal_rank] = feedback_metrics

    def approximation_summary(temporal_rank: int) -> dict[str, object]:
        summary: dict[str, object] = {
            "samples": approximation_samples(temporal_rank),
            "low_rank_factor": {},
            "low_rank_plus_int8": {},
            "blockwise_int4_error_feedback": residual_error_feedback.get(
                temporal_rank, {}
            ),
            "blockwise_int8": {
                "block_size": block_size,
                "scale_mode": scale_mode,
                "storage_bytes_int8": approximation_mean(
                    temporal_rank, "blockwise_int8", None,
                    "storage_bytes_int8",
                ),
                "storage_ratio_to_target_bf16": approximation_mean(
                    temporal_rank, "blockwise_int8", None,
                    "storage_ratio_to_target_bf16",
                ),
                "decode_milliseconds": approximation_mean(
                    temporal_rank, "blockwise_int8", None,
                    "decode_milliseconds",
                ),
                "residual_relative_error": approximation_mean(
                    temporal_rank, "blockwise_int8", None,
                    "residual_relative_error",
                ),
                "update_relative_error": approximation_mean(
                    temporal_rank, "blockwise_int8", None,
                    "update_relative_error",
                ),
                "update_cosine": approximation_mean(
                    temporal_rank, "blockwise_int8", None,
                    "update_cosine",
                ),
                "update_norm_ratio": approximation_mean(
                    temporal_rank, "blockwise_int8", None,
                    "update_norm_ratio",
                ),
            },
            "blockwise_int4": {
                "block_size": block_size,
                "scale_mode": scale_mode,
                "storage_bytes_int4": approximation_mean(
                    temporal_rank, "blockwise_int4", None,
                    "storage_bytes_int4",
                ),
                "storage_ratio_to_target_bf16": approximation_mean(
                    temporal_rank, "blockwise_int4", None,
                    "storage_ratio_to_target_bf16",
                ),
                "decode_milliseconds": approximation_mean(
                    temporal_rank, "blockwise_int4", None,
                    "decode_milliseconds",
                ),
                "residual_relative_error": approximation_mean(
                    temporal_rank, "blockwise_int4", None,
                    "residual_relative_error",
                ),
                "update_relative_error": approximation_mean(
                    temporal_rank, "blockwise_int4", None,
                    "update_relative_error",
                ),
                "update_cosine": approximation_mean(
                    temporal_rank, "blockwise_int4", None,
                    "update_cosine",
                ),
                "update_norm_ratio": approximation_mean(
                    temporal_rank, "blockwise_int4", None,
                    "update_norm_ratio",
                ),
            },
        }
        low_rank_summary = cast(dict[str, object], summary["low_rank_factor"])
        hybrid_summary = cast(
            dict[str, object], summary["low_rank_plus_int8"]
        )
        for spatial_rank in ("1", "2", "4", "8", "16"):
            low_rank_summary[spatial_rank] = {
                "storage_bytes_bf16": approximation_mean(
                    temporal_rank, "low_rank_factor", spatial_rank,
                    "storage_bytes_bf16",
                ),
                "storage_ratio_to_target_bf16": approximation_mean(
                    temporal_rank, "low_rank_factor", spatial_rank,
                    "storage_ratio_to_target_bf16",
                ),
                "decode_milliseconds": approximation_mean(
                    temporal_rank, "low_rank_factor", spatial_rank,
                    "decode_milliseconds",
                ),
                "residual_relative_error": approximation_mean(
                    temporal_rank, "low_rank_factor", spatial_rank,
                    "residual_relative_error",
                ),
                "update_relative_error": approximation_mean(
                    temporal_rank, "low_rank_factor", spatial_rank,
                    "update_relative_error",
                ),
                "update_cosine": approximation_mean(
                    temporal_rank, "low_rank_factor", spatial_rank,
                    "update_cosine",
                ),
                "update_norm_ratio": approximation_mean(
                    temporal_rank, "low_rank_factor", spatial_rank,
                    "update_norm_ratio",
                ),
            }
            hybrid_summary[spatial_rank] = {
                "block_size": block_size,
                "scale_mode": scale_mode,
                "storage_bytes_bf16_factor_int8_remainder": approximation_mean(
                    temporal_rank, "low_rank_plus_int8", spatial_rank,
                    "storage_bytes_bf16_factor_int8_remainder",
                ),
                "storage_ratio_to_target_bf16": approximation_mean(
                    temporal_rank, "low_rank_plus_int8", spatial_rank,
                    "storage_ratio_to_target_bf16",
                ),
                "decode_milliseconds": approximation_mean(
                    temporal_rank, "low_rank_plus_int8", spatial_rank,
                    "decode_milliseconds",
                ),
                "residual_relative_error": approximation_mean(
                    temporal_rank, "low_rank_plus_int8", spatial_rank,
                    "residual_relative_error",
                ),
                "update_relative_error": approximation_mean(
                    temporal_rank, "low_rank_plus_int8", spatial_rank,
                    "update_relative_error",
                ),
                "update_cosine": approximation_mean(
                    temporal_rank, "low_rank_plus_int8", spatial_rank,
                    "update_cosine",
                ),
                "update_norm_ratio": approximation_mean(
                    temporal_rank, "low_rank_plus_int8", spatial_rank,
                    "update_norm_ratio",
                ),
            }
        return summary

    residual_spatial_compression = {
        str(rank): {
            "samples": len(residual_compression[rank]),
            "spatial_effective_rank": compression_mean(
                rank, "spatial_effective_rank"
            ),
            "spatial_rank_95": compression_mean(rank, "spatial_rank_95"),
            "spatial_retained_energy": {
                energy_rank: compression_energy_mean(rank, energy_rank)
                for energy_rank in ("1", "2", "4", "8", "16")
            },
            "global_int8_relative_error": compression_mean(
                rank, "global_int8_relative_error"
            ),
            "blockwise_int8_relative_error": compression_mean(
                rank, "blockwise_int8_relative_error"
            ),
            "blockwise_scalar_relative_error": compression_mean(
                rank, "blockwise_scalar_relative_error"
            ),
            "diagonal_relative_error": compression_mean(
                rank, "diagonal_relative_error"
            ),
        }
        for rank in ranks
    }
    residual_approximation = {
        str(rank): approximation_summary(rank)
        for rank in ranks
    }
    return residual_spatial_compression, residual_approximation


def _rolling_trajectory_pca_metrics(
    values: list[torch.Tensor],
    *,
    max_elements: int,
    window: int,
    matrix_shape: tuple[int, ...] | None = None,
    residual_compression_block_size: int = 256,
    residual_compression_scale_mode: str = "max_abs",
    basis_mode: str = "rolling",
) -> dict[str, object] | None:
    """Fit a PCA basis on past updates and score the next update.

    ``basis_mode="rolling"`` uses only the preceding ``window`` snapshots for
    each prediction. ``basis_mode="fixed"`` fits the basis once from the
    initial window and reuses it for all later predictions. The latter is
    useful for testing temporal error feedback in a stable coordinate system.
    """
    if (
        window < 2
        or len(values) <= window
        or values[0].numel() > max_elements
    ):
        return None
    if basis_mode not in ("rolling", "fixed"):
        raise ValueError("basis_mode must be 'rolling' or 'fixed'")
    if any(value.numel() != values[0].numel() for value in values):
        return None

    matrix = torch.stack([value.detach().float().reshape(-1) for value in values])
    if not torch.isfinite(matrix).all():
        return None

    ranks = (1, 2, 4, 8, 16)
    rank_limit = min(int(matrix.shape[1]), window - 1)
    prediction_errors: dict[int, list[float]] = {rank: [] for rank in ranks}
    prediction_explained: dict[int, list[float]] = {rank: [] for rank in ranks}
    prediction_residual_energy: dict[int, list[float]] = {
        rank: [] for rank in ranks
    }
    prediction_cosines: dict[int, list[float]] = {rank: [] for rank in ranks}
    prediction_norm_ratios: dict[int, list[float]] = {rank: [] for rank in ranks}
    residual_compression: dict[int, list[dict[str, object]]] = {
        rank: [] for rank in ranks
    }
    residual_approximations: dict[int, list[dict[str, object]]] = {
        rank: [] for rank in ranks
    }
    residual_sequences: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {
        rank: [] for rank in ranks
    }
    calibration_effective_ranks: list[float] = []
    calibration_retained: dict[int, list[float]] = {rank: [] for rank in ranks}

    for prediction_index in range(window, len(values)):
        calibration = (
            matrix[:window]
            if basis_mode == "fixed"
            else matrix[prediction_index - window:prediction_index]
        )
        target = matrix[prediction_index:prediction_index + 1]
        calibration_mean = calibration.mean(dim=0, keepdim=True)
        centered_calibration = calibration - calibration_mean
        gram = centered_calibration.matmul(centered_calibration.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
        eigenvectors = eigenvectors.flip(1)
        total_energy = eigenvalues.sum()
        if not torch.isfinite(total_energy) or total_energy <= 0:
            continue

        probabilities = eigenvalues / total_energy
        nonzero = probabilities > 1e-30
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        calibration_effective_ranks.append(float(torch.exp(entropy)))
        cumulative = eigenvalues.cumsum(0) / total_energy
        for rank in ranks:
            calibration_retained[rank].append(
                float(cumulative[min(rank, cumulative.numel()) - 1])
            )

        singular_values = eigenvalues.sqrt()
        valid = singular_values > 1e-15
        components = (
            eigenvectors[:, valid].transpose(0, 1).matmul(centered_calibration)
            / singular_values[valid].unsqueeze(1)
            if valid.any()
            else eigenvalues.new_zeros((0, matrix.shape[1]))
        )
        centered_target = target - calibration_mean
        target_energy = centered_target.square().sum()
        if not torch.isfinite(target_energy) or target_energy <= 0:
            continue
        target_norm = torch.sqrt(target_energy).clamp_min(1e-30)
        raw_target = target.reshape(-1)
        raw_target_energy = raw_target.square().sum()
        if not torch.isfinite(raw_target_energy) or raw_target_energy <= 0:
            continue
        raw_target_norm = torch.sqrt(raw_target_energy).clamp_min(1e-30)
        for rank in ranks:
            component_count = min(rank, components.shape[0])
            if component_count == 0:
                continue
            basis = components[:component_count]
            reconstruction = centered_target.matmul(basis.transpose(0, 1)).matmul(basis)
            residual = centered_target - reconstruction
            full_reconstruction = (calibration_mean + reconstruction).reshape(-1)
            full_residual = raw_target - full_reconstruction
            reconstruction_norm = torch.linalg.vector_norm(full_reconstruction)
            prediction_errors[rank].append(
                float(torch.linalg.vector_norm(residual) / target_norm)
            )
            prediction_explained[rank].append(
                float(
                    (1.0 - residual.square().sum() / target_energy)
                    .clamp(-1.0, 1.0)
                )
            )
            prediction_residual_energy[rank].append(
                float(full_residual.square().sum() / raw_target_energy)
            )
            prediction_cosines[rank].append(
                float(
                    (raw_target * full_reconstruction).sum()
                    / (raw_target_norm * reconstruction_norm).clamp_min(1e-30)
                )
            )
            prediction_norm_ratios[rank].append(
                float(reconstruction_norm / raw_target_norm)
            )
            compression = _residual_compression_metrics(
                full_residual,
                matrix_shape=matrix_shape,
                max_elements=max_elements,
                block_size=residual_compression_block_size,
                scale_mode=residual_compression_scale_mode,
            )
            if compression is not None:
                residual_compression[rank].append(compression)
            approximation = _residual_approximation_metrics(
                raw_target,
                full_residual,
                matrix_shape=matrix_shape,
                max_elements=max_elements,
                block_size=residual_compression_block_size,
                scale_mode=residual_compression_scale_mode,
            )
            if approximation is not None:
                residual_approximations[rank].append(approximation)
                residual_sequences[rank].append(
                    (raw_target.clone(), full_residual.clone())
                )

    if not calibration_effective_ranks:
        return None

    def mean_or_none(values_for_rank: list[float]) -> float | None:
        return statistics.fmean(values_for_rank) if values_for_rank else None

    residual_spatial_compression, residual_approximation = _rolling_residual_summaries(
        ranks,
        residual_compression,
        residual_approximations,
        residual_sequences,
        max_elements=max_elements,
        block_size=residual_compression_block_size,
        scale_mode=residual_compression_scale_mode,
    )

    return {
        "metric_type": "rolling_trajectory_pca",
        "basis_mode": basis_mode,
        "samples": len(values),
        "features": int(matrix.shape[1]),
        "window": window,
        "residual_compression_block_size": residual_compression_block_size,
        "residual_compression_scale_mode": residual_compression_scale_mode,
        "predictions": len(values) - window,
        "valid_calibration_windows": len(calibration_effective_ranks),
        "calibration_rank_limit": rank_limit,
        "calibration_effective_rank": statistics.fmean(calibration_effective_ranks),
        "calibration_retained_energy": {
            str(rank): mean_or_none(calibration_retained[rank])
            for rank in ranks
        },
        "prediction_reconstruction_error": {
            str(rank): mean_or_none(prediction_errors[rank])
            for rank in ranks
        },
        "prediction_explained_variance": {
            str(rank): mean_or_none(prediction_explained[rank])
            for rank in ranks
        },
        "prediction_residual_energy_fraction": {
            str(rank): mean_or_none(prediction_residual_energy[rank])
            for rank in ranks
        },
        "prediction_cosine": {
            str(rank): mean_or_none(prediction_cosines[rank])
            for rank in ranks
        },
        "prediction_norm_ratio": {
            str(rank): mean_or_none(prediction_norm_ratios[rank])
            for rank in ranks
        },
        "residual_spatial_compression": residual_spatial_compression,
        "residual_approximation": residual_approximation,
    }
