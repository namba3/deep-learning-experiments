"""Offline residual compression and reconstruction diagnostics for Text-LM optimizer probes."""

from __future__ import annotations

import math
import statistics
from time import perf_counter

import torch


def _matrix_rank_metrics(
    tensor: torch.Tensor,
    *,
    max_elements: int,
) -> dict[str, object] | None:
    """Return compact singular-spectrum metrics for one matrix-shaped tensor."""
    if tensor.ndim < 2 or tensor.numel() > max_elements:
        return None
    matrix = tensor.detach().float().reshape(tensor.shape[0], -1)
    if min(matrix.shape) < 2:
        return None
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    total_energy = energy.sum()
    if not torch.isfinite(total_energy) or total_energy <= 0:
        return {
            "shape": list(matrix.shape),
            "numel": int(matrix.numel()),
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "rank_90": 0,
            "rank_95": 0,
            "rank_99": 0,
            "retained_energy": {str(rank): 0.0 for rank in (1, 2, 4, 8, 16)},
        }
    probabilities = energy / total_energy
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    cumulative = energy.cumsum(0) / total_energy

    def threshold_rank(value: float) -> int:
        return int(torch.searchsorted(cumulative, value).item()) + 1

    return {
        "shape": list(matrix.shape),
        "numel": int(matrix.numel()),
        "effective_rank": float(torch.exp(entropy)),
        "participation_ratio": float(1.0 / probabilities.square().sum()),
        "rank_90": threshold_rank(0.90),
        "rank_95": threshold_rank(0.95),
        "rank_99": threshold_rank(0.99),
        "retained_energy": {
            str(rank): float(cumulative[min(rank, cumulative.numel()) - 1])
            for rank in (1, 2, 4, 8, 16)
        },
    }

def _quantization_threshold(
    block: torch.Tensor,
    *,
    scale_mode: str,
) -> torch.Tensor:
    """Return the symmetric quantizer clipping threshold for one block."""
    absolute = block.abs()
    if scale_mode == "max_abs":
        return absolute.max()
    if scale_mode == "percentile_99_9":
        return torch.quantile(absolute, 0.999)
    if scale_mode == "rms_3sigma":
        return 3.0 * torch.sqrt(block.square().mean())
    raise ValueError(f"unsupported residual quantization scale mode: {scale_mode}")

def _blockwise_dequantize(
    values: torch.Tensor,
    *,
    block_size: int,
    quantization_bits: int,
    scale_mode: str,
) -> torch.Tensor:
    """Symmetrically quantize and dequantize a flat tensor block by block."""
    quantization_max = 2 ** (quantization_bits - 1) - 1
    dequantized = torch.empty_like(values)
    for start in range(0, values.numel(), block_size):
        end = min(start + block_size, values.numel())
        block = values[start:end]
        threshold = _quantization_threshold(block, scale_mode=scale_mode)
        if threshold <= 0:
            dequantized[start:end] = 0.0
        else:
            scale = threshold / quantization_max
            dequantized[start:end] = (
                torch.round(block / scale).clamp(
                    -quantization_max, quantization_max,
                ) * scale
            )
    return dequantized

def _residual_error_feedback_metrics(
    targets: list[torch.Tensor],
    residuals: list[torch.Tensor],
    *,
    max_elements: int,
    block_size: int,
    scale_mode: str,
    quantization_bits: int = 4,
) -> dict[str, object] | None:
    """Measure temporal error feedback for an offline residual quantizer."""
    if (
        not targets
        or len(targets) != len(residuals)
        or block_size <= 0
        or scale_mode not in ("max_abs", "percentile_99_9", "rms_3sigma")
        or quantization_bits not in (4, 8)
    ):
        return None
    target_values = [target.detach().float().cpu().reshape(-1) for target in targets]
    residual_values = [
        residual.detach().float().cpu().reshape(-1) for residual in residuals
    ]
    if (
        any(value.numel() != target_values[0].numel() for value in target_values)
        or any(value.numel() != target_values[0].numel() for value in residual_values)
        or target_values[0].numel() > max_elements
        or not all(torch.isfinite(value).all() for value in target_values)
        or not all(torch.isfinite(value).all() for value in residual_values)
    ):
        return None

    feedback = torch.zeros_like(residual_values[0])
    target_norms: list[float] = []
    residual_errors: list[float] = []
    update_errors: list[float] = []
    update_cosines: list[float] = []
    update_norm_ratios: list[float] = []
    feedback_ratios: list[float] = []
    started = perf_counter()
    for target, residual in zip(target_values, residual_values):
        quantized_input = residual + feedback
        dequantized = _blockwise_dequantize(
            quantized_input,
            block_size=block_size,
            quantization_bits=quantization_bits,
            scale_mode=scale_mode,
        )
        feedback = quantized_input - dequantized
        compressed_update = target - residual + dequantized
        target_norm = torch.linalg.vector_norm(target).clamp_min(1e-30)
        compressed_norm = torch.linalg.vector_norm(compressed_update).clamp_min(1e-30)
        target_norms.append(float(target_norm))
        residual_errors.append(
            float(torch.linalg.vector_norm(residual - dequantized) / target_norm)
        )
        update_errors.append(
            float(torch.linalg.vector_norm(target - compressed_update) / target_norm)
        )
        update_cosines.append(
            float(target.dot(compressed_update) / (target_norm * compressed_norm))
        )
        update_norm_ratios.append(float(compressed_norm / target_norm))
        feedback_ratios.append(
            float(torch.linalg.vector_norm(feedback) / target_norm)
        )
    decode_milliseconds = (perf_counter() - started) * 1000.0
    elements = target_values[0].numel()
    scale_count = math.ceil(elements / block_size)

    return {
        "samples": len(target_values),
        "block_size": block_size,
        "scale_mode": scale_mode,
        "quantization_bits": quantization_bits,
        "storage_bytes_int4": (
            math.ceil(elements / 2) + scale_count * 4
            if quantization_bits == 4
            else None
        ),
        "storage_ratio_to_target_bf16": (
            (math.ceil(elements / 2) + scale_count * 4)
            / max(1, elements * 2)
            if quantization_bits == 4
            else None
        ),
        "feedback_storage_bytes_fp32": elements * 4,
        "feedback_storage_ratio_to_target_bf16": 2.0,
        "decode_milliseconds": decode_milliseconds,
        "residual_relative_error": statistics.fmean(residual_errors),
        "update_relative_error": statistics.fmean(update_errors),
        "update_cosine": statistics.fmean(update_cosines),
        "update_norm_ratio": statistics.fmean(update_norm_ratios),
        "feedback_norm_ratio_to_target": statistics.fmean(feedback_ratios),
        "final_feedback_norm_ratio_to_target": feedback_ratios[-1],
    }

def _residual_compression_metrics(
    residual: torch.Tensor,
    *,
    matrix_shape: tuple[int, ...] | None,
    max_elements: int,
    block_size: int = 256,
    scale_mode: str = "max_abs",
) -> dict[str, object] | None:
    """Measure spatial rank and simple residual compression surrogates."""
    if (
        matrix_shape is None
        or len(matrix_shape) < 2
        or residual.numel() > max_elements
        or block_size <= 0
        or scale_mode not in ("max_abs", "percentile_99_9", "rms_3sigma")
    ):
        return None
    matrix = residual.detach().float().cpu().reshape(matrix_shape[0], -1)
    if min(matrix.shape) < 2 or not torch.isfinite(matrix).all():
        return None
    denominator = matrix.norm().clamp_min(1e-30)
    flat = matrix.reshape(-1)
    max_abs = flat.abs().max()

    if max_abs <= 0:
        global_int8_error = 0.0
        blockwise_int8_error = 0.0
    else:
        global_scale = max_abs / 127.0
        global_dequantized = (
            torch.round(flat / global_scale).clamp(-127, 127) * global_scale
        )
        global_int8_error = float(
            (flat - global_dequantized).norm() / denominator
        )
        blockwise_dequantized = torch.empty_like(flat)
        for start in range(0, flat.numel(), block_size):
            end = min(start + block_size, flat.numel())
            block = flat[start:end]
            threshold = _quantization_threshold(
                block, scale_mode=scale_mode,
            )
            if threshold <= 0:
                blockwise_dequantized[start:end] = 0.0
            else:
                block_scale = threshold / 127.0
                blockwise_dequantized[start:end] = (
                    torch.round(block / block_scale).clamp(-127, 127)
                    * block_scale
                )
        blockwise_int8_error = float(
            (flat - blockwise_dequantized).norm() / denominator
        )

    scalar_approximation = torch.empty_like(flat)
    for start in range(0, flat.numel(), block_size):
        end = min(start + block_size, flat.numel())
        scalar_approximation[start:end] = flat[start:end].mean()
    blockwise_scalar_error = float(
        (flat - scalar_approximation).norm() / denominator
    )

    diagonal = torch.zeros_like(matrix)
    diagonal_size = min(matrix.shape)
    indices = torch.arange(diagonal_size)
    diagonal[indices, indices] = matrix[indices, indices]
    diagonal_error = float((matrix - diagonal).norm() / denominator)
    rank_metrics = _matrix_rank_metrics(matrix, max_elements=max_elements)
    if rank_metrics is None:
        return None
    return {
        "block_size": block_size,
        "scale_mode": scale_mode,
        "spatial_effective_rank": rank_metrics["effective_rank"],
        "spatial_rank_95": rank_metrics["rank_95"],
        "spatial_retained_energy": rank_metrics["retained_energy"],
        "global_int8_relative_error": global_int8_error,
        "blockwise_int8_relative_error": blockwise_int8_error,
        "blockwise_scalar_relative_error": blockwise_scalar_error,
        "diagonal_relative_error": diagonal_error,
    }

def _residual_approximation_metrics(
    target: torch.Tensor,
    residual: torch.Tensor,
    *,
    matrix_shape: tuple[int, ...] | None,
    max_elements: int,
    block_size: int = 256,
    scale_mode: str = "max_abs",
    factor_ranks: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> dict[str, object] | None:
    """Estimate residual-factor and INT8 decode/compression trade-offs.

    This is an offline reconstruction diagnostic.  It does not replace the
    optimizer update or allocate persistent optimizer state.  Factor storage
    assumes BF16 factors, while blockwise INT8 storage uses one FP32 scale per
    block; both assumptions are reported explicitly in the result.
    """
    if (
        matrix_shape is None
        or len(matrix_shape) < 2
        or target.numel() > max_elements
        or residual.numel() != target.numel()
        or block_size <= 0
        or scale_mode not in ("max_abs", "percentile_99_9", "rms_3sigma")
        or not factor_ranks
    ):
        return None
    residual_flat = residual.detach().float().cpu().reshape(-1)
    matrix = residual_flat.reshape(matrix_shape[0], -1)
    target_flat = target.detach().float().cpu().reshape(-1)
    if (
        min(matrix.shape) < 2
        or not torch.isfinite(matrix).all()
        or not torch.isfinite(target_flat).all()
    ):
        return None

    rows, columns = matrix.shape
    target_norm = torch.linalg.vector_norm(target_flat).clamp_min(1e-30)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    low_rank: dict[str, dict[str, object]] = {}
    low_rank_reconstructions: dict[int, torch.Tensor] = {}
    for requested_rank in factor_ranks:
        rank = min(int(requested_rank), int(singular_values.numel()))
        if rank <= 0:
            continue
        left_factor = u[:, :rank] * singular_values[:rank]
        started = perf_counter()
        reconstruction = left_factor.matmul(vh[:rank, :]).reshape(-1)
        low_rank_reconstructions[int(requested_rank)] = reconstruction
        decode_seconds = perf_counter() - started
        compressed_update = target_flat - residual_flat + reconstruction
        compressed_norm = torch.linalg.vector_norm(compressed_update)
        low_rank[str(requested_rank)] = {
            "rank_used": rank,
            "storage_elements": int((rows + columns) * rank),
            "storage_bytes_bf16": int((rows + columns) * rank * 2),
            "storage_bytes_fp32": int((rows + columns) * rank * 4),
            "storage_ratio_to_target_bf16": float(
                (rows + columns) * rank / max(1, target_flat.numel())
            ),
            "decode_milliseconds": decode_seconds * 1000.0,
            "residual_relative_error": float(
                torch.linalg.vector_norm(residual_flat - reconstruction)
                / torch.linalg.vector_norm(matrix).clamp_min(1e-30)
            ),
            "update_relative_error": float(
                torch.linalg.vector_norm(target_flat - compressed_update) / target_norm
            ),
            "update_cosine": float(
                target_flat.dot(compressed_update)
                / (target_norm * compressed_norm).clamp_min(1e-30)
            ),
            "update_norm_ratio": float(compressed_norm / target_norm),
        }

    started = perf_counter()
    dequantized = torch.empty_like(target_flat)
    scale_count = 0
    for start in range(0, target_flat.numel(), block_size):
        end = min(start + block_size, target_flat.numel())
        block = matrix.reshape(-1)[start:end]
        threshold = _quantization_threshold(block, scale_mode=scale_mode)
        scale_count += 1
        if threshold <= 0:
            dequantized[start:end] = 0.0
        else:
            scale = threshold / 127.0
            dequantized[start:end] = (
                torch.round(block / scale).clamp(-127, 127) * scale
            )
    decode_seconds = perf_counter() - started
    compressed_update = target_flat - residual_flat + dequantized
    compressed_norm = torch.linalg.vector_norm(compressed_update)
    blockwise_int8 = {
        "storage_elements": int(target_flat.numel() + scale_count),
        "storage_bytes_int8": int(target_flat.numel() + scale_count * 4),
        "storage_ratio_to_target_bf16": float(
            (target_flat.numel() + scale_count * 4)
            / max(1, target_flat.numel() * 2)
        ),
        "block_size": block_size,
        "scale_mode": scale_mode,
        "scale_dtype": "fp32",
        "decode_milliseconds": decode_seconds * 1000.0,
        "residual_relative_error": float(
            torch.linalg.vector_norm(residual_flat - dequantized)
            / torch.linalg.vector_norm(matrix).clamp_min(1e-30)
        ),
        "update_relative_error": float(
            torch.linalg.vector_norm(target_flat - compressed_update) / target_norm
        ),
        "update_cosine": float(
            target_flat.dot(compressed_update)
            / (target_norm * compressed_norm).clamp_min(1e-30)
        ),
        "update_norm_ratio": float(compressed_norm / target_norm),
    }
    started = perf_counter()
    int4_dequantized = torch.empty_like(target_flat)
    int4_scale_count = 0
    for start in range(0, target_flat.numel(), block_size):
        end = min(start + block_size, target_flat.numel())
        block = matrix.reshape(-1)[start:end]
        threshold = _quantization_threshold(block, scale_mode=scale_mode)
        int4_scale_count += 1
        if threshold <= 0:
            int4_dequantized[start:end] = 0.0
        else:
            scale = threshold / 7.0
            int4_dequantized[start:end] = (
                torch.round(block / scale).clamp(-7, 7) * scale
            )
    decode_seconds = perf_counter() - started
    compressed_update = target_flat - residual_flat + int4_dequantized
    compressed_norm = torch.linalg.vector_norm(compressed_update)
    blockwise_int4 = {
        "storage_elements": int(
            math.ceil(target_flat.numel() / 2) + int4_scale_count
        ),
        "storage_bytes_int4": int(
            math.ceil(target_flat.numel() / 2) + int4_scale_count * 4
        ),
        "storage_ratio_to_target_bf16": float(
            (math.ceil(target_flat.numel() / 2) + int4_scale_count * 4)
            / max(1, target_flat.numel() * 2)
        ),
        "block_size": block_size,
        "scale_mode": scale_mode,
        "quantization_bits": 4,
        "scale_dtype": "fp32",
        "decode_milliseconds": decode_seconds * 1000.0,
        "residual_relative_error": float(
            torch.linalg.vector_norm(residual_flat - int4_dequantized)
            / torch.linalg.vector_norm(matrix).clamp_min(1e-30)
        ),
        "update_relative_error": float(
            torch.linalg.vector_norm(target_flat - compressed_update) / target_norm
        ),
        "update_cosine": float(
            target_flat.dot(compressed_update)
            / (target_norm * compressed_norm).clamp_min(1e-30)
        ),
        "update_norm_ratio": float(compressed_norm / target_norm),
    }
    low_rank_plus_int8: dict[str, dict[str, object]] = {}
    for requested_rank in factor_ranks:
        reconstruction = low_rank_reconstructions.get(int(requested_rank))
        if reconstruction is None:
            continue
        remainder = residual_flat - reconstruction
        started = perf_counter()
        remainder_dequantized = torch.empty_like(remainder)
        scale_count = 0
        for start in range(0, remainder.numel(), block_size):
            end = min(start + block_size, remainder.numel())
            block = remainder[start:end]
            threshold = _quantization_threshold(
                block, scale_mode=scale_mode,
            )
            scale_count += 1
            if threshold <= 0:
                remainder_dequantized[start:end] = 0.0
            else:
                scale = threshold / 127.0
                remainder_dequantized[start:end] = (
                    torch.round(block / scale).clamp(-127, 127) * scale
                )
        decode_seconds = perf_counter() - started
        compressed_update = (
            target_flat - residual_flat + reconstruction + remainder_dequantized
        )
        compressed_norm = torch.linalg.vector_norm(compressed_update)
        low_rank_bytes = (rows + columns) * min(
            int(requested_rank), int(singular_values.numel())
        ) * 2
        low_rank_plus_int8[str(requested_rank)] = {
            "rank_used": min(int(requested_rank), int(singular_values.numel())),
            "storage_bytes_bf16_factor_int8_remainder": int(
                low_rank_bytes + remainder.numel() + scale_count * 4
            ),
            "storage_ratio_to_target_bf16": float(
                (low_rank_bytes + remainder.numel() + scale_count * 4)
                / max(1, target_flat.numel() * 2)
            ),
            "block_size": block_size,
            "scale_mode": scale_mode,
            "scale_dtype": "fp32",
            "decode_milliseconds": decode_seconds * 1000.0,
            "residual_relative_error": float(
                torch.linalg.vector_norm(remainder - remainder_dequantized)
                / torch.linalg.vector_norm(residual_flat).clamp_min(1e-30)
            ),
            "update_relative_error": float(
                torch.linalg.vector_norm(target_flat - compressed_update) / target_norm
            ),
            "update_cosine": float(
                target_flat.dot(compressed_update)
                / (target_norm * compressed_norm).clamp_min(1e-30)
            ),
            "update_norm_ratio": float(compressed_norm / target_norm),
        }
    return {
        "target_elements": int(target_flat.numel()),
        "target_storage_bytes_bf16": int(target_flat.numel() * 2),
        "low_rank_factor": low_rank,
        "blockwise_int8": blockwise_int8,
        "blockwise_int4": blockwise_int4,
        "low_rank_plus_int8": low_rank_plus_int8,
    }
