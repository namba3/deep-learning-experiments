from typing import Any, cast
import pytest
import torch
from verify.text_lm_optimizer_convergence import _causal_trajectory_pca_metrics, _loss_second_difference_metrics, _residual_compression_metrics, _residual_approximation_metrics, _residual_error_feedback_metrics, _trajectory_curvature_metrics, _trajectory_position_curvature_metrics, _trajectory_pca_metrics, _rolling_trajectory_pca_metrics, _truncated_svd_reconstructions, parse_args, _update_trajectory_before_snapshot, _update_trajectory_snapshot


def test_text_lm_optimizer_probe_loss_curvature_is_enabled_with_trajectory_diagnostics():
    args = parse_args([
        "--record-trajectory-curvature",
        "--eval-interval", "25",
    ])

    assert args.record_trajectory_curvature is True
    assert args.eval_interval == 25

def test_loss_second_difference_metrics_reports_scalar_roughness():
    metrics = cast(dict[str, Any] | None, _loss_second_difference_metrics([1.0, 2.0, 4.0, 7.0]))

    assert metrics is not None
    assert metrics["metric_type"] == "loss_second_difference"
    assert metrics["samples"] == 4
    assert metrics["second_difference_samples"] == 2
    assert metrics["second_difference_mean"] == 1.0
    assert metrics["second_difference_abs_mean"] == 1.0

def test_loss_second_difference_metrics_requires_three_samples():
    assert _loss_second_difference_metrics([1.0, 2.0]) is None

def test_trajectory_pca_metrics_detects_one_temporal_direction():
    values = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0]),
        torch.tensor([3.0, 0.0, 0.0]),
    ]

    metrics = cast(dict[str, Any] | None, _trajectory_pca_metrics(values, max_elements=10))

    assert metrics is not None
    assert metrics["metric_type"] == "trajectory_pca"
    assert metrics["samples"] == 3
    assert metrics["effective_rank"] == 1.0
    assert metrics["rank_90"] == 1
    assert metrics["retained_energy"]["1"] == 1.0

def test_causal_trajectory_pca_reconstructs_future_in_calibration_subspace():
    values = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([-1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([-1.0, -1.0, 0.0, 0.0]),
        torch.tensor([0.5, 0.5, 0.0, 0.0]),
    ]

    metrics = cast(dict[str, Any] | None, _causal_trajectory_pca_metrics(
        values, max_elements=100, calibration_fraction=0.5,
    ))

    assert metrics is not None
    assert metrics["calibration_samples"] == 3
    assert metrics["future_samples"] == 3
    assert metrics["calibration_rank_limit"] == 2
    explained = cast(dict[str, float | None], metrics["future_explained_variance"])
    assert explained["2"] is not None
    assert explained["2"] > 0.99
    assert explained["1"] is not None
    assert explained["1"] < explained["2"]

def test_rolling_trajectory_pca_reconstructs_next_update_from_past_window():
    values = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([4.0, 0.0]),
        torch.tensor([5.0, 0.0]),
    ]

    metrics = cast(dict[str, Any] | None, _rolling_trajectory_pca_metrics(
        values, max_elements=100, window=3,
    ))

    assert metrics is not None
    assert metrics["metric_type"] == "rolling_trajectory_pca"
    assert metrics["window"] == 3
    assert metrics["predictions"] == 2
    assert metrics["calibration_rank_limit"] == 2
    explained = cast(
        dict[str, float | None], metrics["prediction_explained_variance"]
    )
    assert explained["1"] == pytest.approx(1.0)
    residual_energy = cast(
        dict[str, float | None], metrics["prediction_residual_energy_fraction"]
    )
    assert residual_energy["1"] == pytest.approx(0.0)
    cosine = cast(dict[str, float | None], metrics["prediction_cosine"])
    assert cosine["1"] == pytest.approx(1.0)
    norm_ratio = cast(dict[str, float | None], metrics["prediction_norm_ratio"])
    assert norm_ratio["1"] == pytest.approx(1.0)
    residual_compression = cast(
        dict[str, object], metrics["residual_spatial_compression"]
    )
    rank_one_compression = cast(dict[str, object], residual_compression["1"])
    assert rank_one_compression["samples"] == 0
    assert rank_one_compression["global_int8_relative_error"] is None

    fixed_metrics = _rolling_trajectory_pca_metrics(
        values, max_elements=100, window=3, basis_mode="fixed",
    )
    assert fixed_metrics is not None
    assert fixed_metrics["basis_mode"] == "fixed"
    assert fixed_metrics["predictions"] == 2

def test_residual_compression_metrics_reports_spatial_and_quantization_signals():
    residual = torch.zeros(4, 4)
    residual[:, 0] = 1.0

    metrics = cast(dict[str, Any] | None, _residual_compression_metrics(
        residual, matrix_shape=(4, 4), max_elements=100, block_size=4,
    ))

    assert metrics is not None
    assert metrics["block_size"] == 4
    assert metrics["spatial_effective_rank"] == 1.0
    assert metrics["spatial_rank_95"] == 1
    assert metrics["global_int8_relative_error"] == pytest.approx(0.0)
    assert metrics["blockwise_int8_relative_error"] == pytest.approx(0.0)
    assert metrics["diagonal_relative_error"] == pytest.approx((3.0 / 4.0) ** 0.5)

def test_residual_quantization_scale_modes_are_supported():
    residual = torch.tensor([1.0, 2.0, 3.0, 100.0]).reshape(2, 2)

    percentile = _residual_compression_metrics(
        residual,
        matrix_shape=(2, 2),
        max_elements=100,
        block_size=4,
        scale_mode="percentile_99_9",
    )
    rms = _residual_compression_metrics(
        residual,
        matrix_shape=(2, 2),
        max_elements=100,
        block_size=4,
        scale_mode="rms_3sigma",
    )

    assert percentile is not None
    assert rms is not None
    assert percentile["scale_mode"] == "percentile_99_9"
    assert rms["scale_mode"] == "rms_3sigma"

def test_residual_error_feedback_metrics_reports_temporal_quantization():
    targets = [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([1.1, 1.9, 3.2, 3.8]),
    ]
    residuals = [
        torch.tensor([0.1, -0.2, 0.3, -0.4]),
        torch.tensor([0.2, -0.1, 0.4, -0.3]),
    ]

    metrics = cast(dict[str, Any] | None, _residual_error_feedback_metrics(
        targets,
        residuals,
        max_elements=100,
        block_size=4,
        scale_mode="max_abs",
    ))

    assert metrics is not None
    assert metrics["samples"] == 2
    assert metrics["quantization_bits"] == 4
    assert metrics["final_feedback_norm_ratio_to_target"] >= 0.0
    assert metrics["update_cosine"] > 0.99

def test_residual_approximation_metrics_reports_capacity_and_update_quality():
    residual = torch.zeros(4, 4)
    residual[:, 0] = torch.arange(1.0, 5.0)
    target = residual + 1.0

    metrics = cast(dict[str, Any] | None, _residual_approximation_metrics(
        target,
        residual,
        matrix_shape=(4, 4),
        max_elements=100,
        block_size=4,
        factor_ranks=(1, 2),
    ))

    assert metrics is not None
    factor = cast(dict[str, Any], metrics["low_rank_factor"])
    rank_one = cast(dict[str, Any], factor["1"])
    assert rank_one["rank_used"] == 1
    assert rank_one["storage_bytes_bf16"] == 16
    assert rank_one["residual_relative_error"] == pytest.approx(0.0, abs=1e-6)
    assert rank_one["update_cosine"] > 0.99
    int8 = cast(dict[str, Any], metrics["blockwise_int8"])
    assert int8["storage_bytes_int8"] == 32
    assert int8["block_size"] == 4
    assert int8["update_cosine"] > 0.99
    int4 = cast(dict[str, Any], metrics["blockwise_int4"])
    assert int4["storage_bytes_int4"] == 24
    assert int4["quantization_bits"] == 4
    hybrid = cast(dict[str, dict[str, Any]], metrics["low_rank_plus_int8"])
    assert hybrid["1"]["storage_ratio_to_target_bf16"] > 0.5
    assert hybrid["1"]["update_cosine"] > 0.99

def test_trajectory_curvature_metrics_reports_direction_change():
    values = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ]

    metrics = cast(dict[str, Any] | None, _trajectory_curvature_metrics(values, max_elements=100))

    assert metrics is not None
    assert metrics["metric_type"] == "trajectory_curvature"
    assert metrics["valid_turns"] == 2
    assert metrics["path_length"] == 3.0
    assert metrics["turning_angle_mean"] > 0.7
    assert metrics["turning_angle_p95"] > 1.4
    assert metrics["roughness"] == 2.0

def test_trajectory_position_curvature_uses_position_differences():
    positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([2.0, 1.0]),
    ]

    metrics = cast(dict[str, Any] | None, _trajectory_position_curvature_metrics(
        positions, max_elements=100,
    ))

    assert metrics is not None
    assert metrics["metric_type"] == "trajectory_position_curvature"
    assert metrics["position_samples"] == 4
    assert metrics["samples"] == 3
    assert metrics["valid_turns"] == 2
    assert metrics["turning_angle_mean"] > 0.7

def test_update_trajectory_snapshot_returns_realized_parameter_delta():
    model = torch.nn.Linear(2, 3, bias=False)
    before = _update_trajectory_before_snapshot(
        model,
        max_elements=100,
        max_tensors=1,
        parameter_substring=None,
    )
    with torch.no_grad():
        model.weight.add_(2.0)

    snapshots = _update_trajectory_snapshot(model, before)

    assert len(snapshots) == 1
    name, source, value = snapshots[0]
    assert name == "weight"
    assert source == "effective_update"
    assert torch.allclose(value, torch.full((6,), 2.0))

def test_truncated_svd_reconstruction_reports_rank_one_matrix_error():
    tensor = torch.zeros(4, 3)
    tensor[:, 0] = 1.0
    tensor[:, 1] = 2.0

    reconstructions = _truncated_svd_reconstructions(
        tensor, (1, 2), max_elements=100,
    )

    assert [rank for rank, _, _ in reconstructions] == [1, 2]
    assert reconstructions[0][2] < 1e-6
    assert reconstructions[1][2] < 1e-6
