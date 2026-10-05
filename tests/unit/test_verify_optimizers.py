import argparse
from typing import Any, cast

import pytest

from verify.optimizers import parse_args, parse_shape, parse_shapes, parse_ranks, run


def test_optimizer_verify_parsers_preserve_matrix_and_vector_shapes():
    assert parse_shape("64x32x3x3") == (64, 32, 3, 3)
    assert parse_shape("4096") == (4096,)
    assert parse_shapes("4x3,32x32") == ((4, 3), (32, 32))
    assert parse_ranks("1,8") == (1, 8)


@pytest.mark.parametrize("value", ["0x3", "3x0", "x3", "text"])
def test_optimizer_verify_rejects_invalid_shapes(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape(value)


def test_optimizer_verify_runs_cpu_matrix_fallback_cases():
    class Args:
        device = "cpu"
        dtype = "fp32"
        warmup = 0
        steps = 1
        seed = 0
        shapes = ((4, 3),)
        optimizers = ("CAME", "APOLLO", "APOLLO-CAME", "APOLLOMini")
        ranks = (1, 8)
        matrix_fallback = "auto"
        fallback_state_margin = 1.0
        fallback_min_savings_bytes = 0

    result = cast(dict[str, Any], run(Args()))

    assert result["status"] == "passed"
    assert result["device"] == "cpu"
    assert result["dtype"] == "fp32"
    assert result["fallback"]["small_matrix"] == "auto"
    for case in result["cases"].values():
        assert case["status"] == "passed"
        assert case["persistent_state_bytes"] > 0
        assert case["persistent_state_elements"] > 0
        if case["optimizer"] != "CAME":
            assert case["state_estimate_matches_actual"] is True
        else:
            assert case["state_estimate_matches_actual"] is None
        assert case["host_seconds_per_step"] >= 0.0
    assert result["cases"]["APOLLO@rank=8:4x3"]["backend"] == "came"
    assert result["cases"]["APOLLO-CAME@rank=8:4x3"]["backend"] == "came"
    assert result["cases"]["APOLLOMini@rank=1:4x3"]["backend"] == "apollo"


def test_optimizer_verify_counts_bfloat16_state_elements_without_byte_assumption():
    class Args:
        device = "cpu"
        dtype = "bf16"
        warmup = 1
        steps = 1
        seed = 0
        shapes = ((4, 3),)
        optimizers = ("CAME",)
        ranks = (1,)

    result = cast(dict[str, Any], run(Args()))
    case = result["cases"]["CAME:4x3"]

    assert result["status"] == "passed"
    assert case["persistent_state_elements"] == 27
    # The full-size exp_avg now follows the BF16 parameter dtype; CAME's
    # factored statistics and scalar RMS remain FP32.
    assert case["persistent_state_bytes"] == 82


def test_optimizer_verify_separates_projection_refresh_steps():
    class Args:
        device = "cpu"
        dtype = "fp32"
        warmup = 1
        steps = 2
        update_proj_gap = 1
        seed = 0
        shapes = ((16, 16),)
        optimizers = ("APOLLO",)
        ranks = (1,)
        matrix_fallback = "apollo"
        fallback_state_margin = 1.0
        fallback_min_savings_bytes = 0

    result = cast(dict[str, Any], run(Args()))
    case = result["cases"]["APOLLO@rank=1:16x16"]

    assert result["status"] == "passed"
    assert result["update_proj_gap"] == 1
    assert case["projection_refresh_steps"] == 2
    assert case["projection_refresh_active_steps"] == 2
    assert case["host_seconds_per_projection_refresh_step"] is not None
    assert case["host_seconds_per_projection_refresh_active_step"] is not None
    assert "cuda_seconds_per_projection_refresh_step" not in case


def test_optimizer_verify_parser_exposes_smooth_refresh_options():
    args = parse_args([
        "--projection-refresh-mode", "smooth",
        "--projection-refresh-window", "3",
        "--projection-refresh-mix", "linear",
        "--projection-refresh-state", "transport",
    ])

    assert args.projection_refresh_mode == "smooth"
    assert args.projection_refresh_window == 3
    assert args.projection_refresh_mix == "linear"
    assert args.projection_refresh_state == "transport"


def test_optimizer_verify_accepts_stochastic_and_orthogonal_refreshes():
    class Args:
        device = "cpu"
        dtype = "fp32"
        warmup = 1
        steps = 2
        update_proj_gap = 2
        projection_refresh_mode = "smooth"
        projection_refresh_window = 3
        projection_refresh_mix = "stochastic"
        projection_refresh_state = "transport"
        orthogonal_refresh_rate = 0.05
        seed = 13
        shapes = ((16, 16),)
        optimizers = ("APOLLO",)
        ranks = (1,)
        matrix_fallback = "apollo"
        fallback_state_margin = 1.0
        fallback_min_savings_bytes = 0

    result = cast(dict[str, Any], run(Args()))
    case = result["cases"]["APOLLO@rank=1:16x16"]

    assert result["status"] == "passed"
    assert result["projection_refresh"]["mix"] == "stochastic"
    assert result["orthogonal_refresh_rate"] == 0.05
    assert case["orthogonal_refresh_steps"] == 3
    assert case["state_estimate_matches_actual"] is True


@pytest.mark.parametrize("optimizer", ["APOLLO", "APOLLO-CAME", "APOLLOMini"])
def test_optimizer_verify_reports_smooth_refresh_state_and_estimate(optimizer):
    class Args:
        device = "cpu"
        dtype = "fp32"
        warmup = 1
        steps = 2
        update_proj_gap = 2
        projection_refresh_mode = "smooth"
        projection_refresh_window = 3
        projection_refresh_mix = "linear"
        projection_refresh_state = "transport"
        seed = 0
        shapes = ((16, 16),)
        optimizers = (optimizer,)
        ranks = (1,)
        matrix_fallback = "apollo"
        fallback_state_margin = 1.0
        fallback_min_savings_bytes = 0

    result = cast(dict[str, Any], run(Args()))
    case = next(iter(result["cases"].values()))

    assert result["status"] == "passed"
    assert result["projection_refresh"] == {
        "mode": "smooth",
        "interval": 2,
        "window": 3,
        "mix": "linear",
    }
    assert result["projection_refresh_state"] == "transport"
    assert case["projection_refresh_mode"] == "smooth"
    assert case["projection_refresh_window"] == 3
    assert case["projection_refresh_mix"] == "linear"
    assert case["projection_refresh_state"] == "transport"
    assert case["projection_refresh_steps"] == 1
    assert case["projection_refresh_active_steps"] == 2
    assert case["state_estimate_matches_actual"] is True


def test_optimizer_verify_reports_transient_smooth_state_peak():
    class Args:
        device = "cpu"
        dtype = "fp32"
        warmup = 1
        steps = 2
        update_proj_gap = 2
        projection_refresh_mode = "smooth"
        projection_refresh_window = 2
        projection_refresh_mix = "smoothstep"
        projection_refresh_state = "reset"
        seed = 0
        shapes = ((16, 16),)
        optimizers = ("APOLLO",)
        ranks = (1,)
        matrix_fallback = "apollo"
        fallback_state_margin = 1.0
        fallback_min_savings_bytes = 0

    result = cast(dict[str, Any], run(Args()))
    case = result["cases"]["APOLLO@rank=1:16x16"]

    assert result["status"] == "passed"
    assert case["persistent_state_bytes"] == 192
    assert case["peak_persistent_state_bytes"] == 384
    assert case["peak_persistent_state_elements"] == 96
