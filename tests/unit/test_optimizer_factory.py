import argparse
from types import SimpleNamespace

import pytest
import torch

from optimizers.factory import (
    CORE_OPTIMIZER_CHOICES,
    add_optimizer_argument,
    build_optimizer,
    normalize_optimizer_name,
)


def test_common_optimizer_factory_builds_and_steps_all_core_choices():
    for name in CORE_OPTIMIZER_CHOICES:
        parameter = torch.nn.Parameter(torch.ones(4, 4))
        optimizer = build_optimizer(
            name,
            [parameter],
            args=SimpleNamespace(auto_schedule=False),
            lr=1e-3,
            weight_decay=0.0,
        )
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert torch.isfinite(parameter).all()


def test_auto_schedule_is_a_mode_separate_from_core_name():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = build_optimizer(
        "APOLLO-CAME",
        [parameter],
        args=SimpleNamespace(auto_schedule=True),
        lr=1e-3,
        weight_decay=0.0,
    )
    assert type(optimizer).__name__ == "APOLLOCAMEAutoSchedule"


def test_factory_passes_size_aware_apollo_fallback_policy():
    parameter = torch.nn.Parameter(torch.ones(4, 3))
    optimizer = build_optimizer(
        "APOLLO",
        [parameter],
        args=SimpleNamespace(
            auto_schedule=False,
            apollo_fallback="came",
            apollo_matrix_fallback="auto",
            apollo_fallback_state_margin=1.0,
            apollo_fallback_min_savings_bytes=0,
        ),
        lr=1e-3,
        weight_decay=0.0,
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert optimizer.state[parameter]["backend"] == "came"


def test_factory_maps_apollo_refresh_mode_to_shared_policy():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = build_optimizer(
        "APOLLO",
        [parameter],
        args=SimpleNamespace(
            apollo_projection_refresh_mode="none",
            apollo_update_proj_gap=10,
        ),
        lr=1e-3,
        weight_decay=0.0,
    )

    assert optimizer.param_groups[0]["projection_refresh"] == {
        "mode": "none",
        "interval": 10,
        "window": 0,
        "mix": "smoothstep",
    }


def test_factory_maps_apollo_smooth_refresh_options_to_shared_policy():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = build_optimizer(
        "APOLLO",
        [parameter],
        args=SimpleNamespace(
            apollo_projection_refresh_mode="smooth",
            apollo_projection_refresh_window=4,
            apollo_projection_refresh_mix="linear",
            apollo_update_proj_gap=10,
        ),
        lr=1e-3,
        weight_decay=0.0,
    )

    assert optimizer.param_groups[0]["projection_refresh"] == {
        "mode": "smooth",
        "interval": 10,
        "window": 4,
        "mix": "linear",
    }


def test_factory_maps_apollo_stochastic_and_orthogonal_refresh_options():
    parameter = torch.nn.Parameter(torch.ones(6, 4))
    optimizer = build_optimizer(
        "APOLLO",
        [parameter],
        args=SimpleNamespace(
            apollo_projection_refresh_mode="smooth",
            apollo_projection_refresh_window=4,
            apollo_projection_refresh_mix="stochastic",
            apollo_update_proj_gap=10,
            apollo_orthogonal_refresh_rate=0.05,
            seed=13,
        ),
        lr=1e-3,
        weight_decay=0.0,
    )

    assert optimizer.param_groups[0]["projection_refresh"]["mix"] == "stochastic"
    assert optimizer.param_groups[0]["orthogonal_refresh"] == {
        "rate": 0.05, "seed": 13,
    }


def test_factory_maps_loss_directed_orthogonal_refresh_option():
    parameter = torch.nn.Parameter(torch.ones(6, 4))
    optimizer = build_optimizer(
        "APOLLO",
        [parameter],
        args=SimpleNamespace(
            apollo_orthogonal_refresh_rate=0.01,
            apollo_orthogonal_refresh_direction="loss_directed",
            seed=5,
        ),
        lr=1e-3,
        weight_decay=0.0,
    )

    assert optimizer.param_groups[0]["orthogonal_refresh"] == {
        "rate": 0.01, "seed": 5, "direction": "loss_directed",
    }


def test_factory_maps_update_norm_variance_cap_option():
    parameter = torch.nn.Parameter(torch.ones(6, 4))
    optimizer = build_optimizer(
        "APOLLO",
        [parameter],
        args=SimpleNamespace(
            apollo_update_norm_variance_cap=0.001,
        ),
        lr=1e-3,
        weight_decay=0.0,
    )

    assert optimizer.param_groups[0]["update_norm_variance_cap"] == 0.001


def test_common_parser_hides_legacy_choices_but_accepts_them():
    parser = argparse.ArgumentParser()
    add_optimizer_argument(parser)
    assert parser.parse_args([]).optimizer == "AdamW"
    with pytest.warns(DeprecationWarning):
        assert parser.parse_args(["--optimizer", "RAdamSF"]).optimizer == "RAdamSF"


def test_factory_passes_adamw_schedulefree_backend():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = build_optimizer(
        "AdamW-SF",
        [parameter],
        args=SimpleNamespace(
            adamw_sf_backend="torch",
            adamw_lrsf_beta1=0.9,
            adamw_lrsf_beta2=0.999,
        ),
        lr=1e-3,
        weight_decay=0.0,
    )

    assert optimizer.backend == "torch"


def test_common_parser_can_expose_apollo_mini_explicitly():
    parser = argparse.ArgumentParser()
    add_optimizer_argument(parser, include_apollo_mini=True)

    assert parser.parse_args(["--optimizer", "APOLLO-Mini"]).optimizer == "APOLLO-Mini"


def test_common_parser_can_expose_came_lrsf_explicitly():
    parser = argparse.ArgumentParser()
    add_optimizer_argument(parser, include_came_lrsf=True)

    assert parser.parse_args(["--optimizer", "CAME-LRSF"]).optimizer == "CAME-LRSF"


def test_common_parser_can_expose_came_sf_explicitly():
    parser = argparse.ArgumentParser()
    add_optimizer_argument(parser, include_came_sf=True)

    assert parser.parse_args(["--optimizer", "CAME-SF"]).optimizer == "CAME-SF"


def test_common_parser_can_expose_adamw_lrsf_variants():
    parser = argparse.ArgumentParser()
    add_optimizer_argument(parser, include_experimental=True)

    assert parser.parse_args(["--optimizer", "AdamW-SF"]).optimizer == "AdamW-SF"
    assert parser.parse_args(["--optimizer", "AdamW-LRSF"]).optimizer == "AdamW-LRSF"
    assert parser.parse_args(["--optimizer", "AdamW-SF-LR"]).optimizer == "AdamW-SF-LR"
    assert parser.parse_args(["--optimizer", "AdamW-LRSF-LR"]).optimizer == "AdamW-LRSF-LR"
    assert parser.parse_args(["--optimizer", "AdamW-LR-EMA"]).optimizer == "AdamW-LR-EMA"
    assert parser.parse_args(["--optimizer", "AdamW-LR-EMA-Conf"]).optimizer == "AdamW-LR-EMA-Conf"
    assert parser.parse_args(["--optimizer", "AdamW-LR-EMA-Conf-LRSF"]).optimizer == "AdamW-LR-EMA-Conf-LRSF"
    assert parser.parse_args(["--optimizer", "APOLLO-Conf"]).optimizer == "APOLLO-Conf"


def test_factory_builds_low_rank_gradient_ema():
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = build_optimizer(
        "AdamW-LR-EMA",
        [parameter],
        args=SimpleNamespace(
            rank=4, adamw_lr_ema_beta=0.8, adamw_lr_ema_projection_scale="norm",
        ),
    )

    assert type(optimizer).__name__ == "AdamWLowRankGradientEMA"
    assert optimizer.param_groups[0]["lr_ema_rank"] == 4
    assert optimizer.param_groups[0]["lr_ema_beta"] == 0.8


def test_factory_builds_low_rank_gradient_ema_confidence():
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = build_optimizer(
        "AdamW-LR-EMA-Conf",
        [parameter],
        args=SimpleNamespace(
            rank=4,
            adamw_lr_ema_beta=0.8,
            adamw_lr_ema_confidence_beta=0.97,
            adamw_lr_ema_confidence_alpha=0.002,
            adamw_lr_ema_projection_scale="norm",
        ),
    )

    assert type(optimizer).__name__ == "AdamWLowRankGradientEMAConfidence"
    assert optimizer.param_groups[0]["lr_ema_confidence_beta"] == 0.97
    assert optimizer.param_groups[0]["lr_ema_confidence_alpha"] == 0.002


def test_factory_builds_low_rank_gradient_ema_confidence_lrsf():
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = build_optimizer(
        "AdamW-LR-EMA-Conf-LRSF",
        [parameter],
        args=SimpleNamespace(
            rank=4,
            adamw_lrsf_beta1=0.9,
            adamw_lrsf_beta2=0.999,
            adamw_lrsf_projection_refresh={"mode": "none"},
            adamw_lrsf_orthogonal_refresh=None,
            adamw_sf_backend="torch",
            adamw_lr_ema_confidence_beta=0.97,
            adamw_lr_ema_confidence_alpha=0.002,
        ),
    )

    assert type(optimizer).__name__ == "AdamWLRSEMAConfLRSF"
    assert optimizer.param_groups[0]["lr_ema_confidence_beta"] == 0.97
    assert optimizer.param_groups[0]["lr_ema_confidence_alpha"] == 0.002


def test_factory_builds_apollo_confidence_state_variant():
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = build_optimizer(
        "APOLLO-Conf",
        [parameter],
        args=SimpleNamespace(
            apollo_rank=4,
            apollo_confidence_beta=0.97,
            apollo_confidence_alpha=0.002,
            apollo_projection_refresh_mode="none",
            apollo_update_proj_gap=10,
            apollo_matrix_fallback="apollo",
        ),
        lr=1e-3,
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]

    assert type(optimizer).__name__ == "APOLLOConfidence"
    assert state["backend"] == "apollo"
    assert state["apollo_confidence"] is True
    assert state["projection"].shape == (8, 4)
    assert state["exp_avg"].shape == (8, 4)
    assert state["exp_avg_sq"].shape == (8, 4)
    assert "fallback_exp_avg" not in state


def test_confidence_lrsf_keeps_separate_low_rank_states_and_full_fallback():
    matrix = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = build_optimizer(
        "AdamW-LR-EMA-Conf-LRSF",
        [matrix],
        args=SimpleNamespace(rank=2, adamw_sf_backend="torch"),
    )
    optimizer.train()
    matrix.grad = torch.ones_like(matrix)
    optimizer.step()
    matrix_state = optimizer.state[matrix]

    assert matrix_state["backend"] == "lrsf"
    assert matrix_state["lrsf_delta"].shape == (8, 2)
    assert matrix_state["lr_ema_grad"].shape == (8, 2)
    assert matrix_state["lr_ema_residual_sq"].shape == (8, 2)
    assert not torch.equal(
        matrix_state["lrsf_projection"],
        matrix_state["lr_ema_confidence_projection"],
    )
    assert "exp_avg_sq" not in matrix_state

    vector = torch.nn.Parameter(torch.randn(8))
    fallback = build_optimizer(
        "AdamW-LR-EMA-Conf-LRSF",
        [vector],
        args=SimpleNamespace(rank=2, adamw_sf_backend="torch"),
    )
    fallback.train()
    vector.grad = torch.ones_like(vector)
    fallback.step()
    vector_state = fallback.state[vector]

    assert vector_state["backend"] == "sf_full"
    assert vector_state["z"].shape == vector.shape
    assert vector_state["exp_avg_sq"].shape == vector.shape


def test_confidence_lrsf_bfloat16_fallback_interpolates_in_compute_dtype():
    vector = torch.nn.Parameter(torch.randn(8, dtype=torch.bfloat16))
    optimizer = build_optimizer(
        "AdamW-LR-EMA-Conf-LRSF",
        [vector],
        args=SimpleNamespace(rank=2, adamw_sf_backend="torch"),
        lr=1e-3,
    )
    optimizer.train()
    vector.grad = torch.randn_like(vector)

    optimizer.step()

    state = optimizer.state[vector]
    assert state["backend"] == "sf_full"
    assert state["z"].dtype == vector.dtype
    assert state["exp_avg_sq"].dtype == vector.dtype
    assert torch.isfinite(vector).all()


def test_factory_builds_adamw_sf_oracle():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = build_optimizer(
        "AdamW-SF", [parameter], args=SimpleNamespace(),
        lr=1e-3, weight_decay=0.0,
    )
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert type(optimizer).__name__ == "AdamWScheduleFree"


def test_factory_builds_adamw_sf_low_rank_preconditioner():
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = build_optimizer(
        "AdamW-SF-LR", [parameter],
        args=SimpleNamespace(rank=4, adamw_sf_backend="torch"),
    )

    assert type(optimizer).__name__ == "AdamWSFLowRankPreconditioner"
    assert optimizer.param_groups[0]["sf_lr_rank"] == 4


def test_factory_builds_integrated_adamw_lrsf_low_rank_variant():
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = build_optimizer(
        "AdamW-LRSF-LR", [parameter],
        args=SimpleNamespace(rank=4, adamw_sf_backend="torch"),
    )

    assert type(optimizer).__name__ == "AdamWLRSLowRankPreconditioner"


def test_factory_builds_adamw_lrsf_and_uses_full_sf_for_vectors():
    matrix = torch.nn.Parameter(torch.ones(8, 8))
    vector = torch.nn.Parameter(torch.ones(8))
    optimizer = build_optimizer(
        "AdamW-LRSF", [matrix, vector],
        args=SimpleNamespace(adamw_lrsf_rank=2),
        lr=1e-3, weight_decay=0.0,
    )
    optimizer.train()
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer.step()

    assert type(optimizer).__name__ == "AdamWLRSF"
    assert optimizer.state[matrix]["backend"] == "lrsf"
    assert optimizer.state[vector]["backend"] == "sf_full"


def test_factory_builds_came_sf_oracle():
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = build_optimizer(
        "CAME-SF", [parameter], args=SimpleNamespace(),
        lr=1e-3, weight_decay=0.0,
    )
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert type(optimizer).__name__ == "CAMESF"
    assert optimizer.state[parameter]["backend"] == "sf_full"


def test_common_parser_can_expose_apollo_came_lrsf_explicitly():
    parser = argparse.ArgumentParser()
    add_optimizer_argument(parser, include_apollo_came_lrsf=True)

    assert parser.parse_args(["--optimizer", "APOLLO-CAME-LRSF"]).optimizer == "APOLLO-CAME-LRSF"


def test_factory_builds_came_lrsf_and_marks_it_schedule_free():
    parameter = torch.nn.Parameter(torch.ones(8, 8))
    optimizer = build_optimizer(
        "CAME-LRSF",
        [parameter],
        args=SimpleNamespace(came_lrsf_rank=4, came_lrsf_beta1=0.9),
        lr=1e-3,
        weight_decay=0.0,
    )
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert type(optimizer).__name__ == "CAMELRSF"
    assert optimizer.state[parameter]["backend"] == "lrsf"


def test_invalid_optimizer_name_is_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        normalize_optimizer_name("not-an-optimizer", CORE_OPTIMIZER_CHOICES)
