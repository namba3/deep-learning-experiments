import argparse
from typing import Any, cast

import torch
from torch import nn

from verify.image_ae_cifar10_optimizer_convergence import (
    build_model,
    parse_args,
    run_case,
)
from verify.image_ae_optimizer_convergence import build_optimizer


def test_image_ae_cifar10_probe_parser():
    args = parse_args([
        "--device", "cpu",
        "--epochs", "3",
        "--batch-size", "16",
        "--max-train-samples", "128",
        "--max-validation-samples", "64",
        "--optimizers", "CAME,CAME-SF,CAME-LRSF",
        "--rank", "4", "--ranks", "1,4,8,16",
        "--refresh-mode", "smooth", "--refresh-interval", "8",
        "--refresh-window", "4", "--refresh-mix", "linear",
        "--orthogonal-refresh-direction", "loss_directed",
        "--orthogonal-refresh-signal", "effective_update",
        "--record-step-metrics", "--record-update-norms",
    ])

    assert args.device == "cpu"
    assert args.epochs == 3
    assert args.max_train_samples == 128
    assert args.max_validation_samples == 64
    assert args.optimizers == ("CAME", "CAME-SF", "CAME-LRSF")
    assert args.rank == 4
    assert args.ranks == (1, 4, 8, 16)
    assert args.refresh_mode == "smooth"
    assert args.refresh_interval == 8
    assert args.refresh_window == 4
    assert args.refresh_mix == "linear"
    assert args.orthogonal_refresh_direction == "loss_directed"
    assert args.orthogonal_refresh_signal == "effective_update"
    assert args.record_step_metrics is True
    assert args.record_update_norms is True


def test_cifar_probe_forwards_refresh_policy_to_lrsf_optimizers():
    model = nn.Linear(4, 4)
    args = argparse.Namespace(
        learning_rate=2e-4,
        weight_decay=0.0,
        rank=2,
        scale=1.0,
        disable_norm_growth_limiter=True,
        norm_growth_rate=1.01,
        seed=0,
        came_lrsf_refresh_mode="smooth",
        came_lrsf_refresh_interval=8,
        came_lrsf_refresh_window=4,
        came_lrsf_refresh_mix="linear",
        came_lrsf_orthogonal_refresh_direction="loss_directed",
        came_lrsf_orthogonal_refresh_signal="effective_update",
    )

    came_lrsf = build_optimizer("CAME-LRSF", model, args)
    apollo_came_lrsf = build_optimizer("APOLLO-CAME-LRSF", model, args)

    expected = {
        "mode": "smooth",
        "interval": 8,
        "window": 4,
        "mix": "linear",
    }
    assert came_lrsf.param_groups[0]["projection_refresh"] == expected
    assert apollo_came_lrsf.param_groups[0]["delta_refresh"] == expected
    assert came_lrsf.param_groups[0]["orthogonal_refresh"] == {
        "rate": 0.0, "seed": 0, "direction": "loss_directed",
        "signal": "effective_update",
    }


def test_cifar_probe_reports_peak_state_and_cpu_memory_placeholders():
    args = argparse.Namespace(
        latent_channels=4,
        bottleneck_channels=16,
        downsample_stages=2,
        learning_rate=2e-4,
        weight_decay=0.0,
        rank=1,
        seed=0,
        epochs=1,
        batch_size=2,
        refresh_mode="none",
        refresh_interval=1,
        refresh_window=1,
        refresh_mix="ema",
        orthogonal_refresh_rate=0.01,
        orthogonal_refresh_direction="loss_lowering",
        orthogonal_refresh_signal="gradient",
        record_step_metrics=True,
        record_update_norms=True,
    )
    torch.manual_seed(0)
    initial_model = build_model(args, torch.device("cpu"), torch.float32)
    initial_state = {
        key: value.detach().clone()
        for key, value in initial_model.state_dict().items()
    }
    images = torch.rand(4, 3, 32, 32)

    result = cast(dict[str, Any], run_case(
        "CAME-LRSF", args, torch.device("cpu"), torch.float32,
        initial_state, images, images,
    ))

    assert result["peak_persistent_state_bytes"] >= result["persistent_state_bytes"]
    assert result["peak_persistent_state_elements"] >= result["persistent_state_elements"]
    assert result["peak_allocated_bytes"] is None
    assert result["peak_reserved_bytes"] is None
    assert result["peak_delta_allocated_bytes"] is None
    assert result["update_norm_mean"] >= 0.0
    assert result["update_norm_variance"] >= 0.0
    assert result["orthogonal_refresh_steps"] == 2
    assert result["orthogonal_refresh_events"]
    assert any(
        event.get("loss_lowering_proxy_decrease") is not None
        for event in result["orthogonal_refresh_events"]
    )
