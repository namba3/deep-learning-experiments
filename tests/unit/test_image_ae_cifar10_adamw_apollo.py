from typing import Any, cast

import torch

from verify import image_ae_cifar10_adamw_apollo as probe


def test_adamw_apollo_probe_parser_and_projection_options():
    args = probe.parse_args([
        "--device", "cpu",
        "--epochs", "2",
        "--optimizers", "AdamW,APOLLO,APOLLO-Mini",
        "--rank", "4",
        "--update-proj-gap", "3",
        "--projection-refresh-mode", "smooth",
        "--projection-refresh-window", "4",
        "--projection-refresh-mix", "linear",
        "--projection-refresh-state", "transport",
        "--orthogonal-refresh-direction", "loss_directed",
        "--update-norm-variance-cap", "0.001",
        "--freeze-projection",
    ])

    assert args.optimizers == ("AdamW", "APOLLO", "APOLLO-Mini")
    assert args.rank == 4
    assert args.update_proj_gap == 3
    assert args.projection_refresh_mode == "smooth"
    assert args.projection_refresh_window == 4
    assert args.projection_refresh_mix == "linear"
    assert args.projection_refresh_state == "transport"
    assert args.orthogonal_refresh_direction == "loss_directed"
    assert args.update_norm_variance_cap == 0.001
    assert args.freeze_projection is True


def test_adamw_apollo_probe_runs_on_fixed_cpu_images(monkeypatch):
    images = torch.linspace(0.0, 1.0, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)

    def fake_load_images(_data_dir, _max_samples, *, train):
        return images[:2] if train else images[2:]

    monkeypatch.setattr(probe, "_load_images", fake_load_images)
    args = probe.parse_args([
        "--device", "cpu",
        "--dtype", "fp32",
        "--epochs", "2",
        "--batch-size", "2",
        "--max-train-samples", "2",
        "--max-validation-samples", "2",
        "--latent-channels", "1",
        "--bottleneck-channels", "16",
        "--downsample-stages", "2",
        "--optimizers", "AdamW,APOLLO",
        "--update-proj-gap", "1",
        "--projection-refresh-state", "transport",
        "--record-step-metrics",
        "--record-update-norms",
    ])

    result = cast(dict[str, Any], probe.run(args))

    assert result["status"] == "passed"
    assert set(result["cases"]) == {"AdamW", "APOLLO"}
    for case in result["cases"].values():
        assert case["status"] == "passed"
        assert case["total_steps"] == 2
        assert case["persistent_state_elements"] > 0
        assert torch.isfinite(torch.tensor(case["final_validation_loss"]))
        assert len(case["step_loss_history"]) == 2
        assert len(case["update_norm_history"]) == 2
        assert case["update_norm_variance"] >= 0.0

    assert result["cases"]["AdamW"]["projection_refresh_events"] == []
    assert result["cases"]["APOLLO"]["projection_refresh_events"]
    assert result["cases"]["APOLLO"]["projection_refresh_steps"] == 1
    assert result["cases"]["APOLLO"]["orthogonal_refresh_steps"] == 0
    assert result["cases"]["APOLLO"]["projection_refresh_events"][0][
        "projection_change_max_abs"
    ] > 0.0
