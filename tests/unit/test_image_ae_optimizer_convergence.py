import argparse
from typing import Any, cast

from verify.image_ae_optimizer_convergence import parse_args, run


def test_image_ae_optimizer_probe_parser():
    args = parse_args([
        "--device", "cpu", "--dtype", "fp32", "--steps", "2",
        "--batch-size", "2", "--image-size", "16", "--rank", "4",
        "--learning-rate", "3e-4", "--scale", "0.5",
        "--disable-norm-growth-limiter", "--optimizers", "APOLLO,CAME",
    ])

    assert args.image_size == 16
    assert args.rank == 4
    assert args.learning_rate == 3e-4
    assert args.scale == 0.5
    assert args.disable_norm_growth_limiter is True
    assert args.optimizers == ("APOLLO", "CAME")


def test_image_ae_optimizer_probe_parser_accepts_sweep_options():
    args = parse_args([
        "--ranks", "1,4", "--learning-rates", "1e-4,2e-4",
        "--scales", "0.5,1.0", "--norm-growth-rate", "1.05",
        "--norm-growth-rates", "1.01,1.1", "--weight-decays", "0,1e-4",
    ])

    assert args.ranks == (1, 4)
    assert args.learning_rates == (1e-4, 2e-4)
    assert args.scales == (0.5, 1.0)
    assert args.norm_growth_rate == 1.05
    assert args.norm_growth_rates == (1.01, 1.1)
    assert args.weight_decays == (0.0, 1e-4)


def test_image_ae_optimizer_probe_runs_cpu():
    args = argparse.Namespace(
        device="cpu",
        dtype="fp32",
        warmup=0,
        steps=1,
        batch_size=2,
        image_size=16,
        seed=0,
        optimizers=("CAME", "APOLLO", "APOLLO-CAME", "APOLLOMini"),
        rank=4,
        learning_rate=2e-4,
        scale=1.0,
        disable_norm_growth_limiter=False,
        latent_channels=4,
        bottleneck_channels=32,
        downsample_stages=2,
    )

    result = cast(dict[str, Any], run(args))

    assert result["status"] == "passed"
    assert result["model"]["encoder"] == "residual_conv_ffn"
    for case in result["cases"].values():
        assert case["status"] == "passed"
        assert case["final_loss"] is not None
        assert case["persistent_state_bytes"] > 0


def test_image_ae_optimizer_probe_runs_apollo_growth_sweep_on_cpu():
    args = argparse.Namespace(
        device="cpu",
        dtype="fp32",
        warmup=0,
        steps=1,
        batch_size=2,
        image_size=16,
        seed=0,
        optimizers=("APOLLO",),
        rank=1,
        ranks=(1,),
        learning_rate=2e-4,
        learning_rates=(2e-4,),
        scale=1.0,
        scales=(1.0,),
        disable_norm_growth_limiter=False,
        norm_growth_rate=1.01,
        norm_growth_rates=(1.01, 1.05),
        latent_channels=4,
        bottleneck_channels=32,
        downsample_stages=2,
    )

    result = cast(dict[str, Any], run(args))

    assert result["status"] == "passed"
    assert set(result["cases"]) == {
        "APOLLO@rank=1,lr=0.0002,scale=1,growth=1.01",
        "APOLLO@rank=1,lr=0.0002,scale=1,growth=1.05",
    }
    for case in result["cases"].values():
        assert case["norm_growth_rate"] in {1.01, 1.05}
