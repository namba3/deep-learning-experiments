import argparse
from typing import Any, cast

import pytest

from verify.optimizer_convergence import parse_args, run


def test_optimizer_convergence_parser_preserves_probe_options():
    args = parse_args([
        "--device", "cpu", "--dtype", "fp32", "--warmup", "0",
        "--steps", "2", "--batch-size", "8", "--rank", "4",
        "--matrix-fallback", "auto", "--optimizers", "CAME,APOLLO",
    ])

    assert args.device == "cpu"
    assert args.steps == 2
    assert args.batch_size == 8
    assert args.rank == 4
    assert args.optimizers == ("CAME", "APOLLO")


def test_optimizer_convergence_parser_accepts_sweep_options():
    args = parse_args([
        "--ranks", "1,4,8", "--learning-rates", "3e-4,1e-3",
        "--scales", "0.5,1.0", "--disable-norm-growth-limiter",
        "--norm-growth-rate", "1.05", "--norm-growth-rates", "1.01,1.1",
    ])

    assert args.ranks == (1, 4, 8)
    assert args.learning_rates == (3e-4, 1e-3)
    assert args.scales == (0.5, 1.0)
    assert args.disable_norm_growth_limiter is True
    assert args.norm_growth_rate == 1.05
    assert args.norm_growth_rates == (1.01, 1.1)


def test_optimizer_convergence_parser_rejects_invalid_norm_growth_rates():
    with pytest.raises(SystemExit):
        parse_args(["--norm-growth-rates", "1.0,1.05"])


def test_optimizer_convergence_runs_cpu_probe():
    args = argparse.Namespace(
        device="cpu",
        dtype="fp32",
        warmup=1,
        steps=3,
        batch_size=8,
        seed=0,
        optimizers=("CAME", "APOLLO", "APOLLO-CAME", "APOLLOMini"),
        rank=8,
        matrix_fallback="auto",
        learning_rate=1e-3,
        scale=1.0,
        ranks=None,
        learning_rates=None,
        scales=None,
        disable_norm_growth_limiter=False,
    )

    result = cast(dict[str, Any], run(args))

    assert result["status"] == "passed"
    assert result["device"] == "cpu"
    for case in result["cases"].values():
        assert case["status"] == "passed"
        assert case["final_loss"] is not None
        assert case["persistent_state_bytes"] > 0
        assert len(case["loss_history"]) == 3


def test_optimizer_convergence_runs_norm_growth_sweep_on_cpu():
    args = argparse.Namespace(
        device="cpu",
        dtype="fp32",
        warmup=0,
        steps=1,
        batch_size=4,
        seed=0,
        optimizers=("APOLLO",),
        rank=1,
        ranks=(1,),
        matrix_fallback="auto",
        learning_rate=1e-3,
        learning_rates=(1e-3,),
        scale=1.0,
        scales=(1.0,),
        disable_norm_growth_limiter=False,
        norm_growth_rate=1.01,
        norm_growth_rates=(1.01, 1.05),
    )

    result = cast(dict[str, Any], run(args))

    assert result["status"] == "passed"
    assert set(result["cases"]) == {
        "APOLLO@rank=1,lr=0.001,scale=1,growth=1.01",
        "APOLLO@rank=1,lr=0.001,scale=1,growth=1.05",
    }


def test_optimizer_convergence_runs_rank_sweep_on_cpu():
    args = argparse.Namespace(
        device="cpu",
        dtype="fp32",
        warmup=0,
        steps=1,
        batch_size=4,
        seed=0,
        optimizers=("APOLLO",),
        rank=8,
        ranks=(1, 4),
        matrix_fallback="auto",
        learning_rate=1e-3,
        learning_rates=(1e-3,),
        scale=1.0,
        scales=(1.0,),
        disable_norm_growth_limiter=False,
    )

    result = cast(dict[str, Any], run(args))

    assert result["status"] == "passed"
    assert set(result["cases"]) == {
        "APOLLO@rank=1,lr=0.001,scale=1",
        "APOLLO@rank=4,lr=0.001,scale=1",
    }
