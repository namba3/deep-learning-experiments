from typing import Any, cast

import torch

from verify.adapter_metrics import merge_is_equivalent, merge_tolerances
from verify.cifar10_adapter_comparison import parse_args, run


def test_adapter_merge_tolerance_covers_bf16_accumulation_order():
    before = torch.tensor([2.3, -1.1, 0.01])
    after = torch.tensor([2.317, -1.105, 0.012])

    assert merge_tolerances(torch.float32) == (1e-5, 1e-5)
    assert merge_tolerances(torch.bfloat16) == (3e-2, 1e-2)
    assert merge_is_equivalent(before, after, torch.bfloat16)
    assert not merge_is_equivalent(before, after + 0.05, torch.bfloat16)


def test_adapter_comparison_parser_exposes_all_adapter_types():
    args = parse_args([
        "--adapters", "lora,loha,dora,glu_lora,rglu_lora",
        "--rank", "2",
        "--steps", "1",
    ])

    assert args.adapters == (
        "lora", "loha", "dora", "glu_lora", "rglu_lora",
    )
    assert args.rank == 2
    assert args.adapter_init == "identity"


def test_adapter_comparison_runs_all_cases_on_cpu():
    args = parse_args([
        "--device", "cpu",
        "--steps", "1",
        "--batch-size", "2",
        "--rank", "1",
    ])
    result = cast(dict[str, Any], run(args))

    assert result["status"] == "passed"
    assert set(result["cases"]) == {
        "lora", "loha", "dora", "glu_lora", "rglu_lora",
    }
    for case in result["cases"].values():
        assert case["status"] == "passed"
        assert case["trainable_parameters"] > 0
        assert case["optimizer_state_bytes"] > 0
        assert case["merge_equivalent"] is True
        assert case["initial_gradient_norm"] >= 0.0
        if case["adapter"] in {"glu_lora", "rglu_lora"}:
            assert case["gate_min"] <= case["gate_p50"] <= case["gate_max"]
        assert case["seconds_per_step"] >= 0.0
