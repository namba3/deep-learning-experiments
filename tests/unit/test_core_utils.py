import torch
import torch.nn as nn

from core.utils import (
    DepthDistributionScheduler,
    build_parameter_groups,
    convert_rmsnorm_to_dtype_aware,
    format_bytes,
)


def test_format_bytes_supports_large_binary_units():
    assert format_bytes(1023) == "1023.00 B"
    assert format_bytes(1024**2) == "1.00 MiB"
    assert format_bytes(1024**4) == "1.00 TiB"
    assert format_bytes(1024**6) == "1.00 PiB"


def test_depth_distribution_scheduler_is_normalized_and_progresses():
    scheduler = DepthDistributionScheduler(
        min_depth=1,
        max_depth=4,
        total_steps=10,
        initial_bias=1.0,
        final_bias=4.0,
        schedule="linear",
    )

    initial = scheduler.probabilities()
    assert torch.isclose(initial.sum(), torch.tensor(1.0))
    assert torch.allclose(initial, torch.full((4,), 0.25))

    scheduler.step(10)
    final = scheduler.probabilities()
    assert torch.isclose(final.sum(), torch.tensor(1.0))
    assert final[-1] > final[0]
    assert scheduler.progress == 1.0


def test_build_parameter_groups_separates_decay_targets():
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.LayerNorm(4),
        nn.Conv2d(2, 3, kernel_size=1),
    )
    groups = build_parameter_groups(
        model,
        target_param_regexes=[r"linear", r"conv2d"],
        weight_decay=0.1,
    )

    assert {group["weight_decay"] for group in groups} == {0.0, 0.1}
    grouped_parameters = [
        parameter
        for group in groups
        for parameter in group["params"]
    ]
    assert len(grouped_parameters) == len(list(model.parameters()))
    assert len({id(parameter) for parameter in grouped_parameters}) == len(
        grouped_parameters
    )

    decay_group = next(group for group in groups if group["weight_decay"] == 0.1)
    assert {id(parameter) for parameter in decay_group["params"]} == {
        id(model[0].weight),
        id(model[0].bias),
        id(model[2].weight),
        id(model[2].bias),
    }


def test_convert_rmsnorm_to_dtype_aware_handles_mixed_precision_boundary():
    model = nn.Sequential(nn.RMSNorm(4, dtype=torch.bfloat16))
    convert_rmsnorm_to_dtype_aware(model)

    assert model[0].__class__.__name__ == "DtypeAwareRMSNorm"
    output = model(torch.randn(2, 4, dtype=torch.float32))

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
