import math

from types import SimpleNamespace

import pytest

import torch

from optimizers.lr_scheduler import LearningRateSchedule, build_lr_scheduler, resolve_lr_scheduler_name, resolve_warmup_steps

def test_learning_rate_schedule_warmup_and_linear_decay():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    schedule = LearningRateSchedule(
        optimizer,
        name="linear",
        total_steps=10,
        warmup_steps=2,
        min_lr_ratio=0.2,
    )

    assert math.isclose(schedule.step(1), 0.5)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.05)
    assert math.isclose(schedule.step(10), 0.2)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.02)

@pytest.mark.parametrize(
    "name",
    [
        "constant",
        "linear",
        "cosine",
        "cosine-restarts",
        "polynomial",
        "inverse-sqrt",
        "step",
        "multistep",
        "exponential",
    ],
)
def test_learning_rate_schedule_supports_common_step_schedulers(name):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    schedule = LearningRateSchedule(
        optimizer,
        name=name,
        total_steps=20,
        warmup_steps=2,
        min_lr_ratio=0.1,
        step_size=5,
        gamma=0.5,
        milestones=(5, 10),
        num_cycles=2,
        power=2.0,
    )

    schedule.step(1)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.05)
    schedule.step(20)
    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr >= 0.01 - 1e-12
    assert final_lr <= 0.1 + 1e-12

def test_learning_rate_schedule_resolves_auto_and_warmup_ratio():
    assert resolve_lr_scheduler_name("auto", "CAME") == "constant"
    assert resolve_lr_scheduler_name("auto", "AdamW") == "cosine"
    assert resolve_warmup_steps(100, warmup_ratio=0.1) == 10
    with pytest.raises(ValueError, match="only one"):
        resolve_warmup_steps(100, warmup_steps=2, warmup_ratio=0.1)

def test_build_lr_scheduler_uses_cli_namespace():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    args = SimpleNamespace(
        lr_scheduler="step",
        optimizer="AdamW",
        warmup_steps=0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        lr_step_size=2,
        lr_gamma=0.5,
        lr_milestones=(2, 4),
        lr_num_cycles=1,
        lr_power=1.0,
    )
    schedule = build_lr_scheduler(optimizer, args, total_steps=10)
    schedule.step(2)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.05)
