import random

import numpy as np
import torch

from optimizers.lr_scheduler import LearningRateSchedule
from runtime.checkpoint import (
    load_training_state,
    make_training_state,
    resume_state_path,
    restore_rng_state,
    save_training_state,
)
from runtime.sampler import ResumableRandomSampler


def test_training_state_round_trip_restores_optimizer_and_scheduler(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = LearningRateSchedule(
        optimizer, "cosine", total_steps=10, warmup_steps=2,
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step(1)
    state = make_training_state(
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        global_step=7,
        extra={"depth_scheduler": {"step_count": 7}},
    )

    checkpoint = tmp_path / "checkpoint_latest.safetensors"
    saved_path = save_training_state(checkpoint, state)
    loaded = load_training_state(checkpoint)

    assert saved_path == resume_state_path(checkpoint)
    assert loaded is not None
    assert loaded["epoch"] == 3
    assert loaded["global_step"] == 7
    assert loaded["optimizer"]["state"]
    assert loaded["scheduler"]["last_step"] == 1
    assert loaded["extra"]["depth_scheduler"]["step_count"] == 7


def test_rng_state_is_serialized_in_training_state(tmp_path):
    random.seed(123)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    state = make_training_state(
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        global_step=0,
    )
    save_training_state(tmp_path / "model.safetensors", state)
    loaded = load_training_state(tmp_path / "model.safetensors")

    assert loaded is not None
    assert "python" in loaded["rng"]
    assert "torch" in loaded["rng"]


def test_full_resume_round_trip_restores_rng_and_sampler_state(tmp_path):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    dataset = torch.utils.data.TensorDataset(torch.arange(8))
    sampler = ResumableRandomSampler(dataset, seed=77)
    sampler.set_epoch(2, position=3)
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = LearningRateSchedule(
        optimizer, "cosine", total_steps=10, warmup_steps=2,
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step(1)

    state = make_training_state(
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=7,
        extra={"sampler": sampler.state_dict()},
    )
    save_training_state(tmp_path / "model.safetensors", state)

    expected_random = random.random()
    expected_torch = torch.rand(())
    expected_numpy = float(np.random.random())

    loaded = load_training_state(tmp_path / "model.safetensors")
    assert loaded is not None

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored_optimizer = torch.optim.AdamW([restored_parameter], lr=0.1)
    restored_scheduler = LearningRateSchedule(
        restored_optimizer, "cosine", total_steps=10, warmup_steps=2,
    )
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_scheduler.load_state_dict(loaded["scheduler"])
    restore_rng_state(loaded["rng"])

    restored_sampler = ResumableRandomSampler(dataset, seed=999)
    restored_sampler.load_state_dict(loaded["extra"]["sampler"])

    assert random.random() == expected_random
    assert torch.equal(torch.rand(()), expected_torch)
    assert float(np.random.random()) == expected_numpy
    assert list(restored_sampler) == list(sampler)
    assert restored_sampler.state_dict() == sampler.state_dict()
    assert restored_scheduler.state_dict() == scheduler.state_dict()
