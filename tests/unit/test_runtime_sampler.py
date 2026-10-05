import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from runtime.data import build_dataloader_options, seed_worker
from runtime.sampler import (
    ResumableAspectRatioBatchSampler,
    ResumableRandomSampler,
)


def test_resumable_sampler_recreates_remaining_order():
    dataset = TensorDataset(torch.arange(10))
    sampler = ResumableRandomSampler(dataset, seed=123)
    sampler.set_epoch(2, position=4)

    remaining = list(iter(sampler))
    state = sampler.state_dict()

    restored = ResumableRandomSampler(dataset, seed=999)
    restored.load_state_dict(state)

    assert list(iter(restored)) == remaining
    assert len(restored) == 6


def test_resumable_sampler_position_is_not_advanced_by_iteration():
    dataset = TensorDataset(torch.arange(5))
    sampler = ResumableRandomSampler(dataset, seed=1)
    first = list(iter(sampler))[:2]

    assert len(first) == 2
    assert sampler.position == 0
    sampler.set_position(2)
    assert len(sampler) == 3


def test_resumable_sampler_matches_dataloader_after_consumed_batch():
    dataset = TensorDataset(torch.arange(12))
    sampler = ResumableRandomSampler(dataset, seed=7)
    loader = DataLoader(dataset, batch_size=3, sampler=sampler)
    first_batch = next(iter(loader))[0].tolist()
    sampler.set_position(3)

    resumed_loader = DataLoader(dataset, batch_size=3, sampler=sampler)
    remaining = [value for batch in resumed_loader for value in batch[0].tolist()]

    full_sampler = ResumableRandomSampler(dataset, seed=7)
    full_order = [value for value in full_sampler]
    assert first_batch == full_order[:3]
    assert remaining == full_order[3:]


def test_resumable_batch_sampler_keeps_buckets_and_resumes():
    sampler = ResumableAspectRatioBatchSampler(
        [0, 1, 0, 1, 0, 1], batch_size=2, shuffle=False, seed=123,
    )

    assert list(sampler) == [[0, 2], [1, 3]]
    sampler.set_position(1)
    state = sampler.state_dict()

    restored = ResumableAspectRatioBatchSampler(
        [0, 1, 0, 1, 0, 1], batch_size=2, shuffle=False, seed=999,
    )
    restored.load_state_dict(state)

    assert list(restored) == [[1, 3]]
    assert len(restored) == 1


def test_resumable_batch_sampler_restores_shuffled_epoch():
    sampler = ResumableAspectRatioBatchSampler(
        [0, 1, 0, 1, 0, 1, 0, 1], batch_size=2, seed=7,
    )
    sampler.set_epoch(3, position=1)
    remaining = list(sampler)
    state = sampler.state_dict()

    restored = ResumableAspectRatioBatchSampler(
        [0, 1, 0, 1, 0, 1, 0, 1], batch_size=2, seed=999,
    )
    restored.load_state_dict(state)

    assert list(restored) == remaining


def test_dataloader_options_use_independent_reproducible_worker_streams():
    first = build_dataloader_options(
        num_workers=2, pin_memory=False, seed=123, stream=0,
    )
    second = build_dataloader_options(
        num_workers=2, pin_memory=False, seed=123, stream=0,
    )
    eval_options = build_dataloader_options(
        num_workers=2, pin_memory=False, seed=123, stream=1,
    )

    assert first["worker_init_fn"] is seed_worker
    assert first["persistent_workers"] is True
    assert first["generator"].initial_seed() == second["generator"].initial_seed()
    assert first["generator"].initial_seed() != eval_options["generator"].initial_seed()


def test_worker_seeding_reproduces_torch_python_and_numpy_transforms():
    def sample_values():
        torch.manual_seed(321)
        seed_worker(0)
        return (
            torch.rand(()),
            random.random(),
            float(np.random.random()),
        )

    first = sample_values()
    second = sample_values()

    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[2] == second[2]
