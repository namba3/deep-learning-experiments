"""Shared DataLoader seeding helpers for reproducible worker transforms."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy RNGs from PyTorch's per-worker seed.

    PyTorch seeds its worker-local torch RNG when a DataLoader worker starts.
    Mirroring that seed into other commonly used RNGs keeps torchvision and
    user-defined transforms deterministic across workers.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader_options(
    *,
    num_workers: int,
    pin_memory: bool,
    seed: int | None = None,
    stream: int = 0,
) -> dict[str, Any]:
    """Return common DataLoader options with deterministic worker seeding.

    "stream" gives train/eval loaders independent generator streams while
    preserving repeatability for the same base seed. The sampler remains the
    source of sample order; the generator only controls worker base seeds.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if stream < 0:
        raise ValueError("stream must be >= 0")
    base_seed = int(torch.initial_seed() if seed is None else seed)
    if base_seed < 0:
        raise ValueError("seed must be >= 0")
    generator = torch.Generator()
    generator.manual_seed((base_seed + stream * 1_000_003) % (2**63 - 1))
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
