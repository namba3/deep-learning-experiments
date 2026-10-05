"""Checkpointable samplers for deterministic epoch-local data order."""

from __future__ import annotations

from collections.abc import Iterable, Sized
import hashlib
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):
    """Generate a deterministic random permutation with a resumable position.

    The sampler does not advance its position while DataLoader workers prefetch
    indices.  Training loops must call :meth:`set_position` after a batch has
    actually been consumed; this keeps the saved position tied to optimizer
    progress rather than worker prefetch progress.
    """

    state_version = 1

    def __init__(self, data_source: Sized, *, seed: int | None = None) -> None:
        self.num_samples = len(data_source)
        if self.num_samples < 0:
            raise ValueError("data_source length must be non-negative")
        self.seed = int(torch.initial_seed() if seed is None else seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.epoch = 0
        self.position = 0

    def _permutation(self) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(self.num_samples, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        # Keep position read-only here. DataLoader may request indices ahead of
        # the batch currently being processed when workers are enabled.
        yield from self._permutation()[self.position:]

    def __len__(self) -> int:
        return self.num_samples - self.position

    def set_epoch(self, epoch: int, *, position: int = 0) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        self.set_position(position)

    def set_position(self, position: int) -> None:
        position = int(position)
        if not 0 <= position <= self.num_samples:
            raise ValueError(
                f"position must be in [0, {self.num_samples}], got {position}"
            )
        self.position = position

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("state_version") != self.state_version:
            raise ValueError(
                f"Unsupported sampler state version {state.get('state_version')!r}"
            )
        saved_num_samples = int(state["num_samples"])
        if saved_num_samples != self.num_samples:
            raise ValueError(
                "Sampler dataset length mismatch: "
                f"current={self.num_samples}, saved={saved_num_samples}"
            )
        self.seed = int(state["seed"])
        self.set_epoch(int(state["epoch"]), position=int(state["position"]))


class ResumableWeightedRandomSampler(Sampler[int]):
    """Weighted sampler whose consumed position survives checkpoint resume.

    Indices are generated from ``seed + epoch`` on demand, while ``position``
    advances only when the training loop reports examples it actually consumed.
    This avoids treating DataLoader worker prefetch as completed training data.
    """

    state_version = 1

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        *,
        replacement: bool = True,
        seed: int | None = None,
    ) -> None:
        weights = torch.as_tensor(weights, dtype=torch.double, device="cpu").flatten()
        if weights.numel() == 0 or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be a non-empty finite non-negative tensor")
        if not bool(weights.sum() > 0):
            raise ValueError("weights must have a positive sum")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not replacement and num_samples > weights.numel():
            raise ValueError("num_samples cannot exceed weight count without replacement")
        self.weights = weights.contiguous()
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.seed = int(torch.initial_seed() if seed is None else seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.epoch = 0
        self.position = 0
        self._weights_signature = hashlib.sha256(
            self.weights.numpy().tobytes()
        ).hexdigest()

    def _indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=generator,
        ).tolist()

    def __iter__(self) -> Iterator[int]:
        yield from self._indices()[self.position:]

    def __len__(self) -> int:
        return self.num_samples - self.position

    def set_epoch(self, epoch: int, *, position: int = 0) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        self.set_position(position)

    def set_position(self, position: int) -> None:
        position = int(position)
        if not 0 <= position <= self.num_samples:
            raise ValueError(
                f"position must be in [0, {self.num_samples}], got {position}"
            )
        self.position = position

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "weights_signature": self._weights_signature,
            "num_samples": self.num_samples,
            "replacement": self.replacement,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("state_version") != self.state_version:
            raise ValueError(
                f"Unsupported weighted sampler state version {state.get('state_version')!r}"
            )
        if state.get("weights_signature") != self._weights_signature:
            raise ValueError("Weighted sampler weights mismatch")
        for key, current in (
            ("num_samples", self.num_samples),
            ("replacement", self.replacement),
        ):
            if state.get(key) != current:
                raise ValueError(
                    f"Weighted sampler configuration mismatch for {key}: "
                    f"current={current!r}, saved={state.get(key)!r}"
                )
        self.seed = int(state["seed"])
        self.set_epoch(int(state["epoch"]), position=int(state["position"]))


class ResumableAspectRatioBatchSampler(Sampler[list[int]]):
    """Checkpointable batch sampler that keeps samples within one bucket."""

    state_version = 1

    def __init__(
        self,
        bucket_ids: Iterable[int],
        batch_size: int,
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.bucket_ids = list(bucket_ids)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(torch.initial_seed() if seed is None else seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.epoch = 0
        self.position = 0
        self._bucket_signature = hashlib.sha256(
            repr(self.bucket_ids).encode("utf-8")
        ).hexdigest()

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        buckets: dict[int, list[int]] = {}
        for index, bucket in enumerate(self.bucket_ids):
            buckets.setdefault(int(bucket), []).append(index)

        batches: list[list[int]] = []
        for bucket in sorted(buckets):
            indices = buckets[bucket]
            if self.shuffle:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in permutation]
            limit = len(indices)
            if self.drop_last:
                limit -= limit % self.batch_size
            for start in range(0, limit, self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle and len(batches) > 1:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in permutation]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()[self.position:]

    def __len__(self) -> int:
        return max(0, len(self._batches()) - self.position)

    def set_epoch(self, epoch: int, *, position: int = 0) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        self.set_position(position)

    def set_position(self, position: int) -> None:
        position = int(position)
        total = len(self._batches())
        if not 0 <= position <= total:
            raise ValueError(f"position must be in [0, {total}], got {position}")
        self.position = position

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "bucket_signature": self._bucket_signature,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("state_version") != self.state_version:
            raise ValueError(
                f"Unsupported batch sampler state version {state.get('state_version')!r}"
            )
        if state.get("bucket_signature") != self._bucket_signature:
            raise ValueError("Batch sampler bucket assignment mismatch")
        for key, current in (
            ("batch_size", self.batch_size),
            ("drop_last", self.drop_last),
            ("shuffle", self.shuffle),
        ):
            if state.get(key) != current:
                raise ValueError(
                    f"Batch sampler configuration mismatch for {key}: "
                    f"current={current!r}, saved={state.get(key)!r}"
                )
        self.seed = int(state["seed"])
        self.set_epoch(int(state["epoch"]), position=int(state["position"]))
