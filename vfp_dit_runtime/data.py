"""Tensor-manifest datasets for the first VFP-DiT training stages.

Each JSONL row contains a path to a CPU ``torch.save``-d tensor dictionary.
The feature extraction contract is deliberately external: changing the frozen
teacher or aligner changes the meaning of the saved target tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class TensorManifestDataset(Dataset):
    """Load one tensor dictionary per line from a JSONL manifest."""

    def __init__(self, manifest: str | Path, *, required_keys: tuple[str, ...]):
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Tensor manifest not found: {self.manifest}")
        self.required_keys = required_keys
        self.records: list[Path] = []
        for line_number, line in enumerate(
            self.manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {self.manifest}:{line_number}: {error}"
                ) from error
            if isinstance(row, str):
                relative_path = row
            elif isinstance(row, dict) and isinstance(row.get("tensor_file"), str):
                relative_path = row["tensor_file"]
            else:
                raise ValueError(
                    f"{self.manifest}:{line_number} must be a path string or "
                    "an object with a 'tensor_file' string"
                )
            path = Path(relative_path).expanduser()
            if not path.is_absolute():
                path = (self.manifest.parent / path).resolve()
            self.records.append(path)
        if not self.records:
            raise ValueError(f"Tensor manifest is empty: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.records[index]
        if not path.is_file():
            raise FileNotFoundError(f"Tensor record not found: {path}")
        try:
            item: Any = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before weights_only was introduced.
            item = torch.load(path, map_location="cpu")
        if not isinstance(item, dict):
            raise ValueError(f"Tensor record must contain a dictionary: {path}")
        missing = [key for key in self.required_keys if key not in item]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
        result: dict[str, torch.Tensor] = {}
        for key in self.required_keys:
            value = item[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{path}: '{key}' must be a torch.Tensor")
            result[key] = value
        return result


def collate_tensor_records(
    records: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack fixed-shape values and pad variable-length semantic sequences."""
    if not records:
        raise ValueError("Cannot collate an empty batch")
    keys = set(records[0])
    if any(set(record) != keys for record in records):
        raise ValueError("All tensor records in a batch must have the same keys")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [record[key] for record in records]
        if key == "condition_hidden":
            if any(value.ndim != 2 for value in values):
                raise ValueError(
                    "condition_hidden must be (tokens, condition_dim) per record"
                )
            widths = {int(value.shape[1]) for value in values}
            if len(widths) != 1:
                raise ValueError("condition_hidden widths must match within a batch")
            max_tokens = max(int(value.shape[0]) for value in values)
            padded = values[0].new_zeros((len(values), max_tokens, values[0].shape[1]))
            mask = torch.zeros((len(values), max_tokens), dtype=torch.bool)
            for row, value in enumerate(values):
                length = int(value.shape[0])
                padded[row, :length] = value
                mask[row, :length] = True
            result[key] = padded
            result["condition_mask"] = mask
            continue
        shapes = {tuple(value.shape) for value in values}
        if len(shapes) != 1:
            raise ValueError(
                f"'{key}' shapes must match inside each batch; got {sorted(shapes)}"
            )
        result[key] = torch.stack(values)
    return result


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}
