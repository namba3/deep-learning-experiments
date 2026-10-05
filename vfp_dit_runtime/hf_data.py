"""Hugging Face image/text sources for the VFP-DiT Stage 0 input path.

The adapters normalize rows into target/source images, a prompt, and provenance.
They intentionally do not encode or resize images; those operations belong to
the declared VAE/VLM feature-extraction contract.
"""

from __future__ import annotations

import io
import json
import math
import random
from bisect import bisect_right
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset, WeightedRandomSampler


COCO_CAPTIONS_ID = "mm-eval/MS-COCO-Captions"
MULTI_EDIT_ID = "inclusionAI/MultiEdit"


def target_image_size(dataset: Dataset, index: int) -> tuple[int, int]:
    """Read target dimensions through Subset/ConcatDataset wrappers."""
    if isinstance(dataset, Subset):
        return target_image_size(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, ConcatDataset):
        source_index = bisect_right(dataset.cumulative_sizes, index)
        previous = 0 if source_index == 0 else dataset.cumulative_sizes[source_index - 1]
        return target_image_size(dataset.datasets[source_index], index - previous)
    size_fn = getattr(dataset, "target_size", None)
    if not callable(size_fn):
        raise TypeError(f"{type(dataset).__name__} does not expose target_size(index)")
    return size_fn(index)


def nearest_aspect_bucket(width: int, height: int, aspect_ratios: tuple[float, ...]) -> int:
    if width <= 0 or height <= 0 or not aspect_ratios:
        raise ValueError("positive image dimensions and aspect ratios are required")
    aspect = width / height
    return min(range(len(aspect_ratios)),
               key=lambda index: abs(math.log(aspect / aspect_ratios[index])))


def resolution_bucket_size(
    resolution: int, aspect_ratio: float, alignment: int,
) -> tuple[int, int]:
    """Return aligned H/W with approximately resolution**2 pixels."""
    if resolution <= 0 or aspect_ratio <= 0 or alignment <= 0:
        raise ValueError("resolution, aspect ratio, and alignment must be positive")
    height = resolution / math.sqrt(aspect_ratio)
    width = resolution * math.sqrt(aspect_ratio)
    return (
        max(alignment, round(height / alignment) * alignment),
        max(alignment, round(width / alignment) * alignment),
    )


class AspectResolutionDataset(Dataset):
    """Attach an explicit target H/W to rows routed by the bucket sampler."""

    def __init__(
        self, dataset: Dataset, *, resolution_levels: tuple[int, ...],
        aspect_ratios: tuple[float, ...], alignment: int, progress_callback=None,
    ):
        self.dataset = dataset
        self.resolution_levels = resolution_levels
        self.aspect_ratios = aspect_ratios
        self.alignment = alignment
        self.aspect_ids = []
        for index in range(len(dataset)):
            width, height = target_image_size(dataset, index)
            self.aspect_ids.append(nearest_aspect_bucket(width, height, aspect_ratios))
            if progress_callback is not None:
                progress_callback(1)

    def __len__(self) -> int:
        return len(self.dataset)

    def bucket_size(self, resolution_id: int, aspect_id: int) -> tuple[int, int]:
        return resolution_bucket_size(
            self.resolution_levels[resolution_id],
            self.aspect_ratios[aspect_id], self.alignment,
        )

    def __getitem__(self, key):
        if isinstance(key, tuple):
            index, resolution_id, aspect_id = key
            example = dict(self.dataset[index])
            example["_bucket_size"] = self.bucket_size(resolution_id, aspect_id)
            return example
        return self.dataset[key]


class AspectResolutionBatchSampler(Sampler[list[tuple[int, int, int]]]):
    """Weighted source sampling grouped by resolution and target aspect."""

    def __init__(
        self, dataset: AspectResolutionDataset, weights: torch.Tensor, *,
        num_samples: int, batch_size: int, seed: int,
    ):
        if weights.shape != (len(dataset),) or (weights <= 0).any():
            raise ValueError("sampling weights must be positive with one value per row")
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        self.dataset = dataset
        self.weights = weights.double()
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        indices_by_aspect = [
            torch.tensor([i for i, value in enumerate(dataset.aspect_ids) if value == aspect_id])
            for aspect_id in range(len(dataset.aspect_ratios))
        ]
        self.indices_by_aspect = indices_by_aspect
        mass = torch.tensor([
            self.weights[indices].sum().item() if indices.numel() else 0.0
            for indices in indices_by_aspect
        ], dtype=torch.double)
        category_mass = (
            (mass / mass.sum()).repeat(len(dataset.resolution_levels))
            / len(dataset.resolution_levels)
        )
        ideal = category_mass * self.num_samples
        counts = ideal.floor().long()
        remainder = self.num_samples - int(counts.sum().item())
        if remainder:
            fractions = ideal - counts
            counts[torch.argsort(fractions, descending=True)[:remainder]] += 1
        self.category_counts = counts.reshape(
            len(dataset.resolution_levels), len(dataset.aspect_ratios),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(
            math.ceil(int(count) / self.batch_size)
            for count in self.category_counts.flatten().tolist() if count > 0
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for resolution_id in range(len(self.dataset.resolution_levels)):
            for aspect_id, indices in enumerate(self.indices_by_aspect):
                count = int(self.category_counts[resolution_id, aspect_id].item())
                if not count:
                    continue
                draws = torch.multinomial(
                    self.weights[indices], count, replacement=True, generator=generator,
                )
                selected = indices[draws].tolist()
                order = torch.randperm(len(selected), generator=generator).tolist()
                tagged = [(index, resolution_id, aspect_id) for index in
                          (selected[position] for position in order)]
                batches.extend(
                    tagged[start:start + self.batch_size]
                    for start in range(0, len(tagged), self.batch_size)
                )
        order = torch.randperm(len(batches), generator=generator).tolist()
        yield from (batches[position] for position in order)


def _as_rgb_image(
    value: Any, *, field: str, image_root: Path | None = None,
) -> Image.Image:
    """Decode a Hugging Face image value (or a list containing one)."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Image field '{field}' is empty")
        value = value[0]
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute() and image_root is not None:
            path = image_root / path
        if not path.is_file():
            raise FileNotFoundError("Image for field " + repr(field) + " not found: " + str(path))
        return Image.open(path).convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        path = value.get("path")
        if path:
            return Image.open(path).convert("RGB")
    raise TypeError(
        f"Image field '{field}' must decode to PIL.Image, got {type(value).__name__}"
    )


def _image_dimensions(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("image list is empty")
        value = value[0]
    if isinstance(value, Image.Image):
        return value.size
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            with Image.open(io.BytesIO(raw)) as image:
                return image.size
        value = value.get("path")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.size
    raise TypeError(f"cannot read image dimensions from {type(value).__name__}")


def _captions(value: Any) -> list[str]:
    """Flatten the caption representations used by the COCO dataset card."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        for key in ("caption", "text", "captions"):
            if key in value:
                return _captions(value[key])
        return []
    if isinstance(value, (list, tuple)):
        output: list[str] = []
        for item in value:
            output.extend(_captions(item))
        return output
    return []


class CocoCaptionDataset(Dataset):
    """T2I adapter for ``mm-eval/MS-COCO-Captions`` rows."""

    def __init__(self, rows):
        self.rows = rows
        if len(rows) == 0:
            raise ValueError("MS-COCO-Captions split is empty")
        self._image_key, self._caption_key = self._resolve_columns(rows)

    def __getstate__(self):
        """Send Arrow file paths to spawned workers instead of pickling table data."""
        cache_files = getattr(self.rows, "cache_files", ())
        arrow_paths = [entry.get("filename") for entry in cache_files]
        if arrow_paths and all(
            isinstance(path, str) and Path(path).is_file() for path in arrow_paths
        ):
            return {
                "arrow_cache_files": arrow_paths,
                "image_key": self._image_key,
                "caption_key": self._caption_key,
            }
        # Small/in-memory datasets keep their existing generic pickle behavior.
        return {
            "rows": self.rows,
            "image_key": self._image_key,
            "caption_key": self._caption_key,
        }

    def __setstate__(self, state):
        if "arrow_cache_files" in state:
            from datasets import Dataset, concatenate_datasets

            shards = [
                Dataset.from_file(path) for path in state["arrow_cache_files"]
            ]
            rows = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        else:
            rows = state["rows"]
        self.rows = rows
        self._image_key = state["image_key"]
        self._caption_key = state["caption_key"]

    @staticmethod
    def _resolve_columns(rows) -> tuple[str, str]:
        columns = set(getattr(rows, "column_names", ()))
        image_key = next((key for key in ("media", "image") if key in columns), None)
        caption_key = next(
            (key for key in ("answer", "caption", "captions", "text") if key in columns),
            None,
        )
        if image_key is None or caption_key is None:
            raise ValueError(
                "MS-COCO-Captions needs an image column ('media' or 'image') and "
                "caption column ('answer', 'caption', 'captions' or 'text'); "
                f"found columns: {sorted(columns)}"
            )
        return image_key, caption_key

    def __len__(self) -> int:
        return len(self.rows)

    def target_size(self, index: int) -> tuple[int, int]:
        return _image_dimensions(self.rows[index].get(self._image_key))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        captions = _captions(row.get(self._caption_key))
        if not captions:
            raise ValueError(f"MS-COCO-Captions row {index} has no usable caption")
        caption = random.choice(captions)
        return {
            "target_image": _as_rgb_image(row.get(self._image_key), field=self._image_key),
            "source_image": None,
            "prompt": caption,
            "source": COCO_CAPTIONS_ID,
            "sample_id": str(row.get("id", row.get("file_name", index))),
            "conditioning_type": "t2i",
        }


class MultiEditDataset(Dataset):
    """Local MultiEdit train split with source, edited image, and instruction."""

    def __init__(self, rows, *, data_root: str | Path):
        self.rows = rows
        self.data_root = Path(data_root).expanduser().resolve()
        if not rows:
            raise ValueError("MultiEdit split is empty")
        required = {"original_images", "generated_images", "edit_prompt"}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("MultiEdit row " + str(index) + " must be a JSON object")
            missing = required - set(row)
            if missing:
                raise ValueError(
                    "MultiEdit row " + str(index) + " is missing fields " + repr(sorted(missing))
                )
        valid_rows = [
            row for row in rows
            if isinstance(row.get("edit_prompt"), str) and row["edit_prompt"].strip()
        ]
        skipped_rows = len(rows) - len(valid_rows)
        if skipped_rows:
            print(
                "MultiEdit: skipped " + str(skipped_rows)
                + " rows with empty or non-text edit_prompt"
            )
        self.rows = valid_rows
        if not self.rows:
            raise ValueError("MultiEdit split has no rows with a usable edit_prompt")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image_path(self, value: Any, *, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("MultiEdit field " + repr(field) + " must be a path string")
        path = Path(value)
        if path.is_absolute() and path.is_file():
            return path
        roots = (self.data_root, self.data_root / "multiedit", self.data_root.parent)
        relative_candidates = [path]
        if path.parts and path.parts[0].lower() == "multiedit":
            relative_candidates.append(Path(*path.parts[1:]))
        candidates = [root / relative for root in roots for relative in relative_candidates]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "MultiEdit image for " + repr(field) + " was not found; tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def target_size(self, index: int) -> tuple[int, int]:
        path = self._resolve_image_path(
            self.rows[index].get("generated_images"), field="generated_images",
        )
        with Image.open(path) as image:
            return image.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt = row.get("edit_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("MultiEdit row " + str(index) + " has no edit_prompt")
        source_path = self._resolve_image_path(
            row.get("original_images"), field="original_images",
        )
        target_path = self._resolve_image_path(
            row.get("generated_images"), field="generated_images",
        )
        return {
            "target_image": _as_rgb_image(target_path, field="generated_images"),
            "source_image": _as_rgb_image(source_path, field="original_images"),
            "prompt": prompt.strip(),
            "source": MULTI_EDIT_ID,
            "sample_id": str(row.get("id", index)),
            "conditioning_type": "ti2i",
        }


def load_multi_edit_dataset(
    data_root: str | Path, *, split: str = "train",
) -> MultiEditDataset:
    root = Path(data_root).expanduser().resolve()
    manifest_candidates = (
        root / "multiedit" / (split + ".jsonl"),
        root / (split + ".jsonl"),
    )
    manifest = next((path for path in manifest_candidates if path.is_file()), None)
    if manifest is None:
        raise FileNotFoundError(
            "MultiEdit manifest not found; expected one of: "
            + ", ".join(str(path) for path in manifest_candidates)
            + ". Accept the gated HF dataset and download/extract its files first."
        )
    manifest_text = manifest.read_text(encoding="utf-8").strip()
    rows = []
    if manifest_text.startswith("["):
        parsed = json.loads(manifest_text)
        if not isinstance(parsed, list):
            raise ValueError("MultiEdit JSON manifest must contain a list")
        rows = parsed
    else:
        for line_number, line in enumerate(manifest_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON in " + str(manifest) + ":" + str(line_number)
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    "Expected an object in " + str(manifest) + ":" + str(line_number)
                )
            rows.append(row)
    return MultiEditDataset(rows, data_root=root)


def load_hf_image_sources(
    *,
    multi_edit_data_root: str | Path,
    edit_split: str = "train",
    coco_split: str = "val",
    cache_dir: str | None = None,
):
    """Load COCO T2I and local image assets for the gated MultiEdit dataset."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("COCO loading requires the datasets package") from error
    kwargs = {"split": coco_split}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    try:
        coco_rows = load_dataset(COCO_CAPTIONS_ID, **kwargs)
    except Exception as error:
        raise RuntimeError(
            "Could not load split " + repr(coco_split)
            + " from " + COCO_CAPTIONS_ID
        ) from error
    return CocoCaptionDataset(coco_rows), load_multi_edit_dataset(
        multi_edit_data_root, split=edit_split,
    )


def make_mixed_sampler(
    coco: Dataset,
    ti2i: Dataset,
    *,
    coco_weight: float = 1.0,
    ti2i_weight: float = 1.0,
    seed: int = 42,
    num_samples: int | None = None,
    replacement: bool = True,
) -> tuple[ConcatDataset, WeightedRandomSampler]:
    """Build weighted sampling at the requested source-level ratio.

    By default the epoch length is the sum of source lengths. num_samples can
    shorten or extend it while preserving the requested source-level mixture
    when replacement is enabled. With replacement disabled, num_samples must
    not exceed the combined dataset size; requesting the full size visits each
    row exactly once in a weighted random order. Source weights are normalized
    per example, so unequal corpus sizes do not silently set the mixture ratio.
    """
    if coco_weight <= 0 or ti2i_weight <= 0:
        raise ValueError("dataset mixture weights must both be positive")
    if len(coco) == 0 or len(ti2i) == 0:
        raise ValueError("both T2I and TI2I datasets must be non-empty")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive when provided")
    mixed = ConcatDataset((coco, ti2i))
    if not replacement and num_samples is not None and num_samples > len(mixed):
        raise ValueError("num_samples cannot exceed dataset size without replacement")
    per_example_weights = [coco_weight / len(coco)] * len(coco)
    per_example_weights.extend([ti2i_weight / len(ti2i)] * len(ti2i))
    sampler = WeightedRandomSampler(
        per_example_weights,
        num_samples=len(mixed) if num_samples is None else num_samples,
        replacement=replacement,
        generator=torch.Generator().manual_seed(seed),
    )
    return mixed, sampler


def collate_hf_image_examples(
    examples: list[dict[str, Any]], *, resolution: int | None = None,
    resolution_levels: tuple[int, ...] | None = None,
    aspect_ratios: tuple[float, ...] = (0.5, 0.5625, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 16 / 9, 2.0),
    alignment: int = 32,
) -> dict[str, Any]:
    """Resize target/source pairs to one chosen, aspect-aware VAE bucket."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    if resolution is not None and resolution <= 0:
        raise ValueError("resolution must be positive when provided")
    if resolution_levels is None:
        levels = (resolution,) if resolution is not None else ()
        # Preserve the legacy square-only contract for VFP-DiT callers that do
        # not opt into the VFP-DiT Simple bucket metadata.
    else:
        levels = resolution_levels
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("resolution or positive resolution_levels are required")

    bucket_sizes = []
    for example in examples:
        chosen_size = example.get("_bucket_size")
        if chosen_size is None:
            target = example["target_image"]
            if not isinstance(target, Image.Image):
                raise TypeError("dataset target images must be PIL.Image.Image instances")
            if resolution_levels is None:
                chosen_size = (resolution, resolution)
            else:
                target_aspect = target.width / target.height
                chosen_size = resolution_bucket_size(
                    random.choice(levels), target_aspect, alignment,
                )
        bucket_sizes.append(tuple(chosen_size))
    if any(size != bucket_sizes[0] for size in bucket_sizes):
        raise ValueError("batch contains multiple aspect/resolution buckets")

    bucket_height, bucket_width = bucket_sizes[0]

    def fit(image):
        if image is None:
            return None
        if not isinstance(image, Image.Image):
            raise TypeError("dataset images must be PIL.Image.Image instances")
        return ImageOps.fit(
            image.convert("RGB"),
            (bucket_width, bucket_height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

    metadata = torch.tensor(
        [math.log(math.sqrt(bucket_width * bucket_height)),
         math.log(bucket_width / bucket_height)],
        dtype=torch.float32,
    ).expand(len(examples), -1).clone()
    return {
        "target_images": [fit(example["target_image"]) for example in examples],
        "source_images": [fit(example["source_image"]) for example in examples],
        "prompts": [example["prompt"] for example in examples],
        "sources": [example["source"] for example in examples],
        "sample_ids": [example["sample_id"] for example in examples],
        "conditioning_types": [example["conditioning_type"] for example in examples],
        "metadata": metadata,
        "bucket_size": (bucket_height, bucket_width),
    }
