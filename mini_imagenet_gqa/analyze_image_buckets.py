"""Analyze Mini-ImageNet source dimensions and a compute-matched aspect bucket plan.

Run with HF_DATASETS_CACHE pointing at the Hugging Face Datasets cache, or pass
--cache-dir. The image bytes are decoded from the existing cache; no copy is made.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from datasets import DownloadConfig, load_dataset

from mini_imagenet_gqa.buckets import DEFAULT_BUCKET_SIZES, assign_bucket

BUCKETS = tuple(
    (f"{height}x{width}", height, width)
    for height, width in DEFAULT_BUCKET_SIZES
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def analyze_split(data) -> dict:
    shapes: Counter[tuple[int, int]] = Counter()
    aspects: list[float] = []
    areas: list[float] = []
    short_sides: list[float] = []
    bucket_counts = Counter()
    crop_retention: list[float] = []
    for row in data:
        width, height = row["image"].size
        if width <= 0 or height <= 0:
            continue
        aspect = width / height
        shapes[(width, height)] += 1
        aspects.append(aspect)
        areas.append(float(width * height))
        short_sides.append(float(min(width, height)))
        bucket_index = assign_bucket(width, height, DEFAULT_BUCKET_SIZES)
        bucket_name, bucket_h, bucket_w = BUCKETS[bucket_index]
        bucket_counts[bucket_name] += 1
        bucket_aspect = bucket_w / bucket_h
        crop_retention.append(min(aspect / bucket_aspect, bucket_aspect / aspect))

    qs = (0, .01, .05, .25, .5, .75, .95, .99, 1)
    labels = ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")
    return {
        "count": len(aspects),
        "distinct_shapes": len(shapes),
        "common_shapes": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in shapes.most_common(12)
        ],
        "aspect_ratio_quantiles": {
            label: percentile(aspects, q) for label, q in zip(labels, qs)
        },
        "pixel_area_quantiles": {
            label: int(round(percentile(areas, q))) for label, q in zip(labels, qs)
        },
        "short_side_quantiles": {
            label: int(round(percentile(short_sides, q))) for label, q in zip(labels, qs)
        },
        "bucket_assignment": {
            name: {
                "height": height,
                "width": width,
                "tokens_after_three_downsamples": (height // 8) * (width // 8),
                "images": bucket_counts[name],
                "fraction": bucket_counts[name] / len(aspects),
            }
            for name, height, width in BUCKETS
        },
        "crop_retained_area_fraction": {
            "p01": percentile(crop_retention, .01),
            "p05": percentile(crop_retention, .05),
            "p50": percentile(crop_retention, .50),
            "p95": percentile(crop_retention, .95),
            "at_least_90pct": sum(value >= .90 for value in crop_retention) / len(aspects),
            "at_least_80pct": sum(value >= .80 for value in crop_retention) / len(aspects),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="timm/mini-imagenet")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    kwargs = {}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    download_config = DownloadConfig(local_files_only=args.offline)
    dataset = load_dataset(args.dataset, download_config=download_config, **kwargs)
    report = {
        "dataset": args.dataset,
        "bucket_plan": [
            {"name": name, "height": h, "width": w, "aspect_ratio": w / h,
             "tokens": (h // 8) * (w // 8)}
            for name, h, w in BUCKETS
        ],
        "splits": {name: analyze_split(split) for name, split in dataset.items()},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
