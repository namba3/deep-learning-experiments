"""Shared aspect-bucket definitions for Mini-ImageNet input preparation."""

from __future__ import annotations

import math
from collections.abc import Sequence

# Height/width pairs approximate the dominant source aspect modes while keeping
# the last-stage token count close to the 64x64 baseline.
DEFAULT_BUCKET_SIZES: tuple[tuple[int, int], ...] = (
    (80, 56),
    (72, 56),
    (64, 64),
    (56, 72),
    (56, 80),
)


def assign_bucket(width: int, height: int, bucket_sizes: Sequence[tuple[int, int]]) -> int:
    """Choose the target H/W closest in log aspect ratio to a source image."""
    if width <= 0 or height <= 0:
        raise ValueError("source image dimensions must be positive")
    if not bucket_sizes:
        raise ValueError("at least one image bucket is required")
    aspect = width / height
    return min(
        range(len(bucket_sizes)),
        key=lambda index: abs(math.log(aspect / (bucket_sizes[index][1] / bucket_sizes[index][0]))),
    )
