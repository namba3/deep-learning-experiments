"""Device selection helpers shared by training entry points."""

import argparse

import torch


DEVICE_CHOICES = ("auto", "cpu", "cuda")


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common training device option to ``parser``."""
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Training device: auto, cpu, or cuda. Default: auto.",
    )


def resolve_device(
    requested: str, *, default: torch.device | None = None,
) -> torch.device:
    """Resolve a CLI device value and fail early for unavailable CUDA."""
    if requested == "auto":
        if default is not None:
            return default
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is not available"
            )
        return torch.device("cuda")
    return torch.device("cpu")
