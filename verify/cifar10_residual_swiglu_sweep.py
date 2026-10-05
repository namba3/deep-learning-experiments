"""Backward-compatible import path for the renamed RGLU-LoRA sweep."""

from .cifar10_rglu_lora_sweep import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
