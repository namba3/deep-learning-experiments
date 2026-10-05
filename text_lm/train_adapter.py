"""Dedicated entrypoint for text LM low-rank adapter training."""

from __future__ import annotations

from text_lm.train import main as _train_main


def main(argv=None):
    return _train_main(argv, adapter_only=True)


if __name__ == "__main__":
    main()
