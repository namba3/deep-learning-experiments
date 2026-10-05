"""Dedicated entrypoint for CIFAR-10 low-rank adapter training.

The training loop and checkpoint implementation remain shared with
``cifar10.train``. This entrypoint enforces that an adapter is supplied and
supports either a base/adapter checkpoint or deterministic random base
initialization, while keeping the legacy adapter flags in ``train.py``
compatible with existing commands.
"""

from __future__ import annotations

from cifar10.train import main as _train_main


def main(argv=None):
    return _train_main(argv, adapter_only=True)


if __name__ == "__main__":
    main()
