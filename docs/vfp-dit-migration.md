# VFP-DiT migration note

## Current entrypoints

The maintained VFP-DiT implementation is in [`../vfp_dit/`](../vfp_dit/). Shared Qwen, image-data, and training-runtime components are in [`../vfp_dit_runtime/`](../vfp_dit_runtime/).

- Train: `python3 -m vfp_dit.train`
- Generate samples: `python3 -m vfp_dit.generate_samples`
- New run output: `vfp_dit/output/`
- Historical run evidence: [`experiment_data/`](experiment_data/), indexed by [the experiment-data README](experiment_data/README.md)

The old `vfp_dit_simple/` Python source package is retired and is not an active package in this repository. Its former `output/` run tree was moved under `vfp_dit/output/`. Historical run evidence retained in this repository excludes checkpoints and resume state; archived run directory names retain their original IDs.

Current VFP-DiT launchers use the `VFP_DIT_*` prefix for environment overrides. The retired `VFP_SIMPLE_*` launcher variables are no longer supported. The legacy checkpoint stage described below remains accepted for weights compatibility.

## Checkpoint stage compatibility

The current trainer writes checkpoints with stage `vfp_dit.train`. The shared checkpoint loader also accepts the former `vfp_dit_simple.train` stage for loading or initialization from those checkpoints. This compatibility is implemented in [`vfp_dit_runtime/training.py`](../vfp_dit_runtime/training.py).

The legacy staged trainer stage `vfp_dit.train_dit` is not considered equivalent to `vfp_dit.train` and is not accepted as a current trainer checkpoint. Checkpoint metadata remains authoritative; use the matching model configuration and the current CLI when resuming.

For current architecture, CLI, and sampling details, see the [VFP-DiT README](../vfp_dit/README.md).
