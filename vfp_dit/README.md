# VFP-DiT

> **Current model:** this is the primary VFP image-generation implementation. New runs write under `vfp_dit/output/`.

`vfp_dit/` implements VFP-DiT as one model with a condition-aware, spatially compressed front DiT and full-resolution output-refinement blocks. There is no VFCB training stage. For TI2I, stride-2 reference-latent features are fused with Qwen vision features on the resulting half-resolution grid; text tokens remain a separate sequence condition.

## Architecture and checkpoint compatibility

The model shape, conditioning, latent fusion, flow-matching, and checkpoint contracts are documented in the [VFP-DiT architecture guide](../docs/architecture/vfp-dit.md). The recorded adapter ablation is summarized in the [experiment archive](../docs/experiment_data/vfp-simple-adapter-ablation.md).

## Training

The default architecture uses width 1024, depth 24, 16 query heads, and 4 KV heads. APOLLO is the default optimizer (rank 32); its 1D parameters use AdamW-SF, while matrix parameters use state-size-selected `auto-sf` fallback. Start with a small dry-run before loading the frozen models:

```bash
PYTHONPATH=. python3 -m vfp_dit.train --dry-run \
  --device cpu --model-width 24 --depth 1 --heads 2 --kv-heads 1 \
  --condition-dim 20
```

A data preflight loads the configured COCO and MultiEdit splits, but does not load frozen model weights:

```bash
PYTHONPATH=. python3 -m vfp_dit.train --validate-only \
  --multi-edit-data-root data/MultiEdit --resolution 512
```

COCOとMultiEditの画像・annotationにはそれぞれの配布元の利用条件が適用されます。MultiEditはHub上でApache-2.0と表示されていますが、アクセス時に追加条件への同意が必要で、その条件は未確認です。埋め込み画像や生成画像の権利もデータセット全体のlicense表示だけでは確定しません。取得・利用・成果物共有の前に[出典と利用条件](../docs/data-model-provenance.md)を確認してください。

Training example:

```bash
PYTHONPATH=. python3 -m vfp_dit.train \
  --multi-edit-data-root data/MultiEdit --resolution 512 \
  --condition-layer final --condition-dropout 0.1 \
  --apollo-rank 32 \
  --device cuda --amp bf16 --batch-size 1 --epochs 10
```

Detailed encoder placement, memory profiling, comparison launchers, diagnostics, and resume operations are covered in the [VFP-DiT operations guide](../docs/guides/vfp-dit-operations.en.md).

All shared training, optimizer, resume, progress, and checkpoint options are listed by `python3 -m vfp_dit.train --help`. CPU/runtime checks and their limits are documented in the [verification guide](../verify/README.md); full training convergence and CUDA/BF16 training require separate validation.

## Sampling

Generate T2I samples from the checkpoint. The sampler reads the model dimensions, Qwen tap, and frozen model IDs from checkpoint metadata:

```bash
PYTHONPATH=. python3 -m vfp_dit.generate_samples \
  --checkpoint vfp_dit/output/<run>/checkpoints/checkpoint_latest.safetensors \
  --prompt "a small red boat on a quiet lake" --steps 30 --guidance-scale 4 --guidance-method tcfg
```

Use a separate solver and timestep scheduler. `--scheduler flow_match_euler` applies the static FlowMatch transform to a linear time grid; `--flow-shift 1` is the identity. `--scheduler uniform` keeps an unshifted linear grid. The generator writes a `.json` sidecar containing solver, schedule, shift, step count, CFG scale, base/per-sample seeds, and expected callback evaluations. CFG uses two network forwards for each callback evaluation.

Available solver choices are `euler`, `fireflow`, `abm2`, `er_sde`, `rf_ab2`, `rf_2m_warp`, `rf_trust_region`, `rf_er_sde_1`, `rf_er_sde_2m`, `rf_er_sde_trust`, `rf_er_sde_warp_1`, `rf_er_sde_warp_2m`, and `rf_er_sde_warp_trust`. The non-default `rf_*` solvers are experimental; API availability is not a recommendation, and focused numerical and model-quality validation remains pending for the variants in the [solver design index](../docs/sampling-solvers.md). The default is `euler`. Solver behavior and validation limits are summarized in [`flow_sampling/README.md`](../flow_sampling/README.md); detailed design records cover [RF-2M-Warp](../docs/rf-2m-warp-design.md), [RF trust-region](../docs/rf-trust-region-solver-design.md), and [RF-ER-SDE](../docs/rf-er-sde-design.md). For VFP-DiT, `er_sde` defaults to sigma max 80; warped variants require `--flow-shift 1`. The generated JSON sidecar records the selected solver and effective schedule parameters. Example:

```bash
PYTHONPATH=. python3 -m vfp_dit.generate_samples \
  --checkpoint vfp_dit/output/<run>/checkpoints/checkpoint_latest.safetensors \
  --prompt "a small red boat on a quiet lake" --steps 30 --guidance-scale 4 \
  --solver er_sde --scheduler flow_match_euler --flow-shift 1 --er-sde-sigma-max 80
```

For TI2I, repeat `--reference-image` once for each prompt. An empty value selects T2I for that prompt:

```bash
PYTHONPATH=. python3 -m vfp_dit.generate_samples \
  --checkpoint vfp_dit/output/<run>/checkpoints/checkpoint_latest.safetensors \
  --prompt "turn the sky into a sunset" --reference-image input.png
```

The default Euler sampler integrates the learned `epsilon - x0` velocity from noise at `t=1` toward data at `t=0`. Adaptive ABM is not implemented. For solver comparisons, hold checkpoint, grid, seed, and CFG fixed. Experimental solvers have the validation limits described in the shared solver guide.
