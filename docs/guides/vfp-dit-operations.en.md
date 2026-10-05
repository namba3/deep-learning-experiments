# VFP-DiT training operations

This guide collects runtime and experiment procedures that complement the [VFP-DiT README](../../vfp_dit/README.md). Commands are run from the repository root. Launcher defaults are experiment presets; inspect their definitions and the CLI help before changing a run.

`run_vfp_dit_screen.sh` is the shared training runner used by the memory,
checkpointing, and Ada comparison presets. Those launchers set experiment
specific `VFP_DIT_*` values and call the shared runner, which builds the common
training CLI and handles optional sample generation. `run_vfp_dit_lr_screen.sh`
uses the training CLI directly because its fixed-LR arms require a shared
starting checkpoint. Keep these separate entrypoints while their budgets,
initialization, or comparison protocol differ.

## Encoder placement and memory

To free VRAM used by both frozen encoders, load Qwen and the VAE on CPU and
encode the next batches on one background thread while the DiT trains:

```bash
VFP_DIT_ENCODER_DEVICE=cpu \
VFP_DIT_ENCODER_PREFETCH_BATCHES=2 \
bash benchmarks/launchers/run_vfp_dit_screen.sh
```

To keep Qwen on CPU while placing only the VAE on the training GPU (for faster
VAE encode/decode while leaving Qwen weights off VRAM), use:

```bash
VFP_DIT_ENCODER_DEVICE=cpu \
VFP_DIT_VAE_DEVICE=training \
bash benchmarks/launchers/run_vfp_dit_screen.sh
```

Mixed placement encodes synchronously. Batch prefetch is reserved for the case
where both Qwen and the VAE are on CPU, avoiding a background thread that would
run VAE kernels concurrently with DiT training on the GPU.

The screen launcher runs Python GC and flushes unused CUDA allocator cache
every 10 consumed batches by default. Override these with
`VFP_DIT_GC_INTERVAL` and `VFP_DIT_EMPTY_CACHE_INTERVAL`; for example,
set the latter to `1` to return unused cache after every batch. Frequent cache
flushes can slow allocation reuse. Neither GC nor `empty_cache()` can reclaim
live model/optimizer tensors or lower the peak required by an active
forward/backward pass. Set the interval to `0` to disable cache flushing; the
general Python CLI uses the same defaults.

## Memory metrics

Each CUDA training epoch also records `train_peak_allocated_gib` /
`train_peak_reserved_gib` plus a storage breakdown in `metrics.jsonl`:
`train_model_parameters_gib`, `train_model_buffers_gib`,
`train_optimizer_state_gib`, and `train_gradients_at_epoch_end_gib`. The
`train_peak_overhead_over_persistent_gib` remainder is **not pure activation
memory**; it includes activations, gradients and temporary/kernel workspaces at
the peak. `train_end_unclassified_allocated_gib` shows live PyTorch allocations
not accounted for by model and optimizer storage at epoch end. Allocator
reserved memory remains separate from allocated/live tensor storage.

For a bounded CUDA resource check with these fields and synchronized component
timings, run:

```bash
bash benchmarks/launchers/run_vfp_dit_memory_profile.sh
```

The launcher defaults to one epoch of eight 256px samples, one timestep per
image, width 1024 / depth 24, CPU encoders with one prefetched batch, routine
GC every 100 batches, and no CUDA cache flushing, validation, or sample
generation. Override individual `VFP_DIT_*` variables to adjust the probe.
For the next resolution rung, run:

```bash
VFP_DIT_RESOLUTION=384 VFP_DIT_RESOLUTION_LEVELS=384 bash benchmarks/launchers/run_vfp_dit_memory_profile.sh
```

This pins all eight default samples to the 384px level rather than randomly
mixing the nominal resolution's three levels. Without
`VFP_DIT_RESOLUTION_LEVELS`, the 384px nominal resolution samples 192, 288,
and 384 levels. The run name includes `384px` and a timestamp.
Inspect the run's `metrics.jsonl` for persistent-storage and peak-overhead
fields; the peak remainder is not an activation-only measurement.
To compare all completed or in-progress runs that have CUDA memory metrics in
one table, regenerate the report with:

```bash
PYTHONPATH=. python3 -m vfp_dit.summarize_memory_profiles
```

The report is written to `vfp_dit/output/memory_profile_summary.md`.

`--encoder-device` controls Qwen placement; `--vae-device` controls VAE
placement and defaults to the Qwen setting. Each encoder uses BF16 weights when
placed on CUDA with BF16 AMP, and FP32 on CPU. Prefetch is enabled only when
both encoders are on CPU; mixed placement runs encoding synchronously and then
copies latent/condition tensors to the DiT device as needed. The same placement
is used for validation and sample generation. CPU encoding can shift the
bottleneck to host compute and RAM, while GPU VAE placement consumes VRAM, so
compare component times and peak allocation on a short run first. Encoder
placement and queue depth are runtime settings and can change on resume.

## Comparison launchers

The matched metadata Ada screen compares a no-metadata baseline with every configured scale mapping, with shift disabled and enabled. It creates one seeded no-metadata initialization and loads its matching tensors into every arm, so common backbone weights match and newly introduced Ada projections start at identity/zero. For example:

```bash
VFP_DIT_SEED=42 \
VFP_DIT_ADA_SCALE_MAPPINGS=linear,one_plus_silu,two_sigmoid,silu1_normalized,softplus1_normalized \
VFP_DIT_ADA_SHIFT_VALUES=0,1 \
bash benchmarks/launchers/run_vfp_dit_ada_scale_shift.sh
```

The Ada scale/shift comparison launcher writes `ada_scale_shift_comparison.md` under the output directory; it does not select a winning variant. Start a fresh optimizer in every arm; this runner does not resume optimizer state between variants. Its report is assembled by `python3 -m vfp_dit.summarize_metadata_ada`.

The comparison runner is restart-aware: it skips an arm that already has a completed run with the exact arm name, and resumes an incomplete arm from its latest epoch checkpoint when one exists. If it stopped before its first checkpoint, it restarts that arm from the shared no-metadata initialization. Each resumed attempt gets a new run directory, preserving the interrupted run's progress and checkpoint for inspection. The final report lists the configured arms, marks missing or unfinished arms, and shows a short SHA-256 prefix for each arm's shared initialization checkpoint; differing hashes flag a non-matched initialization. It also audits recorded seed, dataset/split, resolution buckets, architecture, optimizer, LR schedule, and training-budget fields across attempts. Hashes are computed from the checkpoint file when the report is generated.

## Diagnostics and resume

To inspect Ada behavior during training, add `--log-metadata-diagnostics` or set `VFP_DIT_LOG_METADATA_DIAGNOSTICS=1` with the launcher. The Ada comparison runner enables it by default. Each optimizer-step progress update and epoch/observation metrics then include attention/FFN metadata-scale mean, min, max, near-zero fraction, negative fraction, and shift RMS/max when shift is enabled. The consolidated comparison report shows these statistics at each run's best validation epoch. These are metadata-derived values; the FFN statistics exclude the separate timestep scale. The option is off by default and recomputes only the small Ada projections for diagnostics.

The screen launcher accepts `VFP_DIT_RESUME=./path/to/checkpoint_latest.safetensors` to continue from a saved full-resume checkpoint while retaining the configured epoch budget.

For non-finite loss diagnosis, the launcher accepts
`VFP_DIT_CHECK_FINITE_UPDATES=1`. The debug mode reports sample IDs and
summarizes non-finite latent/condition/DiT tensors when loss first becomes
invalid; it also checks gradients before and trainable parameters after each
optimizer update. For a forward/backward anomaly trace on one batch, set
`VFP_DIT_ANOMALY_DETECTION_BATCH` to its 1-based index (0 disables tracing).
For example, set it to `9` to inspect the batch where the current short run
first produced non-finite gradients. Anomaly tracing is expensive, so restrict
it to the suspected batch and leave both diagnostics off for normal runs.

Pressing Ctrl-C during VFP training requests a graceful stop: the trainer
finishes the current optimizer step, writes `checkpoint_latest.safetensors`
and its `.resume.pt` sidecar, then exits with `status=interrupted`. For a
mid-epoch stop, the checkpoint also stores the sampler order and consumed
position, so resuming with `VFP_DIT_RESUME=<checkpoint>` continues at the
next unconsumed batch. Worker-side random choices such as caption selection may
not replay bit-for-bit after a restart. A second Ctrl-C aborts immediately and
may interrupt checkpoint writing.

## Bounded training screens

For a bounded quality screen, use the launcher below. It supports `VFP_DIT_*` environment variables for run budget and model configuration; inspect the launcher definitions for available overrides. The trainer records progress and metrics under the run output directory. Training observations and generated samples can be disabled or reconfigured with the training CLI options; sampling pauses training while inference runs. Validation reports mixed, T2I, and TI2I losses with sample counts.

```bash
bash benchmarks/launchers/run_vfp_dit_screen.sh
```

The screen launcher supports a dry-run mode and can generate a consolidated Markdown report with `PYTHONPATH=. python3 -m vfp_dit.summarize_screen --run-dir <run-dir>`. The report combines the current progress snapshot, epoch validation metrics, interval losses, and links to observed/final sample images.

The screen launcher defaults to APOLLO rank 32. Its architecture can be scaled with
VFP_DIT_MODEL_WIDTH, VFP_DIT_DEPTH, VFP_DIT_HEADS,
VFP_DIT_KV_HEADS, VFP_DIT_ADAPTER_DEPTH,
VFP_DIT_ADAPTER_HEADS, and VFP_DIT_FF_MULT; defaults retain the full model.

Historical VFP-DiT resource probes are summarized in the [resource-screening record](../experiment_data/vfp-dit-resource-screening.md). The record retains aggregate measurements only; it does not establish current throughput or model quality.

## Checkpointing comparison

To measure checkpointing overhead under a matched budget, run both short arms with the same seed and data split:

```bash
bash benchmarks/launchers/run_vfp_dit_checkpointing_comparison.sh
```

This defaults to a short matched training/validation budget. The launcher writes a comparison report with each arm's status, losses, peak memory, and throughput. The report is for resource comparison; its short validation does not establish quality. The post-run aggregation helper is `python3 -m vfp_dit.summarize_checkpointing`.
