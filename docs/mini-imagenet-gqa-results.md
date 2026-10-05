# Mini-ImageNet GQA experiment results and follow-ups

This page preserves aggregate results and experiment protocols for Mini-ImageNet GQA screens. It is a dated research record, not a list of active runs. In each subsection, words such as “current”, “next”, and “paused” describe that note's time; completion is only established where a result is reported here. Run directories, generated JSON/Markdown files, and checkpoints are local artifacts and are excluded from the public tree; tables and stated measurements are the retained results. Current model options and launcher entrypoints are documented in the [package README](../mini_imagenet_gqa/README.md).

## Navigation

- [Retained protocol completion status](#completion-status-of-retained-protocols)
- [Initial and repeated architecture screens](#initial-full-split-screen-seed-42)
- [Optimizer and APOLLO-SF comparisons](#optimizer-and-apollo-sf-quantization-comparison)
- [Ada scale-function follow-up](#bounded-ada-scale-function-screen)
- [APOLLO norm-growth limiter ablation](#apollo-norm-growth-limiter-ablation)
- [AdaRMS shift/bias follow-up](#adarms-shiftbias-follow-up)
- [Aggregate evidence and interpretation limits](#aggregate-evidence-recorded-here)

## Completion status of retained protocols

The following protocols have no completion evidence or aggregate result in this public record. Their protocol status is retained for context, while their one-off launchers are omitted from the public tree.

| Protocol | Public record status |
| --- | --- |
| AdamW-SF scale-plus-shift bias sweep, seeds 47–50 | The scale-only and No-Ada controls are recorded. The two additional shift arms have no paired report here, so completion is unconfirmed; their one-off launcher is omitted from the public tree. |
| Query-only attention shift probe, seed 45 | The protocol is described, but this page has no outcome or completion evidence; its one-off launcher is omitted from the public tree. |
| Related six-condition scale sweep, seed 45 | This page has no report establishing whether the full sweep completed. The protocol is retained as context, but its one-off launcher is omitted from the public tree. |

Do not infer run status from launcher existence or local output paths mentioned in historical prose. Only a report-backed result is summarized here.

## Initial full-split screen (seed 42)

All seven variants completed 30 epochs on CUDA/BF16 with batch size 64, 50,000 training examples per epoch, and full 10,000-example validation and 5,000-example test splits. The test set was evaluated once using the checkpoint with the highest validation top-1. Each row below is one run, so `n=1` and no seed-to-seed variation is measured. Throughput is training images/second; CUDA memory is the maximum allocated/reserved training memory in MiB.

| Variant | Best val top-1 | Test top-1 | Images/s | Peak allocated/reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| `naive_gqa` | 37.02% | 35.56% | 536.40 | 686.55 / 734 |
| `naive_gqa_gated_ffn` | 36.58% | 35.56% | 497.55 | 714.18 / 774 |
| `gated_gqa_sigmoid` | 37.04% | 35.40% | 551.65 | 709.89 / 736 |
| `gated_gqa_silu` | 43.14% | 43.18% | 621.57 | 709.89 / 736 |
| `gated_gqa_sigmoid_gated_ffn` | 52.72% | 51.44% | 624.67 | 732.05 / 808 |
| `gated_gqa_silu_gated_ffn` | 53.77% | 52.22% | 558.32 | 732.05 / 808 |
| `ada_gated_gqa_silu_gated_ffn` | 48.32% | 46.22% | 559.71 | 895.45 / 962 |

In this seed-42 screen, a SiLU attention gate alone exceeded naive GQA by 7.62 test points, while a sigmoid gate alone and a GatedFFN alone were close to baseline. Combining gated GQA with GatedFFN reached 51.44% (sigmoid) and 52.22% (SiLU). The Ada and branch-injection results used pre-transform source dimensions, so they are historical results under a different condition definition and do not represent the bucket-conditioned protocol described above. Non-Ada variants have about 2.46M parameters; the Ada variant has 2.62M.

Treat these as screening results, not a stable architecture ranking. The generated JSON/Markdown summary and checkpoints were local artifacts and are not included here. At that point, the multi-seed sweep was paused while the bucketed input pipeline was screened; this records the earlier status, not the later sweep status.

## Archived bucket-conditioned metadata-concat screen (seed 42, three epochs)

This QKV metadata-concat research is closed. The historical exploratory screen used the five default image buckets, batch size 64, BF16, 50,000 training images, complete validation/test splits, and seed 42. Each run trained for three epochs; test top-1 below uses the best-validation checkpoint. Its generated report and checkpoints are local artifacts and are excluded from the public tree.

| Variant | Validation top-1, epochs 1 / 2 / 3 | Test top-1 | Train images/s, epoch 3 | Peak allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| `gated_gqa_silu_gated_ffn` | 6.02 / 8.64 / 12.22% | 12.42% | 458 | 814 / 908 |
| `branch_qkv_meta_gated_gqa_silu_gated_ffn` | 4.15 / 6.96 / 9.21% | 9.54% | 475 | 818 / 912 |
| `branch_qkv_meta_ffn_ada_gated_gqa_silu_gated_ffn` | 6.25 / 11.06 / 15.84% | 15.52% | 423 | 909 / 1,002 |
| `ada_gated_gqa_silu_gated_ffn` | 6.50 / 11.75 / 18.46% | 17.70% | 414 | 994 / 1,086 |
| `branch_qkv_meta_ada_attn_ffn_gated_gqa_silu_gated_ffn` | 6.25 / 10.22 / 16.58% | 15.88% | 406 | 998 / 1,090 |

In this short screen, AdaRMS on both attention and FFN without QKV metadata concat had the highest validation and test accuracy. Adding QKV metadata concat to that arm was lower by 1.88 validation points and 1.82 test points, while adding the concat projection raised peak allocated memory by about 4 MiB and lowered throughput by about 8 images/s. QKV concat without attention AdaRMS was the lowest-scoring arm. These are early signals only: the runs cover three epochs and one seed, and adding variant-specific modules changes random-number consumption during initialization, so same-seed runs do not guarantee identical shared weights. The test split was also evaluated for each screening arm. Use validation for follow-up selection, then confirm finalists with common shared-weight initialization, a longer budget, and fresh seeds before making an architecture claim.

## AdaRMS × gate/SwiGLU screen (seed 42, three epochs)

All four variants in this historical screen used the five default buckets, batch size 64, BF16, 50,000 training images/epoch, and complete validation/test splits. Test top-1 uses the best-validation checkpoint. Per-run reports and checkpoints are local artifacts and are excluded from the public tree.

| Variant | Validation top-1, epochs 1 / 2 / 3 | Test top-1 | Images/s | Peak allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| `naive_gqa` | 3.92 / 5.34 / 6.05% | 5.88% | 393 | 758 / 804 |
| `ada_naive_gqa` | 1.00 / 1.00 / 1.00% | 1.00% | 401 | 941 / 1,008 |
| `gated_gqa_silu_gated_ffn` | 6.36 / 9.46 / 13.29% | 13.12% | 413 | 814 / 908 |
| `ada_gated_gqa_silu_gated_ffn` | 4.38 / 6.27 / 9.07% | 8.94% | 409 | 994 / 1,086 |

In this one-seed screen, the paired SiLU gate + SwiGLU FFN improved test top-1 over the plain branches by 7.24 points without AdaRMS and 7.94 points with AdaRMS. Adding AdaRMS reduced test top-1 by 4.88 points in the plain pair and 4.18 points in the gated pair, while increasing peak allocated memory by about 182 MiB. Treat these as screening signals only. Notably, an earlier same-seed run of `ada_gated_gqa_silu_gated_ffn` under the same recorded protocol reached 18.46% validation and 17.70% test, versus 9.07% / 8.94% here. That repeat discrepancy is large enough that seed-42 accuracy is not stable evidence; inspect repeatability and use multiple seeds/common shared-weight initialization before choosing a winner.

### Repeatability follow-up

A replay probe rebuilt the first training batch twice with seed 42, four DataLoader workers, the bucket sampler, and training augmentation. Images, labels, and bucket metadata matched bit-for-bit. A separate fixed synthetic-batch CUDA/BF16 one-step probe with deterministic PyTorch algorithms produced identical loss and all 51 updated tensors. These checks show that the tested data and one-step paths can replay exactly; they do not isolate the cause of the full-run accuracy gap, because the earlier runs did not preserve initial checkpoints and were not run with deterministic algorithms.

The controlled follow-up used `--common-init --deterministic`, seed 42, BF16, batch 64, five buckets, full train/validation/test splits, and three epochs. Shared stem, attention, QK norm, AdaRMS affine, pooling, and classifier weights were copied from a seed-matched naive-GQA reference; gates, block FFNs, and metadata modules retained variant-specific initialization. Each run records its initial model SHA-256 and copied-parameter count.

| Variant | Validation top-1, epochs 1 / 2 / 3 | Test top-1 | Images/s | Peak allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| `naive_gqa` | 1.00 / 2.63 / 3.00% | 3.16% | 476 | 807 / 862 |
| `ada_naive_gqa` | 4.89 / 6.48 / 9.98% | 9.68% | 439 | 988 / 1,058 |
| `gated_gqa_silu_gated_ffn` | 7.60 / 10.85 / 19.12% | 18.82% | 446 | 861 / 928 |
| `ada_gated_gqa_silu_gated_ffn` | 8.55 / 16.54 / 23.91% | 22.44% | 489 ± 40 | 1,043 / 1,124 |

The Ada+gate condition was repeated with the same flags and seed. Both runs had the same initial-state hash, identical train and validation metrics at every epoch, identical test metrics, and bit-identical best-checkpoint tensors (70/70). This confirms full-run repeatability for this controlled configuration. The earlier same-seed divergence occurred without these controls.

A second seed-42 replay pair used `--deterministic` without `--common-init`. Both runs had identical printed train/validation metrics at every epoch and identical test metrics. This shows that deterministic algorithms were sufficient for same-variant metric replay in this run path; common initialization serves cross-variant shared-weight control. Checkpoint tensors were not compared for this pair, so bitwise checkpoint equality is established only for the common-init replay above. Per-run files are excluded from the public tree.

The same four-arm protocol was repeated at seeds 43 and 44. Each run copied 1,563,636 shared parameters from its seed-matched naive-GQA reference. The paired SiLU gate + SwiGLU condition outperformed naive GQA at all three seeds; Ada+gate varied substantially, from a 4.12% test score at seed 43 to 22.44% at seed 42. Per-run files are excluded from the public tree.

| Variant | Seed-42 best val / test top-1 | Seed-43 best val / test top-1 | Seed-44 best val / test top-1 | Three-seed mean test top-1 |
| --- | ---: | ---: | ---: | ---: |
| `naive_gqa` | 3.00 / 3.16% | 2.74 / 2.80% | 10.29 / 9.78% | 5.25 ± 3.93% |
| `ada_naive_gqa` | 9.98 / 9.68% | 5.44 / 5.42% | 9.93 / 9.36% | 8.15 ± 2.37% |
| `gated_gqa_silu_gated_ffn` | 19.12 / 18.82% | 11.78 / 10.94% | 17.61 / 16.88% | 15.55 ± 4.11% |
| `ada_gated_gqa_silu_gated_ffn` | 23.91 / 22.44% | 4.20 / 4.12% | 16.55 / 15.74% | 14.10 ± 9.27% |

Mean and sample standard deviation use three seeds; every run used three epochs. Gate+SwiGLU had the highest mean test top-1 and beat naive GQA at each seed. Ada's effect depended on the branch condition: it improved the plain pair at every seed, while its effect on the gated pair changed direction and had high variance. These results remain a short screening study, not an architecture ranking. The table retains per-seed validation and test top-1; raw run logs, detailed resource measurements, and initialization fingerprints are omitted from the public record.

## 10-epoch factorial follow-up (seeds 42–44)

The same four variants were trained from scratch for 10 epochs with common initialization and deterministic execution. All runs used the full train/validation/test splits, BF16, batch size 64, five aspect-ratio buckets, LR `1e-3` with cosine decay, and the same augmentation. Aggregate results are retained below; the generated comparison summary and per-run files are excluded from the public tree.

| Variant | Seed 42 | Seed 43 | Seed 44 | Mean test top-1 |
| --- | ---: | ---: | ---: | ---: |
| `naive_gqa` | 1.52% | 8.80% | 21.74% | 10.69% ± 10.24% |
| `ada_naive_gqa` | 22.08% | 12.78% | 19.28% | 18.05% ± 4.77% |
| `gated_gqa_silu_gated_ffn` | 37.20% | 24.20% | 31.94% | **31.11% ± 6.54%** |
| `ada_gated_gqa_silu_gated_ffn` | 38.84% | 3.04% | 34.80% | 25.56% ± 19.61% |

The SiLU attention gate + SwiGLU FFN without AdaRMS beat naive GQA at all three seeds and had the highest mean. The paired gate/FFN change raised the mean by 20.42 points without AdaRMS and by 7.51 points with AdaRMS. AdaRMS raised the plain GQA mean by 7.36 points, while the Ada+gate arm averaged 5.55 points below its non-Ada counterpart because seed 43 collapsed to 3.04%. Thus AdaRMS does not show a stable benefit when combined with the gate/FFN variant. The non-Ada gated model also used less memory (861 MiB peak allocated vs. 1,043 MiB for Ada+gate) with similar throughput (570 vs. 584 images/s).

These are three-seed screening results for a small 64px classifier. The high seed variation, especially for naive GQA and Ada+gate, prevents a definitive architecture claim. In the follow-up decision recorded at the time, `gated_gqa_silu_gated_ffn` was selected as the architecture candidate; the Ada+gate conditioning-LR intervention was treated as a separate stability result described below.

### Ada+gate seed-43 collapse diagnosis

The seed-43 `ada_gated_gqa_silu_gated_ffn` run fell from 3.09% validation top-1 at epoch 1 to 1.76% at epoch 2, then stayed near the 1% chance level. The gate-only run with the same seed reached 24.20% test top-1. Checkpoint inspection found that the Ada conditioning path expanded during training: mean metadata-embedding output norm rose from 6.13 at the best epoch-1 checkpoint to 20.52 at the final checkpoint, and the largest absolute Ada scale rose from 3.50 to 26.02. Scales were large across every bucket, so the collapse was not isolated to one resolution/aspect bucket. In the successful seed-42 and seed-44 Ada+gate runs, the final condition norms were 3.36 and 3.28 and the largest scales were 1.74 and 1.83.

A matched diagnostic lowered only the metadata-embedding and AdaRMS-projection learning rates to 0.1x; the rest of the model retained the same `1e-3` cosine schedule. All three intervention runs had the same initial-state SHA-256 as their corresponding standard-LR runs. Their test results were:

| Seed | Standard conditioning LR | 0.1x conditioning LR | Change | Best-checkpoint mean condition norm | Max absolute Ada scale |
|---:|---:|---:|---:|---:|---:|
| 42 | 38.84% | 38.74% | -0.10 pp | 11.19 | 1.80 |
| 43 | 3.04% | 33.92% | +30.88 pp | 10.14 | 1.79 |
| 44 | 34.80% | 31.76% | -3.04 pp | 10.35 | 1.91 |
| Mean ± sample SD | 25.56% ± 19.61% | 34.81% ± 3.57% | — | — | — |

The validation top-1 rose each epoch in the 0.1x runs, reaching 40.50%, 36.63%, and 33.65% for seeds 42–44. Lowering the conditioning LR recovered the collapsed seed 43, left seed 42 essentially unchanged, and reduced seed 44 by 3.04 points. At the selected test checkpoints, the mean condition embedding norms are higher than the successful standard-LR seed-42/44 checkpoints, but the learned Ada projections keep the largest absolute scales near 1.8–1.9 rather than allowing the scale explosion seen in the collapsed seed-43 checkpoint. This supports conditioning-path instability as a plausible contributor to the collapse, while the three-seed intervention does not establish it as the sole cause. The paired results reduce the Ada+gate test-score spread, but remain a small screening experiment.

The trainer exposes `--conditioning-lr-multiplier` for this ablation. Its default `1.0` keeps the previous single optimizer group and LR behavior; values other than 1.0 separate metadata-embedding/AdaRMS parameters into a group with the scaled LR. Keep the default unchanged until a broader controlled comparison evaluates whether the stability gain generalizes.

Relative to the earlier factorial screen, common initialization and deterministic execution both changed, so treat those measurements as a separate protocol.

Example: `python -m mini_imagenet_gqa.train --variant ada_gated_gqa_silu_gated_ffn --common-init --deterministic --seed 42 --epochs 3 --amp bf16`.

## Optimizer and APOLLO-SF quantization comparison

`train.py` also accepts `APOLLO-SF`, `APOLLO-SF-LRSF`, `APOLLO-SF-INT8-Z`,
`APOLLO-SF-INT8-Delta`, `APOLLO-SF-INT4-Z`, and
`APOLLO-SF-INT4-Delta`. The dedicated runner keeps one summary directory per
optimizer and writes a paired report against AdamW:

```bash
SEEDS="42" EPOCHS=3 \
OUTPUT_DIR=mini_imagenet_gqa/output/bucketed/apollo-sf-quantization \
bash mini_imagenet_gqa/run_optimizer_quantization_comparison.sh
```

For a lower-cost pipeline screen, add `STEPS_PER_EPOCH=100 EVAL_BATCHES=20`.
The runner supports resume and skips completed optimizer/seed arms. Its
default model condition is `gated_gqa_silu_gated_ffn`, the architecture
follow-up candidate selected in this record. `--apollo-sf-quant-block-size` defaults to 256.

For the low-rank `sf_delta` comparison, use the same runner with matched
initialization and rank. `APOLLO-SF-LRSF` stores the delta in a separate
orthonormal low-rank basis and reconstructs a full-rank temporary during the
update. The basis is fixed in this initial variant; APOLLO's own projection
refresh remains independent:

```bash
SEEDS="42" EPOCHS=3 APOLLO_RANK=32 \
OPTIMIZERS="APOLLO-SF-INT8-Delta,APOLLO-SF-LRSF" \
OUTPUT_DIR=mini_imagenet_gqa/output/bucketed/apollo-sf-delta-rank32 \
bash mini_imagenet_gqa/run_optimizer_quantization_comparison.sh
```

This comparison separates persistent state savings from temporary
reconstruction cost and peak VRAM.

The Mini-ImageNet trainer uses FP32 parameter storage with BF16 autocast. Thus
the unquantized `APOLLO-SF` arm currently stores its full-rank `z` in the
parameter dtype (FP32); the INT8/INT4 arms still use their explicitly
quantized state representations. Do not label the unquantized arm as a
full-rank BF16-`z` result without a separate BF16 parameter-storage protocol.

### Completed 10-epoch comparison

The rank-32 comparison completed for seeds 42–44. The table reports test top-1 mean ± sample SD, persistent optimizer state, and throughput:

| Optimizer | Test top-1 mean ± SD | State | Images/s |
|---|---:|---:|---:|
| AdamW | 31.11 ± 6.54% | 18.8 MiB | 584.6 |
| AdamW-SF | 41.04 ± 1.47% | 18.8 MiB | 592.2 |
| APOLLO | 34.05 ± 1.13% | 3.3 MiB | 584.0 |
| APOLLO-SF | 28.75 ± 1.45% | 12.7 MiB | 529.3 |
| APOLLO-SF-INT8-Delta | 28.65 ± 1.62% | 5.7 MiB | 541.1 |
| APOLLO-SF-INT4-Delta | 8.29 ± 0.30% | 4.5 MiB | 538.1 |
| APOLLO-SF-LRSF | 24.07 ± 1.40% | 5.3 MiB | 535.8 |

This is a protocol comparison, not an isolated optimizer-equation comparison: AdamW used epoch-cosine scheduling, APOLLO used optimizer-step warmup/decay, and Schedule-Free variants used their native schedule. INT8-Delta was close to APOLLO-SF while reducing persistent state; INT4-Delta had sharply lower accuracy. Treat these as results for this rank, dataset, and training setup.

The runner accepts `SEEDS`, `VARIANTS`, `EPOCHS`, `BATCH_SIZE`, `STEPS_PER_EPOCH`,
`EVAL_BATCHES`, `OUTPUT_DIR`, and `REFERENCE_OPTIMIZER` environment variables.
The full default matrix uses `AdamW` as the paired reference. For a focused
subset that does not include `AdamW`, the first requested optimizer is used
automatically; set `REFERENCE_OPTIMIZER` to choose another included arm.
For example, a one-epoch, one-batch pipeline screen of the four core factorial conditions can be started with:

```bash
SEEDS="42" \
VARIANTS="naive_gqa ada_naive_gqa gated_gqa_silu_gated_ffn ada_gated_gqa_silu_gated_ffn" \
EPOCHS=1 BATCH_SIZE=8 STEPS_PER_EPOCH=1 EVAL_BATCHES=1 \
OUTPUT_DIR=mini_imagenet_gqa/output/factorial-pipeline-screen \
bash benchmarks/launchers/run_mini_imagenet_gqa_comparison.sh
```

To compare the APOLLO projection-refresh baseline with the delta hard-commit
variant, use the dedicated runner. It keeps the optimizer, rank, quantization
block size, and refresh interval matched, changing only
`--apollo-sf-delta-refresh` between `none` and `blend`:

```bash
SEEDS="42" EPOCHS=3 APOLLO_RANK=32 APOLLO_UPDATE_PROJ_GAP=200 \
OUTPUT_DIR=mini_imagenet_gqa/output/bucketed/apollo-sf-delta-refresh \
bash mini_imagenet_gqa/run_apollo_sf_delta_refresh_comparison.sh
```

The default `blend` window is four optimizer steps. At a refresh event it
moves `1/window` of the current delta from `y` into the live parameter on each
step, preserving `z = y + sf_delta` until the delta reaches zero. This is a
gradual Schedule-Free state merge, not a projection-refresh-equivalent
operation. Set `DELTA_REFRESH_POLICIES=none,commit_z` to reproduce the hard
commit comparison. The JSON report records merge counts and delta norms
separately from persistent state and peak VRAM.

Capped runs validate the pipeline only; they do not support full-split accuracy conclusions. Use a separate `OUTPUT_DIR` for screening so its results stay apart from full comparison runs. `DRY_RUN=1` checks the requested variants on CPU without loading the dataset. Checkpoints and per-epoch metrics are written under `<OUTPUT_DIR>/runs/<run-id>/`. On resume, the latest checkpoint retains the best validation accuracy seen before interruption, and the sibling `checkpoint_best.safetensors` is carried into the resumed run so final test evaluation still uses the best epoch across the full training history. Each run also atomically updates `progress.txt` after every optimizer step with status, epoch/batch/global-step counts, loss, accuracy, LR, step time, elapsed time, throughput, and ETA; this remains readable when stdout is redirected. After the matrix finishes, summarize its test metrics, throughput, and CUDA peak memory with:

```bash
PYTHONPATH=. python3 -m mini_imagenet_gqa.summarize_comparison
```

## Bounded Ada scale-function screen

The conditioning-LR intervention above changes optimization speed. This separate ablation keeps the ordinary `1e-3` model LR and changes only the mapping from the learned raw scale `s(m)` to the RMS-normalized attention/FFN input multiplier. The existing `ada_gated_gqa_silu_gated_ffn` remains the `1+s` reference.

| Variant | Multiplier | Range / initialization |
|---|---|---|
| `ada_gated_gqa_silu_gated_ffn` | `1+s` | Unbounded; starts at 1 |
| `ada_1plus_silu_gated_gqa_silu_gated_ffn` | `1+SiLU(s)` | Lower bounded at about 0.722, unbounded above; starts at 1 |
| `ada_1plus_softplus_halfnorm_gated_gqa_silu_gated_ffn` | `1+Softplus(s-2.5)` | Strictly above 1; starts at about 1.079 |
| `ada_silu_scale_gated_gqa_silu_gated_ffn` | `SiLU(s)` | Lower bounded at about -0.279 and unbounded above; starts at 0 |
| `ada_2sigmoid_gated_gqa_silu_gated_ffn` | `2*sigmoid(s)` | Strictly between 0 and 2; starts at 1 |
| `ada_silu1_norm_gated_gqa_silu_gated_ffn` | `SiLU(1+s)/SiLU(1)` | Unbounded above, can become negative; starts at 1 with nonzero derivative (~1.269) |
| `ada_softplus1_norm_gated_gqa_silu_gated_ffn` | `softplus(1+s)/softplus(1)` | Strictly positive and unbounded above; starts at 1 with nonzero derivative (~0.557) |

All variants use the same LR for every parameter (`--conditioning-lr-multiplier 1.0`) and zero initialization for Ada projection weights and biases. Thus direct `SiLU(s)` starts with multiplier 0. The attention Ada projection can learn through SiLU's derivative at zero, but the FFN Ada projection receives zero gradient because the bias-free SwiGLU branch has zero derivative at zero input; with this initialization the FFN branch remains disabled. Direct SiLU is excluded from the training CLI and retained only as a failure-mode diagnostic.

The common-init, deterministic, 10-epoch same-LR comparison is complete for seeds 42–44. Test top-1 was:

| Scale mapping | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| `1+s` | 38.84% | 3.04% | 34.80% | 25.56 ± 19.61% |
| `1+SiLU(s)` | 40.56% | 37.18% | 28.02% | 35.25 ± 6.49% |
| `2*sigmoid(s)` | 38.52% | 34.90% | 32.48% | 35.30 ± 3.04% |

The two bounded runs at each seed share the exact initial-state hash with that seed's linear reference. The three-seed means are effectively tied; `2*sigmoid(s)` has lower observed seed variation. Keep this as screening evidence, not a definitive winner.

Two follow-up mappings retain a unit multiplier at zero-initialized `s` while keeping a nonzero derivative, so the SwiGLU FFN starts active. `SiLU(1+s)/SiLU(1)` (`ada_silu1_norm_gated_gqa_silu_gated_ffn`) can produce negative multipliers. `softplus(1+s)/softplus(1)` (`ada_softplus1_norm_gated_gqa_silu_gated_ffn`) is strictly positive. Both were screened for 10 epochs at seeds 42–44 with full splits, BF16, common initialization, and deterministic execution. Test top-1 was:

| Scale mapping | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| `1+SiLU(s)` | 40.56% | 37.18% | 28.02% | 35.25 ± 6.49% |
| `SiLU(1+s)/SiLU(1)` | 38.44% | 33.52% | 35.56% | 35.84 ± 2.47% |
| `softplus(1+s)/softplus(1)` | 38.56% | 31.88% | 32.32% | 34.25 ± 3.74% |

All three mappings share the exact initial-state hash within each seed, so these are paired comparisons. The normalized SiLU mean is 0.59 points above `1+SiLU(s)` with lower observed seed variation; Softplus is about 1 point below it. The screen does not establish a winner, but confirms that both unit-initialized mappings train the FFN branch. The table retains the seed-level accuracy results; per-run logs and detailed resource measurements are omitted.

The normalized `1+SiLU(s)` form is already the existing `ada_1plus_silu...` mapping because `1+SiLU(0)=1`; that arm starts at unit scale. The Softplus half-normalized arm uses `1+Softplus(s-2.5)`, giving an initial scale of about 1.079, close to 1.1. It stays above 1, testing an amplification-only conditioning path while retaining the original channel magnitude. Compare it against `1+SiLU(s)` across AdamW and AdamW-SF at seeds 47–50 with the archived launcher `mini_imagenet_gqa/experiments/archive/run_ada_oneplus_softplus_halfnorm_optimizer_seeds47_50.sh`. The runner saves separate optimizer outputs and a paired summary. This remains a protocol comparison because AdamW uses cosine LR and AdamW-SF uses its schedule-free policy.

The four-condition follow-up at seeds 47–50 is complete. Test top-1 and paired difference from the no-Ada model are:

| Variant | Seed 47 | Seed 48 | Seed 49 | Seed 50 | Mean ± SD | Paired Δ vs no-Ada, mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| No Ada | 33.34% | 31.50% | 14.78% | 31.26% | 27.72 ± 8.68% | — |
| `1+s` | 35.50% | 33.66% | 1.00% | 22.68% | 23.21 ± 15.85% | -4.51 ± 7.99 pp |
| `1+SiLU(s)` | 36.10% | 33.78% | 13.32% | 35.20% | 29.60 ± 10.90% | +1.88 ± 2.33 pp |
| `softplus(1+s)/softplus(1)` | 35.40% | 34.64% | 35.36% | 32.88% | 34.57 ± 1.18% | +6.85 ± 9.18 pp |

Normalized Softplus beat the no-Ada baseline at all four seeds. Its seed-47/48/50 gains average +2.27 pp; the larger mean gain is driven partly by seed 49, where no-Ada scored 14.78% and Softplus scored 35.36%. The table above retains the per-seed results. Treat this as robustness evidence for the mapping, not proof that metadata alone caused the difference.

## APOLLO + AdamW-SF fallback with LR warmup

A separate hybrid comparison uses the same four Ada scale variants and seeds 47–50 as the AdamW follow-up. APOLLO rank is 32; 1D parameters use AdamW-SF, and matrix parameters use `auto-sf` state-size selection. The base LR is `1e-3`, with 5% linear warmup over optimizer steps then cosine decay to zero, under the same 10-epoch, BF16, batch-64, deterministic/common-init protocol. The scheduler advances before each optimizer update and its step state is checkpointed for resume. This is a practical hybrid-plus-schedule comparison against AdamW’s epoch cosine and AdamW-SF’s native schedule-free policy; it does not isolate optimizer update equations.

The archived launcher `mini_imagenet_gqa/experiments/archive/run_ada_scale_apollo_warmup_seeds47_50.sh` reproduces this protocol. Results are written separately to `output/bucketed/apollo-sf-warmup-ada-seeds47-50/`; do not mix them with AdamW or AdamW-SF reports. The runner skips completed arms and resumes interrupted arms from the latest checkpoint only when its optimizer/RNG sidecar is present; if an interruption happened before a resumable checkpoint, it restarts that arm from its seed-matched common initialization. The summarizer combines epoch records from resumed attempts and includes both fallback policy fields in its protocol consistency check. GPU execution must be started in an environment with CUDA access.

Once each optimizer matrix has completed, compare per-variant results and paired seed deltas with:

```bash
python3 -m mini_imagenet_gqa.summarize_optimizer_comparison \
  --input AdamW=mini_imagenet_gqa/output/bucketed/ada-scale-vs-no-ada-same-lr-10e-seeds47-50 \
  --input AdamW-SF=mini_imagenet_gqa/output/bucketed/adamw-sf-ada-seeds47-50 \
  --input APOLLO+AdamW-SF=mini_imagenet_gqa/output/bucketed/apollo-sf-warmup-ada-seeds47-50 \
  --output-dir mini_imagenet_gqa/output/bucketed/optimizer-comparison-seeds47-50
```

The report checks for all 16 expected variant/seed runs per optimizer, surfaces missing or unexpected runs, pairs seeds present in both protocols, and flags initial model-state hash mismatches. It records schedule policy because AdamW uses epoch cosine, AdamW-SF uses its schedule-free trajectory, and the APOLLO/AdamW-SF hybrid uses step warmup plus cosine. Backend accounting reports both parameter tensor counts and parameter element counts, so the APOLLO/fallback split is visible by model size as well as tensor count.

## AdamW-SF optimizer follow-up

The same four variants and seeds 47–50 were run with `AdamW-SF`, using the reference `torch` backend. It uses constant LR `1e-3` without an external cosine scheduler; its schedule-free averaging includes the learning-rate weighting internally. This compares practical AdamW+cosine against AdamW-SF's native schedule-free trajectory, so these observations do not isolate optimizer update rules. The trainer transitions Schedule-Free parameters to train weights before updates and to averaged evaluation weights before validation/checkpointing. A CPU synthetic smoke verified those transitions and an exactly matched post-resume update. Only the normalized-Softplus aggregate below is retained; the full per-variant/per-seed matrix and recovery details are omitted.

For the normalized-Softplus Ada arm, the retained aggregate gain is `+5.10 ± 1.44 pp`, with per-seed gains of +3.20, +5.08, +5.44, and +6.68 pp. This is the strongest comparison retained here. AdamW and AdamW-SF use different LR policies, so these observations do not isolate optimizer update equations. The APOLLO/AdamW-SF hybrid passes CPU update, train/eval transition, and checkpoint-restore checks, but its initial CUDA/BF16 runs stayed near 5–6% test top-1 across Ada and No-Ada arms. Detailed pairwise and resource results are omitted; the hybrid matrix remains paused for optimizer/update diagnostics.

The APOLLO fallback design and acceptance checks are in the [optimizer fallback design note](../docs/optimizers.md).

## APOLLO norm-growth limiter ablation

To reproduce the APOLLO per-parameter update-norm growth limiter ablation, use `bash mini_imagenet_gqa/experiments/archive/run_apollo_no_norm_growth_limiter_seeds47_48.sh`. It disables only the limiter while retaining rank 32, the 1D AdamW-SF / `auto-sf` matrix fallback, 5% step warmup, cosine decay, BF16, and common initialization. The two variants are no-Ada and normalized-SiLU Ada scale with additive Ada shift on both branches. Results go to `output/bucketed/apollo-no-norm-growth-limiter-seeds47-48/`.

The no-Ada runs pair directly with the completed limiter-on seed-47/48 APOLLO runs. There is no matching limiter-on run for the normalized-SiLU scale-plus-shift variant yet, so that arm measures its performance with the limiter disabled but does not by itself isolate the limiter effect. The limiter now defaults to disabled; pass `--apollo-norm-growth-limiter` to enable it.

The limiter-off run completed for seeds 47 and 48. Against the existing limiter-on no-Ada runs, test top-1 rose from `5.18 ± 0.31%` to `37.44 ± 0.37%`; paired gains were `+31.78` and `+32.74 pp` (`+32.26 ± 0.68 pp`). Best validation top-1 rose from `5.28 ± 0.51%` to `39.59 ± 0.56%`. Both pairs use identical initial-state hashes, data bucket counts, and training settings apart from the limiter flag and seed. The near-random APOLLO failure is therefore attributable to the default norm-growth limiter in this setup, rather than the low-rank path alone or the LR schedule. The precise limiter activation rate was not recorded, so the per-step clipping pattern remains unmeasured.

With the limiter off, the normalized-SiLU scale-plus-shift Ada variant scored `38.44 ± 0.57%` test top-1 and `40.54 ± 0.54%` best validation top-1 across the same two seeds. Its no-Ada comparison was `+1.00 pp` on test top-1, which is only a small two-seed signal; there is not yet a limiter-on run for that exact Ada variant.

## AdaRMS shift/bias follow-up

The implementation at the time of this follow-up applied scale-only AdaRMS. This screen added a comparison using `(1 + f(s(m))) * RMSNorm(x) + b(m)`, where `b(m)` is a separate zero-initialized linear projection from metadata and is broadcast over tokens. The zero-init keeps each new arm's initial operation equal to ordinary RMSNorm. The CLI variants are paired by mapping:

| Scale-only | Scale + shift |
|---|---|
| `ada_gated_gqa_silu_gated_ffn` | `ada_shift_gated_gqa_silu_gated_ffn` |
| `ada_1plus_silu_gated_gqa_silu_gated_ffn` | `ada_1plus_silu_shift_gated_gqa_silu_gated_ffn` |
| `ada_2sigmoid_gated_gqa_silu_gated_ffn` | `ada_2sigmoid_shift_gated_gqa_silu_gated_ffn` |
| `ada_softplus1_norm_gated_gqa_silu_gated_ffn` | `ada_softplus1_norm_shift_gated_gqa_silu_gated_ffn` |

The shift path is enabled before both attention QKV and FFN input projections. A seed-43, 10-epoch full-split screen used common initialization and deterministic execution. Scale-only and scale-plus-shift arms were run with the same training protocol; the new zero-initialized shift projection starts as an identity addition. Results are:

| Scale mapping | Scale-only test top-1 | Scale + shift test top-1 | Difference |
|---|---:|---:|---:|
| `1+s` | 3.04% | 24.60% | +21.56 pp |
| `1+SiLU(s)` | 37.18% | 33.98% | -3.20 pp |
| `2*sigmoid(s)` | 34.90% | 32.78% | -2.12 pp |

The scale-plus-shift runs reached 25.39%, 35.31%, and 34.19% best validation top-1, respectively. Each shift arm had the same initialization fingerprint as its other shift arms; the added shift parameters mean its full state hash differs from the scale-only state. This one-seed screen is not enough to establish a general benefit: the linear mapping's unusually low scale-only result makes its large gain especially uncertain. A matched repeat at seeds 42 and 44 would be needed to strengthen the conclusion; this public record contains no results for that proposed repeat. The direct-SiLU scale mapping remains excluded because its zero-initialized SwiGLU FFN is inactive; per-run measurements are omitted.

New AdaRMS variants now default to normalized Softplus `softplus(1+s)/softplus(1)`. `ada_gated_gqa_silu_gated_ffn` remains the historical `1+s` arm so existing runs and checkpoints keep their meaning. Other follow-up protocols without completion evidence are summarized in the status table above.


## Interpretation: Ada meta conditioning

## Aggregate evidence recorded here

The Mini-ImageNet classification runs inject bucketed resolution/aspect-ratio metadata through Ada scale modulation. The aggregate recorded here covers four shared conditions with nine matched seeds (42–50):

| Variant | Test top-1 by seed (%) | Mean ± SD (%) |
|---|---:|---:|
| No Ada: Gated GQA + Gated FFN | 37.20 / 24.20 / 31.94 / 34.52 / 30.80 / 33.34 / 31.50 / 14.78 / 31.26 | 29.95 ± 6.68 |
| Ada scale `1 + s` | 38.84 / 3.04 / 34.80 / 38.32 / 18.06 / 35.50 / 33.66 / 1.00 / 22.68 | 25.10 ± 14.84 |
| Ada scale `1 + SiLU(s)` | 40.56 / 37.18 / 28.02 / 38.48 / 38.20 / 36.10 / 33.78 / 13.32 / 35.20 | 33.43 ± 8.35 |
| Ada scale `2 × sigmoid(s)` | 38.52 / 34.90 / 32.48 / 36.52 / 20.94 | 32.67 ± 6.92 (n=5) |
| Ada scale `SiLU(1+s) / SiLU(1)` | 38.44 / 33.52 / 35.56 / 37.86 / 23.82 | 33.84 ± 5.93 (n=5) |
| Ada scale `softplus(1+s) / softplus(1)` | 38.56 / 31.88 / 32.32 / 36.88 / 29.24 / 35.40 / 34.64 / 35.36 / 32.88 | 34.13 ± 2.84 |

Seeds in each row are ordered 42 through 50; the two five-seed rows are ordered 42 through 46. These results compare the Gated GQA + Gated FFN architecture; avoid generalizing them to every architecture variant.

## Interpretation and limits

- Across nine seeds, normalized Softplus has the highest mean among the four fully tested conditions (34.13% vs. 29.95% no-Ada), the lowest observed SD (2.84 pp), and beats no-Ada at 8/9 seeds. Its paired gain averages +4.18 pp, with substantial spread (+6.63 pp SD) due in part to seed 49.
- `1+SiLU(s)` averages 33.43% and beats no-Ada at 7/9 seeds (+3.48 pp paired mean). It remains a plausible alternative, but it falls slightly below baseline at seed 49.
- Raw `1+s` is unstable: it averages 25.10% with 14.84 pp SD, including collapse-like outcomes at seeds 43 and 49. Do not use it as the default scale mapping.
- Seed 49 is not an Ada-only failure: the no-Ada baseline also falls to 14.78%, while normalized Softplus reaches 35.36%. This suggests an interaction between training seed and parameterization, but does not identify its cause.
- Ada behavior depends on the scale parameterization and optimization. A separate lower-learning-rate Ada run also improved over raw `1+s`, suggesting sensitivity to update scale; it is not a direct substitute for a matched mapping comparison.
- `2 × sigmoid(s)` and normalized SiLU have only five seeds and are not directly ranked against the nine-seed rows.
- Scale-and-shift variants were evaluated for seed 43 only. Their results are too limited to conclude whether adding shift/bias helps across seeds.
- This study only covers classification with resolution/aspect-ratio metadata. It does not establish the value of Ada conditioning for diffusion timestep, text, or reference-image information in generation tasks.

Treat these results as screening evidence. The repeated test scores are not independent confirmatory evaluations, and Ada adds 157,248 parameters over no-Ada. A parameter-matched control or shuffled-metadata ablation is needed to attribute gains specifically to resolution/aspect-ratio information.
