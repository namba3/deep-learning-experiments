# VFP-DiT Simple VLM adapter ablation

Date: 2026-09-24<br>
Screen group: `ref-fusion-fix`
Hardware: RTX 3080 Ti, BF16

> **Historical result:** This one-epoch ablation used the retired `vfp_dit_simple` implementation. The results apply only to this checkpoint, data order, and screen protocol; they do not establish a current adapter recommendation for [`vfp_dit`](../../vfp_dit/).

## Setup

All three arms started from the same seed (`42`) with the same width-1024 / depth-24 target DiT initialization and training data order. Each arm trained for one epoch over 128 examples at 512px, batch size 1, four stratified timesteps per image, APOLLO rank 32, LR `1e-4`, and condition dropout `0.1`. Validation used the same 32 rows (14 T2I, 18 TI2I). The train sampler selected 70 T2I and 58 TI2I rows in each arm. Each image used the same prompt and sampling seed for the saved sample.

The first attempted screen group predates the reference-fusion fix and is not valid for TI2I comparison. Its checkpoint directories were moved outside the repository to reclaim workspace disk space. Raw run files and generated images are not retained in the public archive; this document keeps only the summarized comparison below.

## Results

| Adapter | Trainable adapter params | Train loss | Validation loss | T2I validation | TI2I validation | Train samples/s | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 23.10M | 1.8390 | 1.8429 | 1.4659 | 2.1361 | 0.942 | 4.91 / 5.14 GiB |
| FFN-only | 14.71M | 1.8440 | 1.8503 | 1.4651 | 2.1500 | 1.028 | 4.88 / 5.13 GiB |
| Linear-only | 2.12M | 1.8442 | 1.8518 | 1.4658 | 2.1521 | 0.984 | 4.83 / 5.08 GiB |

Validation loss spans only `0.0090` across these arms. T2I losses are nearly identical. Transformer is slightly lower on TI2I in this single epoch, but the difference is small and does not establish a reliable ranking. The throughput spread is also too small and too short to rank. FFN-only cuts trainable adapter parameters by about 36%; Linear-only cuts them by about 91%. Peak VRAM changes by less than 0.1 GiB.

## Gradient diagnostics

Each arm logged 128 `adapter_gradient` events. Across all arms, input-projection gradients were nonzero on all steps. For the 53 TI2I steps not condition-dropped, mean reference-branch gradient norms were 5.55 (Transformer), 5.56 (FFN-only), and 5.57 (Linear-only). This confirms the fixed fusion path participates in training. Transformer attention gradients were nonzero on all steps; FFN-only and Linear-only attention gradients were zero as configured. FFN gradients were nonzero for Transformer and FFN-only and zero for Linear-only as configured.

## Reference-fusion correction

While inspecting the first gradient results, a tensor-reference bug was found in `VLMAdapter`: the fused Qwen vision-token clone was updated, but the later adapter blocks still received the pre-clone tensor. The adapter now assigns the fused tensor to `tokens`. The corrected screen confirms nonzero reference-branch gradients on undropped TI2I batches. Results from the earlier, pre-fix screen must not be used to judge TI2I learning.

## Interpretation

The observation that samples remain noisy is not explained by vanishing gradients in these adapter branches. This one-epoch screen also does not show a meaningful loss advantage for removing the transformer. Keep `transformer` as the default for now; regenerate matched samples before a qualitative comparison, then run a longer matched comparison only if one variant appears consistently better. These runs do not establish convergence, image quality, or seed stability.

## Samples and retained evidence

The sample PNGs, JSON sidecars, step-level `metrics.jsonl`, progress files, and per-run configs were removed from the public archive. The aggregate values and gradient statistics above remain; the original images are no longer available for visual reinspection.
