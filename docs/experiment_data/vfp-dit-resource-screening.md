# VFP-DiT resource screening

This record preserves aggregate results from short historical `vfp_dit_simple`
probes whose raw run files have been removed. These measurements describe the
tested configurations only. They do not establish current throughput, image
quality, or convergence for the current `vfp_dit` implementation.

## 128px probes

On 2026-09-24, APOLLO rank-32 probes used condition tap 10, adapter depth 1,
and four validation examples split evenly between T2I and TI2I:

| Width / depth | Training examples | Throughput | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| 192 / 2 | 8 | 2.36 samples/s | 1.92 GiB |
| 384 / 4 | 8 | 1.55 samples/s | 1.96 GiB |

These sample counts support pipeline/resource checks only. A width-1024,
depth-4, eight-sample probe used 2.44 GiB allocated / 2.54 GiB reserved.

The separate 10-epoch width-384 / depth-4 screen is indexed in the
[experiment aggregate index](README.md#追加screenの集計値).

## 512px probes before and after latent compression

Before target latent compression, depth 4 completed eight samples at
0.62 samples/s and 7.06 / 9.16 GiB peak allocated / reserved. Depth 8 completed
at 0.12 samples/s and 7.64 / 9.41 GiB. The depth-8 probe left 188 MiB free. A
longer run was stopped before its first epoch completed with about 255 MiB
free. These measurements motivated target patchification and reference-latent
fusion into the Qwen vision grid.

After compression, the 2026-09-24 probes used 512px, width 1024, batch 1,
BF16, APOLLO rank 32, adapter depth 1, factor-4 target patchification, VFP
reference fusion, and gradient checkpointing:

| Depth | Training examples | Throughput | Peak allocated / reserved VRAM |
| ---: | ---: | ---: | ---: |
| 4 | 8 | 1.76 samples/s | 2.82 / 3.01 GiB |
| 24 | 8 | 0.85 samples/s | 4.61 / 4.85 GiB |

Each arm validated eight examples (five T2I and three TI2I). The earlier
pre-compression depth-4 probe used 8 query / 2 KV heads, while the later probes
used 16 / 4, so the resource comparison is directional. These single-seed
probes show fit for the tested configuration; their losses do not support
quality or convergence claims.

## Activation checkpointing comparison

A fixed-384px, two-image/four-timestep probe measured 53.62 / 55.46 GiB peak
allocated / reserved without checkpointing and 11.27 / 12.44 GiB with
checkpointing; the checkpointed arm reproduced loss 4.247856. In a reverse-order
repeat, the checkpointed arm completed in 53.67 seconds, while the
non-checkpointed arm ran out of memory in its second batch after 191.57 seconds
for its first. This supports checkpointing for memory fit at those settings;
the throughput comparison is inconclusive because one arm did not complete.

Raw metrics, logs, checkpoints, sample images, and per-run reports are not part
of the public archive. See the [current VFP-DiT guide](../../vfp_dit/README.md)
for current commands and runtime limits.
