# LRTDO research summary

[English](lrtdo-research-summary.en.md) | [日本語](lrtdo-research-summary.md)

This archive summarizes completed diagnostics and comparisons for Low-Rank Trajectory Drift Optimization (LRTDO). Unrun phase proposals and future benchmark plans were removed. The detailed records retain aggregate measurements and their conditions; raw run JSON, logs, and checkpoints are not part of the public tree.

LRTDO studies low-rank representations of Schedule-Free trajectory drift and optimizer state. APOLLO gradient projection `R_update` and Schedule-Free delta projection `R_delta` represent different state and serve different purposes. For current optimizer behavior and defaults, see [the optimizer contract](../optimizers.md) and the implementation.

This archive mainly covers TinyStories experiments involving Schedule-Free low-rank methods, confidence variants, and APOLLO-Conf. Aggregate APOLLO optimizer and projection-refresh results on ImageAE/CIFAR-10 are in the [APOLLO experiment archive](../apollo-experiment-results.md). The tasks and protocols differ, so do not compare their measurements directly.

## How to read the results

The results depend on task, seed, token budget, step count, dtype, and implementation state. An initial rank diagnostic used a mismatched parameter dtype; its optimizer state byte measurements are invalid and marked accordingly. Several comparisons are short probes or use few seeds, so they do not establish general optimizer superiority. The APOLLO candidate results at the end of the detailed archive use rank 4 and do not establish an AdamW-SF quality baseline or generalize to other tasks.

## Recorded results

| Topic | Detailed record |
| --- | --- |
| Initial rank diagnostic, including the dtype limitation | [Initial diagnostic](lrtdo-research-results.md#diagnostic-initial) |
| Effect of EMA on delta rank | [EMA diagnostic rerun](lrtdo-research-results.md#diagnostic-ema-rerun) |
| Rank and refresh-interval comparison | [Rank/refresh sweep](lrtdo-research-results.md#rank-refresh-sweep) |
| Refresh transport error and overlap | [Refresh transport diagnostics](lrtdo-research-results.md#refresh-transport) |
| Low-rank projected-gradient EMA baseline | [Projected-gradient EMA](lrtdo-research-results.md#projected-gradient-ema) |
| Initial innovation-variance confidence results | [Confidence diagnostic](lrtdo-research-results.md#innovation-variance-confidence) |
| Confidence-enabled LRSF prototype | [LRSF prototype](lrtdo-research-results.md#confidence-lrsf-prototype) |
| APOLLO and confidence-variant comparison | [APOLLO comparison](lrtdo-research-results.md#apollo-confidence-comparison) |
| Confidence beta/alpha sensitivity | [APOLLO-Conf sensitivity](lrtdo-research-results.md#apollo-conf-sensitivity) |
| APOLLO rank/scale exploration and 100-step comparison | [Rank-4 LR/scale sweep](lrtdo-research-results.md#apollo-rank-scale-sweep) |
| Matched 300-step comparison with AdamW-SF | [Matched comparison](lrtdo-research-results.md#apollo-adamw-sf-300-step), [rank-4 candidate](lrtdo-research-results.md#apollo-rank4-300-step) |
| 300-step trajectory diagnostic | [Trajectory diagnostic](lrtdo-research-results.md#apollo-trajectory-diagnostic) |

Only reusable aggregate values and their interpretation conditions are retained. Local output files and checkpoints are excluded.
