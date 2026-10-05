# VFCB Qwen layer sweep review

> **Archived workstream.** The old VFCB implementation and its run outputs were retired during VFP-DiT promotion. This report records the original conclusion; it is not a current training plan.

Review month: 2026-09

## Scope and completion

The sweep compared Qwen condition taps 6, 10, 12, 14, 18, and final with seeds 0, 1, and 2. It was paused for result review at 14 of 18 completed cells. Four cells are incomplete: tap 12 / seed 2 has metrics through epoch 3; tap 14, 18, and final / seed 2 have no completed run. The report contains retry records for the interrupted tap-12 cell, while completed-cell aggregates still use the 14 completed cells.

The resumable checkpoint is preserved. A resume attempt exposed a config comparison bug: JSON stores scheduler milestones as a list while CLI parsing provides a tuple. vfp_dit/training.py now normalizes nested lists and tuples during checkpoint config validation. The sweep remains paused.

## Loss trajectory

Mean validation loss across completed seeds, from epoch 1 to epoch 10:

| Qwen tap | Completed seeds | Epoch 1 mean | Epoch 10 mean | Change |
|---|---:|---:|---:|---:|
| 6 | 3 | 1.4724 | 1.4355 | -0.0369 |
| 10 | 3 | 1.4744 | 1.4401 | -0.0343 |
| 12 | 2 | 1.4748 | 1.4445 | -0.0303 |
| 14 | 2 | 1.4776 | 1.4415 | -0.0361 |
| 18 | 2 | 1.4754 | 1.4314 | -0.0440 |
| final | 2 | 1.4742 | 1.4314 | -0.0428 |

Training and validation losses decrease, but validation improvement is modest and reaches only about 1.43-1.44 after 10 epochs. Tap choice does not produce a large or consistent loss difference. The incomplete cells limit seed-level certainty for taps 12, 14, 18, and final.

## Feature diagnostics

Across completed taps, validation feature linear CKA is about 0.070-0.082 and prediction RMS is about 0.106-0.123 of teacher RMS. Prediction effective rank is about 4.8-8.0 while teacher rank is about 430. The predicted representation captures very little of the teacher feature variation, despite the small scalar-loss improvements.

These diagnostics support a collapse or under-representation interpretation for this VFCB setup. They do not prove that every possible VFCB architecture or objective is untrainable; this experiment tests only the recorded model, losses, and 10-epoch budget.

No zero-prediction baseline or downstream Stage 2 image-generation quality comparison was recorded. The decision therefore rejects this VFCB setup for further GPU investment; it does not claim that the broader VFCB idea is impossible.

## Decision

Stop the current VFCB-based VFP-DiT line and mark it archived. Preserve its source, checkpoints, partial sweep data, and reports for reproducibility. Do not spend more GPU time completing the remaining four cells by default: the current representation diagnostics are already far below the teacher and the tap sweep shows no promising loss separation.

This was the decision at the time of the review. The current image-generation implementation has since been promoted to `vfp_dit/`; see [`../../vfp_dit/README.md`](../../vfp_dit/README.md) and [`../vfp-dit-migration.md`](../vfp-dit-migration.md) for the migration record.

## Artifacts

- The original raw sweep report and epoch CSV are no longer present in the current output tree. The aggregate numbers and interpretation above are the retained evidence in this checkout.
- The old implementation and checkpoints were removed during promotion; see [`../vfp-dit-migration.md`](../vfp-dit-migration.md) for the current checkpoint compatibility contract.
