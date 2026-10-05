# Historical research documents

[English](README.en.md) | [日本語](README.md)

This directory contains historical snapshots. Terms such as “current,” “next,” and “candidate” refer to the date of each record; they are not current specifications or work instructions. Proposal-heavy material was condensed to retain completed measurements, aggregate values, and their interpretation limits. Start with a summary when you need results rather than background.

## Monthly overview

Dates in this overview are grouped by month. Experiment dates and conditions needed for reproduction remain in the individual records. The repository has no standalone ADR files; recorded design decisions are included in the related history documents.

| Month | Main records and decisions | Details and current references |
| --- | --- | --- |
| 2026-09 | Optimizer measurements and design, UX/performance reviews, and implementation-audit snapshots. The VFCB Qwen tap sweep was also archived. These results are limited to their recorded conditions and do not state current specifications or recommendations. | [Schedule-Free research index](../schedule-free-methods-summary.md), [optimizer performance design](optimizer-performance-design-2026-09-14.md), [UX review](ux-review-2026-09-12.md), [performance review index](performance-review-2026-09-12.md), [repository audit](repository-audit-2026-09-12.md), [VFCB sweep review](vfcb-layer-sweep-review.md) |
| 2026-10 | A focused CPU-test and static-check follow-up. It is not a full re-audit or GPU validation. | [Validation follow-up](repository-audit-2026-10-05-validation.md) |
| Undated records | VFP-DiT revision history, tap screening and implementation checks, plus Mini-ImageNet GQA decisions. | [VFP-DiT research summary](vfp-dit-research-summary.md), [revision history](VFP-DiT_Research_History_v0.1.47.md), [Mini-ImageNet GQA results](../mini-imagenet-gqa-results.md), [current VFP-DiT architecture](../architecture/vfp-dit.md) |

## Browse by topic

Use these entry points to find related details. Each record retains the conditions and limitations from its stated date.

| Topic | Summary or index | Detailed records |
| --- | --- | --- |
| VFP-DiT design and tap screening | [VFP-DiT research summary](vfp-dit-research-summary.md) | [Revision design and validation history](VFP-DiT_Research_History_v0.1.47.md), [VFCB layer sweep](vfcb-layer-sweep-review.md) |
| Schedule-Free and low-rank optimizers | [Schedule-Free optimizer index](../schedule-free-methods-summary.md), [LRTDO summary (English)](lrtdo-research-summary.en.md) / [日本語](lrtdo-research-summary.md) | [LRTDO aggregate results](lrtdo-research-results.md), [Initial implementation and probe records](low-rank-schedule-free-records-2026-09-13.md) |
| Optimizer and model performance reviews | [Performance review index](performance-review-2026-09-12.md) | [Optimizer review](optimizer-review-2026-09-12.md), [optimizer performance design](optimizer-performance-design-2026-09-14.md), [DiT/LLM adapter review](dit-adapter-performance-review-2026-09-12.md), [Text-LM architecture benchmark](text-lm-architecture-benchmark-2026-09-12.md) |
| UX review | [UX review](ux-review-2026-09-12.md) | The snapshot includes findings and their status at the time of review. |
| Repository audit | [Repository audit snapshot](repository-audit-2026-09-12.md) | [CPU/static validation follow-up](repository-audit-2026-10-05-validation.md) |

For current implementation behavior, use the package READMEs and the architecture and optimizer documents linked from the [documentation index](../README.en.md).

Paths under `output/` in these records refer to local artifacts or former output locations. Experiment artifacts and checkpoints are not included in the public tree.
