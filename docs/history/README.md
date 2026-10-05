# Historical research documents

[日本語](README.md) | [English](README.en.md)

このdirectoryには、現行仕様や優先タスクとして参照すべきではない研究履歴snapshotをまとめています。記載中の「current」「next」などは各文書の記録日時点を指します。公開前整理で、提案中心の2資料は未実施案を圧縮し、実測結果・集計値・解釈条件を残しました。詳しい背景より結果を調べる場合は各summaryから読み始めてください。

## 月別の履歴概要

一覧では日付を月単位に丸めています。実験の再現条件に必要な実施日・計測条件は各記録に残しています。独立したADR文書はなく、設計判断は関連する履歴資料に含まれています。

| 月 | 主な記録・判断 | 詳細と現行資料 |
| --- | --- | --- |
| 2026-09 | optimizer測定・設計、学習UX/性能/実装監査のsnapshot。VFCBのQwen tap sweepをアーカイブする判断も記録。結果は当時の条件に限定され、現行仕様や推奨を示しません。 | [Schedule-Free研究記録索引](../schedule-free-methods-summary.md)、[optimizer performance design](optimizer-performance-design-2026-09-14.md)、[UX review](ux-review-2026-09-12.md)、[性能レビュー索引](performance-review-2026-09-12.md)、[実装監査](repository-audit-2026-09-12.md)、[VFCB sweep review](vfcb-layer-sweep-review.md) |
| 2026-10 | CPU testとstatic checkの限定的なvalidation follow-up。全面再監査やGPU検証ではありません。 | [validation follow-up](repository-audit-2026-10-05-validation.md) |
| 月の記録なし | VFP-DiTのversion別設計判断、tap screening、実装・検証履歴、およびMini-ImageNet GQAの選択判断。 | [VFP-DiT research summary](vfp-dit-research-summary.md) / [version別詳細](VFP-DiT_Research_History_v0.1.47.md)、[Mini-ImageNet GQA results](../mini-imagenet-gqa-results.md)、[現行architecture](../architecture/vfp-dit.md) |

## 主題別の入口

月別一覧に加え、次の索引から詳細記録へ進めます。各記録は記載時点の条件と制約を保持する履歴snapshotです。

| 主題 | 要約・索引 | 詳細記録 |
| --- | --- | --- |
| VFP-DiT設計・tap screening | [VFP-DiT research summary](vfp-dit-research-summary.md) | [revision別の設計・検証履歴](VFP-DiT_Research_History_v0.1.47.md)、[VFCB layer sweep](vfcb-layer-sweep-review.md) |
| Schedule-Free / low-rank optimizer | [Schedule-Free系optimizer索引](../schedule-free-methods-summary.md)、[LRTDO要約（日本語）](lrtdo-research-summary.md) / [English](lrtdo-research-summary.en.md) | [LRTDO集計結果](lrtdo-research-results.md)、[初期implementation/probe記録](low-rank-schedule-free-records-2026-09-13.md) |
| Optimizer・モデル性能レビュー | [性能レビュー索引](performance-review-2026-09-12.md) | [optimizer review](optimizer-review-2026-09-12.md)、[optimizer performance design](optimizer-performance-design-2026-09-14.md)、[DiT/LLM adapter review](dit-adapter-performance-review-2026-09-12.md)、[Text-LM architecture benchmark](text-lm-architecture-benchmark-2026-09-12.md) |
| UXレビュー | [UX review](ux-review-2026-09-12.md) | 同じsnapshot内に所見と記録時点の対応状況を掲載 |
| Repository監査 | [repository audit snapshot](repository-audit-2026-09-12.md) | [CPU/static validation follow-up](repository-audit-2026-10-05-validation.md) |

現行の実装仕様は各package READMEと、[`docs/README.md`](../README.md)から参照できるarchitecture・optimizer文書を確認してください。

この履歴文書に記された`output/`以下のパスは、当時のローカル生成物を示します。実験artifactやcheckpointは公開ツリーに含みません。
