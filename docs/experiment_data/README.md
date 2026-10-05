# Experiment aggregates and analyses

[日本語](README.md) | [English](README.en.md)

このdirectoryには、生データやrun artifactではなく、再利用する集計値とdataset分析を保存します。現行実装の仕様と推奨設定は[`vfp_dit/README.md`](../../vfp_dit/README.md)を参照してください。

## Model measurements

- [Qwen Image VAE latent statistics](image-gen-qwen-latent-statistics.md): historical aggregate over 10,000 Flickr30k images; descriptive only, not a runtime normalization contract

## Dataset analysis

- [Mini-ImageNet GQA resolution and aspect buckets](mini-imagenet-gqa-bucket-analysis.md): cached dataset dimensions, aspect-ratio distribution, and proposed compute-matched buckets

## Archived experiment summaries

以下のVFP-DiT結果は、廃止済み`vfp_dit_simple`実装のscreenです。現行`vfp_dit`の品質や推奨設定を示しません。

- [VFP-DiT resource screening](vfp-dit-resource-screening.md): historical short-run memory and throughput aggregates; not a current performance claim

## Dated implementation probes

- [VFP-DiT implementation probes (2026-09-24)](vfp-dit-implementation-probes-2026-09-24.md): short pipeline and resource checks; not quality, convergence, or general performance evidence

## 比較・集計レポート

- [Adapter ablation](vfp-simple-adapter-ablation.md): 3 adapter variantの1 epoch比較とgradient集計
- [Learning-rate screen](vfp-dit-simple-lr-screen.md): constant LR、cosine follow-up、sampler比較の集計
- [Output-refinement comparison](vfp-dit-simple-output-refinement-comparison-20260929.md): 5構成の短期screen比較

## 追加screenの集計値

個別runのprogressやinterval observationsは保持せず、最終epochの集計値だけを記録します。いずれも旧`vfp_dit_simple`のseed-42 screenであり、現行`vfp_dit`の品質や推奨設定を示しません。

| Screen | 主な条件 | 最終train / validation loss | T2I / TI2I validation loss | Train images/s | Peak allocated / reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 128px pipeline screen | 128px, width 384, depth 4, 10 epochs | 1.1406 / 1.1424 | 1.1769 / 1.1156 | 4.153 | 1.99 / 2.06 |
| 512px quality screen | 512px, width 1024, depth 24, LR `1e-4`, 10 epochs | 1.1995 / 1.1952 | 1.1580 / 1.2324 | 0.962 | 4.92 / 5.68 |
| 512px quality screen | 512px, width 1024, depth 24, LR `1e-3`, 10 epochs | 0.2884 / 0.2851 | 0.2826 / 0.2876 | 1.424 | 5.80 / 6.55 |

最後のscreenはprogress snapshotが`checkpointing`を示していたため、最終保存の完了状態は確認できません。表の数値は記録済みepoch metricsの値です。

公開アーカイブには再利用する集計値と解釈上の制約を残し、生データ、個別run artifact、cleanup作業記録は含めません。
