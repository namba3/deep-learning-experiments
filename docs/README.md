# Documentation

[日本語](README.md) | [English](README.en.md)

このページは、利用者向けの手順、実装仕様、研究記録、メンテナ向け作業記録を案内します。リポジトリ全体の概要と最小実行例は[ルートREADME](../README.md)を参照してください。

## 利用・検証

- [ルートREADME](../README.md): リポジトリの目的、主要ディレクトリ、セットアップ、最小実行例
- [VFP-DiT README](../vfp_dit/README.md): 学習、sampling、checkpoint仕様
- [開発環境と事前検証](guides/development-and-validation.md): 環境構築、preflight、テスト・静的チェック
- [VFP-DiT training operations（English）](guides/vfp-dit-operations.en.md): encoder配置、memory計測、比較launcher、診断、resume運用
- [VFP-DiT shared runtime](../vfp_dit_runtime/README.md): VFP-DiTが共有するQwen、データ、学習runtime
- [Shared training runtime](../runtime/README.md): checkpoint、sampler、progress、preflight等の共通処理
- [Benchmark and experiment launchers](../benchmarks/README.md): reusable benchmarkと研究用shell launcherの索引
- [検証ガイド](../verify/README.md): CPU確認と、外部モデル・CUDAを必要とするruntime検証
- [Mini-ImageNet GQA実験](../mini_imagenet_gqa/README.md): 実験設定と結果の再現手順
- [Dataset and model provenance](data-model-provenance.md): 使用する外部データセット・モデルの出典、条件、未確定の権利事項
- [実験結果アーカイブ索引（日本語）](experiment_data/README.md) | [English](experiment_data/README.en.md): 過去の集計レポートと削除済みraw run artifactの範囲
- Adapter experiment records: [CIFAR-10](adapter-experiments/cifar10.md), [ImageAE](adapter-experiments/image-ae.md), [Text-LM](adapter-experiments/text-lm.md), [TinyImageNet-200](adapter-experiments/tiny-imagenet.md) — dataset別の過去のadapter比較と集計結果

## 実装仕様・監査

- [Image-latent DiT architecture（日本語）](architecture/image-latent-dit.md) | [English](architecture/image-latent-dit.en.md): shape、mask、学習、sampling、checkpointの契約
- [VFP-DiT architecture](architecture/vfp-dit.md): Qwen/VAE conditioning、latent fusion、DiT、Ada、checkpoint compatibility contracts
- [VFP-DiT migration note](vfp-dit-migration.md): 現行entrypoint、旧run出力の配置、checkpoint stage互換性
- [LR schedulers](lr-schedulers.md): train script共通のschedulerとwarmup仕様
- [Optimizers](optimizers.md): optimizer registry、variant、fallback、state設計

## 研究・比較記録

以下は実験時点の仮説・設定・結果を記録した資料です。現行実装の仕様を確認するときは、上の実装仕様と各パッケージのREADMEを参照してください。

### VFP-DiT

- [VFP-DiT documentation index](VFP-DiT_Research_Note.md): 現行README、移行情報、研究履歴への案内
- 廃止済み実装の集計値と過去の短期resource probeは[実験結果アーカイブ](experiment_data/README.md)を参照

### Optimizer・低rank手法

- [APOLLO research and experiment records](apollo-experiment-records.md): 概要、仮説、集計結果、historical protocolへの入口
- [LRTDO研究要約（日本語）](history/lrtdo-research-summary.md) | [English](history/lrtdo-research-summary.en.md): low-rank trajectory実験の要約と集計詳細への案内
- [Schedule-Free手法と関連記録](schedule-free-methods-summary.md): Mini-ImageNet量子化結果と現行optimizer仕様への案内
- [Low-rank Schedule-Free design](low-rank-schedule-free-design.md): 数式、用語、state、train/eval・checkpoint契約の2026-09-13設計snapshot
- [Low-rank adapters](low-rank-adapters.md): 数式・実装契約・検証観点
- [Low-rank adapter experiment records](low-rank-adapters-results.md): 過去の実験条件と集計結果
- [RGLU-LoRA](rglu-lora.md)

### Sampling solver・データセット比較

- [Flow-sampling solver design index](sampling-solvers.md): solverごとの方式・実装状況・検証範囲
- [Mini-ImageNet GQA comparison design](mini-imagenet-gqa-comparison.md): 比較設計とCPU検証記録
- [Mini-ImageNet GQA experiment results](mini-imagenet-gqa-results.md): 完了済みscreenとoptimizer follow-upの集計記録
- [Mini-ImageNet GQA bucket analysis](experiment_data/mini-imagenet-gqa-bucket-analysis.md): dataset解像度の集計とaspect bucket案
- [Text LM pretraining comparison](text-lm-pretraining-comparison.md)

### Historical snapshots

- [履歴索引（日本語）](history/README.md) | [English](history/README.en.md): 履歴snapshotの一覧、対象時点、読み方

## 文書の置き場所と読み方

| 場所 | 用途 |
| --- | --- |
| ルート `README.md` | リポジトリの目的、構成、セットアップ、最小実行例 |
| `*/README.md` | パッケージの責務、公開entrypoint、CLI、関連資料 |
| `docs/architecture/` | 複数パッケージに関わる実装契約 |
| `docs/guides/` | 特定機能の利用・運用手順 |
| `docs/` | 監査、性能記録、研究・比較資料、公開に必要な保守情報 |
| `docs/adapter-experiments/` | dataset別adapter実験記録 |
| `docs/experiment_data/` | 過去実験の集計レポート。step単位metricsやconfig等のraw run artifactは公開ツリーに保持しない |
| `verify/` | pytestから分けた外部モデル・CUDAのruntime検証 |
| `AGENTS.md` | リポジトリ内での開発・検証ルール |

コードとCLI `--help` が現行実装の正です。研究・比較資料は記載時点の仮説や結果として読み、日付や適用範囲を確認してください。CPU/static確認だけではCUDA/Triton/外部モデルの挙動を検証したことにはなりません。

実験記録中の`output/`はローカル生成物の保存先または過去の保存先です。raw run artifact、checkpoint、生成画像は公開ツリーに含めず、再利用する集計値と条件を各レポートに残しています。
