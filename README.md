# AI・深層学習の学習用実験リポジトリ

[日本語](README.md) | [English](README.en.md)

AI・深層学習への理解を深めるため、PyTorchでモデルアーキテクチャや学習手法を実装・検証する実験用リポジトリです。画像モデル、autoencoder、Transformer、optimizerなどを題材にしています。

このREADMEはリポジトリ全体の入口です。個別実験のCLIやモデル内部の仕様は、各ディレクトリのREADMEと[`docs/`](docs/)を参照してください。

## 実験領域

### 基本的なモデルと学習手法

- [`mnist/`](mnist/README.md): MNIST分類と基本的な学習ループ
- [`cifar10/`](cifar10/README.md): CIFAR-10分類とadapter実験
- [`text_lm/`](text_lm/README.md): テキストTransformer
- [`optimizers/`](optimizers/README.md): optimizerと学習率制御

### モデルアーキテクチャと画像モデル

- [`mini_imagenet_gqa/`](mini_imagenet_gqa/README.md): GQA/FFN構成の比較
- [`image_ae/`](image_ae/README.md): 画像autoencoder
- [`image_gen/`](image_gen/README.md): Qwen VAE・Qwen3.5を使うimage-latent DiT/MMDiT
- [`vfp_dit/`](vfp_dit/README.md): 条件付きVFP-DiT

### 共通の実装・検証基盤

- [`core/`](core/README.md)、[`runtime/`](runtime/README.md)、[`vfp_dit_runtime/`](vfp_dit_runtime/README.md): 共有model・学習runtime
- [`flow_sampling/`](flow_sampling/README.md): flow-matching scheduleとsampling solver
- [`benchmarks/`](benchmarks/README.md)、[`tests/`](tests/README.md)、[`verify/`](verify/README.md): 性能測定と各種検証
- [`docs/`](docs/README.md): 設計、監査、性能、研究記録

## 実行例

### MNISTの軽量なCPU事前確認

datasetを読み込まず、引数やdeviceの設定を確認します。

```bash
PYTHONPATH=. python3 -m mnist.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
```

### アーキテクチャ比較: Mini-ImageNet GQA

QK RMSNormを共通にして、naive/Gated GQAとplain/Gated FFNを比較します。augmentation・variant定義・seed実行手順は[`mini_imagenet_gqa/README.md`](mini_imagenet_gqa/README.md)を参照してください。

```bash
PYTHONPATH=. python3 -m mini_imagenet_gqa.train --dry-run --device cpu
```

### 画像autoencoder

以下のFlickr30kを使う例は、画像のダウンロードを伴います。画像の著作権は各権利者にあり、利用・再配布にはFlickrの条件が適用されます。実行前に[データセットとモデルの出典・利用条件](docs/data-model-provenance.md#datasets)を確認してください。このリポジトリのコードライセンスはデータセットに適用されません。

```bash
python3 -m image_ae.train --dataset flickr30k --batch-size 8
```

### Image-latent DiT

`--vae-model`は必須です。データセットを使う例:

次の例はHugging Face上のFlickr30k mirrorから画像を取得します。mirrorには画像の再配布許諾が明示されていません。Flickr30kの提供元は画像を非商用研究・教育向けとし、Flickrの条件への従属を案内しています。データや生成物を共有する前に、[出典と利用条件](docs/data-model-provenance.md#datasets)および各画像の条件を確認してください。

```bash
python3 image_gen/train.py \
  --dataset-name lmms-lab-encoder/flickr30k \
  --vae-model Qwen/Qwen-Image
```

ローカルJSONL/CSVを使う場合:

```bash
python3 image_gen/train.py \
  --records data.jsonl \
  --vae-model Qwen/Qwen-Image
```

### VFP-DiT

現行のVFCBを使わない構成では、Qwen3.5のhidden stateを条件付けに使い、必要に応じてQwen-Imageのreference latentを加えます。DiTは空間情報を圧縮した前段と、全解像度で出力を精緻化する段で構成されます。構成と学習の詳細は[`vfp_dit/README.md`](vfp_dit/README.md)を参照してください。

```bash
python3 -m vfp_dit.train --help
python3 -m vfp_dit.train --dry-run --device cpu
```

### 生成サンプル

学習済みcheckpointから生成する場合:

```bash
python3 image_gen/generate_samples.py \
  --checkpoint image_gen/output/<run>/checkpoint_latest.safetensors \
  --vae-model Qwen/Qwen-Image \
  --prompt "a photograph of a child playing with a dog outdoors"
```

各スクリプトの完全なオプションは`--help`を正とします。

## 開発と検証

環境構築、事前検証、テストと静的チェックの手順は[開発環境と事前検証ガイド](docs/guides/development-and-validation.md)を参照してください。

## ドキュメント

- [開発環境と事前検証ガイド](docs/guides/development-and-validation.md): 仮想環境、依存関係、preflight、テスト・静的チェック
- [`docs/README.md`](docs/README.md): 文書の分類と全体索引
- [`docs/data-model-provenance.md`](docs/data-model-provenance.md): 外部データセット・モデルの出典と利用条件

パッケージ別の起動方法と責務:

| 分類 | ガイド |
| --- | --- |
| 実験 | [`image_gen`](image_gen/README.md)、[`vfp_dit`](vfp_dit/README.md)、[`image_ae`](image_ae/README.md)、[`cifar10`](cifar10/README.md)、[`mini_imagenet_gqa`](mini_imagenet_gqa/README.md)、[`mnist`](mnist/README.md)、[`text_lm`](text_lm/README.md) |
| 共通runtime・model・sampling | [`core`](core/README.md)、[`runtime`](runtime/README.md)、[`vfp_dit_runtime`](vfp_dit_runtime/README.md)、[`optimizers`](optimizers/README.md)、[`flow_sampling`](flow_sampling/README.md) |
| 検証・連携 | [`benchmarks`](benchmarks/README.md)、[`verify`](verify/README.md) |

設計・監査・研究記録:

- [`docs/architecture/image-latent-dit.md`](docs/architecture/image-latent-dit.md): 現行image-latent DiTの構成、shape、学習目的、checkpoint契約
- [`docs/VFP-DiT_Research_Note.md`](docs/VFP-DiT_Research_Note.md): VFP-DiTの現行手順・移行情報・研究履歴への索引
- [`docs/optimizers.md`](docs/optimizers.md): optimizerの分類、選択、低rank state、AutoSchedule
- [`docs/low-rank-adapters.md`](docs/low-rank-adapters.md): LoRA・DoRA・LoHAの対象層、checkpoint、検証方針
- [`docs/rglu-lora.md`](docs/rglu-lora.md): GLU-LoRA / Residual GLU-LoRAの設計仮説と実験計画
- [`docs/lr-schedulers.md`](docs/lr-schedulers.md): train script共通の学習率schedulerとwarmup
- 履歴snapshotの索引と読み方は[`docs/history/README.md`](docs/history/README.md)を参照
- [`AGENTS.md`](AGENTS.md): 実装・レビュー時のリポジトリ規約

## ライセンス

リポジトリ内のプロジェクトコードと文書は、利用者が選択できる[MIT](LICENSE-MIT)または[Apache-2.0](LICENSE-APACHE)のデュアルライセンスです。MITの著作権表示はGitHubアカウント[@namba3](https://github.com/namba3)を示しています。第三者のコードや素材には個別の条件が適用される場合があります。外部データセット・モデルの条件は[provenance一覧](docs/data-model-provenance.md)を確認してください。

## 注意事項

- 大規模モデルの学習には十分なVRAMが必要です。
- `image_gen`の学習可能パラメータは既定でBF16、VAEとtext encoderはfreezeされます。
- `--gradient-checkpointing`と`--mhla-recompute-output`はactivation VRAMを減らす代わりに再計算を追加します。
- `--perf`系オプションは計測用のoverheadを加えるため、通常の速度比較とは分けて使用してください。
- checkpointのarchitecture metadataとCLI設定が一致しない場合、resume前に構成を確認してください。
