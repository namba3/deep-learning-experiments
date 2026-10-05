# Low-rank adapter experiment records

This page preserves dated experiment protocols, aggregate measurements, and reproduction commands. It is a historical record, not a current work plan or a claim that local run artifacts are included. Current adapter contracts are summarized in [low-rank adapters](low-rank-adapters.md), and implementation details are in `core/low_rank` and package READMEs.

## 実験入口

### 実行前提

以下のコマンドは学習済みcheckpointやdatasetを同梱していません。実行前にPython依存関係を
[`requirements.txt`](../requirements.txt)から導入し、対象entrypointのREADMEと`--help`で必要な設定を
確認してください。CIFAR-10/image_aeの例では`cifar10/data/`のdatasetと、同じmodel構成で作成した
`output/base.safetensors`が必要です。text LMの例では、指定したtokenizer・datasetを取得できる環境と、
同じmodel構成・tokenizer・subsetで事前学習したbase checkpointが必要です。これらのファイルはローカル生成物で、
公開ツリーには含めません。base checkpointがない場合の作成方法とデータ取得条件は、各package READMEの手順を
確認してください。

例中の`cifar10/data/`、`output/...`はrepository rootからの相対的なローカル保存先です。存在しない場合は
各CLIがデータ取得または出力先作成を行うことがあります。過去の実験結果を読むだけなら、以下の実行は不要です。

`cifar10`では、既存のmodel checkpointをbaseとして、adapter専用entrypointから次のように実行します。

```bash
PYTHONPATH=. python3 -m cifar10.train_adapter \
  --init-checkpoint output/base.safetensors \
  --lora-rank 4 \
  --lora-alpha 4
```

DoRAまたはLoHAを使う場合は、adapter種別を指定します。

```bash
PYTHONPATH=. python3 -m cifar10.train_adapter \
  --init-checkpoint output/base.safetensors \
  --adapter dora \
  --lora-rank 4
```

`image_ae`では、完全なbase autoencoder checkpointを明示し、adapter専用entrypointを使います。

```bash
PYTHONPATH=. python3 -m image_ae.train_adapter \
  --dataset cifar10 \
  --data-dir cifar10/data \
  --encoder window_transformer \
  --decoder window_transformer \
  --lora-base-checkpoint output/base.safetensors \
  --lora-rank 4 \
  --lora-alpha 4
```

## 実験記録

各ページは当時の集計値とprotocolを保存し、現行推奨や一般的な方式順位を主張しません。run artifactは公開treeに含みません。

- [Text-LM / TinyStories adapter comparisons](adapter-experiments/text-lm.md)
- [ImageAE adapter comparisons](adapter-experiments/image-ae.md)
- [TinyImageNet-200 adapter comparisons](adapter-experiments/tiny-imagenet.md)
- [CIFAR-10 adapter comparisons and probes](adapter-experiments/cifar10.md)

現行の数式・adapter契約は[low-rank adapters](low-rank-adapters.md)、entrypointとCLIは各package READMEを参照してください。
