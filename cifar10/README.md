# `cifar10/`

CIFAR-10分類Transformerの学習scriptです。実行時の完全なCLIは`train.py --help`を確認してください。

## 実行例

```bash
cd cifar10
PYTHONPATH=.. python3 -m cifar10.train --epochs 10 --batch-size 256
```

起動前検証:

```bash
# dataset/modelをロードしない軽量確認
PYTHONPATH=.. python3 -m cifar10.train --dry-run --epochs 1 --batch-size 2 --num-workers 0

# CIFAR-10 datasetと分類modelを確認し、学習せず終了
PYTHONPATH=.. python3 -m cifar10.train --validate-only --epochs 1 --batch-size 2 --num-workers 0
```

`--validate-only`は実データをロードするため、`cifar10/data`にdatasetがない場合は取得が発生します。検証結果は`output/runs/<run-id>/metrics.jsonl`へ保存されます。

## LoRA

既存checkpointをbaseとして、attention projectionだけを低rank更新する実験ができます。adapter実験の専用entrypointは`train_adapter.py`です。

```bash
PYTHONPATH=.. python3 -m cifar10.train_adapter \
  --init-checkpoint output/base.safetensors \
  --lora-rank 4 \
  --lora-alpha 4
```

`--lora-rank 0`が通常のfull fine-tuningです。`--lora-target`は追加の正規表現として複数指定できます。

`--adapter dora`、`--adapter loha`、`--adapter glu_lora`、`--adapter rglu_lora`でDoRA/LoHA/GLU-LoRA/Residual GLU-LoRAも選択できます。adapter引数は`train_adapter.py`専用です。

checkpointなしでランダム初期化したbaseをfreezeしてadapterだけを学習する場合は、`--base-init random`を指定します。

```bash
PYTHONPATH=.. python3 -m cifar10.train_adapter \
  --base-init random \
  --adapter lora \
  --lora-rank 4
```

学習済みadapterを通常のLinearへmergeしてexportする場合:

```bash
PYTHONPATH=.. python3 -m cifar10.export_adapter \
  --checkpoint output/runs/.../checkpoints/model.safetensors \
  --output output/cifar10-merged.safetensors
```

export後はadapter moduleとresume用optimizer stateを含まない、推論用のplain model checkpointになります。
