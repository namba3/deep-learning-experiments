# `mnist/`

MNIST分類Transformerの学習scriptです。実行時の完全なCLIは`train.py --help`を確認してください。

## 実行例

```bash
cd mnist
PYTHONPATH=.. python3 -m mnist.train --epochs 10 --batch-size 256
```

起動前検証:

```bash
# dataset/modelをロードしない軽量確認
PYTHONPATH=.. python3 -m mnist.train --dry-run --epochs 1 --batch-size 2 --num-workers 0

# MNIST datasetと分類modelを確認し、学習せず終了
PYTHONPATH=.. python3 -m mnist.train --validate-only --epochs 1 --batch-size 2 --num-workers 0
```

`--validate-only`は実データをロードするため、`mnist/data`にdatasetがない場合は取得が発生します。検証結果は`output/runs/<run-id>/metrics.jsonl`へ保存されます。
