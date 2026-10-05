# Adapter comparison probes

CIFAR-10とTinyImageNetを使うadapter比較の実行手順です。学習を開始する例を含むため、device、時間、出力先を確認してから実行してください。

## CIFAR-10 adapter comparison

実際の`cifar10.CIFAR10ViT`を使い、固定した画像・ラベル・初期weightでLoRA、LoHA、DoRA、
GLU-LoRA、Residual GLU-LoRAを比較できます。datasetのダウンロードに依存しない短期probeであり、
実データの分類精度を代替するものではありません。

```bash
python3 -m verify.cifar10_adapter_comparison \
  --device cpu --steps 10 --rank 4 \
  --output output/cifar10-adapter-comparison.json
```

JSONには、初期/最終loss、trainable parameter数、optimizer state bytes、step時間、
merge前後の最大絶対誤差、CUDA時のpeak allocated/reservedを記録します。Residual
GLU-LoRA系ではgateの平均値も記録します。短期probeの結果だけでadapterの既定値を変更せず、
実CIFAR-10 datasetでの複数seed・epoch比較を別途行います。

実CIFAR-10画像で複数seed・epochを比較する場合は、dataset比較器を使います。既定では
train 512枚、validation 256枚の短いsubsetを使用します。`--max-train-samples`と
`--max-validation-samples`を増やせば、同じ条件で実験規模を拡大できます。

```bash
python3 -m verify.cifar10_adapter_dataset_comparison \
  --data-dir cifar10/data \
  --seeds 0,1,2 \
  --epochs 2 \
  --max-train-samples 512 \
  --max-validation-samples 256 \
  --output output/cifar10-adapter-dataset-comparison.json
```

この比較器は実データを使いますが、モデルは比較を軽量化した`CIFAR10ViT`構成です。
最終的な分類品質の判断には、full `cifar10.train`で同じseed・optimizer・学習率・rankを
使った検証が必要です。

## TinyImageNet adapter comparison

Hugging Face datasetsの`zh-plus/tiny-imagenet`（`train` / `valid` split）を使い、64x64画像・
200クラスの比較器を実行できます。datasetのパスは直接指定せず、通常のHugging Face
datasetsキャッシュから読み込みます。CIFAR10用のsmoke probeとは分離し、
LoRA/LoHA/DoRA/GLU-LoRA/RGLU-LoRAをparameter-matched条件で比較します。
比較器のmoduleは`verify.tiny_imagenet_adapter_dataset_comparison`で、
下記launcherもこのCLIを呼び出します。引数一覧だけ確認する場合は
`PYTHONPATH=. python3 -m verify.tiny_imagenet_adapter_dataset_comparison --help`を実行してください。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TINY_IMAGENET_ADAPTER_DATASET=zh-plus/tiny-imagenet \
TINY_IMAGENET_ADAPTER_BUDGET=rank32 \
TINY_IMAGENET_ADAPTER_DEVICE=cuda \
TINY_IMAGENET_ADAPTER_DTYPE=bf16 \
TINY_IMAGENET_ADAPTER_EPOCHS=3 \
bash benchmarks/launchers/run_tiny_imagenet_adapter_budget_comparison.sh
```

既定ではtrain 1024枚、validation 512枚、3 seedを使用し、結果を
`output/tiny-imagenet-adapter-budget-comparison.json`へ保存します。キャッシュ場所を
明示する必要がある場合だけ`TINY_IMAGENET_ADAPTER_CACHE_DIR`または`--cache-dir`を指定します。

短期smokeから収束性能の比較へ進める場合は、train 10,000枚、validation 2,000枚、10 epochの
wrapperを使います。ランダムbaseを使う比較では、全方式に共通する分類headも学習対象にします。
rank32 budgetと3 seedは短期条件から引き継ぎます。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
bash benchmarks/launchers/run_tiny_imagenet_adapter_long_comparison.sh
```

結果は既定で`output/tiny-imagenet-adapter-long-head-comparison.json`へ保存されます。
各caseにはepochごとの`train_loss` / `train_loss_std`、`validation_loss` / `validation_accuracy`、
初期値からの`validation_loss_delta`、epoch内のstep時間も記録されます。

```bash
python3 -m verify.optimizer_convergence \
  --device cuda --dtype bf16 --warmup 5 --steps 50 \
  --rank 8 --matrix-fallback auto > output/optimizer-convergence.json
```

比較条件を固定したままrankだけを変える場合:

```bash
python3 -m verify.optimizer_convergence \
  --device cuda --dtype bf16 --warmup 5 --steps 200 \
  --rank 1 --optimizers APOLLO,APOLLO-CAME
```

APOLLO系のrank・learning rate・scaleを組み合わせて一括比較する場合:

```bash
python3 -m verify.optimizer_convergence \
  --device cuda --dtype bf16 --warmup 5 --steps 200 \
  --ranks 1,4,8 --learning-rates 3e-4,1e-3,3e-3 \
  --scales 0.5,1.0 --optimizers APOLLO,APOLLO-CAME,APOLLOMini \
  > output/optimizer-convergence-sweep.json
```

norm-growth limiterの影響を分けて測定する場合は、同じコマンドに
`--disable-norm-growth-limiter`を追加します。CAMEはrank/scaleの組み合わせを重複実行せず、
learning rateごとに1 caseだけ実行します。各caseには実際に使用したrank、learning rate、
scale、limiter設定が記録されます。
