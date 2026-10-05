# 開発環境と事前検証

[日本語](development-and-validation.md) | [English](development-and-validation.en.md)

特に記載がない限り、コマンドはリポジトリのルートから実行します。

## 起動前の検証

`mnist`、`cifar10`、`text_lm`、`image_ae`、`image_gen`の5つの標準trainerは、同じ学習前確認modeを提供します。`vfp_dit`と`mini_imagenet_gqa`にもpackage固有の確認modeがあります。詳細は各READMEを参照してください。

- `--dry-run`: 引数・device・dtype・resume設定だけを確認し、datasetやmodelをロードしない
- `--validate-only`: datasetとmodelをロードして、件数・shape・parameter数・finite性を確認する。学習は開始しない

軽量な確認:

```bash
PYTHONPATH=. python3 -m mnist.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m cifar10.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m text_lm.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m image_ae.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m image_gen.train --dry-run --vae-model Qwen/Qwen-Image
```

実データとmodel構成まで確認する例:

```bash
PYTHONPATH=. python3 -m mnist.train --validate-only --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m cifar10.train --validate-only --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m image_ae.train --validate-only --dataset cifar10 --data-dir cifar10/data --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m image_gen.train --validate-only --vae-model Qwen/Qwen-Image --epochs 1 --batch-size 1 --num-workers 0
```

`--validate-only`は実データ・tokenizer・VAE・model weightsをロードするため、未取得のものは外部から取得される場合があります。両モードは同時に指定できません。検証結果は`<output-dir>/runs/<run-id>/metrics.jsonl`の`preflight`/`validation` eventに保存されます。

## 開発環境と検証

このリポジトリはPythonの対応version rangeを固定していません。利用するPyTorch releaseと、そこで使うOS・Python・CPU/CUDA/ROCmに対応した環境を選んでください。PyTorchの対応環境は[公式install selector](https://pytorch.org/get-started/locally/)で確認できます。torchとtorchvisionはselectorが示す組み合わせで導入してください。

仮想環境を作成します。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShellでは次を使います。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

activateした仮想環境内で、公式selectorが表示するPyTorch/torchvisionのinstall commandを実行してください。

PyTorch/torchvisionのwheelを導入した後、リポジトリの実行時依存関係を導入します。

```bash
python -m pip install -r requirements.txt
```

`requirements.txt`は全experiment共通の直接依存関係一覧です。個別experimentだけを動かす最小依存一覧ではなく、version固定のlockfileでもありません。`requirements-dev.txt`には、この一覧と開発・検証用依存関係が含まれます。

`datasets`、`diffusers`、`huggingface-hub`、`transformers`は主に外部dataset/model連携に使います。`aptx-activation`、`came-pytorch`、`muon-optimizer`、`schedulefree`は対応するactivation/optimizer実装で使います。

```bash
python -m pip install -r requirements-dev.txt
```

GPU用Triton kernelとNVML telemetryは任意依存です。

通常の検証:

```bash
PYTHONPATH=. python3 -m pytest -q
bash scripts/compile_python.sh
python3 -m pyright
ruff check .
git diff --check
```

CUDA/Tritonや長時間学習を必要とする検証は、通常のunit testとは分けて実行してください。CPUでの検証結果は、CUDA/Tritonでの挙動を保証しません。

unit/integration testの区分と個別の実行方法は[`tests/README.md`](../../tests/README.md)を参照してください。GitHub ActionsはCPU suite、compileall、pyright、Ruffを必須チェックとして実行します。

`compileall`の対象directory一覧は[`scripts/compile_python.sh`](../../scripts/compile_python.sh)で管理します。syntaxを確認するもので、import/runtime動作を保証するものではありません。

実VAEのshape、stride、encode→decodeを確認する場合は、`verify/`のruntime検証を使います。

```bash
python3 -m verify.qwen_vae --vae-model Qwen/Qwen-Image --vae-dtype fp32
```
