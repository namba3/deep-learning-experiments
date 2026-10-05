# `image_ae/`

Flickr30k、CIFAR-10、ImageNet系データセットを使って画像autoencoderまたはVAEを学習する実験です。アスペクト比に応じたresolution bucketを作り、同じbucketの画像をbatch化します。

## 入口

```bash
python3 -m image_ae.train --dataset flickr30k --batch-size 8
```

Flickr30kの画像は各権利者に著作権があり、配布元は非商用の研究・教育向けに案内しています。画像の取得・利用条件は[データセットとモデルの出典・利用条件](../docs/data-model-provenance.md)を確認してください。

CIFAR-10:

```bash
python3 -m image_ae.train --dataset cifar10 --data-dir cifar10/data
```

再構成画像の確認:

```bash
python3 -m image_ae.reconstruct --help
```

起動前検証:

```bash
# datasetやmodelをロードせず、共通設定だけを確認
python3 -m image_ae.train --dry-run --dataset cifar10 --epochs 1 --batch-size 2 --num-workers 0

# datasetとautoencoderをロードし、input/latent/reconstruction shapeを確認
python3 -m image_ae.train \
  --validate-only \
  --dataset cifar10 \
  --data-dir cifar10/data \
  --epochs 1 --batch-size 2 --num-workers 0
```

`--validate-only`は学習を開始せず、検証結果を`<output-dir>/runs/<run-id>/metrics.jsonl`へ保存します。未取得のdatasetやmodelがある場合はロード時に取得されます。

adapter学習後の推論用merge checkpointは次で作成できます。

```bash
python3 -m image_ae.export_adapter \
  --checkpoint output/runs/<run-id>/checkpoints/<adapter-checkpoint>.safetensors \
  --output output/image-ae-merged.safetensors
```

出力はadapter moduleを含まないplain Linear checkpointです。resume用のoptimizer sidecarは作成しません。

adapter学習は通常学習と分離した`train_adapter.py`から実行します。

```bash
python3 -m image_ae.train_adapter \
  --dataset cifar10 \
  --data-dir cifar10/data \
  --lora-base-checkpoint output/base.safetensors \
  --adapter rglu_lora \
  --lora-rank 4 \
  --lora-alpha 4
```

`--adapter glu_lora`では`(B1 @ A1) ⊙ SiLU(B2 @ A2)`、`--adapter rglu_lora`では
`(B1 @ A1) ⊙ (1 + SiLU(B2 @ A2))`のweight-space更新を使います。どちらもmerge可能です。

`image_ae.train`は通常のfull-model学習専用です。adapterの注入とadapter parameterだけをoptimizerへ渡す処理は
`image_ae.adapter_training`へ分離しています。

## 主な設定

- `--dataset flickr30k|cifar10|imagenet1k|mini-imagenet`
- `--image-size`: 画像系は既定256、CIFAR-10は既定32
- `--cifar10-train-samples` / `--cifar10-val-samples`: CIFAR-10の使用件数。`0`は全件です。小型base checkpointやadapter実験ではsample数を制限できます。
- `--bucket-step 32`: bucketのheight/width alignment
- `--encoder` / `--decoder`: window、CNN、residual Conv FFNなどのvariant
- `--vae`: variational latentとKL lossを有効化
- `--resume`: checkpointを読み込みます。同じ場所に`.resume.pt` sidecarがあればoptimizer、scheduler、
  epoch、global step、乱数状態も復元し、なければweights-only resumeになります。
- `--init-checkpoint`: 共通層のweightを新規runへ初期化します。`--resume`とは併用できません。
- `train_adapter.py --lora-base-checkpoint` / `--lora-rank` / `--adapter`: adapter学習専用の設定です。通常の`train.py`では受け付けません。
- `--lora-alpha` / `--lora-dropout` / `--lora-target` / `--adapter-init`: adapter scaling、dropout、対象Linear、Residualの初期化です。
- `--num-workers 4`: DataLoaderの既定値。`0`でworkerなし
- `--apollo-rank 8 --apollo-scale 1.0`: APOLLO系のrankとscale
- `--apollo-disable-norm-growth-limiter`: APOLLO系のlimiterを無効化（既定）。有効化する場合は`--no-apollo-disable-norm-growth-limiter`を指定
- `--wavelet-loss`、`--latent-cycle-consistency`、`--image-cycle-consistency`: 追加loss/regularization

通常学習の引数と既定値は`python3 -m image_ae.train --help`、adapter学習の引数は`python3 -m image_ae.train_adapter --help`を正とします。共有層は[`../core/README.md`](../core/README.md)、optimizerの選択は[`../optimizers/README.md`](../optimizers/README.md)を参照してください。

完全resumeではepoch途中のDataLoader位置は保存しないため、途中で再開した場合も現在epochの先頭から再走査します。

## 実装上の注意

可変解像度では、入力の`(B,C,H,W)`、downsample stages、latentのchannel/spatial size、token化後の`(B,T,D)`が対応している必要があります。bucketと実VAE stride、GroupNormのchannel契約、latent channel=1を含む境界検証は[`../docs/history/repository-audit-2026-09-12.md`](../docs/history/repository-audit-2026-09-12.md)に記録しています。
