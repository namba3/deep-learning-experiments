# `image_gen/`

キャプションと画像から、Qwen Image VAEのimage latentを予測するRectified Flow / Flow Matching DiTを学習・評価する実験です。semantic channelやVision Encoder teacherは現行構成に含めません。

## 入口

実装moduleは次の役割に分かれています。

- `data.py`: dataset loading、image transform、aspect-ratio bucket batching
- `layers.py`: timestep/resolution embedding、RMSNorm、RoPE、GQA attention helper
- `attention.py`: spatial RoPE caches, image self-attention, and
  latent/image/text joint attention
- `joint_attention.py`: Joint MHLA layout/cache helpers, reference code, and
  backend orchestration
- `joint_attention_triton.py`: optional autotuned GPU kernels for Joint MHLA
- `dit.py`: context transformer, latent projections, Joint MHLA, and DiT model blocks
- `performance.py`: optimizer-aware memory estimates, GPU telemetry, and opt-in timing reports
- `inference.py`: text/image encoding, VAE shape checks, decoding, and flow sampling helpers
- `checkpoint.py`: image_gen model metadata, weight transfer, and safetensors save/load helpers
- `cli.py`: training defaults and the training argument parser
- `training_config.py`: checkpoint-aware training argument restoration and validation
- `training_data.py`: dataset, VAE bucket validation, latent probe, loader, and scheduler step setup
- `training_model.py`: trainable model construction and initialization checkpoint handling
- `training_optimizer.py`: trainable parameter grouping and optimizer construction
- `training_resume.py`: model, sampler, optimizer, and scheduler checkpoint save and restoration
- `training_runtime.py`: device, precision, and compile setting resolution
- `training_components.py`: pretrained tokenizer and frozen encoder loading
- `training_report.py`: resolved training configuration summary output
- `text_conditioning.py`: bidirectional text adapter and its GQA transformer blocks
- `train.py`: model、学習・検証処理とtraining CLI

学習:

```bash
python3 image_gen/train.py \
  --dataset-name lmms-lab-encoder/flickr30k \
  --vae-model Qwen/Qwen-Image \
  --output-dir image_gen/output
```

ローカルJSONL/CSV:

```bash
python3 image_gen/train.py \
  --records data.jsonl \
  --vae-model Qwen/Qwen-Image \
  --output-dir image_gen/output
```

生成:

```bash
python3 image_gen/generate_samples.py \
  --checkpoint image_gen/output/<run>/checkpoint_latest.safetensors \
  --vae-model Qwen/Qwen-Image \
  --prompt "a photograph of a child playing with a dog outdoors"
```

起動前検証:

```bash
# 引数・device・dtypeだけを確認し、モデルとdatasetをロードしない
python3 image_gen/train.py --dry-run --vae-model Qwen/Qwen-Image

# text encoder、VAE、bucket、latent shape、DiT構成を確認する
python3 image_gen/train.py \
  --validate-only \
  --dataset-name lmms-lab-encoder/flickr30k \
  --vae-model Qwen/Qwen-Image \
  --epochs 1 --batch-size 1 --num-workers 0
```

`--validate-only`はtext encoder、VAE、datasetをロードするため、未取得の重みやdatasetは外部から取得される場合があります。検証結果は`<output-dir>/runs/<run-id>/metrics.jsonl`へ保存され、学習は開始しません。

`--vae-model`は学習・生成の両方で必須です。生成時のtext model、prompt、step数、解像度は`generate_samples.py --help`で確認してください。

## Model dtype inspection

To inspect dtypes recorded in locally cached safetensors without loading the
models into runtime memory, disable runtime inspection explicitly:

```bash
PYTHONPATH=. python3 -m image_gen.inspect_model_dtypes \
  --vae-model Qwen/Qwen-Image --no-runtime --disk
```

Runtime inspection is enabled by default and loads the vision model and VAE;
use `--runtime` only when that model load is intended. `--download-file-weights`
also allows downloading uncached weight files for disk inspection.

## 主なCLI契約

- `--vae-dtype bf16|fp32`: VAEのdtype。既定は`bf16`
- `--trainable-dtype bf16|fp32`: DiTとText Adapterのdtype。既定は`bf16`
- `--model-dim 1024 --depth 12 --heads 16 --kv-heads 8`: main MMDiTの既定構成
- `--context-depth 2 --context-heads 16 --context-kv-heads 8`: context transformerの既定構成
- `--attention-pattern full|mhla|mhla3-full1`: main attentionの構成。既定は`mhla3-full1`
- `--mhla-backend auto|naive|vectorized|triton`: MHLA backend。`triton`はCUDAが必要
- `--num-workers 4`: DataLoaderの既定値。`0`でworkerなし
- `--resume`: 同じnetwork versionのcheckpointから再開
- `--init-checkpoint`: shapeが一致するweightだけを別構成へ部分転送

architecture version、tensor shape、loss、sampling、checkpoint互換性の詳細は[`docs/architecture/image-latent-dit.md`](../docs/architecture/image-latent-dit.md)にまとめています。

## VAE latent統計の計測

必要に応じてQwen Image VAEのlatent channel別統計を再計測できます。既定では最大10,000画像を測定し、生成JSONはignore対象の`output/image_gen/`へ保存します。過去の集計値は[記録資料](../docs/experiment_data/image-gen-qwen-latent-statistics.md)にあります。この集計値は学習時の正規化には使いません。

```bash
python3 -m image_gen.measure_latent_stats --vae-model Qwen/Qwen-Image
```

CPUで実行する場合は自動batch-size探索を無効にし、必要なら測定画像数を減らします。

```bash
python3 -m image_gen.measure_latent_stats \
  --vae-model Qwen/Qwen-Image --no-auto-batch-size \
  --max-images 128 --output output/image_gen/latent_stats.json
```

## 計測とkernel検証

通常の学習と性能計測を分けるため、計測時だけ`--perf`を追加します。詳細なbackwardまたはAPOLLO計測は追加のoverheadを伴います。

```bash
python3 image_gen/train.py --vae-model Qwen/Qwen-Image --perf ...
python3 image_gen/train.py --vae-model Qwen/Qwen-Image --perf-backward-breakdown ...
python3 image_gen/train.py --vae-model Qwen/Qwen-Image --perf-optimizer-breakdown ...
```

MHLAの数値検証:

```bash
python3 -m image_gen.validate_mhla --backend triton --pattern both
```

実Qwen VAEのpixel bucket、latent stride、encode→decodeを確認するruntime検証:

```bash
python3 -m verify.qwen_vae --vae-model Qwen/Qwen-Image --vae-dtype fp32
```

MHLA benchmark:

```bash
python3 -m image_gen.benchmark_mhla \
  --backend triton --patterns full,mhla3-full1 \
  --latent-height 64 --latent-width 64 --text-tokens 128
```

Tritonの初回compile/autotune、GPU上のforward/backward、peak VRAMはCPU testでは代替できません。監査snapshotは[repository audit](../docs/history/repository-audit-2026-09-12.md)、過去の性能計測は[DiT / LLM Adapter performance review](../docs/history/dit-adapter-performance-review-2026-09-12.md)を参照してください。
