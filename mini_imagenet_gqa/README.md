# Mini-ImageNet GQA block comparison

> Paths labeled as local generated artifacts refer to ignored run reports. Aggregate measurements in this README are the public record.

This package compares head-wise attention gates, block FFNs, and metadata conditioning on image classification. QK RMSNorm is enabled in every variant, per the experiment contract. The classifier head (attention pooling plus GatedFFN) and image hierarchy stay fixed.

## Factorial comparison protocol

The controlled comparison uses a 2x2 matrix. AdaNorm means scale-only AdaRMS before both the attention and FFN branches. The second factor enables the SiLU head-wise attention gate and SwiGLU FFN together; both are disabled in the plain branch condition. QK RMSNorm, image hierarchy, attention pooling, classifier head, augmentation, buckets, optimizer, and training budget stay fixed.

| Variant | AdaRMS scale, both branches | SiLU attention gate + SwiGLU FFN |
| --- | --- | --- |
| `naive_gqa` | no | no |
| `ada_naive_gqa` | yes | no |
| `gated_gqa_silu_gated_ffn` | no | yes |
| `ada_gated_gqa_silu_gated_ffn` | yes | yes |

AdaRMS scale projections are zero-initialized, so their initial operation matches ordinary RMSNorm. The plain block FFN uses a GELU MLP with hidden width `4d`; SwiGLU uses `8d/3`, approximately matching projection parameter and multiply-add budgets. The model's attention-pooling head retains its fixed GatedFFN in every condition.

Earlier attention-gate, FFN, stem metadata, and branch metadata variants remain loadable in the model code for historical result/checkpoint inspection. The four variants in this factorial remain its controlled core. Ada scale-function and scale-plus-shift follow-ups, along with the closed QKV metadata-concat screen, are documented as historical measurements in the [results archive](../docs/mini-imagenet-gqa-results.md); they are not part of this factorial.

Q/K have shape `(B,Hq,T,Dh)` and `(B,Hkv,T,Dh)` respectively. The same 2D pair-wise RoPE is applied using the current stage `(height,width)` grid; `Hq % Hkv == 0` and `Dh % 4 == 0`. Attention uses PyTorch SDPA with GQA. Each stage has one stride-2 convolution followed by `M` blocks; the complete hierarchy repeats for each configured width. Attention pooling is shared and ungated, followed by a shared `RMSNorm -> GatedFFN -> classifier` head.

The plain FFN uses hidden width `4d`; the gated FFN uses `8d/3`, approximately matching the two FFNs' projection parameter and multiply-add budgets. The rest of the model, optimizer, augmentation, split, and training settings must match for a controlled comparison.

See the [resolution and aspect bucket analysis](../docs/experiment_data/mini-imagenet-gqa-bucket-analysis.md) for the cached dataset dimension analysis and the proposed compute-matched rectangular buckets. The analysis is implemented by `python -m mini_imagenet_gqa.analyze_image_buckets`.

## Data and augmentation

The trainer reads `timm/mini-imagenet` through Hugging Face Datasets. It uses the dataset's `train`, `validation`, and `test` splits, and discovers class names/count from the `ClassLabel` feature rather than hardcoding a class count. The dataset card reports 50,000 train, 10,000 validation, and 5,000 test images; total download size is about 7.43 GB. Validation selects the best checkpoint; test is evaluated once afterward. See the [dataset card](https://huggingface.co/datasets/timm/mini-imagenet) for split provenance and license details.

Training augmentation follows `cifar10/train.py` order: horizontal flip, `ColorJitter`, `RandomAffine`, `RandomPerspective`, then area/aspect-jittered crop (`scale=(0.5,1.0)`) resized to the selected bucket, tensor conversion, and normalization. Evaluation preserves aspect ratio, resizes to cover the bucket, and center-crops. ImageNet mean/std are used for this ImageNet subset. The proposed default buckets and measured source distributions are documented in the [bucket analysis](../docs/experiment_data/mini-imagenet-gqa-bucket-analysis.md).

## Experiment results

Aggregate results and detailed protocols for completed screens and follow-ups are in [Mini-ImageNet GQA results](../docs/mini-imagenet-gqa-results.md). The page records historical measurements; it does not imply that its launcher commands are active work.

## Analysis utilities

- `python3 -m mini_imagenet_gqa.analyze_ada_scales`: inspect effective AdaRMS scales from best checkpoints at a selected bucket size. It writes JSON and Markdown reports under `--report-dir`; use repeated `--input OPTIMIZER=DIR` options to select run roots.
- `python3 -m mini_imagenet_gqa.summarize_ada_bias_comparison --scale-only-dir <runs> --shift-dir <runs> --output-dir <report-dir>`: compare paired scale-only and scale-plus-shift runs. This helper is used by the archived seed-47–50 comparison protocol; it is not a generic recommendation for new runs.

## Run

### Launcher index

seed、variant、optimizerを固定した過去screenのlauncher一覧は[experiments archive index](experiments/README.md)を参照してください。条件を変えて再利用するrunnerはpackage直下に置いています。一般的な学習入口は `python3 -m mini_imagenet_gqa.train` と [`benchmarks/launchers/run_mini_imagenet_gqa_comparison.sh`](../benchmarks/launchers/run_mini_imagenet_gqa_comparison.sh) です。shell launcherはCUDA学習を含むため、dry-run以外は実行時間とGPU memoryを見積もり、各runで別の出力先を使ってください。結果は通常 `mini_imagenet_gqa/output/` 以下に保存されます。

条件を環境変数で指定できるpackage共通runner:

- [`run_apollo_sf_delta_refresh_comparison.sh`](run_apollo_sf_delta_refresh_comparison.sh)
- [`run_optimizer_quantization_comparison.sh`](run_optimizer_quantization_comparison.sh)

過去screenの個別launcherはarchive indexに記載したprotocolの再現用です。未確認protocolの状態と結果への対応は[実験結果archive](../docs/mini-imagenet-gqa-results.md)を参照してください。

Start with a no-download model/CLI check:

```bash
PYTHONPATH=. python3 -m mini_imagenet_gqa.train --dry-run --device cpu
```

Run one configuration:

```bash
PYTHONPATH=. python3 -m mini_imagenet_gqa.train \
  --variant gated_gqa_silu --device cuda --amp bf16 \
  --run-name gated-silu-seed-42
```

Run the original seven-arm full-split screen with one seed. This keeps the default 30 epochs and evaluates complete validation/test splits; it is a real training screen, not the capped pipeline smoke test below:

```bash
SEEDS="42" \
OUTPUT_DIR=mini_imagenet_gqa/output/bucketed/full-seed42 \
bash benchmarks/launchers/run_mini_imagenet_gqa_comparison.sh
```

This documents the original four-arm bucket-conditioned screen. Original metadata runs used source-file dimensions and remain historical under `output/full-seed42/`; do not pool them with bucket-conditioned results. QKV metadata-concat runs are archived in `output/bucketed/metadata-screen-seed42-3e/` and are not active candidates.
