# Mini-ImageNet GQA block comparison design

> **Document status:** This is the experiment design and CPU-validation record for the Mini-ImageNet GQA comparison. It is not the authority for current CLI defaults or the latest result summary; use [`mini_imagenet_gqa/README.md`](../mini_imagenet_gqa/README.md) for current CLI defaults and [`mini-imagenet-gqa-results.md`](mini-imagenet-gqa-results.md) for aggregate result records.

## Question

Measure whether query/key RMSNorm and per-token, per-head attention gates improve a compact spatial Transformer classifier, and compare gate activation and block FFN choices while holding the surrounding architecture and data pipeline fixed. QK RMSNorm is common across every arm, as specified for this experiment; this sweep therefore does not estimate a QK RMSNorm on/off effect.

## Controlled model

```text
(Conv2d(k=3, stride=2, padding=1) + M x spatial GQA Transformer) x N
  -> learned-query ungated GQA attention pooling
  -> RMSNorm -> SiLU GatedFFN -> linear class projection
```

- Stage widths default to `96,192,256`; each stage has one downsample and one block (`N=3`, `M=1`).
- Head count defaults to 4 and KV head count to 2 at every stage. Each stage width must divide by the query head count, and each head dimension must be divisible by 4.
- Q/K use per-head RMSNorm. 2D RoPE applies pair-wise rotations to Q and K on the row-major grid at each stage. SDPA receives Q `(B,Hq,T,Dh)` and K/V `(B,Hkv,T,Dh)` with native GQA enabled.
- Every gate is token and query-head dependent. Sigmoid uses `2 sigmoid(a)`. SiLU uses `1 + SiLU(a)`. Both initialize as identity scale with zero projection weights and bias.
- Plain block FFN is `Linear(d,4d) -> GELU -> Linear(4d,d)`. Gated block FFN is SiLU GatedFFN with hidden width approximately `8d/3`, matching projection compute/parameters approximately.
- The shared attention pool and final GatedFFN classifier stay fixed across variants.

| Arm | Attention | Block FFN | Metadata conditioning |
| --- | --- | --- | --- |
| `naive_gqa` | QK norm, no gate | plain | none |
| `naive_gqa_gated_ffn` | QK norm, no gate | gated | none |
| `gated_gqa_sigmoid` | QK norm, sigmoid gate | plain | none |
| `gated_gqa_silu` | QK norm, SiLU gate | plain | none |
| `gated_gqa_sigmoid_gated_ffn` | QK norm, sigmoid gate | gated | none |
| `gated_gqa_silu_gated_ffn` | QK norm, SiLU gate | gated | none |
| `ada_gated_gqa_silu_gated_ffn` | QK norm, SiLU gate | gated | source resolution + aspect ratio |

The Ada arm embeds two source-image features, `log(sqrt(H*W))` and `log(W/H)`, before augmentation. It applies separate zero-initialized scale-only AdaRMS projections before attention and FFN in every block. Its SiLU head gate is shared with the non-Ada SiLU arm. The adaptive scales begin at zero, so both norms start as ordinary RMSNorm and then learn metadata conditioning. The non-Ada SiLU + gated-FFN arm is the matched control for the Ada arm, so their difference measures metadata conditioning within this block configuration. The total matrix is not a full factorial study of every gate, FFN, and Ada combination; the report includes parameter counts to make the Ada capacity increase visible. With default widths and 100 classes, the Ada model has 2,616,256 parameters versus 2,459,008 for its control (+157,248, about 6.4%).

## Dataset and image pipeline

Use `timm/mini-imagenet` with its provided `train`, `validation`, and `test` splits. Load class count from the dataset `ClassLabel`; do not assume the card's prose and viewer metadata always agree. The dataset card currently documents train=50,000, validation=10,000, test=5,000 and original-size images. The validation set selects the best epoch; test is evaluated once from that checkpoint. The Hub card reports an ImageNet license and a total download near 7.43 GB, so the trainer leaves HF's default cache location in effect unless `--hf-cache-dir` is explicitly provided.

Training transforms copy the CIFAR-10 sequence from `cifar10/train.py`: horizontal flip -> ColorJitter -> random affine (default ±10 degrees/shear 10) -> perspective distortion 0.1 -> RandomResizedCrop (scale 0.5–1.0) -> tensor -> ImageNet normalization. Resolution defaults to 64 and is configurable. Evaluation uses resize + center crop and ImageNet normalization.

## Comparison protocol

- Use the same resolution, widths, heads, KV heads, block count, optimizer, learning rate, weight decay, batch size, epochs, AMP setting, and seed for all arms.
- Initial screening uses one seed and complete train/validation splits. Repeat the complete matrix for at least three seeds before interpreting small accuracy differences.
- Report best validation top-1, the corresponding test top-1/loss, per-epoch loss/accuracy, parameter count, throughput, and peak CUDA allocated/reserved memory separately. Epoch timing synchronizes CUDA before and after the train loop, so it includes completed device work; throughput is end-to-end and includes data loading and per-step progress/metric reporting, not an isolated kernel benchmark. The report utility aggregates epoch throughput and memory within each run, then computes across-seed mean/std.
- The comparison runner accepts environment overrides for seeds, variants, epochs, batch size, train/eval caps, and output directory. The summarizer verifies protocol settings and rejects mixed configurations; capped output is labeled as screening. Use a separate output directory for each protocol.
- `--steps-per-epoch` and `--eval-batches` exist only for pipeline smoke/screening and must be off for a final comparison.

## Validation status

The implementation has CPU contract tests for all variants, rectangular GQA RoPE, forward/backward shapes, gate initialization, and Ada metadata flow. An offline probe of the cached dataset loaded all three splits and sampled the first 512 images from each: it found 212, 221, and 155 distinct source sizes in train, validation, and test respectively. This sample confirms that source resolution/aspect metadata varies, but it is not a full-split distribution analysis. The Ada `--validate-only` path also completed on CPU with the cached data. Both the matched SiLU/GatedFFN control and Ada arm completed a one-step CPU training smoke using 32px images, widths 16/32, and one validation/test batch; checkpoint save and test evaluation completed. These reduced, capped runs validate the pipeline only and provide no model-quality evidence. No full training run or CUDA/BF16 result is implied.
