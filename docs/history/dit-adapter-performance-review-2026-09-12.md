# DiT / LLM Adapter 性能レビュー（2026-09-12 snapshot）

最終更新: 2026-09-12

この文書は2026-09-12時点の性能計測と改善候補のsnapshotです。実装状態や未完了項目は更新後に変わっている可能性があるため、利用者向けの実行方法は[`../image_gen/README.md`](../../image_gen/README.md)、現行のモデル契約は[`architecture/image-latent-dit.md`](../architecture/image-latent-dit.md)、数値監査は[`repository-audit-2026-09-12.md`](repository-audit-2026-09-12.md)、optimizerの現行契約は[`optimizers.md`](../optimizers.md)を参照してください。benchmark値は記載した条件の記録であり、現行hardwareでの性能保証ではありません。

## 目的

当時の画像生成モデルについて、以下を継続調査の対象としていた。この一覧は2026-09-12時点の記録で、現在の作業計画を示すものではない。

- DiT / MMDiT本体の不具合確認
- LLM Adapterの不具合確認
- MHLAとFlash/SDPAの実運用上の性能比較
- 学習時のVRAM使用量とkernel呼び出し数の削減
- 高解像度・長文tokenでのスケーリング確認

## 記録時点の構成

- VAE: Qwen Image VAE、BF16
- Text Encoder: Qwen3.5-0.8B、凍結、inference mode + AMP
- Text Adapter: `1024 -> 2048 -> 1024 -> 1024`
- Adapter Transformer: 3層、heads=`16, 8, 8`、KV heads=`8, 4, 4`
- Adapter FFN: GatedLinear、hidden multiplier=`3`
- Main DiT: dim=`1024`、depth=`12`、heads=`16`、KV heads=`8`
- Main attention pattern: `mhla3-full1`
- Context Transformer: depth=`2`、image digest + text
- Image Context Embedder: `Conv2d(..., stride=2)`を3層
- Main latent path: `GatedConv2d + GroupNorm + GatedResidualConvFFNBlock`
- Output path: `RMSNorm2d + ConvTranspose2d + ResidualConvFFNBlock`
- 学習パラメータ: BF16、optimizer stateはFP32
- 既定の`timestep_repeats`: `4`

## 確認済みのbenchmark

### 低token数

設定: batch=`2`、latent=`64x48`、text=`9`、dim=`1024`、depth=`1`。

Full AttentionがMHLAより僅かに高速だった。

### 中token数

設定: latent=`64x64`、text=`128`。

概算token数:

```text
main latent:    32x32 = 1024
image context:   4x4  =   16
text context:             128
合計:                   1168
```

結果:

| 指標 | Full | MHLA |
|---|---:|---:|
| forward median | 20.89 ms | 17.85 ms |
| backward median | 29.76 ms | 28.01 ms |
| 1 step合計 | 50.65 ms | 45.86 ms |
| peak allocated | 1291.5 MiB | 890.3 MiB |
| peak reserved | 1412 MiB | 1182 MiB |

MHLAは中央値で約9.5%高速、allocated VRAMで約31%削減。ただしp90はMHLAが僅かに悪い。

### 高token数

設定: latent=`128x128`、text=`9`。

概算token数:

```text
main latent:    64x64 = 4096
image context:   8x8  =   64
text context:               9
合計:                    4169
```

結果:

| 指標 | Full | MHLA |
|---|---:|---:|
| forward median | 1239.57 ms | 160.73 ms |
| backward median | 904.82 ms | 277.03 ms |
| 1 step合計 | 2144.39 ms | 437.76 ms |
| peak allocated | 10092.2 MiB | 1487.3 MiB |
| peak reserved | 11008 MiB | 1826 MiB |

MHLAは1 stepで約4.9倍高速、allocated VRAMは約85%削減。

ただし上記MHLAは`triton_backward=False`であり、Triton forward + vectorized PyTorch backwardである。純Triton backwardの性能比較ではない。

## 記録時点で未完了だった確認事項（2026-09-12）

以下は当時の状態で、現在の対応状況や優先順位を示しません。

| 項目 | 2026-09-12時点の記録 |
| --- | --- |
| 実運用構成のbenchmark | `mhla3-full1 / depth=12 / context_depth=2`を指定できるようにしたが、batch・解像度・text token数・gradient checkpointingを含む実運用測定は未記録。 |
| 純Triton backward | `MHLA_TRITON_BACKWARD=1`の初回compileは長時間化。比較済みのMHLA値は`MHLA_TRITON_BACKWARD=0`（Triton forward + vectorized PyTorch backward）で、純Triton backwardの性能結果ではない。 |

当時の検証メモとして、`--validate-only`は`validation_seconds`とCUDAの`peak_allocated_bytes`/`peak_reserved_bytes`を記録し、CPUではpeak memoryを`null`としていた。また、inference modeで生成したTensorを学習backwardのgraphへ渡さないことを確認項目としていた。

## 大きな性能ボトルネック候補

### 1. Text Adapterの`timestep_repeats`重複計算

Text Adapterはtimestepに依存しないため、学習ループのrepeat間で出力を共有し、勾配を出力境界で集約する実装に変更した。

該当箇所: `image_gen/train.py`の学習ループ内`text_adapter_forward(...)`。

`timestep_repeats=4`でもAdapter forwardは1回だけ実行する。DiTにはAdapter出力のdetach tensorを渡し、各repeatのbackwardで出力勾配を蓄積した後、Adapter本体へ1回だけ逆伝播する。

単純な`text_condition_tokens.detach()`による再利用は採用していない。Adapter共有によるstep時間、peak VRAM、勾配の数値一致は実運用構成で再測定する。

### 2. QKV projectionのkernel呼び出し数

MMDiTの各blockではlatent/image/textそれぞれにQ projectionとKV projectionがあり、さらにstreamごとのoutput projectionとhead gateがある。

該当箇所: `MMDiTJointAttention`、`JointMHLA`。

`GroupedQueryProjection`のQとKVは、1つのfused Linearへ統合した。旧checkpointの`q.*`/`kv.*`はload時に連結する互換処理を持つ。

残りの候補:

- QKV projectionをstream単位でfused化
- output projectionのfuse
- Q/K RMSNorm + RoPEのfuse

当時の`--perf`は、optimizer step単位のhost/GPU時間、CUDA memory、利用可能な場合のGPU telemetry、モデル各段階・data pipelineの時間を`performance.jsonl`へ記録した。backward breakdownとoptimizer breakdownは追加計測のoverheadがあり、包含関係にある区間の値は加算できない。計測項目の実装は[`image_gen/performance.py`](../../image_gen/performance.py)、現行CLI例は[`image_gen/README.md`](../../image_gen/README.md)を参照。

記録時点では`num_workers > 0`で`persistent_workers=True`を使い、data wait、host-to-device投入、batch準備を分けて供給待ちを調べていた。

実測では`apollo_update_apply`がAPOLLO時間の約52%、`apollo_fallback_update`が約24%、`apollo_low_rank_stats`が約16%だった。これは最適化前の計測値で、以下の変更による速度・VRAMの改善量を示すものではない。

記録された一時テンソル削減は次のとおり。

- full-rank updateのparameter適用で、BF16/FP16向けの`update.to(parameter.dtype)`を避けて`parameter.add_(update, alpha=-lr)`を使う。
- CAME fallbackの二次momentを`addcmul_`とscalar epsilon加算で更新し、fallback専用RMS clippingを`torch.linalg.vector_norm(update) / sqrt(numel)`で求める。いずれもfull-sizeの中間テンソルを避ける。APOLLOCAME共通RMS経路はこの変更の対象外。
- 通常APOLLOでは`exp_avg`のcloneを避け、scaleとlimiter比率をscalarとしてparameter applyへ渡す。`scale=1.0`なら恒等scale処理を省き、可能な場合は`parameter.addcmul_`でlimiter比率を適用する。AutoScheduleやupdateをin-place変更する経路では、必要な作業bufferとmaterialized updateを維持する。
- norm-growth limiterでは`scaled_grad_norm`のstate tensorを次stepの上限・比率の作業領域に再利用する。更新後にcurrent normを保存する契約は維持し、`DualRotAPOLLO`にも適用する。

optimizer breakdownでは`apollo_update_apply`をlimiterとparameter applyに分けて計測した。これらの内訳は親区間に含まれるため加算できない。変更後の数値一致・checkpoint互換性・GPU性能を含む検証状況は、このsnapshotからは確認できない。現行の契約は[`optimizer documentation`](../optimizers.md)と実装を参照。

### 3. stream結合の`torch.cat`

MMDiTとMHLAではQ/K/Vを毎block結合している。

該当箇所:

- `MMDiTJointAttention.forward`
- `JointMHLA.forward`

高解像度では大きな一時テンソルのコピーになる。offsetベースのfused attentionで削減できる可能性がある。

### 4. FFNの未融合

Main DiTはlatent/image/textそれぞれに通常FFNを持つ。現在は`Linear -> SiLU -> Linear`が個別kernelとして実行される。

主DiTのパラメータ数は約814.5M、Text Adapterは約79.8Mであり、全体の計算量はDiT側が支配的。

候補:

- fused MLP
- selective `torch.compile`
- FFN専用Triton kernel

## 不具合・保守性リスク

### Text AdapterのIdentity初期化とzero gate

Text Adapterはblock内のresidual gateをzero初期化するが、入力から最初の幅へのprojectionは学習可能なLinear層であるため、Adapter全体は厳密なIdentity初期化ではない。既存conditioningからの転移を行う場合は、部分Identity初期化が候補になる。

このsnapshotでは、Context Transformer、Main DiT、最終ConvTransposeにもzero初期化gateがあることを確認した。学習初期のlossや各モジュールのgradient normを比較した測定結果は記録されていないため、勾配への影響は未評価である。現在の構成は[Image-latent DiT architecture](../architecture/image-latent-dit.md)を参照。

## 実装状況と未検証項目

### RoPE kernelの統合状況

共通kernelは[core/kernels/rope.py](../../core/kernels/rope.py)にあり、naive版とTriton版を切り替えられる。Triton版はforwardをkernel化し、backwardはPyTorchのreference計算を使う。一方、image_genは[image_gen/layers.py](../../image_gen/layers.py)のPyTorch実装を引き続き使っている。このsnapshotにはCUDAでの数値・速度比較を記録していないため、学習経路へ統合する判断は未検証。

### mask生成のcache

DiT forwardではFull Attention用の結合済みmaskを一度生成してFull block間で共有する。MHLAのvalidity metadataもforwardごとに一度作成し、block layoutとpadded layoutは`JointMHLALayoutCache`で形状間共有する。

各block末尾のtext token `masked_fill`は残っている。padding tokenの値を次blockへ持ち越さない役割があるため、削除は正しさを保てる条件と実測効果を確認してから判断する。
