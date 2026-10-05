# Image-latent DiT architecture

[日本語](image-latent-dit.md) | [English](image-latent-dit.en.md)

この文書は、`image_gen/train.py`の現行image-latent-only構成を、実装変更時に確認できる形で記録します。現在値はコードとCLIの`--help`を正とし、実験の経緯や未完了の検証は[`../history/repository-audit-2026-09-12.md`](../history/repository-audit-2026-09-12.md)と関連する実験レポートに分離します。

## 目的と境界

キャプションからQwen Image VAEのimage latentを生成するRectified Flow / Flow Matching DiTです。現在のモデルはVAE latentだけを予測し、semantic channel、Vision Encoder teacher、PCA/whitening projector、semantic loss、参照画像semantic overrideを持ちません。

学習するのはDiT本体とText Conditioning Adapterです。Qwen Image VAEとQwen3.5 text encoderはfreezeし、特徴抽出はinference modeとAMPで行います。

## Tensor shape契約

標準形は次のとおりです。

| 対象 | shape | 意味 |
| --- | --- | --- |
| image latent | `(B, C, H, W)` | VAEが生成するclean/noise/interpolated latent |
| text hidden state | `(B, S, E)` | frozen text encoderの出力 |
| condition tokens | `(B, S, 1024)` | Text Adapterの出力 |
| latent tokens | `(B, T, D)` | `T = H' * W'`、`D = model_dim` |
| attention Q | `(B, Hq, Tq, Dh)` | query heads |
| attention K/V | `(B, Hkv, Tk, Dh)` | GQAのkey/value heads |

`H'`と`W'`はlatentをstride 2の`LatentDownsample`へ通した後の空間サイズです。token化は`(B,D,H',W') -> (B,H'W',D)`、復元はその逆順で行います。image/text tokenを連結する場合、image tokenが先頭に置かれ、text maskにはそのtoken数分のoffsetを加えます。

## Model data flow

```text
image
  -> frozen Qwen Image VAE
  -> clean latent x0: (B, C, H, W)

caption
  -> frozen Qwen3.5 text encoder
  -> TextConditioningAdapter
  -> text tokens: (B, S, 1024)

noisy latent
  -> LatentDownsample: (B, 1024, H/2, W/2)
     ├─ flatten -> main latent tokens
     └─ ImageContextEmbedder: three stride-2 convolutions
        -> image digest tokens

image digest tokens + text tokens
  -> ContextTransformer
  -> three-stream joint attention DiT blocks
  -> LatentUpsample
  -> predicted image-latent velocity: (B, C, H, W)
```

### Text Conditioning Adapter

入力をRMSNormとLinearで最初の幅へ写像し、mask-awareな非causal transformerを通します。既定値は次のとおりです。

```text
text encoder hidden
  -> RMSNorm
  -> Linear(input_dim, 2048)
  -> bidirectional GQA block (dim=2048, heads=16, kv_heads=8)
  -> Linear(2048, 1024)
  -> bidirectional GQA block (dim=1024, heads=8, kv_heads=4)
  -> Identity
  -> bidirectional GQA block (dim=1024, heads=8, kv_heads=4)
  -> RMSNorm
  -> Text Adapter tokens (dim=1024)
```

textには1D RoPEを適用し、同じhead dimensionのcacheをblock間で共有します。padding tokenはmaskで除外し、空captionでは全masked rowを作らないsafe maskを使ってCFG用のnull conditioningを構成します。

### Main DiT / Context Transformer

既定値は`model_dim=1024`、`depth=12`、`heads=16`、`kv_heads=8`、`context_depth=2`、`context_heads=16`、`context_kv_heads=8`、`patch_size=2`です。patch sizeは現在の畳み込みstemがstride 2を要求するため2に固定されています。

Context Transformerは、共有image featureを3段の`Conv2d(kernel=4, stride=2, padding=1) -> GroupNorm -> SiLU`でimage digestへ圧縮し、text tokensと連結します。image tokenには2D RoPE、text tokenには1D RoPEを適用します。

Main DiTの各blockは、latent/image/textの3 streamを保ったまま次を行います。

1. timestep・resolutionでAda modulationしたjoint attention
2. 各stream独立のdense FFN（hidden width `4 * model_dim`）

`--attention-pattern mhla3-full1`では4 block周期でMHLAを3 block、Full Joint Attentionを1 block配置します。`full`は全blockをFull Joint Attention、`mhla`は全blockをMHLAにします。MHLAはreference/native/vectorized/Triton backendを持ち、TritonはCUDAでのみ検証対象になります。

2D RoPEはhead dimensionを座標軸ごとの偶奇pairに分けて適用します。main attentionとcontext attentionのhead数、KV head数、RoPE制約は構築時に検証します。

## 学習目的

clean latentを`x0`、Gaussian noiseを`x1`、`t in [0, 1]`として、現在のstraight pathは次です。

```text
xt = (1 - t) * x0 + t * x1
velocity_target = x1 - x0
```

`rectified_flow`と`flow_matching`は、現行のstraight pathでは同じvelocity targetを使います。モデルの主損失は次です。

```text
diffusion_loss = MSE(predicted_velocity, velocity_target)
x0_prediction = xt - t * predicted_velocity
reconstruction_loss = MSE(x0_prediction, x0)
weighted_reconstruction = reconstruction_loss_weight * reconstruction_loss
loss = diffusion_loss + capped(weighted_reconstruction)
```

既定の`reconstruction_loss_weight`は`0.1`、reconstructionのtotal lossに対する寄与上限は`0.05`です。capはimage latent reconstructionだけに適用されます。

## Sampling

samplingはimage latentだけを`t=1`のGaussian noiseから開始し、`t=1`から`0`へEuler積分します。

```text
samples ~ N(0, I)
for step in range(sample_steps):
    t = 1 - step / sample_steps
    velocity = DiT(samples, t, text_condition)
    samples = samples - velocity / sample_steps
image = frozen Qwen Image VAE decoder(samples)
```

既定のsample step数は30です。checkpointからarchitecture metadataを読み込む生成スクリプトは、構成を復元してstrict loadします。

## Checkpoint契約

- 現行`NETWORK_CONFIG_VERSION`は`43`です。
- `--resume`はnetwork versionが一致しないcheckpointを拒否します。
- checkpointにはDiTとText Adapterのweight、およびarchitecture/学習設定metadataを保存します。
- `--init-checkpoint`は、異なる構成からnameとshapeが一致するweightだけを部分転送できます。完全なresumeではありません。
- semantic channelを含む旧checkpointは、現行image-latent-only構成とはstate shapeと学習目的が異なるため、通常のresume互換性を持ちません。

## 主要な既定値

| 項目 | 既定値 |
| --- | --- |
| VAE dtype | `bf16` |
| trainable dtype | `bf16` |
| image size / bucket step | `256` / `32` |
| text max length | `256` |
| null conditioning probability | `0.05` |
| reconstruction loss weight / cap | `0.1` / `0.05` |
| sample steps | `30` |
| DataLoader workers | `4`（`0`で無効） |
| main optimizer | `APOLLO` |
| external LR scheduler | `constant` |

CLIの追加設定や正確な既定値は`python3 image_gen/train.py --help`を確認してください。
