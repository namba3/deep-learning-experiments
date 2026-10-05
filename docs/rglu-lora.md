# GLU-LoRA family

## Status

これは、LoRA・LoHA・DoRA・GLUの考え方を組み合わせたPEFT（Parameter-Efficient Fine-Tuning）手法の設計メモです。
本書では、offsetなしのGLU-LoRAと、残差offsetを持つResidual GLU-LoRAを同じweight-space gateの系列として扱います。
実装とCIFAR-10、ImageAE、Qwen/TinyStories、TinyImageNet-200の比較は完了していますが、一般的な優位性は確認できませんでした。
本テーマの実験は2026-09-14時点で一旦終了し、未検証項目は再開時候補として保留します。

実装は既存の[`core.low_rank`](../core/low_rank.py)のLinear adapterを基礎に行い、`cifar10`での軽量な数値検証から
`image_ae`、Qwen/TinyStories、TinyImageNet-200へ段階的に展開しました。forward/backward、merge等価性、
checkpoint/resume、parameter-matched比較、初期〜中期収束の確認を完了しています。

## Motivation

既存の低rank adapterには次の特徴があります。

| 手法 | 更新式 | 長所 | 課題 |
| --- | --- | --- | --- |
| LoRA | `ΔW = BA` | 単純、merge可能、学習が安定しやすい | rank制約が強く、directionとmagnitudeが混在する |
| LoHA | `ΔW = (B1A1) ⊙ (B2A2)` | LoRAより表現力が高い可能性がある | 2因子の役割が不明瞭で、初期化の影響を受ける |
| DoRA | directionとmagnitudeを分離 | Full Fine-Tuningに近い表現を期待できる | LoRAとは異なるparameterizationになる |

GLU-LoRAは、LoHAの2因子をValueとGateとして再解釈し、gate側へSiLUを導入する。
Residual GLU-LoRAは、さらにgate側へ残差の基準値`1`を導入する。

## Formal definition

重みを`W ∈ R^(d_out × d_in)`、rankを`r`とする。

```text
V = B1 @ A1
G_glu = SiLU(B2 @ A2)
G_rglu = 1 + G_glu
ΔW_glu = scale * (V ⊙ G_glu)
ΔW_rglu = scale * (V ⊙ G_rglu)
W' = W + ΔW
```

ここで、

```text
A1, A2 ∈ R^(r × d_in)
B1, B2 ∈ R^(d_out × r)
scale = alpha / r
```

である。`⊙`は同じ`(d_out, d_in)` shapeの要素積を表す。

展開すると、

```text
ΔW = scale * V + scale * (V ⊙ SiLU(B2 @ A2))
```

となるため、LoRA相当の更新とgated correctionの和として解釈できる。

## Direction / magnitude interpretation

この手法では、厳密なDoRAのweight norm decompositionではなく、更新行列の内部構造を次のように分けて解釈する。

- `V = B1 @ A1`: 更新の基準方向またはValue
- `G = 1 + SiLU(B2 @ A2)`: 要素ごとの更新強度またはGate
- `V ⊙ G`: Gateで再重み付けされた更新

したがってDoRAとの類似性はあるが、DoRAの出力行単位magnitudeとは異なる。文書・実験結果では、この2つを同一のdirection/magnitude分解として扱わない。

ここでのGLU gateは入力activationではなく更新weight上に適用する。通常のGLUと同じ入力依存の関数ではなく、GLUに着想を得たweight-space gatingである。

## Mergeability contract

Gateを入力`x`から計算するのではなく、parameterだけから計算することが重要である。学習後は次の値を一度計算できる。

```python
value_weight = B1 @ A1
gate_weight = 1.0 + torch.nn.functional.silu(B2 @ A2)
delta_weight = scale * (value_weight * gate_weight)
merged_weight = base_weight + delta_weight
output = torch.nn.functional.linear(input, merged_weight, bias)
```

この定義なら、次を期待できる。

- merge可能
- merge後の推論時adapter overheadなし
- 単一の`Linear`へ畳み込み可能

逆に、`value = lora_value(x)`、`gate = silu(lora_gate(x))`のようにactivation上で積を取る実装は入力依存になるため、このmergeability contractを満たさない。activation-space版を試す場合は、Residual GLU-LoRAとは別の実験として扱う。

## Initialization policy

`SiLU(0) = 0`なので、`B2 @ A2 ≈ 0`の初期状態では`G ≈ 1`となり、更新は次に近づく。

```text
ΔW ≈ scale * (B1 @ A1)
```

これはLoRAに近い初期の最適化経路を与えるという仮説である。ただし、`V`を非zero初期化すると注入直後からbase weightが変わる。したがって、次の2つを実験設定として分ける。

1. **LoRA warm start**: `B1`をLoRA factor相当に非zero初期化し、`B2=0`によって`G=1`を保つ。初期更新は非zeroのLoRA-like updateになる。
2. **identity start**: `B1=0`かつ`B2=0`とし、注入直後の出力をbase Linearと一致させる。

実装では`core.low_rank.RGLULoRALinear(init_mode=...)`と、CIFAR-10の
`train_adapter.py --adapter-init identity|lora_warm`で切り替える。既定値はidentityである。
identity startでは、初期の更新が厳密には`ΔW=0`となる。既存adapterの「注入直後のbase一致」を優先する場合はこちらを採用する。どちらを既定値にするかは、初期loss、gradient norm、短期収束を同一条件で比較して決める。

## Parameter count

```text
W ∈ R^(d_out × d_in), rank = r

LoRA:                  r * (d_in + d_out)
LoHA:                2r * (d_in + d_out)
GLU-LoRA:           2r * (d_in + d_out)
Residual GLU-LoRA: 2r * (d_in + d_out)
```

GLU-LoRAとResidual GLU-LoRAはLoHAと同じparameter countで、gate側のSiLUと、RGLUでは残差offsetを追加する。
DoRAのmagnitudeを併用する場合は、出力channelごとの`d_out`個のparameterを別途加えるため、標準案とは区別して計測する。

## Expected properties and hypotheses

- LoRAより少ないrankでvalidation lossを改善できる可能性がある。
- Gateが更新強度を調整することで、LoHAの2因子より解釈しやすい可能性がある。
- `G`の初期値が1に近いため、LoHAの積より学習初期が安定する可能性がある。
- weight-spaceの要素積はmerge可能性を保つが、full delta weightのmaterializeによりforward時の一時メモリと計算量が増える可能性がある。
- 「学習安定性が高い」「表現力が非常に高い」「DoRAとの親和性が高い」は、実験前の期待であり、確定した性質ではない。

## Experimental plan（実験終了時点の記録）

最初の比較は`cifar10`で行い、初期weight、data order、seed、optimizer、learning rate、rank、alpha、target modulesを共有した。

最低限、次を記録する。

- 初期lossと注入直後のbaseとの差分
- identity / LoRA warm startの初期gradient norm
- trainable parameter数とcheckpoint容量
- adapter parameterのgradient normとfinite性
- validation loss / accuracyとepochごとの収束
- optimizer state bytes、peak allocated/reserved、step時間
- merge前後のforward値とgradientの差分
- `rank ∈ {1, 2, 4, 8}`、LoRA / LoHA / DoRA / GLU-LoRA / Residual GLU-LoRAの比較

許容誤差、dtype、device、target Linearのshapeを固定し、CPUではreferenceと勾配経路を確認した。TinyImageNetでは
CUDA/BF16の性能・収束・mergeも測定したが、Triton実装の性能・数値等価性はこのテーマの実験範囲に含めず未検証とする。

## Open questions（再開時候補）

1. LoHAに対する実性能差と、rankごとの性能差
2. warm startとidentity startの学習安定性
3. Gateの値の分布、飽和、gradient normへの影響
4. `alpha / rank`以外のscale、gate温度、正則化の必要性
5. QLoRAやDoRA magnitudeとの併用効果
6. Hadamard積とSiLUによる更新の理論的なrank・表現力
7. delta weightのmaterialize頻度と、forward/merge時のメモリ・速度
8. Conv2d adapterへ拡張した場合のshape契約

## Summary

GLU-LoRA（canonical identifier: `glu_lora`）とResidual GLU-LoRA（canonical identifier: `rglu_lora`）は、LoHAを再解釈するmerge可能なweight-space adapterである。旧identifier `residual_swiglu_loha`は、既存checkpointと実験commandの互換性のため`rglu_lora`のaliasとして受け付ける。

```text
LoRA:                  ΔW = BA
LoHA:                  ΔW = BA ⊙ DC
GLU-LoRA:           ΔW = BA ⊙ SiLU(DC)
Residual GLU-LoRA:  ΔW = BA ⊙ (1 + SiLU(DC))
```

LoRAのdirectionとLoHAの表現力を保ちつつ、DoRAに似た更新強度の分離と、GLUに着想を得たgatingを導入する。ただし、identity初期化、性能、安定性、理論的優位性は実験条件に依存し、今回の比較では一般的な優位性を確認できなかった。
