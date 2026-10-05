# Low-rank adapters

> This page summarizes adapter design and validation contracts. Dated measurements and local run artifact references are in [the experiment record](low-rank-adapters-results.md); raw runs are not part of the public tree.

## 現在の範囲

低rank追加学習はoptimizerとは別のmodel parameterizationとして扱います。現在は`core.low_rank`にLoRA、DoRA、LoHA、GLU-LoRA、Residual GLU-LoRAを実装しています。
Residual GLU-LoRAのcanonical adapter identifierは`rglu_lora`です。過去checkpointと旧実験commandの`residual_swiglu_loha`は互換aliasとして読み込めます。
GLU-LoRAのcanonical adapter identifierは`glu_lora`です。

## 実験の区切り（2026-09-14）

LoRA・DoRA・LoHA・GLU-LoRA・RGLU-LoRAの実装、merge契約、CIFAR-10、ImageAE、
Qwen/TinyStories、TinyImageNet-200での比較を一旦完了しました。比較器、adapter専用entrypoint、
checkpoint/resume、epoch履歴、parameter-matched条件、VRAM・optimizer state・step時間の計測までを整備しました。

今回の実験から、次を確認しました。

- adapter方式の順位はrank、alpha、learning rate、学習期間、dtype、dataset条件に依存し、特定方式の一般的な優位性は確認できない。
- TinyImageNetのhead学習あり条件では、LoRAが初期〜中期の収束とepoch 10のlossで最良、DoRAがseed間lossの安定性で最小だった。RGLU-LoRAは中間的な軌跡だった。
- LoHA系は同rankならLoRAの約2倍のparameterを使うため、今回の主比較ではLoRA/DoRA=`rank32`、LoHA系=`rank16`のparameter-matched条件を採用した。
- weight-space gateを使うGLU-LoRA/RGLU-LoRAはmerge可能性を維持できたが、TinyImageNetのBF16長期条件では15件中9件のみmerge判定に成功した。既存merge toleranceは緩和しない。
- TinyImageNetの比較は小型モデル、train=`10000`、validation=`2000`、3 seedのscreeningであり、実運用上の性能優位性を示すものではない。

したがって、現時点ではadapterの既定方式・既定rank・既定alphaを変更しません。実験を再開する場合の候補は、
FP32 merge control、事前学習済みbase checkpointを使ったPEFT条件、seed数とdataset規模の拡大です。
これらは現在の作業項目ではなく、再開時に改めて条件を定義します。

LoRAはLinearの出力に次の差分を加えます。

```text
W' = W + (alpha / rank) * B @ A
A: (rank, in_features)
B: (out_features, rank)
```

`B`をゼロ初期化するため、注入直後の出力はbase Linearと一致します。base weightはfreezeし、`A`と`B`だけをoptimizerへ渡します。

DoRAは、LoRAの方向差分に出力行ごとのmagnitudeを加えます。

```text
W' = m[:, None] * (W + delta W) / ||W + delta W||_2
```

LoHAは2組の低rank行列のHadamard積を差分にします。

```text
delta W = (B1 @ A1) * (B2 @ A2)
```

GLU-LoRAはLoHAの一方をValue、もう一方をSiLU Gateとして解釈します。

```text
delta W = scale * ((B1 @ A1) ⊙ SiLU(B2 @ A2))
```

Residual GLU-LoRA（`rglu_lora`）はこのgateへ残差offsetを加えます。

```text
delta W = scale * ((B1 @ A1) ⊙ (1 + SiLU(B2 @ A2)))
```

両方式ともgateは入力activationではなく再構成したweight上で計算するため、LoHAと同じ
`2r * (d_in + d_out)`のtrainable parameter数でmergeできます。

## 設計・実装の補足

LoHAをValueとGateへ再解釈したGLU-LoRAとResidual GLU-LoRAを実装しました。GLU-LoRAは`ΔW = (B1 @ A1) ⊙ SiLU(B2 @ A2)`、Residual GLU-LoRAは`ΔW = (B1 @ A1) ⊙ (1 + SiLU(B2 @ A2))`です。数値契約と基本的なmerge等価性はunit testで確認済みです。入力activation上でgateを計算するのではなく、weight-spaceでdeltaを構成し、既存LoRAと同じmerge可能性を維持することを設計上の必須条件とします。性能・安定性・長期収束の一般的な優位性は確認できなかったため、現時点で既定方式は変更しません。

詳細な定義、identity初期化とLoRA warm startの選択、比較指標は[`rglu-lora.md`](rglu-lora.md)に記録します。

GLU-LoRA系では初期化を`--adapter-init identity|lora_warm`で選べます。
RGLU-LoRAの既定`identity`は`B1/B2=0`で注入直後のbase出力を保ちます。GLU-LoRAの
`identity`は`B2=0`かつ`B1`を非zeroにします。両方式とも注入直後のbase出力を保ちますが、
GLU-LoRAでは`B1=B2=0`にすると積の一次勾配まで消えるためです。`lora_warm`は`B2=0`
（gate=1）を維持しつつ`B1`を非zero初期化するため、初期更新はLoRAに似た非zero更新になります。
この初期化指定はGLU-LoRA系だけを対象とし、LoRA・LoHA・DoRAで`lora_warm`を指定するとエラーになります。

```bash
PYTHONPATH=. python3 -m cifar10.train_adapter \
  --base-init random \
  --adapter rglu_lora --lora-rank 4 \
  --adapter-init lora_warm
```

`cifar10/train_adapter.py`をadapter専用entrypointとして追加しました。既定の`--base-init checkpoint`ではadapterと`--init-checkpoint`または`--resume`を必須にし、`--base-init random`ではcheckpointなしで現在の初期化を使います。通常学習の`train.py`と同じ学習・checkpoint実装を再利用します。`train.py`はadapter引数を受け付けず、adapter checkpointのresumeも`train_adapter.py`へ誘導します。

checkpointを使わずに現在の`init_weights()`でbaseをrandom-initする場合は、`train_adapter.py --base-init random --adapter lora --lora-rank 4`のように指定します。この場合もbase weightはfreezeされ、adapter parameterだけが更新されます。random-initは標準PEFT checkpointの代替ではなく、adapter-only学習のcapacityを測る比較条件です。

## 検証方針

- 注入直後のforwardがbaseと一致すること
- merge/unmerge前後のforwardが一致すること
- base weightにgradientがなく、A/Bだけに有限gradientがあること
- checkpoint/resume後にadapter構成とoptimizer stateが一致すること
- trainable parameter数、checkpoint容量、optimizer state、peak memory、step時間、validation lossを分離して比較すること

GPU/Tritonの性能や数値等価性はCPUテストだけでは確定しないため、CUDA環境で別途測定します。


## 過去の比較結果

詳細な実験条件・集計結果・再実行例は[low-rank adapter experiment records](low-rank-adapters-results.md)に分離しました。
