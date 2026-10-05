# CIFAR-10 adapter experiment records

Residual GLU-LoRA sweepの正規CLIは`verify.cifar10_rglu_lora_sweep`です。
旧import pathの`verify.cifar10_residual_swiglu_sweep`は後方互換aliasとして残っています。

> Historical protocols and aggregate measurements. Local run artifacts are excluded from the public tree.

本文の「次」「今後」は実験記録時点の提案です。後続の結果がある場合は本文に記載しています。
結果が記載されていない案は、現在の作業予定や未実施状況を示すものではありません。

## CIFAR-10 preliminary comparisons

実CIFAR-10画像、seed=`0,1`、2 epoch、train 128枚、validation 64枚、rank=`4`、alpha=`4`、
FP32 CPUで予備比較を実行しました。モデルは比較用の小型`CIFAR10ViT`です。

| adapter | validation loss平均 | validation accuracy平均 | loss delta平均 | step秒平均 | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | 2.41536 | 7.03% | -0.01394 | 0.0563 | 7,720 |
| LoHA | 2.42843 | 7.03% | -0.00087 | 0.0567 | 15,440 |
| DoRA | 2.41209 | 7.81% | -0.01721 | 0.0462 | 8,764 |
| Residual GLU-LoRA | 2.41422 | 7.03% | -0.01508 | 0.0562 | 15,440 |

この規模ではDoRAとResidual GLU-LoRAのloss改善が観測されましたが、seed数・sample数・epoch数が
少なく、方式の優位性を示す結果ではありません。全方式でmerge最大絶対誤差は`1.1e-6`未満でした。

追加でseed=`0,1,2`、5 epoch、train 1024枚、validation 512枚へ拡大しました。rank、alpha、
optimizer、モデル構成は同じで、FP32 CPUの比較です。

| adapter | validation loss平均 | seed間std | validation accuracy平均 | loss delta平均 | step秒平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | 2.31641 | 0.00053 | 9.96% | -0.08559 | 0.1329 |
| LoHA | 2.34770 | 0.01090 | 9.64% | -0.05430 | 0.1479 |
| DoRA | 2.31459 | 0.00185 | 10.42% | -0.08741 | 0.1633 |
| Residual GLU-LoRA | 2.31751 | 0.00491 | 9.38% | -0.08449 | 0.1458 |

この拡大条件ではDoRAとLoRAが近く、Residual GLU-LoRAはLoHAより改善したものの、
DoRA/LoRAを上回る結果ではありません。モデルは比較用の小型構成で、CPU実験でもあるため、
この結果を一般的な優位性や既定adapterの根拠にはしません。全12ケースでmerge最大絶対誤差は`1.1e-6`未満でした。

同じ初期化比較の基準を揃えるため、LoRA・LoHA・DoRA・Residual GLU-LoRAを、
rank=`4`・alpha=`4`・AdamW・FP32 CPU・seed=`0,1,2`・3 epoch・train 512枚・validation 256枚で
比較しました。集計値は以下の通りです。

| adapter | validation loss平均 | validation accuracy平均 | loss delta平均 | step秒平均 | trainable params | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 2.33881 | 8.46% | -0.07245 | 0.1350 | 960 | 7,720 |
| LoHA | 2.39767 | 8.07% | -0.01358 | 0.1353 | 1,920 | 15,440 |
| DoRA | 2.33617 | 8.72% | -0.07509 | 0.1349 | 1,088 | 8,764 |
| Residual GLU-LoRA | 2.33764 | 9.51% | -0.07362 | 0.1414 | 1,920 | 15,440 |

今回の短期条件ではDoRA・LoRA・Residualのvalidation lossは近く、Residualはaccuracyがやや高い一方で
step時間も約5%増えました。seed数・epoch数・モデル規模が限定されているため、優位性や既定値変更の
根拠にはしません。全12ケースでmerge等価性が成立し、最大誤差は`1.3e-6`未満でした。

### Residual GLU-LoRAの初期化予備比較

identity startとLoRA warm startを、同じResidual GLU-LoRA・rank=`4`・alpha=`4`・
AdamW・FP32 CPU・seed=`0,1,2`・3 epoch・train 512枚・validation 256枚で比較しました。
個別JSONは公開せず、初期化別の集計値を以下に示します。

| 初期化 | validation loss平均 | validation accuracy平均 | loss delta平均 | step秒平均 | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 2.33764 | 9.51% | -0.07362 | 0.1023 | 15,440 |
| lora_warm | 2.33122 | 9.11% | -0.06210 | 0.1316 | 15,440 |

warm startは非zero更新を持つため、注入直後のbase一致を保証しません。固定入力のseed=`0`・5 step
probeでは初期gradient normがidentity=`0.454`、warm=`2.546`となり、warmの方が大きい初期勾配を
示しました。今回の短期実データ比較ではwarmの安定した優位性は確認できず、step時間も約29%増えたため、
既定値はidentityのまま維持します。これは小型CPU・短期条件の結果であり、長期収束やGPUでの結論ではありません。


## 追加probeと実験結果

Residualのrank×alpha測定は次のコマンドで再実行できます。`--alphas`を省略するとalpha=rank、
指定すると全rankとの直積になります。結果にはoptimizer state、step時間、validation loss、
merge誤差、gateの平均・標準偏差・min/max・分位点が含まれます。FP32のmerge判定は`atol=1e-5, rtol=1e-5`、
BF16は経路ごとの丸め順序を考慮して`atol=3e-2, rtol=1e-2`です。

```bash
PYTHONPATH=. python3 -m verify.cifar10_rglu_lora_sweep \
  --data-dir cifar10/data --device cpu --seeds 0,1,2 \
  --ranks 1,2,4,8 --alphas 1,4,8 \
  --epochs 2 --output output/rglu-lora-sweep.json
```

GPU/BF16のrank sweep（rank=`1,4,8`、3 seed、3 epoch、train/validation=`512/256`）も実行しました。
個別run JSONは公開せず、集計値を以下に示します。
validation loss平均はrank=`1,4,8`でそれぞれ`2.3482`、`2.3320`、`2.3284`、gateの標準偏差平均は
`0.00182`、`0.00439`、`0.00631`でした。peak allocated/reservedは全9ケースで`20,126,208 / 25,165,824`
bytesでした。rank増加による短期loss改善とgate分散の増加は観測できましたが、単一条件のため一般的な優位性の根拠にはしません。

修正後の検証器で再実行した結果、全9ケースで`merge_equivalent=true`となりました。merge最大誤差は
`0.0150`〜`0.0176`でしたが、BF16のwrapped/merged経路の丸め順序を考慮した`atol=3e-2, rtol=1e-2`
の範囲内です。したがって、今回のRGLU-LoRAのweight-space merge契約はこのBF16条件でも成立しています。

同じ測定は[`benchmarks/launchers/run_cifar10_rglu_lora_sweep.sh`](../../benchmarks/launchers/run_cifar10_rglu_lora_sweep.sh)から再実行できます。
`CIFAR10_RGLU_LORA_ALPHAS=1,4,8`を指定すると、rankとalphaの直積を測定します。

rank=`1,4,8`、alpha=`1,4,8`のGPU/BF16直積sweepも3 seed・3 epochで実行しました。

| rank | alpha | validation loss平均 | loss標準偏差 | gate std平均 | state bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2.34816 | 0.01623 | 0.001824 | 2,000 |
| 1 | 4 | 2.33013 | 0.00569 | 0.001239 | 2,000 |
| 1 | 8 | **2.32348** | 0.00121 | 0.001076 | 2,000 |
| 4 | 1 | 2.34758 | 0.01492 | 0.005620 | 7,760 |
| 4 | 4 | 2.33197 | 0.00574 | 0.004390 | 7,760 |
| 4 | 8 | 2.32841 | 0.00401 | 0.003895 | 7,760 |
| 8 | 1 | 2.34726 | 0.01355 | 0.008191 | 15,440 |
| 8 | 4 | 2.33231 | 0.00492 | 0.006946 | 15,440 |
| 8 | 8 | 2.32839 | 0.00248 | 0.006310 | 15,440 |

この条件ではalpha増加によるloss改善がrank変更より明瞭で、rank=1・alpha=8が最小lossでした。
一方、validation accuracyはseed間の揺れが大きく、peak VRAMは全条件で同一でした。全27ケースのmerge判定も成功しています。
alpha=8を含むLoRA/LoHA/DoRAとのparameter-matched比較は後続の比較で実施し、その結果を次節に記録しています。

CIFAR-10のparameter-matched GPU/BF16比較（3 seed・3 epoch・train/validation=`512/256`）を実行しました。

| adapter | rank/alpha | validation loss平均 | loss標準偏差 | params | optimizer state | step時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 16/16 | **2.32700** | 0.00636 | 3,840 | 15,400 | 12.18 ms |
| DoRA | 16/16 | 2.32796 | 0.00660 | 3,968 | 15,932 | 14.72 ms |
| RGLU-LoRA | 8/8 | 2.32839 | 0.00248 | 3,840 | 15,440 | 14.29 ms |
| LoHA | 8/8 | 2.35150 | 0.01456 | 3,840 | 15,440 | 10.98 ms |

LoRAが平均lossで最小、DoRAとRGLU-LoRAが僅差で続き、LoHAは高いlossでした。RGLU-LoRAはLoHAを改善し、
seed間のloss標準偏差は最小でしたが、今回の短期条件ではLoRAを上回りませんでした。peak allocatedは
LoRA=`20,374,528`、DoRA=`20,128,768`、RGLU-LoRA=`20,126,208`、LoHA=`20,121,088` bytes、
reservedは全方式で`25,165,824` bytesでした。全12ケースでmerge判定に成功しています。

CIFAR-10のparameter-matched比較は、[`benchmarks/launchers/run_cifar10_adapter_budget_comparison.sh`](../../benchmarks/launchers/run_cifar10_adapter_budget_comparison.sh)
で実行できます。既定ではLoRA/DoRAをrank=`16`、LoHA/RGLU-LoRAをrank=`8`、alpha/rank=`1`として、
同一seed・subset・optimizer条件で比較します。方式ごとの設定は`--rank-map` / `--alpha-map`でも指定できます。
LoRA=`32`・LoHA=`16`（DoRA=`32`・RGLU-LoRA=`16`）の高budget条件は
`CIFAR10_ADAPTER_BUDGET=rank32`で選択します。

高budgetのGPU/BF16比較（3 seed・3 epoch・train/validation=`512/256`）も実行しました。4方式と、GLU-LoRAを追加した5方式の集計値を以下に示します。

| adapter | rank/alpha | validation loss平均 | loss標準偏差 | params | optimizer state | step時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 32/32 | **2.31893** | 0.01321 | 7,680 | 30,760 | 16.86 ms |
| RGLU-LoRA | 16/16 | 2.32407 | **0.00037** | 7,680 | 30,800 | 17.10 ms |
| DoRA | 32/32 | 2.32422 | 0.00778 | 7,808 | 31,292 | 19.50 ms |
| LoHA | 16/16 | 2.33711 | 0.00937 | 7,680 | 30,800 | 17.34 ms |

GLU-LoRA追加runの結果は次の通りです。既存4方式の値も同一条件で再現され、GLU-LoRAは
validation loss平均=`2.34761`、標準偏差=`0.01362`、step時間=`14.76 ms`でした。
parameter数はLoHA/RGLU-LoRAと同じ`7,680`、optimizer stateは`30,800` bytesです。
全15ケースで`merge_equivalent=true`となり、peak reservedは全方式で`25,165,824` bytesでした。

| adapter | rank/alpha | validation loss平均 | loss標準偏差 | params | optimizer state | step時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 32/32 | **2.31893** | 0.01079 | 7,680 | 30,760 | 14.48 ms |
| DoRA | 32/32 | 2.32422 | 0.00635 | 7,808 | 31,292 | 16.05 ms |
| RGLU-LoRA | 16/16 | 2.32406 | **0.00030** | 7,680 | 30,800 | 15.79 ms |
| LoHA | 16/16 | 2.33711 | 0.00765 | 7,680 | 30,800 | 15.83 ms |
| GLU-LoRA | 16/16 | 2.34761 | 0.01362 | 7,680 | 30,800 | **14.76 ms** |

この条件ではGLU-LoRAはRGLU-LoRAよりvalidation lossが高く、LoHAも下回りました。GLU-LoRAの
offsetなしgateは計算上単純ですが、今回のidentity初期化・3 epoch条件ではRGLU-LoRAの残差経路を
上回る結果は得られていません。step時間は最短でしたが、短期・単一条件のため速度差を一般化しません。

rank16/8条件からのloss改善はLoRAが最も大きく、RGLU-LoRAも`2.32839`から`2.32407`へ改善しました。
RGLU-LoRAは高budgetでもseed間のばらつきが最小でしたが、平均lossではLoRAを上回りませんでした。全12ケースで
merge判定に成功し、reserved VRAMは全方式で`25,165,824` bytesでした。この測定時点では、rank増加による品質改善と
state/計算量の増加を分離するためepochまたはoptimizer step数を延長する案を記録しました。後続の10 epoch比較を下に記録しています。

高budget条件のlearning rate sweepは[`benchmarks/launchers/run_cifar10_adapter_lr_sweep.sh`](../../benchmarks/launchers/run_cifar10_adapter_lr_sweep.sh)
で実行できます。既定では`3e-4,1e-3,3e-3`を同じrank32/16・seed・subset条件で比較し、
`output/cifar10-adapter-lr-sweep/lr-*.json`へ保存します。

実際に5方式・3 seed・3 epochで実行した結果、全45ケースが完了し、全ケースでmerge判定に成功しました。
各方式のlearning rate別の平均validation lossは次の通りです。

| learning rate | LoRA | LoHA | DoRA | GLU-LoRA | RGLU-LoRA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `3e-4` | 2.33461 | 2.39953 | 2.33512 | 2.40518 | 2.34337 |
| `1e-3` | 2.31893 | 2.33711 | 2.32422 | 2.34761 | 2.32406 |
| `3e-3` | 2.31092 | 2.31986 | 2.31837 | 2.32201 | **2.30950** |

`1e-3`ではLoRAが最良でしたが、`3e-3`ではRGLU-LoRAが最良となり、learning rate依存で順位が変わりました。
LoRAは全learning rateで安定して上位、GLU-LoRAは全条件でRGLU-LoRAを下回りました。`3e-3`ではRGLU-LoRAと
LoRAの差は小さく、3 seed・3 epochの短期比較だけでlearning rateの既定値を変更する根拠にはしません。
optimizer stateは方式ごとに一定で、LoRA=`30,760`、LoHA/GLU-LoRA/RGLU-LoRA=`30,800`、DoRA=`31,292` bytes、
peak reservedは全条件で`25,165,824` bytesでした。個別run出力は公開アーカイブに含めていません。

同じrank32/16の条件を10 epochへ延長した結果の集計値を以下に示します。

| adapter | validation loss平均 | loss標準偏差 | validation accuracy平均 | step時間 |
| --- | ---: | ---: | ---: | ---: |
| LoRA | **2.31258** | 0.06045 | 11.98% | 13.31 ms |
| RGLU-LoRA | 2.31824 | 0.02213 | 11.85% | 15.56 ms |
| DoRA | 2.31912 | 0.04314 | 10.68% | 14.95 ms |
| LoHA | 2.32007 | **0.00365** | 10.42% | 14.45 ms |

10 epochではLoRAが平均lossで最良、RGLU-LoRAが2番手でした。LoRAとDoRAはseed間のばらつきが大きく、
RGLU-LoRAも3 epoch時の標準偏差`0.00037`から増加したため、短期条件だけでは安定性の優位性を確定できません。
全方式のreserved VRAMは`25,165,824` bytesで、rank32/16への増加による増分は主にoptimizer stateとstep時間に現れました。
修正版のBF16 merge閾値で再実行した結果、全12ケースで`merge_equivalent=true`となりました。最大誤差は`0.0254`で、
現在の`atol=3e-2, rtol=1e-2`以内です。

learning rateの長期確認として、同じrank32/16・3 seed・train/validation=`512/256`条件で`1e-3`と`3e-3`を
10 epochへ延長しました。集計値は以下に示します。

| learning rate | LoRA | LoHA | DoRA | GLU-LoRA | RGLU-LoRA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | **2.31258** | 2.32007 | 2.31912 | 2.32285 | 2.31824 |
| `3e-3` | 2.31277 | **2.30076** | 2.31725 | 2.30576 | 2.30475 |

`1e-3`ではLoRAが最良のままでしたが、`3e-3`では10 epoch時点でLoHAが最良となり、RGLU-LoRAは
2番手ではなくGLU-LoRAに近い3番手でした。したがって、短期3 epochで観測したRGLU-LoRAの優位性は
長期収束で再現せず、方式の順位はlearning rateと学習期間の両方に依存します。全30ケースでmerge判定に成功し、
peak reservedは`25,165,824` bytesでした。今回も既定LRやadapter方式の変更は保留します。

さらに同じ条件を20 epochへ延長しました。集計値は以下に示します。

| learning rate | LoRA | LoHA | DoRA | GLU-LoRA | RGLU-LoRA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | 2.33703 | **2.31488** | 2.32234 | 2.31892 | 2.31598 |
| `3e-3` | 2.35385 | 2.32691 | 2.36480 | **2.29817** | 2.31674 |

`1e-3`では10 epoch時点のLoRA優位が20 epochでLoHA優位へ変わり、`3e-3`では10 epoch時点のLoHA優位が
20 epochでGLU-LoRA優位へ変わりました。したがって、この小規模subsetではadapterの順位が学習期間にも強く依存します。
また、`1e-3`の15ケースはmerge判定に成功しましたが、`3e-3`の15ケースは失敗しました。`3e-3`のmerge最大誤差は
`0.0479`で、現行BF16閾値`atol=3e-2, rtol=1e-2`を超えています。閾値を緩和して成功扱いにはせず、
後続のFP32比較でdtypeの影響を切り分けました。品質順位とmerge安全性は分けて記録しています。

同じ`3e-3`・20 epoch条件をFP32でも実行し、dtypeの影響を切り分けました。集計値は以下に示します。

| adapter | validation loss平均 | loss標準偏差 | validation accuracy平均 | 最大merge誤差 |
| --- | ---: | ---: | ---: | ---: |
| LoHA | **2.31059** | 0.05280 | 13.93% | 1.8e-6 |
| DoRA | 2.36135 | 0.05151 | 14.32% | 1.6e-6 |
| GLU-LoRA | 2.36162 | 0.06148 | 10.68% | 1.8e-6 |
| LoRA | 2.36858 | 0.03268 | 14.97% | 2.1e-6 |
| RGLU-LoRA | 2.38920 | 0.08999 | 12.11% | 1.5e-6 |

FP32では全15ケースがmerge成功し、merge誤差は`1e-6`台でした。このため、BF16のmerge失敗は
高LRで学習された重みをBF16のwrapped/merged経路で評価する際の丸め差に起因すると判断できます。
一方、validation lossもBF16とは異なるため、これはmerge誤差だけの比較ではなく、BF16/FP32で学習挙動自体が
変わることを示します。品質比較は同一dtype内で行い、BF16のmerge安全性は別の受入条件として扱います。

短期screeningとして、alpha=rankのrank sweep（3 seed・1 epoch・train/validation=256/128枚）を
実行しました。集計値は以下の通りです。

| rank | validation loss平均 | loss delta平均 | step秒平均 | trainable params | optimizer state bytes | gate std平均 | gate min--max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.38784 | -0.00231 | 0.1561 | 480 | 3,920 | 0.000221 | 0.99917--1.00071 |
| 2 | 2.38349 | -0.00666 | 0.2018 | 960 | 7,760 | 0.000450 | 0.99872--1.00150 |
| 4 | 2.37987 | -0.01028 | 0.2084 | 1,920 | 15,440 | 0.000557 | 0.99848--1.00196 |
| 8 | 2.37105 | -0.01910 | 0.1289 | 3,840 | 30,800 | 0.001087 | 0.99673--1.00444 |

rank増加に伴いtrainable parameterとoptimizer stateはほぼ線形に増え、gate分布のばらつきも広がりました。
この条件ではrank=8のloss deltaが最大でしたが、1 epochのCPU結果であり、step時間にも測定揺れがあるため、
rank=4を既定候補から外す根拠にはしません。

rank=4でalpha=`1,4,8`を比較しました。集計値は以下の通りです。

| alpha | validation loss平均 | loss delta平均 | gate std平均 | gate min--max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.38743 | -0.00272 | 0.000545 | 0.99850--1.00194 |
| 4 | 2.37987 | -0.01028 | 0.000557 | 0.99848--1.00196 |
| 8 | 2.37108 | -0.01907 | 0.000558 | 0.99849--1.00196 |

alpha増加で短期loss deltaは大きくなりましたが、gate分布自体はほぼ変わりませんでした。
peak allocated/reservedはCUDAが利用できないため未測定です。

rankの短期傾向を確認するため、同じseed・subset条件で5 epochへ延長しました。rank=`1,4,8`、
alpha=rank、seed=`0,1,2`、train/validation=512/256枚のFP32 CPU実験です。集計値は以下の通りです。

| rank | validation loss平均 | validation accuracy平均 | loss delta平均 | step秒平均 | state bytes | gate std平均 | gate min--max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.34633 | 8.33% | -0.06493 | 0.1174 | 3,920 | 0.001627 | 0.99388--1.00623 |
| 4 | 2.32497 | 8.20% | -0.08629 | 0.1145 | 15,440 | 0.004621 | 0.98368--1.01845 |
| 8 | 2.32752 | 8.98% | -0.08373 | 0.1620 | 30,800 | 0.008349 | 0.98249--1.05061 |

5 epochでもrank=4がvalidation lossで最良でした。rank=8はaccuracyがやや高いものの、stateはrank=4の
2倍、step時間は約42%増で、loss差も小さいという結果でした。gate飽和や外れ値を含む長期測定は当時の未実施案として記録したもので、
この文書から現在の実施状況は確認できません。

alphaの長期寄り比較として、rank=4・alpha=`1,4,8`・seed=`0,1,2`・5 epoch・
train/validation=512/256枚でも測定しました。集計値は以下の通りです。

| alpha | validation loss平均 | validation accuracy平均 | loss delta平均 | state bytes | gate std平均 | gate min--max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.34427 | 8.85% | -0.06699 | 15,440 | 0.005696 | 0.97837--1.02872 |
| 4 | 2.32497 | 8.20% | -0.08629 | 15,440 | 0.004621 | 0.98368--1.01845 |
| 8 | 2.32114 | 9.77% | -0.09012 | 15,440 | 0.004128 | 0.98590--1.01523 |

この条件ではalpha=8が最良でしたが、alpha=1のstep時間にCPU由来の外れ値があり、速度差は評価しません。
alpha増加によるvalidation改善も小型subset・5 epochの傾向に留まるため、既定alphaはrank相当のまま維持します。
