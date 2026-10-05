# TinyImageNet-200 adapter experiment records

> Historical protocols and aggregate measurements. Local run artifacts are excluded from the public tree.

本文の「次」「今後」は実験記録時点の提案です。後続の結果がある場合は本文に記載しています。
結果が記載されていない案は、現在の作業予定や未実施状況を示すものではありません。

## TinyImageNet-200 comparison entry

CIFAR10よりクラス数と画像内容の多様性が大きい画像分類条件として、TinyImageNet-200用の比較器も追加しました。
Hugging Face datasetsの`zh-plus/tiny-imagenet`をデフォルトキャッシュから読み込み、64x64・200クラスの小型ViT headで
5方式を比較します。CIFAR10は数値・merge・VRAMのsmoke、TinyImageNetはadapterの分類性能を確認する用途として
分離します。実行入口は[`benchmarks/launchers/run_tiny_imagenet_adapter_budget_comparison.sh`](../../benchmarks/launchers/run_tiny_imagenet_adapter_budget_comparison.sh)、
詳細は[`verify/README.md`](../../verify/README.md)に記載します。必要な場合のみcache directoryを明示指定します。

rank32 budget（LoRA/DoRA=`32`、LoHA/GLU-LoRA/RGLU-LoRA=`16`）をBF16・3 epoch・
train/validation=`1024/512`・3 seedで実行しました。全15ケースが完了し、全方式でmerge判定にも成功しました。
個別run JSONは公開せず、集計値のみを以下に記録しています。

| adapter | rank | trainable parameters | validation loss平均 | accuracy平均 | step秒平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DoRA | 32 | 7,808 | **5.29714** | 0.91% | 0.0223 |
| LoRA | 32 | 7,680 | 5.29715 | 0.91% | 0.0230 |
| RGLU-LoRA | 16 | 7,680 | 5.29956 | 0.59% | 0.0229 |
| LoHA | 16 | 7,680 | 5.30197 | 0.78% | 0.0218 |
| GLU-LoRA | 16 | 7,680 | 5.30314 | 0.59% | 0.0222 |

200クラスの一様分類lossは`log(200) ≈ 5.30`であり、今回のaccuracyもランダム水準に近いです。
したがって、現結果は実装・merge・VRAM計測のsmokeとしては有効ですが、adapterの性能順位を示すものではありません。
加えて、この比較器はadapter以外をfreezeするため、ランダム初期化された分類headもfreezeされています。
この条件は性能順位の評価には適しません。共通headも学習する条件での後続runを次に記録しています。

その長期条件には[`benchmarks/launchers/run_tiny_imagenet_adapter_long_comparison.sh`](../../benchmarks/launchers/run_tiny_imagenet_adapter_long_comparison.sh)を使います。
既定値はtrain=`10000`、validation=`2000`、10 epoch、3 seedで、adapterに加えて共通classifier headを学習します。

この条件をCUDA/BF16で実行しました。個別run JSONは公開せず、集計値のみを以下に記録しています。
全15ケース、3,130 steps/caseが完了し、classifier headを学習しない前回の方式平均loss約5.289に対して、
head学習ありでは方式平均lossが約5.086まで下がり、accuracy平均は2.25%（200クラスのランダム水準約0.5%）になりました。

| adapter | rank | trainable parameters | validation loss平均 | accuracy平均 | merge成功 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | 32 | 16,160 | **5.05995** | 2.55% | 2/3 |
| DoRA | 32 | 16,288 | 5.07051 | 2.35% | 3/3 |
| RGLU-LoRA | 16 | 16,160 | 5.07749 | 2.37% | 1/3 |
| LoHA | 16 | 16,160 | 5.10296 | 2.05% | 1/3 |
| GLU-LoRA | 16 | 16,160 | 5.11805 | 1.95% | 2/3 |

head学習によってランダム水準からの改善は確認できましたが、まだ短いscreening条件です。
また、BF16 mergeは9/15ケースに留まり、最大merge誤差は`0.046875`でした。既存のmerge toleranceは緩和せず、
FP32 controlでwrapped/merged経路のdtype依存差を切り分けます。

初期〜中期の安定性を比較できるよう、比較器は各caseの`epoch_metrics`にepochごとのtrain loss平均/std、
validation loss/accuracy、初期値からのloss delta、step時間を保存しています。今回の履歴では、最終lossだけでなく、
epoch 1〜3の初期収束、epoch 5の中期挙動、epoch 10のseed間ばらつきを比較できます。

| adapter | validation loss E1 | E3 | E5 | E10 | E10 seed std | E10 batch loss std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 5.28072 | **5.17602** | **5.13082** | **5.05128** | 0.01257 | 0.13360 |
| DoRA | **5.28026** | 5.17999 | 5.13610 | 5.06930 | **0.00597** | 0.12810 |
| RGLU-LoRA | 5.28310 | 5.19228 | 5.14720 | 5.07835 | 0.02038 | 0.12529 |
| LoHA | 5.28924 | 5.21026 | 5.16430 | 5.09997 | 0.01736 | 0.11780 |
| GLU-LoRA | 5.29030 | 5.21972 | 5.17444 | 5.11966 | 0.01877 | **0.11384** |

LoRAはepoch 1〜5からlossが最も速く下がり、epoch 10でも最良でした。DoRAは初期lossが最小で、
epoch 10のseed間標準偏差も最小です。RGLU-LoRAはLoHA/GLU-LoRAより速く、LoRA/DoRAより遅い中間的な軌跡でした。
一方、batch loss stdはGLU-LoRA/LoHAが小さいため、揺れの小ささと収束速度は同じ指標として扱わない方がよいです。
