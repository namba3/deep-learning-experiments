# ImageAE adapter experiment records

> Historical protocols and aggregate measurements. Local run artifacts are excluded from the public tree.

本文の「次」「今後」は実験記録時点の提案です。後続の結果がある場合は本文に記載しています。
結果が記載されていない案は、現在の作業予定や未実施状況を示すものではありません。

CIFAR-10の小型base checkpointを作る場合は、`--cifar10-train-samples`と`--cifar10-val-samples`で
使用件数を制限できます。`0`は全件です。

実ImageAEでも、window-transformer・latent=`8`・bottleneck=`64`・1 epoch・train 32 / validation 16の
小型条件で、base checkpoint生成、Residual GLU-LoRA adapter学習、epoch延長resume、merge exportを
実行しました。個別checkpointは公開せず、実adapterとmerged modelの出力最大絶対誤差`8.94e-7`のみを記録します。

同じbase checkpointから4方式を各1 epoch（seed=`0`、train 32、validation 16、batch=`16`）実行した短期結果は次の通りです。baseのvalidation MSEは`0.075754`でした。

| adapter | validation MSE | trainable params | optimizer state bytes |
| --- | ---: | ---: | ---: |
| LoRA | 0.075745 | 9,856 | 78,992 |
| LoHA | 0.075754 | 19,712 | 157,984 |
| DoRA | 0.075740 | 10,976 | 88,024 |
| Residual GLU-LoRA | 0.075746 | 19,712 | 157,984 |

4方式とも学習は完了しましたが、差は極めて小さく、データ数・epoch数ともに少ないため性能比較の結論には使いません。

同じ構成でbaseを5 epoch（train 512、validation 256）学習し、そのbest checkpointから各adapterを5 epoch
学習した追加結果も取得しました。seed=`0`、batch=`64`、rank=`4`、alpha=`4`、AdamW、FP32 CPUです。
baseのvalidation MSEは`0.055508`でした。

| adapter | validation MSE | loss delta from base | last-step秒 |
| --- | ---: | ---: | ---: |
| LoRA | 0.055297 | -0.000211 | 0.7513 |
| LoHA | 0.055495 | -0.000013 | 0.5878 |
| DoRA | 0.055224 | -0.000284 | 0.7409 |
| Residual GLU-LoRA | 0.055269 | -0.000239 | 0.7256 |

5 epochでもDoRAとLoRAが近く、Residual GLU-LoRAはLoHAより改善しました。ただし単一seed・小型モデル・
CPU実行であり、CUDA peak memoryや一般的な優位性は未検証です。個別run directoryとcheckpointは公開していません。

同じ5 epoch条件をseed=`0,1,2`へ拡張しました。baseはseed=`0`で学習した同一のbest checkpointに固定し、
adapter側のデータ順序と初期化乱数だけを変更しています。train 512枚、validation 256枚、batch=`64`、
rank=`4`、alpha=`4`、AdamW、FP32 CPUです。validation MSEとloss deltaは各runの最終step、step時間は
各runの最終stepの値です。

| adapter | validation MSE平均 | seed間std | loss delta平均 | last-step秒平均 | trainable params | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 0.055285 | 0.000013 | -0.000223 | 1.1190 | 9,856 | 78,992 |
| LoHA | 0.055494 | 0.000000 | -0.000014 | 0.5672 | 19,712 | 157,984 |
| DoRA | 0.055220 | 0.000006 | -0.000288 | 0.6755 | 10,976 | 88,024 |
| Residual GLU-LoRA | 0.055269 | 0.000035 | -0.000239 | 0.6588 | 19,712 | 157,984 |

3 seedでもDoRAの平均MSEが最小で、LoRAとResidual GLU-LoRAが近い結果でした。ResidualはLoHAを改善しましたが、
DoRAまたはLoRAを安定して上回る根拠は得られていません。LoHAのMSEはseed間のばらつきが小さい一方、平均値は他方式より
高くなりました。CPU実行のstep時間は実行順・キャッシュの影響を受けるため、速度比較には使わず、CUDA peak memoryも
引き続き未測定です。個別run directoryとcheckpointは公開していません。

`image_ae`の初期targetはwindow transformerのattention projectionとFFN出力（`ffn.3`）です。residual Conv FFN系はまだConv adapterの対象ではありません。

実ImageAE比較の再現コマンド例です。CIFAR-10を正規化せず`[0, 1]`のまま使い、MSE再構成lossを評価します。

```bash
PYTHONPATH=. python3 -m verify.image_ae_adapter_dataset_comparison \
  --data-dir cifar10/data --seeds 0,1,2 --epochs 3 \
  --max-train-samples 512 --max-validation-samples 256 \
  --output output/image-ae-adapter-dataset-comparison.json
```

これは短期の方式スクリーニング用であり、完全学習の品質比較や既定値変更の根拠にはしません。実施済みのCPU結果を下に記録しています。CUDA peak allocated/reservedは未測定です。

seed=`0,1,2`、3 epoch、train 512枚、validation 256枚、batch=`64`、rank=`4`、alpha=`4`、
AdamW、FP32 CPUで実行しました。個別JSONは公開せず、集計値を以下に記録します。モデルはlatent channels=`8`、bottleneck channels=`64`、encoder/decoderとも
window-transformer 1層、window size=`4`です。

| adapter | validation MSE平均 | loss delta平均 | step秒平均 | trainable params | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | 0.083955 | -0.000342 | 0.6980 | 9,856 | 78,992 |
| LoHA | 0.084283 | -0.000014 | 0.7276 | 19,712 | 157,984 |
| DoRA | 0.083863 | -0.000434 | 0.5033 | 10,976 | 88,024 |
| Residual GLU-LoRA | 0.084000 | -0.000297 | 0.5265 | 19,712 | 157,984 |

この条件ではDoRAのvalidation MSEが最小でした。Residual GLU-LoRAはLoHAより改善し、LoRAに近い値です。
一方、optimizer stateはLoRA/DoRAの約2倍で、Residualのgateは平均約1、seed間のstdは`5.1e-5`〜`8.6e-5`
でした。全12ケースでmerge等価性が成立し、最大誤差は`2.1e-6`未満です。CPUのstep時間は実行順や
キャッシュの影響を受けるため、性能の結論にはCUDA peak memoryと複数回の測定が必要です。

`image_ae`でもResidual GLU-LoRAの初期化を`--adapter-init identity|lora_warm`で選択できます。
ただし`image_ae`のadapter学習は、既存仕様どおり`--lora-base-checkpoint`またはadapter resume checkpointを
必要とし、CIFAR-10の`--base-init random`とは別契約です。

通常の`image_ae.train`はfull-model学習専用とし、adapter固有のCLI・model注入・optimizer parameter選択は
`image_ae.train_adapter`と`image_ae.adapter_training`へ分離しました。学習ループ、dataset、model構築、checkpoint
形式は共有しています。

学習checkpointは既存形式を使い、base weightとadapter parametersを同じsafetensorsへ保存します。
`cifar10.export_adapter`でadapterを通常のLinearへmergeした推論用checkpointを作成できます。
`image_ae.export_adapter`でも同じ処理を行えます。merged checkpointにはadapter moduleとresume用optimizer stateを含めません。
