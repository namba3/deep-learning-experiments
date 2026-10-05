# Text-LM adapter experiment records

> Historical protocols and aggregate measurements. Local run artifacts are excluded from the public tree.

本文の「次」「今後」は実験記録時点の提案です。後続の結果がある場合は本文に記載しています。
結果が記載されていない案は、現在の作業予定や未実施状況を示すものではありません。

テキスト生成モデルの比較には、リポジトリ内の`text_lm`を使います。`text_lm`はdecoder-only Transformerで
causal next-token lossを使うため、画像再構成MSEよりadapterの表現力差を評価しやすい構成です。最初は
独立weightの`--architecture naive`に限定し、attention projectionとFFN出力を対象にします。

```bash
PYTHONPATH=. python3 -m text_lm.train_adapter \
  --lora-base-checkpoint output/text-lm-base.safetensors \
  --architecture naive \
  --adapter rglu_lora \
  --lora-rank 4 --lora-alpha 4
```

`shared-fixed`とlooped系はsuper weightまたは物理block共有を持つため、通常Linear adapterとの比較に混ぜず、
別のparameterization実験として扱います。TinyStoriesの実測比較は、同一base checkpoint・同一tokenized subset・
同一seedでvalidation loss/perplexity、trainable parameter、optimizer state、step時間、peak memory、merge誤差を
記録する方針です。

同一baseから4方式をseed=`0,1,2`でTinyStories上で比較しました。Qwen tokenizer
`Qwen/Qwen3.5-0.8B`、`roneneldan/TinyStories`、固定長token block、`naive`、dim=`128`、2層、4 heads、
max sequence length=`64`、batch=`4`、train=`256` examples、eval=`64` examples、2 epoch×20 steps、
AdamW、lr=`3e-4`、rank=`4`、alpha=`4`、FP32 CPUです。token budgetはtrain/eval=`16,384/4,096`、
dataset seedは`0`で固定し、adapter側のseedだけを変更しました。baseの最終validation lossは`64.520735`でした。

| adapter | validation loss平均 | seed間std | baseからの差 | trainable params | optimizer state bytes | 最終step秒平均±std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 59.745078 | 0.255033 | -4.775657 | 12,288 | 98,304 | 0.7609 ± 0.3206 |
| LoHA | 64.388882 | 0.002188 | -0.131853 | 24,576 | 196,608 | 0.9253 ± 0.3747 |
| DoRA | **58.426543** | 0.254999 | **-6.094192** | 13,568 | 108,544 | 0.8820 ± 0.3303 |
| Residual GLU-LoRA | 59.672872 | 0.341717 | -4.847863 | 24,576 | 196,608 | 0.9993 ± 0.4055 |

この短期条件ではDoRAが最小loss、次いでLoRAとResidual GLU-LoRAが近く、LoHAの改善は小さい結果でした。
ただし2 epoch・小型subset・小型モデルのscreeningであり、方式の一般的な優位性や既定値変更の根拠にはしません。
perplexityはこのlossの指数変換で定義できますが、未学習に近いlossのため非常に大きく、比較の主指標はvalidation lossとします。

全12ケースを[`text_lm.export_adapter`](../../text_lm/export_adapter.py)でmergeし、通常モデルへstrict loadできることを確認しました。
merged checkpointのadapterキーは残っていません。wrapped modelとの固定入力の最大出力差は`3.8e-5`でした。
これはadapterの意味的な不一致ではなく、分離したbase出力とdelta出力を加算する経路と、merge後の単一Linear経路の
FP32丸め差です。個別checkpointとrun directoryは公開していません。CUDAが利用できなかったため、peak allocated/reservedとGPU step時間は未測定です。

### 長系列・長期収束の比較

64 tokenでの比較を長系列へ拡張するため、[`run_text_lm_adapter_sequence_comparison.sh`](../../benchmarks/launchers/run_text_lm_adapter_sequence_comparison.sh)
を追加しました。`max_seq_len=512,1024,2048`を切り替え、既定の`token_normalized`モードでは
`tokens_per_step=2048`を固定することで、実効batchをそれぞれ`4,2,1`にします。train/eval token budget、epoch数、
steps/epoch、seed、adapter種別は環境変数で変更できます。集計結果は`report.json`へ保存され、validation loss、PPL、
step時間、optimizer state、CUDA peak allocated/reservedをcontext length・adapter・実効tokens/stepごとに分けて記録します。

固定token batch以外の挙動も見る場合は、`fixed_batch`モードでbatch sizeを固定します。この場合、系列長が長いほど
1 optimizer stepのtoken数が増えるため、品質を単純にtoken budgetだけで比較せず、固定step数での最適化挙動として解釈します。
両モードの出力を同じディレクトリへ保存しても、レポートは`tokens_per_step`を別グループとして扱います。

Qwen tokenizerは語彙数が大きく、全系列のlogitsを一度に作るとVRAMを圧迫します。そのため`text_lm.train`は
既定で`--vocab-chunk-size=8192`を使い、語彙方向を分割したFP32 causal cross entropyを計算します。
この経路はtied embeddingへの勾配を含み、全logits経路とのforward/backward一致をテストしています。
`TEXT_LM_ADAPTER_VOCAB_CHUNK_SIZE`でchunk幅を変更でき、0以下を指定すると従来の全logits経路へ戻せます。

GPUでの長期比較例:

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TEXT_LM_ADAPTER_BASE_CHECKPOINT=output/text-lm-base-2048.safetensors \
TEXT_LM_ADAPTER_DEVICE=cuda TEXT_LM_ADAPTER_BF16=1 \
TEXT_LM_ADAPTER_MAX_SEQ_LENS=512,1024,2048 \
TEXT_LM_ADAPTER_TOKENS_PER_STEP=2048 \
TEXT_LM_ADAPTER_TRAIN_TOKENS=1048576 \
TEXT_LM_ADAPTER_EVAL_TOKENS=131072 \
TEXT_LM_ADAPTER_EPOCHS=3 TEXT_LM_ADAPTER_STEPS_PER_EPOCH=200 \
bash benchmarks/launchers/run_text_lm_adapter_sequence_comparison.sh
```

上記のtoken-normalized条件をGPUで実行し、全36ケース（4 adapter × 3 context × 3 seed）が完了しました。
Qwen tokenizer、TinyStories、`naive`、dim=`128`、2層、BF16、rank=`4`、alpha=`4`、3 epoch × 200 steps、
train/eval token budget=`1,048,576/131,072`、tokens/step=`2,048`です。個別run reportは公開せず、集計値を以下に示します。

| adapter | max seq | validation loss平均 | seed std | 最終step秒平均 | peak allocated MiB平均 | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 512 | 30.341973 | 0.083352 | 0.1536 | 507.2 | 98,304 |
| LoRA | 1024 | 30.242207 | 0.054352 | 0.1436 | 515.5 | 98,304 |
| LoRA | 2048 | 30.261696 | 0.070320 | 0.1497 | 532.9 | 98,304 |
| LoHA | 512 | 38.343089 | 0.230499 | 0.1536 | 508.4 | 196,608 |
| LoHA | 1024 | 36.733320 | 0.655844 | 0.1516 | 516.7 | 196,608 |
| LoHA | 2048 | 35.616650 | 0.557072 | 0.1528 | 534.0 | 196,608 |
| DoRA | 512 | 30.618015 | 0.087690 | 0.1520 | 499.6 | 108,544 |
| DoRA | 1024 | 30.485576 | 0.056937 | 0.1509 | 504.7 | 108,544 |
| DoRA | 2048 | 30.465097 | 0.046829 | 0.1492 | 512.6 | 108,544 |
| Residual GLU-LoRA | 512 | 30.242755 | 0.053185 | 0.1472 | 508.8 | 196,608 |
| Residual GLU-LoRA | 1024 | 30.127067 | 0.048232 | 0.1495 | 517.1 | 196,608 |
| Residual GLU-LoRA | 2048 | 30.168303 | 0.030464 | 0.1495 | 534.5 | 196,608 |

この条件ではResidual GLU-LoRAが3系列長で最小lossまたはLoRAと同等、LoHAは一貫して高い結果でした。
ただし同じbase・同じtoken budgetの3 seedによる一条件の結果であり、方式の一般的な優位性とは解釈しません。
peak allocatedは系列長とともに増加しましたが、chunked vocabulary lossにより2048 tokenでも学習は完了しました。
固定batch条件の結果は次節に分けて記録します。adapter merge後の出力一致は、parameter-matched条件でCPU/CUDAの両方を検証済みです。

### Fixed-batch比較

batch=`1`、rank=`4`、alpha=`4`を固定し、token-normalized条件と同じ3 seed・3 epoch×200 stepsで比較しました。
したがってtokens/stepは系列長に応じて`512,1024,2048`となり、同じstep数でも系列長が長いほど総処理token数が増えます。
この結果はtoken budgetを揃えた品質比較ではなく、固定optimizer step数での収束挙動として解釈します。全36ケースが完了し、集計値は以下の通りです。

| adapter | max seq | validation loss平均 | seed std | tokens/step | 最終step秒平均 | peak allocated MiB平均 | peak reserved MiB平均 | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 512 | 30.931968 | 0.081749 | 512 | 0.0757 | 283.2 | 360 | 98,304 |
| LoRA | 1024 | 30.300535 | 0.060591 | 1024 | 0.1014 | 359.8 | 462 | 98,304 |
| LoRA | 2048 | 30.261696 | 0.070320 | 2048 | 0.1530 | 532.9 | 618 | 98,304 |
| LoHA | 512 | 39.692561 | 0.442490 | 512 | 0.0754 | 284.5 | 362 | 196,608 |
| LoHA | 1024 | 37.631540 | 0.490432 | 1024 | 0.1066 | 361.1 | 462 | 196,608 |
| LoHA | 2048 | 35.616561 | 0.557069 | 2048 | 0.1583 | 534.0 | 620 | 196,608 |
| DoRA | 512 | 31.147211 | 0.078645 | 512 | 0.0780 | 282.3 | 358 | 108,544 |
| DoRA | 1024 | 30.505115 | 0.042256 | 1024 | 0.1020 | 354.0 | 458 | 108,544 |
| DoRA | 2048 | 30.465097 | 0.046829 | 2048 | 0.1520 | 512.6 | 584 | 108,544 |
| Residual GLU-LoRA | 512 | **30.863558** | 0.061067 | 512 | 0.0821 | 285.0 | 362 | 196,608 |
| Residual GLU-LoRA | 1024 | **30.232525** | 0.033566 | 1024 | 0.1033 | 361.5 | 464 | 196,608 |
| Residual GLU-LoRA | 2048 | **30.168297** | 0.030464 | 2048 | 0.1514 | 534.5 | 620 | 196,608 |

固定batch条件でもResidual GLU-LoRAが全系列長で最小lossでしたが、LoRAとの差は小さく、LoHAは改善しませんでした。
512では総token数が少ないため全方式のlossがtoken-normalized条件より高く、2048ではtokens/stepが一致するためほぼ同じ結果です。
peak allocatedはbatch=`1`によりtoken-normalized条件より低く、系列長に応じて約`282`→`535 MiB`へ増加しました。
この結果だけでResidualの一般的優位性や既定値変更は判断しません。学習率・rank・alpha・総token budgetを切り分ける比較は後続節に記録しています。

当時の次の切り分けとして、parameter-matchedのrank/alpha（LoRA/DoRA=`16`、LoHA/Residual=`8`）を固定し、
learning rateだけをsweepするwrapperを追加しました。既定値は`1e-4,3e-4,1e-3`、系列長=`2048`、batch=`1`、
seed=`0,1,2`です。1条件あたり12 run、全体で36 runとなり、各LRの結果はreport内で別groupに集計されます。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TEXT_LM_ADAPTER_DEVICE=cuda TEXT_LM_ADAPTER_BF16=1 \
TEXT_LM_ADAPTER_EPOCHS=3 TEXT_LM_ADAPTER_STEPS_PER_EPOCH=200 \
bash benchmarks/launchers/run_text_lm_adapter_lr_sweep.sh
```

全36ケース（4方式×3 LR×3 seed）がCUDA/BF16で完了し、rank/alphaはparameter-matched条件、batch=`1`、
tokens/step=`2048`です。

| adapter | LR | validation loss平均 | seed std | 最終step秒平均 | peak allocated MiB平均 | peak reserved MiB平均 | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 1e-4 | 40.036400 | 0.113925 | 0.1526 | 533.7 | 620 | 393,216 |
| LoRA | 3e-4 | 31.984032 | 0.172801 | 0.1253 | 533.7 | 620 | 393,216 |
| LoRA | 1e-3 | **29.074682** | 0.177925 | 0.1073 | 533.7 | 620 | 393,216 |
| LoHA | 1e-4 | 45.225524 | 0.002393 | 0.1507 | 534.3 | 620 | 393,216 |
| LoHA | 3e-4 | 31.743594 | 0.329511 | 0.1241 | 534.3 | 620 | 393,216 |
| LoHA | 1e-3 | **27.491870** | 0.265053 | 0.1094 | 534.3 | 620 | 393,216 |
| DoRA | 1e-4 | 40.282377 | 0.096181 | 0.1517 | 512.9 | 584 | 403,456 |
| DoRA | 3e-4 | 32.884361 | 0.259771 | 0.1240 | 512.9 | 584 | 403,456 |
| DoRA | 1e-3 | **31.026701** | 0.378491 | 0.1105 | 512.9 | 584 | 403,456 |
| Residual GLU-LoRA | 1e-4 | 36.647255 | 0.181924 | 0.1550 | 534.7 | 620 | 393,216 |
| Residual GLU-LoRA | 3e-4 | 29.446300 | 0.050318 | 0.1236 | 534.7 | 620 | 393,216 |
| Residual GLU-LoRA | 1e-3 | **27.012182** | 0.026656 | 0.1087 | 534.7 | 620 | 393,216 |

全方式で`1e-3`が最良でした。特にResidual GLU-LoRAはLoRA/LoHAより低いlossを維持し、
`3e-4`から`1e-3`への改善も大きくなっています。一方、step時間のLR間比較は実行順やCUDA warm-upの影響を含むため、
性能優劣の根拠にはしません。peak memoryとoptimizer stateは同じ方式内ではLRに依存せず、方式のparameterizationで決まっています。
3 epoch×200 stepsの短期・単一モデル条件のため、`1e-3`を既定値へ変更せず、長期収束と複数seedで再確認します。

### Parameter-matched比較

rankによるparameter数の差を抑えるため、LoRA/DoRAはrank=`16`、LoHA/Residual GLU-LoRAはrank=`8`として比較しました。
alpha/rankはすべて1、その他の条件は上記token-normalized比較と同じです。全36ケースが完了しました。

| adapter | rank | max seq | validation loss平均 | seed std | 最終step秒平均 | peak allocated MiB平均 | optimizer state bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 16 | 512 | 31.842246 | 0.096698 | 0.1213 | 572.1 | 393,216 |
| LoRA | 16 | 1024 | 31.632145 | 0.118621 | 0.1199 | 582.3 | 393,216 |
| LoRA | 16 | 2048 | 31.437477 | 0.182271 | 0.1207 | 598.9 | 393,216 |
| LoHA | 8 | 512 | 30.419158 | 0.063277 | 0.1280 | 573.3 | 393,216 |
| LoHA | 8 | 1024 | 30.367769 | 0.057113 | 0.1214 | 583.5 | 393,216 |
| LoHA | 8 | 2048 | 30.251058 | 0.081713 | 0.1130 | 600.1 | 393,216 |
| DoRA | 16 | 512 | 32.402484 | 0.054242 | 0.1277 | 573.3 | 403,456 |
| DoRA | 16 | 1024 | 32.073106 | 0.164971 | 0.1200 | 583.5 | 403,456 |
| DoRA | 16 | 2048 | 32.040424 | 0.070700 | 0.1165 | 600.1 | 403,456 |
| Residual GLU-LoRA | 8 | 512 | **29.225009** | 0.040316 | 0.1295 | 574.2 | 393,216 |
| Residual GLU-LoRA | 8 | 1024 | **29.101715** | 0.032505 | 0.1190 | 584.3 | 393,216 |
| Residual GLU-LoRA | 8 | 2048 | **29.157862** | 0.030941 | 0.1132 | 601.0 | 393,216 |

このparameter-matched条件ではResidual GLU-LoRAが全系列長で最小lossでした。LoHAもrank=`4`時のloss約35〜38から
約30.25〜30.42へ改善し、rankを増やす効果が確認できます。一方、LoRA/DoRAはrankを増やしても今回の学習率・alpha設定では
rank4結果を上回りませんでした。これはrankそのものの上限というより、初期化・学習率・alpha・seedの相互作用を含む結果として扱います。
DoRAはmagnitude parameterのためoptimizer stateが約403 KBとなり、他のparameter-matched方式より約10 KB大きくなります。
peak allocatedはcontext lengthに応じて約572→601 MiBへ増加しましたが、step時間は約0.11〜0.13秒でした。
fixed-batch比較とmerge後の出力一致・strict load検証まで完了しています。

adapter比較用baseを先に作成する場合は、次の専用スクリプトを使います。既定ではadapter比較と同じ小型`naive`
モデルで、Qwen tokenizer・TinyStories・最大context=`2048`・batch=`1`を使用します。完了時にadapter比較へ渡す
checkpointのパスを表示します。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TEXT_LM_ADAPTER_DEVICE=cuda TEXT_LM_ADAPTER_BF16=1 \
TEXT_LM_ADAPTER_EPOCHS=3 TEXT_LM_ADAPTER_STEPS_PER_EPOCH=200 \
bash benchmarks/launchers/run_text_lm_adapter_base.sh
```

表示された`artifacts/model.safetensors`を`TEXT_LM_ADAPTER_BASE_CHECKPOINT`へ設定して、上記のadapter比較を実行します。
baseのmodel dimensions、tokenizer、dataset、dataset seed、sequence lengthはadapter側と揃えてください。

固定batch sizeでの比較は、専用wrapperで実行します。既定ではbatch=`1`、系列長は`512,1024,2048`、
出力先は`output/text-lm-adapter-sequence-fixed-batch`です。token-normalized比較とは別directoryになるため、
reportを混在させません。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TEXT_LM_ADAPTER_DEVICE=cuda TEXT_LM_ADAPTER_BF16=1 \
TEXT_LM_ADAPTER_EPOCHS=3 TEXT_LM_ADAPTER_STEPS_PER_EPOCH=200 \
TEXT_LM_ADAPTER_MAX_SEQ_LENS=512,1024,2048 \
bash benchmarks/launchers/run_text_lm_adapter_fixed_batch_comparison.sh
```

wrapperは`TEXT_LM_ADAPTER_BASE_CHECKPOINT`が未指定の場合、parameter-matched比較と同じく
`output/text-lm-adapter-base-2048/runs/`の最新base checkpointを自動検出します。

parameter数を揃えたadapter比較には、専用wrapperを使います。既定値はLoRA/DoRAがrank=`16`、LoHA/Residual
GLU-LoRAがrank=`8`で、alpha/rankも1に揃えます。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TEXT_LM_ADAPTER_BASE_CHECKPOINT=output/text-lm-base-2048.safetensors \
TEXT_LM_ADAPTER_DEVICE=cuda TEXT_LM_ADAPTER_BF16=1 \
TEXT_LM_ADAPTER_EPOCHS=3 TEXT_LM_ADAPTER_STEPS_PER_EPOCH=200 \
bash benchmarks/launchers/run_text_lm_adapter_budget_comparison.sh
```

全方式をrank=`8`で比較するrank-matched条件は、次のように切り替えます。

```bash
TEXT_LM_ADAPTER_COMPARISON_MODE=rank_matched \
bash benchmarks/launchers/run_text_lm_adapter_budget_comparison.sh
```

wrapperは`TEXT_LM_ADAPTER_BASE_CHECKPOINT`が未指定の場合、既定の`output/text-lm-adapter-base-2048/runs/`から
最新のbase checkpointを自動検出します。別のbaseを使う場合は、従来どおり環境変数で明示指定してください。

reportはadapter・rank・alpha・context length・tokens/stepごとに集計するため、parameter-matchedとrank-matchedの
結果を同じoutput directoryへ混在させない運用を推奨します。

長系列比較のadapter checkpointをまとめてmerge検証するには、次を実行します。各checkpointについて一時的に
merged safetensorsを作成し、adapter moduleなしのplain modelへstrict loadしたうえで、probe入力に対するwrapped model
との差を`torch.allclose(atol, rtol)`で確認します。検証後のmergedファイルは一時ディレクトリから削除され、元checkpointは変更されません。

```bash
PYTHONPATH=. python3 -m verify.text_lm_adapter_merge \
  --input-dir output/text-lm-adapter-budget-parameter_matched \
  --output output/text-lm-adapter-budget-parameter_matched/merge-report-cuda.json \
  --device cuda --probe-seq-len 16
```

`--device cpu`へ変更すればCUDAなしでも検証できます。許容値はBF16が`atol=5e-3, rtol=1e-3`、FP32が
`atol=3e-4, rtol=1e-5`です。これは長系列のlogitsでmerge前後のmatmul評価順序が異なる影響を含みます。

parameter-matchedの36 checkpoint（4方式×3系列長×3 seed）をCPU・probe=`16`で検証しました。
全件でstrict loadとadapter key除去が成功し、出力一致判定も通過しました。CUDAでも全36件が通過し、
最大絶対差は`2.689e-4`、最大peak allocatedは`351.25 MiB`でした。個別merge reportは公開せず、集計値のみを記録します。
peak値はprobe=`16`のmerge検証時の値であり、長系列学習時のpeak VRAMではありません。

base checkpointは`naive`・同じmodel dimensionsで、可能なら最大context=`2048`の条件で先に作成します。
同じstep数だけを比較すると長系列ほど1 stepあたりのtoken数が増えるため、短期lossの順位だけでなく、
固定token budgetでの比較と、固定optimizer step数での比較を分けて解釈します。現環境はCUDA未使用のため、
512/1024/2048の長期品質比較とpeak VRAM測定は未実行です。
