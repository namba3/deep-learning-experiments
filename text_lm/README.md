# `text_lm/`

小型decoder-only Transformerの事前学習・instruction tuning scriptです。実行時の完全なCLIは`train.py --help`を確認してください。

## Dataset modes

既定値は`--data-mode text`、`--dataset-name roneneldan/TinyStories`です。`text` modeでは`text`列をHF streamingで読み、`--max-train-tokens`と`--max-eval-tokens`の範囲だけを`--max-seq-len`単位の固定長blockへpackします。TinyStoriesは小規模な初期検証向けです。

```bash
# 既定のTinyStoriesで事前学習（各budgetは実際には固定長block単位に切り捨て）
PYTHONPATH=. python3 -m text_lm.train \
  --epochs 1 --steps-per-epoch 100 \
  --max-train-tokens 1_000_000 --max-eval-tokens 100_000

# OpenWebTextへ切り替え
PYTHONPATH=. python3 -m text_lm.train \
  --data-mode text --dataset-name Skylion007/openwebtext \
  --max-train-tokens 10_000_000 --max-eval-tokens 1_000_000

# TinyStories / OpenWebTextを同一token budgetで構成比較
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TEXT_LM_DATASETS=roneneldan/TinyStories,Skylion007/openwebtext \
TEXT_LM_DEVICE=cuda TEXT_LM_BF16=1 \
  bash benchmarks/launchers/run_text_lm_pretraining_comparison.sh

# Alpacaは事前学習ではなくinstruction tuning用に明示する
PYTHONPATH=. python3 -m text_lm.train \
  --data-mode instruction --dataset-name tatsu-lab/alpaca \
  --epochs 1 --batch-size 1
```

`text` modeは同じstreamの先頭を学習、続くblockを評価に割り当てるため、raw documentを全件保持しません。`--text-column`で`text`以外の列も指定できます。`--dataset-config`と`--dataset-split`はHugging Face datasetの構成・splitを明示する場合に使います。

既定のHugging Face cacheが読み取り専用でも、完了済みのArrow shardを直接再利用できます。比較スクリプトの各runは、指定したoutput directoryの`runs/`に設定・metrics・checkpointを保存します。

## Distillation

`--distill-mode logits`を指定すると、学生モデルのhard next-token CEに、`--teacher-model`（既定値は`Qwen/Qwen3.5-0.8B`）のsoft logits KLを加えます。`--distill-alpha`はhard CEの重み、`--distill-temperature`はsoft targetの温度です。teacherは凍結され、checkpointには保存されませんが、学習中はteacher分のVRAMと計算時間が必要です。

`--bf16`を付けると学生モデルのtrainable parameterとteacher weightをBF16 storageにします。Linearの出力と最終logitsはFP32、RMSNormはparameter dtypeで計算して入力dtypeへ戻します。指定しない場合は従来通りFP32です。

Qwen tokenizerのように語彙数が大きい場合、全`(batch, sequence, vocabulary)` logitsを保持するとVRAM使用量が急増します。学習時は既定の`--vocab-chunk-size 8192`で語彙方向を分割してcausal cross entropyを計算します。0以下を指定すると全logitsのreference経路を使います。

学習metricsには、通常の`total loss`に加えてhard next-token CEとPPL（`exp(hard CE)`）を記録します。PPLは解釈しやすい評価指標としてのみ使い、指数変換による勾配スケールの不安定化を避けるためlossへは直接適用しません。蒸留時はraw KL-Divergence、temperature scaling後のsoft loss（`T² * KL`）、および混合後のlossを分離して記録します。実際の蒸留lossは次式です。

```text
total_loss = alpha * hard_CE + (1 - alpha) * (temperature^2 * KL)
```

`metrics.jsonl`と比較reportには`train_ppl`、`eval_ppl`、`train_kl_divergence`、`eval_kl_divergence`などが入り、蒸留なしのKL値は0になります。

```bash
# TinyStoriesをQwen3.5 teacherで蒸留
PYTHONPATH=. python3 -m text_lm.train \
  --data-mode text --distill-mode logits \
  --max-train-tokens 1_000_000 --max-eval-tokens 100_000 \
  --epochs 1 --steps-per-epoch 100 --bf16

# teacherを変更する場合
PYTHONPATH=. python3 -m text_lm.train \
  --data-mode text --distill-mode logits \
  --teacher-model Qwen/Qwen3.5-0.8B \
  --distill-temperature 2.0 --distill-alpha 0.5
```

teacherの語彙に含まれる学生tokenizer非公開の予約IDは、蒸留時の共通語彙から除外して再正規化します。

## Next-token generation

学習済みsafetensorsから、会話templateを使わずに引数の生テキストの続きを生成できます。checkpointの`artifacts/tokenizer`を優先してtokenizerを読み込み、metadataからmodel構成とdtypeを復元します。`--max-new-tokens`の上限は1024で、既定のsamplingはtemperature=`0.8`、top-k=`40`、top-p=`0.95`（95%）です。`--temperature 0`でgreedy decodingにできます。

```bash
PYTHONPATH=. python3 -m text_lm.generate \
  --checkpoint output/runs/<run-id>/artifacts/model.safetensors \
  --prompt "Once upon a time" \
  --device cuda \
  --max-new-tokens 1024
```

samplingを使う場合は`--temperature`、`--top-k`、`--top-p`を指定します。出力はpromptを含む補完済みテキストです。checkpointの`max_seq_len`を超えて生成する場合は、直近contextを使うsliding windowで最大1024 tokenまで継続します。

```bash
PYTHONPATH=. python3 -m text_lm.generate \
  --checkpoint output/runs/<run-id>/artifacts/model.safetensors \
  --prompt "The little robot" \
  --temperature 0.8 --top-k 40 --top-p 0.95 --seed 0
```

## Decoder architecture

`--architecture`で次の5種類を切り替えられます。すべてpre-norm RMSNorm、decoder-only、Q/Kへの1D RoPEを使用します。学習可能なposition embeddingは使用しません。

| 値 | 構成 |
| --- | --- |
| `naive` | 独立した重みを持つGated MHA Transformer block × `--num-layers` |
| `mhla3-gqa` | causal MHLA block × 3 + Gated GQA block × 1 を1 cycleとして、合計 `--num-layers` block |
| `looped` | `--looped-blocks`個の物理blockを、合計`--num-layers` depthになるまで反復 |
| `looped-hybrid` | 独立block × prefix → looped block × repeat → 独立block × suffix |
| `mhla3-gqa-looped-hybrid` | 独立(MHLA×3 + GQA×1) cycle × prefix → 共有cycle × repeat → 独立cycle × suffix |

`shared-fixed`と`shared-variable`はshared super weightのアーカイブ実装です。新規CLIの選択肢と比較スクリプトからは除外していますが、過去checkpointの復元と内部APIでは保持しています。

`mhla3-gqa`では`--num-layers`を総block数として扱い、4で割ったcycle数だけパターンを反復します。そのため`--num-layers`は4の倍数である必要があります。`--kv-heads`はGQA/MHLAのK/V head数で、`--num-heads`を割り切る必要があります。

`looped`では`--num-layers`が展開後の総depth、`--looped-blocks`が保存する物理block数です。`--num-layers / --looped-blocks`回だけ同じblock stackを再利用します。既定値は1物理blockなので、例として`--num-layers 16`では1 blockを16回反復します。複数blockのstackを反復する場合は、`--looped-blocks 4`のように指定します。

`looped-hybrid`では、総depthを`prefix + looped_blocks × repeats + suffix`で指定します。既定値は`4 + 2 × 4 + 4 = 16`です。中央の反復部分だけを共有し、前段・後段は独立重みを持つため、純粋な`looped`と通常Transformerの中間的な構成になります。

`mhla3-gqa-looped-hybrid`では4 block（causal MHLA×3 + Gated GQA×1）を1 cycleとし、`--num-layers / 4` cycleを`prefix-cycles + repeats + suffix-cycles`に分けます。既定値は`1 + 2 + 1 = 4` cycle、つまり総depth 16です。中央cycleだけを共有し、前後のcycleは独立重みになります。

例:

```bash
# 独立weightの基準モデル
PYTHONPATH=. python3 -m text_lm.train --architecture naive

# 合計4 block（MHLA 3段 + Gated GQA 1段）
PYTHONPATH=. python3 -m text_lm.train \
  --architecture mhla3-gqa --num-layers 4 --kv-heads 8

# 1つの物理blockを16回反復するlooped Transformer
PYTHONPATH=. python3 -m text_lm.train \
  --architecture looped --num-layers 16 --looped-blocks 1

# 独立4 block → 共有2 blockを4回 → 独立4 block
PYTHONPATH=. python3 -m text_lm.train \
  --architecture looped-hybrid --num-layers 16 \
  --looped-prefix-layers 4 --looped-blocks 2 \
  --looped-repeats 4 --looped-suffix-layers 4

# 独立cycle → MHLA3+GQA cycleを2回共有 → 独立cycle
PYTHONPATH=. python3 -m text_lm.train \
  --architecture mhla3-gqa-looped-hybrid --num-layers 16 \
  --mhla-looped-prefix-cycles 1 --mhla-looped-repeats 2 \
  --mhla-looped-suffix-cycles 1 --kv-heads 8
```

画像用の`Grid2DMHLA`は双方向2D grid向けのため、Alpacaでは使用していません。`mhla3-gqa`のMHLAは、未来tokenを参照しないcausal sequence MHLAです。`looped`と`looped-hybrid`は標準のcausal Gated MHA blockを反復します。

## 実行例

```bash
python3 -m text_lm.train \
  --data-mode instruction \
  --dataset-name tatsu-lab/alpaca \
  --epochs 1 --batch-size 1
```

起動前検証:

```bash
# tokenizer・dataset・modelをロードしない軽量確認
PYTHONPATH=. python3 -m text_lm.train --dry-run --epochs 1 --batch-size 2 --num-workers 0

# tokenizer・dataset・modelとlogits shapeを確認し、学習せず終了
PYTHONPATH=. python3 -m text_lm.train --validate-only --epochs 1 --batch-size 2 --num-workers 0
```

`--validate-only`はtokenizerとdatasetをロードするため、未取得のものは外部から取得される場合があります。検証結果は`output/runs/<run-id>/metrics.jsonl`へ保存されます。

## Low-rank adapters

adapter比較は通常の`train.py`と分離した`train_adapter.py`で実行します。現時点では、独立weightを持つ
`--architecture naive`だけを対象にしています。`shared-fixed`、`looped`系は共有super weight/blockの
parameterizationが異なるため、別のadapter設計として扱います。

```bash
PYTHONPATH=. python3 -m text_lm.train_adapter \
  --data-mode text --dataset-name roneneldan/TinyStories \
  --tokenizer Qwen/Qwen3.5-0.8B \
  --lora-base-checkpoint output/text-lm-base.safetensors \
  --architecture naive \
  --adapter rglu_lora \
  --lora-rank 4 \
  --lora-alpha 4
```

`--adapter glu_lora`も利用できます。GLU-LoRAは`(B1 @ A1) ⊙ SiLU(B2 @ A2)`、Residual GLU-LoRAは
`(B1 @ A1) ⊙ (1 + SiLU(B2 @ A2))`をweight-spaceで構成するため、学習後に通常Linearへmergeできます。

デフォルトの対象は各blockの`attn.q_proj`、`attn.k_proj`、`attn.v_proj`、`attn.out_proj`、`ffn.2`です。
対象を限定する場合は`--lora-target`を繰り返し指定します。text modeではTinyStoriesを固定長token blockへ
packし、base checkpointは同じtoken budget・tokenizer・subsetで先に通常学習して各adapterへ共有してください。
Alpaca形式のinstruction tuningを比較する場合だけ、`--data-mode instruction --dataset-name tatsu-lab/alpaca`
へ切り替えます。

merge済みcheckpointは次で作成できます。

```bash
PYTHONPATH=. python3 -m text_lm.export_adapter \
  --checkpoint output/runs/<run-id>/checkpoints/epoch_1/model.safetensors \
  --output output/text-lm-merged.safetensors
```

adapter専用のCLIは`python3 -m text_lm.train_adapter --help`、通常学習のCLIは`python3 -m text_lm.train --help`を
参照してください。

短時間の構成比較では、`--max-train-examples`と`--max-eval-examples`で使用するsubsetのサイズを固定できます。既定値は0（全件）で、正の値を指定した場合はsplit後の先頭N件だけを使います。例えば、学習128件・評価32件に制限するには次を指定します。

```bash
PYTHONPATH=. python3 -m text_lm.train \
  --data-mode instruction \
  --architecture mhla3-gqa-looped-hybrid --num-layers 16 \
  --max-train-examples 128 --max-eval-examples 32
```

GPUで候補3構成を複数seed比較するスクリプトは[`benchmarks/launchers/run_text_lm_instruction_comparison.sh`](../benchmarks/launchers/run_text_lm_instruction_comparison.sh)です。事前学習用の全構成比較は[`benchmarks/launchers/run_text_lm_pretraining_comparison.sh`](../benchmarks/launchers/run_text_lm_pretraining_comparison.sh)です。後者は終了時に`report.json`を自動生成します。`TEXT_LM_SEEDS=0,1`、`TEXT_LM_EPOCHS=1`、`TEXT_LM_STEPS_PER_EPOCH=20`などの環境変数で短縮できます。事前学習比較の既定データは`roneneldan/TinyStories`です。蒸留比較では`TEXT_LM_DISTILL_MODE=logits`、`TEXT_LM_TEACHER_MODEL=Qwen/Qwen3.5-0.8B`、`TEXT_LM_DISTILL_TEMPERATURE=2.0`、`TEXT_LM_DISTILL_ALPHA=0.5`を指定します。
GPUで候補3構成を複数seed比較するスクリプトは[`benchmarks/launchers/run_text_lm_instruction_comparison.sh`](../benchmarks/launchers/run_text_lm_instruction_comparison.sh)です。事前学習用の全構成比較は[`benchmarks/launchers/run_text_lm_pretraining_comparison.sh`](../benchmarks/launchers/run_text_lm_pretraining_comparison.sh)です。後者は終了時に`report.json`を自動生成します。`TEXT_LM_SEEDS=0,1`、`TEXT_LM_EPOCHS=1`、`TEXT_LM_STEPS_PER_EPOCH=20`などの環境変数で短縮できます。`TEXT_LM_MAX_SEQ_LENS=128,1024,4096`と指定すると系列長を横断して比較でき、各runとreport summaryに`max_seq_len`が記録されます。事前学習比較の既定データは`roneneldan/TinyStories`です。蒸留比較では`TEXT_LM_DISTILL_MODE=logits`、`TEXT_LM_TEACHER_MODEL=Qwen/Qwen3.5-0.8B`、`TEXT_LM_DISTILL_TEMPERATURE=2.0`、`TEXT_LM_DISTILL_ALPHA=0.5`を指定します。

## Optimizer比較

Optimizerの学習性能比較には、既存のモデル・tokenize・dataset splitを再利用する
[`verify/text_lm_optimizer_convergence.py`](../verify/text_lm_optimizer_convergence.py)を使用します。
初期構成は`naive` decoder-only Transformerに固定し、モデルのparameterization差を避けて
OptimizerのSchedule-Free差分を比較します。データセットは`text` modeと同じTinyStoriesを使い、固定長blockへ
packした全tokenを対象とするcausal language modeling lossです。LRSFの切り分けでは
`AdamW`、full hidden stateの`AdamW-SF`、hidden deltaだけを低rank化した`AdamW-LRSF`、
factorized second momentの`CAME`を既定比較とします。`CAME-LRSF`は主比較から外し、
必要な場合だけSchedule-Free近似の追加ablationとして`--optimizers CAME-LRSF`を明示します。
4-way比較を再実行するwrapperは
[`benchmarks/launchers/run_text_lm_optimizer_comparison.sh`](../benchmarks/launchers/run_text_lm_optimizer_comparison.sh)です。
比較wrapperではAdamW-SFのbackendを既定で`torch`に固定し、AdamW-LRSFのrank=512
（フルSFフォールバック）と更新経路を揃えます。速度検証でTritonを使う場合は
`TEXT_LM_ADAMW_SF_BACKEND=auto`を指定します。
長期Transformer学習のLRSF標準条件は`hard` refreshです。wrapperと
`verify.text_lm_optimizer_convergence`の既定値も`hard`とし、projectionを一定間隔で即時交換します。
refreshなしの比較baselineは`TEXT_LM_REFRESH_MODE=none`または`frozen`指定で選択します。
`TEXT_LM_REFRESH_MODE=shadow`を指定すると、active branchの裏でshadow branchのlow-rank
delta/projectionを毎step更新し、intervalごとにshadowを昇格します。full optimizer stateは
共有されるため、full Schedule-Free stateを二重化しませんが、low-rank stateとprojectionは
二重になります。shadowは実験用であり、既定値は変更していません。
PA/PB smooth refreshは、wrapperの`TEXT_LM_REFRESH_MODE=smooth`、
`TEXT_LM_REFRESH_INTERVAL`、`TEXT_LM_REFRESH_WINDOW`、
`TEXT_LM_REFRESH_MIX=linear|smoothstep|ema|stochastic`で指定できます。orthogonal refreshは
`TEXT_LM_ORTHOGONAL_RATE`と`TEXT_LM_ORTHOGONAL_DIRECTION`で指定します。
learning rateだけを変える比較には
[`benchmarks/launchers/run_text_lm_optimizer_lr_sweep.sh`](../benchmarks/launchers/run_text_lm_optimizer_lr_sweep.sh)を使います。
LRSFのrankだけを変える比較には
[`benchmarks/launchers/run_text_lm_optimizer_rank_sweep.sh`](../benchmarks/launchers/run_text_lm_optimizer_rank_sweep.sh)を使います。
refresh方式を同一条件で長期比較する場合は
[`benchmarks/launchers/run_text_lm_optimizer_refresh_comparison.sh`](../benchmarks/launchers/run_text_lm_optimizer_refresh_comparison.sh)を使います。
既定ではhard、smoothstep、ema、ema_fast、stochasticをrank=8/16・10 epoch×100 stepで比較し、
`TEXT_LM_OPTIMIZER_REFRESH_DIR`以下へ方式別に保存します。短縮時は
`TEXT_LM_EPOCHS`、`TEXT_LM_STEPS_PER_EPOCH`、`TEXT_LM_REFRESH_INTERVAL`、
`TEXT_LM_REFRESH_WINDOW`を指定します。
`hard`は長期学習用の標準条件です。`frozen`はrefreshなし（旧表記`fixed`）の比較baselineで、
`ema_fast`は`ema_decay=0.96`を使う急峻EMAで、
通常のEMAよりPBへの遷移を速めます。個別指定では`TEXT_LM_REFRESH_EMA_DECAY`を使います。
transport損失を調査するときは`TEXT_LM_RECORD_REFRESH_DIAGNOSTICS=1`または
`--record-refresh-diagnostics`を指定します。refresh eventごとにdecoded deltaの相対誤差、
norm比、cosine similarity、最後のrefresh stepをJSONへ記録します。この経路はrefresh時に
一時的なfull decoded deltaを生成するため、step時間の比較には使わないでください。
hard refreshのprojection交換を緩和する実験には`TEXT_LM_REFRESH_TRANSPORT_OVERLAP`または
`--refresh-transport-overlap`を使います。`1.0`は旧basisを保持し、`0.0`は従来の新しい
random basis、途中の値は旧basisと新basisを混ぜてからQR直交化します。これはdecoded deltaの
transport損失を減らすための実験用ノブで、既定値は`None`（従来のrandom refresh）です。
overlapの検証では診断付きの縮小runを先に行い、品質・速度を測る本runでは診断を無効にします。

TinyStoriesとQwen3.5 tokenizerがキャッシュ済みの場合の最小GPU実行例:

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=. python3 \
  -m verify.text_lm_optimizer_convergence \
  --device cuda --dtype bf16 --seeds 0,1,2 \
  --optimizers AdamW,AdamW-SF,AdamW-LRSF,CAME \
  --train-tokens 1000000 --eval-tokens 100000 \
  --epochs 3 --steps-per-epoch 100
```

このprobeはproduction checkpointの代わりではなく、同じ初期weight・split・batch順で
validation loss、perplexity、Optimizer state、peak CUDA memory、Optimizer step時間を比較する
ためのものです。

既定のLR sweepは`1e-4,3e-4,1e-3`です。例えばseed数と実行時間を短縮する場合:

```bash
TEXT_LM_SEEDS=0,1 TEXT_LM_EPOCHS=1 TEXT_LM_STEPS_PER_EPOCH=50 \
  ./benchmarks/launchers/run_text_lm_optimizer_lr_sweep.sh
```

rank=4/8/16とフルSchedule-Freeへフォールバックするrank=512を、learning rate=`1e-3`で比較する場合:

```bash
./benchmarks/launchers/run_text_lm_optimizer_rank_sweep.sh
```

rankと対象optimizerを明示する場合:

```bash
TEXT_LM_RANKS=4,8,16,512 \
TEXT_LM_OPTIMIZERS=AdamW-SF,AdamW-LRSF \
TEXT_LM_LEARNING_RATE=1e-3 \
  ./benchmarks/launchers/run_text_lm_optimizer_rank_sweep.sh
```

実用候補のrank=8/16を長期比較する場合は、既定で10 epoch×100 stepを実行します。
AdamW-SFは修正版Triton経路（`backend=auto`）を使用します。

```bash
./benchmarks/launchers/run_text_lm_optimizer_long_rank_comparison.sh
```

step数を短縮する場合:

```bash
TEXT_LM_LONG_EPOCHS=1 TEXT_LM_LONG_STEPS_PER_EPOCH=50 \
  ./benchmarks/launchers/run_text_lm_optimizer_long_rank_comparison.sh
```

APOLLO-CAMEのnorm-growth limiterを比較する場合は、次を指定します。

```bash
TEXT_LM_APOLLO_DISABLE_NORM_GROWTH_LIMITER=1 \
TEXT_LM_APOLLO_NORM_GROWTH_RATE=1.05 \
  ./benchmarks/launchers/run_text_lm_optimizer_comparison.sh
```

`text` modeのTinyStoriesは、Hugging Faceの通常キャッシュまたはstreaming loaderを利用します。キャッシュが読み取り専用でも、完了済みArrow shardを再利用できます。instruction modeでAlpacaを使う場合は、次のように明示します。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m text_lm.train \
  --data-mode instruction \
  --dataset-name tatsu-lab/alpaca --tokenizer Qwen/Qwen3.5-0.8B
```

キャッシュを使わず明示する場合は、`--dataset-path`にJSON/JSONLまたはHugging Face DatasetsのArrowファイルを指定できます。
