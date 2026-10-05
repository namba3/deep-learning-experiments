# Optimizer runtime and convergence probes

optimizerのruntime、state量、収束probeを再実行する手順です。CUDAを使う例は長時間かかる場合があり、結果の保存先を分けてください。

## Optimizer runtime benchmark

optimizer専用の比較は、モデルやdatasetの影響を分離するため次で実行できます。

```bash
python3 -m verify.optimizers --device auto --dtype fp32
python3 -m verify.optimizers --device cuda --dtype bf16 --steps 20
```

CPU smoke testを短時間で再実行する場合:

```bash
python3 -m verify.optimizers \
  --device cpu --dtype fp32 --warmup 1 --steps 2 \
  --shapes 4x3,16x16,4096
```

projection refresh stepも含めて測定する場合は、`--update-proj-gap`をstepsより
小さくします。

```bash
python3 -m verify.optimizers \
  --device cuda --dtype bf16 --warmup 10 --steps 50 \
  --update-proj-gap 10
```

JSONの`projection_refresh_steps`（開始イベント数）と
`projection_refresh_active_steps`（smooth transition中の更新数）、通常step・refresh
step別のhost/GPU時間を比較できます。smoothでは後者をtransitionの実コストとして
使用してください。

smooth refreshを測定する場合は、次のように指定します。refresh中は新旧の低ランク
projectionとmomentを一時的に二重保持するため、JSONの`persistent_state_bytes`は
定常時より大きくなります。測定区間内にrefreshが完了して定常stateへ戻る場合は、
`peak_persistent_state_bytes/elements`で一時的な二重stateを確認できます。
`estimated_persistent_state_bytes`はactive stateを含めて一致することを確認できます。

APOLLO projectionをintervalなしで毎step微小回転させる場合は、
`--orthogonal-refresh-rate`を指定します。回転回数はJSONの
`orthogonal_refresh_steps`に記録されます。
`--orthogonal-refresh-direction loss_directed`を明示すると、現在gradientの射影エネルギーを
増やすStiefel接空間方向を一次近似として使います。実損失を直接評価するものではないため、
実験用のproxyとしてrandom条件と比較してください。実験スクリプトでは
`--orthogonal-direction loss_directed`に対応します。
LRSFでは`--orthogonal-refresh-signal effective_update`を指定すると、既に計算済みの
CAME/APOLLO-CAME effective updateを`R_delta`の回転signalに使います。既定の`gradient`は
APOLLO互換のraw-gradient proxyです。

LRSFのraw gradient版とeffective update版を比較する場合は、出力先を分けて同じseedで
実行します。

```bash
verify/launchers/run_lrsf_gpu_validation.sh \
  --seeds 0,1,2 --modes fixed,orthogonal \
  --orthogonal-direction loss_directed \
  --orthogonal-signal gradient \
  --output-dir output/lrsf-loss-directed-gradient

verify/launchers/run_lrsf_gpu_validation.sh \
  --seeds 0,1,2 --modes fixed,orthogonal \
  --orthogonal-direction loss_directed \
  --orthogonal-signal effective_update \
  --output-dir output/lrsf-loss-directed-update

verify/launchers/run_lrsf_gpu_validation.sh \
  --seeds 0,1,2 --modes fixed,orthogonal \
  --orthogonal-rate 0.001 \
  --orthogonal-direction loss_lowering \
  --orthogonal-signal gradient \
  --record-step-metrics --record-update-norms \
  --output-dir output/lrsf-loss-lowering-r0001
```

各結果は`python3 -m verify.lrsf_gpu_report <output-dir>`で集計できます。レポートは
signal別に方式を分けて表示し、同一optimizer・rank・seedの`fixed/frozen`がある場合は
paired validation loss差分と改善seed数も表示します。

```bash
python3 -m verify.optimizers \
  --device cuda --dtype bf16 --warmup 10 --steps 50 \
  --update-proj-gap 10 --projection-refresh-mode smooth \
  --projection-refresh-window 4 --projection-refresh-state transport
```

```bash
# Compare the policy choices directly.
python3 -m verify.optimizers \
  --device cuda --dtype bf16 --matrix-fallback apollo
python3 -m verify.optimizers \
  --device cuda --dtype bf16 --matrix-fallback came
```

同じ初期parameterとgradient列を使い、`CAME`、`APOLLO`、`APOLLO-CAME`、
`APOLLOMini`を次のshapeで比較します。

```text
4x3, 16x16, 32x32, 64x32x3x3, 4096
```

JSONにはoptimizerごとの選択backend、persistent state bytes、dtypeに依存しないstate tensor要素数、state bytesの推定値との一致、host/GPU step時間、
`peak allocated`、`peak reserved`、parameter normを記録します。`--device auto`は
CUDAがなければCPUで実行するため、CPU結果は更新式・state選択の確認、CUDA結果は
実運用の性能・peak memory確認として分けて扱います。`--matrix-fallback`で
`apollo`、`came`、`auto`を指定できるため、state-size基準の自動選択と速度差を
同じ初期parameter/gradient列で比較できます。state要素数はdtypeをまたいで直接集計します。

## Optimizer convergence probe

外部datasetやVAEに依存せず、実際のforward/backwardとoptimizer更新を比較する
小型MLPの決定的な回帰probeも実行できます。初期重み、入力、教師出力はoptimizer間で
共有され、更新後のloss、optimizer step時間、state量、CUDA peak memoryをJSONへ記録します。
これはoptimizerの明らかな収束退行を早期検出するためのprobeであり、画像生成の品質や
実モデルの収束を代替するものではありません。

TinyStoriesのQwen tokenizer text-LMでoptimizerの収束とstateを比較する場合:

```bash
verify/launchers/run_text_lm_lr_ema_confidence_sweep.sh \
  --device cuda --dtype bf16 \
  --rank 8 --seeds 0,1,2 \
  --confidence-betas 0.95,0.99 \
  --confidence-alphas 0.0001,0.001,0.01
```

このwrapperは`AdamW-SF`、`AdamW-LRSF`、`AdamW-LR-EMA-Conf`、
`AdamW-LR-EMA-Conf-LRSF`、`APOLLO`、`APOLLO-Conf`を
同じseedで実行し、confidenceのbeta/alphaごとにJSONを分けて保存します。各結果の
`persistent_state_bytes`と`host_seconds_per_optimizer_step`は診断なしの比較に使い、
`confidence_diagnostics`は`m̂²/(m̂²+ĉ)`、innovation RMS、normalized update RMSの
観測値として解釈します。診断を追加したrunはstep時間が変わるため、性能比較では分離してください。

APOLLO-Conf候補のconfidence感度だけを調べる場合は、optimizer、rank、LR、scale、limiterを
絞ります。beta=`0.95,0.99` × alpha=`1e-4,1e-3,1e-2`の6セルが生成されます。

```bash
bash verify/launchers/run_text_lm_lr_ema_confidence_sweep.sh \
  --device cuda --dtype bf16 \
  --optimizers APOLLO,APOLLO-Conf \
  --rank 4 --seeds 0,1,2 \
  --train-tokens 16384 --eval-tokens 1024 \
  --steps-per-epoch 32 --learning-rate 3e-3 \
  --apollo-scale 1.0 --apollo-disable-norm-growth-limiter \
  --confidence-betas 0.95,0.99 \
  --confidence-alphas 0.0001,0.001,0.01 \
  --output-dir output/text-lm-apollo-conf-rank4-confidence-sweep
```

集計表は同じ出力先の`confidence-report.md`へ保存されます。APOLLOをpaired baselineとして
扱い、loss差、改善seed数、state、step時間差を表示します。

APOLLOの更新norm縮退を、learning rate・scale・norm-growth limiterから切り分ける場合は、
次の低負荷sweepを使います。既定では`3e-4,1e-3,3e-3 × 0.5,1.0 × on,off`を
rank=`8`・3 seed・10 stepで比較します。各セルは独立しているため、中断後も完了済みセルを
skipして再開できます。

```bash
bash verify/launchers/run_text_lm_apollo_scale_sweep.sh \
  --device cuda --dtype bf16 \
  --optimizers APOLLO,APOLLO-Conf \
  --seeds 0,1,2 --rank 8 \
  --output-dir output/text-lm-apollo-scale-sweep
```

集計表は`output/text-lm-apollo-scale-sweep/apollo-scale-sweep-report.md`へ出力されます。
このsweepは更新norm診断を有効にするため、step時間は候補比較用です。最終候補が得られたら
診断なしでpeak VRAMと定常step時間を再測定し、lossはより長いpaired runで確認します。

候補条件を診断なしで延長する場合は、同じwrapperに軸を絞って指定します。

```bash
bash verify/launchers/run_text_lm_apollo_scale_sweep.sh \
  --device cuda --dtype bf16 \
  --learning-rates 1e-3,3e-3 --scales 0.5,1.0 \
  --limiters off --steps-per-epoch 32 --train-tokens 16384 \
  --no-diagnostics --output-dir output/text-lm-apollo-candidates
```

この場合、各セルの`peak_allocated_bytes`、`peak_reserved_bytes`、定常step時間を診断なしで
取得できます。集計レポートのupdate norm欄は`not recorded`になります。

rankとscaleを同時に縮小比較する場合は、`--ranks`を指定します。複数rankのセルは
`rank-*`配下へ保存され、集計表にもrank列が追加されます。

```bash
bash verify/launchers/run_text_lm_apollo_scale_sweep.sh \
  --device cuda --dtype bf16 \
  --ranks 4,8,16 --seeds 0,1,2 \
  --learning-rates 3e-4,3e-3 --scales 0.5,1.0 \
  --limiters off --steps-per-epoch 32 --train-tokens 16384 \
  --no-diagnostics --output-dir output/text-lm-apollo-rank-scale-sweep
```

単一rankの既存指定`--rank 8`と従来の出力形式も互換維持しています。

optimizerごとに候補LRを分けて比較する場合は、fair comparison wrapperを使います。既定値は
AdamW-SF/LRSF=`3e-4`、APOLLO/APOLLO-Conf=`3e-3`・scale=`1.0`・limiter無効です。

```bash
bash verify/launchers/run_text_lm_optimizer_fair_comparison.sh \
  --device cuda --dtype bf16 \
  --seeds 0,1,2 --steps-per-epoch 100 \
  --output-dir output/text-lm-optimizer-fair-comparison
```

AdamW系とAPOLLO系を別groupで実行し、同じtoken budget・seed・モデルの結果を
`optimizer-comparison-report.md`へ集計します。LRはoptimizerごとに異なるため、表の比較は
「各候補設定での実用比較」として読み、同一LRの数学的ablationとは分けて扱います。

LRSFのrefresh方式を診断なしで比較する場合は、AdamW groupへ次の設定を渡せます。
`--lrsf-transport-overlap`を指定すると、refresh時のbasis transport overlapも固定されます。
APOLLO groupには適用されません。
なお、`AdamW-LRSF`のSchedule-Free deltaは常にtransportされます。latent second-momentの
`reset/transport`比較は`AdamW-LRSF-LR`専用であり、plain `AdamW-LRSF`に`reset`を指定すると
検証器がエラーにします。

latent moment policyの速度を比較するときは、診断を付けない専用wrapperを使います。
reset/transportを別JSONへ保存し、scalar diagnostic reduction・周期validation・update norm
計測を行わないため、optimizer step時間とpeak VRAMを比較できます。

```bash
bash verify/launchers/run_text_lm_lrsf_lr_speed_comparison.sh \
  --device cuda --dtype bf16 --rank 16 \
  --seeds 0,1,2 --refresh-interval 50 \
  --transport-overlap 0.99 \
  --output-dir output/text-lm-lrsf-lr-speed-comparison
```

既定policyを決める前にrankとrefresh intervalも横断する場合は、sweep wrapperを使います。
既定でrank=`8,16`、interval=`25,50`、各3 seed、3 epoch×100 stepを実行し、完了済みセルは
スキップします。設定は環境変数で上書きできます。

```bash
bash verify/launchers/run_text_lm_lrsf_lr_reset_sweep.sh
```

例：縮小検証にする場合。

```bash
TEXT_LM_LRSF_LR_RANKS=8,16 \
TEXT_LM_LRSF_LR_INTERVALS=25,50 \
TEXT_LM_TRAIN_TOKENS=49152 \
TEXT_LM_EPOCHS=1 TEXT_LM_STEPS_PER_EPOCH=100 \
bash verify/launchers/run_text_lm_lrsf_lr_reset_sweep.sh
```

```bash
bash verify/launchers/run_text_lm_optimizer_fair_comparison.sh \
  --device cuda --dtype bf16 --rank 16 \
  --adamw-learning-rate 3e-4 \
  --lrsf-refresh-mode hard --lrsf-refresh-interval 50 \
  --lrsf-refresh-mix smoothstep --lrsf-transport-overlap 0.99 \
  --output-dir output/text-lm-lrsf-speed-overlap-099 \
  --force
```

`AdamW-LRSF-LR`のlatent second momentについて、hard refresh時の`reset`と`transport`を
診断付きで比較する場合は、専用wrapperを使います。各JSONにはrefresh eventごとのdelta
transport診断に加え、各stepのlatent moment norm・moment age・periodic validation lossが
保存されます。診断用のscalar reductionと定期validationを含むため、step時間の比較には
使わず、速度は診断なしrunで別に測定してください。

trajectory curvatureやSchedule-Freeのtrain/hidden/eval位置も同時に確認する場合は、低負荷の
paired wrapperを使います。`AdamW-LRSF-LR`の`reset`と`transport`を別JSONへ保存し、既定では
rank=`16`・hard refresh interval=`25`・診断snapshot/validation interval=`20`・3 seed・
1 epoch×100 stepで実行します。実効step数がデータローダーで96になる条件でも、position
curvatureに必要な4 snapshotを確保する設定です。

```bash
bash verify/launchers/run_text_lm_lrsf_lr_trajectory_diagnostics.sh
```

このwrapperも診断用snapshotと周期validationを含むため、step時間の比較には使用しません。

refresh前後のvalidation変化をfrozen controlと比較する場合は、次のwrapperを使います。
`frozen`、`hard-reset`、`hard-transport`を同一条件で実行し、policyごとのJSONを保存します。

```bash
bash verify/launchers/run_text_lm_lrsf_lr_refresh_recovery.sh

python3 -m verify.text_lm_lrsf_lr_refresh_recovery_report \
  output/text-lm-lrsf-lr-refresh-recovery \
  > output/text-lm-lrsf-lr-refresh-recovery.md
```

reportのrecovery proxyは、各refresh eventについて「直前のvalidation lossから、最初の
refresh後validation lossまでの差」を集計します。validation間隔がrefresh間隔より長いため、
通常の学習進行も含む粗い指標であり、因果的な回復時間ではありません。

refresh event直後を細かく観測するscreeningでは、validation/snapshot intervalを5 stepへ
揃えます。1 epochなら実効96 stepとなり、interval=`25`のevent（26, 51, 76）の前後を
観測できます。

```bash
bash verify/launchers/run_text_lm_lrsf_lr_refresh_recovery.sh \
  --epochs 1 --steps-per-epoch 100 \
  --eval-interval 5 --snapshot-interval 5 \
  --max-snapshots 20 \
  --output-dir output/text-lm-lrsf-lr-refresh-recovery-event-aligned

python3 -m verify.text_lm_lrsf_lr_refresh_recovery_report \
  output/text-lm-lrsf-lr-refresh-recovery-event-aligned \
  > output/text-lm-lrsf-lr-refresh-recovery-event-aligned.md
```

```bash
bash verify/launchers/run_text_lm_lrsf_lr_refresh_diagnostics.sh \
  --device cuda --dtype bf16 --rank 16 \
  --seeds 0,1,2 --refresh-interval 25 \
  --transport-overlap 0.99 \
  --output-dir output/text-lm-lrsf-lr-refresh-diagnostics
```

対象は`AdamW-LRSF-LR`だけです。plain `AdamW-LRSF`へ`--lrsf-refresh-state reset`を渡す
比較には使用しないでください。

APOLLO第一候補とAdamW-SFのtrajectoryを、品質比較とは別に診断する場合は、次のwrapperを
使います。rank=`4`・LR=`5e-3`・APOLLO scale=`0.75`・limiter無効・3 seed・実効300 stepを
標準条件（3 epoch × 100 step）とし、更新のturning angle / direction change / roughnessと、stepごとのtraining
lossの離散2階差分を記録します。さらに25 stepごとのvalidation lossも記録するため、
validation系列の2階差分も計算できます。診断用のCPU snapshotと周期的validationが入るため、
step時間は速度比較に使いません。
周期validationではSchedule-Freeのeval/train変換後に学習中parameterとtrain modeを復元するため、
validation測定によるBF16の丸め誤差をtrajectoryへ残しません。

```bash
bash verify/launchers/run_text_lm_optimizer_trajectory_diagnostics.sh \
  --device cuda --dtype bf16 \
  --output output/text-lm-optimizer-trajectory-diagnostics.json

python3 -m verify.text_lm_optimizer_trajectory_diagnostics_report \
  output/text-lm-optimizer-trajectory-diagnostics.json \
  > output/text-lm-optimizer-trajectory-diagnostics.md
```

JSONの`trajectory_curvature`はsnapshot間のeffective updateを対象とし、
`loss_curvature.train_step`はupdate前のper-step training lossを対象とします。
後者はminibatch noiseを含むため、同じbatch order・token budgetで比較します。
`validation_loss_history`はepochごとの値ですが、wrapperの`--eval-interval 25`は別に
`loss_curvature.validation_step`を生成します。品質は診断なしrun、曲率とlossの2階差分は
診断付きrunで分離して解釈してください。

## Text-LM residual approximation diagnostics

Text-LMのrolling trajectory PCAで残差の圧縮候補を比較する場合は、同じ出力JSONを次で
集計します。これはoptimizerへ圧縮を適用しないオフライン診断であり、factorの容量、
再構成updateのcosine・norm比、CPU decode時間を方式別に表示します。

```bash
python3 -m verify.text_lm_residual_approximation_report \
  output/text-lm-residual-compression-long \
  > output/text-lm-residual-approximation-report.md
```

`storage / target BF16`は残差表現単体の比率です。GPU step時間やoptimizer全体のstate
削減率とは分けて評価してください。

診断実行とレポート生成を一度に行う場合は、専用wrapperを使えます。既定値は
AdamW/AdamW-SF/AdamW-LRSF、rank=`8`、3 seed、100 step、snapshot interval=`5`、
rolling window=`9`です。GPU負荷を抑える場合は`--steps-per-epoch 50`や
`--max-tensors 1`を指定してください。
量子化block sizeを比較する場合は`--block-size 64`、`--block-size 128`などを出力先を
分けて実行します。複数の出力先をreportへ渡すと、block size列ごとに分離集計されます。
scale方式は`--scale-mode max_abs`、`--scale-mode percentile_99_9`、
`--scale-mode rms_3sigma`から選択でき、reportのscale mode列で分離集計されます。
moving rolling-PCA基底と固定基底を切り分ける場合は`--record-fixed-basis`を追加します。
固定基底の結果はreportのbasis列で`fixed_basis`として表示されます。これはerror-feedbackの
座標不一致を検証する診断であり、optimizer更新には介入しません。

```bash
bash verify/launchers/run_text_lm_residual_approximation_sweep.sh \
  --device cuda --dtype bf16 \
  --output-dir output/text-lm-residual-approximation-sweep
```

固定基底を含む比較例：

```bash
bash verify/launchers/run_text_lm_residual_approximation_sweep.sh \
  --device cuda --dtype bf16 --record-fixed-basis \
  --steps-per-epoch 100 --max-tensors 1 --block-size 128 \
  --output-dir output/text-lm-residual-fixed-basis
```

`results.json`と`report.md`が出力されます。wrapperの結果は診断専用であり、圧縮された
残差をoptimizerへ適用する実験ではありません。
