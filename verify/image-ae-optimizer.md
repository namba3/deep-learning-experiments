# ImageAE optimizer convergence probe

ImageAEの固定入力optimizer probeとLRSFのGPU validation手順です。CIFAR-10実画像を使う比較は[別ガイド](image-ae-cifar10-optimizer.md)を参照してください。

## ImageAE optimizer convergence probe

`image_gen`より軽い実モデル比較として、実際の`image_ae.train.ImageAE`の
forward/backwardと再構成MSEを固定画像で測定できます。dataset、VAE、checkpointを読み込まず、
畳み込みautoencoderのparameter shapeを使ってoptimizerの収束、state量、CUDA peak memoryを
比較します。

Schedule-Freeの full oracle `CAME-SF`、近似 baseline `CAME-LRSF`、低メモリ候補
`APOLLO-CAME-LRSF`も比較できます。LRSF系は既定では固定/frozen random projection
（`refresh-mode=none`）ですが、
`--refresh-mode hard|smooth|shadow`で`R_delta`のrefreshを比較できます。`shadow`では
active branchの裏でlow-rank delta/projectionを毎step育成し、intervalでshadowを昇格します。
full optimizer stateは共有しますが、low-rank stateは二重になります。`smooth`ではPA/PB
double-bufferとtransportを使い、`--refresh-interval`、`--refresh-window`、
`--refresh-mix linear|smoothstep|stochastic|ema`で切り替えを指定します。JSONの`refresh_events`には同一batchの
更新前後lossと次stepの観測lossが入り、spike/recoveryの確認に使えます。APOLLOの
`R_update`側のrefreshは別設定です。
`--refresh-mix stochastic`を指定すると、smooth refreshのwindow中に各stepでPA/PBの
一方を選択します。PBの選択確率はwindowの進行度に対応し、double-bufferの両方は
EMA更新を継続します。
`--refresh-mix ema`ではwindow中のPA/PB importance weightを指数平滑します。decayは
既定で`exp(-1/window)`です。
`--orthogonal-refresh-rate`を指定すると、interval refreshとは独立にLRSFの`R_delta`
projectionを毎step tangent-space上で微小回転します。回転後は低ランクdeltaを新しい
直交基底へtransportします。`0`（既定）は無効で、smooth refreshを併用する場合は
PA/PBの両方を回転します。
`--orthogonal-refresh-direction loss_lowering`では、現在のgradientとhidden deltaから
decoded deltaの一次近似lossを減らす方向へ回転します。実損失を追加評価するものではなく、
LRSFの`R_delta`専用です。APOLLO本体の`R_update`には指定しないでください。
`--record-step-metrics`を併用すると、各rotation eventに回転前の`D`を固定した
`<G, D R^T>`（wide shapeでは`<G, R^T D>`）の回転前後値と低下量が記録されます。
併せてoptimizer step完了後のproxyも記録されますが、これはdelta更新を含むため、
rotation単体の判定には固定Dの低下量を使います。
interval refreshと同じstepはbasis交換の影響を分離するため、proxy低下量の集計から除外します。
`--rank` はLRSFのdelta rankにも使われ、1D・full-rank相当の小行列は自動的にCAMEへ
fallbackします。
weight decayを比較する場合は`--weight-decays`を使います。`0`はweight decayなし、
LRSF経路ではSchedule-Freeのeffective update方向へ、CAME fallback経路では通常CAMEの
decoupled decayとして適用されます。

LRSFのGPU比較を同じ条件で再実行する場合は、専用スクリプトを使います。既定では
本番ImageAEに近いlatent=`16`、bottleneck=`256`、downsample stages=`3`、
CUDA/BF16、CIFAR-10各512枚、5 epochで、fixed/frozen・smoothstep・EMAを比較します。
`fixed`は`refresh-mode=none`であり、hard refreshとは異なります。

```bash
verify/launchers/run_lrsf_gpu_validation.sh
```

全方式を比較する場合:

```bash
verify/launchers/run_lrsf_gpu_validation.sh \
  --modes fixed,hard,smoothstep,ema,stochastic,orthogonal
```

seed=0,1,2を同じ方式で順次比較する場合は`--seeds`を指定します。各seedの結果は
別JSONへ保存されるため、既存のseed=0結果を残したまま追加できます。

```bash
verify/launchers/run_lrsf_gpu_validation.sh \
  --seeds 0,1,2 \
  --modes fixed,smoothstep,ema,orthogonal
```

rankのGPU sweepは`--ranks`で指定します。LRSF caseだけがrankごとに展開され、
`CAME`と`CAME-SF`は一度だけ実行されます。

```bash
verify/launchers/run_lrsf_gpu_validation.sh \
  --ranks 1,4,8,16 \
  --modes fixed
```

rank=4とrank=8でrefresh方式も比較する場合は、次の組み合わせを使います。
既存のrank sweep結果と分けるため、出力先を変更します。

```bash
verify/launchers/run_lrsf_gpu_validation.sh \
  --seeds 0,1,2 \
  --ranks 4,8 \
  --modes fixed,smoothstep,ema,orthogonal \
  --output-dir output/lrsf-rank-refresh
```

実行後は、ディレクトリをそのまま指定してrank・refresh方式・optimizerごとの
平均とseed間標準偏差を集計できます。

```bash
python3 -m verify.lrsf_gpu_report output/lrsf-rank-refresh \
  > output/lrsf-rank-refresh/report.md
```

結果は`output/lrsf-gpu-<mode>-seed<N>.json`、GPU名・driver・VRAMは
`output/lrsf-gpu-info-seed<N>.txt`へ保存されます。CUDAがない場合、既定の`--device cuda`は
CPUへ黙ってfallbackせず失敗します。
各caseにはpersistent stateの現在値とrefresh中を含むピーク値、CUDA時の
`peak_allocated_bytes`、`peak_reserved_bytes`、`peak_delta_allocated_bytes`が記録されます。
診断用に`--record-step-metrics`を付けると、orthogonalを含むrefresh eventの同一batch loss、
次step loss、pre-refresh lossへ戻るまでのrecovery stepsを保存できます。
`--record-update-norms`を追加すると、parameter update normの履歴・平均・分散を保存できます。
後者は更新前parameterのhost copyを伴うため、通常の性能・peak VRAM比較とは分けて実行してください。

loss-directedのsignalと回転率を比較する例:

```bash
verify/launchers/run_lrsf_gpu_validation.sh \
  --seeds 0,1,2 \
  --modes fixed,orthogonal \
  --orthogonal-rate 0.005 \
  --orthogonal-direction loss_directed \
  --orthogonal-signal effective_update \
  --record-step-metrics --record-update-norms \
  --output-dir output/lrsf-loss-directed-update-r005

python3 -m verify.lrsf_gpu_report output/lrsf-loss-directed-update-r005 \
  > output/lrsf-loss-directed-update-r005/report.md
```

集計表には、orthogonal refresh回数、update norm平均・分散、refresh eventの平均recovery
stepsも表示されます。`gradient` signalへ変更する場合は`--orthogonal-signal gradient`を指定します。

```bash
python3 -m verify.image_ae_optimizer_convergence \
  --device cuda --dtype bf16 --warmup 5 --steps 200 \
  --batch-size 8 --image-size 32 \
  --rank 8 --learning-rate 2e-4 --scale 1.0 \
  > output/image-ae-optimizer-convergence.json
```

limiter無効候補を測定する場合は、同じコマンドに
`--disable-norm-growth-limiter`を追加します。既定モデルは
`residual_conv_ffn` encoder/decoder、`latent_channels=8`、`bottleneck_channels=64`、
`downsample_stages=2`です。これはimage_aeの実forward/backwardを使う軽量probeであり、
CIFAR-10やFlickr30kの学習品質を直接代替するものではありません。

limiterを完全に無効化せず、許容するnorm成長率だけを比較する場合は、
`--norm-growth-rates`を使います。値はoptimizerの契約上`1.0`より大きくします。
この軸はAPOLLO、APOLLO-CAME、APOLLOMiniに適用され、CAMEでは重複caseを作りません。

```bash
python3 -m verify.image_ae_optimizer_convergence \
  --device cuda --dtype bf16 --warmup 5 --steps 200 \
  --optimizers APOLLO,APOLLO-CAME,APOLLOMini \
  --ranks 1,8 --learning-rates 1e-3,3e-3 \
  --norm-growth-rates 1.01,1.05,1.1 \
  --batch-size 8 --image-size 32 \
  --scale 1.0 > output/image-ae-optimizer-convergence-growth.json
```

各APOLLO系caseには`norm_growth_rate`、トップレベルには指定した
`norm_growth_rates`が記録されます。`--norm-growth-rate`はsweepを使わない場合の
単一値を変更します。

現行ImageAEに近い構成（latent=`16`、bottleneck=`256`、downsample stages=`3`）で
候補を再検証する場合は、まずlimiter無効条件を次で測定します。

```bash
python3 -m verify.image_ae_optimizer_convergence \
  --device cuda --dtype bf16 --warmup 5 --steps 200 \
  --batch-size 8 --image-size 32 \
  --latent-channels 16 --bottleneck-channels 256 --downsample-stages 3 \
  --optimizers CAME,APOLLO,APOLLO-CAME,APOLLOMini \
  --rank 4 --learning-rate 5e-4 --scale 1.0 \
  --disable-norm-growth-limiter \
  > output/image-ae-current-config-limiter-off.json
```

同じ条件から`--disable-norm-growth-limiter`を外した結果も取得すると、候補設定の
収束差とstate・step時間・peak memoryを直接比較できます。
