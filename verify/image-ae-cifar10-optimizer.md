# ImageAE optimizer comparison with CIFAR-10

ローカルCIFAR-10の固定subsetを使ったImageAE optimizer比較です。dataset配置とCUDA条件を確認し、出力先を分けてください。

ローカルCIFAR-10の固定subsetで実画像を比較する場合は、次のprobeを使います。
epochごとの固定permutationを全optimizerで共有し、train/validation loss、state量、
optimizer step時間を記録します。

```bash
python3 -m verify.image_ae_cifar10_optimizer_convergence \
  --data-dir cifar10/data --device cpu --dtype fp32 \
  --epochs 3 --batch-size 16 \
  --max-train-samples 512 --max-validation-samples 512 \
  --optimizers CAME,CAME-SF,CAME-LRSF,APOLLO-CAME-LRSF --rank 4 \
  > output/cifar10-imageae-lrsf-cpu.json
```

LRSF rankを比較する場合は`--ranks`を追加します。`backend_counts`でLRSF経路と
CAME/APOLLO fallbackの件数を併記してください。

```bash
python3 -m verify.image_ae_cifar10_optimizer_convergence \
  --data-dir cifar10/data --device cpu --dtype fp32 \
  --epochs 3 --batch-size 16 --max-train-samples 512 \
  --max-validation-samples 512 --rank 4 --ranks 1,4,8,16 \
  --optimizers CAME,CAME-SF,CAME-LRSF,APOLLO-CAME-LRSF \
  > output/cifar10-imageae-lrsf-rank-sweep-cpu.json
```

AdamWを基準にAPOLLOの低rank更新を比較する場合は、専用probeを使います。
初期ImageAE重み、CIFAR-10のsubset、epochごとのpermutation、precision、validation
画像をoptimizer間で共有し、persistent optimizer stateとCUDA peak memoryを分けて記録します。
`APOLLO`系では`--update-proj-gap`でprojection refresh間隔を指定でき、
`--projection-refresh-mode none|hard|smooth`でrefresh方式を選択できます。`smooth`では
`--projection-refresh-window`と`--projection-refresh-mix`も指定できます。
`--freeze-projection`を付けるとrefreshを実質的に停止した対照条件になります。
`--projection-refresh-state transport`を指定すると、hard refresh時に低rank momentを
旧新projectionのoverlapで近似移送します。既定の`reset`は従来動作です。
`--projection-refresh-mix ema`では、smooth window中のPA/PB importance weightを指数平滑します。
decayは既定で`exp(-1/window)`です。したがって、APOLLO側の指定は
`--projection-refresh-mode smooth --projection-refresh-mix ema`です。
`--projection-refresh-mix stochastic`では、smooth windowの進行度をPBを選ぶ確率として使い、
各stepでPA/PBの一方をdecodeします。PA/PBのmoment更新は継続します。
`--orthogonal-refresh-rate`を0より大きくすると、interval refreshとは独立にAPOLLOの
`R_update`を毎step微小回転します。この場合、`--projection-refresh-mode none`と組み合わせると
orthogonal-onlyの比較になります。
`--record-step-metrics`を付けるとstep lossと検出したrefresh eventを保存し、
`--record-update-norms`を追加するとparameter update normの履歴・平均・分散も保存します。
update norm計測はhost copyを伴うため、通常のstep時間・peak memory比較とは分けて実行してください。
`--update-norm-variance-cap V`を指定すると、APOLLOのoptimizer updateについて累積Welford統計を
使った実験的な上側variance capを有効にします。過去の分散やmoment stateは変更せず、大きな新規
updateだけを縮小します。JSONにはcap値と`update_norm_variance_capped_steps`が保存されます。

複数cap値・seedを比較する場合は、次のsweep scriptを使います。baselineの`cap=none`を各seedで
保存し、各capのvalidation loss delta、step倍率、state/peak、capped stepsを
`output/apollo-variance-cap-sweep/apollo-variance-cap-sweep-report.md`へ出力します。

```bash
bash verify/launchers/run_apollo_variance_cap_sweep.sh \
  --device cuda --dtype bf16 \
  --caps none,0.0001,0.001,0.01 \
  --seeds 0,1,2 \
  --output-dir output/apollo-variance-cap-sweep \
  --record-update-norms --force
```

refresh eventには`projection_change_max_abs`も記録され、projection seedだけでなく実際の
projection tensorが変化したかを確認できます。

複数runのJSONを同じ表へまとめる場合は、次を使います。ImageAE probeとoptimizer runtime
benchmarkの両方のschemaに対応し、CUDA未実行時のpeak列は`-`として残します。
このコマンドは既存JSONを集計するだけなので、先に下記のprobeを各方式で実行してください。
ファイルが存在しない場合は、対応するprobeの実行を促すエラーを返します。

方式別probeの実行と集計をまとめて行う場合は、次のスクリプトを使えます。デフォルトでは
5方式を順番に実行し、`output/apollo-refresh-*-seed0.json`とMarkdown集計を保存します。
標準出力・標準エラーの進捗ログも`output/apollo-refresh-run-seed0.log`へ保存します。
同じ出力を上書きする場合だけ`--force`を付けてください。

```bash
bash verify/launchers/run_apollo_refresh_experiments.sh \
  --device cuda --dtype bf16 --seed 0
```

orthogonalの回転率と方向を複数条件で比較する場合は、次のsweep scriptを使います。
各cellを`direction/rate/seed`配下へ保存し、noneとの差、step倍率、update norm variance倍率を
`apollo-orthogonal-sweep-report.md`へ集計します。GPU環境での本実験を想定しています。

```bash
bash verify/launchers/run_apollo_orthogonal_sweep.sh \
  --device cuda --dtype bf16 \
  --rates 0.001,0.005,0.01,0.02,0.05 \
  --directions random,loss_directed \
  --seeds 0,1,2 \
  --output-dir output/apollo-orthogonal-sweep
```

CPUで短い実行契約だけを確認する例:

```bash
bash verify/launchers/run_apollo_refresh_experiments.sh \
  --device cpu --dtype fp32 --epochs 1 \
  --max-train-samples 16 --max-validation-samples 16 \
  --latent-channels 1 --bottleneck-channels 16 --downsample-stages 2 \
  --modes none,smooth-ema
```

```bash
python3 -m verify.apollo_refresh_report \
  output/apollo-refresh-none-seed0.json \
  output/apollo-refresh-hard-seed0.json \
  output/apollo-refresh-smooth-ema-seed0.json \
  output/apollo-refresh-smooth-stochastic-seed0.json \
  output/apollo-refresh-orthogonal-seed0.json
```

まずは小subsetで実行契約を確認します。

```bash
python3 -m verify.image_ae_cifar10_adamw_apollo \
  --data-dir cifar10/data --device cuda --dtype bf16 \
  --epochs 5 --batch-size 8 \
  --max-train-samples 512 --max-validation-samples 512 \
  --optimizers AdamW,CAME,APOLLO,APOLLO-CAME,APOLLO-Mini \
  --rank 4 --learning-rate 5e-4 --update-proj-gap 200 \
  --projection-refresh-state transport \
  > output/cifar10-imageae-adamw-apollo-subset.json
```

同じ条件で従来のmoment破棄動作（`reset`）を保存する場合は、
`--projection-refresh-state transport`を省略するか、次のように明示します。

```bash
python3 -m verify.image_ae_cifar10_adamw_apollo \
  --data-dir cifar10/data --device cuda --dtype bf16 \
  --epochs 5 --batch-size 8 \
  --max-train-samples 512 --max-validation-samples 512 \
  --optimizers AdamW,CAME,APOLLO,APOLLO-CAME,APOLLO-Mini \
  --rank 4 --learning-rate 5e-4 --update-proj-gap 200 \
  --projection-refresh-state reset \
  > output/cifar10-imageae-adamw-apollo-reset-subset.json
```

refreshを固定projection対照と比較する場合は、同じseed・subset・条件で次を別実行します。
結果の因果解釈には、AdamWとの単純な品質比較だけでなく、refresh前後のloss spikeと
recoveryを追加記録する必要があります。Flat Minimaや構造化探索ノイズは、これらの対照が
揃うまで仮説として扱います。

```bash
python3 -m verify.image_ae_cifar10_adamw_apollo \
  --data-dir cifar10/data --device cuda --dtype bf16 \
  --epochs 5 --batch-size 8 \
  --max-train-samples 512 --max-validation-samples 512 \
  --optimizers APOLLO,APOLLO-CAME,APOLLO-Mini \
  --rank 4 --learning-rate 5e-4 --freeze-projection \
  --record-step-metrics --record-update-norms \
  > output/cifar10-imageae-apollo-frozen-subset.json
```

```bash
python3 -m verify.image_ae_optimizer_convergence \
  --device cuda --dtype bf16 --warmup 10 --steps 50 \
  --optimizers CAME,CAME-SF,CAME-LRSF,APOLLO-CAME,APOLLO-CAME-LRSF --rank 4 \
  --batch-size 8 --image-size 32 \
  > output/image-ae-came-lrsf.json
```

weight decayの小型sweep例：

```bash
python3 -m verify.image_ae_optimizer_convergence \
  --device cpu --dtype fp32 --warmup 5 --steps 50 \
  --optimizers CAME,CAME-SF,CAME-LRSF --rank 4 \
  --weight-decays 0,1e-3,1e-2 \
  > output/came-lrsf-weight-decay.json
```

比較では、`persistent_state_bytes`、`backend_counts.lrsf`、loss trajectory、
optimizer step時間を確認してください。CAME-SFはfull hidden deltaを持つoracle、
CAME-LRSFはCAMEのstateを保持する近似baselineです。APOLLO-CAME-LRSFは低rank update
stateにLRSF deltaを追加します。CUDAが使えない場合は`--device cpu --dtype fp32`でshape・数値の
確認はできますが、GPU peak memoryと実運用速度の結論には使いません。
