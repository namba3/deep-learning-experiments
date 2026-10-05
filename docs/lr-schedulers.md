# Learning-rate schedulers

5つの`train.py`は、共通の`--lr-scheduler`でstep-basedな学習率制御を選択できます。
実装は[`optimizers/lr_scheduler.py`](../optimizers/lr_scheduler.py)に集約されています。

## 選択肢

| 値 | 用途 | 追加パラメータ |
| --- | --- | --- |
| `auto` | 既存互換。CAMEはconstant、それ以外はcosine | なし |
| `constant` | warmup後のLRを維持 | なし |
| `linear` | 最終LRへ線形減衰 | `--min-lr-ratio` |
| `cosine` | cosine annealing | `--min-lr-ratio` |
| `cosine-restarts` | cosine annealing with hard restarts | `--lr-num-cycles` |
| `polynomial` | polynomial decay | `--lr-power`, `--min-lr-ratio` |
| `inverse-sqrt` | inverse-square-root decay | `--min-lr-ratio` |
| `step` | 一定stepごとの段階減衰 | `--lr-step-size`, `--lr-gamma` |
| `multistep` | 複数milestoneで段階減衰 | `--lr-milestones`, `--lr-gamma` |
| `exponential` | 毎stepの指数減衰 | `--lr-gamma` |

全方式で、optimizer stepの前半にwarmupを設定できます。
`--warmup-steps`または`--warmup-ratio`のどちらか一方を指定してください。

```bash
python3 -m cifar10.train \
  --lr-scheduler cosine-restarts \
  --lr-num-cycles 2 \
  --warmup-ratio 0.05 \
  --min-lr-ratio 0.1
```

`mnist`、`cifar10`、`text_lm`はscheduler指定がない場合、従来互換の10% warmupを維持します。
`image_ae`のデフォルトはconstant、`image_gen`のデフォルトもconstantです。
ScheduleFree系optimizerでは外部schedulerを併用しないのがデフォルトで、必要な場合は
`--force-scheduler`を指定します。

validation lossを入力にする`ReduceLROnPlateau`系は、epoch評価とstep更新の契約が異なるため、
この共通step-based APIには含めていません。
