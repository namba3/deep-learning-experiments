# Optimizer design and selection

optimizerのローカル実装、CLI registry、parameter role別の選択方法をまとめます。実装の接続点は[`../optimizers/README.md`](../optimizers/README.md)、現行の選択肢は`optimizers/factory.py`を正とします。

## 推奨の入口

通常の実験では、まず次の4つから選びます。

| CLI名 | 用途 |
| --- | --- |
| `AdamW` | 汎用的な基準。full-size momentはparameter dtypeで保持する |
| `AdamW-SF` | full hidden stateを持つSchedule-Free AdamW oracle |
| `AdamW-LRSF` | AdamWのSchedule-Free hidden deltaだけを低rank化する実験用variant |
| `AdamW-SF-LR` | Schedule-Freeの`z`をfull-sizeで維持し、matrixの`exp_avg_sq`だけをlatent低rank化するablation |
| `AdamW-LRSF-LR` | hidden driftとlatent preconditionerを同一低rank projectionへ統合するprototype |
| `AdamW-LR-EMA` | Schedule-Freeを使わず、低rank projected gradientの急峻EMAを直接更新へ使うablation |
| `AdamW-LR-EMA-Conf` | 低rank gradient EMAとinnovation varianceによるconfidence normalizationのablation |
| `AdamW-LR-EMA-Conf-LRSF` | confidence stateとSchedule-Free low-rank deltaを別basisで保持する統合prototype |
| `CAME` | factorized second momentとconfidence-guided update |
| `CAME-SF` | full hidden deltaを持つSchedule-Free CAME oracle |
| `CAME-LRSF` | CAMEに固定random projection上のSchedule-Free差分を追加する実験用variant |
| `APOLLO` | matrix gradientを低rank空間で処理するAdam系 |
| `APOLLO-Conf` | APOLLO型のlatent stateにinnovation variance confidenceを適用する比較variant |
| `APOLLO-CAME` | APOLLOの低rank pathにCAMEのconfidence制御を組み合わせる |

`image_gen`では既定のmain optimizerは`APOLLO`です。AutoSchedule対応variantでは、上記の名前に`--auto-schedule`を追加します。`CAME-LRSF`は現在AutoSchedule対象外です。例えば:

```bash
python3 image_gen/train.py \
  --vae-model Qwen/Qwen-Image \
  --optimizer APOLLO \
  --auto-schedule
```

`image_gen`はLinearとConv/ConvTransposeを別optimizerに分けられます。指定しないroleは`--optimizer`を共有します。

```bash
python3 image_gen/train.py \
  --vae-model Qwen/Qwen-Image \
  --optimizer APOLLO \
  --linear-optimizer Muon \
  --conv-optimizer AdamW
```

## Registryの分類

`optimizers.factory`では選択肢を次の3群に分類しています。

### Core choices

`AdamW`、`CAME`、`APOLLO`、`APOLLO-CAME`。小さなtrain scriptでも同じfactory APIから選択できます。`CAME-SF`、`CAME-LRSF`、`APOLLO-CAME-LRSF`はImageAEのCLIで明示的に有効化された実験用choiceです。

### Legacy compatibility

`RAdamSF`、`AdamWSF`。旧実験を読み戻すための互換名です。新しい比較では`AdamW-SF`と
`AdamW-LRSF`を使います。

### Experimental choices

`AdamW-SF`、`AdamW-LRSF`、`AdamW-SF-LR`、`AdamW-LRSF-LR`、`AdamW-LR-EMA-Conf-LRSF`、`APOLLO-Conf`、`CAME-SF`、`CAME-LRSF`、`APOLLO-CAME-LRSF`、`CAME-AutoSchedule`、`AdamW-AutoSchedule`、`APOLLO-AutoSchedule`、`APOLLO-CAME-AutoSchedule`、`APOLLO-Mini`、`APOLLO-AdamW`、`APOLLO-AdamW-AutoSchedule`、`APOLLO-Lion`、`RotAPOLLO`、`DualRotAPOLLO`、`Lion`、`Muon`、`SOAP`、`NorMuon`、`AdaMuon`。

experimental choiceは主に`image_gen`の互換性・比較実験用です。小さな分類・言語実験では、parserが許可するcore choicesを使用してください。

## 更新の特徴

- **AdamW**: decoupled weight decayと一次・二次momentを使う基準optimizer。`AdamWFP32State`は
  互換名を維持しているが、full-size stateはparameterのstorage dtypeで保持する。
- **AdamW-SF**: full hidden `z`とfull second-moment stateを持つSchedule-Free AdamW oracleです。
- **AdamW-LRSF**: AdamWのfull second-moment stateを維持し、hidden Schedule-Free deltaだけを
  固定random projection上の低rank係数で近似します。1Dとfull-rank相当の小行列はfull
  Schedule-Freeへfallbackします。text LMの長期Transformer比較では`hard` refreshを標準条件とし、
  refreshなしのfrozen baselineは`mode="none"`を明示して使います。
- **AdamW-SF-LR**: full Schedule-Freeの`z`を維持し、matrix parameterのgradientだけを
  固定直交basisへ射影してlatent second momentを更新します。normalized latent updateを
  decodeしてから、`AdamW-SF`と同じSchedule-Free補間・weight decay順序で適用します。
  1Dとfull-rank相当の小行列はexact full-state pathへfallbackします。これは`AdamW-LRSF`
  と組み合わせる前のpreconditioner単独ablationで、現時点ではmatrix経路にTriton融合を
  実装していません。
- **AdamW-LR-EMA**: matrix gradientを固定orthonormal projectionへ写像し、latent gradientを
  EMAします。bias correction後にdecodeした勾配でdecoupled weight decay付きの直接更新を行うため、
  `z`、`exp_avg_sq`、Schedule-Free weightingを持ちません。既定の`ema_beta=0.9`は急峻な
  応答を比較する設定で、`0.5/0.9/0.99`の感度比較を行います。`projection_scale="norm"`
  （既定）はrank-r射影で失われる期待更新ノルムを補正しますが、捨てられた勾配成分を
  復元するものではありません。1Dとfull-rank相当の小行列はparameter dtypeのfull gradient
  EMAへfallbackします。
- **AdamW-LR-EMA-Conf**: `AdamW-LR-EMA`のlatent gradient EMAに加え、
  `r_t = g_t^lr - m_{t-1}`の二乗を指数移動平均したinnovation variance `c_t`を保持します。
  bias correction後の`m_hat / sqrt(c_hat + alpha*m_hat²)`をdecodeして更新するため、
  低rank空間内で方向の安定性に応じた正規化を行えます。`alpha`はvariance collapse時の
  分母floorです。既定値は`ema_beta=0.9`、`confidence_beta=0.99`、`alpha=1e-3`です。
- **AdamW-LR-EMA-Conf-LRSF**: confidence用の低rank `m/c`とSchedule-Freeの低rank deltaを
  別projectionで保持します。confidence projectionは固定し、delta projectionだけが既存
  LRSFのrefresh policyに従います。1Dとrank飽和行列はfull Schedule-Freeへfallbackする初期
  prototypeであり、confidence projectionのrefreshやbasis共有は未実装です。
- **AdamW-LRSF-LR**: Schedule-Freeのhidden driftとlatent second momentを同じ固定projection
  の係数として保持する統合prototypeです。`z`、full `exp_avg_sq`、full `sf_delta`を
  持たないため、最終目標に近いstate構成です。hard refreshではdeltaとlatent second
  momentをbasis overlapの二乗で近似transportします。これはfull covarianceのtransportでは
  ありません。`projection_refresh_state="reset"`ではmomentを破棄し、局所bias correctionを
  再開できます。`mode="shadow"`ではdriftとlatent momentを含む二つのbranchを並行更新し、
  intervalごとにshadowをactiveへ昇格します。smoothとorthogonal transportは未対応です。
  1Dと小行列はexact full Schedule-Freeへfallbackします。
- **CAME**: second momentをrow/column factorへ分解し、confidence統計を使って更新を調整します。
- **Muon系**: momentum gradientをNewton-Schulz反復で直交化します。matrix weight向けで、1D parameterにはfallbackが必要です。
- **SOAP**: Shampoo系のpreconditionerをbasis空間で扱います。matrixの相関を利用する代わりにbasis更新のコストがあります。
- **Lion**: momentum方向のsignで更新し、二次momentを持たないためstateが小さくなります。
- **APOLLO**: full-rank gradientを低rank projectionへ写像し、低rank空間に適応stateを保持してからfull-rank updateへ戻します。
- **APOLLO-Conf**: APOLLOと同じ`projection`、latent `exp_avg`、latent `exp_avg_sq`のstate構成を
  維持し、`exp_avg_sq`を投影勾配のinnovation varianceとして更新します。正規化stateから
  channel-wise scalingを作り、APOLLOと同じくfull gradientへ適用します。Schedule-Free deltaは
  持たないため、`AdamW-LR-EMA-Conf-LRSF`との比較で「APOLLO型state」と「LRSF delta」を
  分離できます。
- **APOLLO-Lion**: APOLLOの低rank空間にLionのsign updateを適用します。
- **APOLLO-CAME**: 低rank空間のCAME処理とresidual/confidence統計を使います。
- **APOLLO refresh state**: `projection_refresh`は現行APOLLOでは`none`、`hard`、`smooth`を共有Policyで指定できます。hard refresh時の低rank momentは既定で`reset`し、`projection_refresh_state="transport"`では旧新projectionのoverlapで一次・二次統計を近似移送します。smoothではPA/PB相当の低rank moment double-bufferを使い、`mix="linear"`/`"smoothstep"`はscalingをblendし、`mix="ema"`はimportance weightを指数平滑します。`mix="stochastic"`では同じweightをPB選択確率として使い、PA/PBの一方をstepごとに選びます。`orthogonal_refresh={"rate": rho, "seed": seed}`を併用すると、intervalなしの毎step微小回転とmoment transportを行います。smooth中はpersistent stateが増えるため、低メモリ性の評価とは分けます。
- **CAME-LRSF**: CAMEのfull update directionとstateを維持し、Schedule-Freeのhidden deltaだけを固定random projection上の低rank係数で近似します。1Dとfull-rank相当の小行列はCAMEへfallbackします。追加stateがあるため、低メモリ版ではなくSchedule-Free効果の比較用です。
- **CAME-SF**: CAME-LRSFのfull-size hidden delta oracleです。Schedule-Free効果を測る基準であり、メモリ削減目的ではありません。
- **APOLLO-CAME-LRSF**: APOLLO-CAMEの`R_update`・低rank CAME stateを維持し、別の`R_delta/H`でSchedule-Free差分を近似します。`delta_refresh`にnone/hard/smooth(PA/PB)とlinear/smoothstep/ema/stochastic mixを指定できます。APOLLOの`R_update` refreshとorthogonal refreshは`R_delta`側と独立です。

LRSFと低rank optimizer stateに関する過去の診断・prototype・比較結果は[Schedule-Free研究記録索引](schedule-free-methods-summary.md)から参照してください。記録されたvariantや「next」は当時の履歴snapshotであり、現在の推奨ではありません。現行backendとstate契約は本書とコードを参照してください。

## State dtype contract

parameterと同じ要素数・shapeのpersistent stateは、parameterのstorage dtypeで保持する。
対象はAdamWの`exp_avg`/`exp_avg_sq`、Schedule-Freeの`z`/`exp_avg_sq`、CAMEの
unfactored moment、Muon/Lion/SOAPのfull-size momentである。旧checkpointを読み込んだ場合も、
各parameterの最初のstepでこのdtypeへ変換する。

一方、APOLLO/CAME-LRSF/AdamW-LRSFの低rank moment・delta・projection、CAMEのrow/column
factor、preconditioner basis、診断用scalarはFP32を維持する。これらはFP32 gradientや
projection/QR/eigendecompositionと直接演算するため、parameter dtypeへ揃えてもcastを減らせず、
むしろ混在dtype変換または蓄積精度低下を招く。除算や基底空間GEMMなどのtemporary arithmeticも
FP32を基本とする。この契約変更後のloss・step時間・peak VRAMは、既存のFP32-state測定と
混ぜず、CUDA/BF16で再計測する。

### 現行dtype契約でのstate容量比較

現行実装の容量感を揃えて確認するため、TinyStories/Qwenの比較probeと同じ
`naive` TinyTextLM（`parameter_numel=181,566,592`、parameter=`BF16`、
rank=`4`）について、persistent optimizer stateを見積もった。parameter本体は
`363,133,184` bytes（約`0.338 GiB`）である。下表はparameter、gradient、activation、
一時FP32 tensor、CUDA allocatorの`peak allocated/reserved`を含まない。

| optimizer | 主なpersistent state | state bytes | state GiB | state / parameter | 現行実装での位置づけ |
| --- | --- | ---: | ---: | ---: | --- |
| AdamW | `exp_avg` BF16 + `exp_avg_sq` BF16 | 726,266,368 | 0.676 | 2.000x | full-size AdamW baseline |
| AdamW-SF | `z` BF16 + `exp_avg_sq` BF16 | 726,266,368 | 0.676 | 2.000x | full Schedule-Free oracle |
| AdamW-LRSF (rank=4) | `exp_avg_sq` BF16 + low-rank delta/projection FP32 | 369,701,840 | 0.344 | 1.018x | full second moment + low-rank drift |
| CAME | `exp_avg` BF16 + row/column factors FP32 | 366,468,592 | 0.341 | 1.009x | factorized CAME baseline |
| CAME-SF | CAME state + full `sf_delta` BF16 | 729,601,776 | 0.679 | 2.009x | full Schedule-Free CAME oracle |
| CAME-LRSF (rank=4) | CAME state + low-rank delta/projection FP32 | 372,936,384 | 0.347 | 1.027x | CAME + low-rank drift |
| APOLLO-CAME (rank=4) | low-rank CAME moments/factors, FP32 | 20,856,192 | 0.019 | 0.057x | low-rank APOLLO-CAME |
| APOLLO-CAME-LRSF (rank=4) | APOLLO-CAME + low-rank drift, FP32 | 27,323,724 | 0.025 | 0.075x | low-rank update + low-rank drift |

これは新dtype契約に基づく容量値であり、既存のFP32-stateで取得した過去のJSONを
再実行した測定値ではない。`AdamW`/`AdamW-SF`は従来のFP32 stateから約`1.353 GiB`
減少する一方、`AdamW-LRSF`は低rank stateがFP32のままなので、削減後の主な容量は
full-size `exp_avg_sq`で決まる。低rank stateをparameter dtypeへ変更していないのは、
FP32 gradient・projection・GEMMへ直接接続しており、単純なBF16化ではcastを減らせず、
蓄積精度だけを下げる可能性があるためである。

`shadow` double-buffer refreshを使う場合は、上表に含まれる通常のactive stateへ
低rankのshadow projection/deltaが追加される。したがってrefresh方式を比較するときは、
通常時の`persistent_state_bytes`だけでなく`peak_persistent_state_bytes`、一時tensor、
`peak allocated/reserved`を別々に記録する。

APOLLO系の直接コンストラクタも`rank=8`を既定値とします。`APOLLOMini`だけは
rank-one optimizerとして`rank=1`に固定されます。factoryと`image_gen` CLIの
`--apollo-rank`も通常のAPOLLO系では既定値8です。

APOLLOの低メモリ性に加え、低rank projectionとProjection Refreshが学習探索へ影響する
可能性については、設計目標・実測・未検証仮説を分離して
[`apollo-experiment-records.md`](apollo-experiment-records.md)に整理しています。Flat Minima、構造化
ノイズ、loss spikeからの回復は現時点で証明済みの性質ではありません。

既存設定との互換性のため、`update_proj_gap`はhard refreshのintervalへ変換されます。
APOLLO本体のrefresh PolicyはSchedule-Free/LRSFの`R_delta` Policyとは別に評価し、
`R_update`のmoment transportによる収束・loss spike・recoveryを比較します。

## APOLLO系のstateと計算量

matrix parameterを`m x n`、projection rankを`r`、`M=max(m,n)`、`Q=min(m,n)`とすると、低rank stateの概算は次です。

```text
L = M * r
APOLLO       : 約 2L（first/second moment）
APOLLO-Lion  : 約 L（momentum）
APOLLO-CAME  : 約 3L + 2M + 2r（low-rank workspacesを含む）
APOLLO-CAME-LRSF: APOLLO-CAME + 約 L + Qr（R_delta/Hを含む）
```

これはoptimizer stateの概算であり、parameter、gradient、projection、一時FP32 tensor、activationを含むpeak VRAMではありません。実測では、次を分けて記録します。

- 永続optimizer stateの要素数とdtype
- projectionやBF16→FP32変換を含むstep中の一時tensor
- `peak allocated`と`peak reserved`
- forward/backwardとoptimizer stepの時間
- PyTorch/reference、Triton、fallback間の値・勾配・state差

APOLLO系の低rank projection自体は概ね`O(mnr)`です。低rank updateだけを速くしても、projectionやfull-rank scalingが支配的な場合があります。

## APOLLOのfallback policy

APOLLOはmatrix parameterを低rank pathへ送り、1D parameterは既定でAdamW-SFへ
fallbackします。matrix parameterも既定で`auto-sf` state-size比較を行い、推定stateが
APOLLOより小さいmatrixだけAdamW-SFへ送ります。`fallback="came"`という従来の文字列は
1Dだけを制御し、matrixは従来どおりAPOLLOに残します。backendの選択結果はstateに保存されます。
fallback policyは次のcheckpoint-safe設定です。

```python
fallback = {
    "1d": "adamw-sf",
    "small_matrix": "auto-sf",
    "state_margin": 1.0,
    "min_savings_bytes": 0,
}
```

`small_matrix`は`apollo`、`came`、`auto`、`adamw-sf`、`auto-sf`を選べます。
`auto-sf`はparameterごとにshapeと有効rankから永続optimizer stateを見積もり、
AdamW-SFの方が小さい場合にfallbackします。`auto`は同様にCAMEと比較します。
選択結果はstateの`backend`へ保存され、学習途中では切り替えません。
`APOLLO-CAME-LRSF`は現在AdamW-SF fallbackのstate machineに未対応のため、factory経由では
従来の1D CAME・matrix `auto` policyを維持します。

小さい2D matrixでは、full CAMEの概算は
`mn + 2(m+n) + 1`、APOLLOはprojectionを含めて
`2*max(m,n)*r + min(m,n)*r + 1`です。Convなど3D以上のparameterでは、
CAMEは元の最後の2軸、APOLLOはaxis 0と残りをflattenしたshapeを使うため、
単純なnumel閾値ではなく実shapeから比較します。

明示した1D CAME fallbackはunfactored CAME-like updateです。matrix fallbackの
`came`はこれを使わず、row/column second momentとresidual confidenceを持つ
full CAME pathを使います。full CAME fallbackではAPOLLO固有の`scale`と
norm-growth limiterは適用せず、CAME本体と同じ更新・state構造にします。
weight decayは通常どおり適用されます。

### APOLLOのAdamW-SF fallback

Mini-ImageNetのAdamW-SF比較はseed 47–50の16条件を完了し、AdamW-SFはAdamWとの全16組の
対応比較でtest top-1を上回った。CPUで混成更新、train/eval遷移、復元後の更新をstandalone
AdamW-SFと照合済み。Mini-ImageNetではCUDA/BF16の混成条件も評価済み。
集計値は本文に記録し、ローカル生成レポート`mini_imagenet_gqa/output/bucketed/adamw-vs-sf-ada-seeds47-50/optimizer_comparison.md`は公開ツリーに含めていない。AdamWとAdamW-SFではLR policyも異なるため、これは実用的なoptimizer-plus-schedule
比較である。

APOLLOでは`fallback={"1d": "adamw-sf"}`で1D parameterをAdamW-SFに、
`fallback={"small_matrix": "adamw-sf"}`で2D以上のmatrix parameterをすべてAdamW-SFに委譲できます。
`small_matrix="auto-sf"`は推定persistent stateを比較し、AdamW-SFの`z`と
`exp_avg_sq`の合計bytesがAPOLLOより`state_margin`と`min_savings_bytes`の条件を満たすmatrix
だけを委譲します。これは現在の既定policyです。

既定policyでは1D parameterがAdamW-SF、matrix parameterがstate-sizeによる`auto-sf`です。
CAME比較へ戻す場合は次のoptionsを指定します。

```bash
python3 -m image_gen.train --optimizer APOLLO \
  --apollo-fallback came --apollo-matrix-fallback auto
```

混成optimizerはfallback対象parameterにだけSchedule-Freeの更新とtrain/eval変換を適用し、
APOLLO matrix parameterのlow-rank更新を維持する。group単位の`schedule-free` clockと重み和は
optimizer state dictに含め、parameterごとの`z`/`exp_avg_sq`はparameter dtypeで保持する。
AdamW-SF fallbackはAPOLLOのscale/limiter更新を通らず、Schedule-Free AdamWのparameter更新を
直接適用する。状態量はfallback parameterごとに約2 full-size tensorsとなるため、small-matrix
自動選択ではpersistent state量を比較する。

CPUで次の数値・state契約は確認済み。

- fallback parameterの更新値がstandalone `AdamWScheduleFree`と一致する。
- `eval()`/`train()`往復後もfallback weightが一致し、APOLLO low-rank parameterが変換されない。
- 混成optimizer state dictを保存・復元した後、standalone optimizerと次stepが一致する。

CUDA/BF16混成runの結果は[Mini-ImageNet GQA results](mini-imagenet-gqa-results.md)に記録しています。初期matrixはtest top-1が5〜6%付近に留まり、optimizer/update診断のため停止しました。この結果は当該protocolの記録であり、一般的な品質評価ではありません。既定値と既存checkpointの動作は変更していません。条件固定の再現launcherは[archive](../mini_imagenet_gqa/experiments/archive/run_ada_scale_apollo_warmup_seeds47_50.sh)にあります。

## AutoSchedule

AutoScheduleはparameterごとの統計を集約し、parameter group単位でlearning-rate multiplier/capを更新します。主な制御値は次です。

- target update ratio
- EMA beta
- trust alpha
- min/max factor
- 1 stepあたりの増減幅
- confidence floor、stability/limiter gain
- projection refresh後のcooldown
- adaptation開始までのwarmup

AutoScheduleの更新式を変更するときは、統計の対象（full-rank gradientかlow-rank stateか）、要素数の重み付け、state更新順序を明記してください。

## 1D parameterとparameter role

APOLLOはmatrix parameterを主対象とします。bias・normalizationなどの1D parameterは、image_genでは既定でAdamW-SF fallbackへ送られます。`--apollo-fallback came`または`--apollo-fallback sgd`で変更できます。

`image_gen`のrole分割は次の単位です。

```text
main  : role指定されなかったparameter
linear: Linear weight
conv  : Conv / ConvTranspose weight
```

role別optimizerを指定した場合、対象parameterが存在しないとエラーにします。重複割り当てを避けるため、checkpointとoptimizer stateの保存・復元を変更する際は全roleを確認してください。

## 関連ファイル

- [`../optimizers/factory.py`](../optimizers/factory.py): registry、CLI choices、共通builder
- [`../optimizers/adamw.py`](../optimizers/adamw.py): AdamW、parameter-dtype full-size state、AutoSchedule variant
- [`../optimizers/came.py`](../optimizers/came.py): CAME
- [`../optimizers/apollo.py`](../optimizers/apollo.py): APOLLO family
- [`../optimizers/muon.py`](../optimizers/muon.py): Muon
- [`../optimizers/muon_variants.py`](../optimizers/muon_variants.py): NorMuon、AdaMuon
- [`../optimizers/soap.py`](../optimizers/soap.py): SOAP
- [`../optimizers/schedulefree.py`](../optimizers/schedulefree.py): legacy Schedule-Free variants

実測結果と未完了の性能課題は[`optimizer-review-2026-09-12.md`](history/optimizer-review-2026-09-12.md)、数式・stateの監査は[`repository-audit-2026-09-12.md`](history/repository-audit-2026-09-12.md)を参照してください。
