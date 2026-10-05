# `optimizers/`

共有optimizer実装とlearning-rate制御を置くパッケージです。CLIからの選択は[`factory.py`](factory.py)に集約しています。

## 使用方法

train scriptでは通常、次のcore choicesを使えます。

```text
AdamW  CAME  APOLLO  APOLLO-CAME
```

`image_gen`では、実験用variant・legacy compatibility名も許可されます。正確なchoicesは各scriptの`--help`を確認してください。core choicesにadaptive learning-rateを付ける場合は`--auto-schedule`を使います。

APOLLO系optimizerのper-parameter norm-growth limiterは既定で無効です。
学習CLIでは明示的に有効化できます。小型MLP probeではAPOLLO-CAMEの一部設定がlimiter無効時に
発散したため、APOLLO-CAMEなどではrank・learning rateと合わせて実モデルで確認してください。

`AdamW-SF`、`AdamW-LRSF`、`AdamW-SF-LR`、`AdamW-LRSF-LR`、`AdamW-LR-EMA`、`AdamW-LR-EMA-Conf`、`AdamW-LR-EMA-Conf-LRSF`、`APOLLO-Conf`、`CAME-SF`、`CAME-LRSF`、`APOLLO-CAME-LRSF`、`APOLLO-SF`系は明示的に選択できる
実験用variantです。`AdamW-SF`はfull hidden stateを持つoracle、`AdamW-LRSF`は
AdamWのsecond momentを維持したままhidden deltaだけを低rank化する比較用variantです。
`AdamW-LR-EMA`はSchedule-Freeを使わず、matrix gradientを固定低rank projectionへ写像し、
そのlatent gradientだけを急峻EMA（既定`beta=0.9`）してdecodeし、直接更新へ使う比較用
ablationです。Adamのsecond moment、Schedule-Freeの`z`、deltaは保持しません。1Dと
requested rank以上の小行列はparameter dtypeのfull gradient EMAへfallbackします。
`AdamW-LR-EMA-Conf`はこの比較にlatent innovation varianceを追加し、
`m / sqrt(c + alpha*m²)`で更新を正規化するvariantです。
`AdamW-LR-EMA-Conf-LRSF`はこれにSchedule-Freeの低rank deltaを組み合わせた初期prototypeで、
confidence用projectionとdelta用projectionを分離します。confidence側は固定projection、
delta側は既存LRSFのrefresh policyを使います。1Dとrank飽和行列はfull Schedule-Freeへ
fallbackします。まだ品質・速度の採用判断をする段階ではありません。
`APOLLO-Conf`は同じconfidence正規化をAPOLLO型stateへ適用する比較variantです。
低rankの`projection`、`exp_avg`、innovation varianceとしての`exp_avg_sq`を保持し、
latent stateから作ったchannel-wise scalingをfull gradientへ適用します。
`AdamW-SF-LR`は逆にSchedule-Freeの`z`をfull-sizeで維持し、matrix parameterの
`exp_avg_sq`だけを固定射影上のlatent second momentへ置き換えるablationです。1Dと
requested rank以上の小行列はfull Schedule-Freeへfallbackします。これによりdriftの近似と
preconditionerの近似を分離して検証できます。matrix経路は現在PyTorch/referenceのみで、
Triton指定時もこの経路はreferenceを使用します。
`APOLLO-SF`はAPOLLOの低rank updateとfull-rank Schedule-Free stateを組み合わせる検証用variant
です。`APOLLO-SF`はBF16 `z`、`APOLLO-SF-INT8-Z` / `APOLLO-SF-INT4-Z`はblockwise量子化
`z`、`APOLLO-SF-INT8-Delta` / `APOLLO-SF-INT4-Delta`はblockwise量子化`sf_delta`を保持します。
INT4はpacked nibbleとして保存します。いずれも現時点ではreference実装であり、量子化と
dequantizeのstep時間・workspaceを別途測定する前提です。
`APOLLO-SF-LRSF`は`sf_delta`をAPOLLO更新用とは別の直交projection上のlatent tensorとして
保持するablationです。更新時にはfull-rank deltaを一時再構成するため、persistent stateは削減
できますが、peak VRAMとstep時間は別途検証が必要です。初期variantではdelta projectionは固定し、
APOLLO本体のprojection refreshとは独立に扱います。検証rankは`--rank`で指定します。
`AdamW-LRSF-LR`は`z`、full `exp_avg_sq`、full `sf_delta`を同時に持たず、Schedule-Free
driftとlatent second momentを同じ低rank projectionへ保持する統合prototypeです。現在は
固定projection、hard refresh、state double-bufferのshadow refreshに対応します。
`projection_refresh_state="transport"`ではdeltaを
新しい座標へ移し、latent second momentをoverlap二乗で近似transportします。`"reset"`では
latent second momentを破棄し、局所bias correctionを再開します。どちらもfull covarianceの
transportではありません。shadowではactive/shadowのprojection、delta、latent second
momentを毎step更新して、intervalでshadowを昇格します。smooth・orthogonal transport・
Triton融合は未対応です。
`AdamW-LRSF`の`backend="auto"|"torch"|"triton"`では、Triton使用時にsecond momentの
前処理を融合します。低rankの射影・delta再構成はPyTorch/cuBLAS経路を使い、非対応入力は
従来のPyTorch経路へfallbackします。
text LMの長期Transformer学習では`hard` projection refreshを標準条件とし、
refreshなしの比較baselineは`mode="none"`（frozen）を明示します。これはtext LMの
実験既定値であり、image_aeなど他の学習経路の既定値は変更しません。
LRSF系はコンストラクタの`projection_refresh`または`delta_refresh`で、
`mode="none"`の固定/frozen・hard transport・PA/PB smooth refresh・
`shadow` double-buffer refreshを選択できます。さらに`orthogonal_refresh={"rate": 0.02,
"seed": 0}`を渡すと、intervalを使わず対象projectionを毎step微小回転します。
`rate=0`（既定）は無効で、通常のprojectionは固定です。
smooth refreshの`mix`には`linear`、`smoothstep`、`stochastic`、
`ema`を指定できます。
`stochastic`ではPA/PBの両方をEMA更新しつつ、各stepでPAまたはPBの一方だけを選び、
その選択確率をwindow中にPAからPBへ移します。
`ema`ではPA/PBのdecode importance weightを指数平滑し、既定のdecayは
`exp(-1/window)`です。`ProjectionRefreshPolicy`へ`ema_decay`を渡すと上書きできます。
`shadow`ではactive branchとshadow branchのprojection/deltaを常時更新し、intervalごとに
shadowをactiveへ昇格します。full AdamW second moment、CAME state、APOLLOの`R_update`
stateは共有し、projection依存のLRSF stateだけを二重化します。そのため低rank stateの
メモリは増えますが、refresh時に新branchをcold startしません。shadow更新は追加の低rank
projection計算を伴うため、step時間とstate bytesを別々に測定してください。
`CAME-SF`はfull hidden delta oracle、`CAME-LRSF`はCAMEのfull update/stateを維持して
Schedule-Free hidden deltaだけを固定random projectionへ制限します。`APOLLO-CAME-LRSF`
はAPOLLO-CAMEの更新projectionとSchedule-Free差分projectionを分離します。

低精度parameterでは、parameterと同じ要素数・shapeを持つfull-size state
（AdamWのmoment、Schedule-Freeの`z`/second moment、CAMEのunfactored moment、
Muon/Lion/SOAPのfull-size moment）をparameterのstorage dtypeで保持します。既存checkpointの
FP32 stateは最初の利用時にparameter dtypeへ変換します。低rank moment、projection、CAMEの
row/column factor、QR/eigendecomposition用basis、診断scalarは原則FP32のままです。
これらはFP32のgradient/projection演算へ直結しており、低rank stateだけをBF16化すると
cast削減にならず、蓄積精度も下がるためです。一時計算は必要に応じてFP32で行います。
hard refreshとshadowの新shadow生成では`transport_overlap`を指定して、旧projectionと新しいrandom projectionを
混ぜてからQR直交化できます。`1.0`は旧basisの保持、`0.0`は従来のrandom refresh、途中の
値はdelta transportの急変を緩和する実験設定です。既定値は`None`で、既存のhard refreshの
挙動を維持します。full decoded deltaを作る診断設定と同様、overlapの品質効果とstep時間は
別々に測定してください。

```bash
python3 image_gen/train.py \
  --vae-model Qwen/Qwen-Image \
  --optimizer APOLLO \
  --auto-schedule
```

optimizerのアルゴリズム、state要素数、parameter role分割、AutoScheduleの設計は[`../docs/optimizers.md`](../docs/optimizers.md)にまとめています。

AdamW、AdamW-SF、CAME、APOLLOのkernel融合、workspace、benchmark、受け入れ基準は
[`../docs/history/optimizer-performance-design-2026-09-14.md`](../docs/history/optimizer-performance-design-2026-09-14.md)にまとめています。

APOLLOのfallbackはparameterの分類をdictまたは`APOLLOFallbackPolicy`で渡せます。
既定では1D parameterをAdamW-SFへ送り、matrix parameterはAPOLLOとAdamW-SFの
推定stateサイズを比較します。AdamW-SFの方が小さいmatrixだけをfallbackさせます。
旧来の文字列`fallback="came"`は1D parameterだけをCAME-like pathへ送り、matrixは
APOLLOのままにします。
明示的にCAMEを基準にstateサイズ比較する場合は、次を使います。

```python
from optimizers import APOLLO, APOLLOFallbackPolicy

optimizer = APOLLO(
    model.parameters(),
    fallback=APOLLOFallbackPolicy(
        one_dimensional="came",
        small_matrix="auto",
    ),
)
```

直接生成するAPOLLO系optimizerのrank既定値は8です。`APOLLOMini`のみrank=1に
固定されます。

APOLLO本体・APOLLO-CAMEでは、共通の`ProjectionRefreshPolicy`をrefreshのmode/interval
契約に使います。現行対応は`none`/`hard`/`smooth`です。hard refresh時の低rank momentは
`projection_refresh_state="reset"`（既定）または`"transport"`から選択できます。
`transport`は旧新projectionのoverlapを使ってmomentを近似移送します。smoothでは旧新の
低rank momentを二重bufferし、window中にscalingをmixします。その間のpersistent stateは
概ね倍になるため、低メモリ性とrefreshの連続性を分けて評価してください。
`mix="ema"`では、window中のPA/PB importance weightを指数平滑します。既定のdecayは
`exp(-1/window)`で、`ProjectionRefreshPolicy`へ`ema_decay`を明示指定することもできます。
`mix="stochastic"`では、PA/PBの両方を更新しながら、weightをPB選択確率として使います。
`orthogonal_refresh`を指定すると、APOLLOの`R_update`もintervalなしで毎step回転し、
低rank momentを新しい座標へtransportします。
Schedule-Freeの`R_delta` smooth refreshとはstateの意味が異なるため、別経路として検証します。
`image_ae`/`image_gen`では`--apollo-projection-refresh-mode none|hard|smooth`と
`--apollo-projection-refresh-mix linear|smoothstep|ema|stochastic`で選択でき、
`--apollo-orthogonal-refresh-rate`で毎step回転率を指定できます。
設定はcheckpoint metadataとresume時の設定復元にも含まれます。

## パッケージの責務

- optimizer本体とTriton実装を提供する
- 1D parameter、matrix parameter、複数parameter groupの契約を維持する
- `factory.py`でCLI名からbuilderへの対応を一元管理する
- PyTorch/reference、Triton、fallbackの更新順序とstate構造を揃える

学習スクリプト固有のparameter groupingやcheckpoint保存は、各script側の責務です。optimizerの変更では、parameter更新、moment・preconditioner、weight decay、projection、state保存・復元の順序を確認してください。

## 主な実装

| ファイル | 内容 |
| --- | --- |
| `adamw.py` | AdamW、parameter-dtype full-size state、AutoSchedule variant |
| `adamw_lr_ema.py` | 低rank projected gradientの急峻EMAを直接更新へ使う比較用variant |
| `adamw_lr_ema_conf_lrsf.py` | confidence stateとSchedule-Free low-rank deltaを分離した統合prototype |
| `adamw_lrsf.py` | AdamW-SFのhidden deltaを低rank化したvariant |
| `adamw_sf_lr.py` | AdamW-SFのpreconditionerだけを低rank化したablation |
| `adamw_lrsf_lr.py` | low-rank driftとpreconditionerを統合したLRTDO prototype |
| `came.py` | CAMEとAutoSchedule variant |
| `came_lrsf.py` | 固定projection版CAME-LRSF |
| `apollo_lrsf.py` | 固定projection版APOLLO-CAME-LRSF |
| `apollo_sf.py` | APOLLO low-rank updateとfull-rank/量子化Schedule-Free stateの比較variant |
| `apollo.py` | APOLLO family、`APOLLO-CAME`、`APOLLO-Conf`のprojection/state |
| `apollo.py` | APOLLO familyとprojection/state |
| `muon.py` / `muon_variants.py` | Muon、NorMuon、AdaMuon |
| `soap.py` | SOAP |
| `lion.py` | Lion |
| `schedulefree.py` | legacy Schedule-Free variant |
| `factory.py` | 共通CLI registryと`build_optimizer()` |

optimizerの数値変更を検証するときは、リポジトリ標準の`PYTHONPATH=. python3 -m pytest -q`に加え、可能なら対象dtype・複数group・state restore・CUDA/Tritonのforward/backwardを確認します。
