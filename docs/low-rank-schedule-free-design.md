# APOLLO-SPR and Low-Rank ScheduleFree optimizer design

最終更新: 2026-09-13

この文書は2026-09-13時点の設計snapshotです。variant案や推奨値は当時の設計記録であり、現行backendや既定値を示すものではありません。現在の実装契約はコードと[`optimizers.md`](optimizers.md)を参照してください。実装段階、測定結果、当時の比較計画は[日付付き研究記録](history/low-rank-schedule-free-records-2026-09-13.md)に分離しています。

## 目的と位置付け

`APOLLO-SPR`（APOLLO with Smooth Projection Refresh）は、PA/PB double-bufferとsmooth
weightで更新projectionを切り替える設計案である。LRSF併用時はAPOLLOの更新用
`R_update`とSchedule-Free差分用`R_delta`を別々のpolicy/stateとして扱う。variant名を
増やすよりrefresh policyとして再利用する。

ScheduleFreeの評価時parameterをfull-size hidden parameterなしで近似する。`CAME-LRSF`は
Schedule-Free効果と低rank近似誤差を分離する比較用で、CAMEよりstateが小さくなる設計では
ない。低メモリ候補はCAME部分も低rank化する`APOLLO-CAME-LRSF`である。

## 用語とScheduleFree基準式

実装上の名前と論文・upstream実装の名前が混ざらないよう、次の記号を固定する。

```text
y_t       学習時にforward/backwardするparameter
s_t       ScheduleFreeが本来保持するhidden parameter
h_t       s_t - y_t
x_t       評価時parameter
u_t       CAMEが計算した、learning rate適用前のupdate direction
c_t       ScheduleFreeの重み係数
eta_t     learning rate
beta_sf   ScheduleFree用のbeta1
```

ScheduleFreeの基準式は次である。

```text
y_{t+1} = y_t + c_t h_t
              + eta_t * (beta_sf * (1 - c_t) - 1) * u_t

h_{t+1} = (1 - c_t) * (h_t - eta_t * beta_sf * u_t)
```

評価時はupstream互換の外挿を行う。

```text
x_t = y_t + (1 - 1 / beta_sf) * h_t
```

したがって、ユーザーが想定する
`x_t = z_t + Delta z_t`の`Delta z_t`は、内部stateとしては
`(1 - 1 / beta_sf) * h_t`に相当する。内部では`h_t`を保存し、評価時だけ係数を
適用することで、train/eval切り替え時の符号と復元処理を明確にする。

`c_t`はScheduleFreeの累積weightから計算する。第一版では比較を単純化するため、
`r=0`、外部schedulerなし、warmupなしを基本条件とする。`beta_sf`はCAMEの
`beta1`と独立した設定値にする。CAME自身の`exp_avg`とScheduleFreeの混合を
二重momentumとして評価する必要があるためである。

## 低ランク差分表現

parameterをflatten後の`M x N`行列として扱い、rankを`r`とする。LRSFの差分用
projectionは、APOLLOの更新用projectionとは別stateとして扱う。差分`h_t`は過去の
stepから累積されるため、APOLLOのように独立乱数へ置き換えるrefreshをそのまま共有
してはならない。

`M >= N`の場合:

```text
R : N x r
H : M x r
h = H R^T
u_proj = u R
H_next = (1 - c) * (H - eta * beta_sf * u_proj)
```

`M < N`の場合:

```text
R : r x M
H : r x N
h = R^T H
u_proj = R u
H_next = (1 - c) * (H - eta * beta_sf * u_proj)
```

parameterへの`c*h`の適用は、full-sizeの`h`をpersistent stateとして作らず、
`addmm_`相当の行列積で行う。CAMEのfull update direction `u`自体はCAMEの
既存更新に必要な一時tensorなので、第一版ではそれを新たに低ランク化しない。

この近似は、ScheduleFreeの差分`h`をランダム部分空間へ射影する近似である。
CAMEの更新式とCAME stateの更新順序は維持し、近似対象を`h`の表現だけに限定する。

## Projection refreshの設計（改訂）

### 第一版の既定方針

projectionを次の2種類に分離する。

```text
R_update : APOLLO/APOLLO-CAMEのgradient・moment用
R_delta  : LRSFのSchedule-Free差分h用
```

基準条件では`R_delta`を固定する。APOLLOの更新方向がrefreshで変わっても、累積した
Schedule-Free差分を別basisへ突然移さないためである。更新projectionと差分projectionの
refreshを同時に使う場合も、それぞれのPA/PBを独立して保持する。共有してstateを減らす案は、
数値一致と履歴保持の確認が必要なため採用しない。

`fixed`は`mode="none"`のfrozen基準、`hard`はinterval到達時に即時交換する方式を指す。
現行のmode、mix、constructor、checkpoint設定は[`optimizers.md`](optimizers.md)を参照する。
追加stateは`R_delta`と低rank係数`H`で、full-sizeのhidden parameterは持たない。

### Per-step orthogonal refresh

interval refreshとは別の候補として、`R_delta`へ毎stepの微小ランダム回転を適用する。
コンストラクタでは`orthogonal_refresh={"rate": rho, "seed": seed}`として指定し、
`rho=0`を無効状態とする。これは新しい独立projectionへ置換する方式ではなく、現在の
直交基底の近傍をStiefel接空間上で移動する方式である。

`M >= N`で`P`が`N x r`、`P^T P=I`の場合、Gaussian行列`G`から接空間成分を

```text
T = G - P(P^T G)
P_candidate = P + rho * sqrt(r) * T / max(||T||_F, eps)
P_next = qf(P_candidate)
```

として生成する。`qf`はreduced QRの`Q`である。`M < N`では`P`の向きを転置した
対応式を使い、`P P^T=I`を維持する。毎stepのseedはcheckpointの
`orthogonal_refresh_count`から決定的に導出するため、resume後も回転系列を再現できる。

projectionを回転した後は、累積Schedule-Free deltaを新しい座標へtransportする。
`M >= N`では`H_next = H P_old^T P_next`、`M < N`では
`H_next = P_next P_old^T H`とする。smooth interval refreshが同時に有効な場合は、
activeなPA/PBそれぞれを同じ規則で回転し、対応する`H_A/H_B`もtransportする。これに
よりrefresh intervalの境界でだけprojectionが変わるのではなく、履歴を連続的に保ちながら
部分空間を探索できる。

この方式の追加persistent stateは回転カウンタだけで、full-size hidden parameterや
第三のprojection bufferは増えない。一方、毎stepのGaussian生成・接空間射影・QRが必要な
ため、step時間と一時workspaceは増える可能性がある。従って、評価では固定projectionと
interval refreshから分けて、projection直交誤差、delta transport誤差、step時間、peak
memory、validation lossを記録する。APOLLOの`R_update`にも同じ
`orthogonal_refresh={"rate": rho, "seed": seed}`を適用できる。`R_update`と
`R_delta`は別stateとして回転・transportし、interval refreshと組み合わせる場合も
それぞれのカウンタをcheckpointへ保存する。

### 独立乱数refreshを採用しない理由

旧projectionを`R_old`、新projectionを独立乱数`R_new`とすると、旧差分を新空間へ
移すには少なくとも新空間への射影が必要になる。`M >= N`の表現では、直交化された
projectionを仮定しても

```text
H_new = H_old R_old^T R_new
```

となり、これは旧`h`の新部分空間への射影近似である。独立部分空間では、期待される
保持エネルギーは概ね`r / min(M, N)`である。例えば`rank=4`、短辺256なら、旧差分の
大部分が失われるため、実質的にScheduleFree stateをresetするのと近い。

`M < N`では対応するtransportは次である。

```text
H_new = R_new R_old^T H_old
```

projectionが直交していない場合は単純な内積ではなく、必要に応じて
`(R_new R_new^T)^-1`を含む最小二乗projectionを使う必要がある。ただし逆行列計算と
数値条件の管理が増えるため、第一版では直交projectionを前提にする。

現在のAPOLLO refreshはGaussian projectionを新規生成するだけで、低ランクmomentの
座標transportを行わない。この挙動はAPOLLOの更新stateでは既存仕様として測定できる
が、累積差分`h`を持つLRSFへ直接流用しない。

### delta projection refreshの第一実装

`delta_proj_gap > 0`を有効にする場合は、PA/PBのdouble-buffer refreshを使う。
`ProjectionRefreshPolicy(mode="smooth", interval=200, window=200)`が基準候補である。
refresh開始時に旧`H`を新projectionへtransportし、window中は
`decode(H_A, P_A)`と`decode(H_B, P_B)`を混合する。

`mode="hard"`ではtransport後に即時swapする。`mode="none"`は従来の固定projection
動作と完全に同じである。

このPA/PB方式はprojectionを瞬間的に置き換えず、旧projectionから新projectionへimportanceを
徐々に移す方式である。

#### PA/PBの状態とweight schedule

PA/PB double-bufferでは、現在・次projectionと、それぞれに対応する低rank差分係数を
保持する。smooth refresh中は係数を別々のEMAとして更新し、開始時に旧係数を新projectionへ
transportして次側を初期化する。window中は両方のdecodeをweightで混合し、終端で新側を昇格
する。この構成は旧basisの履歴を新側の更新で上書きせず、full-size EMAも追加しない。

線形とsmoothstepのweightは次の通り。smoothstepは端点で傾きが0になる。

```text
w(u) = u                         # linear
w(u) = 3u^2 - 2u^3               # smoothstep
P(u) = (1 - w(u)) * PA + w(u) * PB
```

EMAでは`w_{t+1} = decay * w_t + (1 - decay)`を使い、window終端でPBを昇格する。
Stochastic mixはweightをPB選択のBernoulli確率に使うため、連続blendと異なるbranch-level
noiseを生む。selection counterとseedがresume再現に必要であり、収束・loss spike・step時間を
独立に比較する。bufferは事前確保してswap・再利用し、refresh中のpersistent allocationを避ける。
具体的なscheduleとstate仕様は[`optimizers.md`](optimizers.md)を参照する。

#### 混合projectionに対するHのtransport

`P(u)`がstepごとに変わるため、`H`を同じ数値のまま使ってはいけない。`M >= N`で
`h = H P^T`と表す場合、次のprojectionへ移す係数を最小二乗で求める。

```text
G_next = P_next^T P_next + lambda I
H_transport = H_current P_current^T P_next G_next^{-1}
```

その後、ScheduleFreeの差分更新を適用する。

```text
H_next = (1 - c) * (H_transport - eta * beta_sf * U_next)
U_next = u_t P_next G_next^{-1}
```

`M < N`では左右を転置した式を使う。`lambda`はprojectionの条件数を管理する
小さなridgeであり、固定projection時にも同じencode/decode契約を使う。直交projection
を採用できる場合は`G_next=I`となり、transportは`H_current P_current^T P_next`
まで簡略化できる。

refresh境界では、swap直後の`P_next`が直前の`P_current`と同じになるようにする。
したがって境界transportは恒等写像になり、`h`に不連続なbasis changeを入れない。

#### refresh spikeの範囲

PA/PB blendはprojection切替による方向の不連続を抑えるが、Gaussian生成、Gram行列solve、
transportのコストは残る。比較ではprojection生成・blend・delta transport/updateの時間を分け、
step時間、peak memory、validation lossも記録する。生成spikeが支配的ならCPU事前生成、非同期
copy、chunked fillを候補とする。

refresh品質は累積差分全体の保持率でなく、隣接step間のlocal transport errorで見る。独立乱数へ
一気に移す場合の保持率を閾値にするとrefreshを常に拒否しうるためである。診断の定義は次の通り。

```text
transport_error = ||h_before - decode(encode(h_before, P_next), P_next)||
                   / max(||h_before||, eps)
accept iff transport_error <= delta_max_transport_error
```

閾値超過時は差分をresetせず、window延長または次candidate生成の延期を行う。`h`をzero化する、
transportせず`H`を新basisの係数として再解釈する方式は、Schedule-Free履歴を壊すため採用しない。

## stateとfallback

matrix parameterでは、追加stateは概ね次である。

```text
fixed R_update       : r * min(M, N)
SPR PA/PB            : 2 * r * min(M, N)
low-rank H or moment : r * max(M, N)
```

`APOLLO-SPR`ではPA/PBが従来の`R_update`を置き換えるため、更新projectionのstateは
2つのprojection buffer分だけ増える。`APOLLO-CAME-LRSF-SPR`で`R_update`と`R_delta`を
両方smooth refreshする場合は2組のPA/PBを持つため、projection共有は数値検証後の
別最適化とする。`CAME-LRSF`単体では、checkpoint再開と数値再現のため`R_delta`を保存する。

1D parameterや、低ランクstateの方がCAME stateより大きくなる小行列は、既定で
CAMEへfallbackする。APOLLOで導入済みのpolicy形式に合わせ、次のcheckpoint-safe
表現を使う。

```python
fallback = {
    "1d": "came",
    "small_matrix": "auto",
    "state_margin": 1.0,
    "min_savings_bytes": 0,
}
```

backendはparameterごとに初回stepで決め、学習途中では変更しない。`auto`の判定は
「CAME本体のstate + LRSF追加state」と比較する。小行列だけを低ランク化して、
stateやprojectionの分だけ不利になる状態を避ける。

## train/evalとcheckpoint契約

`CAME-LRSF`はSchedule-Freeの`train()`/`eval()`契約を持つ。train時は`y_t`、eval時は
`x_t = y_t + (1 - 1 / beta_sf) * h_t`をparameterに反映し、trainへ戻す際に同じ差分を戻す。
mode切替で`h_t`やprojectionは変更しない。

checkpointはparameterとprojection、Schedule-Free累積weight、mode、refresh policy/進行状況を
保存し、resume後にtrain parameterを復元する。`R_update`と`R_delta`のrefresh stateは独立して
記録する。現行の保存schema、fallback設定、refresh optionは[`optimizers.md`](optimizers.md)と
コードを参照する。
