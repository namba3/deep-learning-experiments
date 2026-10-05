# RF-Trust-Region Solver 設計

- Version: v0.2.0
- Status: Prototype implemented; numerical and model-quality validation pending
- Scope: 線形 Rectified Flow / Flow Matching の決定論的 velocity ODE
- Base method: `rf_ab2` の固定grid variable-step AB2
- Training / additional network: 不要
- Target cost: startupを含め1 callback evaluation / interval

この文書はRF-Trust-Regionの実装向け設計書であり、数学・state・validationの契約を定義する。

## 1. Review decisions

前区間でのvelocity予測誤差を使ってAB2補正を弱める研究仮説は検討可能。ただし、実装では次の数式・契約を用いる。

1. **時間と符号を統一する。** リポジトリはモデル時刻をノイズ端1からデータ端0へ下げる。Trust solverの式は正のprogress `r=1-t` とsampling velocity `w=-v` で書く。
2. **可変step予測に正しいstep比を使う。** non-uniform gridでは、次の点への外挿係数に `h_n/h_(n-1)` を使う。
3. **予測が存在しないstartupを明示する。** 最初にTrust signalを評価できるのはstep 2。step 0はEuler、step 1はgatedでないAB2とする。
4. **radius clampは実際に適用する補正にかける。** `alpha_n * (w_n - w_(n-1))` の後にclampしないと、step比が大きいとき補正幅をradiusが制限しない。
5. **trust統計はsampleごとに独立させる。** batch全体をreduceしたscalarは、ある画像の誤差で別画像のsolver更新を変える。非batch軸だけでRMSを計算し、係数を各sampleへbroadcastする。
6. **curvature proxyをraw second differenceで定義しない。** non-uniform stepでは刻み幅に依存する。v0.1の基準法から外し、必要なら刻みで正規化した別ablationにする。
7. **Warp式の微分と符号を揃える。** RF-2M-Warpの座標は `r` から `tau=phi(r)` への変換であり、変換速度は `-v/q`。この契約に合わない時間変数や符号の式は使わない。
8. **「Trust Region」は類推として扱う。** 古典的な最適化trust-regionのacceptance ratioや目的関数評価を持たない。名称・説明ではtrust-region-inspired history gateとする。

## 2. Model-time and sampling-progress contract

共有するmodel-time、sampling velocity、正方向gridの契約は[solver共通契約](sampling-solvers.md#shared-time-velocity-and-grid-contract)を参照する。このunwarped solverでは`h_n=r_(n+1)-r_n>0`を使い、入力gridは非一様でもよい。

## 3. Raw RF-AB2 baseline

Trust variantは共有solverの`rf_ab2`と同じsampling-progress grid、Euler startup、raw velocity historyを使います。variable-step係数と更新式は[RF-2M-Warpのraw baseline](rf-2m-warp-design.md#3-baseline-raw-rf-2m)を参照してください。Trust variantはAB2補正だけにgateを適用します。gammaが1でclampが無効なら、共有baselineと一致しなければなりません。

## 4. Prediction-agreement trust signal

step `n>=2` では、step `n-1` で得た2点の履歴から現在のsampling velocityを予測する。

\[
\widehat w_n=w_{n-1}+\frac{h_{n-1}}{h_{n-2}}(w_{n-1}-w_{n-2}).
\]

現在の通常callbackで `w_n` を得た後、追加NFEなしでsampleごとの誤差を計算する。

\[
e_n=\frac{\operatorname{RMS}_{\mathrm{feature}}(w_n-\widehat w_n)}
{\operatorname{RMS}_{\mathrm{feature}}(w_n)+\epsilon}.
\]

RMSはbatch軸以外の全軸で各sampleごとに計算する。trust係数はshape `(B,)` とし、状態へ適用する前に `(B, 1, ..., 1)` へreshapeする。計算はsolver accumulation dtypeで行い、分母epsilon、finite性、zero-velocity時の挙動を実装で定義・検査する。両方のRMSが0のときは `e_n=0` とする。

基準mapping:

\[
\gamma_n=\exp(-\lambda e_n),\qquad \lambda\ge0.
\]

よって `0 < gamma_n <= 1`。この係数はheuristicであり局所誤差の保証ではない。初期探索値は `lambda in {1, 2, 4, 8}`。推奨値・品質改善を主張しない。

## 5. Trust-gated update and startup

基準solverはradius clampなしとし、

\[
x_{n+1}=x_n+h_n(w_n+\gamma_n c_n)
\]

で更新する。

| step | 利用できる履歴 | update |
|---|---|---|
| `n=0` | `w_0` | Euler |
| `n=1` | `w_0,w_1`、まだ予測誤差なし | ungated AB2 (`gamma_1=1`) |
| `n>=2` | 2点以上の履歴と `w_hat_n` | trust-gated AB2 |

step `n=1` のAB2補正を含む状態から次の予測を

\[
\widehat w_{n+1}=w_n+\frac{h_n}{h_{n-1}}(w_n-w_{n-1})
\]

と保存する。履歴・予測・係数stateは呼び出し単位で初期化し、別sample runに持ち越さない。

上のindex定義では `e_n` がstep `n` のAB2補正 `c_n` を制御する。予測時と更新時のindexをずらさない。

## 6. Optional fixed correction radius

補正の大きさにも上限を設けるvariantを比較する場合、raw差分ではなく適用するAB2補正 `c_n` をclampする。

\[
\tilde c_n=c_n\min\!\left(1,
\frac{\Delta\,\operatorname{RMS}_{\mathrm{feature}}(w_n)}
{\operatorname{RMS}_{\mathrm{feature}}(c_n)+\epsilon}\right),
\]

\[
x_{n+1}=x_n+h_n(w_n+\gamma_n\tilde c_n).
\]

`Delta` はsampleごとの無次元上限である。RMSと係数は各sample内で計算する。初期実装ではradiusをOFFにする。ONの場合は`fixed_radius` variantとして識別し、thresholdによるadaptive radiusの増減は別実験にする。clampが有効な区間では基準AB2の次数主張をしない。

adaptive radius案の「誤差良好なら拡大／不良なら縮小」はthreshold・更新遅延・初期値を含む追加controllerになる。初期版へ同時投入しない。

## 7. Curvature, direction, and spatial gates

curvature gate、global cosine direction gate、token/patch別係数は基準methodに含めない。

non-uniform gridでcurvatureを見るなら、生の `w_n-2w_(n-1)+w_(n-2)` ではなく、例えば隣接区間の差分商

\[
a_n=\frac{2}{h_{n-1}+h_{n-2}}\left[
\frac{w_n-w_{n-1}}{h_{n-1}}-\frac{w_{n-1}-w_{n-2}}{h_{n-2}}\right]
\]

を使う。これでも状態`x`が変化するtrajectory上の診断量であり、独立した厳密な時間二階微分ではない。導入時はtrust error gate単独と別armにし、空間別gateも別variantにする。

## 8. CFG and batch behavior

callback内でCFGを適用し、そのpost-CFG velocityを履歴とtrust計算に使う。CFG scaleがstep/sampleごとに変わる場合は実効scaleを記録する。prediction agreementが悪いとgateが弱まるのは仮説であり、high-CFGで必ず改善する性質ではない。

trust stateはsampleごとに独立する。batchの並び替えや異なるpromptを同一batchに入れた場合でも、各sampleのsolver係数が他sampleのvelocity値で変わらないことを確認する。

## 9. Warp extension (out of initial scope)

最初の実装はraw progress `r` 上の`rf_ab2`だけを対象とする。RF-2M-Warpと統合する場合は、

\[
\tau=\phi(r),\quad q(r)=\frac{d\tau}{dr}>0,\quad
U_n=\frac{dx}{d\tau}=-\frac{v_n}{q(r_n)}
\]

を用いる。trust predictor、error、history、AB2補正はすべて `U` と正の `tau`-step `k_n` で統一する。すなわち

\[
\widehat U_n=U_{n-1}+\frac{k_{n-1}}{k_{n-2}}(U_{n-1}-U_{n-2}),\quad
 e_n^\tau=\frac{\operatorname{RMS}(U_n-\widehat U_n)}
 {\operatorname{RMS}(U_n)+\epsilon}.
\]

RF-2M-Warpが返す時間列・微分との整合をidentity warpで確認する。raw `v` と変換済み `U`、またはモデル時刻差とsolver座標差を混在させない。これは別solver variantとして設計・記録する。

## 10. Shared API and metadata proposal

共有`sample`へ別solver名`rf_trust_region`を追加し、既存`rf_ab2`は変更しない。solver本体は現在のcallback契約を維持し、grid・速度履歴・trust stateを管理する。Trust強度はキーワード引数`rf_trust_lambda`で指定する。

最低限保存するmetadata:

- solver名、scheduler/grid名、step数、モデル時刻端点
- trust signal名、mapping、`lambda`、epsilon policy
- startup規則、radius variantと設定（OFFも明記）
- CFG scale、seed、callback evaluation数
- warp統合時はwarp名、parameter、solver-grid種別と微分設定

CLIのデフォルトは従来のEulerのまま。新variantの設定はcheckpointの学習構成を変えないが、生成sidecarおよびtraining observation記録から再現できること。

## 11. Validation plan

実施済みCPU証拠と未検証範囲は[solver検証索引](sampling-solvers.md)にまとめる。追加確認では、
同一gridでの`gamma=1`退化、variable-stepの手計算reference、zero/near-zero velocityとfinite境界、
per-sample reduction・batch independence・history resetを優先する。trust gate/radiusを有効にした結果へ
raw AB2の収束次数をそのまま当てはめない。

品質比較ではEuler/`rf_ab2`/trust variantをmatched NFE・checkpoint・初期noise・CFG・解像度・seedで
比べ、toy ODE errorと画像品質・時間・memoryを分ける。CPU smokeはGPU品質の証拠にならない。

## 12. Claims and limitations

- prediction agreementは履歴外挿の過去予測性能を測るheuristicで、局所誤差推定器でもaccept/reject testでもない。
- gamma gateは常に1以下だが、真の解への誤差を必ず減らす保証はない。
- variable-step AB2 baselineは滑らかなODEと適切なstep比の条件で二次法。gate/clampの係数変化を含むTrust variantへ同じ次数を自動で継承しない。
- NFE/intervalは1だがRMS reductionなどtensor演算・メモリ読み書きが追加される。実測時間とmemoryを別途計測する。
- Warp、ER-SDE、score drift、stochasticity、adaptive step splitting、likelihood/inversionは初期variantの範囲外。
- 品質改善、high-CFG改善、一般的なstability improvementは実験で確認されるまで主張しない。

## 実装状況

共有`flow_sampling.sample`へ`rf_trust_region`を追加し、正のprogress上でのper-sample prediction-error gateを実装した。VFP-DiTの生成CLIは`--rf-trust-lambda`、学習中の観測samplingは`--sample-rf-trust-lambda`を受け付け、生成sidecarとobservation recordへ実効値を保存する。既存solverと学習checkpointのデフォルトは変更していない。

CPU testの具体的な範囲と未検証項目は[solver検証索引](sampling-solvers.md)を参照する。配線と限定的なCPU確認は、一般的な数値精度や品質の検証済みを意味しない。
