# RF-2M-Warp: Rectified Flow 向け時間変換 multistep solver

- Version: v0.2.0
- Status: prototype integrated into the shared solver and VFP-DiT entrypoints; focused numerical and model-quality validation pending
- Scope: 線形 Rectified Flow / Flow Matching の決定論的 velocity ODE
- Training / additional network: 不要
- Target cost: 初期点を含め原則 1 model evaluation / step

この文書は、sampling direction、座標変換の微分、安定化処理、API責務を整理した実装向け設計書である。

## 1. 目的と位置づけ

モデルはRectified Flowの速度場を予測する。実装上のモデル時刻はノイズ側が t=1、データ側が t=0 で、callbackは

\[
v_\theta(x,t)\approx \frac{dx}{dt}
\]

を返す。生成は t=1 から t=0 へ進む。

直線経路かつ理想的な速度場では速度は一定なのでEulerが厳密である。有限モデル、条件付け、CFG、経路の非直線性などがある場合はstep内で速度が変化し、1 NFE/stepの履歴法で誤差を減らせる可能性がある。

本設計は、sampling progressに単調な座標変換を適用し、その座標上の変換速度をvariable-step Adams-Bashforth 2で積分する。DPM-Solver++ 2Mの式やdiffusion固有のnoise/score parameterizationは移植しない。「座標選択と履歴再利用」という着想のみを参照する。

研究仮説は、適切な座標で変換速度がより滑らかになれば、同一NFEのraw-time AB2より積分誤差を下げられる、である。品質改善は仮定しない。

## 2. 時間・速度の契約

共通するcallback、sampling progress、正方向gridの符号規約は[solver共通契約](sampling-solvers.md#shared-time-velocity-and-grid-contract)を参照する。本solver固有の変換は`tau=phi(r)`、`q(r)=phi'(r)>0`であり、全サンプルに共通する時間のみの関数とする。warped updateでは`U=w/q`を使い、model callbackには元の時刻`t=1-r`を渡す。

## 3. Baseline: raw RF-2M

座標変換なしのr上で

\[
\frac{dx}{dr}=w_\theta(x,r)
\]

を解く。正のsampling-progress刻みを
\(h_n=r_{n+1}-r_n>0\) とすると、variable-step AB2は

\[
x_{n+1}=x_n+
h_n\left[
w_n+\frac{h_n}{2h_{n-1}}(w_n-w_{n-1})
\right].
\]

最初のstepはEuler startup:

\[
x_1=x_0+h_0w_0.
\]

uniform gridでは係数は \(3/2,-1/2\) になる。滑らかな決定論的ODEに対して、startupを含めglobal order 2を期待する。有限精度、粗いstep、極端なstep ratio、clampなどの条件下でこの精度が保証されるという意味ではない。

リポジトリの共有solverにはsampling-progress上のAB2に相当するrf_ab2がある。実装時はこの挙動をraw baselineとして再利用し、別名の重複実装を作らない。グリッド、startup、accumulation dtypeを揃えたidentity equivalenceを確認する。

## 4. Warped RF-2M

\[
\tau=\phi(r)
\]

へ変換し、時刻 \(\tau_n\) でmodelを評価する:

\[
r_n=\phi^{-1}(\tau_n),\quad
t_n=1-r_n,\quad
w_n=-v_\theta(x_n,t_n),\quad
U_n=w_n/q(r_n).
\]

\(k_{n-1}=\tau_n-\tau_{n-1}>0\) とすると、変換速度の一次外挿を積分した更新は

\[
\bar U_n=U_n+
\frac{k_n}{2k_{n-1}}(U_n-U_{n-1}),
\]

\[
x_{n+1}=x_n+k_n\bar U_n.
\]

uniform τ gridでは

\[
\bar U_n=\frac32U_n-\frac12U_{n-1}.
\]

startupはwarped Euler \(x_1=x_0+k_0U_0\)。追加のmodel evaluationは不要である。履歴には外挿後の \(\bar U_n\) ではなく、生の変換速度 \(U_n\) を保存する。

warpは連続ODEの軌道を変えないが、有限stepのmultistep近似は座標に依存する。改善対象はこの離散化誤差であり、別の生成分布を意図するものではない。

## 5. Gridとcoordinate warpの切り分け

比較では以下を分ける:

| Arm | solver coordinate / update | 分離する効果 |
| --- | --- | --- |
| Euler | uniform r, Euler | 基準 |
| RF-2M-Raw | uniform r, raw AB2 | velocity history |
| RF-2M-Raw + warped nodes | warpで選んだr nodes, raw-r AB2 | node allocation |
| RF-2M-Warp | uniform τ, \(U=w/q\) のAB2 | 真のcoordinate変換 |
| Heun / Midpoint | 同じ物理区間上の2-NFE法 | NFE追加の基準 |
| RK4 / Dopri5 | 高精度ODE参照 | endpoint誤差の基準 |

warped nodeのarmでは、状態更新はraw-r AB2のままにする。真のcoordinate warpでは同じ対応ノードで \(w/q\) を外挿する。この二つは一般に同じ離散法ではない。

NFEが同じであるだけでは公平な比較にならない。checkpoint、初期noise、prompt、CFG、解像度、model時刻の端点を揃える。reference solverも同一velocity callbackを使う。

## 6. Warp familyと数値契約

すべてのwarpはsolverで実際に使う写像 \(\phi\)、逆写像 \(\phi^{-1}\)、微分 q が数値的に一貫していること。最低条件:

\[
\phi(0)=0,\quad \phi(1)=1,\quad
\int_0^1 q(r)\,dr=1,\quad q(r)>0.
\]

### 6.1 Identity

\[
\phi(r)=r,\qquad q(r)=1.
\]

raw RF-2Mとの一致を確認するcontrol。

### 6.2 Shifted rational

\[
\phi(r)=\frac{s r}{1+(s-1)r},\qquad
q(r)=\frac{s}{[1+(s-1)r]^2},\qquad s>0.
\]

逆写像は r=τ/[s-(s-1)τ]。写像は正規化済みで、qは正かつ有限（端点値はsと1/s）。s=1はidentity。最初の探索候補とする。

### 6.3 未実装warp候補

Power、logit、Beta-CDF、trajectory-calibrated warpは現行APIに実装されていないため、候補式やパラメータはこの仕様から外した。追加する場合は、endpointと微分の数値契約、独立した検証、利用者向けの適用範囲を確認してから設計を追記する。

## 7. qの正則化とendpoint処理

qだけをclampしたりblendしたりして、元のφと同時に使ってはならない。それは \(q=\phi'\) を壊し、\(\tau\) と \(d\tau/dr\) の契約を破る。

warpが端点で ill-conditioned な場合は、まず正の有界密度を定義する。例:

\[
\tilde q(r)=a(r)q_{\mathrm{raw}}(r)+[1-a(r)]q_{\mathrm{id}},
\quad q_{\mathrm{id}}=1,
\]

\[
c=\int_0^1\tilde q(s)\,ds,\qquad
q_{\mathrm{safe}}(r)=\tilde q(r)/c,
\]

\[
\phi_{\mathrm{safe}}(r)=\int_0^r q_{\mathrm{safe}}(s)\,ds.
\]

実装はこの同じ \(\phi_{\mathrm{safe}}\) から逆写像と微分を得る。表引きの場合も、補間法と誤差許容値を記録する。qのboundsはwarp構築時に満たし、runtimeの分母clampで隠さない。条件を満たせないwarp parameterはエラーにする。

## 8. 残差安定化

基準RF-2M-WarpではRMS clamp、direction gate、adaptive coefficientをすべてOFFとする。これによりraw AB2との同値性と座標変換自体を独立に評価できる。

clamp ablationを追加する場合は、例えば

\[
R_n=\frac{\operatorname{RMS}(U_n-U_{n-1})}
{\operatorname{RMS}(U_n)+\epsilon}
\]

と上限 \(\rho_{\max}\) を定義する。RMSは各sample内のfeature軸だけで計算しbatch軸を混ぜない。clampが発動した区間では標準AB2の形式次数を主張しない。

adaptive 係数やEuler fallbackも別variantとして識別・記録する。特に係数を曲率に応じて変える方法は基準AB2の係数ではなくなるため、次数と安定性を別途測る。

## 9. 可変step安定性

AB2係数は隣接solver刻み比に依存する:

\[
\alpha_n=\frac{k_n}{2k_{n-1}}.
\]

warpが端点で急変したり、合成schedulerで隣接kが大きく変わる場合、qが有限でも更新が不安定になり得る。v0.1ではfixed gridを使い、少なくとも最大/最小 \(k_n/k_{n-1}\)、\(\alpha_n\)、変換速度のfinite性をwarpごとに記録する。許容step ratioを実験で決め、範囲外のwarp設定を拒否するか、明示的なstep-ratio ablationに分ける。

## 10. API / repository integration

responsibilityを分ける:

- **Time warp / grid builder:** uniform solver-coordinate τ nodesを作り、各nodeに対する \(r=\phi^{-1}(\tau)\)、model time \(t=1-r\)、qを返す。warp種類とparameter、補間精度を管理する。
- **Solver:** callbackを呼び、sampling-direction速度と \(U=w/q\) を計算し、AB2履歴とstate updateを管理する。履歴はcallごとにresetする。
- **Model callback:** \(x\)、shape (B,)のmodel time、conditioningを受け、元の \(dx/dt\) conventionによる速度を返す。

実装APIでは、grid builderとsolverを別概念に保ちつつ、solverが同じnodeのτ/model time/qを受け取れる小さなgrid objectまたは同等の型を定義する。単なるmodel-time列だけではqが失われるため、true warpには不十分。warp solver内でscheduler名にsolver履歴やclamp設定を持たせない。

既存のflow shiftなどと二重warpしない。最初の実装はuniform τとidentity / rational warpに限定する。複数のtime mapを合成するときは写像と導関数の積を明記してから対応する。モデルは常に合成後の実model timeを受け取る。

履歴Uとstateはsolver accumulation dtypeで保持する。FP16/BF16 model callbackには既存契約どおり入力dtypeへcastし、結果をmaster dtypeへ戻す。qは時間のみのスカラーで全batch共通、broadcast軸を明示する。CFGを使う場合はcallbackが返すpost-CFG速度をUへ変換し、同じCFG規則で得た履歴を使う。CFG scaleがstepごとに変わる場合はその設定をmetadataに記録する。

生成metadataにはsolver、warp名とparameter、τ grid種別、step数、residual clamp等のstabilization設定、CFG scale、seed、callback NFEを保存する。RF-2M-Warpは決定論的なので、同一入力・同一初期noise・同一実装で再現可能であることを確認する。

## 11. v0.1 parameter defaults

\[
\text{solver}=\text{rf\_2m\_warp},\quad
\text{coordinate}=\tau,\quad
\text{grid}=\text{uniform},\quad
\text{startup}=\text{warped Euler}.
\]

初期warpはidentityを既定とし、rational warpの探索は別armとする。identityはraw RF-2Mの正規化点である。residual clamp、direction gate、adaptive alphaは既定OFF。q boundsと許容step ratioはwarp構築時のvalidity contractであり、危険値をruntime clampする設定ではない。

この設定は安全な初期実験契約であり、推奨品質設定ではない。

## 12. Validation plan

基礎CPU証拠と未検証範囲は[solver検証索引](sampling-solvers.md)に集約する。追加検証では、
identityが同一grid/startupのraw `rf_ab2`と一致すること、rational warpの端点・逆写像・微分、
analytic ODEでのgrid refinement収束、callback NFEとcall間のhistory resetを確認する。
非identity warpは有限stepでraw法と一致するとは限らない。

モデル比較ではcheckpoint、初期noise、prompt、CFG、解像度、seed、NFEを揃え、endpoint error・
画像品質・時間・memoryを分けて報告する。CPU toy ODEはGPU品質・速度の根拠にならない。
補助diagnosticには、同じτ刻みで次の一次/二次差分比を使える。

\[
R_1=\frac{\operatorname{RMS}(U_n-U_{n-1})}
{\operatorname{RMS}(U_n)+\epsilon},\quad
R_2=\frac{\operatorname{RMS}(U_n-2U_{n-1}+U_{n-2})}
{\operatorname{RMS}(U_n)+\epsilon}
\]

平均・最大・時刻別分布を記録できるが、R2低下だけではendpoint errorや画像品質の改善を示さない。

## 13. 未対応範囲と主張の境界

- 固定gridが対象。adaptive step splittingは対象外。
- 追加学習、追加network、確率過程、score推定、likelihood/inversionは対象外。
- residual clamp等が有効な区間は標準AB2と異なるmethodである。
- monotone warpで連続ODEの軌道は不変だが、有限NFEの結果は変わり得る。
- optimal warpはmodel、conditioning、CFG、解像度に依存し得る。calibration setへの過適合を避ける。
- rectificationが良くvelocityが一定に近いほどidentityが有力になる、というsanity hypothesisを持つ。
- いかなるwarpも品質改善を保証しない。

## 実装状況

共有packageにidentity/rational warped-grid builderとrf_2m_warpを追加し、VFP-DiT生成・観測samplingへ配線した。現在の入口はsolver名rf_2m_warp、warp種別identity/rational、rational parameter shiftである。生成sidecarはeffective solver gridとwarp設定を記録する。residual clamp / gateは実装していない。

CPU testの具体的な範囲と未検証項目は[solver検証索引](sampling-solvers.md)を参照する。配線と限定的なCPU確認は、精度・品質の確認済みを意味しない。

## 14. 参考文献

1. Lu et al., DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models, arXiv:2211.01095 (2022).
2. Lipman et al., Flow Matching for Generative Modeling, ICLR 2023.
3. Liu et al., Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, ICLR 2023.
4. Wang et al., Taming Rectified Flow for Inversion and Editing, arXiv:2411.04746 (2024).
