# RF-ER-SDE design

**Version:** v0.2.0 design
**Status:** Prototype implemented; numerical and model-quality validation pending
**Scope:** A model-independent stochastic solver for linear flow-matching paths, integrated through the repository's shared flow-sampling API.

This document defines the implementation-facing v0.2.0 design. It fixes the time convention, states where the marginal-preservation result applies, and separates the deterministic multistep correction from stochasticity.

## 1. Decisions

- Keep the shared sampler's model-time convention: noise at t=1, data at t=0, and model output u(x,t)=dx/dt. The sampling grid is strictly descending.
- Derive RF-ER-SDE in the positive sampling-progress coordinate r=1-t. Convert the model velocity to sampling-direction velocity w=-u.
- First implement a time-only diffusion strength g(r). The continuous-time SDE then has the same marginals as the RF probability-flow ODE when the velocity-derived score is exact.
- Treat curvature-gated diffusion g(r,kappa) as a separate heuristic variant. It is state/history dependent and is not covered by the time-only marginal-preservation derivation.
- Separate the first-order RF-ER-SDE from the velocity-history variant. Call the latter a multistep-corrected Euler-Maruyama method; do not claim second-order accuracy for the combined SDE method without a stochastic convergence analysis.
- Keep the existing solver name er_sde and its update unchanged. The new variants use distinct names: rf_er_sde_1 and rf_er_sde_2m.

## 2. Model and time contract

The model is trained on

\[
x_t=(1-t)x_{data}+t\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
\]

with velocity

\[
u_\theta(x_t,t)\approx \frac{dx_t}{dt}=\epsilon-x_{data}.
\]

共有callback、sampling progress、速度符号、正方向gridの規約は[solver共通契約](sampling-solvers.md#shared-time-velocity-and-grid-contract)を参照する。ここでは線形Gaussian pathに固有のvelocity-to-score導出を続ける。SDE varianceには正のsampling intervalを用い、`sqrt(abs(dt))`を逆時間SDEの導出として扱わない。

## 3. Velocity-derived score

For the linear Gaussian path, the velocity and state give estimates

\[
\widehat{x}_{data}=x_t-t u_\theta(x_t,t),\qquad
\widehat{\epsilon}=x_t+(1-t)u_\theta(x_t,t).
\]

The marginal score identity is

\[
s_t(x_t)=\nabla_x\log p_t(x_t)
\approx -\frac{\widehat{\epsilon}}{t}
=-\frac{x_t+(1-t)u_\theta(x_t,t)}{t}.
\]

This identity is exact for the population-optimal RF velocity under the stated path. With a finite model it is an estimate. Its denominator is the noise coefficient t, so it is singular at the data endpoint t=0. The stochastic branch must turn off before that endpoint; when g=0, skip score evaluation rather than relying on a denominator clamp to justify the stochastic update.

Compute score arithmetic in FP32 (or FP64 for FP64 inputs), preserving the solver's master-state and model-callback dtype contract.

## 4. Continuous-time stochastic process

Let r=1-t and let p_r denote the RF marginal at model time t=1-r. For a deterministic, time-only diffusion coefficient g(r), define

\[
dX_r=b(X_r,r)\,dr+g(r)\,dW_r,
\qquad
b(x,r)=w_\theta(x,r)+\frac12g(r)^2s_{1-r}(x).
\]

Under the exact score, its probability-flow velocity is

\[
b-\frac12g(r)^2s_{1-r}=w_\theta,
\]

so this continuous SDE and the RF ODE share their one-time marginals. This is a continuous-time result. It does not promise identical finite-step samples, exact marginals for an approximate score, or exactness under a curvature-dependent diffusion schedule.

For a state-dependent scalar g(x,r), the probability-flow relation also contains the spatial-divergence term \(\frac12\nabla_x g^2\). A coefficient depending on kappa computed from x or velocity history is state/path dependent. The formula above therefore does not establish marginal preservation for g(r,kappa). Keep that variant explicitly experimental unless its corresponding process and correction are derived.

## 5. Discrete solver variants

### 5.1 RF-ER-SDE-1: time-only Euler-Maruyama

For a step from t_n to t_{n+1}, use Δ_n=t_n-t_{n+1}>0 and evaluate u_n=u_theta(x_n,t_n). Define

\[
s_n=-\frac{x_n+(1-t_n)u_n}{t_n},\qquad
g_n=g(r_n),\qquad
b^{sample}_n=-u_n+\frac12g_n^2s_n.
\]

Then

\[
x_{n+1}=x_n+\Delta_n b^{sample}_n
+g_n\sqrt{\Delta_n}\,z_n,\qquad z_n\sim\mathcal N(0,I).
\]

This is Euler-Maruyama in increasing r. When g=0 it reduces exactly to the shared Euler update x_{n+1}=x_n-Δ_n u_n. Use this variant to validate the stochastic and score paths before adding history.

A simple initial schedule may be a deterministic function of sampling progress, for example

\[
g(r)=\eta\sin(\pi r),
\]

with an explicit terminal cutoff r>=r_{cut} that sets g=0 near data. The function, scale, cutoff, and time grid are experimental configuration and must be recorded. The score term is evaluated only when g is nonzero.

### 5.2 RF-ER-SDE-2M: velocity-history corrected EM

For positive progress steps h_n=r_{n+1}-r_n=Δ_n, and h_{n-1}=r_n-r_{n-1}, variable-step AB2 extrapolates the sampling-direction velocity:

\[
\bar w_n=w_n+\frac{h_n}{2h_{n-1}}(w_n-w_{n-1}).
\]

Use Euler startup when no previous velocity exists. Then update

\[
x_{n+1}=x_n+h_n\left[\bar w_n+\frac12g_n^2s_n\right]
+g_n\sqrt{h_n}\,z_n.
\]

This corrects the RF velocity history but leaves the score drift at the current-step value. It is not a second-order SDE method by declaration; measure its convergence and stability separately. An alternative that extrapolates the complete drift is a separate ablation because it also extrapolates the score term.

Optional residual clipping changes the AB2 formula and removes its formal order where clipping activates. Keep it disabled in the reference variant; expose it only as a named stability ablation.

### 5.3 Optional curvature-gated variant

A future variant may set g_n=g(r_n)q(kappa_n), using a per-sample statistic. This is a heuristic schedule, not the time-only marginal-preserving SDE above. Record its exact reduction, clamp, and q function. Never reduce curvature over the batch dimension: each sample's noise scale must be independent of other samples in the batch.

The initial reference implementation should not include this gate. First establish whether time-only stochasticity helps at all.

## 6. Claims and degeneration contracts

- g=0, history disabled: exactly shared Euler.
- g=0, history enabled: deterministic variable-step AB2 after Euler startup.
- history disabled, g nonzero: RF-ER-SDE-1.
- g=0 and history correction disabled: Euler; do not describe eta=0 alone as Euler when history remains enabled.
- Constant velocity: the AB2 residual is zero.
- The continuous time-only SDE preserves RF marginals only under the exact-score assumption. Numerical integration, approximate model output, CFG, residual clipping, and curvature gating each weaken or remove that guarantee.
- CFG output may be used consistently as the solver's velocity, but the reconstructed score is not automatically the score of the original conditional marginal. Report RF-ER-SDE claims with the guidance setting and treat guided stochasticity as empirical unless separately justified.

## 7. API and integration

Implement these as solver variants in flow_sampling, not as timestep schedulers. The scheduler constructs the descending model-time grid; the solver owns per-call history and RNG use.

The callback contract remains velocity_fn(x, t_batch) -> dx/dt with t_batch shaped (B,). Maintain one previous velocity per sample trajectory, in the solver accumulation dtype. Reset all history for each new sample call. Keep model evaluation state dtype, device, shape, and CFG handling under the existing callback contract.

Use distinct public names rf_er_sde_1 and rf_er_sde_2m so the existing er_sde behavior and saved metadata remain interpretable. Record solver name, schedule and shift, step count, SDE scale and envelope, cutoff, history/clamp mode, CFG scale, sample seeds, and callback NFE in generation metadata. An explicit torch.Generator must make repeated calls reproducible.

## 8. Validation checklist

実装配線と検証状況は[solver検証索引](sampling-solvers.md)を参照する。品質を評価する前に、
analytic velocity-to-score/marginal、deterministic limitと非定数2M historyのgrid refinement、
seeded RNG・call-history reset・batch independence・dtype/broadcast境界を確認する。
その後、matched checkpoint/NFEのモデル比較を行い、数値誤差・画像品質・runtime・memoryを分けて
報告する。Marginal preservationを主張できるのは、記載したtime-only diffusionとexact-score条件に
限る。CPU toy testはGPU品質や速度を示さない。

## 9. Known boundaries

- Fixed grids only; adaptive step splitting and Brownian-bridge bookkeeping are out of scope.
- No exact likelihood or inversion claim.
- The current design is for the repository's linear Gaussian RF parameterization. Other path parameterizations need their own velocity-to-score derivation.
- The design does not assert a quality gain. The stochastic scale and terminal cutoff must be measured on a fixed VFP-DiT checkpoint before defaults are considered.
