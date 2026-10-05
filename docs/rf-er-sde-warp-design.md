# RF-ER-SDE-Warp design

**Version:** v0.1.1 reviewed design
**Status:** Prototype implemented; analytic and model-quality validation pending
**Scope:** Linear Rectified Flow sampling with a deterministic time warp and a time-only stochastic correction

This reviewed design follows the conventions in [RF-ER-SDE](rf-er-sde-design.md) and [RF-2M-Warp](rf-2m-warp-design.md). It specifies the warped time-only stochastic correction and its current validation limits.

## 1. Review decisions

- Start with deterministic, time-only $g(r)$. A curvature/history gate is a separate experimental process and has no same-marginal guarantee from the time-only derivation.
- Apply 2M history only to the transformed RF velocity. Keep the current-step score drift and Brownian increment outside the history extrapolation.
- Treat the resulting method as a multistep-corrected Euler-Maruyama scheme. Do not claim second-order SDE accuracy without a separate stochastic convergence analysis.
- This is a solver using a warp grid, not a scheduler. The grid builder owns τ, inverse mapping to model times, and q; the solver owns score/drift evaluation, history, and RNG.
- Exclude residual clipping, runtime q clamping, curvature gating, trust-region control, and adaptive steps from the reference variant. Add them only as separately named ablations after the baseline is numerically validated.

## 2. Shared RF path and score contract

The linear path, descending model-time convention, and callback velocity are defined in
[RF-ER-SDE's model and time contract](rf-er-sde-design.md#2-model-and-time-contract). The velocity-derived
score and continuous time-only SDE are defined in the [score identity](rf-er-sde-design.md#3-velocity-derived-score)
and [continuous process](rf-er-sde-design.md#4-continuous-time-stochastic-process). Use that score identity;
the expression `-(x-tv)/(1-t)` is incompatible with this path's model-time and velocity convention.
This variant changes only the solver coordinate: it uses increasing progress `r=1-t` and the warp
`tau=phi(r)` with `q=d tau/dr > 0`. The tau-grid and transformed update below are specific to this
solver. The stochastic branch must remain off at the singular `t=0` endpoint; the exact-score
marginal statement applies only under the assumptions in the shared design.

## 3. Time change and discrete reference update

Under τ=φ(r), drift coefficients divide by q and Brownian amplitudes divide by \(\sqrt q\). Define

\[
g_n=g(r_n),\quad
D_n=\frac{g_n^2}{2q_n}s_{t_n}(x_n),\quad
\sigma_{\tau,n}=\frac{g_n}{\sqrt{q_n}},\quad
U_n=-\frac{u_n}{q_n},\qquad q_n=q(r_n).
\]

For (k_n=\tau_{n+1}-\tau_n>0), the first-order warped Euler-Maruyama reference is

\[
x_{n+1}=x_n+k_n(U_n+D_n)+\sigma_{\tau,n}\sqrt{k_n}\,z_n,
\qquad z_n\sim\mathcal N(0,I).
\]

The 2M variant extrapolates only the deterministic RF velocity. For $n\ge1$, with $k_{n-1}=\tau_n-\tau_{n-1}>0$,

\[
\bar U_n=U_n+\frac{k_n}{2k_{n-1}}(U_n-U_{n-1}).
\]

Use Euler startup at $n=0$, setting \(\bar U_0=U_0\). Then

\[
x_{n+1}=x_n+k_n(\bar U_n+D_n)+\sigma_{\tau,n}\sqrt{k_n}\,z_n.
\]

Save raw $U_n$, not \$\bar U_n$. The score drift is evaluated at the current state and is not extrapolated. Full-drift extrapolation is a separate ablation. With $g=0$, the method reduces to deterministic RF-2M-Warp after startup. With history disabled, it reduces to warped RF-ER-SDE-1. Identity warp $q=1,\tau=r$ must reproduce the corresponding unwarped RF-ER-SDE discrete update when grid, startup, dtype, and random draws match.

The time-change identity is continuous. For a finite step and variable $g$, (\sigma_{\tau,n}^2 k_n=g_n^2 k_n/q_n\) is a left-point Euler-Maruyama approximation to the integrated variance. Do not describe it as an exact finite-step variance under nonlinear φ. Adaptive refinement/rejection is out of scope; if later introduced, preserving an SDE path requires Brownian-bridge/tree bookkeeping.

## 4. Warp contract and initial family

The grid builder must return aligned solver progress, descending model times, and q values from the same mathematical warp. Require endpoint mapping, strict monotonicity, finite positive q, and a numerically consistent inverse/derivative. Check q bounds at warp construction. Do not clamp q independently at runtime: that would make the applied derivative inconsistent with φ and its inverse.

Initial families:

- **Identity:** \(\phi(r)=r,\ q=1\). Required equivalence control.
- **Rational:** \(\phi(r)=\frac{s r}{1+(s-1)r}\), \(q(r)=\frac{s}{[1+(s-1)r]^2}\), $s>0$. Its inverse is \(r=\frac{\tau}{s-(s-1)\tau}\). Start with this family because its endpoints and derivative are explicit and normalized.

Beta-CDF or other families with unbounded endpoint derivatives are out of the initial variant unless boundedness and inverse/derivative consistency are established. The initial grid is fixed and uniform in τ. Use q_min/q_max as validation bounds that reject an invalid warp, not as a second transformation.

## 5. Diffusion schedule and terminal handling

Use the baseline

\[
g(r)=\eta\sin(\pi r),
\]

with a documented cutoff $r\ge r_{cut}$ that sets $g=0$ near the data endpoint. The existing RF-ER-SDE prototype uses η=0.2 and cutoff 0.9 as wiring defaults; they are not tuned recommendations. Keep them aligned only if this solver shares the same implementation configuration. Curvature gating, alternative exponents, and warp-dependent noise schedules are separate experiments.

When the cutoff disables stochasticity, bypass score evaluation and RNG draws. The full model-time grid still reaches $t=0$, where deterministic integration can continue. The cutoff is defined in progress r, not in the descending model time t; document the comparison boundary precisely.

## 6. Numerical and tensor contract

- The state and model output retain the existing callback's shape, device, and model-input dtype. Solver-time scalars (t, r, \tau, q, g) broadcast over non-time axes only.
- Compute the reconstructed score and SDE drift in FP32 (FP64 for FP64 inputs); cast the model input to the model parameter dtype according to the existing sampler boundary.
- Curvature, if later added, is per-sample over feature axes only. Never reduce across batch; one sample's noise scale must not depend on other samples in the batch.
- Generate $z_n$ with the supplied generator and the state shape/device. Keep draws independent per fixed-grid step. Reset history for each sample call.
- Ensure all coefficients and updates are finite. Verify broadcasting for image `(B,C,H,W)` and token `(B,T,D)` states and nonuniform positive solver increments.
- Count callback invocations as model evaluations. CFG may perform more than one network forward inside a callback, so report both callback NFE and actual forward count when available.

## 7. API and metadata

The shared solver registry exposes `rf_er_sde_warp_1` and `rf_er_sde_warp_2m`; these do not overload `rf_er_sde_1`, `rf_er_sde_2m`, `rf_2m_warp`, or the legacy `er_sde`.

Responsibilities:

- Scheduler/grid builder: create \(\tau_n\), inverse-map to descending $t_n$, provide aligned $q_n$, and validate the warp.
- Solver: call the velocity callback, reconstruct score only if (g_n\ne0), apply the selected history rule, draw Brownian noise, and reset per-call state.
- Generation entrypoint: pass the grid/solver configuration and persist it with artifacts.

Record solver name/version, base timestep scheduler and shift, warp family/parameter, q bounds, step count, η and time envelope, cutoff in r, history mode, score precision, CFG configuration, seed, callback NFE, and actual model-forward count if available. Checkpoint/sample metadata and resume/reconstruction should not silently infer omitted warp settings.

## 8. Validation plan

現在のCPU証拠と未検証範囲は[solver検証索引](sampling-solvers.md)を参照する。追加検証は、warpと
微分の契約、analytic score/marginal、identity/zero-noise退化とgrid refinement、seeded repeatability・
batch independence・dtype境界の順に行う。その後にmatched checkpoint/NFEの画像比較へ進む。
Marginal preservationはtime-only diffusionとexact scoreの条件に限り、guidedまたはstate-dependentな
stochastic variantには同じ主張を適用しない。品質・runtime・memory・数値誤差は別々に報告する。

## 9. Prototype implementation status

The fixed-grid prototype is wired into `flow_sampling.sample`, the VFP-DiT adapter, generation CLI, and training observation CLI. It uses the identity/rational `build_warped_timesteps` grid and persists warp type/shift with solver metadata. The recorded defaults are integration values, not tuned recommendations.

Analytic score/marginal, convergence, identity-equivalence, seeded-repeatability, CUDA/BF16, and VFP-DiT quality checks remain pending. Treat this as an experimental sampler until those checks are completed.

## 10. Known limits and research hypotheses

- Exact same-marginal reasoning requires the stated linear path, a time-only diffusion schedule, and the exact population score. Approximate model score and CFG weaken this claim.
- The finite-step warped solver is an approximation. The continuous time reparameterization does not ensure that a particular warp improves quality or endpoint error at a given NFE.
- RF-ER-SDE-Warp-2M does not have an established second-order stochastic convergence order.
- Fixed τ grid only; adaptive stepping and Brownian path reuse are deferred.
- The initial design does not combine trust-region control or residual clipping with stochasticity.

Research hypotheses to test, not assumptions:

1. A suitable τ coordinate can reduce deterministic RF history error at low NFE while the correctly transformed continuous SDE remains in the same marginal family.
2. Extrapolating only RF velocity while evaluating score drift at the current state is more stable than extrapolating the full stochastic drift.
3. Ending stochastic activity before $t=0$ avoids score singularity and may preserve endpoint detail; image-quality impact remains empirical.
