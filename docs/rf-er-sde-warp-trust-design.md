# RF-ER-SDE-Warp-Trust design

**Version:** v0.1.1 reviewed design

**Status:** Prototype wired into the shared solver and VFP-DiT paths; focused numerical and model-quality validation pending
**Scope:** Fixed-grid linear Rectified Flow sampling with a time warp, prediction-agreement-gated 2M correction, and experimental trust-error-gated stochasticity

This reviewed design combines the trust-region gate with the warped stochastic solver. It follows [RF-Trust-Region](rf-trust-region-solver-design.md), [RF-ER-SDE-Warp](rf-er-sde-warp-design.md), and [RF-2M-Warp](rf-2m-warp-design.md), and records the current validation limits.

## 1. Review decisions

- Use the repository convention: noise at model time `t=1`, data at `t=0`, callback `u=dx/dt`, and descending model times.
- Write updates in positive progress `r=1-t` and warped progress `tau=phi(r)`. Define `q=d tau/dr > 0`, `w=-u`, and `U=dx/dtau=w/q=-u/q`.
- Calculate trust prediction error and its gate from raw transformed RF velocities `U`, not from score drift or stochastic increments. Use the variable-step predictor even though the initial grid is uniform in tau.
- Gate only the 2M RF-velocity correction. Evaluate the distributional score drift at the current state.
- Treat error-gated stochasticity as an empirical variant. Its scale depends on state and velocity history, so the time-only same-marginal derivation does not apply. “Guardrail” is a hypothesis, not a guarantee.
- Compare against a deterministic time-only SDE envelope to isolate the effect of coupling SDE strength to trust error.
- Defer adaptive trust radius, residual clipping, curvature/direction gates, composite errors, adaptive time steps, and model-evaluation skipping.
- Implement as the distinct shared solver `rf_er_sde_warp_trust`, using the existing identity/rational warped-grid builder. This is a solver, not a scheduler, and does not change existing solver defaults.

## 2. Model and coordinate contract

The shared linear path, model-time sign convention, and velocity-derived score are defined in
[RF-ER-SDE design](rf-er-sde-design.md#2-model-and-time-contract). The coordinate transformation
and its derivative are defined in [RF-ER-SDE-Warp](rf-er-sde-warp-design.md#2-shared-rf-path-and-score-contract).
This trust variant uses `r=1-t`, `tau=phi(r)`, `q=d tau/dr > 0`, and transformed RF velocity
`U=-u/q`. Its trust predictor operates on `U` in the tau grid; the callback continues to receive
model time `t`.

## 3. Trust predictor and per-sample error

For `n>=2`, use the variable-grid predictor from [RF-Trust-Region](rf-trust-region-solver-design.md#4-prediction-agreement-trust-signal), replacing raw velocity `w` and progress step `h` with transformed velocity `U` and tau step `k`. Keep raw `U` values in the predictor history.

After evaluating `U_n`, compute a per-sample relative prediction error:

\[
e_n=\frac{\operatorname{RMS}_{feature}(U_n-\widehat U_n)}
{\max(\operatorname{RMS}_{feature}(U_n),\epsilon_{abs})}.
\]

Reduce over every non-batch axis and reshape the `(B,)` result to `(B,1,...,1)` only when broadcasting over the state. This keeps one sample from changing another sample's solver coefficients. Use the solver accumulation dtype. `epsilon_abs` is an absolute floor in that dtype; if both RMS values are zero, the resulting error is zero.

The predictor diagnoses the previous extrapolation; it is not a local error estimator, accept/reject test, or guarantee of future accuracy. This method has no objective-based acceptance ratio, so “trust region” is an analogy to a prediction-agreement gate.

Use the exponential mapping from [RF-Trust-Region](rf-trust-region-solver-design.md#4-prediction-agreement-trust-signal) with this solver's `lambda_trust` parameter. It scales only the 2M history correction; its initial value is a wiring default, not a tuned recommendation.

## 4. Startup and trust-gated RF update

A prediction residual needs two preceding velocity samples:

| Current node | Trust estimate | RF velocity update |
|---|---|---|
| `n=0` | unavailable | Euler startup, `Ubar_0=U_0` |
| `n=1` | unavailable | ungated AB2, `gamma_1=1` |
| `n>=2` | `e_n` available | trust-gated AB2 |

For `n>=1`, define

\[
\alpha_n=\frac{k_n}{2k_{n-1}},\qquad
R_n=U_n-U_{n-1},
\]

and the RF transport velocity

\[
\bar U_n=U_n+\gamma_n\alpha_nR_n.
\]

Store raw `U_n`, not gated or extrapolated velocities, in history. The initial grid is fixed and uniform in tau, but retain the step ratio in the implementation so later grid changes cannot silently invalidate the formula.

The baseline has no residual clamp or adaptive radius. A fixed correction clamp is a separate method and removes the unmodified AB2 order claim when active.

## 5. Shared score and transformed SDE terms

Use the shared [RF-ER-SDE score identity](rf-er-sde-design.md#3-velocity-derived-score) and
[time-change rule](rf-er-sde-warp-design.md#3-time-change-and-discrete-reference-update). This
variant reconstructs the score from the raw callback velocity `u_n`, not the transformed history
velocity `U_n`. Evaluate score/drift and draw random numbers only when its trust-gated diffusion
coefficient is nonzero. The branch must remain disabled before `t=0`; changing the score denominator
or applying a finite floor does not extend the identity to the endpoint.

## 6. Trust-error-gated stochasticity

The experimental coupled schedule is

\[
g_n=\begin{cases}
0, & n<2,\\
0, & r_n\ge r_{cut},\\
\eta\sin(\pi r_n)\dfrac{e_n}{e_n+c_e}, & \text{otherwise},
\end{cases}
\qquad \eta\ge0,\ c_e>0.
\]

The same pre-update error controls gamma and the stochastic gate. Because this error depends on model outputs and trajectory history, `g_n` is state/path dependent. The time-only probability-flow/marginal-preservation result does not prove this process has RF marginals. State-dependent diffusion adds spatial-derivative terms to the Fokker-Planck relation, and history dependence also changes the process state. Treat this as a practical stochastic heuristic.

The `n<2` rule means the two startup intervals are deterministic because no prediction-error estimate exists. This is an explicit fallback. A separate time-only `g(r)` control may enable noise during startup and isolate the effect of early noise. Do not substitute a missing prediction error with an assumed perfect prediction.

The terminal cutoff is defined in progress r, where `r>=r_cut` is near data. It is not a high-t cutoff. With `g_n=0`, bypass score calculation and random-number generation; deterministic transport may continue to `t=0`.

## 7. Combined fixed-grid update

Evaluate the callback once at `(x_n,t_n)`, form `U_n`, compute trust/error when available, and update:

\[
x_{n+1}=x_n+k_n(\bar U_n+D_n)
+\sigma_{\tau,n}\sqrt{k_n}\,z_n,
\qquad z_n\sim\mathcal N(0,I).
\]

Use Euler startup at `n=0`, ungated AB2 at `n=1`, and trust-gated AB2 from `n=2`. Score drift and Brownian noise are current-node Euler-Maruyama terms, not multistep-extrapolated terms. Draw noise from the caller-provided `torch.Generator` for active fixed-grid intervals.

For variable `g` and nonlinear warp, `g_n^2 k_n/q_n` is a left-endpoint Euler-Maruyama approximation to interval variance, not exact finite-step Brownian variance. Adaptive splitting/rejection is out of scope; it would require Brownian-bridge/tree path bookkeeping.

## 8. Warp and numerical contracts

Use the existing `build_warped_timesteps` identity/rational families and its paired tau nodes, descending model times, and q values. The prototype uses full endpoints, a fixed uniform-tau grid, and external `flow_shift=1`. Require strict order, finite positive q, inverse consistency, and q bounds at grid construction. Never clamp q independently of phi and its inverse.

State, callback output, velocity history, and Brownian tensor have identical data shape/device. Time coefficients broadcast over non-time axes; trust statistics reduce independently per sample. Reset history and prediction state on every `sample()` call. Check that errors, gates, drifts, and state updates remain finite.

The rational warp is

\[
\phi(r)=\frac{s r}{1+(s-1)r},\qquad
q(r)=\frac{s}{[1+(s-1)r]^2},\qquad s>0.
\]

Identity (`s=1`) is a required equivalence control. Keep warp type/shift separate in the API and metadata from RF-2M-Warp settings.

## 9. API and metadata

The shared registry exposes `rf_er_sde_warp_trust` through `flow_sampling.sample` and VFP-DiT generation/training-observation paths. The grid builder remains a scheduler utility; the solver owns velocity history, prediction error, trust coefficients, score/drift, and RNG. Existing `rf_trust_region`, `rf_er_sde_warp_1`, and `rf_er_sde_warp_2m` remain separate.

Record solver/version, callback evaluations, grid kind, step count and endpoints; warp family/shift/q bounds; predictor and error definition; `lambda_trust`, `c_e`, absolute epsilon and startup behavior; SDE gate mode, eta/time envelope/progress cutoff; CFG scale, seeds and callback/network-forward accounting.

Use the guided callback velocity consistently for transport, trust prediction and velocity-derived score unless running an explicitly named score ablation. A CFG-derived score is not automatically the score of the guided field's marginal; make no exact-score marginal claim for guided sampling.

## 10. Prototype implementation status

The fixed-grid prototype is wired into the shared solver, warped-grid VFP-DiT adapter, sample-generation CLI, and training observation CLI. It uses per-sample transformed-velocity prediction error, an exponential 2M gate, and an error-gated diffusion envelope. The first two intervals use deterministic startup. Metadata records warp/trust/SDE settings and callback counts. These defaults are wiring values, not tuned recommendations.

CPU test coverage and remaining validation are tracked in the [solver verification index](sampling-solvers.md). Wiring and its limited CPU checks are not full numerical or model-quality evidence; treat this as an experimental sampler.

## 11. Comparison, validation, and limits

現在のCPU証拠と未検証範囲は[solver検証索引](sampling-solvers.md)を参照する。品質比較前に、analytic
sequenceでtransformed-velocity predictorとper-sample gateを確認し、score/cutoff、deterministic limit、
seeded stochastic momentsを検証する。次にhistory reset、batch independence、dtype/shape境界を確認する。
Marginal testはexact scoreのtime-only diffusionに限る。モデル比較はcheckpoint、noise、CFG、解像度、
seed、NFEを揃える。

2M補正はEuler-Maruyama更新内のdeterministic RF velocity外挿であり、確立した2次stochastic convergence
はない。Adaptive stepping、Brownian bridge、full-drift extrapolation、tokenwise trust、exact marginal/
likelihood/inversion、quality-improvement claimsはこのprototypeの対象外である。
