# Flow-sampling solver design index

The [flow-sampling package README](../flow_sampling/README.md) documents the current API, defaults, and update rules. This index links each experimental RF solver to its design rationale and states the tracked CPU evidence and remaining validation boundary.

<a id="shared-time-velocity-and-grid-contract"></a>
## Shared time, velocity, and grid contract

The model callback uses model time `t`, with noise at `t=1` and data at `t=0`, and returns the model-time velocity `v(x,t)=dx/dt`. Solvers sample in the opposite direction, so use positive progress and velocity

\[
r=1-t,\qquad w(x,r)=\frac{dx}{dr}=-v(x,1-r).
\]

An unwarped grid increases from `r=0` to `r=1`, with positive interval $h_n=r_{n+1}-r_n$. A warped solver uses `tau=phi(r)`, $q(r)=d\phi/dr>0$, and transformed velocity

\[
U(x,\tau)=\frac{dx}{d\tau}=\frac{w(x,r)}{q(r)}=-\frac{v(x,1-r)}{q(r)}.
\]

Its solver grid also increases and has positive intervals $k_n=\tau_{n+1}-\tau_n$. In both cases, the model callback still receives descending model time `t`; it must not receive `r` or `tau`. These sign, direction, and callback rules are shared by the designs below. The RF-ER-SDE score identity is path-specific and remains defined in its [design document](rf-er-sde-design.md#3-velocity-derived-score).

| Solver design | Current CPU test evidence | Still unverified |
|---|---|---|
| [RF-2M-Warp](rf-2m-warp-design.md) | Constant-velocity identity-warp integration, callback count, and first-step rational-warp derivative scaling. | Analytic ODE convergence, identity equivalence to `rf_ab2`, non-constant warped trajectories, GPU behavior, and VFP-DiT quality. |
| [RF-Trust-Region](rf-trust-region-solver-design.md) | Constant-velocity integration and a variable-grid nonlinear case compared with a hand-computed trust-gated reference. | Analytic ODE convergence, per-sample batch independence, GPU behavior, and VFP-DiT quality. |
| [RF-ER-SDE](rf-er-sde-design.md) | Constant-velocity deterministic limit for `rf_er_sde_1`/`rf_er_sde_2m`; one seeded stochastic first step for `rf_er_sde_1`. | Analytic score and marginal checks, convergence, non-constant 2M history behavior, repeated-call RNG behavior, GPU/BF16, and VFP-DiT quality. |
| [RF-ER-SDE-Trust](rf-er-sde-trust-design.md) | Constant-velocity deterministic limit and a variable-grid nonlinear `eta=0` case compared with a hand-computed trust-gated reference. | Trust predictor and stochastic branch checks, analytic score/marginal checks, batch independence, seeded stochastic repeatability, GPU/BF16, and VFP-DiT quality. |
| [RF-ER-SDE-Warp](rf-er-sde-warp-design.md) | Constant-velocity identity-warp limit, callback count, and first-step rational-warp derivative scaling for the Euler and 2M variants. | Analytic score/marginal checks, convergence, non-constant 2M history behavior, seeded stochastic repeatability, GPU/BF16, and VFP-DiT quality. |
| [RF-ER-SDE-Warp-Trust](rf-er-sde-warp-trust-design.md) | Constant-velocity identity-warp `eta=0` path, callback count, and first-step rational-warp derivative scaling. | Trust prediction gate, state-dependent stochastic update, analytic score/marginal checks, batch independence, seeded repeatability, GPU/BF16, and VFP-DiT quality. |

These are implementation-status notes, not sampler recommendations or quality evidence. Passing the listed CPU cases establishes only those specific update paths. It does not establish convergence, marginal preservation, generated-image quality, or GPU performance. For time-only ER-SDE, same-marginal reasoning additionally requires the stated linear path and exact score; the trust-gated, state-dependent diffusion variants make no same-marginal claim.
