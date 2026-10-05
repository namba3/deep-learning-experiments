# Flow sampling

Reusable building blocks for flow-matching inference. Schedules are independent
from integration solvers so experiments can compare either dimension while
holding the other fixed.

## API

    import torch
    from flow_sampling import build_timesteps, sample

    times = build_timesteps(30, scheduler="flow_match_euler", shift=1.0)
    images, info = sample(
        velocity_fn, initial_noise, times, solver="fireflow", return_info=True,
    )

The callback contract is velocity_fn(x, t_batch) -> velocity, where x is
(B, ...) and t_batch is (B,). The callback owns model conditioning and any
output guidance; the solver only receives the resulting velocity. Guidance
transforms live in `flow_sampling.guidance` and can be selected independently
of the solver.
Returned velocity must match x in shape and device and be floating point. Its
dtype may differ, which supports autocast model outputs; the solver converts it
to its accumulation dtype. For FP16/BF16/FP32 samples, solver state updates
accumulate in FP32, are cast to the input dtype at each model-call boundary,
and return in the input dtype.

## Schedules

build_timesteps(num_steps, scheduler, shift, start_t, device, dtype) returns
num_steps + 1 descending points, including the final zero endpoint.

- uniform: linear grid. Requires shift=1.
- flow_match_euler: linear grid followed by the static transform
  t' = shift * t / (1 + (shift - 1) * t).

Schedule construction uses FP32 before casting to the requested dtype. Reduced
precision can collapse adjacent points for large step counts; solvers reject
non-strict grids. `warp_timesteps` pairs an existing full descending model-time
grid with its warped progress coordinates and derivative, preserving the
host scheduler's non-uniform grid. Dynamic shifting, terminal shifting, and Karras/exponential/
beta schedules are not implemented, so this is not a full Diffusers scheduler
port.

flow_match_timesteps remains as a compatibility wrapper for the static
FlowMatch schedule.

## Sampler policies

The schedule above chooses the model-time grid. A sampler policy is a separate
choice that selects an update rule for each interval. The current sample API
uses one solver for the full grid; per-interval switching is not implemented
yet.

The [Sampler Scheduler for Diffusion Models](https://arxiv.org/abs/2311.06845)
paper motivates mixing update rules across the trajectory, including SDE
updates early and ODE updates later. Its equations
are written for a diffusion data-prediction/VE parameterization. VFP-DiT
predicts straight-flow velocity, so those update coefficients are not a
drop-in implementation here. The current API applies one solver to the full
grid; it does not implement per-interval solver plans or the paper's adaptive
switching method.

## Solvers

The non-default `rf_*` solvers are experimental research implementations. Their
availability in the API does not imply that they are recommended for sampling:
focused numerical and model-quality validation is still pending for the variants
listed in the [solver design index](../docs/sampling-solvers.md). The default
remains `euler`. The CPU reference tests do not establish GPU/BF16 behavior or
generated-image quality.

- euler: explicit Euler, one callback evaluation per interval.
- fireflow: reuses the previous midpoint velocity; two evaluations on the
  first interval and one on each later interval. Its implementation follows
  the modified midpoint update; paper results do not establish quality on a
  different model.
- abm2: fixed-grid variable-step AB2 predictor / trapezoidal corrector with
  a Heun startup. It evaluates once on startup and once per later interval
  using a PECE approximation. Adaptive step control is not included.
- er_sde: first-order stochastic solver using the VFP-DiT reference
  parameterization sigma=t/(1-t), y=x_t/(1-t), and f(sigma)=sigma^2.
  It requires a starting time below one. Pass a seeded torch.Generator for
  reproducible noise.
- rf_ab2: deterministic variable-step Adams-Bashforth 2 in positive sampling
  progress, with Euler startup. With zero stochastic scale, the 2M solver uses
  this history rule.
- rf_2m_warp: deterministic AB2 over transformed velocity in a monotone
  progress coordinate. Use build_warped_timesteps to obtain aligned descending
  model times, ascending uniform tau nodes, and q=d tau/dr. v0.1 supports
  identity and rational warps; it requires a full [1, 0] model-time grid.
  Residual clipping is disabled in the reference update.
- rf_trust_region: deterministic RF-AB2 with a per-sample trust gate on the
  history correction. Startup is Euler, then ungated AB2 for one step; from
  step two, the gate uses the previous-step variable-grid velocity prediction
  error and gamma=exp(-lambda*error). It uses one callback evaluation per
  interval, has no radius clamp, and is a heuristic rather than an error
  estimator. Pass rf_trust_lambda to sample; default is 4.0.
- rf_er_sde_1: time-only Euler-Maruyama for the linear RF path, with score
  reconstructed from the current state and velocity. The initial envelope is
  g(r)=eta*sin(pi*r), set to zero at and after the progress cutoff.
- rf_er_sde_2m: the same stochastic update with variable-step AB2 applied only
  to the sampling-direction velocity; the score drift remains at the current
  evaluation. This is not claimed to be a second-order SDE method.
- rf_er_sde_trust: unwarped variable-grid trust-gated AB2 plus experimental
  trust-error-gated stochasticity. It uses the same relative per-sample
  prediction error for the AB2 correction gate and diffusion envelope; the
  first two intervals are deterministic. The state/history-dependent SDE has
  no same-marginal guarantee. Configure `rf_er_sde_trust_lambda`,
  `rf_er_sde_trust_error_c`, and `rf_er_sde_trust_epsilon` separately from the
  deterministic `rf_trust_region` gate.
- rf_er_sde_warp_1 / rf_er_sde_warp_2m: time-changed RF-ER-SDE variants using
  an ascending uniform-tau grid and identity/rational warp. Drift is divided by
  q=d tau/dr and Brownian amplitude by sqrt(q); 2M history applies only to the
  transformed RF velocity. They require flow_shift=1 and accept the same eta
  and progress cutoff options as the unwarped RF-ER-SDE solvers.
- rf_er_sde_warp_trust: gates warped AB2 history with a per-sample prediction
  error and uses that error to scale stochasticity. Its first two intervals use
  deterministic startup because the prediction error is unavailable. Configure
  it with `rf_er_sde_warp_trust_lambda`, `rf_er_sde_warp_trust_error_c`, and
  `rf_er_sde_warp_trust_epsilon`; the state-dependent diffusion is empirical
  and has no time-only same-marginal guarantee.

RF solvers require a grid in [0, 1] and use positive progress r=1-t. The
time-only continuous SDE has the RF marginal-preservation relation only with
the exact score; discretization, CFG, and approximate model scores weaken that
claim. `rf_er_sde_eta=0` gives Euler for `rf_er_sde_1` and
`rf_er_sde_warp_1`, and deterministic multistep updates for the 2M variants.
For `rf_er_sde_trust`, the AB2 history correction remains trust-gated even when
stochasticity is disabled.

sample(..., return_info=True) returns (samples, SamplingInfo) with solver,
interval count, and callback-evaluation count. A callback may perform more
than one model forward for CFG.

## Guidance transforms

`flow_sampling.guidance` exposes `classifier_free_guidance`,
`tangential_damping_cfg`, and `apply_guidance`. Both transforms accept paired
unconditional and conditional model outputs with matching shape, device, and
dtype. They do not depend on solver or scheduler APIs.

- `cfg`: `u + w * (c - u)`; this remains the default.
- `tcfg`: independently per batch item, flatten `u` and `c`, form the 2 x D
  matrix with rows `[c, u]`, compute its reduced SVD in FP32 (FP64 for FP64
  inputs), project `u` onto the leading right singular direction, then apply
  `u_hat + w * (c - u_hat)`. The final guided tensor returns in the model
  output dtype. It adds a small 2 x D SVD per sample to the two model forwards.

The VFP-DiT generation CLI selects this with `--guidance-method tcfg`; training
observations use `--sample-guidance-method tcfg`. Scale 1 reduces exactly to
conditional output and avoids the unconditional model call. As a score/velocity
geometry heuristic, TCFG changes the model output before integration; the
solver's numerical order or SDE marginal statements do not automatically carry
over to this guided field. Image-quality validation is pending.

## Flow convention

The callback returns dx/dt = velocity_fn(x, t). For the path
x_t = (1-t) * data + t * noise, ODE methods integrate from t=1 down to t=0.
The RF variants use the equivalent positive-progress coordinate r=1-t.
ER-SDE starts at the finite sigma_max endpoint, so its initial pure-noise
condition is approximate.

## VFP-DiT integration

vfp_dit.samplers.sample_flow_matching is the model adapter. The
sample-generation CLI and training observation options accept all listed
solvers, plus the uniform and flow_match_euler schedules. Defaults remain euler and flow_match_euler. RF-2M-Warp and RF-ER-SDE-Warp
own a uniform tau grid and reject a non-unit flow shift so the existing
schedule is not silently composed with the warp. `rf_er_sde_warp_trust` also
records its prediction gate and startup settings. Their JSON metadata records
the warp family, parameter, solver grid, and q bounds. RF-ER-SDE variants also
record eta, time envelope, and cutoff. Each generated image grid writes a JSON
sidecar with solver, scheduler, shift, steps, guidance, seed, and callback/NFE
accounting. CFG may make two network forwards per callback evaluation. The
RF-ER-SDE-Warp-Trust prototype has no same-marginal claim and remains pending
analytic predictor checks and VFP-DiT quality validation. The unwarped
`rf_er_sde_trust` applies the same experimental trust-error coupling directly
on the input progress grid; see the solver design record in the source repository.
The cross-solver design index and validation boundaries are collected in
[`docs/sampling-solvers.md`](../docs/sampling-solvers.md).


## References

- [FlowMatchEulerDiscreteScheduler](https://huggingface.co/docs/diffusers/v0.33.1/api/schedulers/flow_match_euler_discrete)
- [FireFlow](https://arxiv.org/abs/2412.07517)
- [ABM Solver](https://arxiv.org/abs/2503.16522)
- [ER-SDE-Solver](https://github.com/QinpengCui/ER-SDE-Solver)
- [TCFG: Tangential Damping Classifier-free Guidance](https://arxiv.org/abs/2503.18137)
