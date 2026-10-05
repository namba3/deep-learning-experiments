# RF-ER-SDE-Trust design

**Status:** Implemented as an experimental shared solver; numerical and model-quality validation pending
**Scope:** Unwarped linear Rectified Flow on the supplied descending time grid, with variable-grid trust-gated AB2 and trust-error-gated Euler-Maruyama terms.

This is the unwarped counterpart of [RF-ER-SDE-Warp-Trust](rf-er-sde-warp-trust-design.md). The deterministic `rf_trust_region` remains a separate solver.

## Contract and update

This is the unwarped specialization of [RF-ER-SDE-Warp-Trust](rf-er-sde-warp-trust-design.md): set `tau=r=1-t`, `q=1`, and `U=w=-u`. Use its prediction-error, trust-gate, startup, and stochastic-update definitions with `k_n` replaced by the supplied-grid interval `h_n=t_n-t_(n+1)>0`. The callback receives strictly descending model times; non-uniform grids are supported.

The first two intervals use deterministic Euler/ungated-AB2 startup because no prediction error is available. The stochastic branch remains disabled near `t=0`; the score identity and its singular endpoint are defined in [RF-ER-SDE](rf-er-sde-design.md#3-velocity-derived-score). Since the diffusion gate depends on model output and trajectory history, this solver has no same-marginal guarantee.

## API

The `flow_sampling.sample` solver name is `rf_er_sde_trust`. The caller supplies a strictly descending model-time grid, including non-uniform grids; warped variants also pair model times with solver progress and `tau/q` metadata. Parameters are `rf_er_sde_eta`, `rf_er_sde_cutoff`, `rf_er_sde_trust_lambda`, `rf_er_sde_trust_error_c`, and `rf_er_sde_trust_epsilon`. Random draws use the caller's `torch.Generator`; state/history use the solver accumulation dtype and per-sample reductions over every non-batch dimension.

## Validation boundary

Before treating this as a quality-improving sampler, validate variable-grid predictor coefficients and error values, zero-velocity/epsilon behavior, batch independence, deterministic repeatability with `eta=0`, trust-gate limits, score identity at interior times, skipped RNG when `g=0`, seeded stochastic repeatability, FP32/BF16/FP64 boundaries, and non-finite diagnostics. Compare matched NFE and seeds against `rf_er_sde_2m`, `rf_trust_region`, and `rf_er_sde_warp_trust`. Current implementation status is not quality or marginal evidence.
