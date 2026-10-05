# Archived Mini-ImageNet GQA launchers

The condition-specific launchers are grouped under [`archive/`](archive/). They preserve fixed-seed and fixed-variant protocols from past screens; they are not current experiment recommendations.

Use the [Mini-ImageNet GQA README](../README.md#launcher-index) for current training entrypoints and the [results archive](../../docs/mini-imagenet-gqa-results.md) for aggregate measurements and completion notes. Run an archived launcher from the repository root only when reproducing its recorded protocol.

The public results record also describes three protocols without completion evidence: the seed-45 six-condition scale sweep, the seed-45 query-only shift probe, and the seeds-47–50 AdamW-SF scale-plus-shift bias sweep. Their one-off launchers were omitted from this public archive; the results page retains the protocol status and limits.

The seven retained launchers each map to a protocol family with aggregate results in the public record. Their seed ranges, variant sets, optimizer options, and resume behavior differ, so they remain separate to keep each reproduction command explicit. Launcher presence alone does not establish that every run completed; use the results record for reported evidence.

## Launcher index

| Protocol family | Archived launchers | Public aggregate record |
| --- | --- | --- |
| Ada scale follow-ups, seeds 46–50 | [`run_ada_scale_seed46.sh`](archive/run_ada_scale_seed46.sh), [`run_ada_scale_seeds47_50.sh`](archive/run_ada_scale_seeds47_50.sh) | [Aggregate evidence](../../docs/mini-imagenet-gqa-results.md#aggregate-evidence-recorded-here) |
| Ada scale-function and optimizer follow-ups | [`run_ada_oneplus_softplus_halfnorm_optimizer_seeds47_50.sh`](archive/run_ada_oneplus_softplus_halfnorm_optimizer_seeds47_50.sh), [`run_ada_scale_adamw_sf_seeds47_50.sh`](archive/run_ada_scale_adamw_sf_seeds47_50.sh), [`run_adamw_sf_normalized_silu_seeds47_50.sh`](archive/run_adamw_sf_normalized_silu_seeds47_50.sh) | [Scale-function screen](../../docs/mini-imagenet-gqa-results.md#bounded-ada-scale-function-screen), [AdamW-SF follow-up](../../docs/mini-imagenet-gqa-results.md#adamw-sf-optimizer-follow-up) |
| APOLLO + AdamW-SF warmup and norm-growth limiter | [`run_ada_scale_apollo_warmup_seeds47_50.sh`](archive/run_ada_scale_apollo_warmup_seeds47_50.sh), [`run_apollo_no_norm_growth_limiter_seeds47_48.sh`](archive/run_apollo_no_norm_growth_limiter_seeds47_48.sh) | [APOLLO + AdamW-SF](../../docs/mini-imagenet-gqa-results.md#apollo--adamw-sf-fallback-with-lr-warmup), [Limiter ablation](../../docs/mini-imagenet-gqa-results.md#apollo-norm-growth-limiter-ablation) |

```bash
bash mini_imagenet_gqa/experiments/archive/<script>.sh
```

These launchers start training and write outputs below `mini_imagenet_gqa/output/`. Check each script's defaults, required checkpoints, CUDA needs, and output directory first.
