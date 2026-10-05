# Experiment aggregates and analyses

[English](README.en.md) | [日本語](README.md)

This directory contains reusable aggregate values and dataset analyses, not raw data or per-run artifacts. For current implementation behavior and recommended settings, see the [VFP-DiT README](../../vfp_dit/README.md).

## Model measurements

- [Qwen Image VAE latent statistics](image-gen-qwen-latent-statistics.md): historical aggregate over 10,000 Flickr30k images; descriptive only and not a runtime normalization contract

## Dataset analysis

- [Mini-ImageNet GQA resolution and aspect buckets](mini-imagenet-gqa-bucket-analysis.md): cached dataset dimensions, aspect-ratio distribution, and proposed compute-matched buckets

## Archived experiment summaries

The following VFP-DiT results are screens from the retired `vfp_dit_simple` implementation. They do not describe the quality or recommended settings of the current `vfp_dit` implementation.

- [VFP-DiT resource screening](vfp-dit-resource-screening.md): historical short-run memory and throughput aggregates, not current performance claims

## Dated implementation probes

- [VFP-DiT implementation probes (2026-09-24)](vfp-dit-implementation-probes-2026-09-24.md): short pipeline and resource checks, not evidence of quality, convergence, or general performance

## Comparisons and aggregate reports

- [Adapter ablation](vfp-simple-adapter-ablation.md): one-epoch comparison of three adapter variants and gradient aggregates
- [Learning-rate screen](vfp-dit-simple-lr-screen.md): constant-LR and cosine follow-up results, plus sampler comparisons
- [Output-refinement comparison](vfp-dit-simple-output-refinement-comparison-20260929.md): short screen comparing five configurations

## Additional screen aggregates

Per-run progress and interval observations are not retained. Only final-epoch aggregates are listed below. These are seed-42 screens from the retired `vfp_dit_simple` implementation and do not establish quality or recommended settings for the current `vfp_dit` implementation.

| Screen | Main conditions | Final train / validation loss | T2I / TI2I validation loss | Train images/s | Peak allocated / reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 128px pipeline screen | 128px, width 384, depth 4, 10 epochs | 1.1406 / 1.1424 | 1.1769 / 1.1156 | 4.153 | 1.99 / 2.06 |
| 512px quality screen | 512px, width 1024, depth 24, LR `1e-4`, 10 epochs | 1.1995 / 1.1952 | 1.1580 / 1.2324 | 0.962 | 4.92 / 5.68 |
| 512px quality screen | 512px, width 1024, depth 24, LR `1e-3`, 10 epochs | 0.2884 / 0.2851 | 0.2826 / 0.2876 | 1.424 | 5.80 / 6.55 |

The final screen's progress snapshot showed `checkpointing`, so completion of the final save was not verified. The values above are from the recorded epoch metrics.

The public archive retains aggregate values and interpretation limits, but not raw data, per-run artifacts, or cleanup records.
