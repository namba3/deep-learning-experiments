# Qwen Image VAE latent statistics

This is a historical aggregate copied from the VAE portion of `image_gen/latent_stats.json`. The source JSON also contained teacher/semantic statistics from an earlier pipeline; those fields are omitted because the current `image_gen` path does not use or produce them.

These measurements describe one dataset/model sample and are not a runtime normalization contract, quality result, or guarantee for another dataset or VAE revision. The original artifact did not record its run date or VAE parameter dtype.

## Recorded conditions

| Field | Value |
| --- | --- |
| Dataset | `lmms-lab-encoder/flickr30k` |
| VAE | `Qwen/Qwen-Image` |
| Images measured | 10,000 |
| Nominal image size / bucket step | 256 / 32 px |
| Latent scale | 1.0 |
| Shape probe | `(1, 16, 28, 40)` |

## Aggregate values

The aggregate covers 10,725,888 latent scalar values across 16 channels.

| Measure | Value |
| --- | ---: |
| Global mean | 0.008092 |
| Global standard deviation | 0.811829 |
| Global RMS | 0.996165 |

| Channel | Mean | Standard deviation |
| ---: | ---: | ---: |
| 0 | -0.305040 | 0.896606 |
| 1 | 0.223484 | 0.584735 |
| 2 | -0.734417 | 0.949580 |
| 3 | 0.010313 | 1.109666 |
| 4 | -0.226494 | 0.604290 |
| 5 | -0.083056 | 0.720464 |
| 6 | 0.238228 | 0.776309 |
| 7 | -0.202363 | 0.844105 |
| 8 | 0.269962 | 1.045131 |
| 9 | -0.295183 | 0.782604 |
| 10 | -0.145076 | 0.846032 |
| 11 | 0.883977 | 0.522314 |
| 12 | 0.251107 | 0.604457 |
| 13 | -1.430596 | 0.539698 |
| 14 | 0.936681 | 1.043185 |
| 15 | 0.737943 | 0.792817 |

The current training and sampling paths use the VAE configuration scaling factor (or an explicit `--latent-scale` override); they do not load this statistics document.
