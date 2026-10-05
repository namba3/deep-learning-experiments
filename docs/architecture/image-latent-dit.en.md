# Image-latent DiT architecture

[English](image-latent-dit.en.md) | [日本語](image-latent-dit.md)

This document records the current image-latent-only configuration in `image_gen/train.py` as an implementation reference. Code and `--help` output are authoritative for current values. Experiment history and incomplete validation are kept in the [repository audit snapshot](../history/repository-audit-2026-09-12.md) and the related experiment reports.

## Purpose and scope

This is a Rectified Flow / Flow Matching DiT that generates image latents from captions using the Qwen Image VAE. The current model predicts only VAE latents. It does not use semantic channels, a vision-encoder teacher, a PCA/whitening projector, a semantic loss, or reference-image semantic overrides.

The DiT and Text Conditioning Adapter are trainable. The Qwen Image VAE and Qwen3.5 text encoder are frozen; feature extraction runs in inference mode with AMP.

## Tensor shape contracts

The standard shapes are:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Image latent | `(B, C, H, W)` | Clean, noise, or interpolated latent from the VAE |
| Text hidden state | `(B, S, E)` | Output from the frozen text encoder |
| Condition tokens | `(B, S, 1024)` | Output from the Text Adapter |
| Latent tokens | `(B, T, D)` | `T = H' * W'`, `D = model_dim` |
| Attention Q | `(B, Hq, Tq, Dh)` | Query heads |
| Attention K/V | `(B, Hkv, Tk, Dh)` | GQA key/value heads |

`H'` and `W'` are the spatial dimensions after the latent passes through the stride-2 `LatentDownsample`. Tokenization is `(B,D,H',W') -> (B,H'W',D)` and restoration reverses that mapping. When image and text tokens are concatenated, image tokens come first and their count is added as an offset to the text mask.

## Model data flow

```text
image
  -> frozen Qwen Image VAE
  -> clean latent x0: (B, C, H, W)

caption
  -> frozen Qwen3.5 text encoder
  -> TextConditioningAdapter
  -> text tokens: (B, S, 1024)

noisy latent
  -> LatentDownsample: (B, 1024, H/2, W/2)
     ├─ flatten -> main latent tokens
     └─ ImageContextEmbedder: three stride-2 convolutions
        -> image digest tokens

image digest tokens + text tokens
  -> ContextTransformer
  -> three-stream joint attention DiT blocks
  -> LatentUpsample
  -> predicted image-latent velocity: (B, C, H, W)
```

### Text Conditioning Adapter

The input is mapped to the first width with RMSNorm and a Linear layer, then passed through a mask-aware, non-causal transformer. Defaults:

```text
text encoder hidden
  -> RMSNorm
  -> Linear(input_dim, 2048)
  -> bidirectional GQA block (dim=2048, heads=16, kv_heads=8)
  -> Linear(2048, 1024)
  -> bidirectional GQA block (dim=1024, heads=8, kv_heads=4)
  -> Identity
  -> bidirectional GQA block (dim=1024, heads=8, kv_heads=4)
  -> RMSNorm
  -> Text Adapter tokens (dim=1024)
```

Text uses 1D RoPE, and blocks share a cache with the same head dimension. Padding tokens are masked. For an empty caption, a safe mask prevents fully masked rows while constructing null conditioning for CFG.

### Main DiT and Context Transformer

Defaults are `model_dim=1024`, `depth=12`, `heads=16`, `kv_heads=8`, `context_depth=2`, `context_heads=16`, `context_kv_heads=8`, and `patch_size=2`. Patch size is fixed at 2 because the current convolutional stem requires stride 2.

The Context Transformer compresses shared image features into image digest tokens with three stages of `Conv2d(kernel=4, stride=2, padding=1) -> GroupNorm -> SiLU`, then concatenates them with text tokens. Image tokens use 2D RoPE; text tokens use 1D RoPE.

Each Main DiT block preserves the latent, image, and text streams and performs:

1. Joint attention with timestep- and resolution-conditioned Ada modulation.
2. An independent dense FFN for each stream, with hidden width `4 * model_dim`.

With `--attention-pattern mhla3-full1`, each four-block cycle contains three MHLA blocks and one Full Joint Attention block. `full` uses Full Joint Attention in every block; `mhla` uses MHLA in every block. MHLA provides reference, native, vectorized, and Triton backends. The Triton backend is only evaluated on CUDA.

2D RoPE is applied to even/odd pairs assigned to each coordinate axis. Main and context attention validate their head counts, KV head counts, and RoPE constraints during construction.

## Training objective

Let clean latent be `x0`, Gaussian noise be `x1`, and `t in [0, 1]`. The current straight path is:

```text
xt = (1 - t) * x0 + t * x1
velocity_target = x1 - x0
```

For this straight path, `rectified_flow` and `flow_matching` use the same velocity target. The main losses are:

```text
diffusion_loss = MSE(predicted_velocity, velocity_target)
x0_prediction = xt - t * predicted_velocity
reconstruction_loss = MSE(x0_prediction, x0)
weighted_reconstruction = reconstruction_loss_weight * reconstruction_loss
loss = diffusion_loss + capped(weighted_reconstruction)
```

The default `reconstruction_loss_weight` is `0.1`, and the reconstruction contribution is capped at `0.05` of total loss. The cap applies only to image-latent reconstruction.

## Sampling

Sampling starts from Gaussian noise at `t=1` and Euler-integrates image latents from `t=1` to `0`.

```text
samples ~ N(0, I)
for step in range(sample_steps):
    t = 1 - step / sample_steps
    velocity = DiT(samples, t, text_condition)
    samples = samples - velocity / sample_steps
image = frozen Qwen Image VAE decoder(samples)
```

The default is 30 sampling steps. The generation script reads architecture metadata from the checkpoint, restores the configuration, and performs a strict load.

## Checkpoint contract

- The current `NETWORK_CONFIG_VERSION` is `43`.
- `--resume` rejects checkpoints with a different network version.
- Checkpoints store DiT and Text Adapter weights plus architecture and training metadata.
- `--init-checkpoint` can partially transfer weights whose names and shapes match across different configurations. It is not a full resume.
- Legacy checkpoints with semantic channels have different state shapes and training objectives; they are not normally resume-compatible with the current image-latent-only configuration.

## Main defaults

| Setting | Default |
| --- | --- |
| VAE dtype | `bf16` |
| Trainable dtype | `bf16` |
| Image size / bucket step | `256` / `32` |
| Maximum text length | `256` |
| Null-conditioning probability | `0.05` |
| Reconstruction loss weight / cap | `0.1` / `0.05` |
| Sampling steps | `30` |
| DataLoader workers | `4` (`0` disables workers) |
| Main optimizer | `APOLLO` |
| External LR scheduler | `constant` |

For additional CLI settings and exact defaults, run `python3 image_gen/train.py --help`.
