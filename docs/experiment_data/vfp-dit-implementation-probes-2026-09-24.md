# VFP-DiT implementation probes (2026-09-24 snapshot)

This record preserves short measurements and runtime checks for the `vfp_dit`
implementation as recorded on 2026-09-24. The probes are narrow pipeline and
resource checks; they do not establish quality, convergence, or general
performance. Raw run logs, per-step metrics, generated images, and checkpoints
are not retained.

## Short 384px screen

A 32-update screen with eight validation examples measured peak allocated /
reserved VRAM of 4.26 / 4.38 GiB and validation MSE 1.188. There was no matched
baseline, so the result is a pipeline/runtime check only.

## Matched 512px profile

An eight-image profile compared one and four stratified timesteps at 512px,
width 1024 / depth 24, batch 1, APOLLO rank 32, factor 4, adapter depth 2, and
BF16. Both completed; peak allocated / reserved VRAM was 4.69 / 4.89 GiB for
one timestep and 4.90 / 5.10 GiB for four. The reported 0.55 versus 0.64
images/s includes CUDA synchronization overhead and is not a speed ranking.
The metrics indicate Qwen/VAE encoding and condition preparation happened once
per image, while target latent/noise and DiT evaluation expanded with the
number of timesteps.

A single cold T2I sample at 512px and 30 steps measured Qwen encode 0.863 s,
condition preparation 0.066 s, DiT sampling 2.168 s, and VAE decode 0.244 s.
This is a timing reference for that sample only.

The current model contracts and commands are in the [VFP-DiT architecture
guide](../architecture/vfp-dit.md) and [package README](../../vfp_dit/README.md).
