# VFP-DiT architecture and implementation contracts

This document records the model-specific contracts for `vfp_dit/`. The package entrypoint, CLI
examples, and training workflow remain in [`vfp_dit/README.md`](../../vfp_dit/README.md). Code and
CLI help define current behavior; this document is a concise implementation guide.

## Model contracts

- Qwen source: `Qwen/Qwen3.5-0.8B`. The frozen encoder provides a selected hidden tap and an exact
  processor-derived vision-token mask. `VLMAdapter` is a per-token SwiGLU/GatedFFN stack; it has no
  self-attention or Transformer mode. The deep target DiT performs multimodal cross-token fusion
  through its cached condition K/V. For TI2I, Qwen vision features are resized to the
  half-resolution latent grid and mapped by Linear to 1024 channels; a stride-2 Conv2d maps the
  16-channel reference latent directly to 1024 channels, and the two branches are added. Text tokens
  are unchanged. At 512px, each 1024-channel BF16 projection on the 32x32 grid is about 2 MiB; the
  concat copy is gone, though both branch projections exist transiently, so peak VRAM savings need
  measurement. The legacy concat+Linear path remains available for checkpoint resume and explicit
  comparison.
- VAE source: `Qwen/Qwen-Image`, loaded as Diffusers `AutoencoderKLQwenImage`. It encodes still
  images through a singleton frame axis. Its 16-channel latent has spatial stride 8 and is
  normalized with the VAE's per-channel `(z - latents_mean) / latents_std` values. Sampling creates
  a latent grid at `resolution / 8`.
- `NoVFCBDiT.prepare_condition()` runs the bidirectional adapter and stores per-layer condition K/V
  in `ConditionKVCache`. Sampling can reuse that cache for every timestep.
- The target path defaults to spatial factor-2 patchification: a
  `Conv2d(16,1024,kernel_size=2,stride=2)` maps a 64x64 VAE latent at 512px to 1024 target tokens
  for the 24-layer DiT. This main DiT uses the visual condition K/V cache and Ada metadata
  modulation. At the exit, `ConvTranspose2d(1024,1024,kernel_size=2,stride=2)` restores the 64x64
  feature grid. A pointwise Linear maps the 16-channel noisy target latent at each location to 1024
  channels; these skip features are added to the upsampled DiT features before refinement. The skip
  projection is zero-initialized, so the initial path is exactly the upsampled feature branch. Set
  `--output-skip-fusion-mode concat_linear` to restore the older concat+Linear fusion. Two shallow
  full-resolution blocks run self-attention without visual condition K/V by default. Set
  `output_refinement_conditioning=cross_attention` to add a target-query cross-attention residual
  over adapted Qwen/reference tokens in each block; those K/V are projected and cached once per
  conditioning pass. Its output projection starts at zero, preserving the existing path at
  initialization. Both variants retain timestep conditioning and Ada scale/optional shift from
  resolution/aspect metadata. Each block also has a token/head-dependent `1+SiLU` attention gate and
  an Ada-derived zero-initialized FFN residual gate. The output head first applies RMSNorm and
  predicts a base 16-channel velocity with Linear. New Ada-enabled runs with output refinement also
  default to a width-1024 Gated Vector Field Head correction: Ada scale from the shared metadata
  embedding is applied only to the SwiGLU correction branch, and its final `W_d` projection is
  zero-initialized. This preserves the original Linear velocity readout exactly at initialization.
  Use `--no-output-head-ada-scale` to disable only this head scale, or `--metadata-conditioning
  none` to disable metadata Ada throughout the model. This new head has not yet been
  quality-screened. Additive skip fusion removes the previous `Linear(2048,1024)`, which cost about
  8.59 billion MACs at a 64x64 latent grid and materialized a 16 MiB BF16 concat tensor per sample.
  Each output block's SwiGLU FFN at width 1024 and `ff_mult=2` is about 25.77 billion MACs at that
  grid. Set `--target-latent-downsample-factor` to `1` or `4` to change the patch factor,
  `--output-refinement-depth` to change or disable the output stack, and
  `--output-refinement-conditioning cross_attention` to carry the visual/reference conditions into
  those full-resolution blocks. The launcher exposes these as `VFP_DIT_OUTPUT_REFINEMENT_DEPTH`
  and `VFP_DIT_OUTPUT_REFINEMENT_CONDITIONING`. Main and output transformer blocks both use
  activation checkpointing when enabled. The default Qwen-fused reference path also uses a 32x32
  grid at 512px: Qwen vision features are resized to match a stride-2 Conv2d projection of the VAE
  latent, and the two 1024-channel branches are added. Thus the reference image contributes 1024
  image tokens to the condition K/V instead of 4096. A matched comparison is still needed to assess
  speed and quality. Checkpoints with no `output_refinement_depth` retain their legacy direct
  prediction head. Older concat-fusion checkpoints should be resumed with their saved architecture
  or used with `--init-checkpoint` for a fresh optimizer. Compatible `qwen_to_half_latent` and
  factor-2 output-fusion weights are folded into the new additive projections during initialization.
- HF training uses aligned aspect/resolution buckets. The default nominal square-equivalent levels
  are 256, 384, and 512, with W:H buckets 1:2, 9:16, 2:3, 3:4, 1:1, 4:3, 3:2, 16:9, and 2:1. Bucket
  H/W are rounded to the VAE/DiT alignment while keeping area close to the selected level squared.
  With batch size 1, each sample independently draws a resolution level and keeps its source image's
  aspect ratio, rounded only to the VAE/DiT alignment; no bucket-index scan or batch grouping is
  needed. With batch size above 1, the loader indexes target-image dimensions once and groups equal
  resolution/aspect shapes together while retaining the configured T2I/TI2I sampling weights. Target
  and reference images use that same H/W before VAE/Qwen encoding.
- `ada_attn_ffn` injects the post-transform target canvas metadata (log square-root pixel area and
  log W:H ratio) at **both** target branches. One per-image embedding drives a per-channel scale at
  the normalized attention input before Q/K/V and at the normalized FFN input before SwiGLU. New
  training runs default to `ada_attn_ffn` with both scale and shift enabled; pass
  `--metadata-conditioning none` to disable metadata conditioning, or `--no-metadata-shift` to
  retain scale-only Ada. `--metadata-scale-mapping` selects `linear` (1+s), `one_plus_silu`
  (1+SiLU(s)), `two_sigmoid` (2 sigmoid(s)), `silu1_normalized` (SiLU(1+s)/SiLU(1)), or
  `softplus1_normalized` (softplus(1+s)/softplus(1)); the default is `softplus1_normalized`, and
  each mapping starts at effective scale 1 with a zero-initialized projection. These defaults follow
  the Mini-ImageNet screen; their effect on generative quality remains to be measured. The
  zero-initialized metadata shift adds a per-channel bias to both branch inputs. Ada-conditioned
  runs also default to a zero-initialized, metadata-derived per-channel FFN residual gate
  (`--no-metadata-ffn-residual-gate` disables it); it broadcasts one gate vector across target
  tokens and starts with the FFN residual path closed, following the AdaLN-Zero gate initialization
  principle. Attention has no separate sample-level residual gate: by default, a per-token/per-head
  `1 + SiLU` gate is computed from the post-Ada attention input (`--attention-head-gate
  input_silu`). `--attention-head-gate timestep_sigmoid` restores the older timestep-derived `2
  sigmoid` head gate for legacy checkpoint resume. When conditioning is disabled, the recorded
  mapping is `linear` and metadata-derived gates are off. The historical `attn_concat_ffn_ada`
  architecture remains loadable for its existing checkpoints, but is no longer offered for new
  training. The mode, mapping, shift, and gate settings are saved in the run/checkpoint config.
  Resume runs inherit these settings from checkpoint config when their CLI options are omitted;
  sampling likewise restores them from the checkpoint and selects legacy gate behavior when the
  newer fields are absent.
- The FFN residual gate is broadcast per metadata-conditioned image and channel over all tokens. New
  runs map its zero-initialized projection through `SiLU(s)` (`--metadata-ffn-gate-mapping silu`):
  it starts at exactly zero, has derivative 0.5 at zero, is bounded below by about -0.278, and grows
  approximately linearly for large positive inputs. Use `--metadata-ffn-gate-mapping linear` to
  select the prior raw gate. Resume configs that predate this option restore `linear` to retain
  their original behavior; new configs save the selected mapping. The benchmark launcher accepts
  `VFP_DIT_METADATA_FFN_GATE_MAPPING`.
- For new TI2I runs, Qwen vision tokens are reshaped to their processor-reported merged grid and
  resized to half resolution. A stride-2 `Conv2d(16,1024,kernel_size=2)` projects the reference
  latent, while a per-token Linear projects Qwen features to model width; the two 1024-channel
  feature maps are added. The fused 32x32 tokens replace the Qwen vision span at 512px; text tokens
  retain their sequence positions, while fused image tokens use center coordinates on the
  corresponding 2x2 latent patches and the same reference ID. `--no-fuse-reference-latent-to-vision`
  appends reference latent tokens separately. Use `--reference-latent-fusion-mode
  qwen_to_half_latent` to select the previous concat+Linear reference path; the launcher exposes
  this as `VFP_DIT_REFERENCE_LATENT_FUSION_MODE`.
  `VFP_DIT_OUTPUT_SKIP_FUSION_MODE=concat_linear` restores the previous output concat+Linear
  path. The selected fusion modes are saved in checkpoint config. Older checkpoint modes remain
  supported by the sampler and resume path; use `--init-checkpoint` rather than `--resume` when
  intentionally switching architectures.
- Main DiT layers issue queries only for target tokens. Each query reads the cached condition
  keys/values plus all target keys/values. Thus target tokens are bidirectional and target
  information never updates the condition prefix. Attention is computed with PyTorch SDPA and GQA.
- The adapter SwiGLU input/value projections share one `ffn_in` GEMM. Same-input projection fusion
  is enabled by default in the target DiT: it packs target Q/K/V and the per-token/head attention
  gate, condition K/V before splitting and caching them, and same-input metadata
  scale/shift/FFN-gate projections. The target split shapes are `(B,T,width)`,
  `(B,T,kv_heads*head_dim)` twice, and `(B,T,heads)`; cached condition K/V remain
  `(B,kv_heads,S,head_dim)`. Use `--no-fuse-same-input-projections` to select the reference path.
  The mode is saved in checkpoint config; when the flag is omitted, resume and initialization runs
  inherit the checkpoint's setting, while a fresh run enables fusion.
- `tests/unit/test_vfp_dit_model_primitives.py` compares fused and separate projections after copying
  equivalent weights, including forward values, condition-cache K/V, input gradients, and
  packed-projection parameter gradients for Ada scale/shift/gate and legacy attention-gate variants.
  CPU FP32 checks pass at tight tolerances; CUDA/BF16 equivalence and performance remain unverified
  in the current environment.
- RoPE coordinates are `(sequence_or_reference, y, x)`. Text uses the first axis as token position.
  Qwen image tokens use reference id 1 and row-major merged-grid coordinates reconstructed from
  `mm_token_type_ids`, `image_grid_thw`, and the configured spatial merge size. Fused reference
  tokens use reference id 1 and center coordinates of their corresponding 2x2 latent patches. Target
  patch tokens use reference id 0 at the center of their original latent-grid patch.
- Flow matching uses `x_t = (1-t)x_0 + t*epsilon`, with target velocity `epsilon - x_0`. By default,
  each image draws four independent `t` values, one from each equal-width interval `[0,.25)`,
  `[.25,.5)`, `[.5,.75)`, and `[.75,1)`, with independent noise per value. The condition adapter and
  K/V cache run once per image; the target DiT receives four repeated target rows, and the loss
  averages over timesteps and images. Set `--timesteps-per-image 1` for the former one-timestep
  behavior. Dataset images and encoder features are processed on the fly; no tensor records are
  produced.
- `--profile-components` synchronizes CUDA around opt-in measurements and records per-step seconds
  for target/reference VAE encode, Qwen encode, condition preparation, stratified noise
  construction, DiT forward, loss, backward, and optimizer step in run metrics. These synchronized
  measurements add overhead and should be used to identify bottlenecks, not to compare throughput.
  Pass the same option to `generate_samples.py` to print per-sample Qwen encode, reference VAE
  encode, condition preparation, DiT sampling, and VAE decode times.
- T2I examples carry no reference latent. TI2I examples carry the source-image Qwen hidden sequence,
  its exact vision-token mask, and its VAE latent. Condition dropout removes Qwen and reference
  conditions together.

## Checkpoint compatibility

The VLM adapter has one architecture: per-token SwiGLU/GatedFFN blocks. The bidirectional adapter
Transformer and linear-only variants were removed; the target DiT remains responsible for cross-token
condition fusion. New runs record
`adapter_type=ffn` in their checkpoint config. Checkpoints from the former FFN mode can be sampled
after their unused frozen attention tensors are dropped under a strict load. Transformer or
linear-only checkpoints are rejected by the sampler because their condition computation differs. Old
runs with unused adapter-attention tensors cannot resume optimizer state; use `--init-checkpoint` to
start a fresh run from compatible weights.

The historical adapter ablation is summarized in the [experiment
archive](../experiment_data/vfp-simple-adapter-ablation.md); it applies only to the recorded
one-epoch comparison.

## Historical runtime measurements

Short resource and pipeline measurements are kept in [VFP-DiT probe
records](../experiment_data/vfp-dit-implementation-probes-2026-09-24.md), separate from the
architecture contract.
