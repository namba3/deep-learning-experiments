#!/usr/bin/env bash
set -euo pipefail

# One VFP-DiT training run. Set VFP_DIT_FULL_DATA_EPOCH=1 to visit every
# row in the mixed HF training set exactly once (normally with --epochs 1).
full_data_epoch="${VFP_DIT_FULL_DATA_EPOCH:-0}"
if [[ "$full_data_epoch" != "0" && "$full_data_epoch" != "1" ]]; then
  echo "VFP_DIT_FULL_DATA_EPOCH must be 0 or 1" >&2
  exit 2
fi
if [[ "$full_data_epoch" == "1" ]]; then
  epochs="${VFP_DIT_EPOCHS:-1}"
  default_validation_fraction="0.05"
else
  epochs="${VFP_DIT_EPOCHS:-10}"
  default_validation_fraction="0.05"
fi
samples_per_epoch="${VFP_DIT_SAMPLES_PER_EPOCH:-1024}"
validation_fraction="${VFP_DIT_VALIDATION_FRACTION:-$default_validation_fraction}"
validation_samples="${VFP_DIT_VALIDATION_SAMPLES:-256}"
resolution="${VFP_DIT_RESOLUTION:-512}"
resolution_levels="${VFP_DIT_RESOLUTION_LEVELS:-}"
batch_size="${VFP_DIT_BATCH_SIZE:-1}"
num_workers="${VFP_DIT_NUM_WORKERS:-4}"
condition_layer="${VFP_DIT_CONDITION_LAYER:-final}"
condition_dropout="${VFP_DIT_CONDITION_DROPOUT:-0.1}"
optimizer="${VFP_DIT_OPTIMIZER:-APOLLO}"
learning_rate="${VFP_DIT_LR:-1e-3}"
lr_scheduler="${VFP_DIT_LR_SCHEDULER:-cosine}"
apollo_rank="${VFP_DIT_APOLLO_RANK:-32}"
seed="${VFP_DIT_SEED:-42}"
model_width="${VFP_DIT_MODEL_WIDTH:-1024}"
depth="${VFP_DIT_DEPTH:-24}"
heads="${VFP_DIT_HEADS:-16}"
kv_heads="${VFP_DIT_KV_HEADS:-4}"
adapter_depth="${VFP_DIT_ADAPTER_DEPTH:-2}"
condition_dim="${VFP_DIT_CONDITION_DIM:-1024}"
ff_mult="${VFP_DIT_FF_MULT:-3.0}"
metadata_conditioning="${VFP_DIT_METADATA_CONDITIONING:-ada_attn_ffn}"
metadata_scale_mapping="${VFP_DIT_METADATA_SCALE_MAPPING:-}"
metadata_ffn_gate_mapping="${VFP_DIT_METADATA_FFN_GATE_MAPPING:-}"
metadata_shift="${VFP_DIT_METADATA_SHIFT:-1}"
init_checkpoint="${VFP_DIT_INIT_CHECKPOINT:-}"
reference_latent_downsample_factor="${VFP_DIT_LATENT_DOWNSAMPLE_FACTOR:-1}"
timesteps_per_image="${VFP_DIT_TIMESTEPS_PER_IMAGE:-4}"
profile_components="${VFP_DIT_PROFILE_COMPONENTS:-0}"
log_metadata_diagnostics="${VFP_DIT_LOG_METADATA_DIAGNOSTICS:-0}"
check_finite_updates="${VFP_DIT_CHECK_FINITE_UPDATES:-0}"
anomaly_detection_batch="${VFP_DIT_ANOMALY_DETECTION_BATCH:-0}"
observe_interval="${VFP_DIT_OBSERVE_INTERVAL:-1000}"
sample_steps="${VFP_DIT_SAMPLE_STEPS:-30}"
sample_guidance_scale="${VFP_DIT_SAMPLE_GUIDANCE_SCALE:-4}"
sample_solver="${VFP_DIT_SAMPLE_SOLVER:-${VFP_DIT_SAMPLE_SAMPLER:-euler}}"
sample_scheduler="${VFP_DIT_SAMPLE_SCHEDULER:-flow_match_euler}"
flow_shift="${VFP_DIT_FLOW_SHIFT:-1.0}"
er_sde_sigma_max="${VFP_DIT_ER_SDE_SIGMA_MAX:-80}"
fuse_reference_latent="${VFP_DIT_FUSE_REFERENCE_LATENT:-1}"
fuse_same_input_projections="${VFP_DIT_FUSE_SAME_INPUT_PROJECTIONS:-1}"
device="${VFP_DIT_DEVICE:-cuda}"
amp="${VFP_DIT_AMP:-bf16}"
encoder_device="${VFP_DIT_ENCODER_DEVICE:-training}"
vae_device="${VFP_DIT_VAE_DEVICE:-}"
encoder_prefetch_batches="${VFP_DIT_ENCODER_PREFETCH_BATCHES:-2}"
gc_interval="${VFP_DIT_GC_INTERVAL:-10}"
empty_cache_interval="${VFP_DIT_EMPTY_CACHE_INTERVAL:-10}"
multi_edit_root="${VFP_DIT_MULTI_EDIT_ROOT:-data/MultiEdit}"
output_dir="${VFP_DIT_OUTPUT_DIR:-vfp_dit/output}"
output_refinement_depth="${VFP_DIT_OUTPUT_REFINEMENT_DEPTH:-}"
output_refinement_conditioning="${VFP_DIT_OUTPUT_REFINEMENT_CONDITIONING:-}"
output_skip_fusion_mode="${VFP_DIT_OUTPUT_SKIP_FUSION_MODE:-}"
output_head_ada_scale="${VFP_DIT_OUTPUT_HEAD_ADA_SCALE:-}"
reference_latent_fusion_mode="${VFP_DIT_REFERENCE_LATENT_FUSION_MODE:-}"
if [[ "$full_data_epoch" == "1" && -z "$init_checkpoint" ]]; then
  init_checkpoint="vfp_dit/output/initial_checkpoints/fullres-reference-init-128px-16.safetensors"
fi
if [[ "$full_data_epoch" == "1" ]]; then
  default_run_name="full-data-1e-cosine-$(date -u +%Y%m%dT%H%M%SZ)"
else
  default_run_name="screen-$(date -u +%Y%m%dT%H%M%SZ)"
fi
run_name="${VFP_DIT_RUN_NAME:-$default_run_name}"
resume_checkpoint="${VFP_DIT_RESUME:-}"

args=(
  --data-mode hf
  --multi-edit-data-root "$multi_edit_root"
  --condition-layer "$condition_layer"
  --condition-dim "$condition_dim"
  --metadata-conditioning "$metadata_conditioning"
  --resolution "$resolution"
  --device "$device"
  --amp "$amp"
  --encoder-device "$encoder_device"
  --encoder-prefetch-batches "$encoder_prefetch_batches"
  --optimizer "$optimizer"
  --lr "$learning_rate"
  --lr-scheduler "$lr_scheduler"
  --apollo-rank "$apollo_rank"
  --batch-size "$batch_size"
  --num-workers "$num_workers"
  --gc-interval "$gc_interval"
  --empty-cache-interval "$empty_cache_interval"
  --condition-dropout "$condition_dropout"
  --reference-latent-downsample-factor "$reference_latent_downsample_factor"
  --timesteps-per-image "$timesteps_per_image"
  --observe-interval "$observe_interval"
  --sample-steps "$sample_steps"
  --sample-guidance-scale "$sample_guidance_scale"
  --sample-solver "$sample_solver"
  --sample-scheduler "$sample_scheduler"
  --sample-flow-shift "$flow_shift"
  --sample-er-sde-sigma-max "$er_sde_sigma_max"
  --model-width "$model_width"
  --depth "$depth"
  --heads "$heads"
  --kv-heads "$kv_heads"
  --adapter-depth "$adapter_depth"
  --ff-mult "$ff_mult"
  --epochs "$epochs"
  --validation-fraction "$validation_fraction"
  --validation-samples "$validation_samples"
  --seed "$seed"
  --output-dir "$output_dir"
  --run-name "$run_name"
  --sample-prompt "A small red wooden boat on a quiet lake at sunrise, realistic photography."
  --sample-prompt "A crowded night market on a rainy city street, bright signs reflected on wet pavement, documentary street photography."
  --sample-prompt "Extreme close-up macro photograph of a metallic blue beetle walking across a vivid green leaf, soft natural light."
  --sample-prompt "A tiny orange fox reading a book beneath oversized mushrooms in a misty forest, whimsical watercolor illustration."
)
if [[ -n "$output_refinement_depth" ]]; then
  args+=(--output-refinement-depth "$output_refinement_depth")
fi
if [[ -n "$output_refinement_conditioning" ]]; then
  args+=(--output-refinement-conditioning "$output_refinement_conditioning")
fi
if [[ -n "$output_skip_fusion_mode" ]]; then
  args+=(--output-skip-fusion-mode "$output_skip_fusion_mode")
fi
if [[ "$output_head_ada_scale" == "1" || "$output_head_ada_scale" == "true" ]]; then
  args+=(--output-head-ada-scale)
elif [[ "$output_head_ada_scale" == "0" || "$output_head_ada_scale" == "false" ]]; then
  args+=(--no-output-head-ada-scale)
elif [[ -n "$output_head_ada_scale" ]]; then
  echo "VFP_DIT_OUTPUT_HEAD_ADA_SCALE must be 1/true or 0/false" >&2
  exit 2
fi
if [[ -n "$reference_latent_fusion_mode" ]]; then
  args+=(--reference-latent-fusion-mode "$reference_latent_fusion_mode")
fi
if [[ -n "$vae_device" ]]; then
  if [[ "$vae_device" != "training" && "$vae_device" != "cpu" ]]; then
    echo "VFP_DIT_VAE_DEVICE must be training or cpu" >&2
    exit 2
  fi
  args+=(--vae-device "$vae_device")
fi
if [[ -n "$resolution_levels" ]]; then
  args+=(--resolution-levels "$resolution_levels")
fi
if [[ "$full_data_epoch" == "1" ]]; then
  if [[ "$epochs" != "1" ]]; then
    echo "VFP_DIT_FULL_DATA_EPOCH=1 requires VFP_DIT_EPOCHS=1" >&2
    exit 2
  fi
  if [[ -n "${VFP_DIT_SAMPLES_PER_EPOCH:-}" ]]; then
    echo "VFP_DIT_SAMPLES_PER_EPOCH cannot be set with VFP_DIT_FULL_DATA_EPOCH=1" >&2
    exit 2
  fi
  args+=(--full-data-epoch)
else
  args+=(--samples-per-epoch "$samples_per_epoch")
fi
if [[ -n "$metadata_scale_mapping" ]]; then
  args+=(--metadata-scale-mapping "$metadata_scale_mapping")
fi
if [[ -n "$metadata_ffn_gate_mapping" ]]; then
  args+=(--metadata-ffn-gate-mapping "$metadata_ffn_gate_mapping")
fi
if [[ "$fuse_same_input_projections" == "1" ]]; then
  args+=(--fuse-same-input-projections)
elif [[ "$fuse_same_input_projections" == "0" ]]; then
  args+=(--no-fuse-same-input-projections)
else
  echo "VFP_DIT_FUSE_SAME_INPUT_PROJECTIONS must be 0 or 1" >&2
  exit 2
fi
if [[ "$metadata_shift" == "1" ]]; then
  args+=(--metadata-shift)
elif [[ "$metadata_shift" == "0" ]]; then
  args+=(--no-metadata-shift)
else
  echo "VFP_DIT_METADATA_SHIFT must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$init_checkpoint" ]]; then
  if [[ ! -f "$init_checkpoint" ]]; then
    echo "VFP_DIT_INIT_CHECKPOINT does not exist: $init_checkpoint" >&2
    exit 2
  fi
  args+=(--init-checkpoint "$init_checkpoint")
fi
if [[ -n "$resume_checkpoint" && -n "$init_checkpoint" ]]; then
  echo "VFP_DIT_RESUME and VFP_DIT_INIT_CHECKPOINT cannot be combined" >&2
  exit 2
fi
if [[ -n "$resume_checkpoint" ]]; then
  if [[ ! -f "$resume_checkpoint" ]]; then
    echo "VFP_DIT_RESUME checkpoint does not exist: $resume_checkpoint" >&2
    exit 2
  fi
  args+=(--resume "$resume_checkpoint")
fi
if [[ "$profile_components" == "1" ]]; then
  args+=(--profile-components)
elif [[ "$profile_components" != "0" ]]; then
  echo "VFP_DIT_PROFILE_COMPONENTS must be 0 or 1" >&2
  exit 2
fi
if [[ "$log_metadata_diagnostics" == "1" ]]; then
  args+=(--log-metadata-diagnostics)
elif [[ "$log_metadata_diagnostics" != "0" ]]; then
  echo "VFP_DIT_LOG_METADATA_DIAGNOSTICS must be 0 or 1" >&2
  exit 2
fi
if [[ "$check_finite_updates" == "1" ]]; then
  args+=(--check-finite-updates)
elif [[ "$check_finite_updates" != "0" ]]; then
  echo "VFP_DIT_CHECK_FINITE_UPDATES must be 0 or 1" >&2
  exit 2
fi
if [[ "$anomaly_detection_batch" =~ ^[0-9]+$ ]]; then
  args+=(--anomaly-detection-batch "$anomaly_detection_batch")
else
  echo "VFP_DIT_ANOMALY_DETECTION_BATCH must be 0 or a positive integer" >&2
  exit 2
fi

no_observe_samples="${VFP_DIT_NO_OBSERVE_SAMPLES:-0}"
if [[ "$no_observe_samples" == "1" ]]; then
  args+=(--no-observe-samples)
elif [[ "$no_observe_samples" != "0" ]]; then
  echo "VFP_DIT_NO_OBSERVE_SAMPLES must be 0 or 1" >&2
  exit 2
fi

if [[ "$fuse_reference_latent" == "1" ]]; then
  args+=(--fuse-reference-latent-to-vision)
elif [[ "$fuse_reference_latent" == "0" ]]; then
  args+=(--no-fuse-reference-latent-to-vision)
else
  echo "VFP_DIT_FUSE_REFERENCE_LATENT must be 0 or 1" >&2
  exit 2
fi

if [[ "${VFP_DIT_GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  args+=(--gradient-checkpointing)
elif [[ "${VFP_DIT_GRADIENT_CHECKPOINTING:-1}" != "0" ]]; then
  echo "VFP_DIT_GRADIENT_CHECKPOINTING must be 0 or 1" >&2
  exit 2
else
  args+=(--no-gradient-checkpointing)
fi

sample_setting="${VFP_DIT_GENERATE_SAMPLES:-1}"
if [[ "$sample_setting" != "0" && "$sample_setting" != "1" ]]; then
  echo "VFP_DIT_GENERATE_SAMPLES must be 0 or 1" >&2
  exit 2
fi
sample_steps="${VFP_DIT_SAMPLE_STEPS:-30}"
sample_solver="${VFP_DIT_SAMPLE_SOLVER:-${VFP_DIT_SAMPLE_SAMPLER:-euler}}"
sample_scheduler="${VFP_DIT_SAMPLE_SCHEDULER:-flow_match_euler}"
flow_shift="${VFP_DIT_FLOW_SHIFT:-1.0}"
er_sde_sigma_max="${VFP_DIT_ER_SDE_SIGMA_MAX:-80}"
if [[ ! "$sample_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "VFP_DIT_SAMPLE_STEPS must be a positive integer" >&2
  exit 2
fi
reference_image="${VFP_DIT_SCREEN_REFERENCE_IMAGE:-}"
if [[ "$sample_setting" == "1" && "${DRY_RUN:-0}" != "1" && -n "$reference_image" && ! -f "$reference_image" ]]; then
  echo "TI2I reference image does not exist: $reference_image" >&2
  exit 2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run --device cpu)
fi

mkdir -p "$output_dir"
echo "Run: ${run_name}"
if [[ "$full_data_epoch" == "1" ]]; then
  echo "Budget: ${epochs} full-data epoch(s), every training row once; optimizer=${optimizer}, lr=${learning_rate}, schedule=${lr_scheduler}"
else
  echo "Budget: ${epochs} epochs x ${samples_per_epoch} samples; optimizer=${optimizer}, lr=${learning_rate}, schedule=${lr_scheduler}"
fi
echo "Validation: fraction=${validation_fraction}, samples/epoch=${validation_samples}"
echo "Output: ${output_dir}"
if [[ -n "$resume_checkpoint" ]]; then
  echo "Resume: ${resume_checkpoint}"
fi
log_path="${output_dir}/${run_name}.log"
PYTHONPATH=".:${PYTHONPATH:-}" python3 -m vfp_dit.train "${args[@]}" \
  2>&1 | tee "$log_path"

if [[ "$sample_setting" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  checkpoint_line="$(rg '^Saved checkpoint: ' "$log_path" | tail -n 1 || true)"
  checkpoint_path="${checkpoint_line#Saved checkpoint: }"
  if [[ -z "$checkpoint_path" || ! -f "$checkpoint_path" ]]; then
    echo "Could not find the completed run checkpoint in $log_path" >&2
    exit 2
  fi

  run_dir="$(dirname -- "$(dirname -- "$checkpoint_path")")"
  sample_args=(
    --checkpoint "$checkpoint_path"
    --output "$run_dir/artifacts/screen_samples.png"
    --steps "$sample_steps"
    --guidance-scale "${VFP_DIT_SAMPLE_GUIDANCE:-4}"
    --solver "$sample_solver"
    --scheduler "$sample_scheduler"
    --flow-shift "$flow_shift"
    --er-sde-sigma-max "$er_sde_sigma_max"
    --seed "$seed"
    --device "$device"
    --prompt "A small red wooden boat on a quiet lake at sunrise, realistic photography."
    --prompt "A crowded night market on a rainy city street, bright signs reflected on wet pavement, documentary street photography."
    --prompt "Extreme close-up macro photograph of a metallic blue beetle walking across a vivid green leaf, soft natural light."
    --prompt "A tiny orange fox reading a book beneath oversized mushrooms in a misty forest, whimsical watercolor illustration."
  )
  if [[ "$profile_components" == "1" ]]; then
    sample_args+=(--profile-components)
  fi
  if [[ -n "$reference_image" ]]; then
    sample_args+=(
      --prompt "${VFP_DIT_SCREEN_EDIT_PROMPT:-Turn the scene into a warm sunset.}"
      --reference-image "" --reference-image "" --reference-image ""
      --reference-image "" --reference-image "$reference_image"
    )
  fi
  sample_log="$run_dir/artifacts/screen_samples.log"
  echo "Generating screen samples: $run_dir/artifacts/screen_samples.png"
  echo "steps=$sample_steps solver=$sample_solver scheduler=$sample_scheduler flow_shift=$flow_shift er_sde_sigma_max=$er_sde_sigma_max guidance=${VFP_DIT_SAMPLE_GUIDANCE:-4} seed=$seed device=$device reference=${reference_image:-none}" \
    | tee "$sample_log"
  PYTHONPATH=".:${PYTHONPATH:-}" python3 -m vfp_dit.generate_samples "${sample_args[@]}" \
    2>&1 | tee -a "$sample_log"
fi
