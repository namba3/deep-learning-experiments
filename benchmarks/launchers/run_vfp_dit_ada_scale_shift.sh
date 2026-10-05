#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

seed="${VFP_DIT_SEED:-42}"
output_dir="${VFP_DIT_OUTPUT_DIR:-vfp_dit/output/ada-scale-shift-seed${seed}}"
condition_dim="${VFP_DIT_CONDITION_DIM:-1024}"
model_width="${VFP_DIT_MODEL_WIDTH:-1024}"
depth="${VFP_DIT_DEPTH:-24}"
heads="${VFP_DIT_HEADS:-16}"
kv_heads="${VFP_DIT_KV_HEADS:-4}"
adapter_depth="${VFP_DIT_ADAPTER_DEPTH:-2}"
ff_mult="${VFP_DIT_FF_MULT:-3.0}"
reference_latent_downsample_factor="${VFP_DIT_LATENT_DOWNSAMPLE_FACTOR:-1}"
fuse_reference_latent="${VFP_DIT_FUSE_REFERENCE_LATENT:-1}"
fuse_same_input_projections="${VFP_DIT_FUSE_SAME_INPUT_PROJECTIONS:-1}"
mappings="${VFP_DIT_ADA_SCALE_MAPPINGS:-linear,one_plus_silu,two_sigmoid,silu1_normalized,softplus1_normalized}"
shift_values="${VFP_DIT_ADA_SHIFT_VALUES:-0,1}"
if [[ "$fuse_same_input_projections" == "1" ]]; then
  projection_suffix="_fused-projections"
elif [[ "$fuse_same_input_projections" == "0" ]]; then
  projection_suffix=""
else
  echo "VFP_DIT_FUSE_SAME_INPUT_PROJECTIONS must be 0 or 1" >&2
  exit 2
fi
init_checkpoint="$output_dir/initialization/no_metadata_seed${seed}${projection_suffix}.safetensors"

mkdir -p "$(dirname "$init_checkpoint")"
init_args=(
  --output "$init_checkpoint"
  --seed "$seed"
  --condition-dim "$condition_dim"
  --model-width "$model_width"
  --depth "$depth"
  --heads "$heads"
  --kv-heads "$kv_heads"
  --adapter-depth "$adapter_depth"
  --ff-mult "$ff_mult"
  --reference-latent-downsample-factor "$reference_latent_downsample_factor"
)
if [[ "$fuse_same_input_projections" == "1" ]]; then
  init_args+=(--fuse-same-input-projections)
else
  init_args+=(--no-fuse-same-input-projections)
fi
if [[ "$fuse_reference_latent" == "0" ]]; then
  init_args+=(--no-fuse-reference-latent-to-vision)
elif [[ "$fuse_reference_latent" != "1" ]]; then
  echo "VFP_DIT_FUSE_REFERENCE_LATENT must be 0 or 1" >&2
  exit 2
fi

export VFP_DIT_OUTPUT_DIR="$output_dir"
export VFP_DIT_SEED="$seed"
export VFP_DIT_GENERATE_SAMPLES=0
export VFP_DIT_OBSERVE_INTERVAL=0
export VFP_DIT_NO_OBSERVE_SAMPLES=1
export VFP_DIT_RESUME=
export VFP_DIT_LOG_METADATA_DIAGNOSTICS="${VFP_DIT_LOG_METADATA_DIAGNOSTICS:-1}"
expected_run_names=()

if [[ ! -f "$init_checkpoint" ]]; then
  echo "Creating shared no-metadata initialization: $init_checkpoint"
  PYTHONPATH=".:${PYTHONPATH:-}" python3 -m vfp_dit.save_init_checkpoint "${init_args[@]}"
else
  echo "Reusing shared initialization: $init_checkpoint"
fi

run_arm() {
  local run_name="$1"
  local conditioning="$2"
  local mapping="$3"
  local shift="$4"
  local arm_state
  local resume_checkpoint=""
  expected_run_names+=("$run_name")
  arm_state="$(PYTHONPATH=".:${PYTHONPATH:-}" python3 -m vfp_dit.summarize_metadata_ada \
    --output-dir "$output_dir" --run-name "$run_name" --arm-state)"
  if [[ "$arm_state" == "completed" ]]; then
    echo "Skipping completed $run_name"
    return
  elif [[ "$arm_state" == resume:* ]]; then
    resume_checkpoint="${arm_state#resume:}"
    echo "Resuming $run_name from $resume_checkpoint"
  else
    echo "Starting $run_name"
  fi

  export VFP_DIT_RUN_NAME="$run_name"
  export VFP_DIT_METADATA_CONDITIONING="$conditioning"
  export VFP_DIT_METADATA_SCALE_MAPPING="$mapping"
  export VFP_DIT_METADATA_SHIFT="$shift"
  if [[ -n "$resume_checkpoint" ]]; then
    export VFP_DIT_RESUME="$resume_checkpoint"
    export VFP_DIT_INIT_CHECKPOINT=
  else
    export VFP_DIT_RESUME=
    export VFP_DIT_INIT_CHECKPOINT="$init_checkpoint"
  fi
  bash benchmarks/launchers/run_vfp_dit_screen.sh
}

# The no-meta arm is the baseline. Every arm starts from the same seed-built
# no-meta state; newly introduced Ada projections are zero-initialized.
run_arm "ada-screen-seed${seed}-no-metadata" "none" "linear" "0"

IFS=',' read -r -a mapping_list <<< "$mappings"
IFS=',' read -r -a shift_list <<< "$shift_values"
for mapping in "${mapping_list[@]}"; do
  for shift in "${shift_list[@]}"; do
    if [[ "$shift" != "0" && "$shift" != "1" ]]; then
      echo "VFP_DIT_ADA_SHIFT_VALUES entries must be 0 or 1" >&2
      exit 2
    fi
    run_arm "ada-screen-seed${seed}-${mapping}-shift${shift}" "ada_attn_ffn" "$mapping" "$shift"
  done
done

summary_args=(--output-dir "$output_dir")
for run_name in "${expected_run_names[@]}"; do
  summary_args+=(--expected-run-name "$run_name")
done
PYTHONPATH=".:${PYTHONPATH:-}" python3 -m vfp_dit.summarize_metadata_ada "${summary_args[@]}"
