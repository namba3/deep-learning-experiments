#!/usr/bin/env bash
set -euo pipefail

# Matched fixed-LR screen initialized from the same completed 2k-step checkpoint.
# Constant LR isolates optimizer step size from the cosine-horizon effect.
run_group="${VFP_DIT_LR_SCREEN_RUN_GROUP:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${VFP_DIT_LR_SCREEN_OUTPUT_DIR:-vfp_dit/output/lr-screen}"
init_checkpoint="${VFP_DIT_LR_SCREEN_INIT_CHECKPOINT:-}"
learning_rates="${VFP_DIT_LR_SCREEN_VALUES:-1e-4,3e-4,1e-3}"
samples_per_epoch="${VFP_DIT_LR_SCREEN_SAMPLES_PER_EPOCH:-512}"
validation_samples="${VFP_DIT_LR_SCREEN_VALIDATION_SAMPLES:-256}"
seed="${VFP_DIT_LR_SCREEN_SEED:-42}"
num_workers="${VFP_DIT_LR_SCREEN_NUM_WORKERS:-4}"
multi_edit_root="${VFP_DIT_LR_SCREEN_MULTI_EDIT_ROOT:-data/MultiEdit}"
device="${VFP_DIT_LR_SCREEN_DEVICE:-cuda}"
amp="${VFP_DIT_LR_SCREEN_AMP:-bf16}"
make_samples="${VFP_DIT_LR_SCREEN_GENERATE_SAMPLES:-1}"

if [[ -z "$init_checkpoint" ]]; then
  echo "Set VFP_DIT_LR_SCREEN_INIT_CHECKPOINT to a completed starting checkpoint" >&2
  exit 2
fi
if [[ ! -f "$init_checkpoint" ]]; then
  echo "Starting checkpoint does not exist: $init_checkpoint" >&2
  exit 2
fi
if [[ ! "$samples_per_epoch" =~ ^[1-9][0-9]*$ || ! "$validation_samples" =~ ^[1-9][0-9]*$ ]]; then
  echo "Training and validation sample counts must be positive integers" >&2
  exit 2
fi
if [[ ! "$make_samples" =~ ^[01]$ ]]; then
  echo "VFP_DIT_LR_SCREEN_GENERATE_SAMPLES must be 0 or 1" >&2
  exit 2
fi

IFS=',' read -r -a rates <<< "$learning_rates"
mkdir -p "$output_dir"

for lr in "${rates[@]}"; do
  lr="${lr//[[:space:]]/}"
  [[ -z "$lr" ]] && continue
  run_name="lr-${lr}-${run_group}"
  log_path="$output_dir/${run_name}.log"
  args=(
    --data-mode hf
    --multi-edit-data-root "$multi_edit_root"
    --condition-layer final
    --resolution 512
    --device "$device"
    --amp "$amp"
    --optimizer APOLLO
    --apollo-rank 32
    --lr "$lr"
    --lr-scheduler constant
    --batch-size 1
    --num-workers "$num_workers"
    --condition-dropout 0.1
    --reference-latent-downsample-factor 1
    --timesteps-per-image 4
    --observe-interval 0
    --no-observe-samples
    --model-width 1024
    --depth 24
    --heads 16
    --kv-heads 4
    --adapter-depth 2
    --ff-mult 3.0
    --epochs 1
    --samples-per-epoch "$samples_per_epoch"
    --validation-fraction 0.05
    --validation-samples "$validation_samples"
    --seed "$seed"
    --output-dir "$output_dir"
    --run-name "$run_name"
    --init-checkpoint "$init_checkpoint"
    --fuse-reference-latent-to-vision
    --gradient-checkpointing
  )

  echo "=== VFP-DiT Simple LR=${lr}; samples=${samples_per_epoch}; seed=${seed}; group=${run_group} ==="
  PYTHONPATH=".:${PYTHONPATH:-}" python3 -m vfp_dit.train "${args[@]}" \
    2>&1 | tee "$log_path"

  if [[ "$make_samples" == "1" ]]; then
    checkpoint_line="$(rg '^Saved checkpoint: ' "$log_path" | tail -n 1 || true)"
    checkpoint_path="${checkpoint_line#Saved checkpoint: }"
    if [[ -z "$checkpoint_path" || ! -f "$checkpoint_path" ]]; then
      echo "Could not find the completed checkpoint for LR=${lr} in ${log_path}" >&2
      exit 2
    fi
    run_dir="$(dirname -- "$(dirname -- "$checkpoint_path")")"
    PYTHONPATH=".:${PYTHONPATH:-}" python3 -m vfp_dit.generate_samples \
      --checkpoint "$checkpoint_path" \
      --output "$run_dir/artifacts/lr_screen_sample.png" \
      --steps 30 --guidance-scale 4 --seed "$seed" --device "$device" \
      --prompt "A small red wooden boat on a quiet lake at sunrise, realistic photography."
  fi
done

echo "LR screen complete: $output_dir (group=$run_group)"
