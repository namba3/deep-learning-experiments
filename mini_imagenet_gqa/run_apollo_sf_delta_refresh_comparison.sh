#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_dir="${OUTPUT_DIR:-mini_imagenet_gqa/output/bucketed/apollo-sf-delta-refresh}"
variant="${VARIANT:-gated_gqa_silu_gated_ffn}"
epochs="${EPOCHS:-3}"
batch_size="${BATCH_SIZE:-64}"
num_workers="${NUM_WORKERS:-4}"
steps_per_epoch="${STEPS_PER_EPOCH:-0}"
eval_batches="${EVAL_BATCHES:-0}"
lr="${LR:-1e-3}"
weight_decay="${WEIGHT_DECAY:-0.05}"
amp="${AMP:-bf16}"
apollo_rank="${APOLLO_RANK:-32}"
quant_block_size="${APOLLO_SF_QUANT_BLOCK_SIZE:-256}"
update_proj_gap="${APOLLO_UPDATE_PROJ_GAP:-200}"
refresh_window="${APOLLO_SF_DELTA_REFRESH_WINDOW:-4}"
seeds_csv="${SEEDS:-42}"
policies_csv="${DELTA_REFRESH_POLICIES:-none,blend}"

IFS=',' read -r -a seeds <<< "$seeds_csv"
IFS=',' read -r -a policies <<< "$policies_csv"

for policy in "${policies[@]}"; do
  label="APOLLO-SF-INT8-Delta-${policy}"
  optimizer_dir="$output_dir/$label"
  for seed in "${seeds[@]}"; do
    run_name="mini-imagenet-${label}-seed${seed}-${variant}"
    arm_state="$(python3 -m mini_imagenet_gqa.summarize_comparison \
      --output-dir "$optimizer_dir" --run-name "$run_name" --arm-state)"
    if [[ "$arm_state" == "completed" ]]; then
      echo "Skipping completed ${label} seed ${seed}"
      continue
    fi

    args=(
      --variant "$variant"
      --seed "$seed"
      --epochs "$epochs"
      --batch-size "$batch_size"
      --num-workers "$num_workers"
      --lr "$lr"
      --optimizer APOLLO-SF-INT8-Delta
      --apollo-rank "$apollo_rank"
      --apollo-sf-quant-block-size "$quant_block_size"
      --apollo-sf-delta-refresh "$policy"
      --apollo-sf-delta-refresh-window "$refresh_window"
      --apollo-update-proj-gap "$update_proj_gap"
      --apollo-fallback came
      --apollo-matrix-fallback auto
      --weight-decay "$weight_decay"
      --amp "$amp"
      --common-init
      --deterministic
      --device cuda
      --steps-per-epoch "$steps_per_epoch"
      --eval-batches "$eval_batches"
      --output-dir "$optimizer_dir"
      --run-name "$run_name"
    )
    if [[ "$arm_state" == resume:* ]]; then
      resume_checkpoint="${arm_state#resume:}"
      echo "Resuming ${label} seed ${seed} from ${resume_checkpoint}"
      args+=(--resume "$resume_checkpoint")
    else
      echo "Starting ${label} seed ${seed}"
    fi
    python3 -m mini_imagenet_gqa.train "${args[@]}"
  done
  python3 -m mini_imagenet_gqa.summarize_comparison --output-dir "$optimizer_dir"
done

reference_label="APOLLO-SF-INT8-Delta-${policies[0]}"
comparison_args=(
  --reference "$reference_label"
  --output-dir "$output_dir/paired-comparison"
  --expected-seeds "${seeds[@]}"
  --expected-variants "$variant"
)
for policy in "${policies[@]}"; do
  label="APOLLO-SF-INT8-Delta-${policy}"
  comparison_args+=(--input "$label=$output_dir/$label")
done
python3 -m mini_imagenet_gqa.summarize_optimizer_comparison "${comparison_args[@]}"
