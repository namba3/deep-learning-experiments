#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

variant="ada_1plus_softplus_halfnorm_gated_gqa_silu_gated_ffn"
seeds=(47 48 49 50)
experiment_dir="mini_imagenet_gqa/output/bucketed/ada-oneplus-softplus-half-normalized-optimizers-seeds47-50"

run_optimizer() {
  local label="$1"
  local optimizer="$2"
  local output_dir="$experiment_dir/$label"
  for seed in "${seeds[@]}"; do
    local run_name="${label}-seed${seed}-oneplus-softplus-half-normalized"
    local arm_state
    arm_state="$(python3 -m mini_imagenet_gqa.summarize_comparison \
      --output-dir "$output_dir" --run-name "$run_name" --arm-state)"
    if [[ "$arm_state" == "completed" ]]; then
      echo "Skipping completed ${optimizer} seed ${seed}"
      continue
    fi

    local args=(
      --variant "$variant"
      --seed "$seed"
      --epochs 10
      --batch-size 64
      --num-workers 4
      --lr 1e-3
      --optimizer "$optimizer"
      --conditioning-lr-multiplier 1.0
      --weight-decay 0.05
      --amp bf16
      --common-init
      --deterministic
      --device cuda
      --steps-per-epoch 0
      --eval-batches 0
      --output-dir "$output_dir"
      --run-name "$run_name"
    )
    if [[ "$optimizer" == "AdamW-SF" ]]; then
      args+=(--adamw-sf-backend torch)
    fi
    if [[ "$arm_state" == resume:* ]]; then
      local resume_checkpoint="${arm_state#resume:}"
      echo "Resuming ${optimizer} seed ${seed} from ${resume_checkpoint}"
      args+=(--resume "$resume_checkpoint")
    else
      echo "Starting ${optimizer} seed ${seed}: ${variant}"
    fi
    python3 -m mini_imagenet_gqa.train "${args[@]}"
  done
  python3 -m mini_imagenet_gqa.summarize_comparison --output-dir "$output_dir"
}

run_optimizer adamw AdamW
run_optimizer adamw-sf AdamW-SF

python3 -m mini_imagenet_gqa.summarize_optimizer_comparison \
  --input "AdamW=$experiment_dir/adamw" \
  --input "AdamW-SF=$experiment_dir/adamw-sf" \
  --expected-variants "$variant" \
  --output-dir "$experiment_dir/optimizer-comparison"
