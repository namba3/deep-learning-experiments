#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
output_dir="mini_imagenet_gqa/output/bucketed/adamw-sf-normalized-silu-seeds47-50"
variant="ada_silu1_norm_gated_gqa_silu_gated_ffn"
seeds=(47 48 49 50)

for seed in "${seeds[@]}"; do
  run_name="adamw-sf-normalized-silu-seed${seed}"
  arm_state="$(python3 -m mini_imagenet_gqa.summarize_comparison \
    --output-dir "$output_dir" --run-name "$run_name" --arm-state)"
  if [[ "$arm_state" == "completed" ]]; then
    echo "Skipping completed AdamW-SF normalized-SiLU seed ${seed}"
    continue
  fi

  args=(
    --variant "$variant" \
    --seed "$seed" \
    --epochs 10 \
    --batch-size 64 \
    --num-workers 4 \
    --lr 1e-3 \
    --optimizer AdamW-SF \
    --adamw-sf-backend torch \
    --conditioning-lr-multiplier 1.0 \
    --weight-decay 0.05 \
    --amp bf16 \
    --common-init \
    --deterministic \
    --device cuda \
    --steps-per-epoch 0 \
    --eval-batches 0 \
    --output-dir "$output_dir" \
    --run-name "$run_name"
  )
  if [[ "$arm_state" == resume:* ]]; then
    resume_checkpoint="${arm_state#resume:}"
    echo "Resuming AdamW-SF normalized-SiLU seed ${seed} from ${resume_checkpoint}"
    args+=(--resume "$resume_checkpoint")
  else
    echo "Starting AdamW-SF normalized-SiLU seed ${seed}"
  fi
  python3 -m mini_imagenet_gqa.train "${args[@]}"
done

python3 -m mini_imagenet_gqa.summarize_comparison --output-dir "$output_dir"
