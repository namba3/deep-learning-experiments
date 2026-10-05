#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
output_dir="mini_imagenet_gqa/output/bucketed/apollo-sf-warmup-ada-seeds47-50"
seeds=(47 48 49 50)
variants=(
  gated_gqa_silu_gated_ffn
  ada_gated_gqa_silu_gated_ffn
  ada_1plus_silu_gated_gqa_silu_gated_ffn
  ada_softplus1_norm_gated_gqa_silu_gated_ffn
)

for seed in "${seeds[@]}"; do
  for variant in "${variants[@]}"; do
    run_name="apollo-sf-warmup-seed${seed}-${variant}"
    arm_state="$(python3 -m mini_imagenet_gqa.summarize_comparison \
      --output-dir "$output_dir" --run-name "$run_name" --arm-state)"
    if [[ "$arm_state" == "completed" ]]; then
      echo "Skipping completed APOLLO + AdamW-SF seed ${seed}: ${variant}"
      continue
    fi

    args=(
      --variant "$variant" \
      --seed "$seed" \
      --epochs 10 \
      --batch-size 64 \
      --num-workers 4 \
      --lr 1e-3 \
      --optimizer APOLLO \
      --apollo-rank 32 \
      --apollo-fallback adamw-sf \
      --apollo-matrix-fallback auto-sf \
      --lr-scheduler cosine \
      --warmup-ratio 0.05 \
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
      echo "Resuming APOLLO + AdamW-SF seed ${seed}: ${variant} from ${resume_checkpoint}"
      args+=(--resume "$resume_checkpoint")
    else
      echo "Starting APOLLO + AdamW-SF seed ${seed}: ${variant}"
    fi
    python3 -m mini_imagenet_gqa.train "${args[@]}"
  done
done

python3 -m mini_imagenet_gqa.summarize_comparison --output-dir "$output_dir"
