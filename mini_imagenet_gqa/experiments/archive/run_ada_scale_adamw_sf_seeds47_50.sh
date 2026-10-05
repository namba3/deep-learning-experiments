#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
output_dir="mini_imagenet_gqa/output/bucketed/adamw-sf-ada-seeds47-50"
seeds=(47 48 49 50)
variants=(
  gated_gqa_silu_gated_ffn
  ada_gated_gqa_silu_gated_ffn
  ada_1plus_silu_gated_gqa_silu_gated_ffn
  ada_softplus1_norm_gated_gqa_silu_gated_ffn
)

for seed in "${seeds[@]}"; do
  for variant in "${variants[@]}"; do
    echo "Starting AdamW-SF seed ${seed}: ${variant}"
    python3 -m mini_imagenet_gqa.train \
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
      --run-name "adamw-sf-seed${seed}-${variant}"
  done
done

python3 -m mini_imagenet_gqa.summarize_comparison --output-dir "$output_dir"
