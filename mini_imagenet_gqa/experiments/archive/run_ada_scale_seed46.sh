#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
output_dir="mini_imagenet_gqa/output/bucketed/ada-scale-vs-no-ada-same-lr-10e-seed46"

variants=(
  gated_gqa_silu_gated_ffn
  ada_gated_gqa_silu_gated_ffn
  ada_1plus_silu_gated_gqa_silu_gated_ffn
  ada_2sigmoid_gated_gqa_silu_gated_ffn
  ada_silu1_norm_gated_gqa_silu_gated_ffn
  ada_softplus1_norm_gated_gqa_silu_gated_ffn
)

for variant in "${variants[@]}"; do
  echo "Starting seed 46: ${variant}"
  python3 -m mini_imagenet_gqa.train \
    --variant "$variant" \
    --seed 46 \
    --epochs 10 \
    --batch-size 64 \
    --num-workers 4 \
    --lr 1e-3 \
    --conditioning-lr-multiplier 1.0 \
    --weight-decay 0.05 \
    --amp bf16 \
    --common-init \
    --deterministic \
    --device cuda \
    --steps-per-epoch 0 \
    --eval-batches 0 \
    --output-dir "$output_dir" \
    --run-name "seed46-${variant}"
done

python3 -m mini_imagenet_gqa.summarize_comparison --output-dir "$output_dir"
