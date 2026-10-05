#!/usr/bin/env bash
set -euo pipefail

# Sweep the LRSF rank on the TinyStories optimizer comparison.
# Each rank gets an independent JSON artifact so optimizer state is never
# reused between rank configurations.  rank=512 is the full-SF fallback
# oracle for the default 512-wide model.

output_dir="${TEXT_LM_OPTIMIZER_RANK_SWEEP_DIR:-output/text-lm-optimizer-rank-sweep}"
ranks="${TEXT_LM_RANKS:-4,8,16,512}"
learning_rate="${TEXT_LM_LEARNING_RATE:-1e-3}"
optimizers="${TEXT_LM_OPTIMIZERS:-AdamW-SF,AdamW-LRSF}"

mkdir -p "$output_dir"
IFS=',' read -r -a rank_values <<< "$ranks"

for rank in "${rank_values[@]}"; do
  rank="${rank// /}"
  if [[ -z "$rank" ]]; then
    continue
  fi
  output_path="$output_dir/rank-${rank}.json"
  echo "=== TinyStories optimizer rank=${rank} LR=${learning_rate} ==="
  TEXT_LM_RANK="$rank" \
    TEXT_LM_LEARNING_RATE="$learning_rate" \
    TEXT_LM_OPTIMIZERS="$optimizers" \
    TEXT_LM_OPTIMIZER_OUTPUT="$output_path" \
    ./benchmarks/launchers/run_text_lm_optimizer_comparison.sh
done
