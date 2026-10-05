#!/usr/bin/env bash
set -euo pipefail

# Sweep learning rates for the AdamW/LRSF TinyStories optimizer comparison.
# Each rate gets an independent JSON artifact so runs can be compared without
# mixing optimizer state or seed results between configurations.

output_dir="${TEXT_LM_OPTIMIZER_SWEEP_DIR:-output/text-lm-optimizer-lr-sweep}"
learning_rates="${TEXT_LM_LEARNING_RATES:-1e-4,3e-4,1e-3}"

mkdir -p "$output_dir"
IFS=',' read -r -a rates <<< "$learning_rates"

for learning_rate in "${rates[@]}"; do
  learning_rate="${learning_rate// /}"
  if [[ -z "$learning_rate" ]]; then
    continue
  fi
  output_path="$output_dir/lr-${learning_rate}.json"
  echo "=== TinyStories optimizer LR=${learning_rate} ==="
  TEXT_LM_LEARNING_RATE="$learning_rate" \
    TEXT_LM_OPTIMIZER_OUTPUT="$output_path" \
    ./benchmarks/launchers/run_text_lm_optimizer_comparison.sh
done
