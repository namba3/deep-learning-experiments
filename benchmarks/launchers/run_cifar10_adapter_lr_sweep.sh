#!/usr/bin/env bash
set -euo pipefail

# Sweep learning rate while keeping the adapter budget and data conditions
# fixed. Each learning rate gets its own JSON file for unambiguous analysis.
output_dir="${CIFAR10_ADAPTER_LR_SWEEP_OUTPUT_DIR:-output/cifar10-adapter-lr-sweep}"
learning_rates="${CIFAR10_ADAPTER_LR_SWEEP_VALUES:-3e-4,1e-3,3e-3}"
budget="${CIFAR10_ADAPTER_LR_SWEEP_BUDGET:-rank32}"
mkdir -p "$output_dir"

IFS=',' read -r -a learning_rate_values <<< "$learning_rates"
for learning_rate in "${learning_rate_values[@]}"; do
  learning_rate="${learning_rate// /}"
  [[ -z "$learning_rate" ]] && continue
  echo "=== CIFAR-10 adapter learning-rate sweep: lr=${learning_rate} ==="
  CIFAR10_ADAPTER_BUDGET="$budget" \
  CIFAR10_ADAPTER_LEARNING_RATE="$learning_rate" \
  CIFAR10_ADAPTER_OUTPUT="$output_dir/lr-${learning_rate}.json" \
  bash benchmarks/launchers/run_cifar10_adapter_budget_comparison.sh
done
