#!/usr/bin/env bash
set -euo pipefail

# Sweep learning rate while keeping the parameter-matched adapter budgets
# fixed.  Each value reuses the same base checkpoint and writes runs into one
# directory; the report verifier groups learning rates separately.

output_dir="${TEXT_LM_ADAPTER_LR_SWEEP_OUTPUT_DIR:-output/text-lm-adapter-lr-sweep}"
learning_rates="${TEXT_LM_ADAPTER_LR_SWEEP_VALUES:-1e-4,3e-4,1e-3}"
max_seq_lens="${TEXT_LM_ADAPTER_MAX_SEQ_LENS:-2048}"
adapters="${TEXT_LM_ADAPTER_ADAPTERS:-lora,loha,dora,glu_lora,rglu_lora}"
seeds="${TEXT_LM_ADAPTER_SEEDS:-0,1,2}"
base_checkpoint="${TEXT_LM_ADAPTER_BASE_CHECKPOINT:-}"

if [[ -z "$base_checkpoint" ]]; then
  base_output_dir="${TEXT_LM_ADAPTER_BASE_OUTPUT_DIR:-output/text-lm-adapter-base-2048}"
  latest_base_run="$(find "$base_output_dir/runs" -mindepth 1 -maxdepth 1 \
    -type d -name 'text_lm.train_*' -print 2>/dev/null | sort | tail -n 1)"
  base_checkpoint="$latest_base_run/artifacts/model.safetensors"
  if [[ ! -f "$base_checkpoint" ]]; then
    echo "No base checkpoint found under $base_output_dir; set TEXT_LM_ADAPTER_BASE_CHECKPOINT." >&2
    exit 2
  fi
fi

mkdir -p "$output_dir"
IFS=',' read -r -a learning_rate_values <<< "$learning_rates"
for learning_rate in "${learning_rate_values[@]}"; do
  learning_rate="${learning_rate// /}"
  [[ -z "$learning_rate" ]] && continue
  echo "=== adapter learning-rate sweep: lr=${learning_rate} ==="
  TEXT_LM_ADAPTER_BASE_CHECKPOINT="$base_checkpoint" \
  TEXT_LM_ADAPTER_OUTPUT_DIR="$output_dir" \
  TEXT_LM_ADAPTER_RUN_TAG="lr-${learning_rate}" \
  TEXT_LM_ADAPTER_LR="$learning_rate" \
  TEXT_LM_ADAPTER_BATCH_MODE=fixed_batch \
  TEXT_LM_ADAPTER_BATCH_SIZE="${TEXT_LM_ADAPTER_BATCH_SIZE:-1}" \
  TEXT_LM_ADAPTER_RANK_MAP="${TEXT_LM_ADAPTER_RANK_MAP:-lora=16,dora=16,loha=8,glu_lora=8,rglu_lora=8}" \
  TEXT_LM_ADAPTER_ALPHA_MAP="${TEXT_LM_ADAPTER_ALPHA_MAP:-lora=16,dora=16,loha=8,glu_lora=8,rglu_lora=8}" \
  TEXT_LM_ADAPTER_MAX_SEQ_LENS="$max_seq_lens" \
  TEXT_LM_ADAPTER_ADAPTERS="$adapters" \
  TEXT_LM_ADAPTER_SEEDS="$seeds" \
  bash benchmarks/launchers/run_text_lm_adapter_sequence_comparison.sh
done
