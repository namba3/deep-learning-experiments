#!/usr/bin/env bash
set -euo pipefail

# Compare adapters at a fixed batch size.  This keeps optimizer-step count
# comparable while intentionally allowing tokens per step to grow with the
# context length.

export TEXT_LM_ADAPTER_BATCH_MODE=fixed_batch
export TEXT_LM_ADAPTER_BATCH_SIZE="${TEXT_LM_ADAPTER_BATCH_SIZE:-1}"
export TEXT_LM_ADAPTER_OUTPUT_DIR="${TEXT_LM_ADAPTER_OUTPUT_DIR:-output/text-lm-adapter-sequence-fixed-batch}"

if [[ -z "${TEXT_LM_ADAPTER_BASE_CHECKPOINT:-}" ]]; then
  base_output_dir="${TEXT_LM_ADAPTER_BASE_OUTPUT_DIR:-output/text-lm-adapter-base-2048}"
  latest_base_run="$(find "$base_output_dir/runs" -mindepth 1 -maxdepth 1 \
    -type d -name 'text_lm.train_*' -print 2>/dev/null | sort | tail -n 1)"
  discovered_checkpoint="$latest_base_run/artifacts/model.safetensors"
  if [[ -f "$discovered_checkpoint" ]]; then
    export TEXT_LM_ADAPTER_BASE_CHECKPOINT="$discovered_checkpoint"
    echo "Using latest base checkpoint: $TEXT_LM_ADAPTER_BASE_CHECKPOINT"
  else
    echo "TEXT_LM_ADAPTER_BASE_CHECKPOINT is not set and no base checkpoint was found under $base_output_dir" >&2
    echo "Run benchmarks/launchers/run_text_lm_adapter_base.sh first or set TEXT_LM_ADAPTER_BASE_CHECKPOINT explicitly." >&2
    exit 2
  fi
fi

exec bash benchmarks/launchers/run_text_lm_adapter_sequence_comparison.sh
