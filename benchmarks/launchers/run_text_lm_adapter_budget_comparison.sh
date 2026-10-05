#!/usr/bin/env bash
set -euo pipefail

# Parameter-matched adapter comparison.  Override COMPARISON_MODE=rank_matched
# to compare all adapters at the same rank instead.

comparison_mode="${TEXT_LM_ADAPTER_COMPARISON_MODE:-parameter_matched}"
case "$comparison_mode" in
  parameter_matched)
    default_rank_map="lora=16,dora=16,loha=8,glu_lora=8,rglu_lora=8"
    default_alpha_map="lora=16,dora=16,loha=8,glu_lora=8,rglu_lora=8"
    ;;
  rank_matched)
    default_rank_map="lora=8,dora=8,loha=8,glu_lora=8,rglu_lora=8"
    default_alpha_map="lora=8,dora=8,loha=8,glu_lora=8,rglu_lora=8"
    ;;
  *)
    echo "TEXT_LM_ADAPTER_COMPARISON_MODE must be parameter_matched or rank_matched" >&2
    exit 2
    ;;
esac

export TEXT_LM_ADAPTER_OUTPUT_DIR="${TEXT_LM_ADAPTER_OUTPUT_DIR:-output/text-lm-adapter-budget-${comparison_mode}}"
export TEXT_LM_ADAPTER_RANK_MAP="${TEXT_LM_ADAPTER_RANK_MAP:-$default_rank_map}"
export TEXT_LM_ADAPTER_ALPHA_MAP="${TEXT_LM_ADAPTER_ALPHA_MAP:-$default_alpha_map}"

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
