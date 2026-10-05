#!/usr/bin/env bash
set -uo pipefail

# Matched short run to measure activation-checkpointing memory and throughput.
# Run only after other GPU jobs release the device. Both arms use the same seed,
# dataset split, sample budget, and model; only gradient checkpointing changes.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
epochs="${VFP_DIT_COMPARE_EPOCHS:-1}"
samples_per_epoch="${VFP_DIT_COMPARE_SAMPLES_PER_EPOCH:-8}"
validation_samples="${VFP_DIT_COMPARE_VALIDATION_SAMPLES:-8}"
seed="${VFP_DIT_COMPARE_SEED:-42}"
base_run_name="${VFP_DIT_COMPARE_RUN_NAME:-checkpointing-compare-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${VFP_DIT_OUTPUT_DIR:-vfp_dit/output}"
comparison_order="${VFP_DIT_COMPARE_ORDER:-disabled-first}"
failed=0

case "$comparison_order" in
  disabled-first) checkpointing_modes=(0 1) ;;
  enabled-first) checkpointing_modes=(1 0) ;;
  *)
    echo "VFP_DIT_COMPARE_ORDER must be disabled-first or enabled-first" >&2
    exit 2
    ;;
esac

for checkpointing in "${checkpointing_modes[@]}"; do
  mode="disabled"
  [[ "$checkpointing" == "1" ]] && mode="enabled"
  echo "=== activation checkpointing ${mode} ==="
  if (
    export VFP_DIT_EPOCHS="$epochs"
    export VFP_DIT_SAMPLES_PER_EPOCH="$samples_per_epoch"
    export VFP_DIT_VALIDATION_SAMPLES="$validation_samples"
    export VFP_DIT_SEED="$seed"
    export VFP_DIT_RUN_NAME="${base_run_name}-${mode}"
    export VFP_DIT_GRADIENT_CHECKPOINTING="$checkpointing"
    export VFP_DIT_OUTPUT_DIR="$output_dir"
    export VFP_DIT_GENERATE_SAMPLES=0
    cd "$repo_root"
    bash benchmarks/launchers/run_vfp_dit_screen.sh
  ); then
    echo "Completed: ${base_run_name}-${mode}"
  else
    status=$?
    echo "Failed (exit ${status}): ${base_run_name}-${mode}" >&2
    failed=1
  fi
done

PYTHONPATH="${repo_root}:${PYTHONPATH:-}" python3 -m vfp_dit.summarize_checkpointing \
  --output-dir "$output_dir" --base-run-name "$base_run_name"

if [[ "$failed" == "1" ]]; then
  echo "At least one arm failed; inspect logs and the summary under ${output_dir}." >&2
  exit 1
fi
