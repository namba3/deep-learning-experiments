#!/usr/bin/env bash
set -euo pipefail

all_variants=(
  naive_gqa
  ada_naive_gqa
  gated_gqa_silu_gated_ffn
  ada_gated_gqa_silu_gated_ffn
)
read -r -a seeds <<< "${SEEDS:-42}"
read -r -a variants <<< "${VARIANTS:-${all_variants[*]}}"
epochs="${EPOCHS:-30}"
batch_size="${BATCH_SIZE:-64}"
steps_per_epoch="${STEPS_PER_EPOCH:-0}"
eval_batches="${EVAL_BATCHES:-0}"
output_dir="${OUTPUT_DIR:-mini_imagenet_gqa/output/bucketed}"

if ((${#seeds[@]} == 0 || ${#variants[@]} == 0)); then
  echo "SEEDS and VARIANTS must each contain at least one value" >&2
  exit 2
fi

for seed in "${seeds[@]}"; do
  for variant in "${variants[@]}"; do
    args=(
      --variant "$variant"
      --seed "$seed"
      --device "${DEVICE:-cuda}"
      --amp "${AMP:-bf16}"
      --epochs "$epochs"
      --batch-size "$batch_size"
      --steps-per-epoch "$steps_per_epoch"
      --eval-batches "$eval_batches"
      --output-dir "$output_dir"
      --run-name "${variant}-seed-${seed}"
    )
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      args+=(--dry-run --device cpu)
    fi
    PYTHONPATH=. python3 -m mini_imagenet_gqa.train "${args[@]}"
  done
done
