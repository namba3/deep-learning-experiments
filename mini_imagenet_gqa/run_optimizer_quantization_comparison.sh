#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_dir="${OUTPUT_DIR:-mini_imagenet_gqa/output/bucketed/apollo-sf-quantization}"
variant="${VARIANT:-gated_gqa_silu_gated_ffn}"
epochs="${EPOCHS:-3}"
batch_size="${BATCH_SIZE:-64}"
num_workers="${NUM_WORKERS:-4}"
steps_per_epoch="${STEPS_PER_EPOCH:-0}"
eval_batches="${EVAL_BATCHES:-0}"
lr="${LR:-1e-3}"
weight_decay="${WEIGHT_DECAY:-0.05}"
amp="${AMP:-bf16}"
apollo_rank="${APOLLO_RANK:-32}"
quant_block_size="${APOLLO_SF_QUANT_BLOCK_SIZE:-256}"
delta_refresh="${APOLLO_SF_DELTA_REFRESH:-none}"
update_proj_gap="${APOLLO_UPDATE_PROJ_GAP:-200}"
seeds_csv="${SEEDS:-42}"
optimizers_csv="${OPTIMIZERS:-AdamW,AdamW-SF,APOLLO,APOLLO-SF,APOLLO-SF-INT8-Z,APOLLO-SF-INT8-Delta,APOLLO-SF-INT4-Z,APOLLO-SF-INT4-Delta}"
reference_optimizer="${REFERENCE_OPTIMIZER:-}"

IFS=',' read -r -a seeds <<< "$seeds_csv"
IFS=',' read -r -a optimizers <<< "$optimizers_csv"

for optimizer in "${optimizers[@]}"; do
  optimizer_dir="$output_dir/$optimizer"
  for seed in "${seeds[@]}"; do
    run_name="mini-imagenet-${optimizer}-seed${seed}-${variant}"
    arm_state="$(python3 -m mini_imagenet_gqa.summarize_comparison \
      --output-dir "$optimizer_dir" --run-name "$run_name" --arm-state)"
    if [[ "$arm_state" == "completed" ]]; then
      echo "Skipping completed ${optimizer} seed ${seed}"
      continue
    fi

    args=(
      --variant "$variant"
      --seed "$seed"
      --epochs "$epochs"
      --batch-size "$batch_size"
      --num-workers "$num_workers"
      --lr "$lr"
      --optimizer "$optimizer"
      --apollo-rank "$apollo_rank"
      --apollo-sf-quant-block-size "$quant_block_size"
      --apollo-sf-delta-refresh "$delta_refresh"
      --apollo-update-proj-gap "$update_proj_gap"
      --apollo-fallback came
      --apollo-matrix-fallback auto
      --lr-scheduler cosine
      --weight-decay "$weight_decay"
      --amp "$amp"
      --common-init
      --deterministic
      --device cuda
      --steps-per-epoch "$steps_per_epoch"
      --eval-batches "$eval_batches"
      --output-dir "$optimizer_dir"
      --run-name "$run_name"
    )
    if [[ "$arm_state" == resume:* ]]; then
      resume_checkpoint="${arm_state#resume:}"
      echo "Resuming ${optimizer} seed ${seed} from ${resume_checkpoint}"
      args+=(--resume "$resume_checkpoint")
    else
      echo "Starting ${optimizer} seed ${seed}"
    fi
    python3 -m mini_imagenet_gqa.train "${args[@]}"
  done
  python3 -m mini_imagenet_gqa.summarize_comparison --output-dir "$optimizer_dir"
done

# Full matrices use AdamW as the paired reference.  Focused subsets (for
# example, INT8-Delta versus LRSF) may not include AdamW, so use the first
# requested optimizer unless the caller explicitly selects another reference.
if [[ -z "$reference_optimizer" ]]; then
  reference_optimizer="${optimizers[0]}"
  for optimizer in "${optimizers[@]}"; do
    if [[ "$optimizer" == "AdamW" ]]; then
      reference_optimizer="AdamW"
      break
    fi
  done
fi

comparison_args=(
  --reference "$reference_optimizer"
  --output-dir "$output_dir/paired-comparison"
  --expected-seeds "${seeds[@]}"
  --expected-variants "$variant"
)
for optimizer in "${optimizers[@]}"; do
  comparison_args+=(
    --input "$optimizer=$output_dir/$optimizer"
  )
done
python3 -m mini_imagenet_gqa.summarize_optimizer_comparison "${comparison_args[@]}"
