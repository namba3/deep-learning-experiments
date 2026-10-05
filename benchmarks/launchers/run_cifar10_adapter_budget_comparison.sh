#!/usr/bin/env bash
set -euo pipefail

# Compare CIFAR-10 adapters under approximately matched trainable-parameter
# budgets. The default profile uses LoRA/DoRA rank 16 and LoHA/RGLU-LoRA
# rank 8; the rank32 profile doubles both ranks.
budget="${CIFAR10_ADAPTER_BUDGET:-standard}"
case "$budget" in
  standard)
    default_rank_map="lora=16,dora=16,loha=8,glu_lora=8,rglu_lora=8"
    default_alpha_map="lora=16,dora=16,loha=8,glu_lora=8,rglu_lora=8"
    ;;
  rank32)
    default_rank_map="lora=32,dora=32,loha=16,glu_lora=16,rglu_lora=16"
    default_alpha_map="lora=32,dora=32,loha=16,glu_lora=16,rglu_lora=16"
    ;;
  *)
    echo "CIFAR10_ADAPTER_BUDGET must be standard or rank32" >&2
    exit 2
    ;;
esac
data_dir="${CIFAR10_ADAPTER_DATA_DIR:-cifar10/data}"
device="${CIFAR10_ADAPTER_DEVICE:-cuda}"
dtype="${CIFAR10_ADAPTER_DTYPE:-bf16}"
seeds="${CIFAR10_ADAPTER_SEEDS:-0,1,2}"
adapters="${CIFAR10_ADAPTERS:-lora,loha,dora,glu_lora,rglu_lora}"
rank_map="${CIFAR10_ADAPTER_RANK_MAP:-$default_rank_map}"
alpha_map="${CIFAR10_ADAPTER_ALPHA_MAP:-$default_alpha_map}"
epochs="${CIFAR10_ADAPTER_EPOCHS:-3}"
batch_size="${CIFAR10_ADAPTER_BATCH_SIZE:-16}"
max_train_samples="${CIFAR10_ADAPTER_MAX_TRAIN_SAMPLES:-512}"
max_validation_samples="${CIFAR10_ADAPTER_MAX_VALIDATION_SAMPLES:-256}"
learning_rate="${CIFAR10_ADAPTER_LEARNING_RATE:-1e-3}"
adapter_init="${CIFAR10_ADAPTER_INIT:-identity}"
output="${CIFAR10_ADAPTER_OUTPUT:-output/cifar10-adapter-budget-comparison.json}"

PYTHONPATH=. python3 -m verify.cifar10_adapter_dataset_comparison \
  --data-dir "$data_dir" \
  --device "$device" \
  --dtype "$dtype" \
  --seeds "$seeds" \
  --adapters "$adapters" \
  --rank-map "$rank_map" \
  --alpha-map "$alpha_map" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --max-train-samples "$max_train_samples" \
  --max-validation-samples "$max_validation_samples" \
  --learning-rate "$learning_rate" \
  --adapter-init "$adapter_init" \
  --output "$output"
