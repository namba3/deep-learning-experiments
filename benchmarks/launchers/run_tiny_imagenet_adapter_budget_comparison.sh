#!/usr/bin/env bash
set -euo pipefail

# Compare adapters on TinyImageNet-200 with approximately matched trainable
# parameter budgets.  The dataset is resolved through Hugging Face datasets.
budget="${TINY_IMAGENET_ADAPTER_BUDGET:-rank32}"
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
    echo "TINY_IMAGENET_ADAPTER_BUDGET must be standard or rank32" >&2
    exit 2
    ;;
esac

dataset_name="${TINY_IMAGENET_ADAPTER_DATASET:-zh-plus/tiny-imagenet}"
cache_dir="${TINY_IMAGENET_ADAPTER_CACHE_DIR:-}"
device="${TINY_IMAGENET_ADAPTER_DEVICE:-cuda}"
dtype="${TINY_IMAGENET_ADAPTER_DTYPE:-bf16}"
seeds="${TINY_IMAGENET_ADAPTER_SEEDS:-0,1,2}"
adapters="${TINY_IMAGENET_ADAPTERS:-lora,loha,dora,glu_lora,rglu_lora}"
rank_map="${TINY_IMAGENET_ADAPTER_RANK_MAP:-$default_rank_map}"
alpha_map="${TINY_IMAGENET_ADAPTER_ALPHA_MAP:-$default_alpha_map}"
epochs="${TINY_IMAGENET_ADAPTER_EPOCHS:-3}"
batch_size="${TINY_IMAGENET_ADAPTER_BATCH_SIZE:-32}"
max_train_samples="${TINY_IMAGENET_ADAPTER_MAX_TRAIN_SAMPLES:-1024}"
max_validation_samples="${TINY_IMAGENET_ADAPTER_MAX_VALIDATION_SAMPLES:-512}"
learning_rate="${TINY_IMAGENET_ADAPTER_LEARNING_RATE:-1e-3}"
adapter_init="${TINY_IMAGENET_ADAPTER_INIT:-identity}"
train_classifier_head="${TINY_IMAGENET_ADAPTER_TRAIN_CLASSIFIER_HEAD:-0}"
output="${TINY_IMAGENET_ADAPTER_OUTPUT:-output/tiny-imagenet-adapter-budget-comparison.json}"

dataset_args=(--dataset-name "$dataset_name")
if [[ -n "$cache_dir" ]]; then
  dataset_args+=(--cache-dir "$cache_dir")
fi
classifier_head_args=()
if [[ "$train_classifier_head" == "1" ]]; then
  classifier_head_args+=(--train-classifier-head)
fi

PYTHONPATH=. python3 -m verify.tiny_imagenet_adapter_dataset_comparison \
  "${dataset_args[@]}" \
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
  "${classifier_head_args[@]}" \
  --output "$output"
