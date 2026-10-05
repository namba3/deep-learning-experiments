#!/usr/bin/env bash
set -euo pipefail

# Run the RGLU-LoRA rank/alpha sweep on the real CIFAR-10 subset. The
# defaults are bounded so the first GPU run remains small enough to inspect.
data_dir="${CIFAR10_RGLU_LORA_DATA_DIR:-cifar10/data}"
device="${CIFAR10_RGLU_LORA_DEVICE:-cuda}"
dtype="${CIFAR10_RGLU_LORA_DTYPE:-bf16}"
seeds="${CIFAR10_RGLU_LORA_SEEDS:-0,1,2}"
ranks="${CIFAR10_RGLU_LORA_RANKS:-1,4,8}"
alphas="${CIFAR10_RGLU_LORA_ALPHAS:-}"
epochs="${CIFAR10_RGLU_LORA_EPOCHS:-3}"
batch_size="${CIFAR10_RGLU_LORA_BATCH_SIZE:-16}"
max_train_samples="${CIFAR10_RGLU_LORA_MAX_TRAIN_SAMPLES:-512}"
max_validation_samples="${CIFAR10_RGLU_LORA_MAX_VALIDATION_SAMPLES:-256}"
learning_rate="${CIFAR10_RGLU_LORA_LEARNING_RATE:-1e-3}"
adapter_init="${CIFAR10_RGLU_LORA_ADAPTER_INIT:-identity}"
output="${CIFAR10_RGLU_LORA_OUTPUT:-output/cifar10-rglu-lora-sweep-gpu.json}"

command_args=(
  --data-dir "$data_dir"
  --device "$device"
  --dtype "$dtype"
  --seeds "$seeds"
  --ranks "$ranks"
  --epochs "$epochs"
  --batch-size "$batch_size"
  --max-train-samples "$max_train_samples"
  --max-validation-samples "$max_validation_samples"
  --learning-rate "$learning_rate"
  --adapter-init "$adapter_init"
  --output "$output"
)
if [[ -n "$alphas" ]]; then
  command_args+=(--alphas "$alphas")
fi

PYTHONPATH=. python3 -m verify.cifar10_rglu_lora_sweep "${command_args[@]}"
