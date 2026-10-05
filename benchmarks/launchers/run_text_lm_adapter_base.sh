#!/usr/bin/env bash
set -euo pipefail

# Train the plain text-lm base checkpoint used by the long-context adapter
# comparison. Keep the model and data settings aligned with the adapter script.

output_dir="${TEXT_LM_ADAPTER_BASE_OUTPUT_DIR:-output/text-lm-adapter-base-2048}"
dataset_name="${TEXT_LM_ADAPTER_DATASET:-roneneldan/TinyStories}"
tokenizer="${TEXT_LM_ADAPTER_TOKENIZER:-Qwen/Qwen3.5-0.8B}"
device="${TEXT_LM_ADAPTER_DEVICE:-auto}"
max_seq_len="${TEXT_LM_ADAPTER_BASE_MAX_SEQ_LEN:-2048}"
train_tokens="${TEXT_LM_ADAPTER_TRAIN_TOKENS:-1048576}"
eval_tokens="${TEXT_LM_ADAPTER_EVAL_TOKENS:-131072}"
batch_size="${TEXT_LM_ADAPTER_BASE_BATCH_SIZE:-1}"
epochs="${TEXT_LM_ADAPTER_EPOCHS:-3}"
steps_per_epoch="${TEXT_LM_ADAPTER_STEPS_PER_EPOCH:-200}"
eval_max_batches="${TEXT_LM_ADAPTER_EVAL_MAX_BATCHES:-64}"
embed_dim="${TEXT_LM_ADAPTER_EMBED_DIM:-128}"
num_layers="${TEXT_LM_ADAPTER_NUM_LAYERS:-2}"
num_heads="${TEXT_LM_ADAPTER_NUM_HEADS:-4}"
kv_heads="${TEXT_LM_ADAPTER_KV_HEADS:-4}"
learning_rate="${TEXT_LM_ADAPTER_LR:-3e-4}"
num_workers="${TEXT_LM_ADAPTER_NUM_WORKERS:-0}"
seed="${TEXT_LM_ADAPTER_BASE_SEED:-0}"
dataset_seed="${TEXT_LM_ADAPTER_DATASET_SEED:-0}"
bf16="${TEXT_LM_ADAPTER_BF16:-0}"
vocab_chunk_size="${TEXT_LM_ADAPTER_VOCAB_CHUNK_SIZE:-8192}"
run_name="${TEXT_LM_ADAPTER_BASE_RUN_NAME:-base-naive-tinystories-ctx-${max_seq_len}}"

if (( max_seq_len <= 1 || batch_size <= 0 )); then
  echo "max sequence length must be greater than one and batch size must be positive" >&2
  exit 2
fi
effective_tokens_per_step=$((max_seq_len * batch_size))
if (( train_tokens < steps_per_epoch * effective_tokens_per_step )); then
  echo "train token budget is too small for the requested steps/epoch" >&2
  exit 2
fi
if (( eval_tokens < eval_max_batches * effective_tokens_per_step )); then
  echo "eval token budget is too small for the requested eval batches" >&2
  exit 2
fi

mkdir -p "$output_dir"
command_args=(
  --data-mode text
  --dataset-name "$dataset_name"
  --tokenizer "$tokenizer"
  --architecture naive
  --device "$device"
  --max-seq-len "$max_seq_len"
  --embed-dim "$embed_dim"
  --num-layers "$num_layers"
  --num-heads "$num_heads"
  --kv-heads "$kv_heads"
  --batch-size "$batch_size"
  --vocab-chunk-size "$vocab_chunk_size"
  --max-train-tokens "$train_tokens"
  --max-eval-tokens "$eval_tokens"
  --eval-max-batches "$eval_max_batches"
  --epochs "$epochs"
  --steps-per-epoch "$steps_per_epoch"
  --num-workers "$num_workers"
  --optimizer AdamW
  --lr "$learning_rate"
  --lr-scheduler constant
  --warmup-steps 0
  --dataset-seed "$dataset_seed"
  --seed "$seed"
  --save-mode final
  --output-dir "$output_dir"
  --run-name "$run_name"
)
if [[ "$bf16" == "1" ]]; then
  command_args+=(--bf16)
fi

HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
PYTHONPATH=. python3 -m text_lm.train "${command_args[@]}"

latest_run="$(find "$output_dir/runs" -mindepth 1 -maxdepth 1 -type d \
  -name "text_lm.train_${run_name}_*" | sort | tail -n 1)"
if [[ -z "$latest_run" || ! -f "$latest_run/artifacts/model.safetensors" ]]; then
  echo "could not locate the generated base checkpoint under $output_dir/runs" >&2
  exit 1
fi
echo "Base checkpoint: $latest_run/artifacts/model.safetensors"
