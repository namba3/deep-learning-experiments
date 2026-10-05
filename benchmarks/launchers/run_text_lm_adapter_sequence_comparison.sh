#!/usr/bin/env bash
set -euo pipefail

# Compare low-rank adapters at several context lengths. By default the number
# of tokens per optimizer step is fixed; fixed_batch mode is also available to
# expose the effect of changing tokens per optimizer step.

output_dir="${TEXT_LM_ADAPTER_OUTPUT_DIR:-output/text-lm-adapter-sequence-comparison}"
base_checkpoint="${TEXT_LM_ADAPTER_BASE_CHECKPOINT:?set TEXT_LM_ADAPTER_BASE_CHECKPOINT}"
dataset_name="${TEXT_LM_ADAPTER_DATASET:-roneneldan/TinyStories}"
tokenizer="${TEXT_LM_ADAPTER_TOKENIZER:-Qwen/Qwen3.5-0.8B}"
max_seq_len_list="${TEXT_LM_ADAPTER_MAX_SEQ_LENS:-512,1024,2048}"
adapter_list="${TEXT_LM_ADAPTER_ADAPTERS:-lora,loha,dora,glu_lora,rglu_lora}"
seed_list="${TEXT_LM_ADAPTER_SEEDS:-0,1,2}"
device="${TEXT_LM_ADAPTER_DEVICE:-auto}"
train_tokens="${TEXT_LM_ADAPTER_TRAIN_TOKENS:-1048576}"
eval_tokens="${TEXT_LM_ADAPTER_EVAL_TOKENS:-131072}"
tokens_per_step="${TEXT_LM_ADAPTER_TOKENS_PER_STEP:-2048}"
batch_mode="${TEXT_LM_ADAPTER_BATCH_MODE:-token_normalized}"
fixed_batch_size="${TEXT_LM_ADAPTER_BATCH_SIZE:-1}"
epochs="${TEXT_LM_ADAPTER_EPOCHS:-3}"
steps_per_epoch="${TEXT_LM_ADAPTER_STEPS_PER_EPOCH:-200}"
eval_max_batches="${TEXT_LM_ADAPTER_EVAL_MAX_BATCHES:-64}"
embed_dim="${TEXT_LM_ADAPTER_EMBED_DIM:-128}"
num_layers="${TEXT_LM_ADAPTER_NUM_LAYERS:-2}"
num_heads="${TEXT_LM_ADAPTER_NUM_HEADS:-4}"
kv_heads="${TEXT_LM_ADAPTER_KV_HEADS:-4}"
condition_dim="${TEXT_LM_ADAPTER_CONDITION_DIM:-64}"
transform_rank="${TEXT_LM_ADAPTER_TRANSFORM_RANK:-10}"
rank="${TEXT_LM_ADAPTER_RANK:-4}"
alpha="${TEXT_LM_ADAPTER_ALPHA:-4}"
rank_map="${TEXT_LM_ADAPTER_RANK_MAP:-}"
alpha_map="${TEXT_LM_ADAPTER_ALPHA_MAP:-}"
learning_rate="${TEXT_LM_ADAPTER_LR:-3e-4}"
num_workers="${TEXT_LM_ADAPTER_NUM_WORKERS:-0}"
bf16="${TEXT_LM_ADAPTER_BF16:-0}"
vocab_chunk_size="${TEXT_LM_ADAPTER_VOCAB_CHUNK_SIZE:-8192}"
run_tag="${TEXT_LM_ADAPTER_RUN_TAG:-}"

IFS=',' read -r -a max_seq_lens <<< "$max_seq_len_list"
IFS=',' read -r -a adapters <<< "$adapter_list"
IFS=',' read -r -a seeds <<< "$seed_list"

resolve_adapter_value() {
  local mapping="$1"
  local adapter_name="$2"
  local fallback="$3"
  local entry key value
  [[ -z "$mapping" ]] && { echo "$fallback"; return; }
  IFS=',' read -r -a entries <<< "$mapping"
  for entry in "${entries[@]}"; do
    entry="${entry// /}"
    key="${entry%%=*}"
    value="${entry#*=}"
    if [[ "$key" == "$adapter_name" ]]; then
      echo "$value"
      return
    fi
  done
  echo "$fallback"
}

case "$batch_mode" in
  token_normalized)
    if (( tokens_per_step <= 0 )); then
      echo "TEXT_LM_ADAPTER_TOKENS_PER_STEP must be positive" >&2
      exit 2
    fi
    ;;
  fixed_batch)
    if (( fixed_batch_size <= 0 )); then
      echo "TEXT_LM_ADAPTER_BATCH_SIZE must be positive" >&2
      exit 2
    fi
    ;;
  *)
    echo "TEXT_LM_ADAPTER_BATCH_MODE must be token_normalized or fixed_batch" >&2
    exit 2
    ;;
esac

mkdir -p "$output_dir"

for max_seq_len in "${max_seq_lens[@]}"; do
  max_seq_len="${max_seq_len// /}"
  if (( max_seq_len <= 1 )); then
    echo "sequence length must be greater than one: ${max_seq_len}" >&2
    exit 2
  fi
  if [[ "$batch_mode" == "token_normalized" ]]; then
    if (( tokens_per_step % max_seq_len != 0 )); then
      echo "sequence length must divide tokens_per_step: ${max_seq_len}" >&2
      exit 2
    fi
    batch_size=$((tokens_per_step / max_seq_len))
  else
    batch_size=$fixed_batch_size
  fi
  effective_tokens_per_step=$((max_seq_len * batch_size))
  if (( train_tokens < steps_per_epoch * effective_tokens_per_step )); then
    echo "train token budget is too small for context ${max_seq_len}" >&2
    exit 2
  fi
  if (( eval_tokens < eval_max_batches * effective_tokens_per_step )); then
    echo "eval token budget is too small for context ${max_seq_len}" >&2
    exit 2
  fi
  for seed in "${seeds[@]}"; do
    for adapter in "${adapters[@]}"; do
      adapter="${adapter// /}"
      adapter_rank="$(resolve_adapter_value "$rank_map" "$adapter" "$rank")"
      adapter_alpha="$(resolve_adapter_value "$alpha_map" "$adapter" "$adapter_rank")"
      if (( adapter_rank <= 0 )); then
        echo "adapter rank must be positive: ${adapter}" >&2
        exit 2
      fi
      run_name="tinystories-ctx-${max_seq_len}-${adapter}-r${adapter_rank}-seed-${seed}-batch-${batch_mode}"
      if [[ -n "$run_tag" ]]; then
        run_name="${run_tag}-${run_name}"
      fi
      log_path="$output_dir/${run_name}.log"
      echo "=== ${run_name} (batch=${batch_size}, tokens/step=${effective_tokens_per_step}) ==="
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
        --condition-dim "$condition_dim"
        --transform-rank "$transform_rank"
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
        --dataset-seed 0
        --seed "$seed"
        --lora-base-checkpoint "$base_checkpoint"
        --adapter "$adapter"
        --lora-rank "$adapter_rank"
        --lora-alpha "$adapter_alpha"
        --save-mode final
        --output-dir "$output_dir"
        --run-name "$run_name"
      )
      if [[ "$bf16" == "1" ]]; then
        command_args+=(--bf16)
      fi
      HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
      HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
      PYTHONPATH=. python3 -m text_lm.train_adapter "${command_args[@]}" \
        2>&1 | tee "$log_path"
    done
  done
done

PYTHONPATH=. python3 -m verify.text_lm_adapter_report \
  --input-dir "$output_dir" \
  --output "$output_dir/report.json"
