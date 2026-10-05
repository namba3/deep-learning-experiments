#!/usr/bin/env bash
set -euo pipefail

# Short CUDA smoke comparison for AdamW-SF and AdamW-LRSF refresh policies.
# This is intentionally not a quality benchmark: it checks that refresh paths
# execute, emit comparable state metrics, and remain numerically finite before
# spending resources on the TinyStories long comparison.

output_root="${TEXT_LM_REFRESH_SMOKE_DIR:-output/text-lm-optimizer-refresh-smoke}"
modes="${TEXT_LM_REFRESH_SMOKE_MODES:-hard,ema_fast,frozen}"
seeds="${TEXT_LM_REFRESH_SMOKE_SEEDS:-0}"
optimizers="${TEXT_LM_REFRESH_SMOKE_OPTIMIZERS:-AdamW-SF,AdamW-LRSF}"
device="${TEXT_LM_REFRESH_SMOKE_DEVICE:-cuda}"
dtype="${TEXT_LM_REFRESH_SMOKE_DTYPE:-bf16}"
dataset_name="${TEXT_LM_REFRESH_SMOKE_DATASET:-roneneldan/TinyStories}"
tokenizer="${TEXT_LM_REFRESH_SMOKE_TOKENIZER:-Qwen/Qwen3.5-0.8B}"
train_tokens="${TEXT_LM_REFRESH_SMOKE_TRAIN_TOKENS:-3072}"
eval_tokens="${TEXT_LM_REFRESH_SMOKE_EVAL_TOKENS:-1024}"
epochs="${TEXT_LM_REFRESH_SMOKE_EPOCHS:-1}"
steps_per_epoch="${TEXT_LM_REFRESH_SMOKE_STEPS_PER_EPOCH:-12}"
batch_size="${TEXT_LM_REFRESH_SMOKE_BATCH_SIZE:-2}"
max_seq_len="${TEXT_LM_REFRESH_SMOKE_MAX_SEQ_LEN:-128}"
embed_dim="${TEXT_LM_REFRESH_SMOKE_EMBED_DIM:-256}"
num_layers="${TEXT_LM_REFRESH_SMOKE_NUM_LAYERS:-4}"
num_heads="${TEXT_LM_REFRESH_SMOKE_NUM_HEADS:-8}"
kv_heads="${TEXT_LM_REFRESH_SMOKE_KV_HEADS:-2}"
learning_rate="${TEXT_LM_REFRESH_SMOKE_LEARNING_RATE:-1e-3}"
rank="${TEXT_LM_REFRESH_SMOKE_RANK:-8}"
interval="${TEXT_LM_REFRESH_SMOKE_INTERVAL:-8}"
window="${TEXT_LM_REFRESH_SMOKE_WINDOW:-8}"
ema_fast_decay="${TEXT_LM_REFRESH_SMOKE_EMA_FAST_DECAY:-0.96}"

mkdir -p "$output_root"
IFS=',' read -r -a mode_values <<< "$modes"

for mode in "${mode_values[@]}"; do
  mode="${mode// /}"
  [[ -z "$mode" ]] && continue

  refresh_mode="none"
  refresh_mix="smoothstep"
  ema_decay=""
  case "$mode" in
    frozen|fixed)
      mode_label="frozen"
      ;;
    hard)
      mode_label="hard"
      refresh_mode="hard"
      refresh_mix="smoothstep"
      ;;
    shadow)
      mode_label="shadow"
      refresh_mode="shadow"
      refresh_mix="smoothstep"
      ;;
    ema)
      mode_label="ema"
      refresh_mode="smooth"
      refresh_mix="ema"
      ;;
    ema_fast)
      mode_label="ema_fast"
      refresh_mode="smooth"
      refresh_mix="ema"
      ema_decay="$ema_fast_decay"
      ;;
    *)
      echo "unsupported refresh mode: $mode" >&2
      exit 2
      ;;
  esac

  output_path="$output_root/${mode_label}.json"
  command_args=(
    -m verify.text_lm_optimizer_convergence
    --dataset-name "$dataset_name"
    --tokenizer "$tokenizer"
    --device "$device"
    --dtype "$dtype"
    --seeds "$seeds"
    --optimizers "$optimizers"
    --train-tokens "$train_tokens"
    --eval-tokens "$eval_tokens"
    --epochs "$epochs"
    --steps-per-epoch "$steps_per_epoch"
    --batch-size "$batch_size"
    --max-seq-len "$max_seq_len"
    --embed-dim "$embed_dim"
    --num-layers "$num_layers"
    --num-heads "$num_heads"
    --kv-heads "$kv_heads"
    --learning-rate "$learning_rate"
    --rank "$rank"
    --refresh-mode "$refresh_mode"
    --refresh-interval "$interval"
    --refresh-window "$window"
    --refresh-mix "$refresh_mix"
  )
  if [[ -n "$ema_decay" ]]; then
    command_args+=(--refresh-ema-decay "$ema_decay")
  fi

  echo "=== TinyStories refresh smoke: ${mode_label} ==="
  PYTHONPATH=. python3 "${command_args[@]}" | tee "$output_path"
done
