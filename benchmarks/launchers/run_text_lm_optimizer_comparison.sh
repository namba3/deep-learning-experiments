#!/usr/bin/env bash
set -euo pipefail

# Compare AdamW, AdamW-SF, AdamW-LRSF, and CAME on TinyStories.
# The verifier writes one JSON object containing all seeds and optimizer cases.

output_path="${TEXT_LM_OPTIMIZER_OUTPUT:-output/text-lm-optimizer-comparison.json}"
dataset_name="${TEXT_LM_DATASET:-roneneldan/TinyStories}"
tokenizer="${TEXT_LM_TOKENIZER:-Qwen/Qwen3.5-0.8B}"
device="${TEXT_LM_DEVICE:-cuda}"
dtype="${TEXT_LM_DTYPE:-bf16}"
seeds="${TEXT_LM_SEEDS:-0,1,2}"
optimizers="${TEXT_LM_OPTIMIZERS:-AdamW,AdamW-SF,AdamW-LRSF,CAME}"
adamw_sf_backend="${TEXT_LM_ADAMW_SF_BACKEND:-torch}"
train_tokens="${TEXT_LM_TRAIN_TOKENS:-1000000}"
eval_tokens="${TEXT_LM_EVAL_TOKENS:-100000}"
epochs="${TEXT_LM_EPOCHS:-3}"
steps_per_epoch="${TEXT_LM_STEPS_PER_EPOCH:-100}"
batch_size="${TEXT_LM_BATCH_SIZE:-4}"
max_seq_len="${TEXT_LM_MAX_SEQ_LEN:-128}"
learning_rate="${TEXT_LM_LEARNING_RATE:-3e-4}"
rank="${TEXT_LM_RANK:-4}"
refresh_mode="${TEXT_LM_REFRESH_MODE:-hard}"
refresh_interval="${TEXT_LM_REFRESH_INTERVAL:-200}"
refresh_window="${TEXT_LM_REFRESH_WINDOW:-200}"
refresh_mix="${TEXT_LM_REFRESH_MIX:-smoothstep}"
refresh_ema_decay="${TEXT_LM_REFRESH_EMA_DECAY:-}"
record_refresh_diagnostics="${TEXT_LM_RECORD_REFRESH_DIAGNOSTICS:-0}"
refresh_transport_overlap="${TEXT_LM_REFRESH_TRANSPORT_OVERLAP:-}"
orthogonal_rate="${TEXT_LM_ORTHOGONAL_RATE:-0.0}"
orthogonal_direction="${TEXT_LM_ORTHOGONAL_DIRECTION:-random}"
orthogonal_signal="${TEXT_LM_ORTHOGONAL_SIGNAL:-gradient}"
apollo_norm_growth_rate="${TEXT_LM_APOLLO_NORM_GROWTH_RATE:-1.01}"
apollo_disable_norm_growth_limiter="${TEXT_LM_APOLLO_DISABLE_NORM_GROWTH_LIMITER:-0}"

output_dir="${output_path%/*}"
if [[ "$output_dir" == "$output_path" ]]; then
  output_dir="."
fi
mkdir -p "$output_dir"

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
  --learning-rate "$learning_rate"
  --rank "$rank"
  --adamw-sf-backend "$adamw_sf_backend"
  --refresh-mode "$refresh_mode"
  --refresh-interval "$refresh_interval"
  --refresh-window "$refresh_window"
  --refresh-mix "$refresh_mix"
  --orthogonal-rate "$orthogonal_rate"
  --orthogonal-direction "$orthogonal_direction"
  --orthogonal-signal "$orthogonal_signal"
  --apollo-norm-growth-rate "$apollo_norm_growth_rate"
)

if [[ -n "$refresh_ema_decay" ]]; then
  command_args+=(--refresh-ema-decay "$refresh_ema_decay")
fi

if [[ "$record_refresh_diagnostics" == "1" ]]; then
  command_args+=(--record-refresh-diagnostics)
fi

if [[ -n "$refresh_transport_overlap" ]]; then
  command_args+=(--refresh-transport-overlap "$refresh_transport_overlap")
fi

if [[ "$apollo_disable_norm_growth_limiter" == "1" ]]; then
  command_args+=(--apollo-disable-norm-growth-limiter)
fi

PYTHONPATH=. python3 "${command_args[@]}" | tee "$output_path"
