#!/usr/bin/env bash
set -euo pipefail

# Compare text-lm architectures on identical streamed pretraining budgets.
# The default dataset is intentionally lightweight; add OpenWebText with:
# TEXT_LM_DATASETS=roneneldan/TinyStories,Skylion007/openwebtext

output_dir="${TEXT_LM_OUTPUT_DIR:-output/text-lm-pretraining-comparison}"
dataset_list="${TEXT_LM_DATASETS:-roneneldan/TinyStories}"
architecture_list="${TEXT_LM_ARCHITECTURES:-naive,mhla3-gqa,looped,looped-hybrid,mhla3-gqa-looped-hybrid}"
seed_list="${TEXT_LM_SEEDS:-0,1,2}"
device="${TEXT_LM_DEVICE:-auto}"
tokenizer="${TEXT_LM_TOKENIZER:-Qwen/Qwen3.5-0.8B}"
train_tokens="${TEXT_LM_TRAIN_TOKENS:-1000000}"
eval_tokens="${TEXT_LM_EVAL_TOKENS:-100000}"
epochs="${TEXT_LM_EPOCHS:-1}"
steps_per_epoch="${TEXT_LM_STEPS_PER_EPOCH:-0}"
eval_max_batches="${TEXT_LM_EVAL_MAX_BATCHES:-100}"
batch_size="${TEXT_LM_BATCH_SIZE:-4}"
max_seq_len_list="${TEXT_LM_MAX_SEQ_LENS:-${TEXT_LM_MAX_SEQ_LEN:-128}}"
embed_dim="${TEXT_LM_EMBED_DIM:-512}"
num_layers="${TEXT_LM_NUM_LAYERS:-16}"
num_heads="${TEXT_LM_NUM_HEADS:-8}"
kv_heads="${TEXT_LM_KV_HEADS:-2}"
num_workers="${TEXT_LM_NUM_WORKERS:-2}"
optimizer="${TEXT_LM_OPTIMIZER:-AdamW}"
lr_scheduler="${TEXT_LM_LR_SCHEDULER:-constant}"
warmup_steps="${TEXT_LM_WARMUP_STEPS:-0}"
bf16="${TEXT_LM_BF16:-1}"
dry_run="${TEXT_LM_DRY_RUN:-0}"
distill_mode="${TEXT_LM_DISTILL_MODE:-none}"
teacher_model="${TEXT_LM_TEACHER_MODEL:-Qwen/Qwen3.5-0.8B}"
distill_temperature="${TEXT_LM_DISTILL_TEMPERATURE:-2.0}"
distill_alpha="${TEXT_LM_DISTILL_ALPHA:-0.5}"

IFS=',' read -r -a datasets <<< "$dataset_list"
IFS=',' read -r -a architectures <<< "$architecture_list"
IFS=',' read -r -a seeds <<< "$seed_list"
IFS=',' read -r -a max_seq_lens <<< "$max_seq_len_list"

mkdir -p "$output_dir"

for dataset in "${datasets[@]}"; do
  dataset="${dataset// /}"
  dataset_slug="${dataset//\//_}"
  for max_seq_len in "${max_seq_lens[@]}"; do
    max_seq_len="${max_seq_len// /}"
    for seed in "${seeds[@]}"; do
      for architecture in "${architectures[@]}"; do
        architecture="${architecture// /}"
        run_variant="$architecture"
        if [[ "$distill_mode" != "none" ]]; then
          run_variant="${run_variant}-distill-${distill_mode}"
        fi
        run_name="${dataset_slug}-ctx-${max_seq_len}-${run_variant}-seed-${seed}"
        log_path="$output_dir/${run_name}.log"
        echo "=== ${run_name} ==="
        command_args=(
          --data-mode text
          --dataset-name "$dataset"
          --tokenizer "$tokenizer"
          --distill-mode "$distill_mode"
          --teacher-model "$teacher_model"
          --distill-temperature "$distill_temperature"
          --distill-alpha "$distill_alpha"
          --architecture "$architecture"
          --device "$device"
          --max-train-tokens "$train_tokens"
          --max-eval-tokens "$eval_tokens"
          --epochs "$epochs"
          --steps-per-epoch "$steps_per_epoch"
          --eval-max-batches "$eval_max_batches"
          --batch-size "$batch_size"
          --max-seq-len "$max_seq_len"
          --embed-dim "$embed_dim"
          --num-layers "$num_layers"
          --num-heads "$num_heads"
          --kv-heads "$kv_heads"
          --num-workers "$num_workers"
          --optimizer "$optimizer"
          --lr-scheduler "$lr_scheduler"
          --warmup-steps "$warmup_steps"
          --output-dir "$output_dir"
          --run-name "$run_name"
          --save-mode final
          --seed "$seed"
        )
        if [[ "$bf16" == "1" ]]; then
          command_args+=(--bf16)
        fi
        if [[ "$dry_run" == "1" ]]; then
          command_args+=(--dry-run)
        fi
        PYTHONPATH=. python3 -m text_lm.train "${command_args[@]}" \
          2>&1 | tee "$log_path"
      done
    done
  done
done

PYTHONPATH=. python3 -m verify.text_lm_pretraining_report \
  --input-dir "$output_dir" \
  --output "$output_dir/report.json"
