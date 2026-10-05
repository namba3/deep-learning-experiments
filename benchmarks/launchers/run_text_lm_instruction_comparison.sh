#!/usr/bin/env bash
set -euo pipefail

# Run the instruction-dataset comparison after the synthetic architecture benchmark.
# The default dataset remains tatsu-lab/alpaca for historical comparability.
# All architectures use the same dataset split, subset sizes, and seed list.
# The model is intentionally smaller than the production-sized configuration
# so that the comparison can fit on a single GPU.

compare_dir="${TEXT_LM_COMPARE_DIR:-${ALPACA_COMPARE_DIR:-output/text-lm-real-architecture}}"
seed_list="${TEXT_LM_SEEDS:-${ALPACA_SEEDS:-0,1,2}}"
train_examples="${TEXT_LM_TRAIN_EXAMPLES:-${ALPACA_TRAIN_EXAMPLES:-2048}}"
eval_examples="${TEXT_LM_EVAL_EXAMPLES:-${ALPACA_EVAL_EXAMPLES:-256}}"
epochs="${TEXT_LM_EPOCHS:-${ALPACA_EPOCHS:-3}}"
steps_per_epoch="${TEXT_LM_STEPS_PER_EPOCH:-${ALPACA_STEPS_PER_EPOCH:-100}}"
eval_batches="${TEXT_LM_EVAL_BATCHES:-${ALPACA_EVAL_BATCHES:-32}}"

IFS=',' read -r -a seeds <<< "$seed_list"
architectures=(mhla3-gqa looped-hybrid mhla3-gqa-looped-hybrid)

mkdir -p "$compare_dir"

for seed in "${seeds[@]}"; do
  for architecture in "${architectures[@]}"; do
    run_name="${architecture}-seed-${seed}"
    log_path="$compare_dir/${run_name}.log"
    echo "=== ${run_name} ==="
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m text_lm.train \
      --data-mode instruction \
      --dataset-name tatsu-lab/alpaca \
      --tokenizer Qwen/Qwen3.5-0.8B \
      --architecture "$architecture" \
      --device cuda \
      --bf16 \
      --max-seq-len 128 \
      --embed-dim 512 \
      --num-layers 16 \
      --num-heads 8 \
      --kv-heads 2 \
      --condition-dim 16 \
      --transform-rank 2 \
      --looped-prefix-layers 4 \
      --looped-blocks 2 \
      --looped-repeats 4 \
      --looped-suffix-layers 4 \
      --mhla-looped-prefix-cycles 1 \
      --mhla-looped-repeats 2 \
      --mhla-looped-suffix-cycles 1 \
      --max-train-examples "$train_examples" \
      --max-eval-examples "$eval_examples" \
      --batch-size 4 \
      --num-workers 4 \
      --epochs "$epochs" \
      --steps-per-epoch "$steps_per_epoch" \
      --eval-max-batches "$eval_batches" \
      --lr 3e-4 \
      --lr-scheduler constant \
      --no-auto-schedule \
      --warmup-steps 0 \
      --save-mode final \
      --seed "$seed" \
      --output-dir "$compare_dir" \
      --run-name "$run_name" \
      2>&1 | tee "$log_path"
  done
done

echo "Results are stored under: $compare_dir"
