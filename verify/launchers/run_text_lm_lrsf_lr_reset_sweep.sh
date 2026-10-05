#!/usr/bin/env bash
# Sweep rank and refresh interval for AdamW-LRSF-LR reset/transport.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="${TEXT_LM_DEVICE:-cuda}"
dtype="${TEXT_LM_DTYPE:-bf16}"
seeds="${TEXT_LM_SEEDS:-0,1,2}"
ranks="${TEXT_LM_LRSF_LR_RANKS:-8,16}"
intervals="${TEXT_LM_LRSF_LR_INTERVALS:-25,50}"
train_tokens="${TEXT_LM_TRAIN_TOKENS:-153600}"
eval_tokens="${TEXT_LM_EVAL_TOKENS:-1024}"
epochs="${TEXT_LM_EPOCHS:-3}"
steps_per_epoch="${TEXT_LM_STEPS_PER_EPOCH:-100}"
learning_rate="${TEXT_LM_LEARNING_RATE:-3e-4}"
transport_overlap="${TEXT_LM_TRANSPORT_OVERLAP:-0.99}"
output_dir="${TEXT_LM_LRSF_LR_SWEEP_DIR:-output/text-lm-lrsf-lr-reset-sweep}"
force="${TEXT_LM_FORCE:-0}"

case "$device" in auto|cpu|cuda) ;; *) printf 'invalid device: %s\n' "$device" >&2; exit 2 ;; esac
case "$dtype" in fp32|bf16) ;; *) printf 'invalid dtype: %s\n' "$dtype" >&2; exit 2 ;; esac
[[ "$transport_overlap" =~ ^(0([.][0-9]+)?|1([.][0]+)?)$ ]] || { printf 'invalid overlap: %s\n' "$transport_overlap" >&2; exit 2; }

IFS=',' read -r -a rank_values <<< "$ranks"
IFS=',' read -r -a interval_values <<< "$intervals"
mkdir -p "$output_dir"

for rank in "${rank_values[@]}"; do
    [[ "$rank" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid rank: %s\n' "$rank" >&2; exit 2; }
    for interval in "${interval_values[@]}"; do
        [[ "$interval" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid interval: %s\n' "$interval" >&2; exit 2; }
        cell_dir="$output_dir/rank-$rank-interval-$interval"
        mkdir -p "$cell_dir"
        for policy in reset transport; do
            result_path="$cell_dir/$policy.json"
            if [[ -e "$result_path" && "$force" != 1 ]]; then
                printf 'skip existing: %s\n' "$result_path"
                continue
            fi
            printf '==> rank=%s interval=%s policy=%s seeds=%s\n' "$rank" "$interval" "$policy" "$seeds"
            "$python_bin" -m verify.text_lm_optimizer_convergence \
                --device "$device" --dtype "$dtype" \
                --optimizers AdamW-LRSF-LR --seeds "$seeds" --rank "$rank" \
                --train-tokens "$train_tokens" --eval-tokens "$eval_tokens" \
                --epochs "$epochs" --steps-per-epoch "$steps_per_epoch" \
                --learning-rate "$learning_rate" --refresh-mode hard \
                --refresh-interval "$interval" --refresh-window "$interval" \
                --refresh-mix smoothstep --lrsf-refresh-state "$policy" \
                --refresh-transport-overlap "$transport_overlap" > "$result_path"
        done
    done
done

printf 'sweep outputs: %s/rank-*/interval-*/{reset,transport}.json\n' "$output_dir"
