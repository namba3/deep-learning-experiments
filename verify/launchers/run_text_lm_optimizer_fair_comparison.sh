#!/usr/bin/env bash
# Compare AdamW-family and APOLLO-family candidates with separate tuned LRs.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
seeds_csv="0,1,2"
rank=8
train_tokens=51200
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=1
steps_per_epoch=100
adamw_learning_rate="3e-4"
apollo_learning_rate="3e-3"
apollo_scale="1.0"
confidence_beta="0.99"
confidence_alpha="1e-3"
lrsf_refresh_mode="hard"
lrsf_refresh_interval=50
lrsf_refresh_mix="smoothstep"
lrsf_transport_overlap=""
output_dir="output/text-lm-optimizer-fair-comparison"
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_optimizer_fair_comparison.sh [options]

Runs diagnostic-free AdamW-SF/AdamW-LRSF and APOLLO/APOLLO-Conf groups with
separate learning rates. The seed, model, and token budgets are shared.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --rank N                        low-rank projection rank (default: 8)
  --train-tokens N                training token budget (default: 51200)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 100)
  --adamw-learning-rate RATE      AdamW-SF/LRSF learning rate (default: 3e-4)
  --apollo-learning-rate RATE     APOLLO learning rate (default: 3e-3)
  --apollo-scale RATE             APOLLO scale (default: 1.0)
  --confidence-beta RATE          APOLLO-Conf innovation EMA decay (default: 0.99)
  --confidence-alpha RATE         APOLLO-Conf mean-square floor (default: 1e-3)
  --lrsf-refresh-mode MODE        LRSF refresh mode (default: hard)
  --lrsf-refresh-interval N       LRSF refresh interval (default: 50)
  --lrsf-refresh-mix MIX          LRSF refresh mix (default: smoothstep)
  --lrsf-transport-overlap RATE   LRSF transport overlap (default: unset)
  --output-dir DIR                comparison output directory
  --force                         overwrite existing group outputs
  -h, --help                      show this help
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--seeds|--rank|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--adamw-learning-rate|--apollo-learning-rate|--apollo-scale|--confidence-beta|--confidence-alpha|--lrsf-refresh-mode|--lrsf-refresh-interval|--lrsf-refresh-mix|--lrsf-transport-overlap|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --seeds) seeds_csv="$value" ;;
                --rank) rank="$value" ;;
                --train-tokens) train_tokens="$value" ;;
                --eval-tokens) eval_tokens="$value" ;;
                --max-seq-len) max_seq_len="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --steps-per-epoch) steps_per_epoch="$value" ;;
                --adamw-learning-rate) adamw_learning_rate="$value" ;;
                --apollo-learning-rate) apollo_learning_rate="$value" ;;
                --apollo-scale) apollo_scale="$value" ;;
                --confidence-beta) confidence_beta="$value" ;;
                --confidence-alpha) confidence_alpha="$value" ;;
                --lrsf-refresh-mode) lrsf_refresh_mode="$value" ;;
                --lrsf-refresh-interval) lrsf_refresh_interval="$value" ;;
                --lrsf-refresh-mix) lrsf_refresh_mix="$value" ;;
                --lrsf-transport-overlap) lrsf_transport_overlap="$value" ;;
                --output-dir) output_dir="$value" ;;
            esac
            shift 2
            ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "invalid --device: $device" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "invalid --dtype: $dtype" ;; esac
case "$lrsf_refresh_mode" in none|hard|smooth|shadow) ;; *) die "invalid --lrsf-refresh-mode: $lrsf_refresh_mode" ;; esac
case "$lrsf_refresh_mix" in linear|smoothstep|stochastic|ema) ;; *) die "invalid --lrsf-refresh-mix: $lrsf_refresh_mix" ;; esac
for value in "$rank" "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done
[[ "$lrsf_refresh_interval" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $lrsf_refresh_interval"
if [[ -n "$lrsf_transport_overlap" ]]; then
    [[ "$lrsf_transport_overlap" =~ ^0?(\.[0-9]+)?$|^1(\.0+)?$ ]] || die "invalid --lrsf-transport-overlap: $lrsf_transport_overlap"
fi

mkdir -p "$output_dir"
run_group() {
    local group="$1"
    local learning_rate="$2"
    local optimizer_names="$3"
    local group_dir="$output_dir/group-$group"
    local result_path="$group_dir/result.json"
    if [[ -e "$result_path" && "$force" -eq 0 ]]; then
        printf 'skip existing: %s\n' "$result_path"
        return
    fi
    mkdir -p "$group_dir"
    printf '==> group=%s optimizers=%s seeds=%s\n' "$group" "$optimizer_names" "$seeds_csv"
    args=(
        --device "$device" --dtype "$dtype" --optimizers "$optimizer_names"
        --seeds "$seeds_csv" --rank "$rank" --train-tokens "$train_tokens"
        --eval-tokens "$eval_tokens" --max-seq-len "$max_seq_len"
        --batch-size "$batch_size" --epochs "$epochs"
        --steps-per-epoch "$steps_per_epoch" --learning-rate "$learning_rate"
    )
    if [[ "$group" == adamw-* ]]; then
        args+=(
            --refresh-mode "$lrsf_refresh_mode"
            --refresh-interval "$lrsf_refresh_interval"
            --refresh-window "$lrsf_refresh_interval"
            --refresh-mix "$lrsf_refresh_mix"
        )
        if [[ -n "$lrsf_transport_overlap" ]]; then
            args+=(--refresh-transport-overlap "$lrsf_transport_overlap")
        fi
    fi
    if [[ "$group" == apollo ]]; then
        args+=(
            --apollo-scale "$apollo_scale"
            --lr-ema-confidence-beta "$confidence_beta"
            --lr-ema-confidence-alpha "$confidence_alpha"
            --apollo-disable-norm-growth-limiter
        )
    fi
    "$python_bin" -m verify.text_lm_optimizer_convergence "${args[@]}" > "$result_path"
}

run_group "adamw-lr-$(printf '%s' "$adamw_learning_rate" | tr '.-' 'p_')" "$adamw_learning_rate" "AdamW-SF,AdamW-LRSF"
run_group "apollo" "$apollo_learning_rate" "APOLLO,APOLLO-Conf"

report_tmp="$output_dir/.optimizer-comparison-report.md.tmp.$$"
trap 'rm -f -- "$report_tmp"' EXIT INT TERM
PYTHONPATH="$repo_root" "$python_bin" -m verify.text_lm_optimizer_comparison_report \
    "$output_dir" > "$report_tmp"
mv -- "$report_tmp" "$output_dir/optimizer-comparison-report.md"
trap - EXIT INT TERM
printf 'comparison report: %s\n' "$output_dir/optimizer-comparison-report.md"
