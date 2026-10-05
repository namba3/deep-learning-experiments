#!/usr/bin/env bash
# Compare reset and transport speed for AdamW-LRSF-LR without diagnostics.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
seeds="0,1,2"
rank=16
train_tokens=153600
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=3
steps_per_epoch=100
learning_rate="3e-4"
refresh_interval=50
transport_overlap="0.99"
output_dir="output/text-lm-lrsf-lr-speed-comparison"
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_lrsf_lr_speed_comparison.sh [options]

Runs a diagnostic-free reset/transport comparison for AdamW-LRSF-LR.
The two policies share model, data, seed, token budget, and refresh settings.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --rank N                        low-rank projection rank (default: 16)
  --train-tokens N                training token budget (default: 153600)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 3)
  --steps-per-epoch N             optimizer steps per epoch (default: 100)
  --learning-rate RATE            AdamW-LRSF-LR learning rate (default: 3e-4)
  --refresh-interval N            hard refresh interval (default: 50)
  --transport-overlap RATE        basis overlap (default: 0.99)
  --output-dir DIR                output directory
  --force                         overwrite existing policy outputs
  -h, --help                      show this help
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--seeds|--rank|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--learning-rate|--refresh-interval|--transport-overlap|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --seeds) seeds="$value" ;;
                --rank) rank="$value" ;;
                --train-tokens) train_tokens="$value" ;;
                --eval-tokens) eval_tokens="$value" ;;
                --max-seq-len) max_seq_len="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --steps-per-epoch) steps_per_epoch="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --refresh-interval) refresh_interval="$value" ;;
                --transport-overlap) transport_overlap="$value" ;;
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
for value in "$rank" "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch" "$refresh_interval"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done
[[ "$transport_overlap" =~ ^(0([.][0-9]+)?|1([.][0]+)?)$ ]] || die "invalid --transport-overlap: $transport_overlap"

mkdir -p "$output_dir"
for policy in reset transport; do
    result_path="$output_dir/$policy.json"
    if [[ -e "$result_path" && "$force" -eq 0 ]]; then
        printf 'skip existing: %s\n' "$result_path"
        continue
    fi
    printf '==> policy=%s seeds=%s rank=%s interval=%s\n' "$policy" "$seeds" "$rank" "$refresh_interval"
    "$python_bin" -m verify.text_lm_optimizer_convergence \
        --device "$device" --dtype "$dtype" \
        --optimizers AdamW-LRSF-LR --seeds "$seeds" --rank "$rank" \
        --train-tokens "$train_tokens" --eval-tokens "$eval_tokens" \
        --max-seq-len "$max_seq_len" --batch-size "$batch_size" \
        --epochs "$epochs" --steps-per-epoch "$steps_per_epoch" \
        --learning-rate "$learning_rate" \
        --refresh-mode hard --refresh-interval "$refresh_interval" \
        --refresh-window "$refresh_interval" --refresh-mix smoothstep \
        --lrsf-refresh-state "$policy" \
        --refresh-transport-overlap "$transport_overlap" > "$result_path"
done

printf 'speed outputs: %s/{reset,transport}.json\n' "$output_dir"
