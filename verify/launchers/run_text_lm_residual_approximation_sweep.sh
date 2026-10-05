#!/usr/bin/env bash
# Run the rolling trajectory residual approximation diagnostic and report it.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
optimizers="AdamW,AdamW-SF,AdamW-LRSF"
seeds="0,1,2"
rank=8
train_tokens=65536
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=1
steps_per_epoch=100
eval_interval=0
learning_rate="5e-3"
interval=5
max_snapshots=24
max_elements=2000000
max_tensors=4
rolling_window=9
residual_compression_block_size=256
residual_compression_scale_mode="max_abs"
record_fixed_basis=0
output_dir="output/text-lm-residual-approximation-sweep"
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_residual_approximation_sweep.sh [options]

Runs a diagnostic-only rolling trajectory PCA probe and creates a residual
approximation report. It does not change optimizer updates or optimizer state.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --optimizers CSV                optimizer names
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --rank N                        LRSF rank (default: 8)
  --train-tokens N                training token budget (default: 65536)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 100)
  --eval-interval N               periodic validation interval (default: 0)
  --learning-rate RATE             learning rate (default: 5e-3)
  --interval N                    snapshot interval (default: 5)
  --max-snapshots N               snapshots per source (default: 24)
  --max-elements N                diagnostic element limit (default: 2000000)
  --max-tensors N                 parameter tensors per case (default: 4)
  --rolling-window N              preceding snapshots (default: 9)
  --block-size N                  residual quantization block size (default: 256)
  --scale-mode MODE               max_abs, percentile_99_9, or rms_3sigma (default: max_abs)
  --record-fixed-basis             also diagnose a fixed initial PCA basis
  --output-dir DIR                output directory
  --force                         overwrite existing outputs
  -h, --help                      show this help
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--optimizers|--seeds|--rank|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--eval-interval|--learning-rate|--interval|--max-snapshots|--max-elements|--max-tensors|--rolling-window|--block-size|--scale-mode|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --optimizers) optimizers="$value" ;;
                --seeds) seeds="$value" ;;
                --rank) rank="$value" ;;
                --train-tokens) train_tokens="$value" ;;
                --eval-tokens) eval_tokens="$value" ;;
                --max-seq-len) max_seq_len="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --steps-per-epoch) steps_per_epoch="$value" ;;
                --eval-interval) eval_interval="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --interval) interval="$value" ;;
                --max-snapshots) max_snapshots="$value" ;;
                --max-elements) max_elements="$value" ;;
                --max-tensors) max_tensors="$value" ;;
                --rolling-window) rolling_window="$value" ;;
                --block-size) residual_compression_block_size="$value" ;;
                --scale-mode) residual_compression_scale_mode="$value" ;;
                --output-dir) output_dir="$value" ;;
            esac
            shift 2
            ;;
        --force) force=1; shift ;;
        --record-fixed-basis) record_fixed_basis=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "invalid --device: $device" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "invalid --dtype: $dtype" ;; esac
case "$residual_compression_scale_mode" in max_abs|percentile_99_9|rms_3sigma) ;; *) die "invalid --scale-mode: $residual_compression_scale_mode" ;; esac
for value in "$rank" "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch" "$interval" "$max_snapshots" "$max_elements" "$max_tensors" "$rolling_window" "$residual_compression_block_size"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done
[[ "$eval_interval" =~ ^[0-9]+$ ]] || die "non-negative integer expected: $eval_interval"
(( max_snapshots > rolling_window )) || die "max-snapshots must exceed rolling-window"
(( epochs * steps_per_epoch >= (rolling_window + 1) * interval )) || die "training must provide a prediction after rolling-window"

mkdir -p "$output_dir"
result_path="$output_dir/results.json"
report_path="$output_dir/report.md"
extra_args=()
if (( record_fixed_basis )); then
    extra_args+=(--record-fixed-basis-residual-approximation)
fi
if [[ -e "$result_path" && "$force" -eq 0 ]]; then
    printf 'skip existing: %s\n' "$result_path"
else
    "$python_bin" -m verify.text_lm_optimizer_convergence \
        --device "$device" --dtype "$dtype" \
        --optimizers "$optimizers" --seeds "$seeds" --rank "$rank" \
        --train-tokens "$train_tokens" --eval-tokens "$eval_tokens" \
        --max-seq-len "$max_seq_len" --batch-size "$batch_size" \
        --epochs "$epochs" --steps-per-epoch "$steps_per_epoch" \
        --eval-interval "$eval_interval" --learning-rate "$learning_rate" \
        --record-update-trajectory-pca \
        --state-rank-interval "$interval" \
        --state-trajectory-max-snapshots "$max_snapshots" \
        --state-rank-max-elements "$max_elements" \
        --state-rank-max-tensors "$max_tensors" \
        --trajectory-pca-calibration-fraction 0.5 \
        --trajectory-pca-rolling-window "$rolling_window" \
        --residual-compression-block-size "$residual_compression_block_size" \
        --residual-compression-scale-mode "$residual_compression_scale_mode" \
        "${extra_args[@]}" \
        > "$result_path"
    printf 'completed: %s\n' "$result_path"
fi

"$python_bin" -m verify.text_lm_residual_approximation_report \
    "$result_path" > "$report_path"
printf 'completed: %s\n' "$report_path"
