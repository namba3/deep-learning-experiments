#!/usr/bin/env bash
# Compare causal and optional rolling trajectory PCA diagnostics.

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
calibration_fractions="0.5,0.75"
rolling_window=0
output_dir="output/text-lm-causal-trajectory-pca-sweep"
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_causal_trajectory_pca_sweep.sh [options]

Runs causal trajectory PCA with multiple calibration fractions. If
--rolling-window is set, each output also includes a rolling PCA score using
the preceding snapshot window. This is a diagnostic sweep, not a speed
benchmark.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --optimizers CSV                optimizer names
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --rank N                        LRSF/APOLLO rank (default: 8)
  --train-tokens N                training token budget (default: 65536)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 100)
  --eval-interval N               periodic validation interval (default: 0)
  --learning-rate RATE             learning rate (default: 5e-3)
  --interval N                    trajectory snapshot interval (default: 5)
  --max-snapshots N               maximum snapshots per source (default: 24)
  --max-elements N                diagnostic element limit (default: 2000000)
  --max-tensors N                 parameter tensors per case (default: 4)
  --calibration-fractions CSV     fractions, e.g. 0.5,0.75
  --rolling-window N              preceding snapshots for rolling PCA (default: 0)
  --output-dir DIR                output directory
  --force                         overwrite existing cell outputs
  -h, --help                      show this help
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--optimizers|--seeds|--rank|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--eval-interval|--learning-rate|--interval|--max-snapshots|--max-elements|--max-tensors|--calibration-fractions|--rolling-window|--output-dir)
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
                --calibration-fractions) calibration_fractions="$value" ;;
                --rolling-window) rolling_window="$value" ;;
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
for value in "$rank" "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch" "$interval" "$max_snapshots" "$max_elements" "$max_tensors"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done
[[ "$eval_interval" =~ ^[0-9]+$ ]] || die "non-negative integer expected: $eval_interval"
[[ "$rolling_window" =~ ^[0-9]+$ ]] || die "non-negative integer expected: $rolling_window"
(( max_snapshots >= 4 )) || die "--max-snapshots must be at least 4"
(( epochs * steps_per_epoch >= 4 * interval )) || die "training must provide at least four trajectory snapshots"
if (( rolling_window > 0 )); then
    (( rolling_window >= 2 )) || die "--rolling-window must be at least 2 when enabled"
    (( max_snapshots > rolling_window )) || die "--max-snapshots must exceed --rolling-window"
fi

mkdir -p "$output_dir"
IFS=',' read -r -a fractions <<< "$calibration_fractions"
(( ${#fractions[@]} > 0 )) || die "at least one calibration fraction is required"

for fraction in "${fractions[@]}"; do
    [[ "$fraction" =~ ^0\.[0-9]+$ ]] || die "calibration fraction must be between 0 and 1: $fraction"
    awk "BEGIN { if ($fraction <= 0 || $fraction >= 1) exit 1 }" || die "calibration fraction must be between 0 and 1: $fraction"
    fraction_tag="${fraction/./p}"
    output_path="$output_dir/calibration-${fraction_tag}.json"
    if [[ -e "$output_path" && "$force" -eq 0 ]]; then
        printf 'skip existing: %s\n' "$output_path"
        continue
    fi
    args=(
        --device "$device" --dtype "$dtype"
        --optimizers "$optimizers" --seeds "$seeds" --rank "$rank"
        --train-tokens "$train_tokens" --eval-tokens "$eval_tokens"
        --max-seq-len "$max_seq_len" --batch-size "$batch_size"
        --epochs "$epochs" --steps-per-epoch "$steps_per_epoch"
        --eval-interval "$eval_interval" --learning-rate "$learning_rate"
        --record-update-trajectory-pca
        --state-rank-interval "$interval"
        --state-trajectory-max-snapshots "$max_snapshots"
        --state-rank-max-elements "$max_elements"
        --state-rank-max-tensors "$max_tensors"
        --trajectory-pca-calibration-fraction "$fraction"
        --trajectory-pca-rolling-window "$rolling_window"
    )
    "$python_bin" -m verify.text_lm_optimizer_convergence "${args[@]}" > "$output_path"
    printf 'completed: fraction=%s output=%s\n' "$fraction" "$output_path"
done

printf 'causal trajectory PCA sweep outputs: %s\n' "$output_dir"
