#!/usr/bin/env bash
# Compare the APOLLO candidate and AdamW-SF with trajectory diagnostics.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
optimizers="AdamW-SF,APOLLO,APOLLO-Conf"
seeds="0,1,2"
rank=4
train_tokens=153600
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=3
steps_per_epoch=100
eval_interval=25
learning_rate="5e-3"
apollo_scale="0.75"
interval=5
max_snapshots=64
max_elements=2000000
max_tensors=4
output_path="output/text-lm-optimizer-trajectory-diagnostics.json"
force=0
disable_limiter=1

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_optimizer_trajectory_diagnostics.sh [options]

Runs the current rank-4 APOLLO candidate and AdamW-SF with sampled update
curvature and per-step training-loss second-difference diagnostics. This is
not a speed benchmark because the diagnostic copies add overhead.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --optimizers CSV                optimizer names (default: AdamW-SF,APOLLO,APOLLO-Conf)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --rank N                        APOLLO rank (default: 4)
  --train-tokens N                training token budget (default: 153600)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 3)
  --steps-per-epoch N             optimizer steps per epoch (default: 100)
  --eval-interval N               validation steps for loss curvature (default: 25)
  --learning-rate RATE             learning rate (default: 5e-3)
  --apollo-scale RATE             APOLLO projection scale (default: 0.75)
  --enable-norm-growth-limiter    enable APOLLO norm-growth limiter
  --interval N                    steps between update snapshots (default: 5)
  --max-snapshots N               maximum snapshots per source (default: 64)
  --max-elements N                diagnostic element limit (default: 2000000)
  --max-tensors N                 parameter tensors per case (default: 4)
  --output PATH                   JSON output path
  --force                         overwrite existing output
  -h, --help                      show this help
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--optimizers|--seeds|--rank|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--eval-interval|--learning-rate|--apollo-scale|--interval|--max-snapshots|--max-elements|--max-tensors|--output)
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
                --apollo-scale) apollo_scale="$value" ;;
                --interval) interval="$value" ;;
                --max-snapshots) max_snapshots="$value" ;;
                --max-elements) max_elements="$value" ;;
                --max-tensors) max_tensors="$value" ;;
                --output) output_path="$value" ;;
            esac
            shift 2
            ;;
        --enable-norm-growth-limiter) disable_limiter=0; shift ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "invalid --device" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "invalid --dtype" ;; esac
for value in "$rank" "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch" "$eval_interval" "$interval" "$max_snapshots" "$max_elements" "$max_tensors"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done
((eval_interval > 0 && epochs * steps_per_epoch >= 3 * eval_interval)) || die "training must provide at least three validation snapshots"
((max_snapshots >= 3)) || die "--max-snapshots must be at least 3"
[[ "$output_path" == *.json ]] || die "--output must end with .json"
if [[ -e "$output_path" && "$force" -eq 0 ]]; then
    die "output exists (use --force): $output_path"
fi
output_parent="${output_path%/*}"
if [[ "$output_parent" != "$output_path" ]]; then
    mkdir -p "$output_parent"
fi

args=(
    --device "$device" --dtype "$dtype"
    --optimizers "$optimizers" --seeds "$seeds" --rank "$rank"
    --train-tokens "$train_tokens" --eval-tokens "$eval_tokens"
    --max-seq-len "$max_seq_len" --batch-size "$batch_size"
    --epochs "$epochs" --steps-per-epoch "$steps_per_epoch"
    --eval-interval "$eval_interval"
    --learning-rate "$learning_rate" --apollo-scale "$apollo_scale"
    --record-trajectory-curvature
    --state-rank-interval "$interval"
    --state-trajectory-max-snapshots "$max_snapshots"
    --state-rank-max-elements "$max_elements"
    --state-rank-max-tensors "$max_tensors"
)
if [[ "$disable_limiter" -eq 1 ]]; then
    args+=(--apollo-disable-norm-growth-limiter)
fi

"$python_bin" -m verify.text_lm_optimizer_convergence "${args[@]}" > "$output_path"
printf 'trajectory diagnostics output: %s\n' "$output_path"
