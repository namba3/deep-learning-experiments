#!/usr/bin/env bash
# Run a multi-seed temporal-PCA probe for full optimizer states.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
optimizers="AdamW-SF,AdamW-LRSF"
seeds="0,1,2"
train_tokens=32768
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=1
steps_per_epoch=50
learning_rate="3e-4"
interval=5
max_snapshots=16
max_elements=2000000
max_tensors=4
output_path="output/text-lm-state-trajectory-pca.json"

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_state_trajectory_pca.sh [options]

Runs a multi-seed temporal-PCA probe for full-rank optimizer states on the
TinyStories text task. APOLLO matrix latent states are excluded by design.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --optimizers CSV                optimizer names (default: AdamW-SF,AdamW-LRSF)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --train-tokens N                training token budget (default: 32768)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 50)
  --learning-rate RATE             learning rate (default: 3e-4)
  --interval N                    steps between snapshots (default: 5)
  --max-snapshots N               maximum snapshots per source (default: 16)
  --max-elements N                SVD/PCA element limit (default: 2000000)
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

force=0
while (($# > 0)); do
    case "$1" in
        --device|--dtype|--optimizers|--seeds|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--learning-rate|--interval|--max-snapshots|--max-elements|--max-tensors|--output)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --optimizers) optimizers="$value" ;;
                --seeds) seeds="$value" ;;
                --train-tokens) train_tokens="$value" ;;
                --eval-tokens) eval_tokens="$value" ;;
                --max-seq-len) max_seq_len="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --steps-per-epoch) steps_per_epoch="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --interval) interval="$value" ;;
                --max-snapshots) max_snapshots="$value" ;;
                --max-elements) max_elements="$value" ;;
                --max-tensors) max_tensors="$value" ;;
                --output) output_path="$value" ;;
            esac
            shift 2
            ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "--device must be auto, cpu, or cuda" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "--dtype must be fp32 or bf16" ;; esac
for value in "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch" "$interval" "$max_snapshots" "$max_elements" "$max_tensors"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done

[[ "$output_path" == *.json ]] || die "--output must end with .json"
if [[ -e "$output_path" && "$force" -eq 0 ]]; then
    die "output exists (use --force): $output_path"
fi
output_parent="${output_path%/*}"
if [[ "$output_parent" != "$output_path" ]]; then
    mkdir -p "$output_parent"
fi

"$python_bin" -m verify.text_lm_optimizer_convergence \
    --device "$device" --dtype "$dtype" \
    --optimizers "$optimizers" --seeds "$seeds" \
    --train-tokens "$train_tokens" --eval-tokens "$eval_tokens" \
    --max-seq-len "$max_seq_len" --batch-size "$batch_size" \
    --epochs "$epochs" --steps-per-epoch "$steps_per_epoch" \
    --learning-rate "$learning_rate" \
    --record-state-trajectory-pca \
    --state-rank-interval "$interval" \
    --state-trajectory-max-snapshots "$max_snapshots" \
    --state-rank-max-elements "$max_elements" \
    --state-rank-max-tensors "$max_tensors" \
    > "$output_path"

printf 'trajectory PCA output: %s\n' "$output_path"
