#!/usr/bin/env bash
# Run a low-load APOLLO state-rank sweep on the TinyStories text task.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${APOLLO_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
ranks_csv="2,4,8,16"
seeds_csv="0,1,2"
optimizers_csv="APOLLO,APOLLO-CAME"
train_tokens=4096
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=1
steps_per_epoch=10
learning_rate="3e-4"
state_rank_interval=5
state_rank_max_elements=2000000
state_rank_max_tensors=4
state_rank_parameter=""
output_dir="output/text-lm-apollo-state-rank-sweep"
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_apollo_state_rank_sweep.sh [options]

Runs a low-load TinyStories APOLLO state-rank sweep. One JSON is written for
each rank and contains all requested seeds and optimizer variants.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --ranks CSV                     ranks (default: 2,4,8,16)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --optimizers CSV                APOLLO variants (default: APOLLO,APOLLO-CAME)
  --train-tokens N                training token budget (default: 4096)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 10)
  --learning-rate RATE             learning rate (default: 3e-4)
  --state-rank-interval N         snapshot interval (default: 5)
  --state-rank-max-elements N     SVD element limit (default: 2000000)
  --state-rank-max-tensors N      tensors per case (default: 4)
  --state-rank-parameter TEXT      parameter-name substring filter
  --output-dir DIR                sweep output directory
  --force                         overwrite existing rank outputs
  -h, --help                      show this help
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--ranks|--seeds|--optimizers|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--learning-rate|--state-rank-interval|--state-rank-max-elements|--state-rank-max-tensors|--state-rank-parameter|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --ranks) ranks_csv="$value" ;;
                --seeds) seeds_csv="$value" ;;
                --optimizers) optimizers_csv="$value" ;;
                --train-tokens) train_tokens="$value" ;;
                --eval-tokens) eval_tokens="$value" ;;
                --max-seq-len) max_seq_len="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --steps-per-epoch) steps_per_epoch="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --state-rank-interval) state_rank_interval="$value" ;;
                --state-rank-max-elements) state_rank_max_elements="$value" ;;
                --state-rank-max-tensors) state_rank_max_tensors="$value" ;;
                --state-rank-parameter) state_rank_parameter="$value" ;;
                --output-dir) output_dir="$value" ;;
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
for value in "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch" "$state_rank_interval" "$state_rank_max_elements" "$state_rank_max_tensors"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done

IFS=',' read -r -a ranks <<< "$ranks_csv"
IFS=',' read -r -a seeds <<< "$seeds_csv"
((${#ranks[@]} > 0)) || die "--ranks must not be empty"
((${#seeds[@]} > 0)) || die "--seeds must not be empty"
for rank in "${ranks[@]}"; do
    [[ "$rank" =~ ^[1-9][0-9]*$ ]] || die "invalid rank: $rank"
done
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed: $seed"
done

mkdir -p "$output_dir"
for rank in "${ranks[@]}"; do
    rank_dir="$output_dir/rank-$rank"
    result_path="$rank_dir/result.json"
    if [[ -e "$result_path" && "$force" -eq 0 ]]; then
        die "output exists (use --force): $result_path"
    fi
    mkdir -p "$rank_dir"
    args=(
        --device "$device" --dtype "$dtype"
        --optimizers "$optimizers_csv" --seeds "$seeds_csv"
        --rank "$rank" --train-tokens "$train_tokens"
        --eval-tokens "$eval_tokens" --max-seq-len "$max_seq_len"
        --batch-size "$batch_size" --epochs "$epochs"
        --steps-per-epoch "$steps_per_epoch"
        --learning-rate "$learning_rate"
        --record-state-rank
        --state-rank-interval "$state_rank_interval"
        --state-rank-max-elements "$state_rank_max_elements"
        --state-rank-max-tensors "$state_rank_max_tensors"
    )
    if [[ -n "$state_rank_parameter" ]]; then
        args+=(--state-rank-parameter "$state_rank_parameter")
    fi
    printf '==> rank=%s seeds=%s optimizers=%s\n' "$rank" "$seeds_csv" "$optimizers_csv"
    "$python_bin" -m verify.text_lm_optimizer_convergence "${args[@]}" \
        > "$result_path"
done

printf 'sweep output: %s\n' "$output_dir"
