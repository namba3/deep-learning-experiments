#!/usr/bin/env bash
# Run a resumable APOLLO/APOLLO-Conf learning-rate and scale sweep.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
optimizers="APOLLO,APOLLO-Conf"
seeds_csv="0,1,2"
ranks_csv="8"
train_tokens=4096
eval_tokens=512
max_seq_len=128
batch_size=4
epochs=1
steps_per_epoch=10
learning_rates_csv="3e-4,1e-3,3e-3"
scales_csv="0.5,1.0"
limiter_csv="on,off"
norm_growth_rate="1.01"
confidence_beta="0.99"
confidence_alpha="1e-3"
output_dir="output/text-lm-apollo-scale-sweep"
force=0
record_diagnostics=1

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_apollo_scale_sweep.sh [options]

Runs paired APOLLO/APOLLO-Conf cells over learning rate, APOLLO scale, and
norm-growth-limiter settings. Each cell is saved separately and existing
cells are skipped unless --force is supplied.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --optimizers CSV                optimizer names (default: APOLLO,APOLLO-Conf)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --rank N                        APOLLO rank (default: 8; alias for --ranks)
  --ranks CSV                     APOLLO ranks; multi-rank output uses rank-* subdirectories
  --train-tokens N                training token budget (default: 4096)
  --eval-tokens N                 evaluation token budget (default: 512)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 10)
  --learning-rates CSV             learning rates (default: 3e-4,1e-3,3e-3)
  --scales CSV                    APOLLO scales (default: 0.5,1.0)
  --limiters CSV                  on/off values (default: on,off)
  --norm-growth-rate RATE         APOLLO limiter growth rate (default: 1.01)
  --confidence-beta RATE          APOLLO-Conf innovation EMA decay (default: 0.99)
  --confidence-alpha RATE         APOLLO-Conf mean-square floor (default: 1e-3)
  --no-diagnostics                omit update-norm and confidence diagnostics
  --output-dir DIR                sweep output directory
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
        --device|--dtype|--optimizers|--seeds|--rank|--ranks|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--learning-rates|--scales|--limiters|--norm-growth-rate|--confidence-beta|--confidence-alpha|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --optimizers) optimizers="$value" ;;
                --seeds) seeds_csv="$value" ;;
                --rank|--ranks) ranks_csv="$value" ;;
                --train-tokens) train_tokens="$value" ;;
                --eval-tokens) eval_tokens="$value" ;;
                --max-seq-len) max_seq_len="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --steps-per-epoch) steps_per_epoch="$value" ;;
                --learning-rates) learning_rates_csv="$value" ;;
                --scales) scales_csv="$value" ;;
                --limiters) limiter_csv="$value" ;;
                --norm-growth-rate) norm_growth_rate="$value" ;;
                --confidence-beta) confidence_beta="$value" ;;
                --confidence-alpha) confidence_alpha="$value" ;;
                --output-dir) output_dir="$value" ;;
            esac
            shift 2
            ;;
        --force) force=1; shift ;;
        --no-diagnostics) record_diagnostics=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "invalid --device: $device" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "invalid --dtype: $dtype" ;; esac
for value in "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done

number_pattern='^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
IFS=',' read -r -a learning_rates <<< "$learning_rates_csv"
IFS=',' read -r -a scales <<< "$scales_csv"
IFS=',' read -r -a limiters <<< "$limiter_csv"
IFS=',' read -r -a ranks <<< "$ranks_csv"
((${#learning_rates[@]} > 0)) || die "--learning-rates must not be empty"
((${#scales[@]} > 0)) || die "--scales must not be empty"
((${#limiters[@]} > 0)) || die "--limiters must not be empty"
((${#ranks[@]} > 0)) || die "--ranks must not be empty"
for rank in "${ranks[@]}"; do
    [[ "$rank" =~ ^[1-9][0-9]*$ ]] || die "invalid rank: $rank"
done
for value in "${learning_rates[@]}" "${scales[@]}"; do
    [[ "$value" =~ $number_pattern ]] || die "invalid positive number: $value"
done
for limiter in "${limiters[@]}"; do
    [[ "$limiter" == "on" || "$limiter" == "off" ]] || die "invalid limiter: $limiter"
done

slug() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    value="${value//+/x}"
    printf '%s' "$value"
}

mkdir -p "$output_dir"
for rank in "${ranks[@]}"; do
    if ((${#ranks[@]} > 1)); then
        rank_output_dir="$output_dir/rank-$rank"
    else
        rank_output_dir="$output_dir"
    fi
    for learning_rate in "${learning_rates[@]}"; do
        for scale in "${scales[@]}"; do
            for limiter in "${limiters[@]}"; do
            cell_dir="$rank_output_dir/lr-$(slug "$learning_rate")/scale-$(slug "$scale")/limiter-$limiter"
            result_path="$cell_dir/result.json"
            if [[ -e "$result_path" && "$force" -eq 0 ]]; then
                printf 'skip existing: %s\n' "$result_path"
                continue
            fi
            mkdir -p "$cell_dir"
            printf '==> rank=%s lr=%s scale=%s limiter=%s seeds=%s\n' "$rank" "$learning_rate" "$scale" "$limiter" "$seeds_csv"
            args=(
                --device "$device" --dtype "$dtype"
                --optimizers "$optimizers" --seeds "$seeds_csv"
                --rank "$rank" --train-tokens "$train_tokens"
                --eval-tokens "$eval_tokens" --max-seq-len "$max_seq_len"
                --batch-size "$batch_size" --epochs "$epochs"
                --steps-per-epoch "$steps_per_epoch"
                --learning-rate "$learning_rate" --apollo-scale "$scale"
                --apollo-norm-growth-rate "$norm_growth_rate"
                --lr-ema-confidence-beta "$confidence_beta"
                --lr-ema-confidence-alpha "$confidence_alpha"
            )
            if ((record_diagnostics)); then
                args+=(--record-update-norms --record-confidence-diagnostics)
            fi
            if [[ "$limiter" == "off" ]]; then
                args+=(--apollo-disable-norm-growth-limiter)
            fi
            "$python_bin" -m verify.text_lm_optimizer_convergence "${args[@]}" > "$result_path"
            done
        done
    done
done

report_tmp="$output_dir/.apollo-scale-sweep-report.md.tmp.$$"
trap 'rm -f -- "$report_tmp"' EXIT INT TERM
PYTHONPATH="$repo_root" "$python_bin" -m verify.text_lm_apollo_scale_report \
    "$output_dir" > "$report_tmp"
mv -- "$report_tmp" "$output_dir/apollo-scale-sweep-report.md"
trap - EXIT INT TERM
printf 'sweep report: %s\n' "$output_dir/apollo-scale-sweep-report.md"
