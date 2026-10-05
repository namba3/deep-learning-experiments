#!/usr/bin/env bash
# Run paired baselines and confidence-EMA sensitivity cells on TinyStories.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
optimizers="AdamW-SF,AdamW-LRSF,AdamW-LR-EMA-Conf,AdamW-LR-EMA-Conf-LRSF,APOLLO,APOLLO-Conf"
seeds_csv="0,1,2"
rank=8
train_tokens=16384
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=1
steps_per_epoch=50
learning_rate="3e-4"
apollo_scale="1.0"
apollo_disable_norm_growth_limiter=0
ema_beta="0.9"
confidence_betas_csv="0.95,0.99"
confidence_alphas_csv="0.0001,0.001,0.01"
output_dir="output/text-lm-lr-ema-confidence-sweep"
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_lr_ema_confidence_sweep.sh [options]

Runs the selected optimizers with paired seeds. By default it includes
AdamW-SF, AdamW-LRSF, AdamW-LR-EMA, AdamW-LR-EMA-Conf, and the integrated
AdamW-LR-EMA-Conf-LRSF, APOLLO, and APOLLO-Conf. Confidence-beta x alpha
cells are saved separately and can be resumed.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --optimizers CSV                optimizer names (default: six-way comparison)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --rank N                        low-rank projection rank (default: 8)
  --train-tokens N                training token budget (default: 16384)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 50)
  --learning-rate RATE             learning rate (default: 3e-4)
  --apollo-scale RATE              APOLLO projection scale (default: 1.0)
  --apollo-disable-norm-growth-limiter
                                   disable APOLLO norm-growth limiter
  --ema-beta RATE                 projected-gradient EMA decay (default: 0.9)
  --confidence-betas CSV           confidence EMA decays (default: 0.95,0.99)
  --confidence-alphas CSV          mean-square floors (default: 0.0001,0.001,0.01)
  --output-dir DIR                output directory
  --force                         overwrite existing cells
  -h, --help                      show this help
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--optimizers|--seeds|--rank|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--learning-rate|--apollo-scale|--ema-beta|--confidence-betas|--confidence-alphas|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --optimizers) optimizers="$value" ;;
                --seeds) seeds_csv="$value" ;;
                --rank) rank="$value" ;;
                --train-tokens) train_tokens="$value" ;;
                --eval-tokens) eval_tokens="$value" ;;
                --max-seq-len) max_seq_len="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --steps-per-epoch) steps_per_epoch="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --apollo-scale) apollo_scale="$value" ;;
                --ema-beta) ema_beta="$value" ;;
                --confidence-betas) confidence_betas_csv="$value" ;;
                --confidence-alphas) confidence_alphas_csv="$value" ;;
                --output-dir) output_dir="$value" ;;
            esac
            shift 2
            ;;
        --force) force=1; shift ;;
        --apollo-disable-norm-growth-limiter) apollo_disable_norm_growth_limiter=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "invalid --device" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "invalid --dtype" ;; esac
for value in "$rank" "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done
number_pattern='^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$'
[[ "$apollo_scale" =~ $number_pattern ]] || die "invalid APOLLO scale: $apollo_scale"

IFS=',' read -r -a confidence_betas <<< "$confidence_betas_csv"
IFS=',' read -r -a confidence_alphas <<< "$confidence_alphas_csv"
((${#confidence_betas[@]} > 0)) || die "--confidence-betas must not be empty"
((${#confidence_alphas[@]} > 0)) || die "--confidence-alphas must not be empty"

safe_name() {
    printf '%s' "$1" | tr '.-' 'p_'
}

mkdir -p "$output_dir"

run_cell() {
    local label="$1"
    local confidence_beta="$2"
    local confidence_alpha="$3"
    local cell_dir="$output_dir/$label"
    local result_path="$cell_dir/result.json"
    if [[ -e "$result_path" && "$force" -eq 0 ]]; then
        printf 'skip existing: %s\n' "$result_path"
        return
    fi
    mkdir -p "$cell_dir"
    printf '==> %s seeds=%s\n' "$label" "$seeds_csv"
    args=(
        --device "$device" --dtype "$dtype"
        --optimizers "$optimizers"
        --seeds "$seeds_csv" --rank "$rank"
        --train-tokens "$train_tokens" --eval-tokens "$eval_tokens"
        --max-seq-len "$max_seq_len" --batch-size "$batch_size"
        --epochs "$epochs" --steps-per-epoch "$steps_per_epoch"
        --learning-rate "$learning_rate" --lr-ema-beta "$ema_beta"
        --apollo-scale "$apollo_scale"
        --lr-ema-confidence-beta "$confidence_beta"
        --lr-ema-confidence-alpha "$confidence_alpha"
        --record-update-norms --record-confidence-diagnostics
        --state-rank-interval 10
    )
    if ((apollo_disable_norm_growth_limiter)); then
        args+=(--apollo-disable-norm-growth-limiter)
    fi
    "$python_bin" -m verify.text_lm_optimizer_convergence \
        "${args[@]}" \
        > "$result_path"
}

for confidence_beta in "${confidence_betas[@]}"; do
    for confidence_alpha in "${confidence_alphas[@]}"; do
        run_cell \
            "confidence-beta-$(safe_name "$confidence_beta")-alpha-$(safe_name "$confidence_alpha")" \
            "$confidence_beta" "$confidence_alpha"
    done
done

report_tmp="$output_dir/.confidence-report.md.tmp.$$"
trap 'rm -f -- "$report_tmp"' EXIT INT TERM
PYTHONPATH="$repo_root" "$python_bin" -m verify.text_lm_apollo_confidence_report \
    "$output_dir" > "$report_tmp"
mv -- "$report_tmp" "$output_dir/confidence-report.md"
trap - EXIT INT TERM
printf 'sweep output: %s\n' "$output_dir"
