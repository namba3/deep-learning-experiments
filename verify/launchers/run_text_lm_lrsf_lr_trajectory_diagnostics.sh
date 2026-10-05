#!/usr/bin/env bash
# Compare AdamW-LRSF-LR reset/transport with trajectory diagnostics.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
seeds="0,1,2"
rank=16
train_tokens=49152
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=1
steps_per_epoch=100
eval_interval=20
learning_rate="3e-4"
refresh_interval=25
snapshot_interval=20
transport_overlap="0.99"
max_snapshots=20
max_elements=2000000
max_tensors=4
output_dir="output/text-lm-lrsf-lr-reset-trajectory-diagnostics"
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_lrsf_lr_trajectory_diagnostics.sh [options]

Runs a low-load paired trajectory diagnostic for AdamW-LRSF-LR reset and
transport latent-moment policies. Diagnostic snapshots add overhead and the
result is not a speed benchmark.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --rank N                        LRSF rank (default: 16)
  --train-tokens N                training token budget (default: 49152)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 100)
  --eval-interval N               periodic validation interval (default: 20)
  --learning-rate RATE            learning rate (default: 3e-4)
  --refresh-interval N            hard refresh interval (default: 25)
  --snapshot-interval N           trajectory snapshot interval (default: 20)
  --transport-overlap VALUE       refresh basis overlap (default: 0.99)
  --max-snapshots N               maximum snapshots per source (default: 20)
  --max-elements N                diagnostic element limit (default: 2000000)
  --max-tensors N                 parameter tensors per case (default: 4)
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
        --device|--dtype|--seeds|--rank|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--eval-interval|--learning-rate|--refresh-interval|--snapshot-interval|--transport-overlap|--max-snapshots|--max-elements|--max-tensors|--output-dir)
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
                --eval-interval) eval_interval="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --refresh-interval) refresh_interval="$value" ;;
                --snapshot-interval) snapshot_interval="$value" ;;
                --transport-overlap) transport_overlap="$value" ;;
                --max-snapshots) max_snapshots="$value" ;;
                --max-elements) max_elements="$value" ;;
                --max-tensors) max_tensors="$value" ;;
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
for value in "$rank" "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch" "$eval_interval" "$refresh_interval" "$snapshot_interval" "$max_snapshots" "$max_elements" "$max_tensors"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done
(( eval_interval <= epochs * steps_per_epoch )) || die "eval interval exceeds total steps"
(( epochs * steps_per_epoch >= 4 * eval_interval )) || die "training must provide at least four validation snapshots"
(( max_snapshots >= 4 )) || die "--max-snapshots must be at least 4"
[[ "$transport_overlap" =~ ^(0([.][0-9]+)?|1([.][0]+)?)$ ]] || die "invalid overlap: $transport_overlap"
mkdir -p "$output_dir"

for policy in reset transport; do
    result_path="$output_dir/$policy.json"
    if [[ -e "$result_path" && "$force" != 1 ]]; then
        printf 'skip existing: %s\n' "$result_path"
        continue
    fi
    printf '==> policy=%s rank=%s interval=%s seeds=%s\n' \
        "$policy" "$rank" "$refresh_interval" "$seeds"
    "$python_bin" -m verify.text_lm_optimizer_convergence \
        --device "$device" --dtype "$dtype" \
        --optimizers AdamW-LRSF-LR --seeds "$seeds" --rank "$rank" \
        --train-tokens "$train_tokens" --eval-tokens "$eval_tokens" \
        --max-seq-len "$max_seq_len" --batch-size "$batch_size" \
        --epochs "$epochs" --steps-per-epoch "$steps_per_epoch" \
        --eval-interval "$eval_interval" --learning-rate "$learning_rate" \
        --refresh-mode hard --refresh-interval "$refresh_interval" \
        --refresh-window "$refresh_interval" --refresh-mix smoothstep \
        --lrsf-refresh-state "$policy" \
        --refresh-transport-overlap "$transport_overlap" \
        --record-refresh-diagnostics --record-trajectory-curvature \
        --state-rank-interval "$snapshot_interval" \
        --state-trajectory-max-snapshots "$max_snapshots" \
        --state-rank-max-elements "$max_elements" \
        --state-rank-max-tensors "$max_tensors" > "$result_path"
done

printf 'trajectory diagnostic outputs: %s/{reset,transport}.json\n' "$output_dir"
