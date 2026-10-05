#!/usr/bin/env bash
# Run a rank x refresh-interval Schedule-Free trajectory sweep.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${TEXT_LM_PYTHON_BIN:-python3}"
device="cuda"
dtype="bf16"
ranks_csv="4,8,16"
intervals_csv="25,50"
seeds_csv="0,1,2"
train_tokens=49152
eval_tokens=1024
max_seq_len=128
batch_size=4
epochs=1
steps_per_epoch=100
learning_rate="3e-4"
refresh_mode="hard"
refresh_mix="smoothstep"
transport_overlap=""
snapshot_interval=5
max_snapshots=20
max_elements=2000000
max_tensors=4
output_dir="output/text-lm-schedulefree-trajectory-sweep"
record_refresh_diagnostics=0
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_text_lm_schedulefree_trajectory_sweep.sh [options]

Runs Schedule-Free trajectory curvature/gap diagnostics for every rank and
refresh interval cell. Existing cell JSON files are skipped unless --force
is supplied.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --ranks CSV                     LRSF ranks (default: 4,8,16)
  --intervals CSV                 refresh intervals (default: 25,50)
  --seeds CSV                     non-negative seeds (default: 0,1,2)
  --train-tokens N                training token budget (default: 49152)
  --eval-tokens N                 evaluation token budget (default: 1024)
  --max-seq-len N                 sequence length (default: 128)
  --batch-size N                  batch size (default: 4)
  --epochs N                      number of epochs (default: 1)
  --steps-per-epoch N             optimizer steps per epoch (default: 100)
  --learning-rate RATE             learning rate (default: 3e-4)
  --refresh-mode MODE              none, hard, smooth, or shadow (default: hard)
  --refresh-mix MIX                linear, smoothstep, stochastic, or ema
  --transport-overlap FLOAT        retain old basis at refresh (default: unset)
  --snapshot-interval N            steps between trajectory snapshots (default: 5)
  --max-snapshots N               maximum snapshots per source (default: 20)
  --max-elements N                diagnostic element limit (default: 2000000)
  --max-tensors N                 parameter tensors per case (default: 4)
  --output-dir DIR                sweep output directory
  --record-refresh-diagnostics    record transport error at refresh events
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
        --device|--dtype|--ranks|--intervals|--seeds|--train-tokens|--eval-tokens|--max-seq-len|--batch-size|--epochs|--steps-per-epoch|--learning-rate|--refresh-mode|--refresh-mix|--transport-overlap|--snapshot-interval|--max-snapshots|--max-elements|--max-tensors|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --ranks) ranks_csv="$value" ;;
                --intervals) intervals_csv="$value" ;;
                --seeds) seeds_csv="$value" ;;
                --train-tokens) train_tokens="$value" ;;
                --eval-tokens) eval_tokens="$value" ;;
                --max-seq-len) max_seq_len="$value" ;;
                --batch-size) batch_size="$value" ;;
                --epochs) epochs="$value" ;;
                --steps-per-epoch) steps_per_epoch="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --refresh-mode) refresh_mode="$value" ;;
                --refresh-mix) refresh_mix="$value" ;;
                --transport-overlap) transport_overlap="$value" ;;
                --snapshot-interval) snapshot_interval="$value" ;;
                --max-snapshots) max_snapshots="$value" ;;
                --max-elements) max_elements="$value" ;;
                --max-tensors) max_tensors="$value" ;;
                --output-dir) output_dir="$value" ;;
            esac
            shift 2
            ;;
        --record-refresh-diagnostics) record_refresh_diagnostics=1; shift ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "invalid --device" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "invalid --dtype" ;; esac
case "$refresh_mode" in none|hard|smooth|shadow) ;; *) die "invalid --refresh-mode" ;; esac
case "$refresh_mix" in linear|smoothstep|stochastic|ema) ;; *) die "invalid --refresh-mix" ;; esac
for value in "$train_tokens" "$eval_tokens" "$max_seq_len" "$batch_size" "$epochs" "$steps_per_epoch" "$snapshot_interval" "$max_snapshots" "$max_elements" "$max_tensors"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "positive integer expected: $value"
done
((epochs * steps_per_epoch >= 4 * snapshot_interval)) || die "training steps must provide at least four snapshots"
((max_snapshots >= 4)) || die "--max-snapshots must be at least 4"

IFS=',' read -r -a ranks <<< "$ranks_csv"
IFS=',' read -r -a intervals <<< "$intervals_csv"
((${#ranks[@]} > 0)) || die "--ranks must not be empty"
((${#intervals[@]} > 0)) || die "--intervals must not be empty"
for rank in "${ranks[@]}"; do
    [[ "$rank" =~ ^[1-9][0-9]*$ ]] || die "invalid rank: $rank"
done
for refresh_interval in "${intervals[@]}"; do
    [[ "$refresh_interval" =~ ^[1-9][0-9]*$ ]] || die "invalid interval: $refresh_interval"
done

mkdir -p "$output_dir"
for rank in "${ranks[@]}"; do
    for refresh_interval in "${intervals[@]}"; do
        cell_dir="$output_dir/rank-$rank/interval-$refresh_interval"
        result_path="$cell_dir/result.json"
        if [[ -e "$result_path" && "$force" -eq 0 ]]; then
            printf 'skip existing: %s\n' "$result_path"
            continue
        fi
        mkdir -p "$cell_dir"
        printf '==> rank=%s refresh_interval=%s seeds=%s\n' "$rank" "$refresh_interval" "$seeds_csv"
        diagnostic_args=()
        if [[ "$record_refresh_diagnostics" -eq 1 ]]; then
            diagnostic_args+=(--record-refresh-diagnostics)
        fi
        overlap_args=()
        if [[ -n "$transport_overlap" ]]; then
            overlap_args+=(--refresh-transport-overlap "$transport_overlap")
        fi
        "$python_bin" -m verify.text_lm_optimizer_convergence \
            --device "$device" --dtype "$dtype" \
            --optimizers AdamW-SF,AdamW-LRSF --seeds "$seeds_csv" \
            --rank "$rank" --train-tokens "$train_tokens" \
            --eval-tokens "$eval_tokens" --max-seq-len "$max_seq_len" \
            --batch-size "$batch_size" --epochs "$epochs" \
            --steps-per-epoch "$steps_per_epoch" \
            --learning-rate "$learning_rate" \
            --refresh-mode "$refresh_mode" \
            --refresh-interval "$refresh_interval" \
            --refresh-window "$refresh_interval" \
            --refresh-mix "$refresh_mix" \
            "${overlap_args[@]}" \
            --record-trajectory-curvature \
            "${diagnostic_args[@]}" \
            --state-rank-interval "$snapshot_interval" \
            --state-trajectory-max-snapshots "$max_snapshots" \
            --state-rank-max-elements "$max_elements" \
            --state-rank-max-tensors "$max_tensors" \
            > "$result_path"
    done
done

printf 'sweep output: %s\n' "$output_dir"
