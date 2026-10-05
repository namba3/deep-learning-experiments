#!/usr/bin/env bash
# Run reproducible GPU validation for CAME-LRSF and APOLLO-CAME-LRSF.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${LRSF_PYTHON_BIN:-python3}"
data_dir="cifar10/data"
device="cuda"
dtype="bf16"
seed=0
seeds_csv=""
epochs=5
batch_size=8
max_train_samples=512
max_validation_samples=512
rank=4
ranks_csv=""
latent_channels=16
bottleneck_channels=256
downsample_stages=3
learning_rate=5e-4
weight_decay=0.0
refresh_interval=64
refresh_window=32
orthogonal_rate=0.01
orthogonal_direction="random"
orthogonal_signal="gradient"
record_step_metrics=0
record_update_norms=0
output_dir="output"
modes_csv="fixed,smoothstep,ema"
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_lrsf_gpu_validation.sh [options]

Runs ImageAE/CIFAR-10 GPU validation for CAME-LRSF and
APOLLO-CAME-LRSF. Each mode is written to a separate JSON file.

Options:
  --device DEVICE                 cuda or cpu (default: cuda)
  --dtype DTYPE                   bf16 or fp32 (default: bf16)
  --data-dir DIR                  CIFAR-10 directory (default: cifar10/data)
  --seed N                        random seed (default: 0)
  --seeds CSV                     optional seed sweep, e.g. 0,1,2
  --epochs N                      number of epochs (default: 5)
  --batch-size N                  batch size (default: 8)
  --max-train-samples N           training subset size (default: 512)
  --max-validation-samples N      validation subset size (default: 512)
  --rank N                        LRSF rank (default: 4)
  --ranks CSV                     optional LRSF rank sweep, e.g. 1,4,8,16
  --latent-channels N              ImageAE latent channels (default: 16)
  --bottleneck-channels N         ImageAE bottleneck channels (default: 256)
  --downsample-stages N           ImageAE downsample stages (default: 3)
  --learning-rate RATE            learning rate (default: 5e-4)
  --weight-decay RATE             decoupled weight decay (default: 0)
  --refresh-interval N             LRSF refresh interval (default: 64)
  --refresh-window N               smooth refresh window (default: 32)
  --orthogonal-rate RATE          per-step orthogonal rate (default: 0.01)
  --orthogonal-direction MODE     random, loss_directed, or loss_lowering (default: random)
  --orthogonal-signal MODE        gradient or effective_update (default: gradient)
  --record-step-metrics            record refresh loss/recovery diagnostics
  --record-update-norms            record parameter update norm statistics
  --modes CSV                     modes to run (default: fixed,smoothstep,ema)
  --output-dir DIR                result directory (default: output)
  --force                         overwrite existing result files
  -h, --help                      show this help

Modes:
  fixed       fixed/frozen projection (refresh-mode=none)
  hard        immediate projection replacement at each interval
  smoothstep  PA/PB double-buffer with smoothstep importance weight
  ema         PA/PB double-buffer with EMA importance weight
  stochastic  PA/PB double-buffer with stochastic branch selection
  orthogonal  interval-free per-step orthogonal refresh

Example:
  verify/launchers/run_lrsf_gpu_validation.sh \
    --modes fixed,hard,smoothstep,ema,stochastic,orthogonal

  verify/launchers/run_lrsf_gpu_validation.sh --seeds 0,1,2 \
    --modes fixed,smoothstep,ema,orthogonal
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--data-dir|--seed|--seeds|--epochs|--batch-size|--max-train-samples|--max-validation-samples|--rank|--ranks|--latent-channels|--bottleneck-channels|--downsample-stages|--learning-rate|--weight-decay|--refresh-interval|--refresh-window|--orthogonal-rate|--orthogonal-direction|--orthogonal-signal|--modes|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --data-dir) data_dir="$value" ;;
                --seed) seed="$value" ;;
                --seeds) seeds_csv="$value" ;;
                --epochs) epochs="$value" ;;
                --batch-size) batch_size="$value" ;;
                --max-train-samples) max_train_samples="$value" ;;
                --max-validation-samples) max_validation_samples="$value" ;;
                --rank) rank="$value" ;;
                --ranks) ranks_csv="$value" ;;
                --latent-channels) latent_channels="$value" ;;
                --bottleneck-channels) bottleneck_channels="$value" ;;
                --downsample-stages) downsample_stages="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --weight-decay) weight_decay="$value" ;;
                --refresh-interval) refresh_interval="$value" ;;
                --refresh-window) refresh_window="$value" ;;
                --orthogonal-rate) orthogonal_rate="$value" ;;
                --orthogonal-direction) orthogonal_direction="$value" ;;
                --orthogonal-signal) orthogonal_signal="$value" ;;
                --modes) modes_csv="$value" ;;
                --output-dir) output_dir="$value" ;;
            esac
            shift 2
            ;;
        --record-step-metrics) record_step_metrics=1; shift ;;
        --record-update-norms) record_update_norms=1; shift ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in cuda|cpu) ;; *) die "--device must be cuda or cpu" ;; esac
case "$dtype" in bf16|fp32) ;; *) die "--dtype must be bf16 or fp32" ;; esac
[[ -d "$data_dir" ]] || die "CIFAR-10 directory not found: $data_dir"
[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be positive"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be positive"
[[ "$max_train_samples" =~ ^[1-9][0-9]*$ ]] || die "--max-train-samples must be positive"
[[ "$max_validation_samples" =~ ^[1-9][0-9]*$ ]] || die "--max-validation-samples must be positive"
[[ "$rank" =~ ^[1-9][0-9]*$ ]] || die "--rank must be positive"
if [[ -n "$ranks_csv" ]]; then
    [[ "$ranks_csv" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] \
        || die "--ranks must be comma-separated positive integers"
fi
[[ "$latent_channels" =~ ^[1-9][0-9]*$ ]] || die "--latent-channels must be positive"
[[ "$bottleneck_channels" =~ ^[1-9][0-9]*$ ]] || die "--bottleneck-channels must be positive"
[[ "$downsample_stages" =~ ^[1-9][0-9]*$ ]] || die "--downsample-stages must be positive"
[[ "$refresh_interval" =~ ^[1-9][0-9]*$ ]] || die "--refresh-interval must be positive"
[[ "$refresh_window" =~ ^[1-9][0-9]*$ ]] || die "--refresh-window must be positive"
case "$orthogonal_direction" in random|loss_directed|loss_lowering) ;; *) die "--orthogonal-direction must be random, loss_directed, or loss_lowering" ;; esac
case "$orthogonal_signal" in gradient|effective_update) ;; *) die "--orthogonal-signal must be gradient or effective_update" ;; esac

if [[ -n "$seeds_csv" ]]; then
    IFS=',' read -r -a requested_seeds <<< "$seeds_csv"
else
    requested_seeds=("$seed")
fi
for requested_seed in "${requested_seeds[@]}"; do
    [[ "$requested_seed" =~ ^[0-9]+$ ]] || die "seed values must be non-negative integers"
done

if [[ "$device" == "cuda" ]]; then
    "$python_bin" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
        || die "CUDA is unavailable; use --device cpu only for a non-GPU contract check"
fi

mkdir -p "$output_dir"
for requested_seed in "${requested_seeds[@]}"; do
    seed="$requested_seed"
gpu_info_path="$output_dir/lrsf-gpu-info-seed${seed}.txt"
if [[ "$device" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
        > "$gpu_info_path" 2>&1; then
        printf 'nvidia-smi failed; torch CUDA check passed or reported the primary error.\n' \
            > "$gpu_info_path"
    fi
else
    printf 'nvidia-smi skipped; CPU mode was selected or nvidia-smi is unavailable.\n' \
        > "$gpu_info_path"
fi

common_args=(
    --data-dir "$data_dir" --device "$device" --dtype "$dtype"
    --seed "$seed" --epochs "$epochs" --batch-size "$batch_size"
    --max-train-samples "$max_train_samples"
    --max-validation-samples "$max_validation_samples"
    --rank "$rank" --latent-channels "$latent_channels"
    --bottleneck-channels "$bottleneck_channels"
    --downsample-stages "$downsample_stages"
    --learning-rate "$learning_rate" --weight-decay "$weight_decay"
    --refresh-interval "$refresh_interval" --refresh-window "$refresh_window"
    --orthogonal-refresh-direction "$orthogonal_direction"
    --orthogonal-refresh-signal "$orthogonal_signal"
)
if ((record_step_metrics)); then
    common_args+=(--record-step-metrics)
fi
if ((record_update_norms)); then
    common_args+=(--record-update-norms)
fi
rank_args=()
if [[ -n "$ranks_csv" ]]; then
    rank_args=(--ranks "$ranks_csv")
fi

IFS=',' read -r -a requested_modes <<< "$modes_csv"
for mode in "${requested_modes[@]}"; do
    case "$mode" in
        fixed)
            mode_args=(
                --optimizers CAME,CAME-SF,CAME-LRSF,APOLLO-CAME-LRSF
                --refresh-mode none --orthogonal-refresh-rate 0
            )
            ;;
        hard)
            mode_args=(
                --optimizers CAME-LRSF,APOLLO-CAME-LRSF
                --refresh-mode hard --orthogonal-refresh-rate 0
            )
            ;;
        smoothstep|ema|stochastic)
            mode_args=(
                --optimizers CAME-LRSF,APOLLO-CAME-LRSF
                --refresh-mode smooth --refresh-mix "$mode"
                --orthogonal-refresh-rate 0
            )
            ;;
        orthogonal)
            mode_args=(
                --optimizers CAME-LRSF,APOLLO-CAME-LRSF
                --refresh-mode none --orthogonal-refresh-rate "$orthogonal_rate"
            )
            ;;
        *) die "unknown mode '$mode'; see --help" ;;
    esac

    result_path="$output_dir/lrsf-gpu-$mode-seed${seed}.json"
    if [[ -e "$result_path" && "$force" != 1 ]]; then
        die "result already exists: $result_path; use --force to overwrite"
    fi
    printf '==> running %s -> %s\n' "$mode" "$result_path"
    PYTHONPATH="$repo_root" "$python_bin" \
        -m verify.image_ae_cifar10_optimizer_convergence \
        "${common_args[@]}" "${rank_args[@]}" "${mode_args[@]}" > "$result_path"
done

printf '==> completed LRSF GPU validation\n'
printf '    device=%s dtype=%s seed=%s modes=%s\n' "$device" "$dtype" "$seed" "$modes_csv"
printf '    GPU info: %s\n' "$gpu_info_path"
done
