#!/usr/bin/env bash
# Run a reproducible APOLLO orthogonal-rate/direction sweep.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${APOLLO_PYTHON_BIN:-python3}"
data_dir="cifar10/data"
device="cuda"
dtype="bf16"
epochs=5
batch_size=8
max_train_samples=512
max_validation_samples=512
rank=4
latent_channels=16
bottleneck_channels=256
downsample_stages=3
learning_rate="5e-4"
refresh_window=32
rates_csv="0.001,0.005,0.01,0.02,0.05"
directions_csv="random,loss_directed"
seeds_csv="0,1,2"
output_dir="output/apollo-orthogonal-sweep"
record_update_norms=1
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_apollo_orthogonal_sweep.sh [options]

Runs APOLLO none/orthogonal comparisons for each rate, direction, and seed.
Each cell is stored below the output directory and receives its own run log,
JSON results, and Markdown report. A sweep-level Markdown report is written
after all cells finish.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --data-dir DIR                  CIFAR-10 directory (default: cifar10/data)
  --epochs N                      number of epochs (default: 5)
  --batch-size N                  batch size (default: 8)
  --max-train-samples N           training subset size (default: 512)
  --max-validation-samples N      validation subset size (default: 512)
  --rank N                        APOLLO rank (default: 4)
  --latent-channels N             ImageAE latent channels (default: 16)
  --bottleneck-channels N         ImageAE bottleneck channels (default: 256)
  --downsample-stages N           ImageAE downsample stages (default: 3)
  --learning-rate RATE             learning rate (default: 5e-4)
  --refresh-window N               smooth refresh window (default: 32)
  --rates CSV                      rates (default: 0.001,0.005,0.01,0.02,0.05)
  --directions CSV                 random,loss_directed (default: both)
  --seeds CSV                      seeds (default: 0,1,2)
  --output-dir DIR                 sweep output directory
  --record-update-norms            record update norm statistics (default)
  --no-record-update-norms         disable update norm statistics
  --force                          overwrite existing cell outputs
  -h, --help                       show this help
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--data-dir|--epochs|--batch-size|--max-train-samples|--max-validation-samples|--rank|--latent-channels|--bottleneck-channels|--downsample-stages|--learning-rate|--refresh-window|--rates|--directions|--seeds|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --data-dir) data_dir="$value" ;;
                --epochs) epochs="$value" ;;
                --batch-size) batch_size="$value" ;;
                --max-train-samples) max_train_samples="$value" ;;
                --max-validation-samples) max_validation_samples="$value" ;;
                --rank) rank="$value" ;;
                --latent-channels) latent_channels="$value" ;;
                --bottleneck-channels) bottleneck_channels="$value" ;;
                --downsample-stages) downsample_stages="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --refresh-window) refresh_window="$value" ;;
                --rates) rates_csv="$value" ;;
                --directions) directions_csv="$value" ;;
                --seeds) seeds_csv="$value" ;;
                --output-dir) output_dir="$value" ;;
            esac
            shift 2
            ;;
        --record-update-norms) record_update_norms=1; shift ;;
        --no-record-update-norms) record_update_norms=0; shift ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "--device must be auto, cpu, or cuda" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "--dtype must be fp32 or bf16" ;; esac
[[ -d "$data_dir" ]] || die "CIFAR-10 directory not found: $data_dir"
[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be positive"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be positive"
[[ "$rank" =~ ^[1-9][0-9]*$ ]] || die "--rank must be positive"
[[ "$refresh_window" =~ ^[1-9][0-9]*$ ]] || die "--refresh-window must be positive"

IFS=',' read -r -a rates <<< "$rates_csv"
IFS=',' read -r -a directions <<< "$directions_csv"
IFS=',' read -r -a seeds <<< "$seeds_csv"
((${#rates[@]} > 0)) || die "--rates must not be empty"
((${#directions[@]} > 0)) || die "--directions must not be empty"
((${#seeds[@]} > 0)) || die "--seeds must not be empty"

for rate in "${rates[@]}"; do
    [[ "$rate" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid rate: $rate"
done
for direction in "${directions[@]}"; do
    case "$direction" in random|loss_directed) ;; *) die "invalid direction: $direction" ;; esac
done
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed: $seed"
done

mkdir -p "$output_dir"
for direction in "${directions[@]}"; do
    for rate in "${rates[@]}"; do
        rate_slug="${rate//./p}"
        cell_root="$output_dir/$direction/rate-$rate_slug"
        for seed in "${seeds[@]}"; do
            cell_dir="$cell_root/seed-$seed"
            args=(
                --device "$device" --dtype "$dtype" --data-dir "$data_dir"
                --seed "$seed" --epochs "$epochs" --batch-size "$batch_size"
                --max-train-samples "$max_train_samples"
                --max-validation-samples "$max_validation_samples"
                --rank "$rank" --latent-channels "$latent_channels"
                --bottleneck-channels "$bottleneck_channels"
                --downsample-stages "$downsample_stages"
                --learning-rate "$learning_rate"
                --refresh-window "$refresh_window"
                --modes none,orthogonal
                --orthogonal-rate "$rate"
                --orthogonal-direction "$direction"
                --output-dir "$cell_dir"
            )
            if ((record_update_norms)); then
                args+=(--record-update-norms)
            fi
            if ((force)); then
                args+=(--force)
            fi
            printf '==> direction=%s rate=%s seed=%s\n' "$direction" "$rate" "$seed"
            bash "$script_dir/run_apollo_refresh_experiments.sh" "${args[@]}"
        done
    done
done

report_tmp="$output_dir/.apollo-orthogonal-sweep-report.md.tmp.$$"
trap 'rm -f -- "$report_tmp"' EXIT INT TERM
PYTHONPATH="$repo_root" "$python_bin" -m verify.apollo_orthogonal_sweep_report \
    "$output_dir" > "$report_tmp"
mv -- "$report_tmp" "$output_dir/apollo-orthogonal-sweep-report.md"
trap - EXIT INT TERM
printf 'sweep report: %s\n' "$output_dir/apollo-orthogonal-sweep-report.md"
