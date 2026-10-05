#!/usr/bin/env bash
# Run a reproducible APOLLO update-norm variance-cap sweep.

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
caps_csv="none,0.0001,0.001,0.01"
seeds_csv="0,1,2"
output_dir="output/apollo-variance-cap-sweep"
record_update_norms=1
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_apollo_variance_cap_sweep.sh [options]

Runs APOLLO/APOLLO-CAME/APOLLO-Mini with and without update-norm variance
caps. Each cell receives its own JSON results, Markdown report, and log.
The sweep-level report is written below the output directory.

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
  --caps CSV                      none or non-negative caps (default: none,0.0001,0.001,0.01)
  --seeds CSV                     seeds (default: 0,1,2)
  --output-dir DIR                sweep output directory
  --record-update-norms           record update norm statistics (default)
  --no-record-update-norms        disable update norm statistics
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
        --device|--dtype|--data-dir|--epochs|--batch-size|--max-train-samples|--max-validation-samples|--rank|--latent-channels|--bottleneck-channels|--downsample-stages|--learning-rate|--caps|--seeds|--output-dir)
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
                --caps) caps_csv="$value" ;;
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

IFS=',' read -r -a caps <<< "$caps_csv"
IFS=',' read -r -a seeds <<< "$seeds_csv"
((${#caps[@]} > 0)) || die "--caps must not be empty"
((${#seeds[@]} > 0)) || die "--seeds must not be empty"
for cap in "${caps[@]}"; do
    [[ "$cap" == "none" || "$cap" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid cap: $cap"
done
has_none=0
for cap in "${caps[@]}"; do
    [[ "$cap" == "none" ]] && has_none=1
done
((has_none)) || die "--caps must include none as baseline"
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed: $seed"
done

mkdir -p "$output_dir"
run_log="$output_dir/apollo-variance-cap-sweep.log"
if [[ -e "$run_log" && "$force" != 1 ]]; then
    die "run log already exists: $run_log; use --force to overwrite"
fi
exec >"$run_log" 2>&1
printf 'APOLLO variance-cap sweep started: %s\n' "$(date --iso-8601=seconds)"

for cap in "${caps[@]}"; do
    cap_slug="${cap//./p}"
    [[ "$cap" == "none" ]] && cap_slug="none"
    for seed in "${seeds[@]}"; do
        cell_dir="$output_dir/cap-$cap_slug/seed-$seed"
        args=(
            --device "$device" --dtype "$dtype" --data-dir "$data_dir"
            --seed "$seed" --epochs "$epochs" --batch-size "$batch_size"
            --max-train-samples "$max_train_samples"
            --max-validation-samples "$max_validation_samples"
            --rank "$rank" --latent-channels "$latent_channels"
            --bottleneck-channels "$bottleneck_channels"
            --downsample-stages "$downsample_stages"
            --learning-rate "$learning_rate"
            --modes none --output-dir "$cell_dir"
        )
        if [[ "$cap" != "none" ]]; then
            args+=(--update-norm-variance-cap "$cap")
        fi
        if ((record_update_norms)); then
            args+=(--record-update-norms)
        fi
        if ((force)); then
            args+=(--force)
        fi
        printf '==> cap=%s seed=%s\n' "$cap" "$seed"
        bash "$script_dir/run_apollo_refresh_experiments.sh" "${args[@]}"
    done
done

report_tmp="$output_dir/.apollo-variance-cap-sweep-report.md.tmp.$$"
trap 'rm -f -- "$report_tmp"' EXIT INT TERM
PYTHONPATH="$repo_root" "$python_bin" -m verify.apollo_variance_cap_report \
    "$output_dir" > "$report_tmp"
mv -- "$report_tmp" "$output_dir/apollo-variance-cap-sweep-report.md"
trap - EXIT INT TERM
printf 'sweep report: %s\n' "$output_dir/apollo-variance-cap-sweep-report.md"
printf 'run log: %s\n' "$run_log"
