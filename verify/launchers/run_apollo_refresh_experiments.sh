#!/usr/bin/env bash
# Run APOLLO refresh variants sequentially and render one comparison report.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${APOLLO_PYTHON_BIN:-python3}"
data_dir="cifar10/data"
device="cuda"
dtype="bf16"
seed=0
epochs=5
batch_size=8
max_train_samples=512
max_validation_samples=512
rank=4
latent_channels=16
bottleneck_channels=256
downsample_stages=3
learning_rate=5e-4
update_proj_gap=200
refresh_window=32
orthogonal_rate=0.01
orthogonal_direction="random"
update_norm_variance_cap=""
output_dir="output"
modes_csv="none,hard,smooth-ema,smooth-stochastic,orthogonal"
record_update_norms=0
force=0

usage() {
    cat <<'EOF'
Usage: verify/launchers/run_apollo_refresh_experiments.sh [options]

Runs the APOLLO refresh matrix sequentially and writes JSON plus a Markdown
summary to the output directory. Script output is saved as
apollo-refresh-run-seed<N>.log in that directory.

Options:
  --device DEVICE                 auto, cpu, or cuda (default: cuda)
  --dtype DTYPE                   fp32 or bf16 (default: bf16)
  --data-dir DIR                  CIFAR-10 directory (default: cifar10/data)
  --seed N                        random seed (default: 0)
  --epochs N                      number of epochs (default: 5)
  --batch-size N                  batch size (default: 8)
  --max-train-samples N           training subset size (default: 512)
  --max-validation-samples N      validation subset size (default: 512)
  --rank N                        APOLLO rank (default: 4)
  --latent-channels N             ImageAE latent channels (default: 16)
  --bottleneck-channels N         ImageAE bottleneck channels (default: 256)
  --downsample-stages N           ImageAE downsample stages (default: 3)
  --learning-rate RATE            learning rate (default: 5e-4)
  --update-proj-gap N             interval refresh gap (default: 200)
  --refresh-window N              smooth refresh window (default: 32)
  --orthogonal-rate RATE          per-step orthogonal rate (default: 0.01)
  --orthogonal-direction MODE     random or loss_directed (default: random)
  --update-norm-variance-cap V    experimental APOLLO update-norm cap
  --modes CSV                     variants to run
  --output-dir DIR                result directory (default: output)
  --record-update-norms           add host-side update norm metrics
  --force                         overwrite existing result files
  -h, --help                      show this help

Modes:
  none, hard, smooth-ema, smooth-stochastic, orthogonal
EOF
}

die() {
    printf 'error: %s\n' "$1" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --device|--dtype|--data-dir|--seed|--epochs|--batch-size|--max-train-samples|--max-validation-samples|--rank|--latent-channels|--bottleneck-channels|--downsample-stages|--learning-rate|--update-proj-gap|--refresh-window|--orthogonal-rate|--orthogonal-direction|--update-norm-variance-cap|--modes|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            option="$1"
            value="$2"
            case "$option" in
                --device) device="$value" ;;
                --dtype) dtype="$value" ;;
                --data-dir) data_dir="$value" ;;
                --seed) seed="$value" ;;
                --epochs) epochs="$value" ;;
                --batch-size) batch_size="$value" ;;
                --max-train-samples) max_train_samples="$value" ;;
                --max-validation-samples) max_validation_samples="$value" ;;
                --rank) rank="$value" ;;
                --latent-channels) latent_channels="$value" ;;
                --bottleneck-channels) bottleneck_channels="$value" ;;
                --downsample-stages) downsample_stages="$value" ;;
                --learning-rate) learning_rate="$value" ;;
                --update-proj-gap) update_proj_gap="$value" ;;
                --refresh-window) refresh_window="$value" ;;
                --orthogonal-rate) orthogonal_rate="$value" ;;
                --orthogonal-direction) orthogonal_direction="$value" ;;
                --update-norm-variance-cap) update_norm_variance_cap="$value" ;;
                --modes) modes_csv="$value" ;;
                --output-dir) output_dir="$value" ;;
            esac
            shift 2
            ;;
        --record-update-norms) record_update_norms=1; shift ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$device" in auto|cpu|cuda) ;; *) die "--device must be auto, cpu, or cuda" ;; esac
case "$dtype" in fp32|bf16) ;; *) die "--dtype must be fp32 or bf16" ;; esac
case "$orthogonal_direction" in random|loss_directed) ;; *) die "--orthogonal-direction must be random or loss_directed" ;; esac
[[ -d "$data_dir" ]] || die "CIFAR-10 directory not found: $data_dir"
[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be positive"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be positive"
[[ "$rank" =~ ^[1-9][0-9]*$ ]] || die "--rank must be positive"
[[ "$latent_channels" =~ ^[1-9][0-9]*$ ]] || die "--latent-channels must be positive"
[[ "$bottleneck_channels" =~ ^[1-9][0-9]*$ ]] || die "--bottleneck-channels must be positive"
[[ "$downsample_stages" =~ ^[1-9][0-9]*$ ]] || die "--downsample-stages must be positive"
[[ "$update_proj_gap" =~ ^[1-9][0-9]*$ ]] || die "--update-proj-gap must be positive"
[[ "$refresh_window" =~ ^[1-9][0-9]*$ ]] || die "--refresh-window must be positive"

mkdir -p "$output_dir"
run_log="$output_dir/apollo-refresh-run-seed$seed.log"
if [[ -e "$run_log" && "$force" != 1 ]]; then
    die "run log already exists: $run_log; use --force to overwrite"
fi
exec >"$run_log" 2>&1
printf 'APOLLO refresh experiment started: %s\n' "$(date --iso-8601=seconds)"
printf 'device=%s dtype=%s seed=%s epochs=%s modes=%s\n' \
    "$device" "$dtype" "$seed" "$epochs" "$modes_csv"

common_args=(
    --data-dir "$data_dir" --device "$device" --dtype "$dtype"
    --seed "$seed" --epochs "$epochs" --batch-size "$batch_size"
    --max-train-samples "$max_train_samples"
    --max-validation-samples "$max_validation_samples"
    --rank "$rank" --latent-channels "$latent_channels"
    --bottleneck-channels "$bottleneck_channels"
    --downsample-stages "$downsample_stages"
    --learning-rate "$learning_rate"
    --update-proj-gap "$update_proj_gap"
    --projection-refresh-state transport --record-step-metrics
)
if ((record_update_norms)); then
    common_args+=(--record-update-norms)
fi
if [[ -n "$update_norm_variance_cap" ]]; then
    common_args+=(--update-norm-variance-cap "$update_norm_variance_cap")
fi

IFS=',' read -r -a requested_modes <<< "$modes_csv"
result_paths=()
for mode in "${requested_modes[@]}"; do
    case "$mode" in
        none)
            mode_args=(--optimizers AdamW,CAME,APOLLO,APOLLO-CAME,APOLLO-Mini --projection-refresh-mode none)
            ;;
        hard)
            mode_args=(--optimizers APOLLO,APOLLO-CAME,APOLLO-Mini --projection-refresh-mode hard)
            ;;
        smooth-ema|smooth-stochastic)
            mix="ema"
            [[ "$mode" == "smooth-stochastic" ]] && mix="stochastic"
            mode_args=(
                --optimizers APOLLO,APOLLO-CAME,APOLLO-Mini
                --projection-refresh-mode smooth
                --projection-refresh-window "$refresh_window"
                --projection-refresh-mix "$mix"
            )
            ;;
        orthogonal)
            mode_args=(
                --optimizers APOLLO,APOLLO-CAME,APOLLO-Mini
                --projection-refresh-mode none
                --orthogonal-refresh-rate "$orthogonal_rate"
                --orthogonal-refresh-direction "$orthogonal_direction"
            )
            ;;
        *) die "unknown mode '$mode'; see --help" ;;
    esac

    result_path="$output_dir/apollo-refresh-$mode-seed$seed.json"
    if [[ -e "$result_path" && "$force" != 1 ]]; then
        die "result already exists: $result_path; use --force to overwrite"
    fi
    printf '==> running %s\n' "$mode"
    result_tmp="$output_dir/.apollo-refresh-$mode-seed$seed.json.tmp.$$"
    trap 'rm -f -- "$result_tmp"' EXIT INT TERM
    PYTHONPATH="$repo_root" "$python_bin" -m verify.image_ae_cifar10_adamw_apollo \
        "${common_args[@]}" "${mode_args[@]}" > "$result_tmp"
    mv -- "$result_tmp" "$result_path"
    trap - EXIT INT TERM
    result_paths+=("$result_path")
done

report_path="$output_dir/apollo-refresh-report-seed$seed.md"
if [[ -e "$report_path" && "$force" != 1 ]]; then
    die "report already exists: $report_path; use --force to overwrite"
fi
printf '==> rendering %s\n' "$report_path"
report_tmp="$output_dir/.apollo-refresh-report-seed$seed.md.tmp.$$"
trap 'rm -f -- "$report_tmp"' EXIT INT TERM
PYTHONPATH="$repo_root" "$python_bin" -m verify.apollo_refresh_report "${result_paths[@]}" \
    > "$report_tmp"
mv -- "$report_tmp" "$report_path"
trap - EXIT INT TERM
printf 'report: %s\n' "$report_path"
printf 'run log: %s\n' "$run_log"
