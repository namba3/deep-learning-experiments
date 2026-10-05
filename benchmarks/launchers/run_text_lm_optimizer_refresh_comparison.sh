#!/usr/bin/env bash
set -euo pipefail

# Compare hard and PA/PB refresh policies for AdamW-LRSF on TinyStories.
# Each policy delegates to the long-rank wrapper and receives an independent
# output directory, while all model, seed, rank, and token-budget settings are
# inherited unchanged.

output_root="${TEXT_LM_OPTIMIZER_REFRESH_DIR:-output/text-lm-optimizer-refresh}"
modes="${TEXT_LM_REFRESH_MODES:-hard,smoothstep,ema,ema_fast,stochastic}"
interval="${TEXT_LM_REFRESH_INTERVAL:-100}"
window="${TEXT_LM_REFRESH_WINDOW:-100}"
requested_ema_decay="${TEXT_LM_REFRESH_EMA_DECAY:-}"
ema_fast_decay="${TEXT_LM_REFRESH_EMA_FAST_DECAY:-0.96}"

mkdir -p "$output_root"
IFS=',' read -r -a mode_values <<< "$modes"

for mode in "${mode_values[@]}"; do
  mode="${mode// /}"
  if [[ -z "$mode" ]]; then
    continue
  fi
  ema_decay="$requested_ema_decay"
  case "$mode" in
    frozen|fixed)
      refresh_mode="none"
      refresh_mix="smoothstep"
      mode_label="frozen"
      ema_decay=""
      ;;
    hard)
      refresh_mode="hard"
      refresh_mix="smoothstep"
      mode_label="hard"
      ;;
    shadow)
      refresh_mode="shadow"
      refresh_mix="smoothstep"
      mode_label="shadow"
      ;;
    smoothstep|ema|stochastic)
      refresh_mode="smooth"
      refresh_mix="$mode"
      mode_label="$mode"
      ;;
    ema_fast)
      refresh_mode="smooth"
      refresh_mix="ema"
      mode_label="ema_fast"
      ema_decay="$ema_fast_decay"
      ;;
    *)
      echo "unsupported refresh mode: $mode" >&2
      exit 2
      ;;
  esac

  echo "=== TinyStories AdamW-LRSF refresh=${mode_label} interval=${interval} window=${window} ==="
  TEXT_LM_OPTIMIZER_LONG_RANK_DIR="$output_root/$mode_label" \
    TEXT_LM_REFRESH_MODE="$refresh_mode" \
    TEXT_LM_REFRESH_INTERVAL="$interval" \
    TEXT_LM_REFRESH_WINDOW="$window" \
    TEXT_LM_REFRESH_MIX="$refresh_mix" \
    TEXT_LM_REFRESH_EMA_DECAY="$ema_decay" \
    ./benchmarks/launchers/run_text_lm_optimizer_long_rank_comparison.sh
done
