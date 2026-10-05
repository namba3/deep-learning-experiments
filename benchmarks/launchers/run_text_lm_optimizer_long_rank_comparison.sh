#!/usr/bin/env bash
set -euo pipefail

# Long TinyStories comparison for the practical LRSF ranks.
# This delegates to the rank sweep so each rank still receives an independent
# artifact and the same backend/seed contract as the short probe.

output_dir="${TEXT_LM_OPTIMIZER_LONG_RANK_DIR:-output/text-lm-optimizer-long-rank}"
ranks="${TEXT_LM_LONG_RANKS:-8,16}"
epochs="${TEXT_LM_LONG_EPOCHS:-10}"
steps_per_epoch="${TEXT_LM_LONG_STEPS_PER_EPOCH:-100}"
learning_rate="${TEXT_LM_LEARNING_RATE:-1e-3}"
adamw_sf_backend="${TEXT_LM_LONG_ADAMW_SF_BACKEND:-auto}"

TEXT_LM_OPTIMIZER_RANK_SWEEP_DIR="$output_dir" \
  TEXT_LM_RANKS="$ranks" \
  TEXT_LM_EPOCHS="$epochs" \
  TEXT_LM_STEPS_PER_EPOCH="$steps_per_epoch" \
  TEXT_LM_LEARNING_RATE="$learning_rate" \
  TEXT_LM_ADAMW_SF_BACKEND="$adamw_sf_backend" \
  ./benchmarks/launchers/run_text_lm_optimizer_rank_sweep.sh
