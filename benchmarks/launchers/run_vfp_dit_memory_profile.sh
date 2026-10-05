#!/usr/bin/env bash
set -euo pipefail

# Short, full-width VFP-DiT resource probe. Every setting can still be
# overridden by the caller, while defaults keep the screen bounded and small.
resolution="${VFP_DIT_RESOLUTION:-256}"
export VFP_DIT_RESOLUTION="$resolution"
export VFP_DIT_RUN_NAME="${VFP_DIT_RUN_NAME:-memory-profile-${resolution}px-1t-$(date -u +%Y%m%dT%H%M%SZ)}"
export VFP_DIT_TIMESTEPS_PER_IMAGE="${VFP_DIT_TIMESTEPS_PER_IMAGE:-1}"
export VFP_DIT_EPOCHS="${VFP_DIT_EPOCHS:-1}"
export VFP_DIT_SAMPLES_PER_EPOCH="${VFP_DIT_SAMPLES_PER_EPOCH:-8}"
export VFP_DIT_VALIDATION_FRACTION="${VFP_DIT_VALIDATION_FRACTION:-0}"
export VFP_DIT_PROFILE_COMPONENTS="${VFP_DIT_PROFILE_COMPONENTS:-1}"
export VFP_DIT_ENCODER_DEVICE="${VFP_DIT_ENCODER_DEVICE:-cpu}"
export VFP_DIT_ENCODER_PREFETCH_BATCHES="${VFP_DIT_ENCODER_PREFETCH_BATCHES:-1}"
export VFP_DIT_GC_INTERVAL="${VFP_DIT_GC_INTERVAL:-100}"
export VFP_DIT_EMPTY_CACHE_INTERVAL="${VFP_DIT_EMPTY_CACHE_INTERVAL:-0}"
export VFP_DIT_OBSERVE_INTERVAL="${VFP_DIT_OBSERVE_INTERVAL:-0}"
export VFP_DIT_GENERATE_SAMPLES="${VFP_DIT_GENERATE_SAMPLES:-0}"
export VFP_DIT_NO_OBSERVE_SAMPLES="${VFP_DIT_NO_OBSERVE_SAMPLES:-1}"
export VFP_DIT_MODEL_WIDTH="${VFP_DIT_MODEL_WIDTH:-1024}"
export VFP_DIT_DEPTH="${VFP_DIT_DEPTH:-24}"
export VFP_DIT_HEADS="${VFP_DIT_HEADS:-16}"
export VFP_DIT_KV_HEADS="${VFP_DIT_KV_HEADS:-4}"

exec bash benchmarks/launchers/run_vfp_dit_screen.sh
