#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
source_dirs=(
  benchmarks
  cifar10
  core
  flow_sampling
  image_ae
  image_gen
  mini_imagenet_gqa
  mnist
  optimizers
  runtime
  tests
  text_lm
  verify
  vfp_dit
  vfp_dit_runtime
)

"$python_bin" -m compileall -q "${source_dirs[@]}"
