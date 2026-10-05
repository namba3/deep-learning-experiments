# Development Setup and Preflight Checks

[English](development-and-validation.en.md) | [日本語](development-and-validation.md)

Unless noted otherwise, run commands from the repository root.

## Preflight checks

The five standard trainers—`mnist`, `cifar10`, `text_lm`, `image_ae`, and `image_gen`—provide the same preflight modes. `vfp_dit` and `mini_imagenet_gqa` also provide package-specific preflight modes; see their READMEs for details.

- `--dry-run`: checks arguments, device, dtype, and resume settings without loading the dataset or model.
- `--validate-only`: loads the dataset and model, checks sample count, shapes, parameter count, and finite values, then exits without training.

Lightweight checks:

```bash
PYTHONPATH=. python3 -m mnist.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m cifar10.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m text_lm.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m image_ae.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m image_gen.train --dry-run --vae-model Qwen/Qwen-Image
```

Examples that also check real data and model setup:

```bash
PYTHONPATH=. python3 -m mnist.train --validate-only --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m cifar10.train --validate-only --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m image_ae.train --validate-only --dataset cifar10 --data-dir cifar10/data --epochs 1 --batch-size 2 --num-workers 0
PYTHONPATH=. python3 -m image_gen.train --validate-only --vae-model Qwen/Qwen-Image --epochs 1 --batch-size 1 --num-workers 0
```

`--validate-only` loads real data, tokenizers, VAE weights, and model weights; it may download assets that are not already available. The two modes cannot be used together. Results are written as `preflight` or `validation` events to `<output-dir>/runs/<run-id>/metrics.jsonl`.

## Development and validation

This repository does not define a fixed supported Python version range. Choose a Python environment supported by the PyTorch release and the OS/CPU/CUDA/ROCm setup you intend to use. Check PyTorch's [official install selector](https://pytorch.org/get-started/locally/) and install the matching `torch` and `torchvision` pair.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

In Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

In the activated environment, run the PyTorch/torchvision install command shown by the official selector.

After installing the selected PyTorch/torchvision wheels, install the repository runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists direct dependencies shared across the experiments. It is not a minimal per-experiment install or a version-pinned lockfile. `requirements-dev.txt` includes these runtime dependencies and the development/verification tools.

`datasets`, `diffusers`, `huggingface-hub`, and `transformers` are mainly used for external dataset/model integration. `aptx-activation`, `came-pytorch`, `muon-optimizer`, and `schedulefree` are used by their corresponding activation/optimizer implementations.

```bash
python -m pip install -r requirements-dev.txt
```

Triton GPU kernels and NVML telemetry are optional dependencies.

Standard checks:

```bash
PYTHONPATH=. python3 -m pytest -q
bash scripts/compile_python.sh
python3 -m pyright
ruff check .
git diff --check
```

Checks that require CUDA, Triton, or long-running training should be run separately from the standard unit tests. CPU validation does not guarantee CUDA or Triton behavior.

See [`tests/README.md`](../../tests/README.md) for the unit/integration split and commands to run each suite. GitHub Actions runs the CPU suite, compileall, Pyright, and Ruff as required checks.

The `compileall` target directories are maintained in [`scripts/compile_python.sh`](../../scripts/compile_python.sh). This checks syntax; it does not validate imports or runtime behavior.

To check the real VAE's shapes, strides, and encode/decode path, use the runtime check in `verify/`:

```bash
python3 -m verify.qwen_vae --vae-model Qwen/Qwen-Image --vae-dtype fp32
```
