# AI and Deep Learning Experiments

[English](README.en.md) | [日本語](README.md)

This repository is for studying AI and deep learning by implementing and evaluating model architectures and training methods with PyTorch. Its topics include image models, autoencoders, Transformers, and optimizers.

This README is the entry point to the repository. For experiment-specific CLIs and model details, see the README files in each package and the [`docs/`](docs/) directory.

## Experiment areas

### Core models and training methods

- [`mnist/`](mnist/README.md): MNIST classification and a basic training loop
- [`cifar10/`](cifar10/README.md): CIFAR-10 classification and adapter experiments
- [`text_lm/`](text_lm/README.md): text Transformers
- [`optimizers/`](optimizers/README.md): optimizers and learning-rate scheduling

### Model architectures and image models

- [`mini_imagenet_gqa/`](mini_imagenet_gqa/README.md): GQA/FFN architecture comparisons
- [`image_ae/`](image_ae/README.md): image autoencoders
- [`image_gen/`](image_gen/README.md): image-latent DiT/MMDiT with Qwen VAE and Qwen3.5
- [`vfp_dit/`](vfp_dit/README.md): conditional VFP-DiT

### Shared implementation and validation

- [`core/`](core/README.md), [`runtime/`](runtime/README.md), [`vfp_dit_runtime/`](vfp_dit_runtime/README.md): shared model and training runtime
- [`flow_sampling/`](flow_sampling/README.md): flow-matching schedules and sampling solvers
- [`benchmarks/`](benchmarks/README.md), [`tests/`](tests/README.md), [`verify/`](verify/README.md): performance measurement and validation
- [`docs/`](docs/README.en.md): design, audits, performance, and research records

## Examples

### Lightweight MNIST CPU preflight

This checks the arguments and device configuration without loading a dataset.

```bash
PYTHONPATH=. python3 -m mnist.train --dry-run --epochs 1 --batch-size 2 --num-workers 0
```

### Architecture comparison: Mini-ImageNet GQA

This comparison uses QK RMSNorm across variants and compares naive versus gated GQA and plain versus gated FFNs. See [`mini_imagenet_gqa/README.md`](mini_imagenet_gqa/README.md) for augmentation, variant definitions, and the seed protocol.

```bash
PYTHONPATH=. python3 -m mini_imagenet_gqa.train --dry-run --device cpu
```

### Image autoencoder

The Flickr30k examples below download images. The images remain copyrighted by their respective owners, and their use and redistribution are subject to Flickr's terms. Review the [dataset and model provenance notes](docs/data-model-provenance.md#datasets) before running them. This repository's code license does not apply to the dataset.

```bash
python3 -m image_ae.train --dataset flickr30k --batch-size 8
```

### Image-latent DiT

`--vae-model` is required. To use a dataset:

The following example downloads images from a Flickr30k mirror on Hugging Face. The mirror does not state a license for redistributing the images. The Flickr30k source limits its image distribution to non-commercial research and education and points users to Flickr's terms. Check the [provenance notes](docs/data-model-provenance.md#datasets) and the terms for each image before sharing data or generated artifacts.

```bash
python3 image_gen/train.py \
  --dataset-name lmms-lab-encoder/flickr30k \
  --vae-model Qwen/Qwen-Image
```

To use a local JSONL or CSV file:

```bash
python3 image_gen/train.py \
  --records data.jsonl \
  --vae-model Qwen/Qwen-Image
```

### VFP-DiT

The current VFCB-free architecture uses Qwen3.5 hidden states, optional Qwen-Image reference latents, a compressed conditional DiT front stage, and full-resolution output refinement. See [`vfp_dit/README.md`](vfp_dit/README.md) for architecture and training details.

```bash
python3 -m vfp_dit.train --help
python3 -m vfp_dit.train --dry-run --device cpu
```

### Generate samples

To generate samples from a trained checkpoint:

```bash
python3 image_gen/generate_samples.py \
  --checkpoint image_gen/output/<run>/checkpoint_latest.safetensors \
  --vae-model Qwen/Qwen-Image \
  --prompt "a photograph of a child playing with a dog outdoors"
```

Use each script's `--help` output as the authoritative list of options.

## Development and validation

For environment setup, preflight checks, tests, and static checks, see the [development setup and validation guide](docs/guides/development-and-validation.en.md).

## Documentation

- [Development setup and validation](docs/guides/development-and-validation.en.md): environments, dependencies, preflight, tests, and static checks
- [`docs/README.en.md`](docs/README.en.md): documentation categories and index
- [`docs/data-model-provenance.md`](docs/data-model-provenance.md): sources and terms for external datasets and models

Package guides for entry points and responsibilities:

| Category | Guides |
| --- | --- |
| Experiments | [`image_gen`](image_gen/README.md), [`vfp_dit`](vfp_dit/README.md), [`image_ae`](image_ae/README.md), [`cifar10`](cifar10/README.md), [`mini_imagenet_gqa`](mini_imagenet_gqa/README.md), [`mnist`](mnist/README.md), [`text_lm`](text_lm/README.md) |
| Shared runtime, model, and sampling | [`core`](core/README.md), [`runtime`](runtime/README.md), [`vfp_dit_runtime`](vfp_dit_runtime/README.md), [`optimizers`](optimizers/README.md), [`flow_sampling`](flow_sampling/README.md) |
| Verification and integrations | [`benchmarks`](benchmarks/README.md), [`verify`](verify/README.md) |

Design, audit, and research records:

- [`docs/architecture/image-latent-dit.md`](docs/architecture/image-latent-dit.md): architecture, shapes, training objectives, and checkpoint contract for the current image-latent DiT
- [`docs/VFP-DiT_Research_Note.md`](docs/VFP-DiT_Research_Note.md): index to current usage, migration details, and research history
- [`docs/optimizers.md`](docs/optimizers.md): optimizer categories, selection, low-rank state, and AutoSchedule
- [`docs/low-rank-adapters.md`](docs/low-rank-adapters.md): LoRA, DoRA, and LoHA target layers, checkpoints, and validation policy
- [`docs/rglu-lora.md`](docs/rglu-lora.md): GLU-LoRA / Residual GLU-LoRA design hypotheses and experiment plan
- [`docs/lr-schedulers.md`](docs/lr-schedulers.md): shared learning-rate schedulers and warmup for training scripts
- For historical snapshots, use the [history index](docs/history/README.md).
- [`AGENTS.md`](AGENTS.md): repository development guidelines

## License

The project code and documentation in this repository are dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your choice. The MIT copyright identifies [GitHub user @namba3](https://github.com/namba3). Third-party code and assets may have separate terms; see the [provenance inventory](docs/data-model-provenance.md) for dataset and model terms.

## Notes

- Training large models requires sufficient VRAM.
- `image_gen` trains with BF16 parameters by default and freezes the VAE and text encoder.
- `--gradient-checkpointing` and `--mhla-recompute-output` reduce activation VRAM at the cost of additional recomputation.
- `--perf` options add measurement overhead and should be kept separate from normal speed comparisons.
- If checkpoint architecture metadata does not match the CLI configuration, review the configuration before resuming.
