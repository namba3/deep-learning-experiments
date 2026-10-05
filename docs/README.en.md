# Documentation

[English](README.en.md) | [日本語](README.md)

This index covers usage guides, implementation contracts, research records, and maintenance notes. For the repository overview and minimal examples, see the [root README](../README.en.md).

## Usage and validation

- [Root README](../README.en.md): project purpose, directory overview, setup, and minimal examples
- [VFP-DiT README](../vfp_dit/README.md): training, sampling, and checkpoint behavior
- [Development setup and validation](guides/development-and-validation.en.md): environment setup, preflight, tests, and static checks
- [VFP-DiT training operations (English)](guides/vfp-dit-operations.en.md): encoder setup, memory measurement, comparison launchers, diagnostics, and resume workflows
- [VFP-DiT shared runtime](../vfp_dit_runtime/README.md): shared Qwen, data, and training runtime used by VFP-DiT
- [Shared training runtime](../runtime/README.md): common checkpoint, sampler, progress, and preflight utilities
- [Benchmarks and experiment launchers](../benchmarks/README.md): reusable benchmarks and research launcher index
- [Runtime verification](../verify/README.md): CPU checks and runtime checks requiring external models or CUDA
- [Mini-ImageNet GQA experiments](../mini_imagenet_gqa/README.md): experiment setup and reproduction steps
- [Dataset and model provenance](data-model-provenance.md): sources, terms, and unresolved rights questions for external datasets and models
- [Experiment results archive (English)](experiment_data/README.en.md) | [日本語](experiment_data/README.md): aggregate reports and the scope of deleted raw run artifacts; individual reports use different languages
- Adapter experiment records (mixed Japanese and English): [CIFAR-10](adapter-experiments/cifar10.md), [ImageAE](adapter-experiments/image-ae.md), [Text-LM](adapter-experiments/text-lm.md), and [TinyImageNet-200](adapter-experiments/tiny-imagenet.md)

## Implementation contracts and audits

- [Image-latent DiT architecture (English)](architecture/image-latent-dit.en.md) | [日本語](architecture/image-latent-dit.md): shape, mask, training, sampling, and checkpoint contracts
- [VFP-DiT architecture](architecture/vfp-dit.md): Qwen/VAE conditioning, latent fusion, DiT, Ada, and checkpoint compatibility
- [VFP-DiT migration note](vfp-dit-migration.md): current entrypoints, legacy run outputs, and checkpoint-stage compatibility
- [Learning-rate schedulers](lr-schedulers.md): shared scheduler and warmup behavior
- [Optimizers (Japanese)](optimizers.md): optimizer registry, variants, fallbacks, and state design

## Research and comparison records

These documents record hypotheses, settings, and results from the time of each experiment. For current implementation behavior, use the contracts above and the relevant package README.

### VFP-DiT

- [VFP-DiT documentation index](VFP-DiT_Research_Note.md): links to current usage, migration notes, and research history
- Aggregate results from retired implementations and historical short resource probes are in the [experiment archive](experiment_data/README.md)

### Optimizers and low-rank methods

- [APOLLO research and experiment records](apollo-experiment-records.md): overview, hypotheses, aggregate results, and historical protocols
- [LRTDO research summary (English)](history/lrtdo-research-summary.en.md) | [日本語](history/lrtdo-research-summary.md): completed low-rank trajectory experiments and links to detailed aggregate records
- [Schedule-Free methods and related records (Japanese)](schedule-free-methods-summary.md): Mini-ImageNet quantization results and links to current optimizer documentation
- [Low-rank Schedule-Free design (Japanese)](low-rank-schedule-free-design.md): equations, terminology, state, train/eval, and checkpoint contract snapshot from 2026-09-13
- [Low-rank adapters](low-rank-adapters.md): equations, implementation contracts, and validation considerations
- [Low-rank adapter experiment records](low-rank-adapters-results.md): historical experiment conditions and aggregate results
- [RGLU-LoRA](rglu-lora.md)

### Sampling solvers and dataset comparisons

- [Flow-sampling solver index](sampling-solvers.md): solver methods, implementation status, and validation scope
- [Mini-ImageNet GQA comparison design](mini-imagenet-gqa-comparison.md): comparison design and CPU validation record
- [Mini-ImageNet GQA results](mini-imagenet-gqa-results.md): completed screens and optimizer follow-up measurements
- [Mini-ImageNet GQA bucket analysis](experiment_data/mini-imagenet-gqa-bucket-analysis.md): dataset resolution summary and aspect-bucket proposal
- [Text-LM pretraining comparison](text-lm-pretraining-comparison.md)

### Historical snapshots

- [History index (English)](history/README.en.md) | [日本語](history/README.md): dated snapshots, their scope, and how to interpret them

## Document locations and interpretation

| Location | Purpose |
| --- | --- |
| Root `README.md` | Project purpose, structure, setup, and minimal examples |
| `*/README.md` | Package responsibilities, public entrypoints, CLIs, and related documents |
| `docs/architecture/` | Implementation contracts shared across packages |
| `docs/guides/` | Usage and operations guides for specific features |
| `docs/` | Audits, performance records, research comparisons, and publication maintenance notes |
| `docs/adapter-experiments/` | Dataset-specific adapter experiment records |
| `docs/experiment_data/` | Aggregate historical reports; step-level metrics and raw run artifacts are not retained in the public tree |
| `verify/` | Runtime validation requiring external models or CUDA, separate from pytest |
| `AGENTS.md` | Repository development and validation rules |

Code and CLI `--help` output define current behavior. Research and comparison records describe hypotheses and results from their stated dates and conditions. CPU or static checks do not verify CUDA, Triton, or external-model behavior. Language labels identify Japanese-only or mixed-language pages; paired English documents are linked where available.

Paths under `output/` in experiment records refer to local generated files or former output locations. Raw run artifacts, checkpoints, and generated images are not included in the public tree; retained reports contain aggregate values and conditions for reuse.
