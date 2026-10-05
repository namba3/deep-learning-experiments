# Performance review archive (2026-09-12)

This index groups the dated performance review by topic. Detailed measurements and implementation notes remain in the linked snapshots. They describe the 2026-09 review state and are not current performance guarantees or work plans.

- [DiT and LLM Adapter performance review](dit-adapter-performance-review-2026-09-12.md): configuration, attention and adapter measurements, optimizer profiling, ImageAE comparisons, and unverified GPU paths.
- [Optimizer, Scheduler, and AutoSchedule review](optimizer-review-2026-09-12.md): optimizer state, update cost, checkpoint, and AutoSchedule findings.
- [Text-LM architecture benchmarks](text-lm-architecture-benchmark-2026-09-12.md): synthetic and Alpaca subset measurements, including CPU-only short runs.

## Core and shared layers

The review recorded a split of shared layers into responsibility-based packages while preserving `core.layers` and pickle paths. It also recorded shared pre-norm convolutional FFN residual logic for ImageAE and image generation. The original package export, strict-load, rectangular-shape, backward, and identity checks are listed in [test_core_layer_package.py](../../tests/unit/test_core_layer_package.py) and [test_core_conv_ffn.py](../../tests/unit/test_core_conv_ffn.py). See [core README](../../core/README.md) for the current layout.

For current behavior, use package READMEs, code, and the current verification guidance. Historical measurements retain their original conditions and limitations.
