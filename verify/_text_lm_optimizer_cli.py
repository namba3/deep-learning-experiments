"""Command-line interface for the Text LM optimizer convergence probe."""

from __future__ import annotations

import argparse

SUPPORTED_OPTIMIZERS = (
    "AdamW",
    "AdamW-SF",
    "AdamW-LRSF",
    "AdamW-SF-LR",
    "AdamW-LRSF-LR",
    "AdamW-LR-EMA",
    "AdamW-LR-EMA-Conf",
    "AdamW-LR-EMA-Conf-LRSF",
    "CAME",
    "CAME-SF",
    "CAME-LRSF",
    "APOLLO",
    "APOLLO-Conf",
    "APOLLO-CAME",
    "APOLLO-CAME-LRSF",
    "APOLLO-SF",
    "APOLLO-SF-INT8-Z",
    "APOLLO-SF-INT8-Delta",
    "APOLLO-SF-INT4-Z",
    "APOLLO-SF-INT4-Delta",
)
OPTIMIZERS = ("AdamW", "AdamW-SF", "AdamW-LRSF", "CAME")


def _parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or any(item not in SUPPORTED_OPTIMIZERS for item in values):
        raise argparse.ArgumentTypeError(
            "optimizers must be a comma-separated subset of: "
            + ",".join(SUPPORTED_OPTIMIZERS)
        )
    return values


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    return values


def _parse_positive_int_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated positive integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "values must be comma-separated positive integers"
        )
    return values



def _validate_args(args) -> None:
    if args.seed < 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("seed must be non-negative; epochs and batch-size must be positive")
    if args.seeds is not None and any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be non-negative")
    if args.train_tokens <= 0 or args.eval_tokens <= 0:
        raise ValueError("train-tokens and eval-tokens must be positive")
    if args.eval_interval < 0:
        raise ValueError("eval-interval must be non-negative")
    if args.max_seq_len <= 1 or args.embed_dim <= 0 or args.num_layers <= 0:
        raise ValueError("model dimensions must be positive")
    if args.num_heads <= 0 or args.embed_dim % args.num_heads != 0:
        raise ValueError("embed-dim must be divisible by num-heads")
    if args.kv_heads <= 0 or args.num_heads % args.kv_heads != 0:
        raise ValueError("num-heads must be divisible by kv-heads")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if not 0.0 <= args.adamw_lr_ema_beta < 1.0:
        raise ValueError("lr-ema-beta must be in [0, 1)")
    if not 0.0 <= args.adamw_lr_ema_confidence_beta < 1.0:
        raise ValueError("lr-ema-confidence-beta must be in [0, 1)")
    if args.adamw_lr_ema_confidence_alpha < 0.0:
        raise ValueError("lr-ema-confidence-alpha must be non-negative")
    if args.apollo_norm_growth_rate <= 1.0:
        raise ValueError("apollo-norm-growth-rate must be greater than 1")
    if args.apollo_scale <= 0.0:
        raise ValueError("apollo-scale must be positive")
    if args.rank <= 0 or args.num_workers < 0 or args.grad_clip <= 0:
        raise ValueError("rank, num-workers, and grad-clip must be valid")
    if not args.update_reconstruction_ranks or any(
        rank <= 0 for rank in args.update_reconstruction_ranks
    ):
        raise ValueError("update-reconstruction-ranks must be positive")
    if args.refresh_interval < 0 or args.refresh_window < 0 or args.orthogonal_rate < 0:
        raise ValueError("refresh values must be non-negative")
    if args.refresh_mode != "none" and args.refresh_interval <= 0:
        raise ValueError("refresh-interval must be positive when refresh is enabled")
    if args.refresh_mode == "smooth" and args.refresh_window <= 0:
        raise ValueError("refresh-window must be positive for smooth refresh")
    if (
        args.lrsf_refresh_state != "transport"
        and "AdamW-LRSF" in args.optimizers
    ):
        raise ValueError(
            "lrsf-refresh-state reset is supported only by AdamW-LRSF-LR; "
            "AdamW-LRSF uses transport implicitly"
        )
    if args.refresh_ema_decay is not None and not 0.0 <= args.refresh_ema_decay < 1.0:
        raise ValueError("refresh-ema-decay must be in [0, 1)")
    if args.refresh_transport_overlap is not None and not 0.0 <= args.refresh_transport_overlap <= 1.0:
        raise ValueError("refresh-transport-overlap must be in [0, 1]")
    if not 0.0 < args.trajectory_pca_calibration_fraction < 1.0:
        raise ValueError("trajectory-pca-calibration-fraction must be in (0, 1)")
    if args.trajectory_pca_rolling_window < 0:
        raise ValueError("trajectory-pca-rolling-window must be non-negative")
    if args.residual_compression_block_size <= 0:
        raise ValueError("residual-compression-block-size must be positive")
    if not 0.0 <= args.state_rank_delta_ema_decay < 1.0:
        raise ValueError("state-rank-delta-ema-decay must be in [0, 1)")
    if (
        args.state_rank_interval <= 0
        or args.state_rank_max_elements <= 0
        or args.state_rank_max_tensors <= 0
        or args.state_trajectory_max_snapshots <= 0
    ):
        raise ValueError(
            "state-rank and trajectory-PCA limits must be positive"
        )

def parse_args(argv=None, *, description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=_parse_seeds, default=None)
    parser.add_argument("--train-tokens", type=int, default=1_000_000)
    parser.add_argument("--eval-tokens", type=int, default=100_000)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument(
        "--eval-interval", type=int, default=0,
        help=(
            "Run validation every N optimizer steps for trajectory loss "
            "diagnostics; 0 disables periodic validation."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--transform-rank", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--lr-ema-beta", dest="adamw_lr_ema_beta", type=float, default=0.9,
        help="EMA decay for AdamW-LR-EMA projected gradients (default: 0.9).",
    )
    parser.add_argument(
        "--lr-ema-projection-scale",
        dest="adamw_lr_ema_projection_scale",
        choices=("norm", "none"),
        default="norm",
        help="Projected-gradient scale correction (default: norm).",
    )
    parser.add_argument(
        "--lr-ema-confidence-beta",
        dest="adamw_lr_ema_confidence_beta",
        type=float,
        default=0.99,
        help="EMA decay for projected-gradient innovation variance.",
    )
    parser.add_argument(
        "--lr-ema-confidence-alpha",
        dest="adamw_lr_ema_confidence_alpha",
        type=float,
        default=1e-3,
        help="Mean-square floor coefficient for confidence normalization.",
    )
    parser.add_argument("--apollo-norm-growth-rate", type=float, default=1.01)
    parser.add_argument("--apollo-scale", type=float, default=1.0)
    parser.add_argument(
        "--apollo-sf-quant-block-size", type=int, default=256,
        help="Block size for APOLLO-SF INT8/INT4 state storage.",
    )
    parser.add_argument(
        "--apollo-disable-norm-growth-limiter",
        action="store_true",
        help="Disable APOLLO's per-parameter norm-growth limiter.",
    )
    parser.add_argument("--lrsf-beta1", type=float, default=0.9)
    parser.add_argument("--lrsf-warmup-steps", type=int, default=0)
    parser.add_argument("--lrsf-r", type=float, default=0.0)
    parser.add_argument("--lrsf-weight-lr-power", type=float, default=2.0)
    parser.add_argument(
        "--adamw-sf-backend", choices=("torch", "triton", "auto"),
        default="torch",
        help=(
            "Backend for AdamW-SF. The default torch backend keeps the SF "
            "and LRSF rank-oracle comparison numerically comparable."
        ),
    )
    parser.add_argument(
        "--refresh-mode", choices=("none", "hard", "smooth", "shadow"), default="hard",
        help=(
            "LRSF projection refresh mode. The text-LM default is hard for "
            "long-training trajectory coverage; use none for frozen control. "
            "shadow keeps a trained low-rank branch ready for promotion."
        ),
    )
    parser.add_argument("--refresh-interval", type=int, default=200)
    parser.add_argument(
        "--lrsf-refresh-state", choices=("reset", "transport"),
        default="transport",
        help=(
            "Latent second-moment handling for AdamW-LRSF-LR hard refresh. "
            "Transport is the legacy approximation; reset also resets its "
            "local bias-correction age."
        ),
    )
    parser.add_argument("--refresh-window", type=int, default=200)
    parser.add_argument(
        "--refresh-mix", choices=("linear", "smoothstep", "stochastic", "ema"),
        default="smoothstep",
    )
    parser.add_argument(
        "--refresh-ema-decay", type=float, default=None,
        help="Override EMA refresh decay; smaller values make the transition steeper.",
    )
    parser.add_argument(
        "--record-refresh-diagnostics", action="store_true",
        help=(
            "Record decoded-delta distortion at projection refresh events. "
            "This adds temporary full-matrix diagnostics and is not a timing run."
        ),
    )
    parser.add_argument(
        "--refresh-transport-overlap", type=float, default=None,
        help=(
            "Blend old and new refresh bases before transport: 1 keeps the "
            "old basis, 0 uses the fully random basis."
        ),
    )
    parser.add_argument("--orthogonal-rate", type=float, default=0.0)
    parser.add_argument(
        "--orthogonal-direction",
        choices=("random", "loss_directed", "loss_lowering"),
        default="random",
    )
    parser.add_argument(
        "--orthogonal-signal", choices=("gradient", "effective_update"),
        default="gradient",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizers", type=_parse_csv, default=OPTIMIZERS)
    parser.add_argument("--record-update-norms", action="store_true")
    parser.add_argument(
        "--record-confidence-diagnostics", action="store_true",
        help=(
            "Record confidence and innovation statistics for "
            "AdamW-LR-EMA-Conf."
        ),
    )
    parser.add_argument(
        "--record-update-trajectory-pca", action="store_true",
        help=(
            "Record temporal PCA diagnostics for effective parameter updates. "
            "This is a diagnostic-only, sampled CPU copy."
        ),
    )
    parser.add_argument(
        "--trajectory-pca-calibration-fraction", type=float, default=0.5,
        help=(
            "Fraction of sampled trajectory updates used to fit the causal "
            "PCA basis (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--trajectory-pca-rolling-window", type=int, default=0,
        help=(
            "Number of preceding trajectory snapshots used for rolling PCA; "
            "0 disables rolling PCA (default: 0)."
        ),
    )
    parser.add_argument(
        "--record-fixed-basis-residual-approximation", action="store_true",
        help=(
            "Also fit one residual-PCA basis from the initial rolling window "
            "and reuse it for later approximation diagnostics."
        ),
    )
    parser.add_argument(
        "--residual-compression-block-size", type=int, default=256,
        help=(
            "Block size for residual INT8/INT4 diagnostics "
            "(default: 256)."
        ),
    )
    parser.add_argument(
        "--residual-compression-scale-mode",
        choices=("max_abs", "percentile_99_9", "rms_3sigma"),
        default="max_abs",
        help=(
            "Scale mode for blockwise residual INT8/INT4 diagnostics "
            "(default: max_abs)."
        ),
    )
    parser.add_argument(
        "--record-trajectory-curvature", action="store_true",
        help=(
            "Record turning-angle and roughness diagnostics for sampled "
            "effective parameter updates. For Schedule-Free optimizers, also "
            "measure train/hidden/eval parameter positions."
        ),
    )
    parser.add_argument(
        "--record-update-reconstruction", action="store_true",
        help=(
            "Measure rank-2/4/8 state replacements against the realized "
            "Schedule-Free update; diagnostic-only."
        ),
    )
    parser.add_argument(
        "--update-reconstruction-ranks", type=_parse_positive_int_csv,
        default=(2, 4, 8),
        help="Comma-separated reconstruction ranks (default: 2,4,8).",
    )
    parser.add_argument(
        "--record-state-rank", action="store_true",
        help=(
            "Record per-state singular-spectrum diagnostics for eligible "
            "matrix parameters."
        ),
    )
    parser.add_argument(
        "--state-rank-interval", type=int, default=100,
        help="Steps between state-rank snapshots.",
    )
    parser.add_argument(
        "--state-rank-max-elements", type=int, default=2_000_000,
        help="Skip matrices larger than this many elements during SVD.",
    )
    parser.add_argument(
        "--state-rank-max-tensors", type=int, default=4,
        help="Maximum number of eligible parameter tensors per case.",
    )
    parser.add_argument(
        "--state-rank-parameter", default=None,
        help="Optional substring used to select state-rank parameter names.",
    )
    parser.add_argument(
        "--state-rank-delta-ema-decay", type=float, default=0.9,
        help="EMA decay for the diagnostic Schedule-Free delta rank.",
    )
    parser.add_argument(
        "--record-state-trajectory-pca", action="store_true",
        help=(
            "Record temporal PCA diagnostics for full-rank optimizer states; "
            "APOLLO latent states are excluded."
        ),
    )
    parser.add_argument(
        "--state-trajectory-max-snapshots", type=int, default=16,
        help="Maximum snapshots retained per parameter/state source for PCA.",
    )
    return parser.parse_args(argv)
