"""Command-line configuration for image-latent training."""

import argparse

from runtime.device import add_device_argument
from optimizers.factory import add_optimizer_argument, add_optimizer_override_argument
from optimizers.lr_scheduler import add_lr_scheduler_arguments

DEFAULT_TEXT_MODEL = "Qwen/Qwen3.5-0.8B"

DEFAULT_DATASET_NAME = "lmms-lab-encoder/flickr30k"

def parse_int_tuple(value):
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values

def parse_args():
    parser = argparse.ArgumentParser(description="Train a caption-conditioned image-latent DiT")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME,
                        help=f"Hugging Face dataset name (default: {DEFAULT_DATASET_NAME})")
    parser.add_argument("--records", default=None, help="Local JSONL/CSV records file")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--vae-model", required=True)
    parser.add_argument(
        "--vae-dtype", choices=["bf16", "fp32"], default="bf16",
        help="VAE parameter dtype. Default: bf16; use fp32 for compatibility.",
    )
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL,
                        help=f"Text Encoder model (default: {DEFAULT_TEXT_MODEL})")
    parser.add_argument("--text-adapter-dim", type=int, default=1024,
                        help="Token-wise Text Adapter output width. Default: 1024.")
    parser.add_argument(
        "--text-adapter-transformer-dims", type=parse_int_tuple,
        default=(2048, 1024, 1024), metavar="DIM[,DIM...]",
        help="Non-causal text transformer widths. Default: 2048,1024,1024.",
    )
    parser.add_argument(
        "--text-adapter-transformer-heads", type=parse_int_tuple,
        default=(16, 8, 8), metavar="HEADS[,HEADS...]",
        help="Attention heads for each text transformer layer. Default: 16,8,8.",
    )
    parser.add_argument(
        "--text-adapter-transformer-kv-heads", type=parse_int_tuple,
        default=(8, 4, 4), metavar="HEADS[,HEADS...]",
        help=(
            "K/V heads for each text transformer layer; must divide Q heads. "
            "Default: 8,4,4."
        ),
    )
    parser.add_argument(
        "--text-adapter-transformer-ff-mult", type=float, default=3.0,
        help="FFN width multiplier for each text transformer layer. Default: 3.0.",
    )
    parser.add_argument(
        "--text-adapter-rope-theta", type=float, default=10000.0,
        help="Base theta for 1D text Adapter RoPE. Default: 10000.",
    )
    parser.add_argument("--latent-channels", type=int, default=None,
                        help="Expected VAE latent channels. Default: infer from the loaded VAE.")
    parser.add_argument("--output-dir", default="image_gen/output")
    parser.add_argument(
        "--run-name", default=None,
        help="Optional human-readable name included in the run directory.",
    )
    add_device_argument(parser)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="共通設定を検証し、データやモデルを読み込まずに終了",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="データとモデルの構成を検証し、学習せずに終了",
    )
    parser.add_argument(
        "--no-artifacts", action="store_true",
        help=(
            "Do not save safetensors checkpoints or sampling images. "
            "Performance JSONL and TensorBoard logs remain enabled."
        ),
    )
    parser.add_argument("--tensorboard-dir", default=None,
                        help="TensorBoard root directory. Default: sibling of --output-dir.")
    parser.add_argument(
        "--resume", default=None,
        help=(
            "Load model weights. If a sibling .resume.pt file exists, also "
            "restore optimizer, scheduler, and RNG state; otherwise use "
            "weights-only resume."
        ),
    )
    parser.add_argument("--init-checkpoint", default=None,
                        help="Initialize matching weights from a checkpoint with a different architecture.")
    parser.add_argument("--init-freeze-steps", type=int, default=1000,
                        help="Freeze transferred parameters for this many optimizer steps. Default: 1000.")
    parser.add_argument("--resume-epoch", type=int, default=0,
                        help="Completed epochs when the checkpoint has no usable epoch metadata.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bucket-step", type=int, default=32,
                        help="Image-size alignment for aspect-ratio buckets; must suit the VAE stride.")
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument(
        "--kv-heads", type=int, default=8,
        help="K/V heads in the main MMDiT; must divide --heads. Default: 8.",
    )
    parser.add_argument("--context-depth", type=int, default=2,
                        help="Number of Image/Text Context Transformer blocks. Default: 2.")
    parser.add_argument("--context-heads", type=int, default=16,
                        help="Attention heads in the Context Transformer. Default: 16.")
    parser.add_argument(
        "--context-kv-heads", type=int, default=8,
        help=(
            "K/V heads in the Context Transformer; must divide --context-heads. "
            "Default: 8."
        ),
    )
    parser.add_argument(
        "--attention-gate", choices=["none", "head"], default="head",
        help=(
            "SDPA output gating mode. 'head' enables query-dependent head-wise "
            "sigmoid gates; default: head."
        ),
    )
    parser.add_argument(
        "--attention-pattern",
        choices=["full", "mhla", "mhla3-full1"],
        default="mhla3-full1",
        help=(
            "MMDiT attention pattern: full, all MHLA, or three MHLA blocks "
            "followed by one full Joint Attention block. Default: mhla3-full1."
        ),
    )
    parser.add_argument(
        "--mhla-latent-blocks", type=int, default=16,
        help="Spatial token blocks for latent MHLA. Default: 16.",
    )
    parser.add_argument(
        "--mhla-image-blocks", type=int, default=4,
        help="Spatial token blocks for image-digest MHLA. Default: 4.",
    )
    parser.add_argument(
        "--mhla-text-blocks", type=int, default=4,
        help="1D token blocks for text MHLA. Default: 4.",
    )
    parser.add_argument(
        "--mhla-backend", choices=["auto", "naive", "vectorized", "triton"],
        default="auto",
        help=(
            "MHLA implementation backend. 'naive' is for reference checks, "
            "'triton' requires CUDA; default: auto."
        ),
    )
    parser.add_argument(
        "--mhla-recompute-output", action="store_true",
        help=(
            "Do not save the MHLA output for backward; recompute it instead. "
            "Reduces activation VRAM at the cost of one extra MHLA forward."
        ),
    )
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--null-conditioning-prob", type=float, default=0.05,
                        help="Probability of replacing a caption with an empty caption for CFG training. Default: 0.05.")
    parser.add_argument("--time-scale", type=float, default=1000.0,
                        help="Frequency scale for the continuous Flow time embedding.")
    parser.add_argument("--prediction-type", choices=["rectified_flow", "flow_matching"],
                        default="rectified_flow",
                        help="Straight-path velocity objective; default: rectified_flow.")
    parser.add_argument("--latent-scale", type=float, default=None, help="Override VAE config scaling_factor")
    parser.add_argument("--reconstruction-loss-weight", type=float, default=0.1,
                        help="Weight of image x0 reconstruction loss. Default: 0.1.")
    parser.add_argument(
        "--max-reconstruction-contribution", type=float, default=0.05,
        help="Upper bound for reconstruction share in total loss. Default: 0.05.",
    )
    parser.add_argument("--observe-interval", type=int, default=1000,
                        help="Observe training every N optimizer steps by saving the latest sample image, checkpoint, and optional performance log; --no-artifacts disables image/checkpoint saving. Default: 1000.")
    parser.add_argument("--sample-steps", type=int, default=30,
                        help="Euler steps for Rectified Flow sampling. Default: 30.")
    parser.add_argument("--sample-prompt", action="append", default=None,
                        help="Sampling prompt; repeat this option to save multiple samples.")
    parser.add_argument("--gc-interval", type=int, default=100,
                        help="Run Python GC every N optimizer steps. Default: 100.")
    parser.add_argument(
        "--empty-cache-interval", type=int, default=0,
        help="Run CUDA empty_cache every N optimizer steps; 0 disables periodic calls. Default: 0.",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False,
        help="Compile the trainable DiT and Text Adapter forward on CUDA. Default: disabled.",
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Checkpoint main DiT blocks during training to reduce activation VRAM; "
            "adds recomputation overhead. Default: disabled."
        ),
    )
    parser.add_argument(
        "--compile-mode", choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode. Default: default.",
    )
    parser.add_argument(
        "--perf", "--performance", "--timing", dest="performance", action="store_true",
        help=(
            "Enable performance metrics: per-step timing and CUDA memory "
            "usage. --performance and --timing are aliases. Default: disabled."
        ),
    )
    parser.add_argument(
        "--perf-backward-breakdown", action="store_true",
        help=(
            "Add detailed MMDiT backward hooks to performance.jsonl. "
            "Implies --perf and adds measurement overhead."
        ),
    )
    parser.add_argument(
        "--perf-optimizer-breakdown", action="store_true",
        help=(
            "Add detailed APOLLO optimizer stages to performance.jsonl. "
            "Implies --perf and adds measurement overhead."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--timestep-repeats", type=int, default=4,
                        help="Independent timesteps/noises per encoded batch. Default: 4.")
    parser.add_argument("--lr", type=float, default=1e-4)
    add_lr_scheduler_arguments(parser, default="constant")
    parser.add_argument(
        "--auto-schedule-target-update-ratio", type=float, default=1e-3,
        help="Target parameter update ratio for AutoSchedule. Default: 1e-3.",
    )
    parser.add_argument(
        "--auto-schedule-ema-beta", type=float, default=0.99,
        help="EMA beta for AutoSchedule statistics. Default: 0.99.",
    )
    parser.add_argument(
        "--auto-schedule-trust-alpha", type=float, default=0.1,
        help="Trust-ratio controller gain. Default: 0.1.",
    )
    parser.add_argument(
        "--auto-schedule-min-factor", type=float, default=0.5,
        help="Minimum AutoSchedule multiplier/cap. Default: 0.5.",
    )
    parser.add_argument(
        "--auto-schedule-max-factor", type=float, default=4.0,
        help="Maximum AutoSchedule multiplier. Default: 4.0.",
    )
    parser.add_argument(
        "--auto-schedule-max-increase", type=float, default=1.05,
        help="Maximum multiplier increase per optimizer step. Default: 1.05.",
    )
    parser.add_argument(
        "--auto-schedule-max-decrease", type=float, default=0.95,
        help="Minimum multiplier change per optimizer step. Default: 0.95.",
    )
    parser.add_argument(
        "--auto-schedule-confidence-floor", type=float, default=0.25,
        help="Minimum confidence-based LR cap. Default: 0.25.",
    )
    parser.add_argument(
        "--auto-schedule-stability-gain", type=float, default=2.0,
        help="APOLLO scaling-instability cap gain. Default: 2.0.",
    )
    parser.add_argument(
        "--auto-schedule-limiter-gain", type=float, default=2.0,
        help="APOLLO norm-limiter cap gain. Default: 2.0.",
    )
    parser.add_argument(
        "--auto-schedule-cooldown-steps", type=int, default=4,
        help="Steps to ignore APOLLO scale instability after projection refresh. Default: 4.",
    )
    parser.add_argument(
        "--auto-schedule-warmup-steps", type=int, default=200,
        help=(
            "Optimizer steps before AutoSchedule multiplier/cap adaptation starts. "
            "Default: 200."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    add_optimizer_argument(
        parser,
        default="APOLLO",
        include_experimental=True,
        include_legacy=True,
    )
    add_optimizer_override_argument(parser, "--linear-optimizer", role="Linear")
    add_optimizer_override_argument(parser, "--conv-optimizer", role="Conv/ConvTranspose")
    parser.add_argument(
        "--linear-lr", type=float, default=None,
        help="Learning rate for the Linear optimizer. Muon/NorMuon/AdaMuon default to 0.02; SOAP to 0.001.",
    )
    parser.add_argument(
        "--linear-weight-decay", type=float, default=None,
        help="Weight decay for the Linear optimizer. Default: --weight-decay.",
    )
    parser.add_argument(
        "--conv-lr", type=float, default=None,
        help="Learning rate for the Conv optimizer. Default: --lr.",
    )
    parser.add_argument(
        "--conv-weight-decay", type=float, default=None,
        help="Weight decay for the Conv optimizer. Default: --weight-decay.",
    )
    parser.add_argument(
        "--linear-muon-momentum", type=float, default=0.95,
        help="Muon momentum for Linear weights. Default: 0.95.",
    )
    parser.add_argument(
        "--apollo-rank", type=int, default=8,
        help="Auxiliary rank for APOLLO Linear weights. Default: 8.",
    )
    parser.add_argument(
        "--apollo-scale", type=float, default=1.0,
        help="Gradient scale for APOLLO. Default: 1.0.",
    )
    parser.add_argument(
        "--apollo-mini-scale", type=float, default=128.0,
        help="Gradient scale for APOLLO-Mini. Default: 128.0.",
    )
    parser.add_argument(
        "--apollo-update-proj-gap", type=int, default=200,
        help="Optimizer steps between APOLLO random projection refreshes. Default: 200.",
    )
    parser.add_argument(
        "--apollo-projection-refresh-mode", choices=("none", "hard", "smooth"),
        default="hard",
        help="APOLLO projection refresh mode: none, hard, or smooth. Default: hard.",
    )
    parser.add_argument(
        "--apollo-projection-refresh-window", type=int, default=200,
        help="APOLLO smooth refresh window in optimizer steps. Default: 200.",
    )
    parser.add_argument(
        "--apollo-projection-refresh-mix",
        choices=("linear", "smoothstep", "stochastic", "ema"),
        default="smoothstep",
        help="APOLLO smooth refresh mixing curve. Default: smoothstep.",
    )
    parser.add_argument(
        "--apollo-projection-refresh-state", choices=("reset", "transport"),
        default="reset",
        help="APOLLO moment handling at refresh: reset or overlap transport. Default: reset.",
    )
    parser.add_argument(
        "--apollo-orthogonal-refresh-rate", type=float, default=0.0,
        help=(
            "Per-step tangent-space rotation rate for the APOLLO projection. "
            "Zero disables it. Default: 0."
        ),
    )
    parser.add_argument(
        "--rot-apollo-frequency", type=int, default=10,
        help="Optimizer steps between RotAPOLLO basis rotations. Default: 10.",
    )
    parser.add_argument(
        "--rot-apollo-rate", type=float, default=0.02,
        help="RotAPOLLO basis rotation rate. Default: 0.02.",
    )
    parser.add_argument(
        "--rot-apollo-exploration-ratio", type=float, default=0.2,
        help="RotAPOLLO random exploration ratio in [0, 1]. Default: 0.2.",
    )
    parser.add_argument(
        "--dual-rot-apollo-frequency", type=int, default=10,
        help="Steps between DualRotAPOLLO rotations. Default: 10.",
    )
    parser.add_argument(
        "--dual-rot-apollo-rate", type=float, default=0.02,
        help="DualRotAPOLLO basis rotation rate. Default: 0.02.",
    )
    parser.add_argument(
        "--dual-rot-apollo-exploration-ratio", type=float, default=0.2,
        help="DualRotAPOLLO exploration ratio in [0, 1]. Default: 0.2.",
    )
    parser.add_argument(
        "--dual-rot-apollo-roughness-beta", type=float, default=0.95,
        help="EMA coefficient for branch roughness. Default: 0.95.",
    )
    parser.add_argument(
        "--dual-rot-apollo-branch-temperature", type=float, default=5.0,
        help="Soft branch-selection temperature. Default: 5.0.",
    )
    parser.add_argument(
        "--apollo-scale-front", action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply APOLLO scale before the norm-growth limiter. Default: disabled.",
    )
    parser.add_argument(
        "--apollo-disable-norm-growth-limiter", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable APOLLO's norm-growth limiter. This is the default; pass "
            "--no-apollo-disable-norm-growth-limiter to enable the limiter."
        ),
    )
    parser.add_argument(
        "--apollo-norm-growth-rate", type=float, default=1.01,
        help="Maximum consecutive APOLLO update-norm growth. Default: 1.01.",
    )
    parser.add_argument(
        "--apollo-fallback", choices=["came", "sgd", "adamw-sf"], default="adamw-sf",
        help=(
            "Fallback optimizer for APOLLO 1D parameters. Default: adamw-sf."
        ),
    )
    parser.add_argument(
        "--apollo-matrix-fallback",
        choices=["apollo", "came", "auto", "adamw-sf", "auto-sf"],
        default="auto-sf",
        help=(
            "Fallback for APOLLO matrices: apollo, came, adamw-sf, auto (CAME "
            "state comparison), or auto-sf (AdamW-SF state comparison). Default: auto-sf."
        ),
    )
    parser.add_argument(
        "--apollo-fallback-state-margin", type=float, default=1.0,
        help=(
            "Use CAME when its estimated state is at most this multiple of "
            "APOLLO's. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--apollo-fallback-min-savings-bytes", type=int, default=0,
        help=(
            "Minimum estimated state savings required for matrix CAME "
            "fallback. Default: 0."
        ),
    )
    parser.add_argument(
        "--apollo-came-backend", choices=["auto", "torch", "triton"],
        default="torch",
        help=(
            "Backend for APOLLO-CAME low-rank adaptive updates. "
            "Default: torch."
        ),
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes. Default: 4.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--trainable-dtype", choices=["bf16", "fp32"], default="bf16",
        help=(
            "Storage dtype for DiT and Text Adapter parameters. Default: bf16; "
            "use fp32 for compatibility or debugging."
        ),
    )
    return parser.parse_args()
