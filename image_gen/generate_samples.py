"""Generate images from a trained image-latent DiT checkpoint.

Example::

    python image_gen/generate_samples.py \
        --checkpoint image_gen/output/20260908_120000/checkpoint_latest.safetensors \
        --vae-model Qwen/Qwen-Image \
        --prompt "a photograph of a child playing with a dog outdoors" \
        --prompt "a photograph of a horse running across a grassy field"

The VAE model cannot currently be recovered from checkpoint metadata, so
``--vae-model`` is required.  The text model is read from checkpoint metadata
unless ``--text-model`` is supplied.
"""

import argparse
import os
import sys
from contextlib import nullcontext

import torch
from safetensors.torch import load_file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.memory import collect_memory  # noqa: E402
from runtime.progress import RichProgress  # noqa: E402

if __package__:
    from .train import (
        DiT,
        NETWORK_CONFIG_VERSION,
        TextConditioningAdapter,
    )
    from .cli import DEFAULT_TEXT_MODEL
    from .inference import (
        decode_images,
        encode_images,
        encode_text,
        resolve_vae_dtype,
        resolve_vae_latent_scale,
        save_labeled_sample_grid,
    )
else:
    from train import (
        DiT,
        NETWORK_CONFIG_VERSION,
        TextConditioningAdapter,
    )
    from cli import DEFAULT_TEXT_MODEL
    from inference import (
        decode_images,
        encode_images,
        encode_text,
        resolve_vae_dtype,
        resolve_vae_latent_scale,
        save_labeled_sample_grid,
    )


DEFAULT_PROMPTS = (
    "a photograph of a child playing with a dog outdoors",
    "a photograph of a person riding a bicycle on a city street",
    "a photograph of a horse running across a grassy field",
    "a photograph of people and dogs relaxing in a park",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-model", required=True)
    parser.add_argument(
        "--vae-dtype", choices=["auto", "bf16", "fp32"], default="auto",
        help="VAE parameter dtype. Default: checkpoint value or bf16.",
    )
    parser.add_argument("--text-model", default=None,
                        help="Override the text model stored in checkpoint metadata.")
    parser.add_argument("--output", default="image_gen/generated_samples.png")
    parser.add_argument("--prompt", action="append", default=None,
                        help="Prompt to generate; repeat this option for multiple images.")
    parser.add_argument("--prompt-file", default=None,
                        help="UTF-8 text file with one prompt per line.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Euler sampling steps. Default: checkpoint value or 30.")
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="CFG scale; 1.0 disables guidance. Default: 1.0.")
    parser.add_argument("--height", type=int, default=None,
                        help="Output height in pixels. Default: checkpoint image_size.")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width in pixels. Default: checkpoint image_size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "fp32", "bf16"], default="auto",
        help="Autocast dtype for inference. Default: bf16 on CUDA, fp32 on CPU.",
    )
    parser.add_argument("--save-individual", action="store_true",
                        help="Also save one PNG per prompt beside the grid.")
    return parser.parse_args()


def read_checkpoint_config(path):
    if __package__:
        from .checkpoint import checkpoint_info
    else:
        from checkpoint import checkpoint_info

    config = checkpoint_info(path)
    if not config:
        raise ValueError(f"Checkpoint has no image_gen.config metadata: {path}")
    return config


def hidden_size_from_config(config):
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and hasattr(config, "text_config"):
        hidden_size = getattr(config.text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not determine Text Encoder hidden size")
    return int(hidden_size)


def _config_int_tuple(config, key, default):
    values = config.get(key, default)
    return tuple(int(value) for value in values)


def build_checkpoint_models(
    checkpoint_config, text_dim, latent_channels,
    reference_latent_height, reference_latent_width, device,
):
    """Build DiT and text adapter using the checkpoint architecture metadata."""
    text_adapter_dim = int(checkpoint_config.get("text_adapter_dim", 1024))
    transformer_dims = _config_int_tuple(
        checkpoint_config, "text_adapter_transformer_dims", (2048, 1024, 1024),
    )
    transformer_heads = _config_int_tuple(
        checkpoint_config, "text_adapter_transformer_heads", (16, 8, 8),
    )
    transformer_kv_heads = _config_int_tuple(
        checkpoint_config, "text_adapter_transformer_kv_heads", transformer_heads,
    )
    use_head_gate = checkpoint_config.get("attention_gate", "head") == "head"
    model = DiT(
        latent_channels,
        text_adapter_dim,
        dim=int(checkpoint_config.get("model_dim", 768)),
        depth=int(checkpoint_config.get("depth", 12)),
        heads=int(checkpoint_config.get("heads", 12)),
        patch_size=int(checkpoint_config.get("patch_size", 2)),
        reference_height=reference_latent_height,
        reference_width=reference_latent_width,
        context_depth=int(checkpoint_config.get("context_depth", 2)),
        context_heads=int(checkpoint_config.get("context_heads", 16)),
        gradient_checkpointing=bool(
            checkpoint_config.get("gradient_checkpointing", False)
        ),
        kv_heads=int(checkpoint_config.get("kv_heads", checkpoint_config.get("heads", 12))),
        context_kv_heads=int(
            checkpoint_config.get(
                "context_kv_heads", checkpoint_config.get("context_heads", 16),
            )
        ),
        use_head_gate=use_head_gate,
        attention_pattern=checkpoint_config.get(
            "attention_pattern", "mhla3-full1",
        ),
        mhla_latent_blocks=int(checkpoint_config.get("mhla_latent_blocks", 16)),
        mhla_image_blocks=int(checkpoint_config.get("mhla_image_blocks", 4)),
        mhla_text_blocks=int(checkpoint_config.get("mhla_text_blocks", 4)),
        mhla_backend=checkpoint_config.get("mhla_backend", "auto"),
        mhla_recompute_output=bool(
            checkpoint_config.get("mhla_recompute_output", False)
        ),
    ).to(device).eval()
    text_adapter = TextConditioningAdapter(
        text_dim,
        output_dim=text_adapter_dim,
        transformer_dims=transformer_dims,
        transformer_heads=transformer_heads,
        transformer_kv_heads=transformer_kv_heads,
        transformer_ff_mult=float(
            checkpoint_config.get("text_adapter_transformer_ff_mult", 3.0)
        ),
        rope_theta=float(
            checkpoint_config.get("text_adapter_rope_theta", 10000.0)
        ),
        use_head_gate=use_head_gate,
    ).to(device).eval()
    return model, text_adapter


def load_models(args, checkpoint_config, device):
    from transformers import AutoModel, AutoTokenizer
    from diffusers import AutoencoderKL

    text_model_name = args.text_model or checkpoint_config.get("text_model") or DEFAULT_TEXT_MODEL
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_encoder = AutoModel.from_pretrained(text_model_name).to(device).eval()
    text_dim = hidden_size_from_config(text_encoder.config)
    requested_vae_dtype = args.vae_dtype
    if requested_vae_dtype == "auto":
        requested_vae_dtype = checkpoint_config.get("vae_dtype", "bf16")
    vae_dtype = resolve_vae_dtype(device, requested_vae_dtype)
    if "qwen" in args.vae_model.lower():
        from diffusers import AutoencoderKLQwenImage
        vae = AutoencoderKLQwenImage.from_pretrained(
            args.vae_model, subfolder="vae", torch_dtype=vae_dtype,
        ).to(device).eval()
    else:
        vae = AutoencoderKL.from_pretrained(
            args.vae_model, subfolder="vae", torch_dtype=vae_dtype,
        ).to(device).eval()

    image_size = int(checkpoint_config.get("image_size", 256))
    reference_probe = torch.zeros(1, 3, image_size, image_size, device=device)
    height = args.height or image_size
    width = args.width or image_size
    if height <= 0 or width <= 0:
        raise ValueError("--height and --width must be positive")
    probe = torch.zeros(1, 3, height, width, device=device)
    latent_scale = checkpoint_config.get("latent_scale")
    latent_scale = resolve_vae_latent_scale(vae, latent_scale)
    with torch.no_grad():
        reference_latent_probe = encode_images(vae, reference_probe, float(latent_scale))
        latent_probe = encode_images(vae, probe, float(latent_scale))
    reference_latent_height, reference_latent_width = (
        int(value) for value in reference_latent_probe.shape[-2:]
    )
    latent_channels = int(latent_probe.shape[1])
    latent_height, latent_width = (int(value) for value in latent_probe.shape[-2:])
    del reference_probe, reference_latent_probe, probe, latent_probe

    expected_latent_channels = checkpoint_config.get("latent_channels")
    if expected_latent_channels is not None and int(expected_latent_channels) != latent_channels:
        raise ValueError(
            f"Checkpoint expects {expected_latent_channels} latent channels, "
            f"but VAE produced {latent_channels}"
        )
    model, text_adapter = build_checkpoint_models(
        checkpoint_config,
        text_dim,
        latent_channels,
        reference_latent_height,
        reference_latent_width,
        device,
    )

    state = load_file(args.checkpoint, device="cpu")
    dit_state = {
        key[len("dit."):]: value
        for key, value in state.items() if key.startswith("dit.")
    }
    adapter_state = {
        key[len("text_adapter."):]: value
        for key, value in state.items() if key.startswith("text_adapter.")
    }
    if not dit_state or not adapter_state:
        raise ValueError(f"Checkpoint does not contain DiT and text adapter weights: {args.checkpoint}")
    model.load_state_dict(dit_state)
    text_adapter.load_state_dict(adapter_state)
    del state, dit_state, adapter_state
    return (
        model, text_adapter, vae, tokenizer, text_encoder,
        latent_channels, latent_height, latent_width,
        float(latent_scale), height, width, text_model_name,
    )


def build_prompts(args):
    prompts = list(args.prompt or [])
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as file:
            prompts.extend(line.strip() for line in file if line.strip())
    return prompts or list(DEFAULT_PROMPTS)


def autocast_context(device, dtype_name):
    if dtype_name == "fp32":
        return nullcontext()
    if dtype_name == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def generate(model, text_adapter, vae, tokenizer, text_encoder, device,
             prompts, latent_channels, latent_height,
             latent_width, latent_scale, max_length, steps, guidance_scale,
             time_scale, amp_dtype):
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if guidance_scale < 0:
        raise ValueError("--guidance-scale must be non-negative")

    empty_prompts = [""] * len(prompts)
    with autocast_context(device, amp_dtype):
        text_hidden_states, text_condition_mask = encode_text(
            tokenizer, text_encoder, prompts, device, max_length,
        )
        if guidance_scale != 1.0:
            unconditional_text_hidden_states, unconditional_text_condition_mask = encode_text(
                tokenizer, text_encoder, empty_prompts, device, max_length,
            )
        adapter_dtype = next(text_adapter.parameters()).dtype
        text_hidden_states = text_hidden_states.to(dtype=adapter_dtype)
        text_condition_tokens = text_adapter(
            text_hidden_states, text_condition_mask,
        )
        if guidance_scale != 1.0:
            unconditional_text_hidden_states = unconditional_text_hidden_states.to(
                dtype=adapter_dtype,
            )
            unconditional_text_condition_tokens = text_adapter(
                unconditional_text_hidden_states,
                unconditional_text_condition_mask,
            )

        samples = torch.randn(
            len(prompts), latent_channels,
            latent_height, latent_width, device=device,
        )
        for step_index in RichProgress(range(steps), description="sampling"):
            time = torch.full(
                (len(prompts),), 1.0 - step_index / steps, device=device,
            )
            prediction = model(
                samples, time * time_scale,
                text_condition_tokens, text_condition_mask,
            )
            if guidance_scale != 1.0:
                unconditional_prediction = model(
                    samples, time * time_scale,
                    unconditional_text_condition_tokens,
                    unconditional_text_condition_mask,
                )
                prediction = unconditional_prediction + guidance_scale * (
                    prediction - unconditional_prediction
                )
            samples = samples - prediction / steps
        images = decode_images(vae, samples, latent_scale)
    return images


def save_individual_images(images, output_path):
    from torchvision.transforms.functional import to_pil_image

    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    stem, extension = os.path.splitext(os.path.basename(output_path))
    for index, image in enumerate(images, start=1):
        path = os.path.join(output_dir, f"{stem}_{index:02d}{extension or '.png'}")
        to_pil_image(((image.float().clamp(-1, 1) + 1) / 2).cpu()).save(path)
        print(f"saved {path}")


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.manual_seed(args.seed)
    checkpoint_config = read_checkpoint_config(args.checkpoint)
    expected_version = checkpoint_config.get("network_version")
    print(f"checkpoint network_version={expected_version}")
    if expected_version is not None and expected_version != NETWORK_CONFIG_VERSION:
        raise ValueError(
            f"Checkpoint network_version {expected_version!r} is incompatible with "
            f"current network_version {NETWORK_CONFIG_VERSION!r}"
        )
    prompts = build_prompts(args)
    steps = args.steps or int(checkpoint_config.get("sample_steps", 30))
    max_length = int(checkpoint_config.get("text_max_length", 256))
    time_scale = float(checkpoint_config.get("time_scale", 1000.0))
    if time_scale != 1000.0:
        print(f"warning: checkpoint time_scale={time_scale}; using it for sampling")

    (
        model, text_adapter, vae, tokenizer, text_encoder,
        latent_channels, latent_height, latent_width,
        latent_scale, height, width, text_model_name,
    ) = load_models(args, checkpoint_config, device)
    amp_dtype = args.dtype
    if args.dtype == "auto":
        amp_dtype = "bf16" if device.type == "cuda" else "fp32"
    requested_vae_dtype = args.vae_dtype
    if requested_vae_dtype == "auto":
        requested_vae_dtype = checkpoint_config.get("vae_dtype", "bf16")
    effective_vae_dtype = resolve_vae_dtype(device, requested_vae_dtype)
    print(
        f"device={device} dtype={amp_dtype} vae_dtype={effective_vae_dtype} "
        f"image_size={height}x{width} "
        f"latent={latent_channels} "
        f"latent_grid={latent_height}x{latent_width} steps={steps} "
        f"guidance_scale={args.guidance_scale} text_model={text_model_name}"
    )
    images = generate(
        model, text_adapter, vae, tokenizer, text_encoder, device, prompts,
        latent_channels, latent_height, latent_width,
        latent_scale, max_length, steps, args.guidance_scale,
        time_scale, amp_dtype,
    )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_labeled_sample_grid(images, prompts, output_path)
    print(f"saved {output_path}")
    if args.save_individual:
        save_individual_images(images, output_path)
    del images
    collect_memory()


if __name__ == "__main__":
    main()
