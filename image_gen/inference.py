"""VAE, text, and sampling helpers for image-latent models."""

from contextlib import nullcontext

from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms.functional import to_pil_image

def amp_context_for(device, dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()

def resolve_vae_dtype(device, requested):
    if requested == "bf16" and device.type != "cuda":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[requested]

def encode_text(tokenizer, text_encoder, captions, device, max_length):
    if max_length <= 0:
        raise ValueError(f"text_max_length must be positive, got {max_length}")
    empty_captions = [not str(caption).strip() for caption in captions]
    tokenizer_captions = captions
    if empty_captions and all(empty_captions):
        # Qwen3.5 returns a zero-length sequence for an all-empty batch.
        # Keep one masked token so its attention implementation can reshape it.
        null_token = tokenizer.pad_token or tokenizer.eos_token or " "
        tokenizer_captions = [null_token] * len(captions)
    tokens = tokenizer(
        tokenizer_captions, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    if empty_captions and all(empty_captions):
        tokens["attention_mask"].zero_()
    tokens = {key: value.to(device) for key, value in tokens.items()}
    if hasattr(text_encoder, "text_model"):
        output = text_encoder.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
        )
    else:
        output = text_encoder(**tokens)
    return output.last_hidden_state, tokens["attention_mask"].bool()

def save_labeled_sample_grid(images, prompts, path):
    """Save generated images in a 2x2 grid with white prompt labels below them."""
    pil_images = [to_pil_image(((image.float().clamp(-1, 1) + 1) / 2).cpu()) for image in images]
    if not pil_images:
        raise ValueError("Cannot save an empty sample grid")
    tile_width, tile_height = pil_images[0].size
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 16)
    except OSError:
        font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    label_lines = []
    for index, prompt in enumerate(prompts[:len(pil_images)], start=1):
        label = f"[{index}] {prompt}"
        lines = []
        current = ""
        for word in label.split():
            candidate = f"{current} {word}".strip()
            if current and draw_probe.textlength(candidate, font=font) > tile_width - 16:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        label_lines.append(lines or [label])
    line_height = max(font.getbbox("Ag")[3] - font.getbbox("Ag")[1], 16)
    label_height = max(24, max(len(lines) for lines in label_lines) * line_height + 16)
    cols = 2
    rows = (len(pil_images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile_width, rows * (tile_height + label_height)), "white")
    for index, image in enumerate(pil_images):
        x = (index % cols) * tile_width
        y = (index // cols) * (tile_height + label_height)
        draw = ImageDraw.Draw(grid)
        grid.paste(image, (x, y))
        draw.multiline_text(
            (x + 8, y + tile_height + 8),
            "\n".join(label_lines[index]),
            fill="black", font=font, spacing=2,
        )
    grid.save(path)

def encode_images(vae, images, latent_scale=None):
    # QwenImage's VAE is implemented as a causal 3D VAE and expects
    # [batch, channels, frames, height, width], even for a still image.
    vae_dtype = next(
        parameter.dtype for parameter in vae.parameters()
        if parameter.is_floating_point()
    )
    images = images.to(dtype=vae_dtype)
    qwen_image_vae = vae.__class__.__name__ == "AutoencoderKLQwenImage"
    vae_input = images.unsqueeze(2) if qwen_image_vae and images.ndim == 4 else images
    encoded = vae.encode(vae_input)
    latent = encoded.latent_dist.sample() if hasattr(encoded, "latent_dist") else encoded.latents
    if qwen_image_vae and latent.ndim == 5:
        if latent.shape[2] != 1:
            raise ValueError(f"Expected one latent frame for a still image, got {latent.shape[2]}")
        latent = latent[:, :, 0]
    scale = resolve_vae_latent_scale(vae, latent_scale)
    return latent * scale

def resolve_vae_latent_scale(vae, override=None):
    """Return a usable latent scale when a VAE config omits one."""
    if override is not None:
        return float(override)
    config_scale = getattr(getattr(vae, "config", None), "scaling_factor", None)
    return 1.0 if config_scale is None else float(config_scale)

def validate_bucket_shapes_with_vae(vae, bucket_shapes, latent_scale=None):
    """Verify that every pixel bucket maps to a consistent integer VAE stride."""
    try:
        parameter = next(vae.parameters())
    except StopIteration as error:
        raise ValueError("VAE must contain at least one parameter") from error
    strides = []
    with torch.no_grad():
        for height, width in bucket_shapes:
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"bucket dimensions must be positive, got {height}x{width}"
                )
            probe = torch.zeros(
                1, 3, int(height), int(width),
                device=parameter.device, dtype=parameter.dtype,
            )
            latent = encode_images(vae, probe, latent_scale)
            if latent.ndim != 4:
                raise ValueError(
                    "VAE must return image latents with shape (B, C, H, W), "
                    f"got {tuple(latent.shape)} for bucket {height}x{width}"
                )
            latent_height, latent_width = latent.shape[-2:]
            if latent_height <= 0 or latent_width <= 0:
                raise ValueError(
                    f"VAE returned an empty latent grid for bucket {height}x{width}"
                )
            if height % latent_height or width % latent_width:
                raise ValueError(
                    "bucket dimensions must be exact multiples of the VAE latent "
                    f"grid: bucket={height}x{width}, latent="
                    f"{latent_height}x{latent_width}"
                )
            strides.append((height // latent_height, width // latent_width))
    if not strides:
        raise ValueError("at least one bucket shape is required")
    if len(set(strides)) != 1:
        raise ValueError(f"VAE stride differs across buckets: {strides}")
    return strides[0]

def decode_images(vae, latents, latent_scale=None):
    scale = resolve_vae_latent_scale(vae, latent_scale)
    qwen_image_vae = vae.__class__.__name__ == "AutoencoderKLQwenImage"
    vae_dtype = next(
        parameter.dtype for parameter in vae.parameters()
        if parameter.is_floating_point()
    )
    decoder_input = (latents / scale).to(dtype=vae_dtype)
    if qwen_image_vae and decoder_input.ndim == 4:
        decoder_input = decoder_input.unsqueeze(2)
    decoded = vae.decode(decoder_input)
    images = decoded.sample if hasattr(decoded, "sample") else decoded[0]
    if qwen_image_vae and images.ndim == 5:
        if images.shape[2] != 1:
            raise ValueError(f"Expected one decoded frame for a still image, got {images.shape[2]}")
        images = images[:, :, 0]
    return images

def flow_velocity_target(clean, noise, prediction_type):
    """Return the velocity target for the supported straight-path objectives.

    Rectified Flow and conditional Flow Matching use the same target for the
    current path ``x_t=(1-t)clean+t noise``: ``dx_t/dt=noise-clean``.
    Keeping the dispatch explicit prevents the CLI value from becoming a
    silently ignored checkpoint setting when another objective is added.
    """
    if prediction_type not in {"rectified_flow", "flow_matching"}:
        raise ValueError(f"unknown prediction type: {prediction_type}")
    return noise - clean

@torch.inference_mode()
def save_flow_samples(dit, text_adapter, vae, tokenizer, text_encoder, device, dtype,
                      latent_channels, latent_height, latent_width, latent_scale,
                      prompts, max_length, sample_steps, time_scale, path):
    dit.eval()
    text_adapter.eval()
    sample_amp_context = amp_context_for(device, dtype)
    with sample_amp_context:
        text_hidden_states, text_condition_mask = encode_text(
            tokenizer, text_encoder, prompts, device, max_length,
        )
        adapter_dtype = next(text_adapter.parameters()).dtype
        text_hidden_states = text_hidden_states.to(dtype=adapter_dtype)
        text_condition_tokens = text_adapter(
            text_hidden_states, text_condition_mask,
        )
        dit_dtype = next(dit.parameters()).dtype
        text_condition_tokens = text_condition_tokens.to(dtype=dit_dtype)
        samples = torch.randn(
            len(prompts), latent_channels,
            latent_height, latent_width, device=device,
        )
        step_size = 1.0 / sample_steps
        for step in range(sample_steps):
            # Integrate backward from t=1 (pure noise) to t=0.  Evaluate the
            # velocity at the current endpoint of each backward interval so the
            # sampler never performs an update beyond t=0.
            time = torch.full((len(prompts),), 1.0 - step / sample_steps, device=device)
            prediction = dit(
                samples, time * time_scale,
                text_condition_tokens, text_condition_mask,
            )
            samples = samples - step_size * prediction
        images = decode_images(vae, samples, latent_scale)
    save_labeled_sample_grid(images, prompts, path)
    dit.train()
    text_adapter.train()
