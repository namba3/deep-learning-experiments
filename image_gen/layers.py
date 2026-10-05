"""Transformer layers and attention helpers for image generation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_GATE_SCALE = 2.0


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, frequency_dim=256):
        super().__init__()
        if frequency_dim % 2 != 0:
            raise ValueError("frequency_dim must be even")
        self.frequency_dim = frequency_dim
        half = frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        # Frequencies are fixed and should follow the module across devices,
        # but they are not learned model state and need not be checkpointed.
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        freqs = self.frequencies.to(device=timestep.device)
        args = timestep.float()[:, None] * freqs[None]
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(embedding)

class ResolutionEmbedding(nn.Module):
    """Embed the latent grid size without tying conditioning to a bucket ID."""
    def __init__(self, dim, reference_height, reference_width):
        super().__init__()
        self.reference_height = float(reference_height)
        self.reference_width = float(reference_width)
        hidden = max(dim // 4, 64)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.output_dim = hidden

    def forward(self, height, width, batch_size, device, dtype):
        height = torch.as_tensor(height, device=device, dtype=torch.float32)
        width = torch.as_tensor(width, device=device, dtype=torch.float32)
        features = torch.stack([
            torch.log(height / self.reference_height),
            torch.log(width / self.reference_width),
            torch.log(height / width),
            torch.log((height * width) / (self.reference_height * self.reference_width)),
        ]).expand(batch_size, -1)
        return self.mlp(features).to(dtype=dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        # PyTorch's CUDA RMSNorm kernel avoids the temporary ``x.float()``
        # tensor used by the portable implementation.  Keep the latter as a
        # fallback because older PyTorch versions and CPU execution do not
        # necessarily provide the fused operator.
        if x.is_cuda and hasattr(F, "rms_norm"):
            weight = None if self.weight is None else self.weight.to(dtype=x.dtype)
            return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=self.eps)
        output = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output.to(dtype=x.dtype)
        return output * self.weight.to(dtype=x.dtype) if self.weight is not None else output

def apply_rope_pairs(tensor, cos, sin):
    """Apply RoPE from compact pair-wise cosine/sine tables.

    ``cos`` and ``sin`` have half the head dimension and are broadcast over
    batch and heads.  Keeping the tables compact also avoids constructing the
    full repeated table on every attention path.
    """
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1,
    ).flatten(-2)

def _ensure_autograd_tensor(tensor):
    """Convert an inference-mode cache tensor before autograd reuse."""
    is_inference = getattr(tensor, "is_inference", None)
    if is_inference is not None and is_inference():
        # clone() inside inference_mode would produce another inference tensor.
        with torch.inference_mode(False):
            return tensor.clone()
    return tensor

def validate_attention_head_counts(heads, kv_heads, label="attention"):
    """Validate the Q/KV head relationship required by grouped-query attention."""
    if heads <= 0 or kv_heads <= 0:
        raise ValueError(f"{label} head counts must be positive")
    if kv_heads > heads or heads % kv_heads:
        raise ValueError(
            f"{label} kv heads ({kv_heads}) must divide q heads ({heads}) "
            "and must not exceed them"
        )

def validate_spatial_token_grid(tokens, height, width, label):
    """Validate the N(token) <-> H*W contract at an attention boundary."""
    if tokens.ndim != 3:
        raise ValueError(
            f"{label} tokens must have shape (batch, tokens, dim), "
            f"got {tuple(tokens.shape)}"
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} spatial dimensions must be positive")
    expected_tokens = int(height) * int(width)
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"{label} token count {tokens.shape[1]} does not match "
            f"height*width={expected_tokens}"
        )
    return tokens.shape[0]

def validate_attention_mask_shape(mask, batch, tokens, label):
    """Validate a 2D batch/token mask before it is broadcast into attention."""
    if mask is None:
        return
    if mask.ndim != 2 or mask.shape != (batch, tokens):
        raise ValueError(
            f"{label} mask must have shape ({batch}, {tokens}), "
            f"got {tuple(mask.shape)}"
        )

def scaled_dot_product_attention_gqa(query, key, value, attn_mask=None):
    """SDPA with native GQA when Q and KV head counts differ.

    The fallback keeps the code usable with older PyTorch versions.  Native
    GQA avoids materializing repeated K/V heads before the attention kernel.
    """
    if query.shape[1] == key.shape[1]:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask,
        )
    try:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, enable_gqa=True,
        )
    except TypeError:
        repeats = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask,
        )
