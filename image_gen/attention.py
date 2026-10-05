"""Spatial RoPE cache and self-attention layer for image tokens."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_gen.layers import (
    HEAD_GATE_SCALE,
    RMSNorm,
    _ensure_autograd_tensor,
    apply_rope_pairs,
    scaled_dot_product_attention_gqa,
    validate_attention_head_counts,
    validate_attention_mask_shape,
    validate_spatial_token_grid,
)
from image_gen.text_conditioning import GroupedQueryProjection, OneDRoPECache


class TwoDRoPECache:
    """Share spatial 2D RoPE tensors across all DiT self-attention blocks."""
    def __init__(self, head_dim):
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        self.head_dim = head_dim
        self.frequency_count = head_dim // 4
        self._cache = {}
        self._device_key = None

    def clear(self):
        self._cache.clear()
        self._device_key = None

    def get(self, height, width, device, dtype):
        device_key = (device.type, device.index)
        if self._device_key != device_key:
            self._cache.clear()
            self._device_key = device_key
        key = (int(height), int(width), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            cos, sin = cached
            cos = _ensure_autograd_tensor(cos)
            sin = _ensure_autograd_tensor(sin)
            cached = (cos, sin)
            self._cache[key] = cached
            return cached
        with torch.no_grad():
            frequency = torch.exp(
                -math.log(10000) * torch.arange(
                    self.frequency_count, device=device, dtype=torch.float32,
                ) / max(self.frequency_count - 1, 1)
            )
            yy, xx = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device), indexing="ij",
            )
            angles = torch.cat([
                yy.flatten()[:, None] * frequency[None],
                xx.flatten()[:, None] * frequency[None],
            ], dim=-1)
            # The returned tensors index even/odd pairs directly.  This
            # avoids a repeat_interleave and lets the application use the
            # same compact representation for 1D and 2D RoPE.
            cos = angles.cos().to(dtype=dtype)[None, None]
            sin = angles.sin().to(dtype=dtype)[None, None]
        self._cache[key] = (cos, sin)
        return cos, sin

class TwoDRoPESelfAttention(nn.Module):
    def __init__(self, dim, heads, kv_heads=None, rope_cache=None,
                 use_head_gate=True):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        validate_attention_head_counts(heads, kv_heads, "2D RoPE")
        if dim % heads or (dim // heads) % 4:
            raise ValueError("head dimension must be divisible by 4 for 2D RoPE")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.rope_cache = (
            rope_cache if rope_cache is not None else TwoDRoPECache(self.head_dim)
        )
        self.qkv = GroupedQueryProjection(dim, heads, kv_heads)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.head_gate = nn.Linear(dim, heads) if use_head_gate else None
        if self.head_gate is not None:
            nn.init.zeros_(self.head_gate.weight)
            nn.init.zeros_(self.head_gate.bias)
        self.out = nn.Linear(dim, dim)

    def apply_rope(self, tensor, cos, sin):
        return apply_rope_pairs(tensor, cos, sin)

    def forward(self, x, height, width):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rope_cache.get(height, width, q.device, q.dtype)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)
        output = scaled_dot_product_attention_gqa(q, k, v)
        if self.head_gate is not None:
            gate = HEAD_GATE_SCALE * torch.sigmoid(self.head_gate(x))
            output = output * gate.transpose(1, 2).unsqueeze(-1)
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        return self.out(output)


class MMDiTJointAttention(nn.Module):
    """Joint attention over latent, image-context, and text streams.

    QKV and output projections remain stream-specific, while the attention
    score/value mixing is shared by concatenating the three streams.  This
    removes the asymmetric query/context structure of CrossAttention.
    """
    def __init__(self, dim, heads, kv_heads=None, rope_cache=None,
                 text_rope_cache=None, use_head_gate=True):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        validate_attention_head_counts(heads, kv_heads, "MMDiT")
        if dim % heads or (dim // heads) % 4:
            raise ValueError("MMDiT head dimension must be divisible by 4")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.rope_cache = (
            rope_cache if rope_cache is not None else TwoDRoPECache(self.head_dim)
        )
        self.text_rope_cache = (
            text_rope_cache
            if text_rope_cache is not None
            else OneDRoPECache(self.head_dim)
        )
        self.latent_qkv = GroupedQueryProjection(dim, heads, kv_heads)
        self.image_qkv = GroupedQueryProjection(dim, heads, kv_heads)
        self.text_qkv = GroupedQueryProjection(dim, heads, kv_heads)
        self.latent_head_gate = nn.Linear(dim, heads) if use_head_gate else None
        self.image_head_gate = nn.Linear(dim, heads) if use_head_gate else None
        self.text_head_gate = nn.Linear(dim, heads) if use_head_gate else None
        for gate in (
            self.latent_head_gate, self.image_head_gate, self.text_head_gate,
        ):
            if gate is not None:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
        self.latent_q_norm = RMSNorm(self.head_dim)
        self.latent_k_norm = RMSNorm(self.head_dim)
        self.image_q_norm = RMSNorm(self.head_dim)
        self.image_k_norm = RMSNorm(self.head_dim)
        self.text_q_norm = RMSNorm(self.head_dim)
        self.text_k_norm = RMSNorm(self.head_dim)
        self.latent_out = nn.Linear(dim, dim)
        self.image_out = nn.Linear(dim, dim)
        self.text_out = nn.Linear(dim, dim)

    def _project(self, tokens, qkv_projection, q_norm, k_norm, head_gate):
        q, k, v = qkv_projection(tokens)
        gate = (
            HEAD_GATE_SCALE * torch.sigmoid(head_gate(tokens))
            if head_gate is not None else None
        )
        return q_norm(q), k_norm(k), v, gate

    def _apply_rope(self, tensor, cos, sin):
        return apply_rope_pairs(tensor, cos, sin)

    def _apply_2d_rope(self, q, k, height, width):
        cos, sin = self.rope_cache.get(height, width, q.device, q.dtype)
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

    def _apply_text_rope(self, q, k):
        cos, sin = self.text_rope_cache.get(q.shape[2], q.device, q.dtype)
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

    def forward(
        self,
        latent_tokens,
        image_context_tokens,
        text_context_tokens,
        latent_height,
        latent_width,
        image_height,
        image_width,
        text_context_mask=None,
        attention_mask=None,
    ):
        batch = validate_spatial_token_grid(
            latent_tokens, latent_height, latent_width, "latent"
        )
        image_batch = validate_spatial_token_grid(
            image_context_tokens, image_height, image_width, "image context"
        )
        if image_batch != batch:
            raise ValueError("latent and image context batch sizes must match")
        if text_context_tokens.ndim != 3:
            raise ValueError(
                "text context tokens must have shape (batch, tokens, dim), "
                f"got {tuple(text_context_tokens.shape)}"
            )
        if text_context_tokens.shape[0] != batch:
            raise ValueError("latent and text context batch sizes must match")
        validate_attention_mask_shape(
            text_context_mask, batch, text_context_tokens.shape[1],
            "text context",
        )
        latent_q, latent_k, latent_v, latent_gate = self._project(
            latent_tokens, self.latent_qkv,
            self.latent_q_norm, self.latent_k_norm, self.latent_head_gate,
        )
        image_q, image_k, image_v, image_gate = self._project(
            image_context_tokens, self.image_qkv,
            self.image_q_norm, self.image_k_norm, self.image_head_gate,
        )
        text_q, text_k, text_v, text_gate = self._project(
            text_context_tokens, self.text_qkv,
            self.text_q_norm, self.text_k_norm, self.text_head_gate,
        )
        latent_q, latent_k = self._apply_2d_rope(
            latent_q, latent_k, latent_height, latent_width,
        )
        image_q, image_k = self._apply_2d_rope(
            image_q, image_k, image_height, image_width,
        )
        text_q, text_k = self._apply_text_rope(text_q, text_k)

        query = torch.cat((latent_q, image_q, text_q), dim=2)
        key = torch.cat((latent_k, image_k, text_k), dim=2)
        value = torch.cat((latent_v, image_v, text_v), dim=2)
        if attention_mask is None and text_context_mask is not None:
            text_mask = text_context_mask.to(
                device=query.device, dtype=torch.bool,
            )
            attention_mask = F.pad(
                text_mask, (latent_tokens.shape[1] + image_context_tokens.shape[1], 0),
                value=True,
            )[:, None, None, :]
        output = scaled_dot_product_attention_gqa(
            query, key, value, attn_mask=attention_mask,
        )
        latent_count = latent_tokens.shape[1]
        image_count = image_context_tokens.shape[1]
        latent_output, image_output, text_output = output.split(
            (latent_count, image_count, text_context_tokens.shape[1]), dim=2,
        )
        if latent_gate is not None:
            latent_output = latent_output * latent_gate.transpose(1, 2).unsqueeze(-1)
            image_output = image_output * image_gate.transpose(1, 2).unsqueeze(-1)
            text_output = text_output * text_gate.transpose(1, 2).unsqueeze(-1)

        def merge(stream):
            return stream.transpose(1, 2).reshape(
                stream.shape[0], stream.shape[2], -1,
            )

        return (
            self.latent_out(merge(latent_output)),
            self.image_out(merge(image_output)),
            self.text_out(merge(text_output)),
        )
