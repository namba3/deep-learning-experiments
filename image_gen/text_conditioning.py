"""Text conditioning transformer and grouped-query projection layers."""

import torch
import torch.nn as nn

from core.layers import GatedLinear
from image_gen.layers import (
    HEAD_GATE_SCALE,
    RMSNorm,
    _ensure_autograd_tensor,
    apply_rope_pairs,
    scaled_dot_product_attention_gqa,
    validate_attention_head_counts,
)


class GroupedQueryProjection(nn.Module):
    """Project one token stream into Q heads and a smaller number of KV heads."""
    def __init__(self, dim, heads, kv_heads):
        super().__init__()
        validate_attention_head_counts(heads, kv_heads)
        if dim % heads:
            raise ValueError("attention dimension must be divisible by q heads")
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.kv_dim = kv_heads * self.head_dim
        self.qkv = nn.Linear(dim, dim + self.kv_dim * 2)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load both fused checkpoints and legacy separate q/kv weights."""
        fused_weight_key = prefix + "qkv.weight"
        fused_bias_key = prefix + "qkv.bias"
        legacy_keys = (
            prefix + "q.weight", prefix + "q.bias",
            prefix + "kv.weight", prefix + "kv.bias",
        )
        if fused_weight_key not in state_dict:
            legacy_q_weight = state_dict.get(prefix + "q.weight")
            legacy_kv_weight = state_dict.get(prefix + "kv.weight")
            if legacy_q_weight is not None and legacy_kv_weight is not None:
                state_dict[fused_weight_key] = torch.cat(
                    (legacy_q_weight, legacy_kv_weight), dim=0,
                )
        if fused_bias_key not in state_dict:
            legacy_q_bias = state_dict.get(prefix + "q.bias")
            legacy_kv_bias = state_dict.get(prefix + "kv.bias")
            if legacy_q_bias is not None and legacy_kv_bias is not None:
                state_dict[fused_bias_key] = torch.cat(
                    (legacy_q_bias, legacy_kv_bias), dim=0,
                )
        for key in legacy_keys:
            state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        batch, tokens, _ = x.shape
        query, key, value = self.qkv(x).split(
            (self.dim, self.kv_dim, self.kv_dim), dim=-1,
        )
        query = query.reshape(
            batch, tokens, self.heads, self.head_dim,
        ).transpose(1, 2)
        key = key.reshape(
            batch, tokens, self.kv_heads, self.head_dim,
        ).transpose(1, 2)
        value = value.reshape(
            batch, tokens, self.kv_heads, self.head_dim,
        ).transpose(1, 2)
        return query, key, value

class OneDRoPECache:
    """Cache 1D text RoPE tensors shared by all text transformer blocks."""
    def __init__(self, head_dim, theta=10000.0):
        if head_dim % 2:
            raise ValueError("text RoPE head dimension must be even")
        if theta <= 0:
            raise ValueError("text RoPE theta must be positive")
        self.head_dim = head_dim
        self.theta = float(theta)
        self._cache = {}
        self._device_key = None

    def clear(self):
        self._cache.clear()
        self._device_key = None

    def get(self, tokens, device, dtype):
        device_key = (device.type, device.index)
        if self._device_key != device_key:
            self._cache.clear()
            self._device_key = device_key
        key = (int(tokens), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            cos, sin = cached
            cos = _ensure_autograd_tensor(cos)
            sin = _ensure_autograd_tensor(sin)
            cached = (cos, sin)
            self._cache[key] = cached
            return cached
        with torch.no_grad():
            positions = torch.arange(tokens, device=device, dtype=torch.float32)
            exponent = torch.arange(
                0, self.head_dim, 2, device=device, dtype=torch.float32,
            ) / self.head_dim
            inverse_frequency = self.theta ** (-exponent)
            angles = positions[:, None] * inverse_frequency[None, :]
            # Store one value per even/odd pair.  Expanding to the full head
            # dimension is unnecessary and doubles the persistent cache.
            cos = angles.cos().to(dtype=dtype)
            sin = angles.sin().to(dtype=dtype)
        cos = cos[None, None]
        sin = sin[None, None]
        self._cache[key] = (cos, sin)
        return cos, sin

class BidirectionalTextTransformerBlock(nn.Module):
    """A non-causal, mask-aware transformer block for text conditioning."""
    def __init__(self, dim, heads, kv_heads=None, ff_mult=3.0,
                 rope_cache=None, use_head_gate=True):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        validate_attention_head_counts(heads, kv_heads, "text transformer")
        if dim % heads or (dim // heads) % 2:
            raise ValueError(
                "text transformer dimension must have an even per-head dimension"
            )
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.norm1 = RMSNorm(dim)
        self.qkv = GroupedQueryProjection(dim, heads, kv_heads)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope_cache = (
            rope_cache if rope_cache is not None else OneDRoPECache(self.head_dim)
        )
        self.attn_out = nn.Linear(dim, dim)
        self.head_gate = nn.Linear(dim, heads) if use_head_gate else None
        if self.head_gate is not None:
            nn.init.zeros_(self.head_gate.weight)
            nn.init.zeros_(self.head_gate.bias)
        self.norm2 = RMSNorm(dim)
        hidden = max(int(dim * ff_mult), 1)
        self.ffn = nn.Sequential(GatedLinear(dim, hidden), nn.Linear(hidden, dim))
        # Start as an identity block so the new conditioning path grows
        # gradually from the existing adapter behavior.
        self.residual_attn_gate = nn.Parameter(torch.zeros(()))
        self.residual_ffn_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _safe_mask(text_mask):
        if text_mask is None:
            return None, None
        valid_mask = text_mask.to(dtype=torch.bool)
        fallback = torch.zeros_like(valid_mask)
        fallback[:, 0] = True
        safe_mask = torch.where(
            valid_mask.any(dim=-1, keepdim=True), valid_mask, fallback,
        )
        return valid_mask, safe_mask

    def apply_rope(self, tensor, cos, sin):
        return apply_rope_pairs(tensor, cos, sin)

    def forward(self, x, text_mask=None):
        valid_mask, safe_mask = self._safe_mask(text_mask)
        h = self.norm1(x)
        gate_input = h
        batch, tokens, dim = h.shape
        q, k, v = self.qkv(h)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rope_cache.get(tokens, q.device, q.dtype)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)
        attention_mask = (
            safe_mask[:, None, None, :] if safe_mask is not None else None
        )
        h = scaled_dot_product_attention_gqa(
            q, k, v, attn_mask=attention_mask,
        )
        if self.head_gate is not None:
            gate = HEAD_GATE_SCALE * torch.sigmoid(self.head_gate(gate_input))
            h = h * gate.transpose(1, 2).unsqueeze(-1)
        h = h.transpose(1, 2).reshape(batch, tokens, dim)
        x = x + torch.tanh(self.residual_attn_gate) * self.attn_out(h)
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0)
        x = x + torch.tanh(self.residual_ffn_gate) * self.ffn(self.norm2(x))
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0)
        return x

class TextConditioningAdapter(nn.Module):
    """Bidirectionally enrich token states while preserving sequence length."""
    def __init__(self, input_dim, output_dim=1024,
                 transformer_dims=(2048, 1024, 1024), transformer_heads=(16, 8, 8),
                 transformer_kv_heads=None, transformer_ff_mult=3.0,
                 rope_theta=10000.0, use_head_gate=True):
        super().__init__()
        if len(transformer_dims) == 0:
            raise ValueError("text transformer must contain at least one layer")
        if len(transformer_dims) != len(transformer_heads):
            raise ValueError("text transformer dims and heads must have equal lengths")
        if transformer_kv_heads is None:
            transformer_kv_heads = transformer_heads
        if len(transformer_heads) != len(transformer_kv_heads):
            raise ValueError(
                "text transformer heads and KV heads must have equal lengths"
            )
        self.input_norm = RMSNorm(input_dim)
        self.transformer_input = nn.Linear(input_dim, transformer_dims[0])
        rope_caches = {}
        for dim, heads in zip(transformer_dims, transformer_heads):
            head_dim = dim // heads
            rope_caches.setdefault(
                head_dim, OneDRoPECache(head_dim, theta=rope_theta),
            )
        self.transformer_blocks = nn.ModuleList([
            BidirectionalTextTransformerBlock(
                dim, heads, kv_heads, transformer_ff_mult,
                rope_caches[dim // heads], use_head_gate,
            )
            for dim, heads, kv_heads in zip(
                transformer_dims, transformer_heads, transformer_kv_heads,
            )
        ])
        self.transformer_projections = nn.ModuleList([
            (
                nn.Linear(transformer_dims[index], transformer_dims[index + 1])
                if transformer_dims[index] != transformer_dims[index + 1]
                else nn.Identity()
            )
            for index in range(len(transformer_dims) - 1)
        ])
        self.transformer_norm = RMSNorm(transformer_dims[-1])
        self.output = (
            nn.Identity()
            if transformer_dims[-1] == output_dim
            else nn.Linear(transformer_dims[-1], output_dim)
        )

    def forward(self, text_hidden_states, text_condition_mask=None):
        x = self.transformer_input(self.input_norm(text_hidden_states))
        for index, block in enumerate(self.transformer_blocks):
            x = block(x, text_condition_mask)
            if index < len(self.transformer_projections):
                x = self.transformer_projections[index](x)
        x = self.transformer_norm(x)
        output = self.output(x)
        if text_condition_mask is not None:
            output = output.masked_fill(
                ~text_condition_mask.to(dtype=torch.bool)[..., None], 0,
            )
        return output
