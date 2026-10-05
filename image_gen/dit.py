"""Image-latent DiT architecture and its multimodal attention blocks."""

from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from image_gen.attention import MMDiTJointAttention, TwoDRoPECache
from image_gen.joint_attention import (
    JointMHLAAttentionMetadata,
    JointMHLALayoutCache,
    _joint_mhla_prepare_padded_layout,
    joint_mhla_attention,
)
from image_gen.layers import (
    HEAD_GATE_SCALE,
    RMSNorm,
    apply_rope_pairs,
    scaled_dot_product_attention_gqa,
    validate_attention_head_counts,
    validate_attention_mask_shape,
    validate_spatial_token_grid,
    ResolutionEmbedding,
    TimestepEmbedding,
)
from image_gen.text_conditioning import GroupedQueryProjection, OneDRoPECache
from core.layers import (
    GatedConv2d,
    PreNormConvFFNResidual2d,
    PreNormGatedConvFFNResidual2d,
    RMSNorm2d,
)


class JointMHLA(nn.Module):
    """Stream-aware Multi-Head Linear Attention for the MMDiT streams.

    Latent and image-digest tokens are partitioned into 2D spatial blocks and
    text tokens into 1D blocks.  Every block produces a KV summary; each query
    block then mixes all source summaries using content-dependent block-level
    weights before applying token-level kernel attention.  This preserves the
    latent/image/text joint-attention path without constructing an N x N
    attention matrix.
    """
    def __init__(self, dim, heads, kv_heads=None, rope_cache=None,
                 text_rope_cache=None, latent_blocks=16, image_blocks=4,
                 text_blocks=4, use_head_gate=True, backend="auto",
                 layout_cache=None, recompute_output=False):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        validate_attention_head_counts(heads, kv_heads, "Joint MHLA")
        if dim % heads or (dim // heads) % 4:
            raise ValueError("Joint MHLA head dimension must be divisible by 4")
        if latent_blocks <= 0 or image_blocks <= 0 or text_blocks <= 0:
            raise ValueError("Joint MHLA block counts must be positive")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.latent_blocks = int(latent_blocks)
        self.image_blocks = int(image_blocks)
        self.text_blocks = int(text_blocks)
        self.layout_cache = (
            layout_cache if layout_cache is not None else JointMHLALayoutCache()
        )
        if backend not in {"auto", "naive", "vectorized", "triton"}:
            raise ValueError(f"unknown Joint MHLA backend: {backend}")
        self.backend = backend
        self.recompute_output = bool(recompute_output)
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
        self.latent_q_norm = RMSNorm(self.head_dim)
        self.latent_k_norm = RMSNorm(self.head_dim)
        self.image_q_norm = RMSNorm(self.head_dim)
        self.image_k_norm = RMSNorm(self.head_dim)
        self.text_q_norm = RMSNorm(self.head_dim)
        self.text_k_norm = RMSNorm(self.head_dim)
        self.latent_head_gate = nn.Linear(dim, heads) if use_head_gate else None
        self.image_head_gate = nn.Linear(dim, heads) if use_head_gate else None
        self.text_head_gate = nn.Linear(dim, heads) if use_head_gate else None
        for gate in (
            self.latent_head_gate, self.image_head_gate, self.text_head_gate,
        ):
            if gate is not None:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
        # A small learned prior lets the model prefer or suppress modality
        # pairs while the content-dependent block mixing remains dominant.
        self.modality_bias = nn.Parameter(torch.zeros(3, 3))
        self.latent_out = nn.Linear(dim, dim)
        self.image_out = nn.Linear(dim, dim)
        self.text_out = nn.Linear(dim, dim)

    def _apply_rope(self, tensor, cos, sin):
        return apply_rope_pairs(tensor, cos, sin)

    @staticmethod
    def _grid_block_indices(height, width, target_blocks, offset, device):
        target_blocks = max(1, min(int(target_blocks), int(height * width)))
        aspect = float(height) / max(float(width), 1.0)
        rows = max(1, min(int(height), round(math.sqrt(target_blocks * aspect))))
        cols = max(1, min(int(width), round(math.sqrt(target_blocks / aspect))))
        while rows * cols > target_blocks:
            if cols >= rows and cols > 1:
                cols -= 1
            elif rows > 1:
                rows -= 1
            else:
                break
        row_ids = torch.arange(height, device=device) * rows // height
        col_ids = torch.arange(width, device=device) * cols // width
        block_ids = (row_ids[:, None] * cols + col_ids[None, :]).flatten()
        return [
            torch.nonzero(block_ids == block, as_tuple=False).flatten() + offset
            for block in range(rows * cols)
        ]

    @staticmethod
    def _text_block_indices(tokens, target_blocks, offset, device):
        block_count = max(1, min(int(target_blocks), int(tokens)))
        return [
            indices + offset
            for indices in torch.arange(tokens, device=device).chunk(block_count)
        ]

    def _make_layout(self, latent_height, latent_width, image_height, image_width,
                     text_tokens, device):
        cache_key = (
            device.type, device.index,
            int(latent_height), int(latent_width),
            int(image_height), int(image_width), int(text_tokens),
            self.latent_blocks, self.image_blocks, self.text_blocks,
        )
        cached = self.layout_cache.get(cache_key)
        if cached is not None:
            return cached
        latent_count = int(latent_height * latent_width)
        image_count = int(image_height * image_width)
        blocks = self._grid_block_indices(
            latent_height, latent_width, self.latent_blocks, 0, device,
        )
        modalities = [0] * len(blocks)
        blocks.extend(self._grid_block_indices(
            image_height, image_width, self.image_blocks,
            latent_count, device,
        ))
        modalities.extend([1] * (len(blocks) - len(modalities)))
        blocks.extend(self._text_block_indices(
            text_tokens, self.text_blocks, latent_count + image_count, device,
        ))
        modalities.extend([2] * (len(blocks) - len(modalities)))
        block_modalities = torch.tensor(
            modalities, device=device, dtype=torch.long,
        )
        padded_layout = _joint_mhla_prepare_padded_layout(blocks, device)
        cached = (blocks, block_modalities, padded_layout)
        self.layout_cache.put(cache_key, cached)
        return cached

    def prepare_attention_metadata(
        self,
        latent_height,
        latent_width,
        image_height,
        image_width,
        text_tokens,
        batch_size,
        device,
        text_context_mask=None,
    ):
        """Build MHLA tensors once for all blocks in one DiT forward."""
        if min(latent_height, latent_width, image_height, image_width) <= 0:
            raise ValueError("MHLA spatial dimensions must be positive")
        if text_tokens <= 0 or batch_size <= 0:
            raise ValueError("MHLA text token and batch counts must be positive")
        validate_attention_mask_shape(
            text_context_mask, batch_size, text_tokens, "text context",
        )
        block_indices, block_modalities, padded_layout = self._make_layout(
            latent_height, latent_width, image_height, image_width,
            text_tokens, device,
        )
        latent_count = int(latent_height * latent_width)
        image_count = int(image_height * image_width)
        total_tokens = latent_count + image_count + int(text_tokens)
        valid = torch.ones(
            batch_size, total_tokens, device=device, dtype=torch.bool,
        )
        if text_context_mask is not None:
            valid[:, latent_count + image_count:] = text_context_mask.to(
                device=device, dtype=torch.bool,
            )
        return JointMHLAAttentionMetadata(
            block_indices, block_modalities, padded_layout, valid,
        )

    def _project(self, tokens, projection, q_norm, k_norm, head_gate):
        query, key, value = projection(tokens)
        gate = (
            HEAD_GATE_SCALE * torch.sigmoid(head_gate(tokens))
            if head_gate is not None else None
        )
        query = q_norm(query)
        key = k_norm(key)
        # Native RMSNorm implementations may choose a wider accumulation
        # dtype.  Triton MHLA requires Q/K/V to have one common dtype, so
        # restore the projection dtype at this boundary.
        if query.dtype != value.dtype:
            query = query.to(dtype=value.dtype)
        if key.dtype != value.dtype:
            key = key.to(dtype=value.dtype)
        return query, key, value, gate

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
        attention_metadata=None,
        timing=None,
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
        latent_q, latent_k = self._apply_rope(
            latent_q, *self.rope_cache.get(
                latent_height, latent_width, latent_q.device, latent_q.dtype,
            )
        ), self._apply_rope(
            latent_k, *self.rope_cache.get(
                latent_height, latent_width, latent_k.device, latent_k.dtype,
            )
        )
        image_q, image_k = self._apply_rope(
            image_q, *self.rope_cache.get(
                image_height, image_width, image_q.device, image_q.dtype,
            )
        ), self._apply_rope(
            image_k, *self.rope_cache.get(
                image_height, image_width, image_k.device, image_k.dtype,
            )
        )
        text_q, text_k = self._apply_rope(
            text_q, *self.text_rope_cache.get(
                text_q.shape[2], text_q.device, text_q.dtype,
            )
        ), self._apply_rope(
            text_k, *self.text_rope_cache.get(
                text_k.shape[2], text_k.device, text_k.dtype,
            )
        )

        query = torch.cat((latent_q, image_q, text_q), dim=2)
        key = torch.cat((latent_k, image_k, text_k), dim=2)
        value = torch.cat((latent_v, image_v, text_v), dim=2)
        latent_count = latent_tokens.shape[1]
        image_count = image_context_tokens.shape[1]
        text_count = text_context_tokens.shape[1]
        if attention_metadata is None:
            attention_metadata = self.prepare_attention_metadata(
                latent_height, latent_width, image_height, image_width,
                text_count, query.shape[0], query.device,
                text_context_mask,
            )
        measure = timing.measure if timing is not None else lambda _: nullcontext()
        with measure("mhla_forward"):
            output = joint_mhla_attention(
                query, key, value,
                attention_metadata.block_indices,
                attention_metadata.block_modalities,
                attention_metadata.valid,
                self.heads, self.kv_heads, self.modality_bias, self.backend,
                attention_metadata.padded_layout, self.recompute_output,
                timing,
            )

        if latent_gate is not None:
            gate = torch.cat((latent_gate, image_gate, text_gate), dim=1)
            output = output * gate.transpose(1, 2).unsqueeze(-1)
        latent_output, image_output, text_output = output.split(
            (latent_count, image_count, text_context_tokens.shape[1]), dim=2,
        )

        def merge(stream):
            return stream.transpose(1, 2).reshape(
                stream.shape[0], stream.shape[2], -1,
            )

        return (
            self.latent_out(merge(latent_output)),
            self.image_out(merge(image_output)),
            self.text_out(merge(text_output)),
        )


class MMDiTBlock(nn.Module):
    """Timestep-conditioned multimodal DiT block with joint attention."""
    def __init__(self, dim, heads, kv_heads=None, rope_cache=None,
                 text_rope_cache=None, use_head_gate=True, use_mhla=False,
                 mhla_latent_blocks=16, mhla_image_blocks=4,
                 mhla_text_blocks=4, mhla_backend="auto",
                 mhla_layout_cache=None, mhla_recompute_output=False):
        super().__init__()
        self.dim = dim
        self.use_mhla = bool(use_mhla)
        self.latent_attn_norm = RMSNorm(dim, elementwise_affine=False)
        self.image_attn_norm = RMSNorm(dim, elementwise_affine=False)
        self.text_attn_norm = RMSNorm(dim, elementwise_affine=False)
        if self.use_mhla:
            self.joint_attn = JointMHLA(
                dim, heads, kv_heads=kv_heads, rope_cache=rope_cache,
                text_rope_cache=text_rope_cache, use_head_gate=use_head_gate,
                latent_blocks=mhla_latent_blocks,
                image_blocks=mhla_image_blocks,
                text_blocks=mhla_text_blocks,
                backend=mhla_backend,
                layout_cache=mhla_layout_cache,
                recompute_output=mhla_recompute_output,
            )
        else:
            self.joint_attn = MMDiTJointAttention(
                dim, heads, kv_heads=kv_heads, rope_cache=rope_cache,
                text_rope_cache=text_rope_cache, use_head_gate=use_head_gate,
            )
        self.latent_ffn_norm = RMSNorm(dim, elementwise_affine=False)
        self.image_ffn_norm = RMSNorm(dim, elementwise_affine=False)
        self.text_ffn_norm = RMSNorm(dim, elementwise_affine=False)
        hidden = int(dim * 4)

        def make_ffn():
            return nn.Sequential(
                nn.Linear(dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, dim),
            )

        self.latent_ffn = make_ffn()
        self.image_ffn = make_ffn()
        self.text_ffn = make_ffn()
        # Each stream gets independent Ada modulation and residual gates.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 18))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    @staticmethod
    def _modulate(norm, tokens, shift, scale):
        return norm(tokens) * (1 + scale[:, None]) + shift[:, None]

    def forward(
        self,
        latent_tokens,
        image_context_tokens,
        text_context_tokens,
        ada_condition,
        latent_height,
        latent_width,
        image_height,
        image_width,
        text_context_mask=None,
        attention_mask=None,
        mhla_attention_metadata=None,
        timing=None,
    ):
        modulation = self.ada(ada_condition).chunk(18, dim=-1)
        (
            latent_attn_shift, latent_attn_scale, latent_attn_gate,
            image_attn_shift, image_attn_scale, image_attn_gate,
            text_attn_shift, text_attn_scale, text_attn_gate,
            latent_ffn_shift, latent_ffn_scale, latent_ffn_gate,
            image_ffn_shift, image_ffn_scale, image_ffn_gate,
            text_ffn_shift, text_ffn_scale, text_ffn_gate,
        ) = modulation
        joint_attention_inputs = (
            self._modulate(
                self.latent_attn_norm, latent_tokens,
                latent_attn_shift, latent_attn_scale,
            ),
            self._modulate(
                self.image_attn_norm, image_context_tokens,
                image_attn_shift, image_attn_scale,
            ),
            self._modulate(
                self.text_attn_norm, text_context_tokens,
                text_attn_shift, text_attn_scale,
            ),
            latent_height, latent_width, image_height, image_width,
            text_context_mask,
        )
        measure = timing.measure if timing is not None else lambda _: nullcontext()
        with measure("dit_attention"):
            if self.use_mhla:
                latent_attn, image_attn, text_attn = self.joint_attn(
                    *joint_attention_inputs,
                    mhla_attention_metadata,
                    timing=timing,
                )
            else:
                latent_attn, image_attn, text_attn = self.joint_attn(
                    *joint_attention_inputs,
                    attention_mask,
                )
            latent_tokens = torch.addcmul(
                latent_tokens, latent_attn, latent_attn_gate[:, None],
            )
            image_context_tokens = torch.addcmul(
                image_context_tokens, image_attn, image_attn_gate[:, None],
            )
            text_context_tokens = torch.addcmul(
                text_context_tokens, text_attn, text_attn_gate[:, None],
            )

        with measure("dit_ffn"):
            latent_tokens = torch.addcmul(
                latent_tokens,
                self.latent_ffn(
                    self._modulate(
                        self.latent_ffn_norm, latent_tokens,
                        latent_ffn_shift, latent_ffn_scale,
                    )
                ),
                latent_ffn_gate[:, None],
            )
            image_context_tokens = torch.addcmul(
                image_context_tokens,
                self.image_ffn(
                    self._modulate(
                        self.image_ffn_norm, image_context_tokens,
                        image_ffn_shift, image_ffn_scale,
                    )
                ),
                image_ffn_gate[:, None],
            )
            text_context_tokens = torch.addcmul(
                text_context_tokens,
                self.text_ffn(
                    self._modulate(
                        self.text_ffn_norm, text_context_tokens,
                        text_ffn_shift, text_ffn_scale,
                    )
                ),
                text_ffn_gate[:, None],
            )
        if text_context_mask is not None:
            text_context_tokens = text_context_tokens.masked_fill(
                ~text_context_mask[..., None].to(dtype=torch.bool), 0,
            )
        return latent_tokens, image_context_tokens, text_context_tokens


def group_norm(channels):
    if channels <= 0:
        raise ValueError("GroupNorm channels must be positive")
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class SpatialRMSNorm(nn.Module):
    """Channel-wise RMS normalization for a possibly 1x1 NCHW feature map."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, features):
        output = features * torch.rsqrt(
            features.float().pow(2).mean(dim=1, keepdim=True) + self.eps
        )
        return output.to(dtype=features.dtype) * self.weight.to(
            dtype=features.dtype
        )[None, :, None, None]

def context_group_norm(channels):
    """Choose a GroupNorm safe for batch=1 and a 1x1 spatial context grid."""
    if channels <= 0:
        raise ValueError("context channels must be positive")
    if channels == 1:
        return SpatialRMSNorm(channels)
    groups = min(32, channels // 2)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class ImageContextEmbedder(nn.Module):
    """Compress shared image features into a low-resolution context grid."""
    def __init__(self, in_channels, context_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(3):
            self.layers.append(nn.Sequential(
                nn.Conv2d(
                    channels, context_dim,
                    kernel_size=4, stride=2, padding=1,
                ),
                context_group_norm(context_dim),
                nn.SiLU(),
            ))
            channels = context_dim

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError(
                "ImageContextEmbedder expects NCHW features, "
                f"got shape {tuple(features.shape)}"
            )
        height, width = features.shape[-2:]
        if height < 8 or width < 8:
            raise ValueError(
                "ImageContextEmbedder requires spatial dimensions >= 8 "
                f"before three stride-2 convolutions, got {height}x{width}"
            )
        for layer in self.layers:
            features = layer(features)
        return features

class ContextSelfAttention(nn.Module):
    """Self-attention with separate 2D image and 1D text RoPE coordinates."""
    def __init__(self, dim, heads, kv_heads=None, image_rope_cache=None,
                 text_rope_cache=None, use_head_gate=True):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        validate_attention_head_counts(heads, kv_heads, "context transformer")
        if dim % heads or (dim // heads) % 4:
            raise ValueError(
                "context head dimension must be divisible by 4 for 2D RoPE"
            )
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.image_rope_cache = (
            image_rope_cache
            if image_rope_cache is not None
            else TwoDRoPECache(self.head_dim)
        )
        self.text_rope_cache = (
            text_rope_cache
            if text_rope_cache is not None
            else OneDRoPECache(self.head_dim)
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

    @staticmethod
    def _safe_mask(context_mask):
        if context_mask is None:
            return None, None
        valid_mask = context_mask.to(dtype=torch.bool)
        fallback = torch.zeros_like(valid_mask)
        fallback[:, 0] = True
        safe_mask = torch.where(
            valid_mask.any(dim=-1, keepdim=True), valid_mask, fallback,
        )
        return valid_mask, safe_mask

    def forward(self, x, image_height, image_width, image_token_count,
                context_mask=None):
        batch, tokens, dim = x.shape
        if image_token_count != int(image_height) * int(image_width):
            raise ValueError(
                "context image token count does not match image_height*image_width"
            )
        if image_token_count < 0 or image_token_count > tokens:
            raise ValueError("context image token count must be within the token sequence")
        validate_attention_mask_shape(
            context_mask, batch, tokens, "context",
        )
        valid_mask, safe_mask = self._safe_mask(context_mask)
        q, k, v = self.qkv(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        image_q, text_q = q[:, :, :image_token_count], q[:, :, image_token_count:]
        image_k, text_k = k[:, :, :image_token_count], k[:, :, image_token_count:]
        image_cos, image_sin = self.image_rope_cache.get(
            image_height, image_width, q.device, q.dtype,
        )
        q = torch.cat((
            self.apply_rope(image_q, image_cos, image_sin),
            self.apply_rope(
                text_q,
                *self.text_rope_cache.get(
                    tokens - image_token_count, q.device, q.dtype,
                ),
            ),
        ), dim=2)
        k = torch.cat((
            self.apply_rope(image_k, image_cos, image_sin),
            self.apply_rope(
                text_k,
                *self.text_rope_cache.get(
                    tokens - image_token_count, k.device, k.dtype,
                ),
            ),
        ), dim=2)
        attention_mask = (
            safe_mask[:, None, None, :] if safe_mask is not None else None
        )
        output = scaled_dot_product_attention_gqa(
            q, k, v, attn_mask=attention_mask,
        )
        if self.head_gate is not None:
            gate = HEAD_GATE_SCALE * torch.sigmoid(self.head_gate(x))
            output = output * gate.transpose(1, 2).unsqueeze(-1)
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        output = self.out(output)
        if valid_mask is not None:
            output = output.masked_fill(~valid_mask[..., None], 0)
        return output

class ContextTransformerBlock(nn.Module):
    """AdaRMS-conditioned multimodal context transformer block."""
    def __init__(self, dim, heads, condition_dim, kv_heads=None,
                 image_rope_cache=None, text_rope_cache=None,
                 use_head_gate=True):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.self_attn = ContextSelfAttention(
            dim, heads, kv_heads, image_rope_cache, text_rope_cache,
            use_head_gate,
        )
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        hidden = dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, condition, image_height, image_width,
                image_token_count, context_mask=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(condition).chunk(
            6, dim=-1,
        )
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = torch.addcmul(
            x,
            self.self_attn(
                h, image_height, image_width, image_token_count, context_mask,
            ),
            gate1[:, None],
        )
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return torch.addcmul(x, self.mlp(h), gate2[:, None])

class ContextTransformer(nn.Module):
    """Fuse image digest and adapted text tokens before the main DiT."""
    def __init__(self, image_dim, context_dim, condition_dim,
                 depth=2, heads=16, kv_heads=None, use_head_gate=True):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        validate_attention_head_counts(heads, kv_heads, "context transformer")
        if context_dim % heads:
            raise ValueError("context dimension must be divisible by context heads")
        if (context_dim // heads) % 4:
            raise ValueError(
                "context head dimension must be divisible by 4 for 2D RoPE"
            )
        self.context_dim = context_dim
        self.image_embedder = ImageContextEmbedder(image_dim, context_dim)
        image_rope_cache = TwoDRoPECache(context_dim // heads)
        text_rope_cache = OneDRoPECache(context_dim // heads)
        self.image_modality = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.text_modality = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.condition_projection = nn.Linear(condition_dim, context_dim)
        self.blocks = nn.ModuleList([
            ContextTransformerBlock(
                context_dim, heads, context_dim, kv_heads,
                image_rope_cache=image_rope_cache,
                text_rope_cache=text_rope_cache,
                use_head_gate=use_head_gate,
            )
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(context_dim)

    def forward(self, image_features, text_condition_tokens,
                text_condition_mask, ada_condition):
        image_features = self.image_embedder(image_features)
        grid_h, grid_w = image_features.shape[-2:]
        image_tokens = image_features.flatten(2).transpose(1, 2)
        image_tokens = image_tokens + self.image_modality.to(
            dtype=image_tokens.dtype,
        )

        text_condition_tokens = text_condition_tokens.to(dtype=image_tokens.dtype)
        text_tokens = text_condition_tokens + self.text_modality.to(
            dtype=text_condition_tokens.dtype,
        )
        if text_condition_mask is None:
            text_mask = torch.ones(
                text_condition_tokens.shape[:2],
                device=text_condition_tokens.device, dtype=torch.bool,
            )
            context_mask = None
        else:
            text_mask = text_condition_mask.to(dtype=torch.bool)
            context_mask = F.pad(text_mask, (image_tokens.shape[1], 0), value=True)
        context_tokens = torch.cat((image_tokens, text_tokens), dim=1)
        context_condition = self.condition_projection(ada_condition)
        for block in self.blocks:
            context_tokens = block(
                context_tokens, context_condition, grid_h, grid_w,
                image_tokens.shape[1], context_mask,
            )
        context_tokens = self.final_norm(context_tokens)
        if context_mask is not None:
            context_tokens = context_tokens.masked_fill(
                ~context_mask[..., None], 0,
            )
        image_token_count = image_tokens.shape[1]
        return (
            context_tokens[:, :image_token_count],
            context_tokens[:, image_token_count:],
            text_mask,
            grid_h,
            grid_w,
        )

class ResidualConvFFNBlock(PreNormConvFFNResidual2d):
    """Checkpoint-compatible wrapper for the shared pre-norm Conv FFN."""
    def __init__(self, channels, hidden_channels=None, zero_last=False):
        # Preserve image_gen's historical behavior where falsy widths select
        # the default width (including an explicitly supplied zero).
        hidden_channels = hidden_channels or channels * 2
        super().__init__(
            channels,
            hidden_channels,
            zero_last=zero_last,
            state_layout="named",
        )

class GatedResidualConvFFNBlock(PreNormGatedConvFFNResidual2d):
    """Checkpoint-compatible wrapper for the shared gated Conv FFN."""
    def __init__(self, channels, hidden_channels=None):
        hidden_channels = hidden_channels or channels * 2
        super().__init__(channels, hidden_channels, state_layout="named")

class LatentDownsample(nn.Module):
    """Downsample the VAE latent and refine it before tokenization."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = GatedConv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1,
        )
        self.norm = group_norm(out_channels)
        self.refine = GatedResidualConvFFNBlock(out_channels)

    def forward(self, latent):
        return self.refine(self.norm(self.downsample(latent)))

class LatentUpsample(nn.Module):
    """Restore the VAE latent grid after DiT token processing."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = RMSNorm2d(in_channels)
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1,
        )
        self.refine = ResidualConvFFNBlock(
            out_channels, zero_last=True,
        )
        # Preserve the previous zero-output initialization of the DiT head.
        nn.init.zeros_(self.upsample.weight)
        nn.init.zeros_(self.upsample.bias)

    def forward(self, features):
        return self.refine(self.upsample(self.norm(features)))

class DiT(nn.Module):
    def __init__(self, image_channels, context_dim,
                 dim=768, depth=12, heads=12, patch_size=2,
                 reference_height=32, reference_width=32,
                 context_depth=2, context_heads=16,
                 gradient_checkpointing=False, kv_heads=None,
                 context_kv_heads=None, use_head_gate=True,
                 attention_pattern="mhla3-full1",
                 mhla_latent_blocks=16, mhla_image_blocks=4,
                 mhla_text_blocks=4, mhla_backend="auto",
                 mhla_recompute_output=False):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        if context_kv_heads is None:
            context_kv_heads = context_heads
        validate_attention_head_counts(heads, kv_heads, "MMDiT")
        validate_attention_head_counts(
            context_heads, context_kv_heads, "context transformer",
        )
        if attention_pattern not in {"full", "mhla", "mhla3-full1"}:
            raise ValueError(f"unknown attention pattern: {attention_pattern}")
        if mhla_backend not in {"auto", "naive", "vectorized", "triton"}:
            raise ValueError(f"unknown MHLA backend: {mhla_backend}")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if patch_size != 2:
            raise ValueError("the convolutional latent stem currently requires patch_size=2")
        if image_channels <= 0:
            raise ValueError("image_channels must be positive")
        self.image_channels = image_channels
        self.in_channels = image_channels
        self.patch_size = patch_size
        self.dim = dim
        self.attention_pattern = attention_pattern
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.image_input_dim = dim
        self.mhla_layout_cache = JointMHLALayoutCache()
        self.image_input_projection = LatentDownsample(
            image_channels, self.image_input_dim,
        )
        self.latent_modality = nn.Parameter(torch.zeros(1, 1, dim))
        self.time = TimestepEmbedding(dim)
        self.resolution = ResolutionEmbedding(dim, reference_height // patch_size, reference_width // patch_size)
        self.condition = nn.Sequential(
            nn.Linear(dim + self.resolution.output_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.context_transformer = ContextTransformer(
            dim, context_dim, dim, depth=context_depth, heads=context_heads,
            kv_heads=context_kv_heads, use_head_gate=use_head_gate,
        )
        self.rope_cache = TwoDRoPECache(dim // heads)
        self.text_rope_cache = OneDRoPECache(dim // heads)
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                dim, heads, kv_heads=kv_heads,
                rope_cache=self.rope_cache,
                text_rope_cache=self.text_rope_cache,
                use_head_gate=use_head_gate,
                use_mhla=(
                    attention_pattern == "mhla"
                    or (
                        attention_pattern == "mhla3-full1"
                        and index % 4 != 3
                    )
                ),
                mhla_latent_blocks=mhla_latent_blocks,
                mhla_image_blocks=mhla_image_blocks,
                mhla_text_blocks=mhla_text_blocks,
                mhla_backend=mhla_backend,
                mhla_layout_cache=self.mhla_layout_cache,
                mhla_recompute_output=mhla_recompute_output,
            )
            for index in range(depth)
        ])
        self.context_projection = (
            nn.Identity()
            if context_dim == dim
            else nn.Linear(context_dim, dim)
        )
        self.final_norm = RMSNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.latent_output_projection = LatentUpsample(dim, image_channels)

    def forward(self, x, timestep, text_context_tokens,
                text_context_mask=None, timing=None):
        if x.ndim != 4:
            raise ValueError(
                "DiT expects latent input with shape (B, C, H, W), "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} latent channels, got {x.shape[1]}")
        latent_height, latent_width = x.shape[-2:]
        if (
            latent_height < 16 or latent_width < 16
            or latent_height % 2 or latent_width % 2
        ):
            raise ValueError(
                "DiT latent height and width must be even and >= 16 so the "
                "latent stem and three context downsamplers preserve a valid grid, "
                f"got {latent_height}x{latent_width}"
            )
        measure = timing.measure if timing is not None else lambda _: nullcontext()
        with measure("dit_input_projection"):
            features = self.image_input_projection(x)
        grid_h, grid_w = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.latent_modality.to(dtype=tokens.dtype)
        timestep_condition = self.time(timestep)
        resolution_condition = self.resolution(
            grid_h, grid_w, x.shape[0], x.device, timestep_condition.dtype,
        )
        ada_condition = self.condition(torch.cat([timestep_condition, resolution_condition], dim=-1))
        with measure("dit_context_transformer"):
            (
                image_context_tokens,
                text_context_tokens,
                text_context_mask,
                context_grid_h,
                context_grid_w,
            ) = self.context_transformer(
                features, text_context_tokens, text_context_mask, ada_condition,
            )
            image_context_tokens = self.context_projection(image_context_tokens)
            text_context_tokens = self.context_projection(text_context_tokens)
        if text_context_mask is not None:
            text_context_mask = text_context_mask.to(
                device=tokens.device, dtype=torch.bool,
            )
        attention_mask = None
        has_full_attention = any(not block.use_mhla for block in self.blocks)
        if text_context_mask is not None and has_full_attention:
            attention_mask = F.pad(
                text_context_mask,
                (tokens.shape[1] + image_context_tokens.shape[1], 0),
                value=True,
            )[:, None, None, :]
        mhla_attention_metadata = None
        for block in self.blocks:
            if block.use_mhla:
                mhla_attention_metadata = block.joint_attn.prepare_attention_metadata(
                    grid_h, grid_w, context_grid_h, context_grid_w,
                    text_context_tokens.shape[1], tokens.shape[0], tokens.device,
                    text_context_mask,
                )
                break
        with measure("dit_main_blocks"):
            for block in self.blocks:
                if self.training and self.gradient_checkpointing:
                    tokens, image_context_tokens, text_context_tokens = checkpoint(
                        block,
                        tokens,
                        image_context_tokens,
                        text_context_tokens,
                        ada_condition,
                        grid_h,
                        grid_w,
                        context_grid_h,
                        context_grid_w,
                    text_context_mask,
                    attention_mask,
                    mhla_attention_metadata,
                    timing,
                    use_reentrant=False,
                )
                else:
                    tokens, image_context_tokens, text_context_tokens = block(
                        tokens,
                        image_context_tokens,
                        text_context_tokens,
                        ada_condition,
                        grid_h,
                        grid_w,
                        context_grid_h,
                        context_grid_w,
                    text_context_mask,
                    attention_mask,
                    mhla_attention_metadata,
                    timing,
                )
        with measure("dit_output_projection"):
            final_shift, final_scale = self.final_ada(ada_condition).chunk(2, dim=-1)
            tokens = self.final_norm(tokens) * (1 + final_scale[:, None]) + final_shift[:, None]
            features = tokens.transpose(1, 2).reshape(
                x.shape[0], self.dim, grid_h, grid_w,
            )
            return self.latent_output_projection(features)
