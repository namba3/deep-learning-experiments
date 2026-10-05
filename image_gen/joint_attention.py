"""Joint MHLA layouts, reference implementations, and backend orchestration."""

from contextlib import nullcontext
import math
import os
import torch
import torch.nn.functional as F

from image_gen.joint_attention_triton import (
    _TRITON_IMPORT_ERROR as _TRITON_IMPORT_ERROR,
    triton as triton,
)
if triton is not None:
    from image_gen.joint_attention_triton import (
        _joint_mhla_block_means_kernel as _joint_mhla_block_means_kernel,
        _joint_mhla_block_statistics_kernel as _joint_mhla_block_statistics_kernel,
        _joint_mhla_output_kernel as _joint_mhla_output_kernel,
        _joint_mhla_backward_mix_kernel as _joint_mhla_backward_mix_kernel,
        _joint_mhla_backward_summary_kernel as _joint_mhla_backward_summary_kernel,
        _joint_mhla_backward_query_kernel as _joint_mhla_backward_query_kernel,
        _joint_mhla_backward_kv_kernel as _joint_mhla_backward_kv_kernel,
    )



def _joint_mhla_prepare_padded_layout(block_indices, device):
    """Prepare a padded block-major gather layout for vectorized MHLA."""
    if not block_indices:
        raise ValueError("Joint MHLA requires at least one token block")
    block_count = len(block_indices)
    max_tokens = max(int(indices.numel()) for indices in block_indices)
    gather_index = torch.zeros(
        block_count, max_tokens, device=device, dtype=torch.long,
    )
    block_token_mask = torch.zeros(
        block_count, max_tokens, device=device, dtype=torch.bool,
    )
    for block, indices in enumerate(block_indices):
        token_count = int(indices.numel())
        gather_index[block, :token_count] = indices
        block_token_mask[block, :token_count] = True
    return gather_index, block_token_mask

class JointMHLALayoutCache:
    """Share geometry-only MHLA layouts across all DiT blocks."""

    def __init__(self, max_entries=16):
        self.max_entries = max(int(max_entries), 1)
        self._cache = {}

    def clear(self):
        self._cache.clear()

    @staticmethod
    def _normal_tensor(tensor):
        """Keep cached geometry tensors usable by a later autograd pass.

        Sampling runs under ``torch.inference_mode()``.  If a new resolution
        is encountered there, the resulting layout tensors are inference
        tensors too.  Reusing them as inputs to the custom MHLA autograd
        Function during the next training batch raises
        ``Inference tensors cannot be saved for backward``.  Geometry is
        tiny and created only on cache misses, so make a normal detached copy
        when needed.
        """
        is_inference = getattr(torch, "is_inference", None)
        if is_inference is None or not is_inference(tensor):
            return tensor
        with torch.inference_mode(False):
            return tensor.detach().clone()

    @classmethod
    def _normalize_layout(cls, value):
        block_indices, block_modalities, padded_layout = value
        normalized_blocks = [
            cls._normal_tensor(indices) for indices in block_indices
        ]
        normalized_modalities = cls._normal_tensor(block_modalities)
        normalized_padded_layout = tuple(
            cls._normal_tensor(layout_tensor)
            for layout_tensor in padded_layout
        )
        return (
            normalized_blocks,
            normalized_modalities,
            normalized_padded_layout,
        )

    def get(self, key):
        cached = self._cache.get(key)
        if cached is not None:
            normalized = self._normalize_layout(cached)
            self._cache[key] = normalized
            cached = normalized
        return cached

    def put(self, key, value):
        self._cache[key] = self._normalize_layout(value)
        while len(self._cache) > self.max_entries:
            self._cache.pop(next(iter(self._cache)))

class JointMHLAAttentionMetadata:
    """Per-forward MHLA geometry and validity tensors shared by DiT blocks."""

    __slots__ = (
        "block_indices", "block_modalities", "padded_layout", "valid",
    )

    def __init__(self, block_indices, block_modalities, padded_layout, valid):
        self.block_indices = block_indices
        self.block_modalities = block_modalities
        self.padded_layout = padded_layout
        self.valid = valid

def _joint_mhla_block_statistics(key, value, valid_mask, block_indices):
    """Reference implementation of per-block KV summary construction."""
    summaries = []
    normalizers = []
    for indices in block_indices:
        block_key = key.index_select(2, indices)
        block_value = value.index_select(2, indices)
        mask = valid_mask.index_select(1, indices).to(dtype=block_key.dtype)
        mask = mask[:, None, :, None]
        feature_key = (F.elu(block_key) + 1.0) * mask
        block_value = block_value * mask
        summaries.append(torch.einsum(
            "bhnd,bhne->bhde", feature_key, block_value,
        ))
        normalizers.append(feature_key.sum(dim=2))
    return torch.stack(summaries, dim=2), torch.stack(normalizers, dim=2)


def _joint_mhla_block_mix(
    query_mean, key_mean, block_modalities, block_valid, modality_bias,
):
    """Build the small block-level softmax used by MHLA backward."""
    with torch.no_grad():
        logits = torch.einsum(
            "bkmd,bknd->bkmn", query_mean, key_mean,
        ) / math.sqrt(query_mean.shape[-1])
        modality_bias = modality_bias.to(dtype=logits.dtype)
        modality_bias = modality_bias[
            block_modalities[:, None], block_modalities[None, :],
        ]
        logits = logits + modality_bias[None, None]
        logits = logits.masked_fill(
            ~block_valid[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        return torch.softmax(logits, dim=-1)


def _joint_mhla_triton_available(query, key, value):
    return (
        triton is not None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
    )

def _joint_mhla_triton_unavailable_reason(query, key, value):
    reasons = []
    if triton is None:
        reasons.append(f"triton import failed: {_TRITON_IMPORT_ERROR}")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        reasons.append(
            f"cuda flags query={query.is_cuda} key={key.is_cuda} value={value.is_cuda}"
        )
    if not (
        query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
    ):
        reasons.append(
            f"dtypes query={query.dtype} key={key.dtype} value={value.dtype}"
        )
    return "; ".join(reasons) or "unknown Triton availability failure"




def _joint_mhla_triton_backward(
    query, key, value, block_modalities, valid_mask, modality_bias,
    grad_output, output, query_mean, key_mean, summary, normalizer,
    block_valid, padded_layout, heads, kv_heads,
):
    """Compute MHLA token gradients with Triton kernels."""
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    grad_output = grad_output.contiguous()
    output = output.contiguous()
    valid_mask = valid_mask.to(device=query.device, dtype=torch.bool).contiguous()
    layout, block_token_mask = padded_layout
    layout = layout.contiguous()
    block_token_mask = block_token_mask.contiguous()
    block_valid = block_valid.contiguous()
    query_mean = query_mean.contiguous()
    key_mean = key_mean.contiguous()
    summary = summary.contiguous()
    normalizer = normalizer.contiguous()
    batch_size, _, token_count, head_dim = query.shape
    block_count, max_tokens = layout.shape
    group = heads // kv_heads
    mix = _joint_mhla_block_mix(
        query_mean, key_mean, block_modalities, block_valid, modality_bias,
    ).contiguous()
    d_mix = torch.empty_like(mix, dtype=torch.float32)
    token_tile = 128
    output_tile = 32
    block_dim = triton.next_power_of_2(head_dim)
    _joint_mhla_backward_mix_kernel[
        (batch_size * kv_heads, block_count, block_count)
    ](
        query, output, grad_output,
        summary, normalizer, mix,
        valid_mask, layout, block_token_mask,
        d_mix,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS=max_tokens,
        GROUP=group,
        TOKEN_TILE=token_tile,
        BLOCK_D=block_dim,
    )
    d_logits = mix * (
        d_mix - (mix * d_mix).sum(dim=-1, keepdim=True)
    )
    query_mean_grad = torch.einsum(
        "bkmn,bknd->bkmd", d_logits, key_mean,
    ) / math.sqrt(head_dim)
    key_mean_grad = torch.einsum(
        "bkmn,bkmd->bknd", d_logits, query_mean,
    ) / math.sqrt(head_dim)
    summary_grad = torch.empty_like(summary)
    normalizer_grad = torch.empty_like(normalizer)
    _joint_mhla_backward_summary_kernel[
        (
            batch_size * kv_heads * block_count,
            triton.cdiv(head_dim, output_tile),
            triton.cdiv(head_dim, output_tile),
        )
    ](
        query, output, grad_output,
        summary_grad, normalizer_grad, normalizer, mix,
        valid_mask, layout, block_token_mask,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS=max_tokens,
        GROUP=group,
        TOKEN_TILE=token_tile,
        BLOCK_D=block_dim,
        OUTPUT_TILE=output_tile,
    )
    query_grad = torch.zeros_like(query)
    token_tiles = triton.cdiv(max_tokens, token_tile)
    feature_tiles = triton.cdiv(head_dim, output_tile)
    _joint_mhla_backward_query_kernel[
        (batch_size * heads, block_count * token_tiles, feature_tiles)
    ](
        query, output, grad_output,
        summary, normalizer, mix, query_mean_grad,
        valid_mask, layout, block_token_mask, query_grad,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS=max_tokens,
        GROUP=group,
        TOKEN_TILE=token_tile,
        BLOCK_D=block_dim,
        OUTPUT_TILE=output_tile,
        TOKEN_TILES=token_tiles,
    )
    key_grad = torch.zeros_like(key)
    value_grad = torch.zeros_like(value)
    _joint_mhla_backward_kv_kernel[
        (batch_size * kv_heads, block_count * token_tiles, feature_tiles)
    ](
        key, value, summary_grad, normalizer_grad, key_mean_grad,
        valid_mask, layout, block_token_mask, key_grad, value_grad,
        batch_size, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS=max_tokens,
        TOKEN_TILE=token_tile,
        BLOCK_D=block_dim,
        OUTPUT_TILE=output_tile,
        TOKEN_TILES=token_tiles,
    )
    modality_grad = torch.zeros_like(modality_bias, dtype=torch.float32)
    for query_modality in range(3):
        for source_modality in range(3):
            selected = (
                (block_modalities[:, None] == query_modality)
                & (block_modalities[None, :] == source_modality)
            )
            modality_grad[query_modality, source_modality] = (
                d_logits * selected[None, None].to(dtype=d_logits.dtype)
            ).sum()
    return (
        query_grad,
        key_grad,
        value_grad,
        modality_grad.to(dtype=modality_bias.dtype),
    )

def _joint_mhla_triton_forward(
    query, key, value, block_indices, block_modalities, valid_mask,
    heads, kv_heads, modality_bias, padded_layout=None, return_aux=False,
):
    """Run the MHLA forward path with Triton kernels.

    The summaries and output are accumulated in FP32.  This is intentional:
    the trainable Q/K/V activations can remain BF16 while the small block
    summaries avoid unnecessary precision loss.
    """
    if not _joint_mhla_triton_available(query, key, value):
        raise RuntimeError(
            "MHLA Triton backend requires CUDA, Triton, and matching "
            "float16/bfloat16/float32 QKV tensors: "
            + _joint_mhla_triton_unavailable_reason(query, key, value)
        )
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    valid_mask = valid_mask.to(device=query.device, dtype=torch.bool).contiguous()
    block_modalities = block_modalities.to(
        device=query.device, dtype=torch.int32,
    ).contiguous()
    modality_bias = modality_bias.contiguous()
    if padded_layout is None:
        layout, block_token_mask = _joint_mhla_prepare_padded_layout(
            block_indices, query.device,
        )
    else:
        layout, block_token_mask = padded_layout
    layout = layout.contiguous()
    block_token_mask = block_token_mask.contiguous()
    block_count, max_tokens = layout.shape
    batch_size, _, token_count, head_dim = query.shape
    if heads % kv_heads:
        raise ValueError("MHLA heads must be divisible by kv_heads")
    group = heads // kv_heads
    block_valid = valid_mask.index_select(1, layout.flatten()).reshape(
        batch_size, block_count, max_tokens,
    ) & block_token_mask[None]
    block_valid = block_valid.any(dim=2).contiguous()
    block_dim = triton.next_power_of_2(head_dim)
    query_mean = torch.empty(
        batch_size, kv_heads, block_count, head_dim,
        device=query.device, dtype=torch.float32,
    )
    key_mean = torch.empty_like(query_mean)
    summary = torch.empty(
        batch_size, kv_heads, block_count, head_dim, head_dim,
        device=query.device, dtype=torch.float32,
    )
    normalizer = torch.empty(
        batch_size, kv_heads, block_count, head_dim,
        device=query.device, dtype=torch.float32,
    )
    output = torch.zeros_like(query)
    token_tile = 128
    output_tile = 32
    _joint_mhla_block_means_kernel[(batch_size, kv_heads, block_count)](
        query, key, valid_mask, layout, block_token_mask,
        query_mean, key_mean,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS=max_tokens,
        GROUP=group,
        TOKEN_TILE=token_tile,
        BLOCK_D=block_dim,
    )
    _joint_mhla_block_statistics_kernel[
        (
            batch_size * kv_heads * block_count,
            triton.cdiv(head_dim, output_tile),
            triton.cdiv(head_dim, output_tile),
        )
    ](
        key, value, valid_mask, layout, block_token_mask,
        summary, normalizer,
        batch_size, kv_heads, token_count, head_dim, block_count,
        MAX_TOKENS=max_tokens,
        TOKEN_TILE=token_tile,
        OUTPUT_TILE=output_tile,
    )
    _joint_mhla_output_kernel[
        (
            batch_size, heads,
            block_count
            * triton.cdiv(max_tokens, token_tile)
            * triton.cdiv(head_dim, output_tile),
        )
    ](
        query, query_mean, key_mean, summary, normalizer,
        valid_mask, layout, block_token_mask, block_valid,
        block_modalities, modality_bias, output,
        batch_size, heads, kv_heads, token_count, head_dim, block_count,
        math.sqrt(head_dim),
        MAX_TOKENS=max_tokens,
        GROUP=group,
        TOKEN_TILE=token_tile,
        BLOCK_D=block_dim,
        OUTPUT_TILE=output_tile,
        BLOCKS=block_count,
    )
    if return_aux:
        return output, (
            query_mean, key_mean, summary, normalizer, block_valid,
        )
    return output


def _joint_mhla_naive(
    query, key, value, block_indices, block_modalities, valid_mask,
    heads, kv_heads, modality_bias, padded_layout=None,
):
    """Readable block-loop MHLA reference path."""
    query_for_mix = query.reshape(
        query.shape[0], kv_heads, heads // kv_heads,
        query.shape[2], query.shape[3],
    ).mean(dim=2)
    query_blocks = []
    key_blocks = []
    for indices in block_indices:
        mask = valid_mask.index_select(1, indices).to(dtype=query.dtype)
        mask = mask[:, None, :, None]
        query_blocks.append(
            (query_for_mix.index_select(2, indices) * mask).sum(dim=2)
            / mask.sum(dim=2).clamp_min(1.0)
        )
        key_blocks.append(
            (key.index_select(2, indices) * mask).sum(dim=2)
            / mask.sum(dim=2).clamp_min(1.0)
        )
    query_blocks = torch.stack(query_blocks, dim=2)
    key_blocks = torch.stack(key_blocks, dim=2)
    mix_logits = torch.einsum(
        "bhmd,bhnd->bhmn", query_blocks, key_blocks,
    ) / math.sqrt(query.shape[-1])
    modality_bias = modality_bias[
        block_modalities[:, None], block_modalities[None, :],
    ]
    mix_logits = mix_logits + modality_bias[None, None]
    block_has_tokens = torch.stack([
        valid_mask.index_select(1, indices).any(dim=1)
        for indices in block_indices
    ], dim=1)
    mix_logits = mix_logits.masked_fill(
        ~block_has_tokens[:, None, None, :],
        torch.finfo(mix_logits.dtype).min,
    )
    mix = torch.softmax(mix_logits, dim=-1)
    summaries, normalizers = _joint_mhla_block_statistics(
        key, value, valid_mask, block_indices,
    )
    mixed_summaries = torch.einsum(
        "bhmn,bhnde->bhmde", mix, summaries,
    )
    mixed_normalizers = torch.einsum(
        "bhmn,bhnd->bhmd", mix, normalizers,
    )
    if heads != kv_heads:
        repeats = heads // kv_heads
        mixed_summaries = mixed_summaries.repeat_interleave(repeats, dim=1)
        mixed_normalizers = mixed_normalizers.repeat_interleave(repeats, dim=1)

    output = torch.zeros_like(query)
    for block, indices in enumerate(block_indices):
        feature_query = F.elu(query.index_select(2, indices)) + 1.0
        block_output = torch.einsum(
            "bhnd,bhde->bhne",
            feature_query, mixed_summaries[:, :, block],
        )
        denominator = torch.einsum(
            "bhnd,bhd->bhn",
            feature_query, mixed_normalizers[:, :, block],
        ).unsqueeze(-1)
        block_output = block_output / denominator.clamp_min(1e-6)
        block_output = block_output * valid_mask.index_select(
            1, indices,
        )[:, None, :, None].to(dtype=block_output.dtype)
        output = output.index_copy(2, indices, block_output)
    return output

def _joint_mhla_vectorized(
    query, key, value, block_indices, block_modalities, valid_mask,
    heads, kv_heads, modality_bias, padded_layout=None,
):
    """Vectorized MHLA path using padded block-major batched contractions."""
    if padded_layout is None:
        gather_index, block_token_mask = _joint_mhla_prepare_padded_layout(
            block_indices, query.device,
        )
    else:
        gather_index, block_token_mask = padded_layout
    block_count, max_tokens = gather_index.shape
    flat_index = gather_index.flatten()
    packed_query = query.index_select(2, flat_index).reshape(
        query.shape[0], heads, block_count, max_tokens, query.shape[-1],
    )
    packed_key = key.index_select(2, flat_index).reshape(
        key.shape[0], kv_heads, block_count, max_tokens, key.shape[-1],
    )
    packed_value = value.index_select(2, flat_index).reshape(
        value.shape[0], kv_heads, block_count, max_tokens, value.shape[-1],
    )
    packed_valid = valid_mask.index_select(1, flat_index).reshape(
        valid_mask.shape[0], block_count, max_tokens,
    ) & block_token_mask[None]
    packed_mask = packed_valid[:, None, :, :, None].to(dtype=query.dtype)

    query_for_mix = packed_query.reshape(
        query.shape[0], kv_heads, heads // kv_heads,
        block_count, max_tokens, query.shape[-1],
    ).mean(dim=2)
    query_for_mix = query_for_mix * packed_mask
    key_for_mix = packed_key * packed_mask
    counts = packed_mask.sum(dim=3).clamp_min(1.0)
    query_blocks = query_for_mix.sum(dim=3) / counts
    key_blocks = key_for_mix.sum(dim=3) / counts
    mix_logits = torch.einsum(
        "bhmd,bhnd->bhmn", query_blocks, key_blocks,
    ) / math.sqrt(query.shape[-1])
    modality_bias = modality_bias[
        block_modalities[:, None], block_modalities[None, :],
    ]
    mix_logits = mix_logits + modality_bias[None, None]
    block_has_tokens = packed_valid.any(dim=2)
    mix_logits = mix_logits.masked_fill(
        ~block_has_tokens[:, None, None, :],
        torch.finfo(mix_logits.dtype).min,
    )
    mix = torch.softmax(mix_logits, dim=-1)

    feature_key = (F.elu(packed_key) + 1.0) * packed_mask
    packed_value = packed_value * packed_mask
    summaries = torch.einsum(
        "bhmld,bhmle->bhmde", feature_key, packed_value,
    )
    normalizers = feature_key.sum(dim=3)
    mixed_summaries = torch.einsum(
        "bhmn,bhnde->bhmde", mix, summaries,
    )
    mixed_normalizers = torch.einsum(
        "bhmn,bhnd->bhmd", mix, normalizers,
    )
    if heads != kv_heads:
        repeats = heads // kv_heads
        mixed_summaries = mixed_summaries.repeat_interleave(repeats, dim=1)
        mixed_normalizers = mixed_normalizers.repeat_interleave(repeats, dim=1)

    feature_query = F.elu(packed_query) + 1.0
    packed_output = torch.einsum(
        "bhmld,bhmde->bhmle", feature_query, mixed_summaries,
    )
    denominator = torch.einsum(
        "bhmld,bhmd->bhml", feature_query, mixed_normalizers,
    ).unsqueeze(-1)
    packed_output = packed_output / denominator.clamp_min(1e-6)
    packed_output = packed_output * packed_mask
    packed_output = packed_output.reshape(
        query.shape[0], heads, block_count * max_tokens, query.shape[-1],
    )
    scatter_index = gather_index.flatten()[None, None, :, None].expand(
        query.shape[0], heads, block_count * max_tokens, query.shape[-1],
    )
    return torch.zeros_like(query).scatter_add(
        2, scatter_index, packed_output,
    )


class _JointMHLATritonFunction(torch.autograd.Function):
    """Triton forward/backward with small block-softmax PyTorch support."""

    @staticmethod
    def forward(
        ctx, query, key, value, block_modalities, valid_mask, modality_bias,
        block_indices, padded_layout, heads, kv_heads, recompute_output,
        timing,
    ):
        output, aux = _joint_mhla_triton_forward(
            query, key, value, block_indices, block_modalities, valid_mask,
            heads, kv_heads, modality_bias, padded_layout, return_aux=True,
        )
        ctx.recompute_output = bool(recompute_output)
        saved = (
            query, key, value, block_modalities, valid_mask, modality_bias,
        )
        if not ctx.recompute_output:
            saved = saved + (output,)
        ctx.save_for_backward(*(saved + aux))
        ctx.block_indices = tuple(block_indices)
        ctx.padded_layout = padded_layout
        ctx.heads = heads
        ctx.kv_heads = kv_heads
        ctx.timing = timing
        return output

    @staticmethod
    def backward(ctx, grad_output):
        measure = (
            ctx.timing.measure
            if ctx.timing is not None
            else lambda _: nullcontext()
        )
        with measure("mhla_backward"):
            saved = ctx.saved_tensors
            query, key, value, block_modalities, valid_mask, modality_bias = saved[:6]
            offset = 6
            if ctx.recompute_output:
                output = _joint_mhla_triton_forward(
                    query, key, value, ctx.block_indices, block_modalities,
                    valid_mask, ctx.heads, ctx.kv_heads, modality_bias,
                    ctx.padded_layout, return_aux=False,
                )
            else:
                output = saved[offset]
                offset += 1
            query_mean, key_mean, summary, normalizer, block_valid = saved[offset:]
            gradients = _joint_mhla_triton_backward(
                query, key, value, block_modalities, valid_mask, modality_bias,
                grad_output, output, query_mean, key_mean, summary, normalizer,
                block_valid, ctx.padded_layout, ctx.heads, ctx.kv_heads,
            )
        return (
            gradients[0], gradients[1], gradients[2], None, None,
            gradients[3], None, None, None, None, None, None,
        )

class _JointMHLATritonVectorizedBackwardFunction(torch.autograd.Function):
    """Triton forward with the vectorized PyTorch backward fallback.

    The custom Triton backward is useful once its kernels are compiled, but
    its first compilation can be disproportionately expensive for the large
    block layout used by the full model.  This fallback keeps the low-memory
    Triton forward while delegating backward graph construction to the tested
    vectorized implementation.
    """

    @staticmethod
    def forward(
        ctx, query, key, value, block_modalities, valid_mask, modality_bias,
        block_indices, padded_layout, heads, kv_heads, recompute_output,
        timing,
    ):
        output = _joint_mhla_triton_forward(
            query, key, value, block_indices, block_modalities, valid_mask,
            heads, kv_heads, modality_bias, padded_layout, return_aux=False,
        )
        ctx.save_for_backward(
            query, key, value, block_modalities, valid_mask, modality_bias,
        )
        ctx.block_indices = tuple(block_indices)
        ctx.padded_layout = padded_layout
        ctx.heads = heads
        ctx.kv_heads = kv_heads
        ctx.timing = timing
        return output

    @staticmethod
    def backward(ctx, grad_output):
        measure = (
            ctx.timing.measure
            if ctx.timing is not None
            else lambda _: nullcontext()
        )
        with measure("mhla_backward"):
            query, key, value, block_modalities, valid_mask, modality_bias = (
                ctx.saved_tensors
            )
            with torch.enable_grad():
                query = query.detach().requires_grad_(True)
                key = key.detach().requires_grad_(True)
                value = value.detach().requires_grad_(True)
                modality_bias = modality_bias.detach().requires_grad_(True)
                output = _joint_mhla_vectorized(
                    query, key, value, ctx.block_indices, block_modalities,
                    valid_mask, ctx.heads, ctx.kv_heads, modality_bias,
                    ctx.padded_layout,
                )
                gradients = torch.autograd.grad(
                    output,
                    (query, key, value, modality_bias),
                    grad_output,
                    allow_unused=True,
                )
        return (
            gradients[0], gradients[1], gradients[2], None, None,
            gradients[3], None, None, None, None, None, None,
        )


def joint_mhla_attention(
    query, key, value, block_indices, block_modalities, valid_mask,
    heads, kv_heads, modality_bias, backend="auto", padded_layout=None,
    recompute_output=False, timing=None,
):
    """Functional Joint-MHLA API with naive/vectorized/Triton backends."""
    if backend not in {"auto", "naive", "vectorized", "triton"}:
        raise ValueError(f"unknown Joint MHLA backend: {backend}")
    if backend == "auto":
        backend = "triton" if _joint_mhla_triton_available(query, key, value) else "vectorized"
    if backend == "triton":
        if not _joint_mhla_triton_available(query, key, value):
            raise RuntimeError(
                "MHLA backend='triton' requires CUDA, Triton, and matching "
                "float16/bfloat16/float32 QKV tensors: "
                + _joint_mhla_triton_unavailable_reason(query, key, value)
            )
        function = _JointMHLATritonFunction
        if os.environ.get("MHLA_TRITON_BACKWARD", "0") == "0":
            function = _JointMHLATritonVectorizedBackwardFunction
        return function.apply(
            query, key, value, block_modalities, valid_mask, modality_bias,
            block_indices, padded_layout, heads, kv_heads, recompute_output,
            timing,
        )
    implementation = (
        _joint_mhla_naive if backend == "naive" else _joint_mhla_vectorized
    )
    return implementation(
        query, key, value, block_indices, block_modalities, valid_mask,
        heads, kv_heads, modality_bias, padded_layout,
    )
