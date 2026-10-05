from __future__ import annotations

import pytest
import torch
from jaxtyping import Float

from core.layers import (
    GatedConv2d,
    GatedConvTranspose2d,
    GatedMultiheadAttention,
    GroupedQueryAttention,
    KVSelfAttention,
    RotaryEmbedding2D,
    RMSNorm2d,
    _default_group_norm,
    build_2d_sincos_pos_embed,
)
from image_gen.train import TimestepEmbedding


def _assert_nchw_tensor(
    tensor: Float[torch.Tensor, "..."],
) -> None:
    """Keep Tensor shape intent visible to static type checkers."""
    assert tensor.ndim == 4


def test_build_2d_sincos_pos_embed_shape():
    embedding = build_2d_sincos_pos_embed(16, height=3, width=5)
    assert embedding.shape == (1, 15, 16)
    assert embedding.dtype == torch.float32


def test_build_2d_sincos_pos_embed_rejects_invalid_dimension():
    with pytest.raises(ValueError, match="divisible by 4"):
        build_2d_sincos_pos_embed(10, height=2, width=2)


def test_default_group_norm_chooses_a_divisor_for_nonstandard_channels():
    norm = _default_group_norm(48)

    assert norm.num_channels == 48
    assert norm.num_groups == 24
    with pytest.raises(ValueError, match="channels must be positive"):
        _default_group_norm(0)


def test_rotary_embedding_2d_rotates_adjacent_pairs_and_backpropagates():
    rope = RotaryEmbedding2D(head_dim=8, height=2, width=3)
    query = torch.randn(2, 3, 6, 8, requires_grad=True)
    key = torch.randn(2, 3, 6, 8, requires_grad=True)

    actual_query, actual_key = rope(query, key)
    cos = rope.cos.to(dtype=query.dtype)
    sin = rope.sin.to(dtype=query.dtype)
    query_even, query_odd = query[..., 0::2], query[..., 1::2]
    expected_query = torch.stack(
        (query_even * cos[..., 0::2] - query_odd * sin[..., 0::2],
         query_even * sin[..., 0::2] + query_odd * cos[..., 0::2]),
        dim=-1,
    ).flatten(-2)
    assert torch.allclose(actual_query, expected_query)
    assert torch.allclose(actual_key, torch.stack(
        (key[..., 0::2] * cos[..., 0::2] - key[..., 1::2] * sin[..., 0::2],
         key[..., 0::2] * sin[..., 0::2] + key[..., 1::2] * cos[..., 0::2]),
        dim=-1,
    ).flatten(-2))

    (actual_query.square().mean() + actual_key.square().mean()).backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()


def test_rotary_embedding_2d_rebuilds_exact_tables_for_rectangular_grid():
    rope = RotaryEmbedding2D(head_dim=8, height=2, width=3, base=100.0)
    query = torch.randn(2, 3, 15, 8, requires_grad=True)
    key = torch.randn(2, 3, 15, 8, requires_grad=True)

    actual_query, actual_key = rope(query, key, grid_shape=(3, 5))
    frequency = 1.0 / (
        100.0 ** (torch.arange(2, dtype=torch.float32) / 2)
    )
    y, x = torch.meshgrid(
        torch.arange(3, dtype=torch.float32),
        torch.arange(5, dtype=torch.float32),
        indexing="ij",
    )
    phase = torch.cat(
        (
            (y.flatten()[:, None] * frequency).repeat_interleave(2, dim=-1),
            (x.flatten()[:, None] * frequency).repeat_interleave(2, dim=-1),
        ),
        dim=-1,
    )
    cos, sin = phase.cos()[None, None], phase.sin()[None, None]
    expected_query = torch.stack(
        (
            query[..., 0::2] * cos[..., 0::2]
            - query[..., 1::2] * sin[..., 0::2],
            query[..., 0::2] * sin[..., 0::2]
            + query[..., 1::2] * cos[..., 0::2],
        ),
        dim=-1,
    ).flatten(-2)
    expected_key = torch.stack(
        (
            key[..., 0::2] * cos[..., 0::2]
            - key[..., 1::2] * sin[..., 0::2],
            key[..., 0::2] * sin[..., 0::2]
            + key[..., 1::2] * cos[..., 0::2],
        ),
        dim=-1,
    ).flatten(-2)
    assert torch.allclose(actual_query, expected_query)
    assert torch.allclose(actual_key, expected_key)

    (actual_query.square().mean() + actual_key.square().mean()).backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()


def test_rotary_embedding_2d_rejects_invalid_grid_configuration():
    with pytest.raises(ValueError, match="divisible by 4"):
        RotaryEmbedding2D(head_dim=6, height=2, width=2)
    with pytest.raises(ValueError, match="positive"):
        RotaryEmbedding2D(head_dim=8, height=0, width=2)

    rope = RotaryEmbedding2D(head_dim=8, height=2, width=2)
    query = torch.randn(1, 1, 5, 8)
    with pytest.raises(ValueError, match="token count"):
        rope(query, query, grid_shape=(2, 2))
    with pytest.raises(ValueError, match="required"):
        rope(query, query)
    with pytest.raises(ValueError, match="head_dim"):
        rope(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
    with pytest.raises(ValueError, match="matching batch"):
        rope(torch.randn(1, 1, 4, 8), torch.randn(2, 1, 4, 8))


@pytest.mark.parametrize(
    ("layer", "expected_height", "expected_width"),
    [
        (GatedConv2d(3, 5, kernel_size=3, stride=2, padding=1), 4, 4),
        (GatedConvTranspose2d(3, 5), 16, 16),
    ],
)
def test_gated_convolution_shapes(layer, expected_height, expected_width):
    output = layer(torch.randn(2, 3, 8, 8))
    _assert_nchw_tensor(output)
    assert output.shape == (2, 5, expected_height, expected_width)
    assert torch.isfinite(output).all()


def test_rmsnorm2d_preserves_nchw_shape():
    layer = RMSNorm2d(4)
    output = layer(torch.randn(2, 4, 5, 7))
    _assert_nchw_tensor(output)
    assert output.shape == (2, 4, 5, 7)
    assert torch.isfinite(output).all()


def test_rmsnorm2d_has_no_unused_layernorm_parameters():
    layer = RMSNorm2d(4)

    assert list(layer.state_dict()) == ["rmsnorm.weight"]
    assert [name for name, _ in layer.named_parameters()] == ["rmsnorm.weight"]


@pytest.mark.parametrize("gate_mode", ["none", "static", "query"])
def test_grouped_query_attention_gate_modes(gate_mode):
    layer = GroupedQueryAttention(
        embed_dim=32,
        num_heads=8,
        kv_heads=2,
        gate_mode=gate_mode,
        dropout=0.0,
    )
    query = torch.randn(2, 5, 32, requires_grad=True)
    source = torch.randn(2, 7, 32)
    output = layer(query, source, source)
    assert output.shape == query.shape
    output.square().mean().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_grouped_query_attention_all_key_mask_returns_finite_zero_weights():
    layer = GroupedQueryAttention(
        embed_dim=16, num_heads=4, kv_heads=2,
        gate_mode="none", dropout=0.0,
    )
    query = torch.randn(1, 3, 16, requires_grad=True)
    source = torch.randn(1, 5, 16)

    output, weights = layer(
        query, source, source,
        key_padding_mask=torch.ones(1, 5, dtype=torch.bool),
        need_weights=True,
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert torch.equal(weights, torch.zeros_like(weights))
    output.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_gated_multihead_attention_compatibility_wrapper_keeps_return_contract():
    layer = GatedMultiheadAttention(
        embed_dim=32,
        num_heads=8,
        batch_first=True,
        dropout=0.0,
    )
    query = torch.randn(2, 5, 32)
    output, weights = layer(query, query, query)
    assert output.shape == query.shape
    assert weights.shape == (2, 5, 5)
    output_without_weights, no_weights = layer(
        query, query, query, need_weights=False,
    )
    assert output_without_weights.shape == query.shape
    assert no_weights is None
    assert torch.allclose(
        torch.sigmoid(layer.gate),
        torch.full_like(layer.gate, 0.5),
    )


def test_kv_self_attention_scales_full_dot_product_once():
    layer = KVSelfAttention(
        embed_dim=8, num_heads=2, batch_first=True,
    )
    layer.k_proj = torch.nn.Identity()
    layer.v_proj = torch.nn.Identity()
    layer.out_proj = torch.nn.Identity()
    inputs = torch.randn(1, 3, 8)

    _, actual_weights = layer(inputs, need_weights=True)
    key = inputs.reshape(1, 3, 2, 4)
    expected_scores = torch.einsum("blhd,bshd->bhls", key, key) / 2.0
    expected_weights = torch.softmax(expected_scores, dim=-1).mean(dim=1)

    assert torch.allclose(actual_weights, expected_weights)


def test_rmsnorm2d_loads_legacy_layernorm_keys():
    source = RMSNorm2d(4)
    legacy_state = source.state_dict()
    legacy_state["weight"] = torch.ones(4)
    legacy_state["bias"] = torch.zeros(4)

    restored = RMSNorm2d(4)
    restored.load_state_dict(legacy_state, strict=True)

    assert torch.equal(restored.rmsnorm.weight, source.rmsnorm.weight)


def test_timestep_embedding_caches_nonpersistent_frequencies():
    embedding = TimestepEmbedding(dim=8, frequency_dim=8)
    timesteps = torch.tensor([0.0, 0.25, 1.0])

    first = embedding(timesteps)
    second = embedding(timesteps)

    assert first.shape == (3, 8)
    assert torch.allclose(first, second)
    assert embedding.frequencies.shape == (4,)
    assert "frequencies" not in embedding.state_dict()


def test_timestep_embedding_rejects_odd_frequency_dimension():
    with pytest.raises(ValueError, match="frequency_dim must be even"):
        TimestepEmbedding(dim=8, frequency_dim=7)


def test_rotary_embedding_2d_supports_gqa_head_counts():
    rope = RotaryEmbedding2D(head_dim=8, height=2, width=3)
    query = torch.randn(2, 8, 6, 8, requires_grad=True)
    key = torch.randn(2, 2, 6, 8, requires_grad=True)

    rotated_query, rotated_key = rope(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    (rotated_query.square().mean() + rotated_key.square().mean()).backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()
