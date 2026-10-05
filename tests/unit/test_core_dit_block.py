from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from core.layers import DiTTransformerBlock, SwiGLUFeedForward


def _reference_block(
    block: DiTTransformerBlock,
    tokens: torch.Tensor,
    z: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch, token_count, _ = tokens.shape
    normalized = block.norm1(tokens)
    query = block.q_proj(normalized).reshape(
        batch, token_count, block.heads, block.head_dim,
    ).transpose(1, 2)
    key = block.k_proj(normalized).reshape(
        batch, token_count, block.kv_heads, block.head_dim,
    ).transpose(1, 2)
    value = block.v_proj(normalized).reshape(
        batch, token_count, block.kv_heads, block.head_dim,
    ).transpose(1, 2)
    query = block.q_norm(query)
    key = block.k_norm(key)
    key = key.repeat_interleave(block.heads // block.kv_heads, dim=1)
    value = value.repeat_interleave(block.heads // block.kv_heads, dim=1)

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
    scores = scores / block.head_dim**0.5
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask
    probabilities = scores.softmax(dim=-1).to(value.dtype)
    attended = torch.matmul(probabilities, value)
    if block.head_gate is not None:
        assert z is not None
        gate = 2 * torch.sigmoid(block.head_gate(z))
        attended = attended * gate[:, :, None, None]
    attended = attended.transpose(1, 2).contiguous().reshape(
        batch, token_count, block.width,
    )
    hidden = tokens + block.attn_out(attended)
    normalized = block.norm2(hidden)
    mlp = block.mlp
    ffn = mlp.out_proj(F.silu(mlp.gate_proj(normalized)) * mlp.value_proj(normalized))
    return hidden + ffn


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("qk_norm", [True, False])
def test_dit_block_matches_explicit_reference_and_gradients(kv_heads: int, qk_norm: bool) -> None:
    torch.manual_seed(13)
    block = DiTTransformerBlock(
        width=16,
        heads=4,
        kv_heads=kv_heads,
        ff_mult=2.0,
        condition_dim=7,
        qk_norm=qk_norm,
        bias=True,
    ).eval()
    reference_block = deepcopy(block)
    tokens = torch.randn(2, 5, 16, requires_grad=True)
    z = torch.randn(2, 7, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    reference_z = z.detach().clone().requires_grad_()
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
    mask[0, :, :, -1] = False

    actual = block(tokens, z, attention_mask=mask)
    expected = _reference_block(reference_block, reference_tokens, reference_z, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(z.grad, reference_z.grad, rtol=2e-4, atol=2e-6)
    for (name, parameter), (ref_name, ref_parameter) in zip(
        block.named_parameters(), reference_block.named_parameters(), strict=True,
    ):
        assert name == ref_name
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, rtol=2e-4, atol=2e-6,
            msg=f"gradient mismatch: {name}",
        )


def test_dit_head_gate_zero_initializes_to_identity_and_depends_on_z() -> None:
    torch.manual_seed(8)
    block = DiTTransformerBlock(16, 4, kv_heads=2, condition_dim=5).eval()
    tokens = torch.randn(2, 3, 16)
    z = torch.randn(2, 5)

    assert torch.count_nonzero(block.head_gate.weight) == 0
    assert torch.count_nonzero(block.head_gate.bias) == 0
    initial = block(tokens, z)
    with torch.no_grad():
        block.head_gate.weight[0, 0] = 1.0
    changed = block(tokens, z)

    assert not torch.allclose(initial, changed)


def test_dit_block_supports_position_transform_and_rectangular_mask() -> None:
    block = DiTTransformerBlock(12, 3, kv_heads=1, condition_dim=4).eval()
    tokens = torch.randn(2, 6, 12)
    z = torch.randn(2, 4)
    mask = torch.ones(2, 6, 6, dtype=torch.bool)
    mask[:, :, 0] = False

    def transform(q: torch.Tensor, k: torch.Tensor):
        return q + 0.05, k - 0.05

    result = block(tokens, z, attention_mask=mask, qk_transform=transform)
    assert result.shape == tokens.shape
    assert torch.isfinite(result).all()


def test_dit_block_all_masked_attention_rows_are_finite_and_zero() -> None:
    torch.manual_seed(23)
    block = DiTTransformerBlock(12, 3, kv_heads=1).eval()
    tokens = torch.randn(2, 4, 12)
    mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)

    actual = block(tokens, attention_mask=mask)
    expected = tokens + block.mlp(block.norm2(tokens))

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dit_block_boolean_and_additive_masks_match() -> None:
    torch.manual_seed(29)
    block = DiTTransformerBlock(12, 3, kv_heads=1).eval()
    tokens = torch.randn(2, 4, 12)
    boolean_mask = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    boolean_mask[:, :, :, -1] = False
    additive_mask = torch.zeros(boolean_mask.shape)
    additive_mask.masked_fill_(~boolean_mask, float("-inf"))

    boolean_result = block(tokens, attention_mask=boolean_mask)
    additive_result = block(tokens, attention_mask=additive_mask)

    torch.testing.assert_close(boolean_result, additive_result, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("tokens", "z", "mask"),
    [
        (torch.randn(2, 4, 12), torch.randn(2, 5), None),
        (torch.randn(2, 4, 12), None, None),
        (torch.randn(2, 4, 12), torch.randn(2, 5), torch.ones(2, 3, dtype=torch.bool)),
    ],
)
def test_dit_block_rejects_bad_condition_or_mask(
    tokens: torch.Tensor, z: torch.Tensor | None, mask: torch.Tensor | None,
) -> None:
    block = DiTTransformerBlock(12, 3, kv_heads=1, condition_dim=4)
    with pytest.raises(ValueError):
        block(tokens, z, attention_mask=mask)


def test_swiglu_separate_projections_match_concatenated_gemm_reference() -> None:
    torch.manual_seed(2)
    layer = SwiGLUFeedForward(6, 9, bias=True)
    reference = deepcopy(layer)
    x = torch.randn(2, 4, 6, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()

    actual = layer(x)
    gate_value_weight = torch.cat(
        (reference.gate_proj.weight, reference.value_proj.weight), dim=0,
    )
    gate_value_bias = torch.cat(
        (reference.gate_proj.bias, reference.value_proj.bias), dim=0,
    )
    combined = F.linear(reference_x, gate_value_weight, gate_value_bias)
    gate, value = combined.chunk(2, dim=-1)
    expected = reference.out_proj(F.silu(gate) * value)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        layer.gate_proj.weight.grad,
        reference.gate_proj.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        layer.value_proj.weight.grad,
        reference.value_proj.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_dit_qkv_separate_projections_match_combined_gemm_reference() -> None:
    torch.manual_seed(5)
    block = DiTTransformerBlock(12, 3, kv_heads=1, condition_dim=None, bias=True)
    x = torch.randn(2, 4, 12, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = [
        block.q_proj.weight.detach().clone().requires_grad_(),
        block.k_proj.weight.detach().clone().requires_grad_(),
        block.v_proj.weight.detach().clone().requires_grad_(),
    ]
    biases = [
        block.q_proj.bias.detach().clone().requires_grad_(),
        block.k_proj.bias.detach().clone().requires_grad_(),
        block.v_proj.bias.detach().clone().requires_grad_(),
    ]

    naive = torch.cat(
        (block.q_proj(x), block.k_proj(x), block.v_proj(x)), dim=-1,
    )
    fused_weight = torch.cat(weights, dim=0)
    fused_bias = torch.cat(biases, dim=0)
    fused = F.linear(reference_x, fused_weight, fused_bias)
    torch.testing.assert_close(naive, fused, rtol=1e-6, atol=1e-7)

    naive.square().sum().backward()
    fused.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-5, atol=1e-6)
    for module, weight, bias in zip(
        (block.q_proj, block.k_proj, block.v_proj), weights, biases, strict=True,
    ):
        torch.testing.assert_close(module.weight.grad, weight.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(module.bias.grad, bias.grad, rtol=1e-5, atol=1e-6)
