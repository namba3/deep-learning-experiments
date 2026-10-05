from typing import Any

import pytest
import torch
from vfp_dit.model import ConditionKVCache, NoVFCBDiT, _TargetBlock, apply_multimodal_rope, build_qwen_condition_positions, grid_positions


def _copy_target_block_to_fused(reference: _TargetBlock, fused: _TargetBlock) -> None:
    skipped = {
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "condition_k_proj.weight", "condition_v_proj.weight",
    }
    if reference.attention_head_gate == "input_silu":
        skipped.update(("attn_gate.weight", "attn_gate.bias"))
    if reference.metadata_conditioning == "ada_attn_ffn":
        skipped.update((
            "attn_meta_scale.weight", "attn_meta_scale.bias",
            "ffn_meta_scale.weight", "ffn_meta_scale.bias",
            "attn_meta_shift.weight", "attn_meta_shift.bias",
            "ffn_meta_shift.weight", "ffn_meta_shift.bias",
            "ffn_residual_gate.weight",
        ))
    with torch.no_grad():
        _copy_matching_parameters(reference, fused, excluded=skipped)
        qkv_parts = [
            reference.q_proj.weight,
            reference.k_proj.weight,
            reference.v_proj.weight,
        ]
        if reference.attention_head_gate == "input_silu":
            qkv_parts.append(reference.attn_gate.weight)
            fused.attn_head_gate_bias.copy_(reference.attn_gate.bias)
        fused.target_qkv_proj.weight.copy_(torch.cat(qkv_parts, dim=0))
        fused.condition_kv_proj.weight.copy_(torch.cat((
            reference.condition_k_proj.weight,
            reference.condition_v_proj.weight,
        ), dim=0))
        if reference.metadata_conditioning == "ada_attn_ffn":
            source_by_name = {
                "attn_scale": reference.attn_meta_scale,
                "ffn_scale": reference.ffn_meta_scale,
            }
            if reference.metadata_shift:
                source_by_name.update({
                    "attn_shift": reference.attn_meta_shift,
                    "ffn_shift": reference.ffn_meta_shift,
                })
            if reference.metadata_ffn_residual_gate:
                source_by_name["ffn_gate"] = reference.ffn_residual_gate
            fused.metadata_projection.weight.copy_(torch.cat([
                source_by_name[name].weight for name in fused.metadata_projection_names
            ], dim=0))
            fused.metadata_projection_bias.copy_(torch.cat([
                source_by_name[name].bias
                for name in fused.metadata_projection_bias_names
            ]))


def _copy_matching_parameters(reference, fused, *, excluded: set[str]) -> None:
    reference_state = reference.state_dict()
    fused_state = fused.state_dict()
    for name, value in reference_state.items():
        if name in excluded or name not in fused_state:
            continue
        if value.shape == fused_state[name].shape:
            fused_state[name].copy_(value)


def _tiny_model() -> NoVFCBDiT:
    return NoVFCBDiT(
        qwen_dim=20,
        latent_channels=4,
        width=24,
        depth=2,
        heads=2,
        kv_heads=1,
        adapter_depth=1,
        ff_mult=1.5,
    )


@pytest.mark.parametrize(
    ("metadata_conditioning", "metadata_shift", "attention_head_gate"),
    [
        ("none", False, "input_silu"),
        ("none", False, "timestep_sigmoid"),
        ("ada_attn_ffn", False, "input_silu"),
        ("ada_attn_ffn", True, "input_silu"),
    ],
)
def test_fused_target_projections_match_separate_forward_cache_and_backward(
    metadata_conditioning: str,
    metadata_shift: bool,
    attention_head_gate: str,
):
    torch.manual_seed(103)
    common: dict[str, Any] = dict(
        width=24,
        heads=2,
        kv_heads=1,
        ff_mult=1.5,
        metadata_conditioning=metadata_conditioning,
        metadata_scale_mapping="softplus1_normalized",
        metadata_shift=metadata_shift,
        attention_head_gate=attention_head_gate,
        metadata_ffn_residual_gate=metadata_conditioning == "ada_attn_ffn",
    )
    reference = _TargetBlock(**common, fuse_same_input_projections=False)
    fused = _TargetBlock(**common, fuse_same_input_projections=True)
    _copy_target_block_to_fused(reference, fused)
    batch, condition_count, target_count = 2, 5, 6
    condition = torch.randn(batch, condition_count, 24)
    condition_mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    condition_positions = torch.zeros(batch, condition_count, 3)
    condition_positions[:, :, 0] = torch.arange(condition_count)
    ref_condition_input = condition.clone().requires_grad_()
    fused_condition_input = condition.clone().requires_grad_()
    reference_cache = reference.prepare_condition(
        ref_condition_input, condition_mask, condition_positions,
    )
    fused_cache = fused.prepare_condition(
        fused_condition_input, condition_mask, condition_positions,
    )
    torch.testing.assert_close(fused_cache[0], reference_cache[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fused_cache[1], reference_cache[1], rtol=1e-5, atol=1e-6)

    target = torch.randn(batch, target_count, 24)
    target_positions = grid_positions(batch, 2, 3, device=torch.device("cpu"))
    timestep = torch.rand(batch, 24)
    metadata = (
        torch.randn(batch, 24, requires_grad=True)
        if metadata_conditioning != "none" else None
    )
    ref_target = target.clone().requires_grad_()
    fused_target = target.clone().requires_grad_()
    ref_timestep = timestep.clone().requires_grad_()
    fused_timestep = timestep.clone().requires_grad_()
    ref_metadata = metadata.detach().clone().requires_grad_() if metadata is not None else None
    fused_metadata = metadata.detach().clone().requires_grad_() if metadata is not None else None
    reference_output = reference(
        ref_target, ref_timestep, target_positions, *reference_cache,
        condition_mask, ref_metadata,
    )
    fused_output = fused(
        fused_target, fused_timestep, target_positions, *fused_cache,
        condition_mask, fused_metadata,
    )
    torch.testing.assert_close(fused_output, reference_output, rtol=1e-5, atol=1e-6)
    reference_output.square().mean().backward()
    fused_output.square().mean().backward()
    for fused_grad, reference_grad in (
        (fused_target.grad, ref_target.grad),
        (fused_timestep.grad, ref_timestep.grad),
        (fused_condition_input.grad, ref_condition_input.grad),
    ):
        torch.testing.assert_close(fused_grad, reference_grad, rtol=2e-5, atol=2e-6)
    if ref_metadata is not None:
        torch.testing.assert_close(fused_metadata.grad, ref_metadata.grad, rtol=2e-5, atol=2e-6)

    kv_width = reference.kv_heads * reference.head_dim
    fused_slices = [
        ("q_proj", 0, reference.width),
        ("k_proj", reference.width, reference.width + kv_width),
        ("v_proj", reference.width + kv_width, reference.width + 2 * kv_width),
    ]
    for name, start, end in fused_slices:
        torch.testing.assert_close(
            fused.target_qkv_proj.weight.grad[start:end],
            getattr(reference, name).weight.grad,
            rtol=2e-5, atol=2e-6,
        )
    torch.testing.assert_close(
        fused.condition_kv_proj.weight.grad[:kv_width],
        reference.condition_k_proj.weight.grad,
        rtol=2e-5, atol=2e-6,
    )
    torch.testing.assert_close(
        fused.condition_kv_proj.weight.grad[kv_width:],
        reference.condition_v_proj.weight.grad,
        rtol=2e-5, atol=2e-6,
    )
    if attention_head_gate == "input_silu":
        gate_start = reference.width + 2 * kv_width
        torch.testing.assert_close(
            fused.target_qkv_proj.weight.grad[gate_start:],
            reference.attn_gate.weight.grad,
            rtol=2e-5, atol=2e-6,
        )
        torch.testing.assert_close(
            fused.attn_head_gate_bias.grad,
            reference.attn_gate.bias.grad,
            rtol=2e-5, atol=2e-6,
        )
    if metadata_conditioning == "ada_attn_ffn":
        reference_metadata_layers = {
            "attn_scale": reference.attn_meta_scale,
            "ffn_scale": reference.ffn_meta_scale,
            "ffn_gate": reference.ffn_residual_gate,
        }
        if metadata_shift:
            reference_metadata_layers.update({
                "attn_shift": reference.attn_meta_shift,
                "ffn_shift": reference.ffn_meta_shift,
            })
        for index, name in enumerate(fused.metadata_projection_names):
            start, end = index * reference.width, (index + 1) * reference.width
            reference_layer = reference_metadata_layers[name]
            torch.testing.assert_close(
                fused.metadata_projection.weight.grad[start:end],
                reference_layer.weight.grad,
                rtol=2e-5, atol=2e-6,
            )
        for index, name in enumerate(fused.metadata_projection_bias_names):
            start, end = index * reference.width, (index + 1) * reference.width
            torch.testing.assert_close(
                fused.metadata_projection_bias.grad[start:end],
                reference_metadata_layers[name].bias.grad,
                rtol=2e-5, atol=2e-6,
            )


def test_multimodal_rope_preserves_shape_and_backpropagates():
    value = torch.randn(2, 2, 6, 12, requires_grad=True)
    positions = grid_positions(2, 2, 3, device=value.device, reference_id=4)

    output = apply_multimodal_rope(value, positions)

    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_multimodal_rope_rejects_invalid_contracts():
    value = torch.randn(1, 2, 3, 12)
    with pytest.raises(ValueError, match="positions must have shape"):
        apply_multimodal_rope(value, torch.zeros(1, 3, 2))
    with pytest.raises(ValueError, match="even and at least 6"):
        apply_multimodal_rope(torch.randn(1, 2, 3, 4), torch.zeros(1, 3, 3))


def test_grid_positions_are_row_major_and_keep_reference_index():
    positions = grid_positions(1, 2, 3, device=torch.device("cpu"), reference_id=7)

    assert positions.tolist() == [[
        [7, 0, 0], [7, 0, 1], [7, 0, 2],
        [7, 1, 0], [7, 1, 1], [7, 1, 2],
    ]]


def test_condition_cache_validates_layer_shapes_and_mask():
    key = torch.randn(2, 1, 3, 12)
    value = torch.randn_like(key)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    cache = ConditionKVCache((key,), (value,), mask)

    assert cache.keys[0].shape == (2, 1, 3, 12)
    with pytest.raises(ValueError, match="valid token"):
        ConditionKVCache((key,), (value,), torch.zeros_like(mask))


def test_no_vfcb_dit_prepares_bidirectional_condition_cache_and_trains():
    torch.manual_seed(17)
    model = _tiny_model()
    qwen_hidden = torch.randn(2, 5, 20)
    qwen_mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, False, False],
    ])
    qwen_positions = torch.zeros(2, 5, 3)
    qwen_positions[:, :, 0] = torch.arange(5)
    # The final two Qwen visual tokens identify ref 1. Reference latent tokens
    # generated by grid_positions() use that same ref index.
    qwen_positions[:, 3:, 0] = 1
    qwen_positions[:, 3:, 1:] = torch.tensor([[0, 0], [0, 1]])
    reference_latent = torch.randn(2, 4, 2, 2)
    reference_positions = grid_positions(
        2, 2, 2, device=torch.device("cpu"), reference_id=1,
    )

    cache = model.prepare_condition(
        qwen_hidden, qwen_mask, qwen_positions,
        reference_latent, reference_positions=reference_positions,
    )
    noisy_latent = torch.randn(2, 4, 2, 3, requires_grad=True)
    timestep = torch.tensor([0.2, 0.8])
    velocity = model(noisy_latent, timestep, cache)

    assert len(cache.keys) == 2
    assert cache.mask.shape == (2, 9)
    assert velocity.shape == noisy_latent.shape
    assert torch.isfinite(velocity).all()
    velocity.square().mean().backward()
    assert noisy_latent.grad is not None and torch.isfinite(noisy_latent.grad).all()
    assert model.adapter.qwen_in.weight.grad is not None
    assert model.blocks[0].condition_kv_proj.weight.grad is not None
    assert model.target_in.weight.grad is not None


def test_condition_cache_can_be_reused_across_target_steps():
    torch.manual_seed(23)
    model = _tiny_model().eval()
    hidden = torch.randn(1, 4, 20)
    mask = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.zeros(1, 4, 3)
    positions[:, :, 0] = torch.arange(4)
    cache = model.prepare_condition(hidden, mask, positions)
    original_keys = tuple(key.clone() for key in cache.keys)
    first = model(torch.randn(1, 4, 2, 3), torch.tensor([0.1]), cache)
    second = model(torch.randn(1, 4, 2, 3), torch.tensor([0.9]), cache)

    assert first.shape == second.shape == (1, 4, 2, 3)
    assert all(torch.equal(before, after) for before, after in zip(
        original_keys, cache.keys, strict=True,
    ))


def test_default_vlm_adapter_uses_gated_ffn_without_attention_gradients():
    model = _tiny_model()
    assert model.adapter.adapter_type == "ffn"
    assert not hasattr(model.adapter.blocks[0], "qkv_proj")
    assert not hasattr(model.adapter.blocks[0], "norm1")

    hidden = torch.randn(1, 4, 20)
    mask = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.zeros(1, 4, 3)
    positions[:, :, 0] = torch.arange(4)
    output, _, _ = model.adapter(hidden, mask, positions)
    output.square().mean().backward()

    assert model.adapter.blocks[0].ffn_in.weight.grad is not None


def test_no_vfcb_dit_rejects_cache_depth_mismatch():
    model = _tiny_model()
    cache = ConditionKVCache(
        (torch.randn(1, 1, 3, 12),),
        (torch.randn(1, 1, 3, 12),),
        torch.ones(1, 3, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="depth"):
        model(torch.randn(1, 4, 2, 2), torch.tensor([0.5]), cache)


def test_qwen_position_builder_aligns_image_tokens_with_latent_reference_id():
    modality_types = torch.tensor([[0, 1, 1, 1, 1, 0]])
    attention_mask = torch.ones_like(modality_types, dtype=torch.bool)
    positions = build_qwen_condition_positions(
        modality_types, attention_mask, torch.tensor([[1, 4, 4]]),
        spatial_merge_size=2, reference_id=1,
    )
    latent_positions = grid_positions(
        1, 2, 2, device=torch.device("cpu"), reference_id=1,
    )

    assert positions[0, 1:5].tolist() == latent_positions[0].tolist()
    assert positions[0, 0, 0].item() == 0
    assert positions[0, 5, 0].item() == 5


def test_qwen_position_builder_rejects_grid_token_mismatch():
    with pytest.raises(ValueError, match="token count"):
        build_qwen_condition_positions(
            torch.tensor([[0, 1, 1]]), torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([[1, 4, 4]]), spatial_merge_size=2,
        )


def test_qwen_position_builder_supports_mixed_text_and_reference_batch():
    modality_types = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0]])
    attention_mask = torch.ones_like(modality_types, dtype=torch.bool)
    positions = build_qwen_condition_positions(
        modality_types, attention_mask, torch.tensor([[1, 2, 1]]),
        spatial_merge_size=1, reference_id=3,
    )

    assert positions[0, :, 0].tolist() == [0, 1, 2, 3]
    assert positions[1, 1:3].tolist() == [[3, 0, 0], [3, 1, 0]]


def test_multimodal_rope_supports_uneven_axis_sections():
    value = torch.randn(1, 2, 5, 64)
    positions = torch.randn(1, 5, 3)

    assert apply_multimodal_rope(value, positions).shape == value.shape
