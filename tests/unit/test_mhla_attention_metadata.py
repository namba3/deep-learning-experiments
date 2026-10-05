import json
from typing import Any, cast

import pytest
import torch
from torch import nn

from image_gen.train import (
    ContextSelfAttention,
    HEAD_GATE_SCALE,
    JointMHLA,
    MMDiTJointAttention,
    PerformanceAccumulator,
    install_backward_timing_hooks,
    summarize_gpu_telemetry,
    TwoDRoPECache,
    TimingAccumulator,
    _joint_mhla_naive,
    _joint_mhla_vectorized,
    apply_rope_pairs,
    write_performance_jsonl,
)


def test_image_gen_head_gate_zero_initialization_preserves_magnitude():
    assert HEAD_GATE_SCALE == 2.0
    zero_gate = torch.zeros(4)
    assert torch.allclose(
        HEAD_GATE_SCALE * torch.sigmoid(zero_gate),
        torch.ones_like(zero_gate),
    )


def _inputs():
    torch.manual_seed(0)
    latent = torch.randn(2, 4, 16)
    image = torch.randn(2, 1, 16)
    text = torch.randn(2, 3, 16)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    return latent, image, text, mask


def test_joint_mhla_metadata_matches_fallback_path():
    module = JointMHLA(
        dim=16,
        heads=4,
        kv_heads=2,
        latent_blocks=2,
        image_blocks=1,
        text_blocks=2,
        backend="vectorized",
    ).float().eval()
    latent, image, text, mask = _inputs()
    metadata = module.prepare_attention_metadata(
        2, 2, 1, 1, text.shape[1], latent.shape[0], latent.device, mask,
    )
    fallback = module(latent, image, text, 2, 2, 1, 1, mask)
    prepared = module(
        latent, image, text, 2, 2, 1, 1, mask, metadata,
    )
    for expected, actual in zip(fallback, prepared):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_joint_mhla_timing_records_forward_without_changing_output():
    module = JointMHLA(
        dim=16,
        heads=4,
        kv_heads=2,
        latent_blocks=2,
        image_blocks=1,
        text_blocks=2,
        backend="vectorized",
    ).float().eval()
    latent, image, text, mask = _inputs()
    timing = TimingAccumulator(torch.device("cpu"), enabled=True)
    timing.step_completed()
    expected = module(latent, image, text, 2, 2, 1, 1, mask)
    actual = module(latent, image, text, 2, 2, 1, 1, mask, timing=timing)
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.allclose(expected_tensor, actual_tensor, atol=1e-6, rtol=1e-5)
    report = timing.report_and_reset()
    assert report["mhla_forward"] >= 0.0


def test_performance_report_writes_jsonl_with_timing_and_metadata(tmp_path):
    performance = PerformanceAccumulator(torch.device("cpu"), enabled=True)
    with performance.measure("example"):
        _ = torch.zeros(1)
    with performance.measure_host("data_wait"):
        _ = torch.zeros(1)
    performance.add_metric("items", 2)
    performance.step_completed()
    report = performance.report_and_reset()
    report["host_seconds_per_optimizer_step"] = {"example": 0.25}
    path = tmp_path / "performance.jsonl"
    write_performance_jsonl(path, report, epoch=1, global_step=3)
    record = json.loads(path.read_text())
    assert record["epoch"] == 2
    assert record["global_step"] == 3
    assert record["optimizer_steps"] == 1
    assert record["seconds_per_optimizer_step"]["example"] >= 0.0
    assert "cuda_memory" not in record["seconds_per_optimizer_step"]
    assert "metrics" not in record["seconds_per_optimizer_step"]
    assert "host_seconds_per_optimizer_step" not in record["seconds_per_optimizer_step"]
    assert record["host_seconds_per_optimizer_step"]["example"] == 0.25
    assert record["metrics"]["items"] == 2.0


def test_host_measurement_is_reported_in_cpu_timing():
    performance = PerformanceAccumulator(torch.device("cpu"), enabled=True)
    with performance.measure_host("data_wait"):
        _ = torch.zeros(1)
    performance.step_completed()
    report = performance.report_and_reset()
    assert report["data_wait"] >= 0.0


def test_optimizer_breakdown_is_opt_in():
    performance = PerformanceAccumulator(
        torch.device("cpu"), enabled=True, optimizer_breakdown=True,
    )
    with performance.measure_optimizer("optimizer_stage"):
        _ = torch.zeros(1)
    performance.step_completed()
    report = performance.report_and_reset()
    assert report["optimizer_stage"] >= 0.0


def test_rope_cache_converts_inference_tensors_before_autograd_reuse():
    cache = TwoDRoPECache(head_dim=8)
    with torch.inference_mode():
        inference_cos, inference_sin = cache.get(
            2, 2, torch.device("cpu"), torch.float32,
        )
    assert inference_cos.is_inference()
    assert inference_sin.is_inference()

    cos, sin = cache.get(2, 2, torch.device("cpu"), torch.float32)
    assert not cos.is_inference()
    assert not sin.is_inference()
    tensor = torch.randn(1, 1, 4, 8, requires_grad=True)
    apply_rope_pairs(tensor, cos, sin).square().mean().backward()
    assert tensor.grad is not None


def test_gpu_telemetry_summary_contains_range_statistics():
    summary = cast(dict[str, Any], summarize_gpu_telemetry([
        {"gpu_utilization_percent": 40.0, "temperature_c": 60.0},
        {"gpu_utilization_percent": 80.0, "temperature_c": 70.0},
    ]))
    assert summary["samples"] == 2
    assert summary["gpu_utilization_percent"] == {
        "mean": 60.0, "min": 40.0, "max": 80.0,
    }
    assert summary["temperature_c"]["max"] == 70.0


def test_backward_timing_hooks_report_component_spans():
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.joint_attn = nn.Linear(4, 4)
            self.latent_ffn = nn.Linear(4, 4)
            self.image_ffn = nn.Linear(4, 4)
            self.text_ffn = nn.Linear(4, 4)

        def forward(self, x):
            return (
                self.joint_attn(x)
                + self.latent_ffn(x)
                + self.image_ffn(x)
                + self.text_ffn(x)
            )

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([TinyBlock()])

        def forward(self, x):
            return self.blocks[0](x)

    model = TinyModel()
    performance = PerformanceAccumulator(torch.device("cpu"), enabled=True)
    handles = install_backward_timing_hooks(model, performance)
    try:
        model(torch.randn(2, 4)).square().mean().backward()
        performance.step_completed()
        report = performance.report_and_reset()
    finally:
        for handle in handles:
            handle.remove()

    assert report["dit_block_backward"] >= 0.0
    assert report["dit_attention_backward"] >= 0.0
    assert report["dit_ffn_backward"] >= 0.0


def test_full_attention_mask_matches_fallback_path():
    module = MMDiTJointAttention(dim=16, heads=4, kv_heads=2).float().eval()
    latent, image, text, mask = _inputs()
    attention_mask = torch.nn.functional.pad(
        mask, (latent.shape[1] + image.shape[1], 0), value=True,
    )[:, None, None, :]
    fallback = module(latent, image, text, 2, 2, 1, 1, mask)
    prepared = module(
        latent, image, text, 2, 2, 1, 1, mask, attention_mask,
    )
    for expected, actual in zip(fallback, prepared):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_joint_mhla_preserves_rectangular_stream_shapes_and_masks():
    module = JointMHLA(
        dim=16,
        heads=4,
        kv_heads=2,
        latent_blocks=3,
        image_blocks=2,
        text_blocks=2,
        backend="vectorized",
    ).float().eval()
    latent = torch.randn(2, 6, 16, requires_grad=True)
    image = torch.randn(2, 2, 16)
    text = torch.randn(2, 3, 16)
    text_mask = torch.tensor([[True, False, True], [False, False, False]])

    outputs = module(latent, image, text, 2, 3, 1, 2, text_mask)

    assert [output.shape for output in outputs] == [
        (2, 6, 16), (2, 2, 16), (2, 3, 16),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    sum(output.square().mean() for output in outputs).backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
def test_joint_mhla_naive_and_vectorized_match_forward_backward(
    dtype, atol, rtol,
):
    torch.manual_seed(7)
    batch, heads, kv_heads, tokens, head_dim = 2, 4, 2, 15, 4
    block_indices = [
        torch.tensor([0, 1, 4, 5]),
        torch.tensor([2, 3, 6]),
        torch.tensor([7, 8, 9, 10, 11, 12, 13, 14]),
    ]
    block_modalities = torch.tensor([0, 1, 2], dtype=torch.long)
    valid_mask = torch.tensor([
        [True, True, True, True, True, False, True, True, True,
         True, True, True, False, True, True],
        [False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False],
    ])
    modality_bias = torch.tensor([
        [0.0, 0.1, -0.2],
        [0.1, 0.0, 0.3],
        [-0.2, 0.3, 0.0],
    ], dtype=dtype)
    query = torch.randn(batch, heads, tokens, head_dim, dtype=dtype)
    key = torch.randn(batch, kv_heads, tokens, head_dim, dtype=dtype)
    value = torch.randn(batch, kv_heads, tokens, head_dim, dtype=dtype)
    naive_inputs = [tensor.detach().clone().requires_grad_() for tensor in
                    (query, key, value, modality_bias)]
    vectorized_inputs = [tensor.detach().clone().requires_grad_() for tensor in
                         (query, key, value, modality_bias)]

    naive = _joint_mhla_naive(
        *naive_inputs[:3], block_indices, block_modalities, valid_mask,
        heads, kv_heads, naive_inputs[3],
    )
    vectorized = _joint_mhla_vectorized(
        *vectorized_inputs[:3], block_indices, block_modalities, valid_mask,
        heads, kv_heads, vectorized_inputs[3],
    )

    assert naive.shape == (batch, heads, tokens, head_dim)
    assert torch.isfinite(naive).all()
    assert torch.isfinite(vectorized).all()
    assert torch.allclose(naive, vectorized, atol=atol, rtol=rtol)

    naive.square().float().mean().backward()
    vectorized.square().float().mean().backward()
    for naive_input, vectorized_input in zip(naive_inputs, vectorized_inputs):
        assert naive_input.grad is not None
        assert vectorized_input.grad is not None
        assert torch.isfinite(naive_input.grad).all()
        assert torch.isfinite(vectorized_input.grad).all()
        assert torch.allclose(
            naive_input.grad,
            vectorized_input.grad,
            atol=atol * 2,
            rtol=rtol * 2,
        )


def test_attention_boundaries_reject_grid_and_mask_mismatches():
    module = MMDiTJointAttention(dim=16, heads=4, kv_heads=2).float().eval()
    latent, image, text, mask = _inputs()
    with pytest.raises(ValueError, match=r"does not match height\*width"):
        module(latent, image, text, 2, 3, 1, 1, mask)
    with pytest.raises(ValueError, match="mask must have shape"):
        module(latent, image, text, 2, 2, 1, 1, torch.ones(2, 2, dtype=torch.bool))

    context = ContextSelfAttention(dim=16, heads=4, kv_heads=2).float().eval()
    tokens = torch.randn(2, 6, 16)
    context_mask = torch.ones(2, 6, dtype=torch.bool)
    output = context(tokens, 2, 2, 4, context_mask)
    assert output.shape == tokens.shape
    with pytest.raises(ValueError, match=r"does not match image_height\*image_width"):
        context(tokens, 1, 2, 4, context_mask)
