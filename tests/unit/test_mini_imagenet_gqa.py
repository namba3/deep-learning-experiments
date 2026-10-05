from __future__ import annotations

import json
from typing import cast

import pytest
import torch

from core.layers import AdaRMSNorm, GatedFFN, ScaleOnlyAdaRMSNorm
from mini_imagenet_gqa.model import (
    GQATransformerBlock2D,
    MiniImageNetGQAModel,
    SpatialGQAStage,
    VARIANTS,
    _ada_scale_offset,
)
from mini_imagenet_gqa.summarize_comparison import summarize, write_reports
from mini_imagenet_gqa.train import (
    APOLLO_SF_OPTIMIZERS,
    build_transforms,
    parse_args,
    parse_widths,
    validate_args,
)


def test_mini_imagenet_optimizer_parser_exposes_apollo_sf_storage_variants():
    for optimizer in APOLLO_SF_OPTIMIZERS:
        args = parse_args([
            "--optimizer", optimizer,
            "--apollo-sf-quant-block-size", "128",
            "--device", "cpu",
        ])
        validate_args(args)
        assert args.optimizer == optimizer
        assert args.apollo_sf_quant_block_size == 128


def test_mini_imagenet_apollo_sf_delta_refresh_parser():
    args = parse_args([
        "--optimizer", "APOLLO-SF-INT8-Delta",
        "--apollo-sf-delta-refresh", "blend",
        "--apollo-sf-delta-refresh-window", "4",
        "--apollo-update-proj-gap", "17",
        "--device", "cpu",
    ])
    validate_args(args)
    assert args.apollo_sf_delta_refresh == "blend"
    assert args.apollo_sf_delta_refresh_window == 4
    assert args.apollo_update_proj_gap == 17


@pytest.mark.parametrize("variant", VARIANTS)
def test_gqa_classifier_variant_forward_and_backward(variant):
    model = MiniImageNetGQAModel(
        num_classes=7,
        widths=(16, 32),
        heads=4,
        kv_heads=2,
        blocks_per_stage=1,
        image_size=32,
        variant=variant,
        dropout=0.0,
        ff_mult=3.0,
    )
    images = torch.randn(2, 3, 32, 32)
    metadata = torch.tensor([[4.0, 0.0], [4.4, 0.2]])
    labels = torch.tensor([1, 5])

    logits = model(images, metadata=metadata)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    from mini_imagenet_gqa.model import SpatialGQAStage

    stage = cast(SpatialGQAStage, model.stages[0])
    block = stage.blocks[0]
    assert isinstance(block.q_norm, torch.nn.RMSNorm)
    assert isinstance(block.k_norm, torch.nn.RMSNorm)
    if variant in {
        "ada_gated_gqa_silu_gated_ffn",
        "ada_1plus_silu_gated_gqa_silu_gated_ffn",
        "ada_silu_scale_gated_gqa_silu_gated_ffn",
        "ada_2sigmoid_gated_gqa_silu_gated_ffn",
        "ada_silu1_norm_gated_gqa_silu_gated_ffn",
        "ada_softplus1_norm_gated_gqa_silu_gated_ffn",
        "ada_shift_gated_gqa_silu_gated_ffn",
        "ada_1plus_silu_shift_gated_gqa_silu_gated_ffn",
        "ada_2sigmoid_shift_gated_gqa_silu_gated_ffn",
        "ada_softplus1_norm_shift_gated_gqa_silu_gated_ffn",
        "branch_qkv_meta_ada_attn_ffn_gated_gqa_silu_gated_ffn",
    }:
        from core.layers import AdaRMSScaleProjection, ScaleOnlyAdaRMSNorm

        if variant in {
            "ada_shift_gated_gqa_silu_gated_ffn",
            "ada_1plus_silu_shift_gated_gqa_silu_gated_ffn",
            "ada_2sigmoid_shift_gated_gqa_silu_gated_ffn",
            "ada_softplus1_norm_shift_gated_gqa_silu_gated_ffn",
        }:
            assert isinstance(block.norm1, AdaRMSNorm)
            assert isinstance(block.norm2, AdaRMSNorm)
            assert isinstance(block.norm1_shift, AdaRMSScaleProjection)
            assert isinstance(block.norm2_shift, AdaRMSScaleProjection)
            for projection in (block.norm1_shift, block.norm2_shift):
                assert projection is not None
                assert torch.count_nonzero(projection.proj.weight) == 0
                assert torch.count_nonzero(projection.proj.bias) == 0
                assert projection.proj.weight.grad is not None
                assert torch.count_nonzero(projection.proj.weight.grad) > 0
        else:
            assert isinstance(block.norm1, ScaleOnlyAdaRMSNorm)
            assert isinstance(block.norm2, ScaleOnlyAdaRMSNorm)
            assert block.norm1_shift is None
            assert block.norm2_shift is None
        assert isinstance(block.norm1_scale, AdaRMSScaleProjection)
        assert isinstance(block.norm2_scale, AdaRMSScaleProjection)
        norm1_weight = block.norm1_scale.proj.weight
        norm2_weight = block.norm2_scale.proj.weight
        assert torch.count_nonzero(norm1_weight) == 0
        assert torch.count_nonzero(norm2_weight) == 0
        assert norm1_weight.grad is not None and torch.count_nonzero(norm1_weight.grad) > 0
        assert norm2_weight.grad is not None
        if variant == "ada_silu_scale_gated_gqa_silu_gated_ffn":
            # SiLU(0)=0 and the bias-free SwiGLU product has zero derivative
            # when its input is zero, so this FFN Ada projection is initially dead.
            assert torch.count_nonzero(norm2_weight.grad) == 0
            assert block.norm2_scale.proj.bias is not None
            assert block.norm2_scale.proj.bias.grad is not None
            assert torch.count_nonzero(block.norm2_scale.proj.bias.grad) == 0
            assert isinstance(block.ffn, torch.nn.Sequential)
            gated_ffn = block.ffn[0]
            assert isinstance(gated_ffn, GatedFFN)
            assert gated_ffn.gated.proj.weight.grad is not None
            assert torch.count_nonzero(gated_ffn.gated.proj.weight.grad) == 0
        else:
            assert torch.count_nonzero(norm2_weight.grad) > 0
    if block.head_gate is not None:
        assert isinstance(block.head_gate, torch.nn.Linear)
        assert torch.count_nonzero(block.head_gate.weight) == 0
        assert block.head_gate.bias is not None
        assert torch.count_nonzero(block.head_gate.bias) == 0



def test_ada_rms_norm_shift_broadcast_and_identity_initialization():
    norm = AdaRMSNorm(4, eps=1e-6)
    x = torch.randn(2, 3, 4)
    zero = torch.zeros(2, 4)

    actual = norm(x, zero, zero)
    expected = torch.nn.functional.rms_norm(x, (4,), eps=1e-6)
    torch.testing.assert_close(actual, expected)

    shift = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    shifted = norm(x, zero, shift)
    torch.testing.assert_close(shifted - actual, shift[:, None, :].expand_as(x))

    with pytest.raises(ValueError, match="shift must have shape"):
        norm(x, zero, torch.zeros(2, 1, 4))



@pytest.mark.parametrize(
    ("scale_only", "scale_shift"),
    [
        ("ada_gated_gqa_silu_gated_ffn", "ada_shift_gated_gqa_silu_gated_ffn"),
        (
            "ada_1plus_silu_gated_gqa_silu_gated_ffn",
            "ada_1plus_silu_shift_gated_gqa_silu_gated_ffn",
        ),
        (
            "ada_2sigmoid_gated_gqa_silu_gated_ffn",
            "ada_2sigmoid_shift_gated_gqa_silu_gated_ffn",
        ),
    ],
)
def test_scale_and_shift_variants_preserve_matching_initialization(scale_only, scale_shift):
    torch.manual_seed(71)
    scale_model = MiniImageNetGQAModel(
        num_classes=5, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant=scale_only, dropout=0.0,
    )
    torch.manual_seed(71)
    shift_model = MiniImageNetGQAModel(
        num_classes=5, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant=scale_shift, dropout=0.0,
    )

    scale_state = scale_model.state_dict()
    shift_state = shift_model.state_dict()
    for name, tensor in scale_state.items():
        torch.testing.assert_close(tensor, shift_state[name], rtol=0, atol=0)
    for name, tensor in shift_state.items():
        if any(name.endswith(suffix) for suffix in (
            "norm1_shift.proj.weight", "norm1_shift.proj.bias",
            "norm2_shift.proj.weight", "norm2_shift.proj.bias",
        )):
            assert torch.count_nonzero(tensor) == 0


def test_ada_arm_matches_nonadaptive_silu_gated_ffn_control():
    control = MiniImageNetGQAModel(
        num_classes=5, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant="gated_gqa_silu_gated_ffn",
        dropout=0.0, ff_mult=3.0,
    )
    adaptive = MiniImageNetGQAModel(
        num_classes=5, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant="ada_gated_gqa_silu_gated_ffn",
        dropout=0.0, ff_mult=3.0,
    )
    control_block = cast(SpatialGQAStage, control.stages[0]).blocks[0]
    adaptive_block = cast(SpatialGQAStage, adaptive.stages[0]).blocks[0]

    assert control_block.gate_mode == adaptive_block.gate_mode == "silu"
    assert isinstance(control_block.norm1, torch.nn.RMSNorm)
    assert isinstance(adaptive_block.norm1, ScaleOnlyAdaRMSNorm)
    assert isinstance(control_block.ffn, torch.nn.Sequential)
    assert isinstance(adaptive_block.ffn, torch.nn.Sequential)
    assert isinstance(control_block.ffn[0], GatedFFN)
    assert isinstance(adaptive_block.ffn[0], GatedFFN)
    assert sum(parameter.numel() for parameter in adaptive.parameters()) > sum(
        parameter.numel() for parameter in control.parameters()
    )


def test_block_gate_variants_start_at_unit_residual_scale():
    tokens = torch.randn(2, 6, 16)
    sigmoid = GQATransformerBlock2D(
        16, 4, 2, grid_size=2, variant="gated_gqa_sigmoid", dropout=0.0,
    )
    silu = GQATransformerBlock2D(
        16, 4, 2, grid_size=2, variant="gated_gqa_silu", dropout=0.0,
    )

    sigmoid_projection = sigmoid.head_gate
    silu_projection = silu.head_gate
    assert sigmoid_projection is not None and silu_projection is not None
    sigmoid_gate = 2 * torch.sigmoid(sigmoid_projection(sigmoid.norm1(tokens)))
    silu_gate = 1 + torch.nn.functional.silu(silu_projection(silu.norm1(tokens)))

    assert torch.equal(sigmoid_gate, torch.ones_like(sigmoid_gate))
    assert torch.equal(silu_gate, torch.ones_like(silu_gate))




@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("linear", lambda value: 1 + value),
        ("one_sided_shifted_silu", lambda value: 1 + torch.nn.functional.silu(value)),
        ("one_sided_silu", lambda value: torch.nn.functional.silu(value)),
        ("two_sided_sigmoid", lambda value: 2 * torch.sigmoid(value)),
        (
            "normalized_shifted_silu",
            lambda value: torch.nn.functional.silu(1 + value)
            / torch.nn.functional.silu(torch.ones_like(value)),
        ),
        (
            "normalized_shifted_softplus",
            lambda value: torch.nn.functional.softplus(1 + value)
            / torch.nn.functional.softplus(torch.ones_like(value)),
        ),
    ],
)
def test_ada_scale_modes_match_declared_multipliers(mode, expected):
    raw = torch.tensor([[-4.0, -1.0, 0.0, 1.0, 4.0]])
    actual = 1 + _ada_scale_offset(raw, mode)
    torch.testing.assert_close(actual, expected(raw))


def test_normalized_shifted_silu_ada_starts_at_identity_and_learns():
    variant = "ada_silu1_norm_gated_gqa_silu_gated_ffn"
    model = MiniImageNetGQAModel(
        num_classes=3, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant=variant, dropout=0.0,
    )
    block = cast(SpatialGQAStage, model.stages[0]).blocks[0]
    assert block.norm1_scale is not None and block.norm2_scale is not None
    for projection in (block.norm1_scale, block.norm2_scale):
        assert torch.count_nonzero(projection.proj.weight) == 0
        assert projection.proj.bias is not None
        torch.testing.assert_close(
            1 + _ada_scale_offset(projection.proj.bias, "normalized_shifted_silu"),
            torch.ones_like(projection.proj.bias),
        )

    logits = model(
        torch.randn(2, 3, 16, 16),
        metadata=torch.tensor([[4.0, 0.0], [4.2, 0.1]]),
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert block.norm1_scale.proj.weight.grad is not None
    assert torch.count_nonzero(block.norm1_scale.proj.weight.grad) > 0
    assert block.norm2_scale.proj.weight.grad is not None
    assert torch.count_nonzero(block.norm2_scale.proj.weight.grad) > 0


def test_normalized_softplus_ada_starts_at_identity_and_stays_positive():
    variant = "ada_softplus1_norm_gated_gqa_silu_gated_ffn"
    model = MiniImageNetGQAModel(
        num_classes=3, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant=variant, dropout=0.0,
    )
    block = cast(SpatialGQAStage, model.stages[0]).blocks[0]
    assert block.norm1_scale is not None and block.norm2_scale is not None
    for projection in (block.norm1_scale, block.norm2_scale):
        assert torch.count_nonzero(projection.proj.weight) == 0
        assert projection.proj.bias is not None
        torch.testing.assert_close(
            1 + _ada_scale_offset(projection.proj.bias, "normalized_shifted_softplus"),
            torch.ones_like(projection.proj.bias),
        )
    raw = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    multiplier = 1 + _ada_scale_offset(raw, "normalized_shifted_softplus")
    assert torch.all(multiplier > 0)

    logits = model(
        torch.randn(2, 3, 16, 16),
        metadata=torch.tensor([[4.0, 0.0], [4.2, 0.1]]),
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert block.norm1_scale.proj.weight.grad is not None
    assert torch.count_nonzero(block.norm1_scale.proj.weight.grad) > 0
    assert block.norm2_scale.proj.weight.grad is not None
    assert torch.count_nonzero(block.norm2_scale.proj.weight.grad) > 0


def test_direct_silu_ada_retains_zero_projection_initialization():
    model = MiniImageNetGQAModel(
        num_classes=3, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant="ada_silu_scale_gated_gqa_silu_gated_ffn",
        dropout=0.0,
    )
    block = cast(SpatialGQAStage, model.stages[0]).blocks[0]
    for projection in (block.norm1_scale, block.norm2_scale):
        assert projection is not None
        assert torch.count_nonzero(projection.proj.weight) == 0
        assert projection.proj.bias is not None
        assert torch.count_nonzero(projection.proj.bias) == 0
        raw_scale = projection.proj.bias
        actual_multiplier = torch.nn.functional.silu(raw_scale)
        torch.testing.assert_close(actual_multiplier, torch.zeros_like(actual_multiplier))

    logits = model(
        torch.randn(2, 3, 16, 16),
        metadata=torch.tensor([[4.0, 0.0], [4.2, 0.1]]),
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert block.norm1_scale.proj.weight.grad is not None
    assert torch.count_nonzero(block.norm1_scale.proj.weight.grad) > 0
    assert block.norm2_scale.proj.weight.grad is not None
    assert torch.count_nonzero(block.norm2_scale.proj.weight.grad) == 0
    assert isinstance(block.ffn, torch.nn.Sequential)
    assert isinstance(block.ffn[0], GatedFFN)
    assert block.ffn[0].gated.proj.weight.grad is not None
    assert torch.count_nonzero(block.ffn[0].gated.proj.weight.grad) == 0


def test_cifar_augmentation_order_and_imagenet_normalization():
    train_transform, eval_transform = build_transforms((64, 64))

    assert [type(transform).__name__ for transform in train_transform.transforms] == [
        "RandomHorizontalFlip", "ColorJitter", "RandomAffine", "RandomPerspective",
        "RandomResizedBucketCrop", "ToTensor", "Normalize",
    ]
    assert train_transform.transforms[2].degrees == [-10.0, 10.0]
    assert train_transform.transforms[3].distortion_scale == 0.1
    assert train_transform.transforms[4].scale == (0.5, 1.0)
    assert eval_transform.transforms[-1].mean == (0.485, 0.456, 0.406)


def test_dataset_view_returns_source_resolution_and_aspect_metadata():
    import math

    from PIL import Image

    from mini_imagenet_gqa.train import HFDatasetView

    class FakeDataset:
        def __getitem__(self, _index):
            return {"image": Image.new("RGB", (80, 40)), "label": 2}

        def __len__(self):
            return 1

    view = HFDatasetView(
        FakeDataset(), [lambda image: torch.zeros(3, 8, 8)],
        ("a", "b", "c"), ((8, 8),), "test",
    )
    image, label, metadata = view[0]

    assert image.shape == (3, 8, 8)
    assert label == 2
    assert metadata.tolist() == pytest.approx([math.log(8), 0.0])


def test_adaptive_variant_requires_metadata():
    model = MiniImageNetGQAModel(
        num_classes=3, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant="ada_gated_gqa_silu_gated_ffn", dropout=0.0,
    )
    with pytest.raises(ValueError, match="metadata"):
        model(torch.randn(1, 3, 16, 16))


def test_parse_widths_rejects_nonpositive_values():
    assert parse_widths("64, 128,256") == (64, 128, 256)
    from argparse import ArgumentTypeError

    with pytest.raises(ArgumentTypeError):
        parse_widths("64,nope")


def test_comparison_report_aggregates_completed_runs(tmp_path):
    runs = tmp_path / "runs"
    for seed, top1 in ((42, 0.7), (43, 0.8)):
        run_dir = runs / f"run-{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "run_id": f"run-{seed}",
            "args": {"variant": "gated_gqa_silu", "seed": seed},
        }))
        events = [
            {"event": "model", "parameters": 1000},
            {"event": "epoch", "epoch": 1, "train": {
                "samples": 40, "seconds": 2.0, "peak_allocated_mb": 300.0,
                "peak_reserved_mb": 400.0,
            }, "validation": {"top1": top1}},
            {"event": "test", "metrics": {"top1": top1 - 0.01, "loss": 0.9}},
        ]
        (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(item) for item in events))

    report = summarize(tmp_path)
    _json_path, markdown_path = write_reports(report, tmp_path)

    summary = report["variants"]["gated_gqa_silu"]
    assert summary["runs"] == 2
    assert summary["best_validation_top1"]["mean"] == pytest.approx(0.75)
    assert summary["test_top1"]["mean"] == pytest.approx(0.74)
    assert summary["train_samples_per_second"]["mean"] == pytest.approx(20.0)
    assert "gated_gqa_silu" in markdown_path.read_text()


def test_comparison_summary_rejects_mixed_protocols(tmp_path):
    runs = tmp_path / "runs"
    for run_id, step_cap in (("screen", 1), ("full", 0)):
        run_dir = runs / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "run_id": run_id,
            "args": {
                "variant": "gated_gqa_silu_gated_ffn",
                "seed": 42,
                "steps_per_epoch": step_cap,
                "eval_batches": 1,
                "device": "cuda",
            },
        }))
        events = [
            {"event": "model", "parameters": 1000},
            {"event": "epoch", "epoch": 1, "train": {"samples": 2, "seconds": 1.0},
             "validation": {"top1": 0.5}},
            {"event": "test", "metrics": {"top1": 0.4, "loss": 1.0}},
        ]
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events),
        )

    with pytest.raises(ValueError, match="incompatible comparison settings"):
        summarize(tmp_path)


def test_one_training_epoch_and_checkpoint_resume_round_trip(tmp_path):
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, TensorDataset

    from mini_imagenet_gqa.train import restore_checkpoint, run_epoch, save_checkpoint

    args = SimpleNamespace(
        variant="ada_gated_gqa_silu_gated_ffn",
        epochs=1,
        image_size=16,
        widths=(16,),
        heads=4,
        kv_heads=2,
        blocks_per_stage=1,
        dropout=0.0,
        ff_mult=3.0,
        dataset="synthetic",
        transform_degrees=10.0,
        transform_shear=10.0,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.01,
        amp="none",
        seed=17,
        steps_per_epoch=0,
        eval_batches=0,
        resume=None,
        output_dir="unused",
        run_name=None,
        hf_cache_dir=None,
        device="cpu",
        dry_run=False,
        validate_only=False,
        num_workers=0,
    )
    model = MiniImageNetGQAModel(
        num_classes=3, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant=args.variant, dropout=0.0, ff_mult=3.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(
        torch.randn(4, 3, 16, 16),
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([[2.0, 0.0], [2.1, 0.1], [2.2, -0.1], [2.3, 0.2]]),
    ), batch_size=2)

    metrics = run_epoch(model, loader, optimizer, torch.device("cpu"), amp="none")
    assert metrics["samples"] == 4
    assert metrics["samples_per_second"] > 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_checkpoint(
        checkpoint, model, args, epoch=1, global_step=2,
        optimizer=optimizer, scheduler=scheduler, save_resume_state=True,
    )
    restored = MiniImageNetGQAModel(
        num_classes=3, widths=(16,), heads=4, kv_heads=2,
        image_size=16, variant=args.variant, dropout=0.0, ff_mult=3.0,
    )
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=args.lr)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=args.epochs,
    )
    restored_position = restore_checkpoint(
        checkpoint, restored, restored_optimizer, restored_scheduler,
        torch.device("cpu"), args,
    )
    assert len(restored_position) == 2
    epoch, global_step = restored_position

    assert (epoch, global_step) == (1, 2)
    for actual, expected in zip(restored.parameters(), model.parameters(), strict=True):
        assert torch.equal(actual, expected)
