from __future__ import annotations

import pytest
import torch

from core.layers import (
    PreNormConvFFNResidual2d,
    PreNormGatedConvFFNResidual2d,
)
from image_ae.train import (
    GatedResidualConvFFNBlock as ImageAEGatedBlock,
    ResidualConvFFNBlock as ImageAEBlock,
)
from image_gen.train import (
    GatedResidualConvFFNBlock as ImageGenGatedBlock,
    ResidualConvFFNBlock as ImageGenBlock,
)


@pytest.mark.parametrize(
    ("factory", "expected_keys"),
    [
        (
            lambda: ImageAEBlock(4, 7),
            [
                "block.0.rmsnorm.weight",
                "block.1.weight", "block.1.bias",
                "block.3.weight", "block.3.bias",
            ],
        ),
        (
            lambda: ImageAEGatedBlock(4, 7),
            [
                "block.0.rmsnorm.weight",
                "block.1.projection.weight", "block.1.projection.bias",
                "block.2.projection.weight", "block.2.projection.bias",
            ],
        ),
        (
            lambda: ImageGenBlock(4, 7),
            [
                "norm.rmsnorm.weight",
                "conv1.weight", "conv1.bias",
                "conv2.weight", "conv2.bias",
            ],
        ),
        (
            lambda: ImageGenGatedBlock(4, 7),
            [
                "norm.rmsnorm.weight",
                "conv1.projection.weight", "conv1.projection.bias",
                "conv2.projection.weight", "conv2.projection.bias",
            ],
        ),
    ],
)
def test_experiment_wrappers_keep_historic_checkpoint_keys(factory, expected_keys):
    block = factory()

    assert list(block.state_dict()) == expected_keys
    restored = factory()
    restored.load_state_dict(block.state_dict(), strict=True)


@pytest.mark.parametrize(
    "block",
    [
        PreNormConvFFNResidual2d(4, 7),
        PreNormConvFFNResidual2d(4, 7, state_layout="sequential"),
        PreNormGatedConvFFNResidual2d(4, 7),
        PreNormGatedConvFFNResidual2d(4, 7, state_layout="sequential"),
    ],
)
def test_shared_conv_ffn_blocks_preserve_rectangular_shape_and_backprop(block):
    x = torch.randn(2, 4, 5, 7, requires_grad=True)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
    )


def test_zero_last_conv_ffn_starts_as_identity():
    block = PreNormConvFFNResidual2d(4, 7, zero_last=True)
    x = torch.randn(2, 4, 5, 7)

    assert torch.equal(block(x), x)


def test_experiment_wrappers_keep_historic_default_hidden_width():
    assert ImageAEBlock(4).block[1].out_channels == 8
    assert ImageGenBlock(4, 0).conv1.out_channels == 8
    assert ImageGenGatedBlock(4, 0).conv1.projection.out_channels == 16


def test_shared_conv_ffn_blocks_reject_invalid_dimensions_and_layouts():
    with pytest.raises(ValueError, match="channels must be positive"):
        PreNormConvFFNResidual2d(0)
    with pytest.raises(ValueError, match="hidden_channels must be positive"):
        PreNormGatedConvFFNResidual2d(4, 0)
    with pytest.raises(ValueError, match="state_layout"):
        PreNormConvFFNResidual2d(4, state_layout="unknown")


@pytest.mark.parametrize(
    "block",
    [
        ImageAEBlock(4, 7),
        ImageAEGatedBlock(4, 7),
        ImageGenBlock(4, 7),
        ImageGenGatedBlock(4, 7),
    ],
)
def test_experiment_wrappers_support_pre_layout_full_object_state(block):
    # Simulate older full-object pickles that predate the state_layout field.
    del block.state_layout
    x = torch.randn(1, 4, 3, 5)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
