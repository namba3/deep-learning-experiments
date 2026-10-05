import pickle

import torch
import core.layers as core_layers

from core.layers import (
    DiTTransformerBlock, DynamicTransformerEncoder, GatedConv2d,
    GroupedQueryAttention, SharedInputDepthAttention,
    SharedInputDepthTransformerBlock, SharedInputDepthTransformerEncoder,
)


def test_core_layers_exports_preserve_historic_pickle_module_path() -> None:
    for cls in (DiTTransformerBlock, GatedConv2d, GroupedQueryAttention):
        assert cls.__module__ == "core.layers"

    layer = GatedConv2d(3, 5)
    restored = pickle.loads(pickle.dumps(layer))
    assert type(restored) is GatedConv2d
    torch.testing.assert_close(restored.projection.weight, layer.projection.weight)


def test_depth_layer_exports_keep_historic_pickle_module_path() -> None:
    depth_layers = (
        DynamicTransformerEncoder,
        SharedInputDepthAttention,
        SharedInputDepthTransformerBlock,
        SharedInputDepthTransformerEncoder,
    )
    assert all(layer.__module__ == "core.layers" for layer in depth_layers)
    assert all(layer.__name__ in core_layers.__all__ for layer in depth_layers)
    assert all(pickle.loads(pickle.dumps(layer)) is layer for layer in depth_layers)
