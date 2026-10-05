import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verify.qwen_vae import encode_probe, parse_size


class _DummyLatentDistribution:
    def __init__(self, images):
        self.images = images

    def sample(self):
        height, width = self.images.shape[-2:]
        return torch.zeros(
            self.images.shape[0], 4, height // 8, width // 8,
            dtype=self.images.dtype, device=self.images.device,
        )


class _DummyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def encode(self, images):
        return SimpleNamespace(latent_dist=_DummyLatentDistribution(images))


def test_verify_size_parser_preserves_height_and_width():
    assert parse_size("32x40") == (32, 40)
    assert parse_size("16X8") == (16, 8)


@pytest.mark.parametrize("value", ["32", "0x8", "8x0", "textx8"])
def test_verify_size_parser_rejects_invalid_sizes(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_size(value)


def test_encode_probe_reports_image_and_latent_shape_contract():
    result = encode_probe(_DummyVAE(), (16, 24), latent_scale=1.0)

    assert result["input_shape"] == [1, 3, 16, 24]
    assert result["latent_shape"] == [1, 4, 2, 3]
    assert result["stride"] == [8, 8]
    assert result["finite"] is True
