import torch

import pytest
from types import SimpleNamespace
from torch import nn

from image_gen.generate_samples import build_checkpoint_models
from image_gen.train import (
    DiT,
    ImageContextEmbedder,
    encode_images,
    flow_velocity_target,
    resolve_vae_latent_scale,
    validate_bucket_shapes_with_vae,
)


class _DummyLatentDistribution:
    def __init__(self, images):
        self.images = images

    def sample(self):
        height, width = self.images.shape[-2:]
        return torch.zeros(
            self.images.shape[0], 4, max(1, height // 8), max(1, width // 8),
            device=self.images.device, dtype=self.images.dtype,
        )


class _DummyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def encode(self, images):
        return SimpleNamespace(latent_dist=_DummyLatentDistribution(images))


def _small_checkpoint_config():
    return {
        "text_adapter_dim": 16,
        "text_adapter_transformer_dims": (16,),
        "text_adapter_transformer_heads": (4,),
        "text_adapter_transformer_kv_heads": (2,),
        "text_adapter_transformer_ff_mult": 2.0,
        "text_adapter_rope_theta": 1000.0,
        "model_dim": 32,
        "depth": 1,
        "heads": 4,
        "kv_heads": 2,
        "patch_size": 2,
        "context_depth": 1,
        "context_heads": 4,
        "context_kv_heads": 2,
        "attention_gate": "head",
        "attention_pattern": "mhla",
        "mhla_latent_blocks": 2,
        "mhla_image_blocks": 2,
        "mhla_text_blocks": 2,
        "mhla_backend": "naive",
        "mhla_recompute_output": True,
    }


def test_checkpoint_model_builder_restores_architecture_and_strict_loads():
    config = _small_checkpoint_config()
    first_dit, first_adapter = build_checkpoint_models(
        config, text_dim=8, latent_channels=4,
        reference_latent_height=4, reference_latent_width=4,
        device=torch.device("cpu"),
    )
    second_dit, second_adapter = build_checkpoint_models(
        config, text_dim=8, latent_channels=4,
        reference_latent_height=4, reference_latent_width=4,
        device=torch.device("cpu"),
    )

    second_dit.load_state_dict(first_dit.state_dict(), strict=True)
    second_adapter.load_state_dict(first_adapter.state_dict(), strict=True)
    assert second_dit.blocks[0].joint_attn.kv_heads == 2
    assert second_dit.context_transformer.blocks[0].self_attn.kv_heads == 2
    assert second_adapter.transformer_blocks[0].kv_heads == 2
    assert second_dit.blocks[0].joint_attn.recompute_output is True


@pytest.mark.parametrize("context_dim", [1, 16, 32])
def test_image_context_embedder_handles_one_by_one_context_for_batch_one(context_dim):
    embedder = ImageContextEmbedder(in_channels=4, context_dim=context_dim)
    output = embedder(torch.randn(1, 4, 8, 8))
    assert output.shape == (1, context_dim, 1, 1)
    assert torch.isfinite(output).all()


def test_image_context_embedder_rejects_spatial_inputs_below_eight():
    embedder = ImageContextEmbedder(in_channels=4, context_dim=16)
    with pytest.raises(ValueError, match="spatial dimensions >= 8"):
        embedder(torch.randn(1, 4, 4, 8))


def test_dit_minimum_latent_grid_preserves_rectangular_output_shape():
    model = DiT(
        image_channels=4,
        context_dim=32,
        dim=32,
        depth=1,
        heads=4,
        patch_size=2,
        reference_height=32,
        reference_width=32,
        context_depth=1,
        context_heads=4,
        kv_heads=2,
        context_kv_heads=2,
        attention_pattern="full",
        mhla_backend="vectorized",
    ).eval()
    latent = torch.randn(1, 4, 16, 24)
    output = model(
        latent,
        torch.rand(1),
        torch.randn(1, 1, 32),
        torch.ones(1, 1, dtype=torch.bool),
    )

    assert output.shape == latent.shape
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="even and >= 16"):
        model(
            torch.randn(1, 4, 17, 16),
            torch.rand(1),
            torch.randn(1, 1, 32),
            torch.ones(1, 1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="even and >= 16"):
        model(
            torch.randn(1, 4, 8, 8),
            torch.rand(1),
            torch.randn(1, 1, 32),
            torch.ones(1, 1, dtype=torch.bool),
        )


def test_bucket_shapes_have_a_consistent_integer_vae_stride():
    vae = _DummyVAE()

    assert validate_bucket_shapes_with_vae(
        vae, ((32, 32), (64, 40)), latent_scale=1.0,
    ) == (8, 8)
    with pytest.raises(ValueError, match="VAE stride differs across buckets"):
        validate_bucket_shapes_with_vae(vae, ((32, 32), (36, 40)))


def test_vae_latent_scale_defaults_to_one_when_config_is_none():
    vae = _DummyVAE()
    vae.config = SimpleNamespace(scaling_factor=None)

    assert resolve_vae_latent_scale(vae) == 1.0
    assert resolve_vae_latent_scale(vae, 0.25) == 0.25
    encoded = encode_images(vae, torch.zeros(1, 3, 16, 16))
    assert encoded.shape == (1, 4, 2, 2)
    assert torch.equal(encoded, torch.zeros_like(encoded))


def test_bucket_shapes_probe_a_real_diffusers_vae_on_cpu():
    diffusers = pytest.importorskip("diffusers")
    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=32,
    ).eval()

    assert validate_bucket_shapes_with_vae(
        vae, ((32, 32), (64, 40)), latent_scale=1.0,
    ) == (2, 2)


@pytest.mark.parametrize("prediction_type", ["rectified_flow", "flow_matching"])
def test_flow_prediction_types_use_the_declared_straight_path_target(prediction_type):
    clean = torch.randn(2, 4, 3, 3)
    noise = torch.randn_like(clean)

    target = flow_velocity_target(clean, noise, prediction_type)

    assert torch.equal(target, noise - clean)
