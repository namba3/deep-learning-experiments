import pytest
import torch
from vfp_dit.model import NoVFCBDiT


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


def test_single_stage_euler_sampler_supports_cfg_and_rejects_missing_unconditional_cache():
    from vfp_dit.generate_samples import sample_latents

    model = _tiny_model().eval()
    hidden = torch.randn(1, 4, 20)
    mask = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.zeros(1, 4, 3)
    positions[0, :, 0] = torch.arange(4)
    condition = model.prepare_condition(hidden, mask, positions)
    unconditional = model.prepare_condition(
        torch.zeros(1, 1, 20), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, 3),
    )
    generator = torch.Generator().manual_seed(4)
    sample = sample_latents(
        model, condition, latent_height=2, latent_width=3, steps=2,
        device=torch.device("cpu"), guidance_scale=2.0,
        unconditional_cache=unconditional, generator=generator,
    )

    assert sample.shape == (1, 4, 2, 3)
    assert torch.isfinite(sample).all()
    with pytest.raises(ValueError, match="unconditional"):
        sample_latents(
            model, condition, latent_height=2, latent_width=3, steps=2,
            device=torch.device("cpu"), guidance_scale=2.0,
        )


def test_original_qwen_image_vae_decode_undoes_latent_normalization():
    from types import SimpleNamespace

    from vfp_dit.encoders import decode_qwen_image_latents

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(latents_mean=[1.0] * 16, latents_std=[2.0] * 16)
            self.received: torch.Tensor | None = None

        def decode(self, value):
            self.received = value
            return SimpleNamespace(sample=torch.zeros(1, 3, 1, 4, 6))

    vae = TinyVAE()
    decoded = decode_qwen_image_latents(vae, torch.full((1, 16, 2, 3), 0.5))

    assert vae.received is not None
    assert vae.received.shape == (1, 16, 1, 2, 3)
    assert torch.equal(vae.received, torch.full_like(vae.received, 2.0))
    assert decoded.shape == (1, 3, 4, 6)


def test_simple_checkpoint_strict_roundtrip_preserves_cfg_sampler_output(tmp_path):
    from types import SimpleNamespace

    from safetensors.torch import load_file

    from vfp_dit_runtime.training import restore_training, save_model_checkpoint
    from vfp_dit.generate_samples import build_model, read_checkpoint, sample_latents

    torch.manual_seed(41)
    model = NoVFCBDiT(
        qwen_dim=20, latent_channels=4, width=24, depth=1,
        heads=2, kv_heads=1, adapter_depth=1, ff_mult=1.5,
    ).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Initialize optimizer state so the resume sidecar exercises state restore too.
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    config = {
        "condition_dim": 20,
        "latent_channels": 4,
        "model_width": 24,
        "depth": 1,
        "heads": 2,
        "kv_heads": 1,
        "adapter_depth": 1,
        "ff_mult": 1.5,
        "adapter_type": "ffn",
        "fuse_same_input_projections": True,
        "attention_head_gate": "input_silu",
        "metadata_ffn_residual_gate": False,
        "condition_layer": "final",
        "resolution": 32,
    }
    checkpoint = tmp_path / "checkpoint_latest.safetensors"
    save_model_checkpoint(
        model,
        checkpoint,
        metadata={"stage": "vfp_dit.train"},
        optimizer=optimizer,
        epoch=3,
        global_step=17,
        config=config,
    )

    network_metadata, saved_config = read_checkpoint(checkpoint)
    assert network_metadata["stage"] == "vfp_dit.train"
    assert saved_config == config
    restored = build_model(saved_config, torch.device("cpu"))
    restored.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    with pytest.raises(ValueError, match="only GatedFFN"):
        build_model({**saved_config, "adapter_type": "transformer"}, torch.device("cpu"))
    resume_args = SimpleNamespace(
        **config,
        resume=str(checkpoint),
        init_checkpoint=None,
    )
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    epoch, global_step = restore_training(
        resume_args,
        restored,
        restored_optimizer,
        torch.device("cpu"),
        expected_stage="vfp_dit.train",
    )
    assert (epoch, global_step) == (3, 17)
    assert len(restored_optimizer.state) == len(optimizer.state)

    hidden = torch.randn(1, 4, 20)
    mask = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.zeros(1, 4, 3)
    positions[0, :, 0] = torch.arange(4)
    conditional_cache = restored.prepare_condition(hidden, mask, positions)
    unconditional_cache = restored.prepare_condition(
        torch.zeros(1, 1, 20),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 3),
    )
    output = sample_latents(
        restored,
        conditional_cache,
        latent_height=2,
        latent_width=3,
        steps=2,
        device=torch.device("cpu"),
        guidance_scale=2.0,
        unconditional_cache=unconditional_cache,
        generator=torch.Generator().manual_seed(9),
    )
    reference_output = sample_latents(
        model,
        model.prepare_condition(hidden, mask, positions),
        latent_height=2,
        latent_width=3,
        steps=2,
        device=torch.device("cpu"),
        guidance_scale=2.0,
        unconditional_cache=model.prepare_condition(
            torch.zeros(1, 1, 20),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, 3),
        ),
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.equal(output, reference_output)


def test_sampler_end_to_end_for_t2i_and_ti2i_on_cpu(monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    from vfp_dit import generate_samples
    generate_one = generate_samples.generate_one

    class Posterior:
        def mode(self):
            return torch.zeros(1, 16, 1, 4, 4)

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
            self.encode_calls = 0

        def encode(self, image_video):
            self.encode_calls += 1
            assert image_video.shape == (1, 3, 1, 32, 32)
            return SimpleNamespace(latent_dist=Posterior())

        def decode(self, latents):
            assert latents.shape == (1, 16, 1, 4, 4)
            return SimpleNamespace(sample=torch.zeros(1, 3, 1, 32, 32))

    class TinyEncoder:
        def encode_condition(self, prompt, *, source_image, condition_layer, include_positions, include_vision_mask=False):
            assert prompt == "edit the scene"
            assert condition_layer == 1 and include_positions
            token_count = 5 if source_image is not None else 3
            hidden = torch.randn(token_count, 20)
            mask = torch.ones(token_count, dtype=torch.bool)
            positions = torch.zeros(token_count, 3)
            positions[:, 0] = torch.arange(token_count)
            if source_image is not None:
                positions[-2:, 0] = 1
                positions[-2:, 2] = torch.tensor([0, 1])
            vision_mask = torch.zeros(token_count, dtype=torch.bool)
            if source_image is not None:
                vision_mask[-2:] = True
            if include_vision_mask:
                return hidden, mask, positions, vision_mask
            return hidden, mask, positions

    model = NoVFCBDiT(
        qwen_dim=20, latent_channels=16, width=24, depth=1, heads=2,
        kv_heads=1, adapter_depth=1, ff_mult=1.5,
    ).eval()
    vae = TinyVAE()
    encoder = TinyEncoder()
    generator = torch.Generator().manual_seed(12)
    sampler_arguments = {}
    original_sample_latents = generate_samples.sample_latents

    def capture_sampler_arguments(*args, **kwargs):
        sampler_arguments.update(kwargs)
        return original_sample_latents(*args, **kwargs)

    monkeypatch.setattr(generate_samples, "sample_latents", capture_sampler_arguments)

    text_only = generate_one(
        "edit the scene", None, model=model, vae=vae, encoder=encoder,
        condition_layer=1, resolution=32, steps=2, guidance_scale=1.0,
        device=torch.device("cpu"), generator=generator,
    )
    source = Image.new("RGB", (24, 40), color="blue")
    edited = generate_one(
        "edit the scene", source, model=model, vae=vae, encoder=encoder,
        condition_layer=1, resolution=32, steps=2, guidance_scale=2.0,
        device=torch.device("cpu"), generator=torch.Generator().manual_seed(13),
        rf_er_sde_warp_trust_lambda=2.5,
        rf_er_sde_warp_trust_error_c=0.25,
        rf_er_sde_warp_trust_epsilon=1e-7,
    )

    assert text_only.shape == edited.shape == (3, 32, 32)
    assert torch.isfinite(text_only).all() and torch.isfinite(edited).all()
    assert vae.encode_calls == 1
    assert sampler_arguments["rf_er_sde_warp_trust_lambda"] == 2.5
    assert sampler_arguments["rf_er_sde_warp_trust_error_c"] == 0.25
    assert sampler_arguments["rf_er_sde_warp_trust_epsilon"] == 1e-7
