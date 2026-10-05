import torch
from vfp_dit.model import NoVFCBDiT


def test_original_qwen_image_vae_latent_encoding_contract():
    from types import SimpleNamespace

    from vfp_dit.encoders import encode_qwen_image_latents

    class Posterior:
        def mode(self):
            return torch.full((2, 16, 1, 2, 3), 2.0)

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                latents_mean=[1.0] * 16,
                latents_std=[2.0] * 16,
            )
            self.input_shape = None

        def encode(self, value):
            self.input_shape = tuple(value.shape)
            return SimpleNamespace(latent_dist=Posterior())

    vae = TinyVAE()
    latent = encode_qwen_image_latents(vae, torch.zeros(2, 3, 16, 24))

    assert vae.input_shape == (2, 3, 1, 16, 24)
    assert latent.shape == (2, 16, 2, 3)
    assert torch.equal(latent, torch.full_like(latent, 0.5))


def test_single_stage_training_step_encodes_mixed_t2i_ti2i_batch_and_backprops():
    from types import SimpleNamespace

    from PIL import Image

    from vfp_dit.train import _flow_step

    class Posterior:
        def __init__(self, value):
            self.value = value

        def mode(self):
            return self.value

        def sample(self):
            return self.value

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                latents_mean=[0.0] * 16,
                latents_std=[1.0] * 16,
            )

        def encode(self, image_video):
            batch = image_video.shape[0]
            latent = image_video.new_zeros(batch, 16, 1, 2, 2)
            return SimpleNamespace(latent_dist=Posterior(latent))

    class TinyEncoder:
        def encode_condition(self, prompt, *, source_image, condition_layer, include_positions, include_vision_mask=False):
            count = 5 if source_image is not None else 3
            hidden = torch.randn(count, 20)
            mask = torch.ones(count, dtype=torch.bool)
            positions = torch.zeros(count, 3)
            positions[:, 0] = torch.arange(count)
            if source_image is not None:
                positions[-2:, 0] = 1
                positions[-2:, 2] = torch.tensor([0, 1])
            assert condition_layer == "final"
            assert include_positions
            vision_mask = torch.zeros(count, dtype=torch.bool)
            if source_image is not None:
                vision_mask[-2:] = True
            if include_vision_mask:
                return hidden, mask, positions, vision_mask
            return hidden, mask, positions

    model = NoVFCBDiT(
        qwen_dim=20,
        latent_channels=16,
        width=24,
        depth=1,
        heads=2,
        kv_heads=1,
        adapter_depth=1,
        ff_mult=1.5,
    )
    image_a = Image.new("RGB", (16, 16), (80, 100, 120))
    image_b = Image.new("RGB", (16, 16), (150, 120, 90))
    batch = {
        "target_images": [image_a, image_b],
        "source_images": [None, image_a],
        "prompts": ["draw a landscape", "change the sky"],
    }
    args = SimpleNamespace(
        condition_layer="final",
        resolution=16,
        vae_latent_mode="mode",
        condition_dropout=0.0,
    )

    loss, metrics = _flow_step(
        model, batch, args, torch.device("cpu"),
        vae=TinyVAE(), encoder=TinyEncoder(),
    )

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["condition_drop_fraction"].item() == 0.0
    loss.backward()
    assert model.adapter.reference_latent_in.weight.grad is not None
    assert model.blocks[0].condition_kv_proj.weight.grad is not None
    assert model.output.weight.grad is not None


def test_condition_dropout_replaces_all_modalities_with_valid_null_token(monkeypatch):
    from vfp_dit.train import _drop_conditions

    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    positions = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    reference = torch.ones(2, 2, 2, 2)
    reference_mask = torch.tensor([
        [True, True, True, True],
        [False, False, False, False],
    ])
    encoded = {
        "qwen_hidden": hidden,
        "qwen_mask": mask,
        "qwen_positions": positions,
        "reference_latent": reference,
        "reference_mask": reference_mask,
    }
    monkeypatch.setattr(
        torch, "rand",
        lambda size, *, device: torch.tensor([0.1, 0.9], device=device),
    )

    (
        dropped_hidden,
        dropped_mask,
        dropped_positions,
        dropped_reference,
        dropped_reference_mask,
        fraction,
    ) = _drop_conditions(encoded, 0.5)

    assert dropped_reference is not None
    assert dropped_reference_mask is not None
    assert torch.equal(dropped_hidden[0], torch.zeros_like(hidden[0]))
    assert dropped_mask[0].tolist() == [True, False, False]
    assert torch.equal(dropped_positions[0], torch.zeros_like(positions[0]))
    assert torch.equal(dropped_reference[0], torch.zeros_like(reference[0]))
    assert not dropped_reference_mask[0].any()
    assert torch.equal(dropped_hidden[1], hidden[1])
    assert torch.equal(dropped_mask[1], mask[1])
    assert torch.equal(dropped_positions[1], positions[1])
    assert torch.equal(dropped_reference[1], reference[1])
    assert torch.equal(dropped_reference_mask[1], reference_mask[1])
    assert fraction.item() == 0.5


def test_qwen35_encoder_returns_multimodal_positions_on_request():
    from types import SimpleNamespace

    from PIL import Image

    from vfp_dit_runtime.qwen35 import Qwen35FeatureEncoder

    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            includes_image = any(
                item.get("type") == "image"
                for item in messages[0]["content"]
            )
            input_ids = torch.tensor([[1, 99, 99, 99, 99, 2]]) if includes_image else torch.tensor([[1, 2]])
            mm_types = torch.zeros_like(input_ids, dtype=torch.int32)
            if includes_image:
                mm_types[:, 1:5] = 1
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
                "mm_token_type_ids": mm_types,
                "image_grid_thw": torch.tensor([[1, 4, 4]]) if includes_image else None,
            }

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.norm = torch.nn.Identity()

    class TinyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()
            self.config = SimpleNamespace(
                image_token_id=99,
                vision_config=SimpleNamespace(spatial_merge_size=2),
            )

        def forward(self, input_ids, **_kwargs):
            hidden = torch.nn.functional.one_hot(input_ids.remainder(4), num_classes=4).float()
            self.language_model.layers[0](hidden)
            return SimpleNamespace()

    encoder = Qwen35FeatureEncoder(
        model=TinyQwen(),
        processor=Processor(),
        device=torch.device("cpu"),
        layer_count=1,
        condition_dim=4,
        teacher_dim=4,
    )
    from typing import cast

    hidden, mask, positions = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], encoder.encode_condition(
        "edit the image",
        source_image=Image.new("RGB", (16, 16)),
        condition_layer=1,
        include_positions=True,
    ))

    assert hidden.shape == (6, 4)
    assert mask.shape == (6,)
    assert positions.shape == (6, 3)
    assert positions[1:5].tolist() == [
        [1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],
    ]
