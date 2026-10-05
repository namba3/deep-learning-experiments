import pytest
import torch

from core.low_rank import iter_adapter_modules
from verify.image_ae_adapter_dataset_comparison import (
    _build_model,
    _make_initial_state,
    parse_args,
    resolve_dtype,
)


def test_image_ae_adapter_probe_parser_and_model_contract():
    args = parse_args([
        "--seeds", "2,3",
        "--adapters", "lora,rglu_lora",
        "--rank", "2",
        "--alpha", "4",
    ])
    assert args.seeds == (2, 3)
    assert args.adapters == ("lora", "rglu_lora")
    assert args.alpha == 4.0

    dtype = resolve_dtype("fp32", torch.device("cpu"))
    initial_state = _make_initial_state(args, seed=0, dtype=dtype)
    model, trainable, matched = _build_model(
        args, "rglu_lora", initial_state,
        torch.device("cpu"), dtype,
    )
    assert trainable > 0
    assert len(matched) == 18
    assert sum(1 for _ in iter_adapter_modules(model)) == len(matched)
    output, latent = model(torch.rand(1, 3, 32, 32))
    assert output.shape == (1, 3, 32, 32)
    assert latent.shape == (1, args.latent_channels, 4, 4)


def test_image_ae_adapter_probe_rejects_warm_mixed_adapters():
    with pytest.raises(SystemExit):
        parse_args([
            "--adapter-init", "lora_warm",
            "--adapters", "lora,rglu_lora",
        ])
