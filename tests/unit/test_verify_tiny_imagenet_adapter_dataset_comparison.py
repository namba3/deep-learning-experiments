from torch import nn

from verify.tiny_imagenet_adapter_dataset_comparison import (
    _enable_classifier_head,
    parse_args,
)


def test_tiny_imagenet_adapter_comparison_parser_defaults():
    args = parse_args([])

    assert args.dataset_name == "zh-plus/tiny-imagenet"
    assert args.cache_dir is None
    assert args.train_classifier_head is False
    assert args.seeds == (0, 1, 2)
    assert args.adapters == (
        "lora", "loha", "dora", "glu_lora", "rglu_lora",
    )
    assert args.epochs == 3
    assert args.batch_size == 32
    assert args.max_train_samples == 1024
    assert args.max_validation_samples == 512


def test_tiny_imagenet_adapter_comparison_parses_matched_budget():
    args = parse_args([
        "--rank-map", "lora=32,loha=16,dora=32,glu_lora=16,rglu_lora=16",
        "--alpha-map", "lora=32,loha=16,dora=32,glu_lora=16,rglu_lora=16",
        "--adapters", "lora,glu_lora,rglu_lora",
    ])

    assert args.rank_map == {
        "lora": 32, "loha": 16, "dora": 32,
        "glu_lora": 16, "rglu_lora": 16,
    }
    assert args.alpha_map == {
        "lora": 32.0, "loha": 16.0, "dora": 32.0,
        "glu_lora": 16.0, "rglu_lora": 16.0,
    }
    assert args.adapters == ("lora", "glu_lora", "rglu_lora")


def test_tiny_imagenet_adapter_comparison_can_train_common_head():
    args = parse_args(["--train-classifier-head"])

    assert args.train_classifier_head is True


def test_enable_classifier_head_keeps_backbone_frozen():
    model = nn.Module()
    model.head = nn.Linear(3, 2)
    model.backbone = nn.Linear(3, 3)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable = _enable_classifier_head(model)

    assert trainable == 8
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
