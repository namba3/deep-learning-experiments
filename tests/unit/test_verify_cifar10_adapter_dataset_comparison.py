from verify.cifar10_adapter_dataset_comparison import parse_args


def test_dataset_adapter_comparison_parser_defaults_to_three_seeds():
    args = parse_args([])

    assert args.seeds == (0, 1, 2)
    assert args.adapters == (
        "lora", "loha", "dora", "glu_lora", "rglu_lora",
    )
    assert args.max_train_samples == 512
    assert args.max_validation_samples == 256
    assert args.adapter_init == "identity"


def test_dataset_adapter_comparison_parses_per_adapter_budget_maps():
    args = parse_args([
        "--adapters", "lora,loha,rglu_lora",
        "--rank-map", "lora=16,loha=8,rglu_lora=8",
        "--alpha-map", "lora=16,loha=8,rglu_lora=8",
    ])

    assert args.rank_map == {"lora": 16, "loha": 8, "rglu_lora": 8}
    assert args.alpha_map == {"lora": 16.0, "loha": 8.0, "rglu_lora": 8.0}
