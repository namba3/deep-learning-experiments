from verify.cifar10_rglu_lora_sweep import parse_args


def test_rglu_lora_sweep_parser_defaults_to_rank_alpha_tied():
    args = parse_args(["--output", "output.json"])

    assert args.ranks == (1, 2, 4, 8)
    assert args.alphas is None
    assert args.adapter_init == "identity"
