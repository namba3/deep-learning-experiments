import json
from typing import Any

import pytest
import pyarrow as pa
import torch
from datasets import Dataset

from text_lm.adapter_training import (
    DEFAULT_ADAPTER_TARGETS,
    enable_adapter,
)
from text_lm.train import (
    ALL_ARCHITECTURE_CHOICES,
    ARCHITECTURE_CHOICES,
    DEFAULT_ARCHITECTURE,
    LEGACY_ALPACA_CONFIG_METADATA_KEY,
    TEXT_LM_CONFIG_METADATA_KEY,
    TinyTextLM,
    build_optimizer,
    compute_causal_lm_loss,
    limit_dataset,
    load_instruction_dataset,
    compute_distillation_loss,
    pack_text_datasets,
    read_text_lm_checkpoint_config,
    _load_cached_arrow_dataset,
    _compute_distillation_components,
    perplexity_from_loss,
    main,
)


def test_chunked_causal_lm_loss_matches_full_logits_forward_and_backward():
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7)
    weight = torch.randn(11, 7)
    labels = torch.tensor([
        [3, 1, -100, 4, 2],
        [0, 2, 5, -100, 1],
    ])
    full_hidden = hidden.clone().requires_grad_()
    full_weight = weight.clone().requires_grad_()
    full_loss = compute_causal_lm_loss(
        full_hidden, full_weight, labels, vocab_chunk_size=0,
    )
    full_loss.backward()

    chunked_hidden = hidden.clone().requires_grad_()
    chunked_weight = weight.clone().requires_grad_()
    chunked_loss = compute_causal_lm_loss(
        chunked_hidden, chunked_weight, labels, vocab_chunk_size=3,
    )
    chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, full_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        chunked_hidden.grad, full_hidden.grad, atol=1e-5, rtol=1e-5,
    )
    torch.testing.assert_close(
        chunked_weight.grad, full_weight.grad, atol=1e-5, rtol=1e-5,
    )


def test_shared_architectures_are_archived_not_active_cli_choices():
    assert DEFAULT_ARCHITECTURE == "naive"
    assert ARCHITECTURE_CHOICES == (
        "naive", "mhla3-gqa", "looped", "looped-hybrid",
        "mhla3-gqa-looped-hybrid",
    )
    assert set(ALL_ARCHITECTURE_CHOICES) == {
        *ARCHITECTURE_CHOICES, "shared-fixed", "shared-variable",
    }
    with pytest.raises(SystemExit):
        main(["--architecture", "shared-fixed"])


@pytest.mark.parametrize(
    "architecture",
    [
        "naive", "shared-fixed", "shared-variable", "mhla3-gqa", "looped",
        "looped-hybrid", "mhla3-gqa-looped-hybrid",
    ],
)
def test_text_lm_architectures_are_causal_and_use_rope(architecture):
    num_layers = 4 if architecture == "mhla3-gqa" else 2
    kwargs = {}
    if architecture == "looped-hybrid":
        num_layers = 4
        kwargs.update(
            looped_prefix_layers=1,
            looped_blocks=1,
            looped_repeats=2,
            looped_suffix_layers=1,
        )
    elif architecture == "mhla3-gqa-looped-hybrid":
        num_layers = 4
        kwargs.update(
            mhla_looped_prefix_cycles=0,
            mhla_looped_repeats=1,
            mhla_looped_suffix_cycles=0,
        )
    model = TinyTextLM(
        vocab_size=31,
        max_seq_len=8,
        embed_dim=16,
        num_layers=num_layers,
        num_heads=4,
        kv_heads=2,
        condition_dim=4,
        transform_rank=2,
        architecture=architecture,
        **kwargs,
    ).eval()
    assert not hasattr(model, "position_embedding")

    prefix = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 9, 10]])
    forward_kwargs = {"depth": 1} if architecture == "shared-variable" else {}
    prefix_logits = model(prefix, **forward_kwargs)
    changed_logits = model(changed_future, **forward_kwargs)

    assert prefix_logits.shape == (1, 5, 31)
    assert torch.isfinite(prefix_logits).all()
    torch.testing.assert_close(
        prefix_logits[:, :3], changed_logits[:, :3],
        atol=1e-5, rtol=1e-5,
    )

    prefix_logits[:, :-1].float().mean().backward()


def test_text_lm_shared_and_hybrid_depth_layouts():
    shared = TinyTextLM(
        vocab_size=17, max_seq_len=6, embed_dim=16, num_layers=3,
        num_heads=4, kv_heads=2, condition_dim=4, transform_rank=1,
        architecture="shared-fixed",
    )
    assert shared.decoder.layers[0].attn is shared.decoder.layers[1].attn
    assert shared.decoder.layers[0].ffn_in is shared.decoder.layers[1].ffn_in

    hybrid = TinyTextLM(
        vocab_size=17, max_seq_len=6, embed_dim=16, num_layers=4,
        num_heads=4, kv_heads=2, condition_dim=4, transform_rank=1,
        architecture="mhla3-gqa",
    )
    assert hybrid.decoder.num_layers == 4
    assert [layer.__class__.__name__ for layer in hybrid.decoder.layers] == [
        "MHLATransformerBlock", "MHLATransformerBlock",
        "MHLATransformerBlock", "GatedGQATransformerBlock",
    ]


def test_text_lm_looped_decoder_uses_physical_block_stack():
    model = TinyTextLM(
        vocab_size=17, max_seq_len=6, embed_dim=16, num_layers=8,
        num_heads=4, kv_heads=2, architecture="looped", looped_blocks=2,
    )

    assert model.decoder.num_layers == 8
    assert model.decoder.looped_blocks == 2
    assert model.decoder.num_loops == 4
    assert len(model.decoder.layers) == 2
    assert model.decoder.layers[0].residual_scale == pytest.approx(0.5)


def test_text_lm_hybrid_looped_decoder_layout():
    model = TinyTextLM(
        vocab_size=17, max_seq_len=6, embed_dim=16, num_layers=8,
        num_heads=4, kv_heads=2, architecture="looped-hybrid",
        looped_prefix_layers=2, looped_blocks=1, looped_repeats=4,
        looped_suffix_layers=2,
    )

    decoder = model.decoder
    assert decoder.num_layers == 8
    assert decoder.prefix_layers_count == 2
    assert decoder.looped_blocks == 1
    assert decoder.num_loops == 4
    assert decoder.suffix_layers_count == 2
    assert len(decoder.prefix_layers) == 2
    assert len(decoder.looped_layers) == 1
    assert len(decoder.suffix_layers) == 2
    assert decoder.looped_layers[0].residual_scale == pytest.approx(0.5)


def test_text_lm_mhla_looped_hybrid_decoder_layout():
    model = TinyTextLM(
        vocab_size=17, max_seq_len=6, embed_dim=16, num_layers=16,
        num_heads=4, kv_heads=2, architecture="mhla3-gqa-looped-hybrid",
        mhla_looped_prefix_cycles=1, mhla_looped_repeats=2,
        mhla_looped_suffix_cycles=1,
    )

    decoder = model.decoder
    assert decoder.num_layers == 16
    assert decoder.num_cycles == 4
    assert decoder.prefix_cycles == 1
    assert decoder.looped_repeats == 2
    assert decoder.suffix_cycles == 1
    assert len(decoder.prefix_layers) == 4
    assert len(decoder.looped_layers) == 4
    assert len(decoder.suffix_layers) == 4
    assert [layer.__class__.__name__ for layer in decoder.looped_layers] == [
        "MHLATransformerBlock", "MHLATransformerBlock",
        "MHLATransformerBlock", "GatedGQATransformerBlock",
    ]
    assert decoder.looped_layers[0].residual_scale == pytest.approx(2 ** -0.5)


def test_text_lm_mhla_looped_hybrid_strict_state_dict_round_trip():
    model_kwargs: dict[str, Any] = dict(
        vocab_size=17,
        max_seq_len=6,
        embed_dim=16,
        num_layers=8,
        num_heads=4,
        kv_heads=2,
        architecture="mhla3-gqa-looped-hybrid",
        mhla_looped_prefix_cycles=0,
        mhla_looped_repeats=2,
        mhla_looped_suffix_cycles=0,
    )
    torch.manual_seed(123)
    source = TinyTextLM(**model_kwargs).eval()
    torch.manual_seed(456)
    restored = TinyTextLM(**model_kwargs).eval()

    restored.load_state_dict(source.state_dict(), strict=True)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        source_logits = source(input_ids)
        restored_logits = restored(input_ids)
    torch.testing.assert_close(source_logits, restored_logits)


def test_text_lm_adapter_targets_cover_naive_attention_and_ffn():
    model = TinyTextLM(
        vocab_size=31, max_seq_len=8, embed_dim=16, num_layers=2,
        num_heads=4, kv_heads=2, architecture="naive",
    )
    matched, trainable = enable_adapter(
        model,
        type(
            "AdapterArgs", (), {
                "adapter": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "lora_dropout": 0.0,
                "lora_target": None,
                "adapter_init": "identity",
            },
        )(),
    )

    assert trainable > 0
    assert len(matched) == 10
    assert any(name.endswith("attn.k_proj") for name in matched)
    assert any(name.endswith("attn.v_proj") for name in matched)
    assert any(name.endswith("ffn.2") for name in matched)
    assert DEFAULT_ADAPTER_TARGETS[0].endswith("out_proj)$")


def test_text_lm_hybrid_num_layers_is_total_block_count():
    with pytest.raises(ValueError, match="multiple of 4"):
        TinyTextLM(
            vocab_size=17,
            max_seq_len=6,
            embed_dim=16,
            num_layers=3,
            num_heads=4,
            kv_heads=2,
            architecture="mhla3-gqa",
        )

    with pytest.raises(ValueError, match="kv_heads"):
        TinyTextLM(
            vocab_size=17,
            max_seq_len=6,
            embed_dim=16,
            num_layers=4,
            num_heads=4,
            kv_heads=3,
            architecture="mhla3-gqa-looped-hybrid",
            mhla_looped_prefix_cycles=0,
            mhla_looped_repeats=1,
            mhla_looped_suffix_cycles=0,
        )


def test_text_lm_arrow_dataset_path_splits_without_cache_writes(tmp_path):
    arrow_path = tmp_path / "alpaca-train.arrow"
    table = Dataset.from_dict({
        "instruction": [f"instruction-{i}" for i in range(10)],
        "input": [""] * 10,
        "output": [f"output-{i}" for i in range(10)],
        "text": [f"text-{i}" for i in range(10)],
    }).data.table
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    train, evaluation = load_instruction_dataset(
        "unused/dataset", str(arrow_path), eval_ratio=0.2, seed=42,
    )

    assert len(train) == 8
    assert len(evaluation) == 2
    assert train.column_names == ["instruction", "input", "output", "text"]


def test_text_lm_dataset_limits_are_deterministic_and_keep_zero_unbounded():
    dataset = Dataset.from_dict({"value": list(range(5))})

    assert len(limit_dataset(dataset, 0)) == 5
    assert limit_dataset(dataset, 3)["value"] == [0, 1, 2]
    assert limit_dataset(dataset, 8) is dataset


def test_text_lm_checkpoint_config_reads_legacy_metadata(monkeypatch):
    calls = []

    def fake_read_checkpoint_config(path, key):
        calls.append((path, key))
        if key == LEGACY_ALPACA_CONFIG_METADATA_KEY:
            return {"architecture": "naive"}
        return None

    monkeypatch.setattr(
        "text_lm.train.read_checkpoint_config", fake_read_checkpoint_config,
    )

    assert read_text_lm_checkpoint_config("legacy.safetensors") == {
        "architecture": "naive",
    }
    assert calls == [
        ("legacy.safetensors", TEXT_LM_CONFIG_METADATA_KEY),
        ("legacy.safetensors", LEGACY_ALPACA_CONFIG_METADATA_KEY),
    ]


def test_text_lm_reads_cached_arrow_shards_without_writing_locks(tmp_path, monkeypatch):
    cache_root = tmp_path / "datasets"
    dataset_root = cache_root / "org-name___tiny_stories" / "default" / "0.0.0" / "hash"
    dataset_root.mkdir(parents=True)
    for index, values in enumerate((["first"], ["second"])):
        table = pa.table({"text": values})
        arrow_path = dataset_root / f"tiny_stories-train-0000{index}-of-00002.arrow"
        with pa.OSFile(str(arrow_path), "wb") as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
    (dataset_root / "dataset_info.json").write_text(
        json.dumps({
            "config_name": "default",
            "splits": {"train": {"shard_lengths": [1, 1]}},
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_root))

    dataset = _load_cached_arrow_dataset("org-name/TinyStories", None, "train")

    assert dataset["text"] == ["first", "second"]


def test_text_lm_text_data_is_packed_and_split_into_disjoint_blocks():
    class FakeTokenizer:
        eos_token_id = 99

        def __call__(self, text, **_kwargs):
            return {"input_ids": [ord(char) for char in text]}

    train, evaluation = pack_text_datasets(
        [
            {"text": "abcdefgh"},
            {"text": "ijklmnop"},
            {"text": "qrstuvwx"},
            {"text": "yzabcdef"},
        ],
        FakeTokenizer(),
        text_column="text",
        max_seq_len=4,
        train_tokens=8,
        eval_tokens=8,
    )

    assert len(train) == 2
    assert len(evaluation) == 2
    assert all(len(row) == 4 for row in train["input_ids"])
    assert all(len(row) == 4 for row in evaluation["input_ids"])
    assert train["input_ids"][-1] != evaluation["input_ids"][0]
    assert evaluation["attention_mask"] == [[1, 1, 1, 1]] * 2


def test_text_lm_distillation_loss_handles_teacher_extra_vocab():
    torch.manual_seed(7)
    student_logits = torch.randn(2, 5, 5, requires_grad=True)
    teacher_logits = torch.randn(2, 5, 7)
    labels = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, -100],
    ])

    loss, hard_loss, soft_loss = compute_distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=2.0,
        alpha=0.5,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(hard_loss)
    assert torch.isfinite(soft_loss)
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_text_lm_metrics_separate_raw_kl_from_temperature_scaled_loss():
    torch.manual_seed(11)
    student_logits = torch.randn(1, 4, 5, requires_grad=True)
    teacher_logits = torch.randn(1, 4, 7)
    labels = torch.tensor([[0, 1, 2, 3]])

    total, hard, soft, kl = _compute_distillation_components(
        student_logits, teacher_logits, labels,
        temperature=2.0, alpha=0.25,
    )

    assert torch.isfinite(total)
    assert torch.isfinite(hard)
    assert torch.isfinite(soft)
    assert torch.isfinite(kl)
    assert soft.item() == pytest.approx(kl.item() * 4.0, rel=1e-6)
    assert perplexity_from_loss(torch.log(torch.tensor(10.0)).item()) == pytest.approx(10.0)


@pytest.mark.parametrize("name", ["APOLLO", "APOLLO-CAME"])
def test_text_lm_low_rank_optimizer_branches_construct_and_step(name):
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = build_optimizer(
        name, [parameter], lr=0.01, weight_decay=0.0,
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()
