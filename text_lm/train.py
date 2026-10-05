"""Configurable small decoder-only Transformer for text pretraining and SFT."""

import argparse
import errno
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.layers import (
    CausalTransformerDecoder,
    HybridMHLA3GQADecoder,
    HybridMHLA3GQALoopedDecoder,
    HybridLoopedTransformerDecoder,
    LoopedTransformerDecoder,
    SharedDepthTransformerDecoder,
)
from runtime.progress import RichProgress
from runtime.data import build_dataloader_options
from runtime.metrics import build_standard_progress_rows, write_standard_training_metrics
from runtime.memory import maybe_collect_memory
from runtime.preflight import build_training_preflight
from runtime.validation import ValidationTimer, build_validation_report
from runtime.run import RunRecorder
from runtime.signal import GracefulStop
from runtime.sampler import ResumableRandomSampler
from runtime.checkpoint import (
    load_training_state,
    make_training_state,
    restore_rng_state,
    save_training_state,
)
from runtime.config import (
    apply_saved_config,
    checkpoint_config_metadata,
    cli_option_provided,
    read_checkpoint_config,
)
from runtime.device import add_device_argument, resolve_device
from optimizers.factory import (
    add_optimizer_argument,
    build_optimizer,
    is_schedule_free_optimizer,
)
from optimizers.lr_scheduler import add_lr_scheduler_arguments, build_lr_scheduler
from core.utils import (
    DepthDistributionScheduler,
    compact_state_dict,
    convert_linear_to_bf16,
    convert_rmsnorm_to_dtype_aware,
    format_bytes,
    print_model_info,
)
from text_lm.adapter_training import (  # noqa: E402
    add_adapter_arguments,
    enable_adapter,
    optimizer_parameters as adapter_optimizer_parameters,
    resolve_adapter_config,
)

# image_gen/train.pyと同じQwen3.5 tokenizerを使い、画像・テキスト実験間で
# token idと語彙を揃える。必要に応じて --tokenizer で変更できる。
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B"
DEFAULT_ARCHITECTURE = "naive"
DEFAULT_DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_DATASET_MODE = "text"
DATASET_MODE_CHOICES = ("instruction", "text")
ARCHITECTURE_CHOICES = (
    "naive",
    "mhla3-gqa",
    "looped",
    "looped-hybrid",
    "mhla3-gqa-looped-hybrid",
)
# Retained only for loading old checkpoints and direct internal API tests.
ARCHIVED_ARCHITECTURE_CHOICES = ("shared-fixed", "shared-variable")
ALL_ARCHITECTURE_CHOICES = ARCHITECTURE_CHOICES + ARCHIVED_ARCHITECTURE_CHOICES
ROPE_BASE = 10000.0

# ========= モデル・学習スーパーパラメータ =========
# 下記の初期値は、各active architectureで約1Bパラメータを目標にした設定。
MAX_SEQ_LEN = 512
EMBED_DIM = 2048
NUM_LAYERS = 16
NUM_HEADS = 32
NUM_KV_HEADS = 8
LOOPED_BLOCKS = 1
LOOPED_HYBRID_BLOCKS = 2
LOOPED_HYBRID_PREFIX_LAYERS = 4
LOOPED_HYBRID_REPEATS = 4
LOOPED_HYBRID_SUFFIX_LAYERS = 4
MHLA_LOOPED_PREFIX_CYCLES = 1
MHLA_LOOPED_REPEATS = 2
MHLA_LOOPED_SUFFIX_CYCLES = 1
CONDITION_DIM = 64
TRANSFORM_RANK = 10
DEFAULT_VOCAB_CHUNK_SIZE = 8192
EMBEDDING_DROPOUT = 0.0

BATCH_SIZE = 1
# 1epochの処理バッチ数を抑え、その分epochを増やす設定。
# Noneにすると、従来通りtrain_loader全体を1epochで処理する。
STEPS_PER_EPOCH = 1000
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
EVAL_RATIO = 0.02
# 評価は学習より高コストになりやすいため、全データ・全depthで毎回評価しない。
EVAL_INTERVAL = 5
EVAL_MAX_BATCHES = 100
SEED = 42
OUTPUT_DIR = "text_lm/output"
CHECKPOINT_RETENTION = 3
MIN_DEPTH = 1
DEPTH_MAX_BIAS = 4.0
DEPTH_INITIAL_BIAS = 1.0
TEXT_LM_CONFIG_METADATA_KEY = "text_lm.config"
LEGACY_ALPACA_CONFIG_METADATA_KEY = "alpaca.config"
TEXT_TRAIN_TOKENS = 10_000_000
TEXT_EVAL_TOKENS = 1_000_000
DEFAULT_DISTILL_MODE = "none"
DISTILL_MODE_CHOICES = ("none", "logits")
DISTILL_TEMPERATURE = 2.0
DISTILL_ALPHA = 0.5


def resolve_resume_path(path, output_dir):
    if os.path.isfile(path):
        return path
    candidate = os.path.join(output_dir, path)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Resume checkpoint not found: {path}")


def read_text_lm_checkpoint_config(path):
    """Read current text-LM metadata, with support for old Alpaca checkpoints."""
    config = read_checkpoint_config(path, TEXT_LM_CONFIG_METADATA_KEY)
    if config is not None:
        return config
    return read_checkpoint_config(path, LEGACY_ALPACA_CONFIG_METADATA_KEY)

def format_instruction_example(example, tokenizer):
    instruction = str(example.get("instruction", "")).strip()
    user_input = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()
    user_text = instruction
    if user_input:
        user_text += "\n\n" + user_input

    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": output},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return (
        "### Instruction:\n"
        + user_text
        + "\n\n### Response:\n"
        + output
    )

class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

class TinyTextLM(nn.Module):
    supports_chunked_causal_loss = True

    def __init__(self, vocab_size, max_seq_len=MAX_SEQ_LEN,
                 embed_dim=EMBED_DIM, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, condition_dim=CONDITION_DIM,
                 transform_rank=TRANSFORM_RANK,
                 dropout=EMBEDDING_DROPOUT, compute_dtype=None,
                 architecture=DEFAULT_ARCHITECTURE,
                 kv_heads=NUM_KV_HEADS, rope_base=ROPE_BASE,
                 looped_blocks=None,
                 looped_prefix_layers=LOOPED_HYBRID_PREFIX_LAYERS,
                 looped_repeats=LOOPED_HYBRID_REPEATS,
                 looped_suffix_layers=LOOPED_HYBRID_SUFFIX_LAYERS,
                 mhla_looped_prefix_cycles=MHLA_LOOPED_PREFIX_CYCLES,
                 mhla_looped_repeats=MHLA_LOOPED_REPEATS,
                 mhla_looped_suffix_cycles=MHLA_LOOPED_SUFFIX_CYCLES):
        super().__init__()
        if looped_blocks is None:
            looped_blocks = (
                LOOPED_HYBRID_BLOCKS
                if architecture == "looped-hybrid" else LOOPED_BLOCKS
            )
        if architecture not in ALL_ARCHITECTURE_CHOICES:
            raise ValueError(
                f"architecture must be one of {ALL_ARCHITECTURE_CHOICES}, "
                f"got {architecture!r}"
            )
        if architecture == "mhla3-gqa" and num_layers % 4 != 0:
            raise ValueError(
                "num_layers must be a positive multiple of 4 for mhla3-gqa"
            )
        if architecture == "looped" and (
            not 1 <= looped_blocks <= num_layers
            or num_layers % looped_blocks != 0
        ):
            raise ValueError(
                "num_layers must be divisible by looped_blocks for looped"
            )
        if architecture == "looped-hybrid" and (
            looped_prefix_layers < 0
            or looped_suffix_layers < 0
            or looped_blocks < 1
            or looped_repeats < 1
            or (
                looped_prefix_layers
                + looped_blocks * looped_repeats
                + looped_suffix_layers
                != num_layers
            )
        ):
            raise ValueError(
                "num_layers must equal prefix + looped_blocks * repeats + suffix "
                "for looped-hybrid"
            )
        if architecture == "mhla3-gqa-looped-hybrid" and (
            mhla_looped_prefix_cycles < 0
            or mhla_looped_suffix_cycles < 0
            or mhla_looped_repeats < 1
            or 4 * (
                mhla_looped_prefix_cycles
                + mhla_looped_repeats
                + mhla_looped_suffix_cycles
            )
            != num_layers
        ):
            raise ValueError(
                "num_layers must equal 4 * (prefix_cycles + repeats + "
                "suffix_cycles) for mhla3-gqa-looped-hybrid"
            )
        if (
            architecture in {"mhla3-gqa", "mhla3-gqa-looped-hybrid"}
            and (not 1 <= kv_heads <= num_heads or num_heads % kv_heads != 0)
        ):
            raise ValueError(
                "kv_heads must be positive, no greater than num_heads, "
                "and divide num_heads"
            )
        self.max_seq_len = max_seq_len
        self.architecture = architecture
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.looped_blocks = looped_blocks
        self.looped_prefix_layers = looped_prefix_layers
        self.looped_repeats = looped_repeats
        self.looped_suffix_layers = looped_suffix_layers
        self.mhla_looped_prefix_cycles = mhla_looped_prefix_cycles
        self.mhla_looped_repeats = mhla_looped_repeats
        self.mhla_looped_suffix_cycles = mhla_looped_suffix_cycles
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        # すべてのAttention variantがQ/Kへ1D RoPEを適用するため、
        # 学習可能なposition embeddingは持たない。
        if architecture == "naive":
            self.decoder = CausalTransformerDecoder(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                drop_out=dropout,
                rope_base=rope_base,
            )
        elif architecture in {"shared-fixed", "shared-variable"}:
            self.decoder = SharedDepthTransformerDecoder(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                condition_dim=condition_dim,
                transform_rank=transform_rank,
                drop_out=dropout,
                compute_dtype=compute_dtype,
                rope_base=rope_base,
            )
        elif architecture == "looped":
            self.decoder = LoopedTransformerDecoder(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                looped_blocks=looped_blocks,
                drop_out=dropout,
                rope_base=rope_base,
            )
        elif architecture == "looped-hybrid":
            self.decoder = HybridLoopedTransformerDecoder(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                prefix_layers=looped_prefix_layers,
                looped_blocks=looped_blocks,
                looped_repeats=looped_repeats,
                suffix_layers=looped_suffix_layers,
                drop_out=dropout,
                rope_base=rope_base,
            )
        elif architecture == "mhla3-gqa-looped-hybrid":
            self.decoder = HybridMHLA3GQALoopedDecoder(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                kv_heads=kv_heads,
                prefix_cycles=mhla_looped_prefix_cycles,
                looped_repeats=mhla_looped_repeats,
                suffix_cycles=mhla_looped_suffix_cycles,
                drop_out=dropout,
                rope_base=rope_base,
            )
        else:
            self.decoder = HybridMHLA3GQADecoder(
                num_cycles=num_layers // 4,
                embed_dim=embed_dim,
                num_heads=num_heads,
                kv_heads=kv_heads,
                drop_out=dropout,
                rope_base=rope_base,
            )

    def forward(
        self, input_ids, attention_mask=None, depth=None, return_hidden=False,
    ):
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("input sequence is longer than max_seq_len")
        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)
        if self.architecture in {"shared-fixed", "shared-variable"}:
            x = self.decoder(x, depth=depth, attention_mask=attention_mask)
        else:
            if depth is not None:
                raise ValueError(
                    "depth is only supported by shared-depth architectures"
                )
            x = self.decoder(x, attention_mask=attention_mask)
        if return_hidden:
            return x
        # Token embeddingと重み共有。出力logitsはFP32で計算する。
        return F.linear(x.float(), self.token_embedding.weight.float())

def _load_cached_arrow_dataset(dataset_name, dataset_config, split):
    """Load finished HF Arrow shards without creating cache lock files."""
    cache_root = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            os.path.expanduser("~/.cache/huggingface/datasets"),
        )
    )
    # datasets' cache layout keeps hyphens in the namespace, e.g.
    # ``tatsu-lab___alpaca``.  Keep that canonical spelling first, while the
    # fallback below also accepts older/custom cache naming conventions.
    cache_name = dataset_name.replace("/", "___").lower()
    dataset_root = cache_root / cache_name
    if not dataset_root.is_dir() and cache_root.is_dir():
        normalized_name = re.sub(r"[_-]", "", cache_name)
        matching_roots = [
            path for path in cache_root.iterdir()
            if path.is_dir()
            and re.sub(r"[_-]", "", path.name.lower()) == normalized_name
        ]
        if matching_roots:
            dataset_root = matching_roots[0]
    candidates = []
    if dataset_root.is_dir():
        for info_path in dataset_root.rglob("dataset_info.json"):
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                dataset_config is not None
                and info.get("config_name") != dataset_config
            ):
                continue
            split_info = info.get("splits", {}).get(split)
            if not isinstance(split_info, dict):
                continue
            expected_shards = len(split_info.get("shard_lengths", ()))
            arrow_paths = sorted(info_path.parent.glob("*.arrow"))
            split_marker = f"-{split.lower()}"
            selected = [
                path for path in arrow_paths
                if path.stem.lower().startswith(split_marker)
                or split_marker in path.stem.lower()
                or path.stem.lower().endswith(f"_{split.lower()}")
            ]
            if not selected and len(info.get("splits", {})) == 1:
                selected = arrow_paths
            if expected_shards and len(selected) != expected_shards:
                continue
            if selected:
                candidates.append(
                    (info_path.stat().st_mtime, tuple(selected))
                )
    if not candidates:
        raise FileNotFoundError(
            f"no cached Arrow files found for {dataset_name!r} split={split!r}"
        )
    _, paths = max(candidates, key=lambda item: item[0])
    datasets = [Dataset.from_file(str(path)) for path in paths]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def load_instruction_dataset(dataset_name, dataset_path, eval_ratio, seed):
    if dataset_path:
        if dataset_path.endswith(".arrow"):
            dataset = Dataset.from_file(dataset_path)
        elif dataset_path.endswith((".json", ".jsonl")):
            dataset = load_dataset(
                "json", data_files=dataset_path, split="train"
            )
        else:
            dataset = load_dataset(dataset_path, split="train")
    else:
        try:
            dataset = load_dataset(dataset_name, split="train")
        except OSError as error:
            if error.errno != errno.EROFS:
                raise
            # A read-only shared HF cache can contain the finished Arrow
            # artifact while datasets still fails when creating its lock file.
            # Reuse that artifact directly; Dataset.from_file is read-only.
            try:
                dataset = _load_cached_arrow_dataset(
                    dataset_name, None, "train"
                )
            except FileNotFoundError:
                raise error
    # Reproduce datasets.train_test_split's random partition while keeping the
    # split indices in memory.  The default HF cache may be readable but not
    # writable, and some datasets versions still derive index-cache filenames
    # even when keep_in_memory=True.
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be between 0 and 1, got {eval_ratio}")
    test_size = int(np.ceil(eval_ratio * len(dataset)))
    permutation = np.random.default_rng(seed).permutation(len(dataset))
    test_indices = permutation[:test_size]
    train_indices = permutation[test_size:]
    train_dataset = dataset.select(train_indices, keep_in_memory=True)
    eval_dataset = dataset.select(test_indices, keep_in_memory=True)
    return train_dataset, eval_dataset


def limit_dataset(dataset, max_examples):
    if max_examples <= 0 or len(dataset) <= max_examples:
        return dataset
    return dataset.select(range(max_examples), keep_in_memory=True)


def _make_packed_dataset(chunks, max_seq_len):
    if not chunks:
        return Dataset.from_dict({
            "input_ids": [],
            "attention_mask": [],
        })
    return Dataset.from_dict({
        "input_ids": chunks,
        "attention_mask": [[1] * max_seq_len for _ in chunks],
    })


def pack_text_datasets(
    examples,
    tokenizer,
    text_column,
    max_seq_len,
    train_tokens,
    eval_tokens,
):
    """Tokenize a text stream once and split it into fixed-length LM blocks.

    Complete blocks are assigned to training first and the following complete
    blocks to evaluation.  The document that fills the training budget is not
    reused for evaluation, keeping the two sets document-disjoint while
    allowing a streaming HF dataset to be used without materializing raw
    documents.  The requested budgets are rounded down to whole blocks.
    """
    if max_seq_len <= 1:
        raise ValueError("max_seq_len must be greater than 1")
    if train_tokens <= 0 or eval_tokens <= 0:
        raise ValueError("text train/eval token budgets must both be > 0")
    train_blocks = train_tokens // max_seq_len
    eval_blocks = eval_tokens // max_seq_len
    if train_blocks < 1 or eval_blocks < 1:
        raise ValueError(
            "text train/eval token budgets must each contain at least one "
            "max_seq_len block"
        )

    train_chunks = []
    eval_chunks = []
    buffer = []
    buffer_start = 0
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("text datasets require a tokenizer with eos_token_id")

    phase = "train"
    for example in examples:
        text = example.get(text_column, "")
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = encoded.get("input_ids", [])
        if not input_ids:
            continue
        buffer.extend(input_ids)
        buffer.append(eos_token_id)
        while len(buffer) - buffer_start >= max_seq_len:
            chunk = buffer[buffer_start:buffer_start + max_seq_len]
            buffer_start += max_seq_len
            if phase == "train":
                train_chunks.append(chunk)
                if len(train_chunks) == train_blocks:
                    # Do not let the tail of this document leak into eval.
                    buffer = []
                    buffer_start = 0
                    phase = "eval"
                    break
            else:
                eval_chunks.append(chunk)
                if len(eval_chunks) == eval_blocks:
                    return (
                        _make_packed_dataset(train_chunks, max_seq_len),
                        _make_packed_dataset(eval_chunks, max_seq_len),
                    )
            if buffer_start >= 1024 * max_seq_len:
                buffer = buffer[buffer_start:]
                buffer_start = 0

    if len(train_chunks) < train_blocks or len(eval_chunks) < eval_blocks:
        available = len(train_chunks) + len(eval_chunks)
        required = train_blocks + eval_blocks
        raise ValueError(
            f"text dataset yielded only {available} complete blocks; "
            f"{required} required for train_tokens={train_tokens} and "
            f"eval_tokens={eval_tokens}"
        )
    return (
        _make_packed_dataset(train_chunks, max_seq_len),
        _make_packed_dataset(eval_chunks, max_seq_len),
    )


def load_text_datasets(
    dataset_name,
    dataset_config,
    dataset_path,
    text_column,
    tokenizer,
    max_seq_len,
    train_tokens,
    eval_tokens,
    split="train",
):
    """Load and pack a text dataset, streaming HF rows when possible."""
    if dataset_path:
        if dataset_path.endswith(".arrow"):
            dataset = Dataset.from_file(dataset_path)
        elif dataset_path.endswith((".json", ".jsonl")):
            dataset = load_dataset(
                "json",
                data_files=dataset_path,
                split="train",
                streaming=True,
            )
        else:
            dataset = load_dataset(
                dataset_path,
                split=split,
                streaming=True,
            )
    else:
        load_kwargs = {"split": split, "streaming": True}
        if dataset_config:
            load_kwargs["name"] = dataset_config
        try:
            dataset = load_dataset(dataset_name, **load_kwargs)
        except OSError as error:
            if error.errno != errno.EROFS:
                raise
            try:
                dataset = _load_cached_arrow_dataset(
                    dataset_name, dataset_config, split
                )
            except FileNotFoundError:
                raise error
    return pack_text_datasets(
        dataset,
        tokenizer,
        text_column,
        max_seq_len,
        train_tokens,
        eval_tokens,
    )


def _compute_distillation_components(
    student_logits,
    teacher_logits,
    labels,
    temperature,
    alpha,
):
    """Combine hard next-token CE with teacher soft-target KL divergence.

    The Qwen3.5 teacher vocabulary can contain reserved multimodal IDs that
    are not exposed by the tokenizer length used by the student.  Distillation
    therefore uses the common prefix and renormalizes that teacher support.
    """
    if temperature <= 0:
        raise ValueError("distillation temperature must be > 0")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("distillation alpha must be between 0 and 1")
    student_shifted = student_logits[:, :-1].float()
    teacher_shifted = teacher_logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    hard_loss = F.cross_entropy(
        student_shifted.contiguous().view(-1, student_shifted.size(-1)),
        shifted_labels.contiguous().view(-1),
        ignore_index=-100,
    )

    common_vocab = student_shifted.size(-1)
    if teacher_shifted.size(-1) < common_vocab:
        raise ValueError(
            "teacher vocabulary is smaller than the student vocabulary: "
            f"teacher={teacher_shifted.size(-1)} student={common_vocab}"
        )
    valid = shifted_labels != -100
    if not valid.any():
        raise ValueError("distillation batch contains no valid target tokens")
    student_log_probs = F.log_softmax(
        student_shifted / temperature, dim=-1
    )
    teacher_probs = F.softmax(
        teacher_shifted[..., :common_vocab] / temperature, dim=-1
    )
    kl_divergence = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)
    kl_divergence = kl_divergence.masked_select(valid).mean()
    soft_loss = kl_divergence * temperature ** 2
    total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
    return (
        total_loss,
        hard_loss.detach(),
        soft_loss.detach(),
        kl_divergence.detach(),
    )


def compute_distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature,
    alpha,
):
    """Return the blended loss and its legacy hard/soft components."""
    total_loss, hard_loss, soft_loss, _ = _compute_distillation_components(
        student_logits, teacher_logits, labels, temperature, alpha,
    )
    return total_loss, hard_loss, soft_loss


def perplexity_from_loss(loss: float) -> float:
    """Convert mean token cross-entropy to a finite-or-infinite PPL value."""
    try:
        return math.exp(float(loss))
    except OverflowError:
        return float("inf")


class _ChunkedCausalCrossEntropy(torch.autograd.Function):
    """Causal CE that avoids materializing the complete vocabulary logits.

    The forward pass keeps only one vocabulary chunk at a time.  The backward
    pass recomputes each chunk and applies the softmax derivative directly, so
    autograd does not retain all chunk logits until backward.  The hidden state
    and tied embedding weight are both inputs because the latter receives the
    output-projection gradient in addition to its input-embedding gradient.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_size):
        if hidden.ndim != 3 or labels.ndim != 2:
            raise ValueError("hidden must be [B,T,D] and labels must be [B,T]")
        if hidden.shape[:2] != labels.shape:
            raise ValueError(
                "hidden and labels must have matching batch and sequence axes"
            )
        if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
            raise ValueError("tied vocabulary weight has incompatible shape")
        if chunk_size <= 0:
            raise ValueError("vocabulary chunk size must be positive")

        shifted_hidden = hidden[:, :-1, :].float().reshape(-1, hidden.shape[-1])
        shifted_labels = labels[:, 1:].reshape(-1)
        valid = shifted_labels != -100
        valid_indices = valid.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError("causal LM batch contains no valid target tokens")
        hidden_valid = shifted_hidden.index_select(0, valid_indices)
        targets = shifted_labels.index_select(0, valid_indices)
        vocab_size = weight.shape[0]
        if targets.min() < 0 or targets.max() >= vocab_size:
            raise ValueError("causal LM target is outside the vocabulary")

        log_z = torch.full(
            (targets.numel(),), float("-inf"),
            dtype=torch.float32, device=hidden.device,
        )
        target_logits = torch.empty_like(log_z)
        target_seen = torch.zeros_like(targets, dtype=torch.bool)
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            chunk_logits = F.linear(
                hidden_valid, weight[start:end].float(),
            )
            log_z = torch.logaddexp(
                log_z, torch.logsumexp(chunk_logits, dim=-1),
            )
            target_mask = (targets >= start) & (targets < end)
            target_rows = target_mask.nonzero(as_tuple=False).flatten()
            if target_rows.numel() > 0:
                target_logits[target_rows] = chunk_logits[
                    target_rows, targets[target_rows] - start,
                ]
                target_seen[target_rows] = True
        if not bool(target_seen.all()):
            raise RuntimeError("failed to compute a target vocabulary logit")

        ctx.save_for_backward(
            hidden, weight, valid_indices, targets, log_z,
        )
        ctx.chunk_size = int(chunk_size)
        return (log_z - target_logits).mean()

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, valid_indices, targets, log_z = ctx.saved_tensors
        shifted_hidden = hidden[:, :-1, :].float().reshape(-1, hidden.shape[-1])
        hidden_valid = shifted_hidden.index_select(0, valid_indices)
        grad_hidden_valid = torch.zeros_like(hidden_valid)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        scale = grad_output.float() / targets.numel()
        vocab_size = weight.shape[0]

        for start in range(0, vocab_size, ctx.chunk_size):
            end = min(start + ctx.chunk_size, vocab_size)
            weight_chunk = weight[start:end].float()
            chunk_logits = F.linear(hidden_valid, weight_chunk)
            probabilities = torch.exp(chunk_logits - log_z[:, None])
            probabilities.mul_(scale)
            target_mask = (targets >= start) & (targets < end)
            target_rows = target_mask.nonzero(as_tuple=False).flatten()
            if target_rows.numel() > 0:
                probabilities[
                    target_rows, targets[target_rows] - start,
                ] -= scale
            grad_hidden_valid.add_(F.linear(probabilities, weight_chunk.t()))
            grad_weight[start:end].add_(probabilities.transpose(0, 1) @ hidden_valid)

        grad_shifted = torch.zeros(
            (hidden.shape[0], hidden.shape[1] - 1, hidden.shape[2]),
            dtype=torch.float32, device=hidden.device,
        )
        grad_shifted.reshape(-1, hidden.shape[-1]).index_copy_(
            0, valid_indices, grad_hidden_valid,
        )
        grad_hidden_float = torch.zeros(
            hidden.shape, dtype=torch.float32, device=hidden.device,
        )
        grad_hidden_float[:, :-1, :] = grad_shifted
        return (
            grad_hidden_float.to(dtype=hidden.dtype),
            grad_weight.to(dtype=weight.dtype),
            None,
            None,
        )


def compute_causal_lm_loss(
    hidden, weight, labels, *, vocab_chunk_size=DEFAULT_VOCAB_CHUNK_SIZE,
):
    """Compute tied-weight causal CE with optional vocabulary chunking.

    ``vocab_chunk_size <= 0`` selects the reference full-logits path.  The
    chunked path uses FP32 projection and accumulation, matching the existing
    numerical policy while reducing peak temporary memory.
    """
    if vocab_chunk_size <= 0:
        logits = F.linear(hidden.float(), weight.float())
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
    return _ChunkedCausalCrossEntropy.apply(
        hidden, weight, labels, int(vocab_chunk_size),
    )


def main(argv=None, *, adapter_only=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-mode", choices=DATASET_MODE_CHOICES,
        default=DEFAULT_DATASET_MODE,
        help=(
            "instruction=Alpaca形式をSFT、text=テキストを固定長packして"
            "事前学習。既定値はTinyStoriesのtext。"
        ),
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-config", default=None,
        help="Hugging Face dataset configuration name（必要な場合のみ）",
    )
    parser.add_argument("--dataset-path", default=None,
                        help="ローカルJSON/JSONL/Arrowを使う場合のパス")
    parser.add_argument(
        "--dataset-split", default="train",
        help="text modeで読むHugging Face split（既定値: train）",
    )
    parser.add_argument(
        "--text-column", default="text",
        help="text modeで読むテキスト列（既定値: text）",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="Qwen系TokenizerのHugging Face IDまたはローカルパス")
    parser.add_argument(
        "--distill-mode", choices=DISTILL_MODE_CHOICES,
        default=DEFAULT_DISTILL_MODE,
        help=(
            "none=通常のcausal LM、logits=teacherのsoft logitsを併用。"
            "teacherは--teacher-modelで指定"
        ),
    )
    parser.add_argument(
        "--teacher-model", default=DEFAULT_TOKENIZER,
        help="logits distillationに使うHugging Face causal LM",
    )
    parser.add_argument(
        "--distill-temperature", type=float, default=DISTILL_TEMPERATURE,
        help="soft targetのtemperature（既定値: 2.0）",
    )
    parser.add_argument(
        "--distill-alpha", type=float, default=DISTILL_ALPHA,
        help="hard CEの重み。0でteacherのみ、1で通常CE（既定値: 0.5）",
    )
    parser.add_argument(
        "--architecture", choices=ARCHITECTURE_CHOICES,
        default=DEFAULT_ARCHITECTURE,
        help=(
            "Active decoder architecture: naive (independent MHA), "
            "mhla3-gqa (MHLA x3 + "
            "Gated-GQA x1 repeated), or looped (a physical block stack "
            "repeated across depth), or looped-hybrid (independent "
            "prefix/suffix around a repeated middle stack), or "
            "mhla3-gqa-looped-hybrid (independent MHLA3+GQA cycles around "
            "a repeated middle cycle)."
        ),
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--run-name", default=None,
        help="Optional human-readable name included in the run directory.",
    )
    add_device_argument(parser)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="共通設定を検証し、データやモデルを読み込まずに終了",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="データとモデルの構成を検証し、学習せずに終了",
    )
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument(
        "--looped-blocks", type=int, default=None,
        help="looped/looped-hybridで保存する物理block数（構成別既定値あり）",
    )
    parser.add_argument(
        "--looped-prefix-layers", type=int,
        default=LOOPED_HYBRID_PREFIX_LAYERS,
        help="looped-hybridの前段に置く独立block数",
    )
    parser.add_argument(
        "--looped-repeats", type=int, default=LOOPED_HYBRID_REPEATS,
        help="looped-hybridの中央stack反復回数",
    )
    parser.add_argument(
        "--looped-suffix-layers", type=int,
        default=LOOPED_HYBRID_SUFFIX_LAYERS,
        help="looped-hybridの後段に置く独立block数",
    )
    parser.add_argument(
        "--mhla-looped-prefix-cycles", type=int,
        default=MHLA_LOOPED_PREFIX_CYCLES,
        help="mhla3-gqa-looped-hybridの前段cycle数",
    )
    parser.add_argument(
        "--mhla-looped-repeats", type=int, default=MHLA_LOOPED_REPEATS,
        help="mhla3-gqa-looped-hybridの中央cycle反復回数",
    )
    parser.add_argument(
        "--mhla-looped-suffix-cycles", type=int,
        default=MHLA_LOOPED_SUFFIX_CYCLES,
        help="mhla3-gqa-looped-hybridの後段cycle数",
    )
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument(
        "--kv-heads", type=int, default=NUM_KV_HEADS,
        help="Number of K/V heads for GQA and MHLA architectures.",
    )
    parser.add_argument("--condition-dim", type=int, default=CONDITION_DIM)
    parser.add_argument("--transform-rank", type=int, default=TRANSFORM_RANK)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--vocab-chunk-size", type=int, default=DEFAULT_VOCAB_CHUNK_SIZE,
        help=(
            "causal CEの語彙chunk幅。正の値でlogitsを分割し、0以下で"
            "従来の全vocabulary logits経路を使う"
        ),
    )
    parser.add_argument(
        "--max-train-examples", type=int, default=0,
        help="学習に使う最大example数。0以下で全件。",
    )
    parser.add_argument(
        "--max-eval-examples", type=int, default=0,
        help="評価に使う最大example数。0以下で全件。",
    )
    parser.add_argument(
        "--max-train-tokens", type=int, default=TEXT_TRAIN_TOKENS,
        help=(
            "text modeの学習token budget。max_seq_len単位に切り捨てる。"
        ),
    )
    parser.add_argument(
        "--max-eval-tokens", type=int, default=TEXT_EVAL_TOKENS,
        help=(
            "text modeの評価token budget。max_seq_len単位に切り捨てる。"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoaderのワーカープロセス数。デフォルト: 4")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument(
        "--steps-per-epoch", type=int, default=STEPS_PER_EPOCH,
        help="1epochで処理する最大バッチ数。0以下で全バッチを処理する",
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument(
        "--eval-interval", type=int, default=EVAL_INTERVAL,
        help="min/mid/max depth別評価を行う間隔。最終epochでは必ず評価する",
    )
    parser.add_argument(
        "--eval-max-batches", type=int, default=EVAL_MAX_BATCHES,
        help="評価に使う最大バッチ数。0以下で全評価データを使う",
    )
    parser.add_argument(
        "--save-mode", choices=("final", "interval", "epoch"),
        default="epoch",
        help="モデル保存頻度。final=最後のみ、interval=指定間隔、epoch=毎epoch",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1,
        help="save-mode=interval時の保存間隔（epoch数）",
    )
    parser.add_argument(
        "--checkpoint-retention", type=int, default=CHECKPOINT_RETENTION,
        help="保持する途中checkpointの世代数（デフォルト: 3）",
    )
    parser.add_argument(
        "--resume", default=None,
        help=(
            "Load a text LM checkpoint. If a sibling .resume.pt exists, also "
            "restore optimizer, scheduler, depth scheduler, epoch, and RNG state; "
            "otherwise use weights-only resume."
        ),
    )
    parser.add_argument(
        "--init-checkpoint", default=None,
        help=(
            "Initialize model weights only from a checkpoint; optimizer, "
            "scheduler, depth, epoch, and RNG state are not restored."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dataset-seed", type=int, default=None,
        help="Seed for the train/eval split. Defaults to --seed.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps", type=int, default=0,
        help="Save a resumable latest checkpoint every N optimizer steps; 0 disables it.",
    )
    add_optimizer_argument(parser, default="AdamW")
    add_lr_scheduler_arguments(
        parser, default="cosine", include_force_scheduler=True,
    )
    if adapter_only:
        add_adapter_arguments(parser)
    # Keep these fields parseable so legacy checkpoint metadata can be
    # restored, but do not advertise the archived shared-depth interface.
    parser.add_argument("--variable-depth", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--min-depth", type=int, default=MIN_DEPTH,
                        help=argparse.SUPPRESS)
    parser.add_argument("--depth-max-bias", type=float, default=DEPTH_MAX_BIAS,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--bf16", action="store_true",
        help=(
            "trainable parameterをBF16で保存する。Linear出力と語彙"
            "projection/lossは数値安定性のためFP32（デフォルトは全FP32）"
        ),
    )
    parser.add_argument("--gc-interval", type=int, default=100,
                        help="NバッチごとにPython GCを実行。0で無効")
    parser.add_argument("--empty-cache-interval", type=int, default=0,
                        help="NバッチごとにCUDAキャッシュを解放。0で無効")
    args = parser.parse_args(argv)
    if not adapter_only:
        # Keep checkpoint metadata fields stable without exposing adapter
        # options from the ordinary full-model training entrypoint.
        args.lora_base_checkpoint = None
        args.lora_rank = 0
        args.adapter = "none"
        args.lora_alpha = None
        args.lora_dropout = 0.0
        args.lora_target = None
        args.adapter_init = "identity"
    cli_argv = sys.argv[1:] if argv is None else argv
    script_name = "text_lm.train_adapter" if adapter_only else "text_lm.train"
    if args.dry_run and args.validate_only:
        raise ValueError("--dry-run and --validate-only cannot be used together")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    resume_path = resolve_resume_path(args.resume, args.output_dir) if args.resume else None
    init_path = resolve_resume_path(args.init_checkpoint, args.output_dir) if args.init_checkpoint else None
    lora_base_path = None
    if args.lora_base_checkpoint:
        lora_base_path = args.lora_base_checkpoint
        if not os.path.isfile(lora_base_path):
            candidate = os.path.join(args.output_dir, lora_base_path)
            if os.path.isfile(candidate):
                lora_base_path = candidate
        if not os.path.isfile(lora_base_path):
            raise FileNotFoundError(
                f"LoRA base checkpoint not found: {args.lora_base_checkpoint}"
            )
    if resume_path:
        resume_config = read_text_lm_checkpoint_config(resume_path)
        if (
            not adapter_only
            and resume_config
            and resume_config.get("adapter", "none") != "none"
        ):
            raise ValueError(
                "adapter checkpoint must be resumed with text_lm.train_adapter"
            )
        overridden_config_keys = []
        restored_config_keys = apply_saved_config(
            args,
            resume_config,
            {"auto_schedule": ("--auto-schedule", "--no-auto-schedule")},
            argv=cli_argv,
            keys=tuple(
                key for key in vars(args)
                if key not in {
                    "resume", "init_checkpoint", "output_dir", "run_name", "device", "dry_run", "validate_only", "dataset_path",
                }
            ),
            overridden_keys=overridden_config_keys,
        )
        if restored_config_keys:
            print(
                "Restored settings from checkpoint: "
                + ", ".join(restored_config_keys)
            )
        if overridden_config_keys:
            print(
                "CLI overrides checkpoint settings: "
                + ", ".join(overridden_config_keys)
            )
        if (
            resume_config is not None
            and "data_mode" not in resume_config
            and not cli_option_provided(cli_argv, "--data-mode")
        ):
            # Checkpoints created before text pretraining was introduced were
            # always instruction-format runs. Do not silently switch their
            # data path to the new TinyStories default on resume.
            args.data_mode = "instruction"
            print(
                "Legacy checkpoint has no data_mode; using instruction mode."
            )
    adapter_type = (
        resolve_adapter_config(
            args, resume_path=resume_path, base_path=lora_base_path,
        )
        if adapter_only
        else "none"
    )
    if args.looped_blocks is None:
        args.looped_blocks = (
            LOOPED_HYBRID_BLOCKS
            if args.architecture == "looped-hybrid" else LOOPED_BLOCKS
        )
    uses_shared_depth = args.architecture in {
        "shared-fixed", "shared-variable",
    }
    if args.variable_depth:
        if args.architecture == "shared-fixed":
            args.architecture = "shared-variable"
        elif args.architecture != "shared-variable":
            raise ValueError(
                "--variable-depth is only compatible with shared-depth "
                "architectures"
            )
    args.variable_depth = args.architecture == "shared-variable"
    if args.warmup_steps is None and args.warmup_ratio is None:
        # Preserve the historical 10% warmup when no scheduler warmup flag
        # was supplied explicitly.
        args.warmup_ratio = 0.1
    elif args.warmup_steps is None:
        args.warmup_steps = 0
    elif args.warmup_ratio is None:
        args.warmup_ratio = 0.0

    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be > 0")
    if args.save_interval <= 0:
        raise ValueError("--save-interval must be > 0")
    if args.checkpoint_retention <= 0:
        raise ValueError("--checkpoint-retention must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.max_train_examples < 0 or args.max_eval_examples < 0:
        raise ValueError("--max-*-examples must be >= 0")
    if (
        args.data_mode == "text"
        and (args.max_train_tokens <= 0 or args.max_eval_tokens <= 0)
    ):
        raise ValueError("--max-*-tokens must be > 0")
    if args.distill_temperature <= 0:
        raise ValueError("--distill-temperature must be > 0")
    if not 0.0 <= args.distill_alpha <= 1.0:
        raise ValueError("--distill-alpha must be between 0 and 1")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0")
    if args.dataset_seed is not None and args.dataset_seed < 0:
        raise ValueError("--dataset-seed must be >= 0")
    if args.checkpoint_interval_steps < 0:
        raise ValueError("--checkpoint-interval-steps must be >= 0")
    if args.num_layers < 1:
        raise ValueError("--num-layers must be >= 1")
    if args.architecture == "looped":
        if not 1 <= args.looped_blocks <= args.num_layers:
            raise ValueError("--looped-blocks must be in [1, --num-layers]")
        if args.num_layers % args.looped_blocks != 0:
            raise ValueError(
                "--num-layers must be divisible by --looped-blocks for looped"
            )
    if args.architecture == "looped-hybrid":
        if args.looped_prefix_layers < 0 or args.looped_suffix_layers < 0:
            raise ValueError(
                "--looped-prefix-layers and --looped-suffix-layers must be >= 0"
            )
        if args.looped_blocks < 1 or args.looped_repeats < 1:
            raise ValueError(
                "--looped-blocks and --looped-repeats must be >= 1"
            )
        expected_layers = (
            args.looped_prefix_layers
            + args.looped_blocks * args.looped_repeats
            + args.looped_suffix_layers
        )
        if expected_layers != args.num_layers:
            raise ValueError(
                "--num-layers must equal prefix + looped-blocks * repeats + "
                "suffix for looped-hybrid"
            )
    if args.architecture in {"mhla3-gqa", "mhla3-gqa-looped-hybrid"}:
        if args.num_layers % 4 != 0:
            raise ValueError("--num-layers must be divisible by 4 for mhla3-gqa")
        if not 1 <= args.kv_heads <= args.num_heads:
            raise ValueError("--kv-heads must be in [1, --num-heads]")
        if args.num_heads % args.kv_heads != 0:
            raise ValueError("--kv-heads must divide --num-heads")
    if args.architecture == "mhla3-gqa-looped-hybrid":
        if (
            args.mhla_looped_prefix_cycles < 0
            or args.mhla_looped_suffix_cycles < 0
            or args.mhla_looped_repeats < 1
        ):
            raise ValueError(
                "MHLA looped prefix/suffix cycles must be >= 0 and repeats >= 1"
            )
        expected_layers = 4 * (
            args.mhla_looped_prefix_cycles
            + args.mhla_looped_repeats
            + args.mhla_looped_suffix_cycles
        )
        if expected_layers != args.num_layers:
            raise ValueError(
                "--num-layers must equal 4 * (prefix-cycles + repeats + "
                "suffix-cycles) for mhla3-gqa-looped-hybrid"
            )
    if (
        args.architecture in {"shared-fixed", "shared-variable"}
        and not 1 <= args.min_depth <= args.num_layers
    ):
        raise ValueError(
            f"--min-depth must be in [1, {args.num_layers}]"
        )

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    run_recorder = RunRecorder(
        args.output_dir,
        script=script_name,
        config=vars(args),
        run_name=args.run_name,
    )
    run_recorder.install_exception_hook()
    preflight = build_training_preflight(
        script=script_name,
        device=device,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        extra={
            "optimizer": args.optimizer,
            "distill_mode": args.distill_mode,
            "teacher_model": args.teacher_model,
        },
    )
    run_recorder.record("preflight", **preflight)
    if args.dry_run:
        run_recorder.finish(status="dry_run")
        print("Dry run completed; no dataset or model was loaded.")
        return
    validation_timer = ValidationTimer(device) if args.validate_only else None
    checkpoint_dir = str(run_recorder.checkpoints_dir)
    artifact_dir = str(run_recorder.artifacts_dir)
    tensorboard_dir = str(run_recorder.tensorboard_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model = None
    if args.distill_mode == "logits":
        teacher_dtype = torch.bfloat16 if args.bf16 else torch.float32
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            trust_remote_code=True,
            dtype=teacher_dtype,
        ).to(device).eval()
        teacher_model.requires_grad_(False)
        teacher_config = teacher_model.config
        teacher_vocab_size = getattr(teacher_config, "vocab_size", None)
        if teacher_vocab_size is None and hasattr(teacher_config, "text_config"):
            teacher_vocab_size = getattr(
                teacher_config.text_config, "vocab_size", None
            )
        if (
            teacher_vocab_size is not None
            and teacher_vocab_size < len(tokenizer)
        ):
            raise ValueError(
                "teacher vocabulary is smaller than the tokenizer vocabulary: "
                f"teacher={teacher_vocab_size} tokenizer={len(tokenizer)}"
            )
        print(
            f"Loaded distillation teacher: {args.teacher_model} "
            f"(dtype={teacher_dtype}, vocab={teacher_vocab_size or 'unknown'})"
        )

    dataset_seed = args.seed if args.dataset_seed is None else args.dataset_seed
    if args.data_mode == "text":
        train_dataset, eval_dataset = load_text_datasets(
            args.dataset_name,
            args.dataset_config,
            args.dataset_path,
            args.text_column,
            tokenizer,
            args.max_seq_len,
            args.max_train_tokens,
            args.max_eval_tokens,
            split=args.dataset_split,
        )
    else:
        train_dataset, eval_dataset = load_instruction_dataset(
            args.dataset_name, args.dataset_path, args.eval_ratio, dataset_seed
        )
    train_dataset = limit_dataset(train_dataset, args.max_train_examples)
    eval_dataset = limit_dataset(eval_dataset, args.max_eval_examples)

    if args.data_mode == "instruction":
        def tokenize(example):
            text = format_instruction_example(example, tokenizer)
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_seq_len,
            )
            if len(encoded["input_ids"]) == args.max_seq_len:
                encoded["input_ids"][-1] = tokenizer.eos_token_id
            return encoded

        def tokenize_dataset(dataset):
            map_kwargs = {"remove_columns": dataset.column_names}
            if isinstance(dataset, Dataset):
                map_kwargs["keep_in_memory"] = True
            return dataset.map(tokenize, **map_kwargs)

        train_dataset = tokenize_dataset(train_dataset)
        eval_dataset = tokenize_dataset(eval_dataset)
    collator = CausalCollator(tokenizer)
    train_sampler = ResumableRandomSampler(train_dataset, seed=args.seed)
    train_loader_options = build_dataloader_options(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        seed=args.seed,
        stream=0,
    )
    eval_loader_options = build_dataloader_options(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        seed=args.seed,
        stream=1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collator,
        **train_loader_options,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        **eval_loader_options,
    )

    model = TinyTextLM(
        vocab_size=len(tokenizer),
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kv_heads=args.kv_heads,
        condition_dim=args.condition_dim,
        transform_rank=args.transform_rank,
        dropout=EMBEDDING_DROPOUT,
        compute_dtype=torch.bfloat16 if args.bf16 else None,
        architecture=args.architecture,
        looped_blocks=args.looped_blocks,
        looped_prefix_layers=args.looped_prefix_layers,
        looped_repeats=args.looped_repeats,
        looped_suffix_layers=args.looped_suffix_layers,
        mhla_looped_prefix_cycles=args.mhla_looped_prefix_cycles,
        mhla_looped_repeats=args.mhla_looped_repeats,
        mhla_looped_suffix_cycles=args.mhla_looped_suffix_cycles,
    ).to(
        device=device,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    if args.bf16:
        # 全trainable parameterをBF16 storageへ揃える。BF16LinearとLM headの
        # 出力は数値安定性のためFP32へ戻すが、重み自体はFP32へ戻さない。
        convert_rmsnorm_to_dtype_aware(model.decoder)
        depth_embedding = getattr(model.decoder, "depth_embedding", None)
        convert_linear_to_bf16(
            model.decoder,
            skip_modules=()
            if depth_embedding is None else (depth_embedding,),
        )
    model_depth = getattr(
        getattr(model, "decoder", None), "num_layers", args.num_layers,
    )
    resume_state = None
    if lora_base_path:
        print(f"Loading LoRA base weights: {lora_base_path}")
        base_weights = load_file(lora_base_path, device="cpu")
        load_result = model.load_state_dict(base_weights, strict=False)
        if load_result.unexpected_keys or load_result.missing_keys:
            raise ValueError(
                "Text LM base checkpoint does not match the current model: "
                f"missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )
        del base_weights
    if adapter_type != "none" and resume_path:
        matched_adapter_targets, adapter_trainable_count = enable_adapter(
            model, args,
        )
        print(
            f"Enabled {adapter_type}: rank={args.lora_rank} "
            f"alpha={args.lora_alpha or args.lora_rank:g} "
            f"targets={len(matched_adapter_targets)} "
            f"trainable_parameters={adapter_trainable_count:,}"
        )
    if resume_path or init_path:
        load_path = resume_path or init_path
        resume_weights = load_file(load_path, device="cpu")
        load_result = model.load_state_dict(resume_weights, strict=False)
        if load_result.unexpected_keys:
            raise ValueError(
                "Unexpected keys in text LM resume checkpoint: "
                + ", ".join(load_result.unexpected_keys)
            )
        full_state = model.state_dict()
        non_alias_missing = [
            name for name in load_result.missing_keys
            if not any(
                key in resume_weights
                and full_state[key].data_ptr() == full_state[name].data_ptr()
                for key in full_state
            )
        ]
        if non_alias_missing:
            raise ValueError(
                "Text LM resume checkpoint is missing parameters: "
                + ", ".join(non_alias_missing)
            )
        if load_result.missing_keys:
            print(
                "Resume checkpoint omitted "
                f"{len(load_result.missing_keys)} shared parameter aliases"
            )
        if resume_path:
            resume_state = load_training_state(resume_path)
            print(
                f"Loaded model weights for resume: {resume_path} "
                + (
                    "(full state sidecar found)"
                    if resume_state is not None
                    else "(weights-only resume)"
                )
            )
        else:
            print(f"Initialized model weights only: {init_path}")
    if adapter_type != "none" and not resume_path:
        matched_adapter_targets, adapter_trainable_count = enable_adapter(
            model, args,
        )
        print(
            f"Enabled {adapter_type}: rank={args.lora_rank} "
            f"alpha={args.lora_alpha or args.lora_rank:g} "
            f"targets={len(matched_adapter_targets)} "
            f"trainable_parameters={adapter_trainable_count:,}"
        )
    print(
        "Model parameter dtype: "
        f"{'torch.bfloat16' if args.bf16 else 'torch.float32'}; "
        f"vocab_chunk_size={args.vocab_chunk_size}"
    )
    print_model_info(model, cast_bf16=args.bf16)

    if args.validate_only:
        assert validation_timer is not None
        model.eval()
        with torch.inference_mode():
            sample_batch = {
                key: value.to(device)
                for key, value in next(iter(eval_loader)).items()
            }
            use_chunked_loss = (
                args.vocab_chunk_size > 0
                and getattr(model, "supports_chunked_causal_loss", False)
            )
            if use_chunked_loss:
                sample_hidden = model(
                    sample_batch["input_ids"],
                    sample_batch["attention_mask"],
                    return_hidden=True,
                )
                sample_output_shape = (
                    sample_hidden.size(0), sample_hidden.size(1), len(tokenizer),
                )
                sample_loss = compute_causal_lm_loss(
                    sample_hidden,
                    model.token_embedding.weight,
                    sample_batch["labels"],
                    vocab_chunk_size=args.vocab_chunk_size,
                )
                sample_finite = torch.isfinite(sample_hidden).all()
            else:
                sample_logits = model(
                    sample_batch["input_ids"],
                    sample_batch["attention_mask"],
                )
                sample_output_shape = tuple(sample_logits.shape)
                sample_loss = None
                sample_finite = torch.isfinite(sample_logits).all()
        expected_shape = (
            sample_batch["input_ids"].size(0),
            sample_batch["input_ids"].size(1),
            len(tokenizer),
        )
        if sample_output_shape != expected_shape:
            raise ValueError(
                "Text LM validation produced an unexpected output shape: "
                f"{sample_output_shape} expected={expected_shape}"
            )
        if not sample_finite:
            raise ValueError("Text LM validation produced non-finite values")
        validation = build_validation_report(
            script=script_name,
            device=device,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
            train_examples=len(train_dataset),
            eval_examples=len(eval_dataset),
            model_parameters=sum(parameter.numel() for parameter in model.parameters()),
            trainable_parameters=sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            steps_per_epoch=len(train_loader),
            measurements=validation_timer.finish(),
            extra={
                "input_shape": list(sample_batch["input_ids"].shape),
                "output_shape": list(sample_output_shape),
                "vocabulary_size": len(tokenizer),
                "max_seq_len": args.max_seq_len,
                "vocab_chunk_size": args.vocab_chunk_size,
                "sample_loss": (
                    float(sample_loss) if sample_loss is not None else None
                ),
            },
        )
        run_recorder.record("validation", **validation)
        run_recorder.finish(status="validate_only")
        print("Validation completed; training was not started.")
        return

    optimizer = build_optimizer(
        args.optimizer,
        adapter_optimizer_parameters(model)
        if adapter_type != "none" else model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
        args=args,
    )
    is_schedule_free = is_schedule_free_optimizer(args.optimizer)
    scheduler = None
    steps_per_epoch = (
        len(train_loader)
        if args.steps_per_epoch <= 0
        else min(args.steps_per_epoch, len(train_loader))
    )
    if not is_schedule_free or args.force_scheduler:
        scheduler = build_lr_scheduler(
            optimizer, args, max(1, args.epochs * steps_per_epoch),
        )

    if args.variable_depth:
        if args.depth_max_bias <= 0.0:
            raise ValueError("--depth-max-bias must be > 0")
        depth_choices = torch.arange(
            args.min_depth, args.num_layers + 1, dtype=torch.long
        )
        depth_scheduler = DepthDistributionScheduler(
            min_depth=args.min_depth,
            max_depth=args.num_layers,
            total_steps=max(1, args.epochs * steps_per_epoch),
            initial_bias=DEPTH_INITIAL_BIAS,
            final_bias=args.depth_max_bias,
        )
        depth_probabilities = depth_scheduler.probabilities()
        expected_depth = depth_scheduler.expected_depth()

        def format_depth_probabilities(probabilities):
            return ", ".join(
                f"{int(depth)}={prob:.3f}"
                for depth, prob in zip(
                    depth_scheduler.depth_choices, probabilities
                )
            )

        print(
            "Initial depth probabilities: "
            + format_depth_probabilities(
                depth_scheduler.probabilities(
                    bias=depth_scheduler.initial_bias
                )
            )
        )
        print(
            "Final depth probabilities: "
            + format_depth_probabilities(
                depth_scheduler.probabilities(
                    bias=depth_scheduler.final_bias
                )
            )
        )
        print(f"Initial depth bias: {depth_scheduler.initial_bias:.3f}")
        print(f"Initial expected depth: {expected_depth:.3f}")
    else:
        depth_scheduler = None
        depth_choices = None
        depth_probabilities = None
        expected_depth = float(model_depth)

    start_epoch = 0
    global_step = 0
    if resume_state is not None:
        start_epoch = int(resume_state["epoch"])
        global_step = int(resume_state["global_step"])
        optimizer.load_state_dict(resume_state["optimizer"])
        saved_scheduler = resume_state.get("scheduler")
        if scheduler is not None and saved_scheduler is not None:
            scheduler.load_state_dict(saved_scheduler)
        elif scheduler is None and saved_scheduler is not None:
            raise ValueError(
                "resume checkpoint contains an LR scheduler, but the current "
                "run disabled it"
            )
        elif scheduler is not None:
            scheduler.step(global_step)
        saved_depth = (resume_state.get("extra") or {}).get("depth_scheduler")
        saved_sampler = (resume_state.get("extra") or {}).get("sampler")
        if saved_sampler is not None:
            train_sampler.load_state_dict(saved_sampler)
            start_epoch = train_sampler.epoch
        if saved_depth is not None and depth_scheduler is None:
            raise ValueError(
                "resume checkpoint contains variable-depth state, but the "
                "current architecture does not enable variable depth"
            )
        if depth_scheduler is not None and saved_depth is not None:
            for key in ("min_depth", "max_depth", "schedule"):
                if saved_depth.get(key) != getattr(depth_scheduler, key):
                    raise ValueError(
                        "Depth scheduler configuration mismatch for "
                        f"{key}: current={getattr(depth_scheduler, key)!r}, "
                        f"saved={saved_depth.get(key)!r}"
                    )
            depth_scheduler.step_count = int(saved_depth["step_count"])
        restore_rng_state(resume_state["rng"])
        print(
            f"Restored full training state: epoch={start_epoch}, "
            f"global_step={global_step}"
        )
    elif resume_path:
        print("Starting a new optimizer schedule from weights-only resume")
    if start_epoch >= args.epochs:
        raise ValueError(
            f"resume starts at epoch {start_epoch}, but --epochs is {args.epochs}; "
            "set --epochs to a larger total epoch count"
        )

    def resume_extra_state():
        extra = {"sampler": train_sampler.state_dict()}
        if depth_scheduler is not None:
            extra["depth_scheduler"] = {
                "min_depth": depth_scheduler.min_depth,
                "max_depth": depth_scheduler.max_depth,
                "schedule": depth_scheduler.schedule,
                "step_count": depth_scheduler.step_count,
            }
        return extra

    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    train_tokens = sum(len(item["input_ids"]) for item in train_dataset)
    eval_tokens = sum(len(item["input_ids"]) for item in eval_dataset)
    print(f"Device: {device}")
    print(f"Data mode: {args.data_mode}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Distillation: {args.distill_mode}")
    if args.distill_mode == "logits":
        print(
            f"Teacher: {args.teacher_model}, temperature={args.distill_temperature:.3f}, "
            f"alpha={args.distill_alpha:.3f}"
        )
    print(f"Optimizer: {args.optimizer}")
    print(f"Vocabulary: {len(tokenizer):,}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Train/Eval examples: {len(train_dataset):,}/{len(eval_dataset):,}")
    print(f"Train/Eval tokens: {train_tokens:,}/{eval_tokens:,}")
    print(f"Architecture: {args.architecture}")
    print(
        f"Model: dim={args.embed_dim}, layers={model_depth}, "
        f"heads={args.num_heads}, kv_heads={args.kv_heads}, "
        f"max_seq_len={args.max_seq_len}"
    )
    print(
        f"LR scheduler: {scheduler.name if scheduler is not None else 'disabled'} "
        f"warmup_steps={scheduler.warmup_steps if scheduler is not None else 0}"
    )
    print(
        f"Batch size: {args.batch_size}, epochs: {args.epochs}, "
        f"steps/epoch: {steps_per_epoch}, learning rate: {args.lr:.3e}"
    )
    evaluation_description = (
        f"depths=min/mid/max every {args.eval_interval} epochs"
        if uses_shared_depth
        else "full architecture depth"
    )
    print(
        f"Evaluation: max_batches={args.eval_max_batches or 'all'}, "
        + evaluation_description
    )
    print(
        f"Checkpoint saving: mode={args.save_mode}, "
        f"interval={args.save_interval} epochs, "
        f"retention={args.checkpoint_retention} generations"
    )
    print(f"Total training steps: {args.epochs * steps_per_epoch}")
    print(f"Expected depth: {expected_depth:.3f}")
    print("Starting training...")

    tensorboard_log_dir = tensorboard_dir
    writer = SummaryWriter(tensorboard_log_dir)
    print(f"TensorBoard log: {tensorboard_log_dir}")

    latest_step_path = os.path.join(
        checkpoint_dir, "latest", "model.safetensors"
    )

    def write_checkpoint(checkpoint_path, epoch_number):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        save_file(
            compact_state_dict(model, cast_bf16=args.bf16),
            checkpoint_path,
            metadata=checkpoint_config_metadata(args, TEXT_LM_CONFIG_METADATA_KEY),
        )
        save_training_state(
            checkpoint_path,
            make_training_state(
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_number,
                global_step=global_step,
                extra=resume_extra_state(),
            ),
        )

    def save_epoch_checkpoint(epoch_number):
        if args.save_mode == "final":
            return None
        if (
            args.save_mode == "interval"
            and epoch_number % args.save_interval != 0
        ):
            return None
        epoch_checkpoint_dir = os.path.join(
            str(run_recorder.checkpoints_dir),
            f"epoch_{epoch_number:04d}",
        )
        os.makedirs(epoch_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            epoch_checkpoint_dir, "model.safetensors"
        )
        write_checkpoint(checkpoint_path, epoch_number)
        print(
            f"Saved checkpoint: {checkpoint_path} "
            f"({format_bytes(os.path.getsize(checkpoint_path))}, full resume state saved)"
        )
        checkpoint_root = str(run_recorder.checkpoints_dir)
        checkpoint_dirs = []
        for entry in os.scandir(checkpoint_root):
            if not entry.is_dir():
                continue
            match = re.fullmatch(r"epoch_(\d+)", entry.name)
            if match is not None:
                checkpoint_dirs.append((int(match.group(1)), entry.path))
        checkpoint_dirs.sort(reverse=True)
        for _, old_checkpoint_dir in checkpoint_dirs[args.checkpoint_retention:]:
            shutil.rmtree(old_checkpoint_dir)
            print(f"Removed old checkpoint: {old_checkpoint_dir}")
        return checkpoint_path

    stop_controller = GracefulStop(
        "Ctrl-C received; finishing the current batch and saving a checkpoint..."
    )
    stop_controller.install()

    def finish_interrupted(epoch_number):
        write_checkpoint(latest_step_path, epoch_number)
        run_recorder.record(
            "checkpoint",
            kind="interrupted",
            epoch=epoch_number,
            global_step=global_step,
            sampler_position=train_sampler.position,
            path=latest_step_path,
        )
        writer.close()
        stop_controller.restore()
        run_recorder.finish(
            status="interrupted",
            checkpoints=[latest_step_path],
        )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler.epoch != epoch:
            train_sampler.set_epoch(epoch)
        samples_seen = train_sampler.position
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        model.train()
        if is_schedule_free:
            optimizer.train()
        train_loss = 0.0
        train_hard_loss = 0.0
        train_soft_loss = 0.0
        train_kl_divergence = 0.0
        grad_norm_total = 0.0
        train_steps = 0
        epoch_loader = (
            train_loader
            if args.steps_per_epoch <= 0
            else islice(train_loader, steps_per_epoch)
        )
        progress_total = min(steps_per_epoch, len(train_loader))
        progress = RichProgress(
            epoch_loader,
            total=progress_total,
            description=f"Epoch {epoch + 1}/{args.epochs}",
        )
        for batch_index, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            train_depth = None
            if args.variable_depth:
                depth_probabilities = depth_scheduler.probabilities()
                depth_index = torch.multinomial(
                    depth_probabilities, num_samples=1
                )
                train_depth = int(depth_choices[depth_index].item())
            optimizer.zero_grad(set_to_none=True)
            use_chunked_loss = (
                teacher_model is None
                and args.vocab_chunk_size > 0
                and getattr(model, "supports_chunked_causal_loss", False)
            )
            model_output = model(
                batch["input_ids"],
                batch["attention_mask"],
                depth=train_depth,
                return_hidden=use_chunked_loss,
            )
            if use_chunked_loss:
                loss = compute_causal_lm_loss(
                    model_output,
                    model.token_embedding.weight,
                    batch["labels"],
                    vocab_chunk_size=args.vocab_chunk_size,
                )
                hard_loss = loss.detach()
                soft_loss = torch.zeros_like(hard_loss)
                kl_divergence = torch.zeros_like(hard_loss)
            elif teacher_model is None:
                logits = model_output
                loss = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    batch["labels"][:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                hard_loss = loss.detach()
                soft_loss = torch.zeros_like(hard_loss)
                kl_divergence = torch.zeros_like(hard_loss)
            else:
                logits = model_output
                with torch.no_grad():
                    teacher_logits = teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    ).logits
                (
                    loss,
                    hard_loss,
                    soft_loss,
                    kl_divergence,
                ) = _compute_distillation_components(
                    logits,
                    teacher_logits,
                    batch["labels"],
                    args.distill_temperature,
                    args.distill_alpha,
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            samples_seen += batch["input_ids"].size(0)
            train_sampler.set_position(samples_seen)
            if scheduler is not None:
                scheduler.step(global_step)
            if depth_scheduler is not None:
                depth_scheduler.step()
            train_loss += loss.item()
            train_hard_loss += hard_loss.item()
            train_soft_loss += soft_loss.item()
            train_kl_divergence += kl_divergence.item()
            grad_norm_total += float(grad_norm)
            train_steps += 1
            progress.set_status(build_standard_progress_rows(
                step=batch_index,
                total_steps=progress_total,
                global_step=global_step,
                loss=f"{loss.item():.4f}",
                learning_rate=f"{optimizer.param_groups[0]['lr']:.4e}",
            ))
            maybe_collect_memory(
                global_step,
                gc_interval=args.gc_interval,
                empty_cache_interval=args.empty_cache_interval,
            )
            if (
                args.checkpoint_interval_steps > 0
                and global_step % args.checkpoint_interval_steps == 0
            ):
                write_checkpoint(latest_step_path, epoch)
                run_recorder.record(
                    "checkpoint",
                    kind="latest",
                    epoch=epoch,
                    global_step=global_step,
                    sampler_position=train_sampler.position,
                    path=latest_step_path,
                )
            if stop_controller.requested:
                finish_interrupted(epoch)
                return

        if is_schedule_free:
            optimizer.eval()
        model.eval()
        depth_eval_due = (
            (epoch + 1) % args.eval_interval == 0
            or epoch + 1 == args.epochs
        )
        if uses_shared_depth:
            mid_depth = (args.min_depth + args.num_layers) // 2
            depth_eval_labels = {
                args.min_depth: "min",
                mid_depth: "mid",
                args.num_layers: "max",
            }
            eval_depths = (
                sorted(depth_eval_labels)
                if depth_eval_due
                else [args.num_layers]
            )
        else:
            depth_eval_labels = {}
            eval_depths = [None]

        depth_eval_losses = {}
        depth_eval_hard_losses = {}
        depth_eval_kl_divergences = {}
        with torch.no_grad():
            for eval_depth in eval_depths:
                depth_loss = 0.0
                depth_hard_loss = 0.0
                depth_kl_divergence = 0.0
                eval_steps = 0
                for batch_index, batch in enumerate(eval_loader):
                    if (
                        args.eval_max_batches > 0
                        and batch_index >= args.eval_max_batches
                    ):
                        break
                    batch = {
                        key: value.to(device) for key, value in batch.items()
                    }
                    use_chunked_loss = (
                        teacher_model is None
                        and args.vocab_chunk_size > 0
                        and getattr(model, "supports_chunked_causal_loss", False)
                    )
                    model_output = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        depth=eval_depth,
                        return_hidden=use_chunked_loss,
                    )
                    if use_chunked_loss:
                        loss = compute_causal_lm_loss(
                            model_output,
                            model.token_embedding.weight,
                            batch["labels"],
                            vocab_chunk_size=args.vocab_chunk_size,
                        )
                        hard_loss = loss
                        kl_divergence = torch.zeros_like(loss)
                    elif teacher_model is None:
                        logits = model_output
                        loss = F.cross_entropy(
                            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                            batch["labels"][:, 1:].contiguous().view(-1),
                            ignore_index=-100,
                        )
                        hard_loss = loss
                        kl_divergence = torch.zeros_like(loss)
                    else:
                        logits = model_output
                        teacher_logits = teacher_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False,
                        ).logits
                        (
                            loss,
                            hard_loss,
                            _,
                            kl_divergence,
                        ) = _compute_distillation_components(
                            logits,
                            teacher_logits,
                            batch["labels"],
                            args.distill_temperature,
                            args.distill_alpha,
                        )
                    depth_loss += loss.item()
                    depth_hard_loss += hard_loss.item()
                    depth_kl_divergence += kl_divergence.item()
                    eval_steps += 1
                depth_eval_losses[eval_depth] = (
                    depth_loss / max(1, eval_steps)
                )
                depth_eval_hard_losses[eval_depth] = (
                    depth_hard_loss / max(1, eval_steps)
                )
                depth_eval_kl_divergences[eval_depth] = (
                    depth_kl_divergence / max(1, eval_steps)
                )

        eval_loss = depth_eval_losses[
            args.num_layers if uses_shared_depth else None
        ]
        eval_hard_loss = depth_eval_hard_losses[
            args.num_layers if uses_shared_depth else None
        ]
        eval_kl_divergence = depth_eval_kl_divergences[
            args.num_layers if uses_shared_depth else None
        ]
        if stop_controller.requested:
            finish_interrupted(epoch)
            return
        epoch_elapsed = time.perf_counter() - epoch_start
        cuda_peak_allocated_mib = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            if device.type == "cuda" else None
        )
        cuda_peak_reserved_mib = (
            torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
            if device.type == "cuda" else None
        )
        current_lr = optimizer.param_groups[0].get(
            "scheduled_lr", optimizer.param_groups[0]["lr"]
        )
        current_expected_depth = (
            depth_scheduler.expected_depth()
            if depth_scheduler is not None
            else float(model_depth)
        )
        mean_train_loss = train_loss / max(1, train_steps)
        mean_train_hard_loss = train_hard_loss / max(1, train_steps)
        mean_train_soft_loss = train_soft_loss / max(1, train_steps)
        mean_train_kl_divergence = (
            train_kl_divergence / max(1, train_steps)
        )
        train_ppl = perplexity_from_loss(mean_train_hard_loss)
        eval_ppl = perplexity_from_loss(eval_hard_loss)
        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train Loss: {mean_train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Train PPL: {train_ppl:.3f} | Eval PPL: {eval_ppl:.3f} | "
            f"LR: {current_lr:.4e} | "
            f"Grad Norm: {grad_norm_total / max(1, train_steps):.4e} | "
            f"Expected Depth: {current_expected_depth:.3f} | "
            f"Steps/s: {train_steps / max(epoch_elapsed, 1e-6):.2f} | "
            f"Time: {epoch_elapsed:.1f}s"
        )
        if teacher_model is not None:
            print(
                f"Distillation metrics: Train KL: "
                f"{mean_train_kl_divergence:.6f} | Eval KL: "
                f"{eval_kl_divergence:.6f} | "
                f"Train soft loss (T²·KL): {mean_train_soft_loss:.6f}"
            )
        epoch_index = epoch + 1
        writer.add_scalar(
            "Loss/train", mean_train_loss, epoch_index
        )
        writer.add_scalar("Loss/eval", eval_loss, epoch_index)
        writer.add_scalar("Loss/train_hard", mean_train_hard_loss, epoch_index)
        writer.add_scalar("Loss/eval_hard", eval_hard_loss, epoch_index)
        writer.add_scalar("Perplexity/train", train_ppl, epoch_index)
        writer.add_scalar("Perplexity/eval", eval_ppl, epoch_index)
        writer.add_scalar("LearningRate/current", current_lr, epoch_index)
        writer.add_scalar(
            "GradientNorm/mean",
            grad_norm_total / max(1, train_steps),
            epoch_index,
        )
        if teacher_model is not None:
            writer.add_scalar("Loss/train_soft", mean_train_soft_loss, epoch_index)
            writer.add_scalar(
                "Loss/train_kl_divergence",
                mean_train_kl_divergence,
                epoch_index,
            )
            writer.add_scalar(
                "Loss/eval_kl_divergence",
                eval_kl_divergence,
                epoch_index,
            )
        writer.add_scalar(
            "Performance/steps_per_second",
            train_steps / max(epoch_elapsed, 1e-6),
            epoch_index,
        )
        standard_extra = {
            "train/grad_norm/mean": grad_norm_total / max(1, train_steps),
            "train/loss/hard": mean_train_hard_loss,
            "eval/loss/hard": eval_hard_loss,
            "train/perplexity": train_ppl,
            "eval/perplexity": eval_ppl,
            "model/parameters": float(parameter_count),
            "model/trainable_parameters": float(trainable_count),
            "data/train_tokens": float(train_tokens),
            "data/eval_tokens": float(eval_tokens),
        }
        if teacher_model is not None:
            standard_extra.update({
                "train/loss/kl_divergence": mean_train_kl_divergence,
                "eval/loss/kl_divergence": eval_kl_divergence,
                "train/loss/soft_scaled": mean_train_soft_loss,
            })
        if cuda_peak_allocated_mib is not None:
            standard_extra.update({
                "memory/cuda_peak_allocated_mib": cuda_peak_allocated_mib,
                "memory/cuda_peak_reserved_mib": cuda_peak_reserved_mib,
            })
        write_standard_training_metrics(
            writer,
            step=epoch_index,
            train_loss=mean_train_loss,
            eval_loss=eval_loss,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            scheduled_learning_rate=float(current_lr),
            steps_per_second=train_steps / max(epoch_elapsed, 1e-6),
            extra=standard_extra,
        )
        writer.add_scalar(
            "Depth/expected",
            current_expected_depth,
            epoch_index,
        )
        if depth_scheduler is not None:
            current_probabilities = depth_scheduler.probabilities()
            print(
                "Depth probabilities: "
                + ", ".join(
                    f"{int(depth)}={prob:.3f}"
                    for depth, prob in zip(
                        depth_scheduler.depth_choices, current_probabilities
                    )
                )
                + f" | bias={depth_scheduler.bias:.3f}"
                + f" | expected={depth_scheduler.expected_depth():.3f}"
            )
            writer.add_scalar(
                "Depth/bias", depth_scheduler.bias, epoch_index
            )
            writer.add_scalar(
                "Depth/expected_scheduler",
                depth_scheduler.expected_depth(),
                epoch_index,
            )
            for depth, probability in zip(
                depth_scheduler.depth_choices, current_probabilities
            ):
                writer.add_scalar(
                    f"Depth/probability_{int(depth)}",
                    probability.item(),
                    epoch_index,
                )
        if uses_shared_depth and depth_eval_due:
            print(
                "Eval loss by depth "
                f"(max_batches={args.eval_max_batches or 'all'}): "
                + " | ".join(
                    f"{depth_eval_labels[depth]}(d={depth}): "
                    f"{depth_eval_losses[depth]:.4f}"
                    for depth in eval_depths
                )
            )
        if uses_shared_depth:
            for depth, loss in depth_eval_losses.items():
                depth_label = depth_eval_labels[depth]
                writer.add_scalar(
                    f"Loss/eval_depth_{depth_label}", loss, epoch_index
                )

        train_sampler.set_epoch(epoch + 1)
        checkpoint_path = save_epoch_checkpoint(epoch_index)
        run_recorder.record(
            "epoch",
            epoch=epoch_index,
            global_step=global_step,
            train_loss=mean_train_loss,
            eval_loss=eval_loss,
            train_hard_loss=mean_train_hard_loss,
            eval_hard_loss=eval_hard_loss,
            train_soft_loss=mean_train_soft_loss,
            train_kl_divergence=mean_train_kl_divergence,
            eval_kl_divergence=eval_kl_divergence,
            train_ppl=train_ppl,
            eval_ppl=eval_ppl,
            learning_rate=current_lr,
            steps_per_second=train_steps / max(epoch_elapsed, 1e-6),
            sampler_position=train_sampler.position,
            checkpoint=checkpoint_path,
            model_parameters=parameter_count,
            trainable_parameters=trainable_count,
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            cuda_peak_allocated_mib=cuda_peak_allocated_mib,
            cuda_peak_reserved_mib=cuda_peak_reserved_mib,
        )
        run_recorder.record_training_step(
            global_step=global_step,
            epoch=epoch_index,
            train_loss=mean_train_loss,
            eval_loss=eval_loss,
            effective_lr=float(optimizer.param_groups[0]["lr"]),
            scheduled_lr=float(current_lr),
            step_time_sec=epoch_elapsed / max(train_steps, 1),
            steps_per_second=train_steps / max(epoch_elapsed, 1e-6),
            metrics={
                "train_grad_norm_mean": grad_norm_total / max(1, train_steps),
                "train_hard_loss": mean_train_hard_loss,
                "eval_hard_loss": eval_hard_loss,
                "train_soft_loss": mean_train_soft_loss,
                "train_kl_divergence": mean_train_kl_divergence,
                "eval_kl_divergence": eval_kl_divergence,
                "train_ppl": train_ppl,
                "eval_ppl": eval_ppl,
                "expected_depth": current_expected_depth,
                "model_parameters": parameter_count,
                "trainable_parameters": trainable_count,
                "train_tokens": train_tokens,
                "eval_tokens": eval_tokens,
                "cuda_peak_allocated_mib": cuda_peak_allocated_mib,
                "cuda_peak_reserved_mib": cuda_peak_reserved_mib,
            },
        )

    if depth_scheduler is not None:
        final_probabilities = depth_scheduler.probabilities(
            bias=depth_scheduler.final_bias
        )
        print(
            "Final depth probabilities: "
            + ", ".join(
                f"{int(depth)}={prob:.3f}"
                for depth, prob in zip(
                    depth_scheduler.depth_choices, final_probabilities
                )
            )
        )
        print(
            f"Final expected depth: "
            f"{depth_scheduler.expected_depth(bias=depth_scheduler.final_bias):.3f}"
        )

    writer.close()
    stop_controller.restore()

    save_path = os.path.join(artifact_dir, "model.safetensors")
    save_file(
        compact_state_dict(model, cast_bf16=args.bf16),
        save_path,
        metadata=checkpoint_config_metadata(args, TEXT_LM_CONFIG_METADATA_KEY),
    )
    save_training_state(
        save_path,
        make_training_state(
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            global_step=global_step,
            extra=resume_extra_state(),
        ),
    )
    model.cpu()
    tokenizer_dir = os.path.join(artifact_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    torch.save(vars(args), os.path.join(artifact_dir, "config.pt"))
    print(
        f"Saved model and tokenizer to {artifact_dir} "
        f"({format_bytes(os.path.getsize(save_path))}, full resume state saved)"
    )
    run_recorder.record("checkpoint", kind="final", path=save_path)
    checkpoints = [save_path]
    if os.path.isfile(latest_step_path):
        checkpoints.insert(0, latest_step_path)
    run_recorder.finish(checkpoints=checkpoints)

if __name__ == "__main__":
    main()
