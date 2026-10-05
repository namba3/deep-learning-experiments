"""Generate a raw text continuation from a trained text LM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from core.utils import convert_linear_to_bf16, convert_rmsnorm_to_dtype_aware
from runtime.device import add_device_argument, resolve_device
from text_lm.train import (
    DEFAULT_TOKENIZER,
    TinyTextLM,
    read_text_lm_checkpoint_config,
)


MAX_GENERATION_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.95


def _value(config: dict, name: str, default):
    value = config.get(name)
    return default if value is None else value


def _resolve_tokenizer_path(
    checkpoint: Path, config: dict, tokenizer_arg: str | None,
) -> str:
    if tokenizer_arg:
        return tokenizer_arg
    sibling = checkpoint.parent / "tokenizer"
    if sibling.is_dir():
        return str(sibling)
    return str(_value(config, "tokenizer", DEFAULT_TOKENIZER))


def _resolve_dtype(
    config: dict, dtype_arg: str, device: torch.device,
) -> tuple[torch.dtype, bool]:
    use_bf16 = (
        bool(_value(config, "bf16", False))
        if dtype_arg == "auto" else dtype_arg == "bf16"
    )
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported(device):
        raise RuntimeError("BF16 was requested, but this CUDA device does not support it")
    return (torch.bfloat16 if use_bf16 else torch.float32), use_bf16


def _build_model(
    checkpoint: Path,
    config: dict,
    tokenizer_vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
    use_bf16: bool,
) -> TinyTextLM:
    state_dict = load_file(str(checkpoint), device="cpu")
    embedding = state_dict.get("token_embedding.weight")
    if embedding is None:
        raise ValueError("checkpoint is missing token_embedding.weight")
    checkpoint_vocab_size = embedding.shape[0]
    if checkpoint_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            "tokenizer vocabulary does not match checkpoint: "
            f"tokenizer={tokenizer_vocab_size}, checkpoint={checkpoint_vocab_size}"
        )

    model = TinyTextLM(
        vocab_size=checkpoint_vocab_size,
        max_seq_len=int(_value(config, "max_seq_len", 512)),
        embed_dim=int(_value(config, "embed_dim", 2048)),
        num_layers=int(_value(config, "num_layers", 16)),
        num_heads=int(_value(config, "num_heads", 32)),
        kv_heads=int(_value(config, "kv_heads", 8)),
        condition_dim=int(_value(config, "condition_dim", 64)),
        transform_rank=int(_value(config, "transform_rank", 10)),
        architecture=str(_value(config, "architecture", "naive")),
        looped_blocks=_value(config, "looped_blocks", None),
        looped_prefix_layers=int(_value(config, "looped_prefix_layers", 4)),
        looped_repeats=int(_value(config, "looped_repeats", 4)),
        looped_suffix_layers=int(_value(config, "looped_suffix_layers", 4)),
        mhla_looped_prefix_cycles=int(
            _value(config, "mhla_looped_prefix_cycles", 1)
        ),
        mhla_looped_repeats=int(_value(config, "mhla_looped_repeats", 2)),
        mhla_looped_suffix_cycles=int(
            _value(config, "mhla_looped_suffix_cycles", 1)
        ),
        compute_dtype=dtype if use_bf16 else None,
    ).to(device=device, dtype=dtype)
    if use_bf16:
        convert_rmsnorm_to_dtype_aware(model.decoder)
        depth_embedding = getattr(model.decoder, "depth_embedding", None)
        convert_linear_to_bf16(
            model.decoder,
            skip_modules=()
            if depth_embedding is None else (depth_embedding,),
        )

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(
            "unexpected checkpoint parameters: "
            + ", ".join(load_result.unexpected_keys)
        )
    full_state = model.state_dict()
    non_alias_missing = [
        name for name in load_result.missing_keys
        if not any(
            key in state_dict
            and full_state[key].data_ptr() == full_state[name].data_ptr()
            for key in full_state
        )
    ]
    if non_alias_missing:
        raise ValueError(
            "checkpoint is missing model parameters: "
            + ", ".join(non_alias_missing)
        )
    model.eval()
    return model


def _next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


class TextGenerationModel(Protocol):
    max_seq_len: int

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor: ...


@torch.inference_mode()
def generate_token_ids(
    model: "TextGenerationModel",
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = MAX_GENERATION_TOKENS,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Append up to ``max_new_tokens`` using the model's context window."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape (1, sequence_length)")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token")
    if not 1 <= max_new_tokens <= MAX_GENERATION_TOKENS:
        raise ValueError(f"max_new_tokens must be in [1, {MAX_GENERATION_TOKENS}]")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    generated = input_ids
    for _ in range(max_new_tokens):
        context = generated[:, -model.max_seq_len:]
        logits = model(context)[:, -1, :].float()
        token = _next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated = torch.cat((generated, token), dim=1)
        if eos_token_id is not None and int(token.item()) == eos_token_id:
            break
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", required=True,
                        help="続きを生成する生テキスト。会話templateは適用しない")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer path or Hugging Face ID（既定: checkpoint同梱）")
    parser.add_argument(
        "--dtype", choices=("auto", "fp32", "bf16"), default="auto",
        help="checkpoint metadataに従うauto、またはfp32/bf16で上書き",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_GENERATION_TOKENS,
        help=f"生成する最大token数（1〜{MAX_GENERATION_TOKENS}、既定: 1024）",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="0でgreedy、それより大きい値でsampling（既定: 0.8）",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="sampling時のtop-k。0で無効（既定: 40）",
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P,
        help="sampling時のnucleus probability（既定: 0.95 = 95%%）",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling用seed")
    add_device_argument(parser)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    config = read_text_lm_checkpoint_config(str(args.checkpoint))
    if not config:
        raise ValueError(
            "checkpoint has no text_lm.config metadata; "
            "a metadata-bearing training checkpoint is required"
        )
    device = resolve_device(args.device)
    dtype, use_bf16 = _resolve_dtype(config, args.dtype, device)
    tokenizer_path = _resolve_tokenizer_path(
        args.checkpoint, config, args.tokenizer,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _build_model(
        args.checkpoint, config, len(tokenizer), device, dtype, use_bf16,
    )
    encoded = tokenizer(
        args.prompt, add_special_tokens=False, return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] == 0:
        raise ValueError("prompt produced no tokens")
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("seed must be >= 0")
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    generated = generate_token_ids(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
