"""Model, optimizer-argument, and device helpers for the Text-LM probe."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.utils import convert_linear_to_bf16, convert_rmsnorm_to_dtype_aware
from text_lm.train import CausalCollator, TinyTextLM
from runtime.sampler import ResumableRandomSampler


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return torch.device(name)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if device.type != "cuda" or not torch.cuda.is_bf16_supported(device):
        raise ValueError("BF16 requires a CUDA device with BF16 support")
    return torch.bfloat16


def _state_metrics(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    state_bytes = 0
    state_elements = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                state_elements += value.numel()
                state_bytes += value.numel() * value.element_size()
    return state_bytes, state_elements


def _optimizer_args(args, seed: int) -> argparse.Namespace:
    """Translate probe names to the shared optimizer factory contract."""
    return argparse.Namespace(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        came_lrsf_rank=args.rank,
        came_lrsf_beta1=args.lrsf_beta1,
        came_lrsf_warmup_steps=args.lrsf_warmup_steps,
        came_lrsf_r=args.lrsf_r,
        came_lrsf_weight_lr_power=args.lrsf_weight_lr_power,
        came_lrsf_seed=seed,
        came_lrsf_refresh_mode=args.refresh_mode,
        came_lrsf_refresh_interval=args.refresh_interval,
        came_lrsf_refresh_window=args.refresh_window,
        came_lrsf_refresh_mix=args.refresh_mix,
        came_lrsf_orthogonal_refresh_rate=args.orthogonal_rate,
        came_lrsf_orthogonal_refresh_direction=args.orthogonal_direction,
        came_lrsf_orthogonal_refresh_signal=args.orthogonal_signal,
        adamw_lrsf_rank=args.rank,
        adamw_sf_lr_rank=args.rank,
        adamw_lrsf_lr_rank=args.rank,
        adamw_lrsf_beta1=args.lrsf_beta1,
        adamw_lrsf_beta2=0.999,
        adamw_lrsf_warmup_steps=args.lrsf_warmup_steps,
        adamw_lrsf_r=args.lrsf_r,
        adamw_lrsf_weight_lr_power=args.lrsf_weight_lr_power,
        adamw_sf_backend=args.adamw_sf_backend,
        adamw_lrsf_seed=seed,
        adamw_sf_lr_seed=seed,
        adamw_lrsf_lr_seed=seed,
        adamw_lrsf_eps=1e-8,
        adamw_lr_ema_rank=args.rank,
        adamw_lr_ema_beta=args.adamw_lr_ema_beta,
        adamw_lr_ema_confidence_beta=args.adamw_lr_ema_confidence_beta,
        adamw_lr_ema_confidence_alpha=args.adamw_lr_ema_confidence_alpha,
        adamw_lr_ema_projection_scale=args.adamw_lr_ema_projection_scale,
        adamw_lr_ema_seed=seed,
        adamw_lr_ema_eps=1e-8,
        adamw_lrsf_projection_refresh={
            "mode": args.refresh_mode,
            "interval": args.refresh_interval,
            "window": args.refresh_window,
            "mix": args.refresh_mix,
            "ema_decay": args.refresh_ema_decay,
            "diagnostics": args.record_refresh_diagnostics,
            "transport_overlap": args.refresh_transport_overlap,
        },
        adamw_lrsf_projection_refresh_state=args.lrsf_refresh_state,
        adamw_lrsf_orthogonal_refresh={
            "rate": args.orthogonal_rate,
            "seed": seed,
            "direction": args.orthogonal_direction,
            "signal": args.orthogonal_signal,
        },
        apollo_rank=args.rank,
        apollo_scale=args.apollo_scale,
        apollo_update_proj_gap=200,
        apollo_projection_refresh_mode="hard",
        apollo_projection_refresh_state="reset",
        apollo_orthogonal_refresh_rate=0.0,
        apollo_disable_norm_growth_limiter=args.apollo_disable_norm_growth_limiter,
        apollo_norm_growth_rate=args.apollo_norm_growth_rate,
        apollo_confidence_beta=args.adamw_lr_ema_confidence_beta,
        apollo_confidence_alpha=args.adamw_lr_ema_confidence_alpha,
    )


def _loss(model: TinyTextLM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(batch["input_ids"], batch["attention_mask"])
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
        batch["labels"][:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def _build_model(args, vocab_size: int, device: torch.device, dtype: torch.dtype):
    model = TinyTextLM(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kv_heads=args.kv_heads,
        condition_dim=args.condition_dim,
        transform_rank=args.transform_rank,
        architecture="naive",
        compute_dtype=dtype if dtype == torch.bfloat16 else None,
    ).to(device=device, dtype=dtype)
    if dtype == torch.bfloat16:
        convert_rmsnorm_to_dtype_aware(model.decoder)
        convert_linear_to_bf16(model.decoder)
    return model


def _load_initial_state(
    args, tokenizer, device: torch.device, dtype: torch.dtype, seed: int,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial_model = _build_model(args, len(tokenizer), torch.device("cpu"), torch.float32)
    return {
        key: value.detach().clone().to(device=device, dtype=dtype)
        for key, value in initial_model.state_dict().items()
    }


def _build_case_loaders(args, tokenizer, train_dataset, eval_dataset,
                        device: torch.device, seed: int):
    sampler = ResumableRandomSampler(train_dataset, seed=seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=CausalCollator(tokenizer),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CausalCollator(tokenizer),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    return sampler, loader, eval_loader
