import pytest
import torch
from verify.text_lm_optimizer_convergence import _build_model, _loss, _optimizer_args, _validate_args, parse_args


def test_text_lm_optimizer_probe_parser_exposes_lrsf_controls():
    args = parse_args([
        "--device", "cpu",
        "--seeds", "0,2",
        "--optimizers", "CAME,CAME-LRSF,APOLLO,APOLLO-Conf,APOLLO-CAME,APOLLO-CAME-LRSF",
        "--rank", "4",
        "--refresh-mode", "smooth",
        "--refresh-mix", "ema",
        "--lrsf-refresh-state", "reset",
        "--refresh-ema-decay", "0.96",
        "--record-refresh-diagnostics",
        "--refresh-transport-overlap", "0.95",
        "--orthogonal-direction", "loss_lowering",
        "--apollo-norm-growth-rate", "1.05",
        "--apollo-scale", "0.5",
        "--apollo-disable-norm-growth-limiter",
    ])

    assert args.seeds == (0, 2)
    assert args.optimizers == (
        "CAME", "CAME-LRSF", "APOLLO", "APOLLO-Conf", "APOLLO-CAME",
        "APOLLO-CAME-LRSF",
    )
    assert args.rank == 4
    assert args.refresh_mode == "smooth"
    assert args.refresh_mix == "ema"
    assert args.lrsf_refresh_state == "reset"
    assert args.refresh_ema_decay == 0.96
    assert args.record_refresh_diagnostics is True
    assert args.refresh_transport_overlap == 0.95
    assert args.orthogonal_direction == "loss_lowering"
    assert args.apollo_norm_growth_rate == 1.05
    assert args.apollo_scale == 0.5
    assert args.apollo_disable_norm_growth_limiter is True

def test_text_lm_optimizer_probe_parser_exposes_confidence_ema_controls():
    args = parse_args([
        "--optimizers", "AdamW-LR-EMA-Conf",
        "--lr-ema-confidence-beta", "0.97",
        "--lr-ema-confidence-alpha", "0.002",
    ])

    assert args.optimizers == ("AdamW-LR-EMA-Conf",)
    assert args.adamw_lr_ema_confidence_beta == 0.97
    assert args.adamw_lr_ema_confidence_alpha == 0.002

def test_text_lm_optimizer_probe_can_record_confidence_diagnostics():
    args = parse_args(["--record-confidence-diagnostics"])

    assert args.record_confidence_diagnostics is True

def test_text_lm_optimizer_probe_defaults_to_qwen_tokenizer():
    args = parse_args([])

    assert args.dataset_name == "roneneldan/TinyStories"
    assert args.train_tokens == 1_000_000
    assert args.eval_tokens == 100_000
    assert args.tokenizer == "Qwen/Qwen3.5-0.8B"
    assert args.refresh_mode == "hard"
    assert args.lrsf_refresh_state == "transport"
    assert args.trajectory_pca_calibration_fraction == 0.5
    assert args.trajectory_pca_rolling_window == 0
    assert args.record_trajectory_curvature is False

def test_text_lm_optimizer_probe_defaults_to_adamw_lrsf_came_comparison():
    args = parse_args([])

    assert args.optimizers == (
        "AdamW", "AdamW-SF", "AdamW-LRSF", "CAME",
    )
    assert args.adamw_sf_backend == "torch"

def test_text_lm_optimizer_probe_accepts_preconditioner_ablation():
    args = parse_args([
        "--device", "cpu",
        "--optimizers", "AdamW-SF,AdamW-SF-LR,AdamW-LRSF",
        "--rank", "8",
    ])

    assert args.optimizers == ("AdamW-SF", "AdamW-SF-LR", "AdamW-LRSF")

def test_text_lm_optimizer_probe_accepts_integrated_low_rank_variant():
    args = parse_args([
        "--device", "cpu",
        "--optimizers", "AdamW-LRSF-LR",
        "--rank", "8",
    ])

    assert args.optimizers == ("AdamW-LRSF-LR",)

def test_text_lm_optimizer_probe_rejects_reset_for_plain_lrsf():
    args = parse_args([
        "--optimizers", "AdamW-LRSF",
        "--lrsf-refresh-state", "reset",
    ])

    with pytest.raises(ValueError, match="AdamW-LRSF-LR"):
        _validate_args(args)

def test_text_lm_optimizer_probe_rejects_non_positive_residual_block_size():
    args = parse_args([
        "--residual-compression-block-size", "0",
    ])

    with pytest.raises(ValueError, match="residual-compression-block-size"):
        _validate_args(args)

def test_text_lm_optimizer_probe_accepts_confidence_lrsf_variant():
    args = parse_args([
        "--device", "cpu",
        "--optimizers", "AdamW-LR-EMA-Conf-LRSF",
        "--rank", "8",
    ])

    assert args.optimizers == ("AdamW-LR-EMA-Conf-LRSF",)

def test_text_lm_optimizer_probe_exposes_state_rank_diagnostics():
    args = parse_args([
        "--record-state-rank",
        "--state-rank-interval", "7",
        "--state-rank-max-elements", "1234",
        "--state-rank-max-tensors", "3",
        "--state-rank-parameter", "decoder",
        "--record-state-trajectory-pca",
        "--state-trajectory-max-snapshots", "12",
        "--record-update-trajectory-pca",
        "--trajectory-pca-calibration-fraction", "0.5",
        "--residual-compression-block-size", "128",
        "--residual-compression-scale-mode", "percentile_99_9",
        "--record-trajectory-curvature",
        "--record-update-reconstruction",
        "--update-reconstruction-ranks", "2,4,8",
    ])

    assert args.record_state_rank is True
    assert args.state_rank_interval == 7
    assert args.state_rank_max_elements == 1234
    assert args.state_rank_max_tensors == 3
    assert args.state_rank_parameter == "decoder"
    assert args.state_rank_delta_ema_decay == 0.9
    assert args.record_state_trajectory_pca is True
    assert args.state_trajectory_max_snapshots == 12
    assert args.record_update_trajectory_pca is True
    assert args.trajectory_pca_calibration_fraction == 0.5
    assert args.trajectory_pca_rolling_window == 0
    assert args.residual_compression_block_size == 128
    assert args.residual_compression_scale_mode == "percentile_99_9"
    assert args.record_trajectory_curvature is True
    assert args.record_update_reconstruction is True
    assert args.update_reconstruction_ranks == (2, 4, 8)

def test_text_lm_optimizer_probe_builds_lrsf_contract_and_loss():
    args = parse_args([
        "--device", "cpu",
        "--embed-dim", "16",
        "--num-layers", "2",
        "--num-heads", "4",
        "--kv-heads", "2",
        "--max-seq-len", "8",
        "--rank", "2",
    ])
    model = _build_model(args, 31, torch.device("cpu"), torch.float32)
    optimizer_args = _optimizer_args(args, seed=7)

    assert optimizer_args.came_lrsf_rank == 2
    assert optimizer_args.came_lrsf_seed == 7
    assert optimizer_args.came_lrsf_refresh_mode == "hard"
    assert optimizer_args.adamw_sf_backend == "torch"
    assert optimizer_args.adamw_lr_ema_rank == 2
    assert optimizer_args.adamw_lr_ema_seed == 7
    assert optimizer_args.adamw_lr_ema_confidence_beta == 0.99
    assert optimizer_args.adamw_lr_ema_confidence_alpha == 1e-3
    assert optimizer_args.adamw_lrsf_projection_refresh_state == "transport"
    assert optimizer_args.adamw_lrsf_projection_refresh["diagnostics"] is False
    assert optimizer_args.adamw_lrsf_projection_refresh["transport_overlap"] is None

    assert optimizer_args.apollo_norm_growth_rate == 1.01
    assert optimizer_args.apollo_scale == 1.0
    assert optimizer_args.apollo_disable_norm_growth_limiter is False

    input_ids = torch.randint(0, 31, (2, 8))
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }
    loss = _loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
