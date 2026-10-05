import torch

from optimizers import APOLLOScheduleFree
from optimizers.factory import build_optimizer


STORAGE_MODES = (
    "bf16_z",
    "blockwise_int8_z",
    "blockwise_int4_z",
    "blockwise_int8_delta",
    "blockwise_int4_delta",
    "low_rank_delta",
)


def _state_bytes(optimizer):
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def test_apollo_schedule_free_storage_modes_step_and_round_trip():
    sizes = {}
    for mode in STORAGE_MODES:
        torch.manual_seed(7)
        parameter = torch.nn.Parameter(torch.randn(8, 12))
        optimizer = APOLLOScheduleFree(
            [parameter],
            lr=1e-3,
            rank=2,
            sf_state_storage=mode,
            sf_quant_block_size=16,
            update_proj_gap=1000,
        )
        optimizer.train()
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        train_value = parameter.detach().clone()
        optimizer.eval()
        optimizer.train()

        assert torch.isfinite(parameter).all()
        assert torch.allclose(parameter, train_value, atol=2e-3, rtol=2e-3)
        sizes[mode] = _state_bytes(optimizer)

    assert sizes["blockwise_int8_z"] < sizes["bf16_z"]
    assert sizes["blockwise_int8_delta"] < sizes["bf16_z"]
    assert sizes["blockwise_int4_z"] < sizes["blockwise_int8_z"]
    assert sizes["blockwise_int4_delta"] < sizes["blockwise_int8_delta"]


def test_low_rank_delta_uses_latent_state_for_apollo_matrix():
    torch.manual_seed(9)
    parameter = torch.nn.Parameter(torch.randn(16, 32))
    optimizer = APOLLOScheduleFree(
        [parameter],
        lr=1e-3,
        rank=4,
        sf_state_storage="low_rank_delta",
        update_proj_gap=1,
    )
    optimizer.train()
    for _ in range(2):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    assert tuple(state["sf_delta_latent"].shape) == (4, 32)
    assert "sf_delta_full" not in state
    assert state["step"] == 2
    assert torch.isfinite(parameter).all()


def test_low_rank_delta_uses_orthonormal_basis_without_reconstruction_growth():
    torch.manual_seed(17)
    parameter = torch.nn.Parameter(torch.randn(96, 256))
    initial_norm = float(parameter.detach().norm())
    optimizer = APOLLOScheduleFree(
        [parameter],
        lr=1e-3,
        rank=32,
        sf_state_storage="low_rank_delta",
        update_proj_gap=200,
    )
    optimizer.train()
    for _ in range(20):
        parameter.grad = parameter.detach().clone()
        optimizer.step()

    state = optimizer.state[parameter]
    basis = state["sf_delta_projection"]
    assert torch.allclose(
        basis @ basis.transpose(0, 1),
        torch.eye(basis.shape[0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(parameter).all()
    assert float(parameter.norm()) < initial_norm * 2.0


def test_factory_builds_all_apollo_schedule_free_variants():
    names = (
        "APOLLO-SF",
        "APOLLO-SF-LRSF",
        "APOLLO-SF-INT8-Z",
        "APOLLO-SF-INT8-Delta",
        "APOLLO-SF-INT4-Z",
        "APOLLO-SF-INT4-Delta",
    )
    for name in names:
        parameter = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = build_optimizer(
            name,
            [parameter],
            args=type(
                "Args", (), {"rank": 2, "seed": 0, "apollo_sf_quant_block_size": 16}
            )(),
            lr=1e-3,
            weight_decay=0.0,
        )
        parameter.grad = torch.ones_like(parameter)
        optimizer.train()
        optimizer.step()
        assert torch.isfinite(parameter).all()


def test_delta_commit_merges_hidden_z_at_projection_refresh():
    torch.manual_seed(11)
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = APOLLOScheduleFree(
        [parameter],
        lr=1e-3,
        rank=2,
        sf_state_storage="blockwise_int8_delta",
        sf_delta_refresh="commit_z",
        sf_quant_block_size=16,
        update_proj_gap=1,
    )
    optimizer.train()
    for _ in range(2):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    group = optimizer.param_groups[0]
    state = optimizer.state[parameter]
    assert group["sf_delta_commit_count"] == 1
    decoded_delta = optimizer._decode_state(parameter, state, group)
    assert torch.allclose(decoded_delta, torch.zeros_like(decoded_delta))
    assert torch.isfinite(parameter).all()


def test_delta_blend_reaches_zero_without_changing_hidden_z():
    torch.manual_seed(13)
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = APOLLOScheduleFree(
        [parameter],
        lr=1e-3,
        rank=2,
        sf_state_storage="blockwise_int8_delta",
        sf_delta_refresh="blend",
        sf_delta_refresh_window=2,
        sf_quant_block_size=16,
        update_proj_gap=2,
    )
    optimizer.train()
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    group = optimizer.param_groups[0]
    state = optimizer.state[parameter]
    assert group["sf_delta_blend_count"] == 1
    assert group["sf_delta_blend_step_count"] == 2
    decoded_delta = optimizer._decode_state(parameter, state, group)
    assert torch.allclose(decoded_delta, torch.zeros_like(decoded_delta))
    assert torch.isfinite(parameter).all()
