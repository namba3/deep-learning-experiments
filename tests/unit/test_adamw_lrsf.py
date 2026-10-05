import torch

from optimizers import AdamWLRSF


def test_adamw_lrsf_shadow_refresh_updates_both_low_rank_branches():
    torch.manual_seed(43)
    parameter = torch.nn.Parameter(torch.randn(16, 8))
    optimizer = AdamWLRSF(
        [parameter],
        lr=1e-3,
        rank=4,
        backend="torch",
        projection_refresh={
            "mode": "shadow",
            "interval": 2,
            "diagnostics": True,
        },
    )
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["shadow_active"] is True
    assert not torch.equal(
        state["lrsf_projection"], state["lrsf_shadow_projection"]
    )
    assert torch.linalg.vector_norm(state["lrsf_delta"]) > 0
    assert torch.linalg.vector_norm(state["lrsf_shadow_delta"]) > 0

    active_projection = state["lrsf_projection"].clone()
    shadow_projection = state["lrsf_shadow_projection"].clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    torch.testing.assert_close(state["lrsf_projection"], shadow_projection)
    assert not torch.equal(state["lrsf_projection"], active_projection)
    assert state["refresh_count"] == 1
    assert state["shadow_gap_count"] == 1
    assert torch.linalg.vector_norm(state["lrsf_shadow_delta"]) > 0
