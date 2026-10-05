import pytest
import torch

from optimizers.update_norm import (
    cap_update_norm_variance,
    cap_update_norm_variance_scale,
)


def test_update_norm_variance_cap_limits_large_upper_tail_sample():
    state = {}
    first, first_capped = cap_update_norm_variance(
        torch.tensor([1.0]), state, max_variance=1.0,
    )
    second, second_capped = cap_update_norm_variance(
        torch.tensor([10.0]), state, max_variance=1.0,
    )

    assert first.item() == pytest.approx(1.0)
    assert first_capped is False
    assert second.item() == pytest.approx(3.0)
    assert second_capped is True
    assert state["update_norm_variance_count"] == 2
    assert state["update_norm_variance_capped_count"].item() == 1
    assert state["update_norm_variance_m2"].item() == pytest.approx(2.0)


def test_update_norm_variance_cap_does_not_change_history_when_disabled():
    update = torch.tensor([2.0, -1.0])
    state = {}
    result, capped = cap_update_norm_variance(update, state, 0.0)

    assert capped is False
    torch.testing.assert_close(result, update)
    assert state["update_norm_variance_count"] == 1


def test_update_norm_variance_cap_returns_scalar_scale_without_materializing_update():
    state = {}
    first_scale, first_capped = cap_update_norm_variance_scale(
        torch.tensor([1.0]), state, max_variance=1.0, scale=2.0,
    )
    second_scale, second_capped = cap_update_norm_variance_scale(
        torch.tensor([10.0]), state, max_variance=1.0, scale=2.0,
    )

    assert first_scale.item() == pytest.approx(2.0)
    assert not bool(first_capped)
    assert bool(second_capped)
    assert (torch.tensor([10.0]) * second_scale).norm().item() == pytest.approx(4.0)
