import torch

from optimizers.lion import Lion

from optimizers.muon import SingleDeviceMuon

def _assign_gradient(parameter, value=1.0):
    parameter.grad = torch.full_like(parameter, value)

def test_lion_matches_manual_momentum_sign_and_decoupled_decay():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = Lion(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.99),
        weight_decay=0.2,
        backend="torch",
    )
    gradient = torch.tensor([0.5, -1.0])
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = gradient.clone()
    optimizer.step()

    expected_momentum = gradient * (1.0 - 0.99)
    expected_parameter = (
        torch.tensor([1.0, -2.0]) * (1.0 - 0.1 * 0.2)
        - 0.1 * expected_momentum.sign()
    )
    assert torch.equal(parameter, expected_parameter)
    assert torch.allclose(
        optimizer.state[parameter]["exp_avg"], expected_momentum,
        atol=1e-7, rtol=0.0,
    )

def test_single_device_muon_skips_parameters_without_gradients():
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = SingleDeviceMuon(
        [parameter], lr=0.01, weight_decay=0.1,
        momentum=0.9, backend="torch",
    )
    _assign_gradient(parameter)
    optimizer.step()
    before = parameter.detach().clone()
    momentum_before = optimizer.state[parameter]["momentum_buffer"].clone()

    parameter.grad = None
    optimizer.step()

    assert torch.equal(parameter, before)
    assert torch.equal(
        optimizer.state[parameter]["momentum_buffer"], momentum_before,
    )

def test_lion_updates_bfloat16_parameters_with_parameter_dtype_momentum():
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = Lion([parameter], lr=0.01, weight_decay=0.0)
    _assign_gradient(parameter)
    before = parameter.detach().clone()
    optimizer.step()

    assert parameter.dtype == torch.bfloat16
    assert not torch.equal(before, parameter)
    assert optimizer.state[parameter]["exp_avg"].dtype == parameter.dtype
    assert "exp_avg_sq" not in optimizer.state[parameter]
