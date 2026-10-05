
import torch

from flow_sampling import available_solvers
from vfp_dit.samplers import SOLVERS, sample_flow_matching
from vfp_dit.schedulers import build_timesteps as build_vfp_timesteps


def test_vfp_adapter_routes_fixed_grid_solvers_with_batched_times():
    initial = torch.zeros(2, 3, 4, 6)
    times_seen = []

    def field(state, time_batch):
        times_seen.append(time_batch)
        return torch.full_like(state, 0.25)

    for solver in ("euler", "fireflow", "abm2"):
        times_seen.clear()
        result = sample_flow_matching(
            field, initial, steps=6, solver=solver, scheduler="uniform",
        )
        assert result.shape == initial.shape
        assert torch.allclose(result, torch.full_like(initial, -0.25), atol=1e-6)
        assert all(times.shape == (2,) for times in times_seen)


def test_vfp_adapter_preserves_er_sde_generator_contract():
    initial = torch.randn(1, 2, 3, 5)
    def field(state, times):
        return torch.zeros_like(state)

    first = sample_flow_matching(
        field, initial, steps=5, solver="er_sde",
        generator=torch.Generator().manual_seed(91),
    )
    second = sample_flow_matching(
        field, initial, steps=5, solver="er_sde",
        generator=torch.Generator().manual_seed(91),
    )

    assert torch.equal(first, second)
    assert "er_sde" in SOLVERS
    assert set(SOLVERS) == set(available_solvers())


def test_generation_cli_accepts_the_shared_solver_choices():
    from vfp_dit.generate_samples import parse_args

    for solver in SOLVERS:
        args = parse_args(["--checkpoint", "unused.safetensors", "--solver", solver])
        assert args.solver == solver


def test_vfp_scheduler_compatibility_wrapper_uses_shared_schedule():
    times = build_vfp_timesteps(4, scheduler="flow_match_euler", flow_shift=2.0)
    assert isinstance(times, tuple)
    assert len(times) == 5
    assert times[0] == 1.0 and times[-1] == 0.0
    assert all(left > right for left, right in zip(times[:-1], times[1:]))


def test_sampling_metadata_records_solver_grid_cfg_seeds_and_nfe():
    from vfp_dit.generate_samples import build_sampling_metadata

    metadata = build_sampling_metadata(
        solver="fireflow", scheduler="flow_match_euler", flow_shift=2.0,
        steps=10, guidance_scale=4.0, seed=42, num_samples=3,
        er_sde_sigma_max=80.0, output="samples.png", checkpoint="model.safetensors",
        prompts=["a", "b", "c"], reference_images=[None, None, None],
        resolution=512, device="cuda", dtype="bf16",
    )
    assert metadata["sample_seeds"] == [42, 43, 44]
    assert metadata["callback_evaluations_per_sample"] == 11
    assert metadata["callback_evaluations_total"] == 33
    assert metadata["cfg_network_forward_multiplier"] == 2
    assert metadata["steps"] == 10 and metadata["flow_shift"] == 2.0
