import json

import pytest

from verify.lrsf_gpu_report import aggregate_rows, load_rows, render_markdown


def _write_result(path, seed, loss, state, peak_state):
    path.write_text(json.dumps({
        "seed": seed,
        "refresh": {
            "mode": "smooth",
            "mix": "ema",
            "orthogonal_rate": 0.0,
        },
        "cases": {
            "APOLLO-CAME-LRSF": {
                "status": "passed",
                "optimizer": "APOLLO-CAME-LRSF",
                "rank": 4,
                "refresh_mode": "smooth",
                "final_validation_loss": loss,
                "persistent_state_bytes": state,
                "peak_persistent_state_bytes": peak_state,
                "peak_allocated_bytes": 5000 + seed,
                "peak_reserved_bytes": 8000,
                "host_seconds_per_optimizer_step": 0.01 + seed / 1000,
            },
        },
    }), encoding="utf-8")


def test_report_expands_directory_and_aggregates_seed_statistics(tmp_path):
    _write_result(tmp_path / "lrsf-gpu-ema-seed0.json", 0, 0.10, 100, 120)
    _write_result(tmp_path / "lrsf-gpu-ema-seed1.json", 1, 0.14, 120, 160)

    rows = load_rows([tmp_path])
    aggregated = aggregate_rows(rows)

    assert len(rows) == 2
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["refresh"] == "smooth/ema"
    assert row["runs"] == 2
    assert row["seeds"] == [0, 1]
    assert row["validation_loss_mean"] == pytest.approx(0.12)
    assert row["validation_loss_std"] == pytest.approx(0.02)
    assert row["peak_state_bytes_mean"] == pytest.approx(140.0)
    assert "smooth/ema" in render_markdown(aggregated)


def test_report_keeps_loss_directed_signal_variants_separate(tmp_path):
    path = tmp_path / "lrsf-gpu-orthogonal-seed0.json"
    path.write_text(json.dumps({
        "seed": 0,
        "refresh": {
            "mode": "none",
            "orthogonal_rate": 0.01,
            "orthogonal_direction": "loss_directed",
            "orthogonal_signal": "effective_update",
        },
        "cases": {
            "CAME-LRSF": {
                "status": "passed",
                "optimizer": "CAME-LRSF",
                "rank": 4,
                "final_validation_loss": 0.1,
                "persistent_state_bytes": 100,
                "peak_persistent_state_bytes": 100,
                "peak_allocated_bytes": 200,
                "peak_reserved_bytes": 300,
                "host_seconds_per_optimizer_step": 0.01,
                "orthogonal_refresh_events": [
                    {"loss_lowering_proxy_decrease": 0.25},
                ],
            },
        },
    }), encoding="utf-8")

    rows = aggregate_rows(load_rows([path]))

    assert rows[0]["refresh"] == (
        "orthogonal(0.01)/loss_directed/effective_update"
    )


def test_report_aggregates_loss_lowering_proxy_decrease(tmp_path):
    path = tmp_path / "lrsf-gpu-orthogonal-seed0.json"
    path.write_text(json.dumps({
        "seed": 0,
        "refresh": {
            "mode": "none",
            "orthogonal_rate": 0.001,
            "orthogonal_direction": "loss_lowering",
        },
        "cases": {
            "CAME-LRSF": {
                "status": "passed",
                "optimizer": "CAME-LRSF",
                "rank": 4,
                "final_validation_loss": 0.1,
                "persistent_state_bytes": 100,
                "orthogonal_refresh_events": [
                    {"loss_lowering_proxy_decrease": 0.5},
                    {"loss_lowering_proxy_decrease": 1.0},
                ],
            },
        },
    }), encoding="utf-8")

    rows = aggregate_rows(load_rows([path]))

    assert rows[0]["loss_lowering_proxy_decrease_mean"] == pytest.approx(0.75)
    assert "loss-lowering proxy decrease" in render_markdown(rows)


def test_report_computes_paired_loss_delta_against_fixed(tmp_path):
    for mode, loss in (("fixed", 0.10), ("orthogonal", 0.08)):
        path = tmp_path / f"lrsf-gpu-{mode}-seed0.json"
        path.write_text(json.dumps({
            "seed": 0,
            "refresh": {
                "mode": "none",
                "orthogonal_rate": 0.001 if mode == "orthogonal" else 0.0,
                "orthogonal_direction": "loss_lowering",
            },
            "cases": {
                "CAME-LRSF": {
                    "status": "passed",
                    "optimizer": "CAME-LRSF",
                    "rank": 4,
                    "final_validation_loss": loss,
                    "persistent_state_bytes": 100,
                },
            },
        }), encoding="utf-8")

    rows = aggregate_rows(load_rows([tmp_path]))
    orthogonal = next(row for row in rows if row["refresh"] != "fixed/frozen")

    assert orthogonal["paired_loss_delta_mean"] == pytest.approx(-0.02)
    assert orthogonal["paired_improved_count"] == 1
    assert orthogonal["paired_comparison_count"] == 1
    assert "-0.02" in render_markdown(rows)
