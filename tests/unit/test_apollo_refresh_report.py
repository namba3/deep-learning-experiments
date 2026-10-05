import json

from verify.apollo_refresh_report import (
    _refresh_label,
    load_rows,
    main,
    render_markdown,
)


def test_report_supports_imageae_and_runtime_result_shapes(tmp_path):
    imageae = tmp_path / "smooth-ema.json"
    imageae.write_text(json.dumps({
        "status": "passed",
        "cases": {
            "APOLLO": {
                "status": "passed",
                "optimizer": "APOLLO",
                "rank": 4,
                "projection_refresh_mode": "smooth",
                "projection_refresh_mix": "ema",
                "projection_refresh_state": "transport",
                "final_validation_loss": 0.25,
                "persistent_state_bytes": 128,
                "host_seconds_per_optimizer_step": 0.01,
                "projection_refresh_steps": 2,
                "projection_refresh_active_steps": 4,
                "orthogonal_refresh_steps": 8,
                "update_norm_mean": 0.125,
                "update_norm_variance": 0.0625,
                "projection_refresh_events": [
                    {"projection_change_max_abs": 0.5},
                ],
            },
        },
    }), encoding="utf-8")
    runtime = tmp_path / "orthogonal.json"
    runtime.write_text(json.dumps({
        "status": "passed",
        "cases": {
            "APOLLO@rank=1:16x16": {
                "optimizer": "APOLLO@rank=1",
                "rank": 1,
                "projection_refresh_mode": "none",
                "orthogonal_refresh_rate": 0.01,
                "persistent_state_bytes": 64,
                "cuda_seconds_per_step": 0.02,
                "peak_allocated_bytes": 1024,
                "peak_reserved_bytes": 2048,
                "orthogonal_refresh_steps": 3,
            },
        },
    }), encoding="utf-8")

    rows = load_rows([imageae, runtime])

    assert rows[0]["refresh"] == "smooth/ema/transport"
    assert rows[0]["step_seconds"] == 0.01
    assert rows[0]["projection_change_max_abs"] == 0.5
    assert rows[0]["update_norm_mean"] == 0.125
    assert rows[0]["update_norm_variance"] == 0.0625
    assert rows[1]["refresh"] == "none/orth=0.01"
    assert rows[1]["step_seconds"] == 0.02
    table = render_markdown(rows)
    assert "smooth/ema/transport" in table
    assert "| smooth-ema.json | APOLLO |" in table
    assert _refresh_label({"projection_mode": "frozen"}) == "frozen"
    assert "orthogonal.json" in table


def test_report_keeps_missing_cuda_metrics_visible(tmp_path):
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps({"cases": {"CAME": {"optimizer": "CAME"}}}), encoding="utf-8")

    rows = load_rows([path])

    assert rows[0]["peak_allocated_bytes"] is None
    assert "| - |" in render_markdown(rows)


def test_report_returns_actionable_error_for_missing_result(capsys, tmp_path):
    missing = tmp_path / "not-generated.json"

    assert main([str(missing)]) == 2
    assert "run the corresponding probe first" in capsys.readouterr().err
