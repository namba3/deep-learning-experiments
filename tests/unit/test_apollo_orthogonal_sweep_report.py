import json

from verify.apollo_orthogonal_sweep_report import load_rows, render


def _write_result(path, *, loss, step, variance, status="passed"):
    cases = {
        "APOLLO": {
            "optimizer": "APOLLO",
            "final_validation_loss": loss,
            "host_seconds_per_optimizer_step": step,
            "update_norm_variance": variance,
            "orthogonal_refresh_steps": 4,
            "status": status,
        },
        "APOLLO-CAME": {
            "optimizer": "APOLLO-CAME",
            "final_validation_loss": loss + 1.0,
            "host_seconds_per_optimizer_step": step,
            "update_norm_variance": variance,
            "orthogonal_refresh_steps": 4,
            "status": status,
        },
        "APOLLO-Mini": {
            "optimizer": "APOLLO-Mini",
            "final_validation_loss": loss + 2.0,
            "host_seconds_per_optimizer_step": step,
            "update_norm_variance": variance,
            "orthogonal_refresh_steps": 4,
            "status": status,
        },
    }
    path.write_text(json.dumps({"status": status, "cases": cases}))


def test_load_rows_pairs_none_and_orthogonal_results(tmp_path):
    cell = tmp_path / "random" / "rate-0p01" / "seed-0"
    cell.mkdir(parents=True)
    _write_result(
        cell / "apollo-refresh-none-seed0.json",
        loss=1.0, step=2.0, variance=3.0,
    )
    _write_result(
        cell / "apollo-refresh-orthogonal-seed0.json",
        loss=0.75, step=3.0, variance=12.0,
    )

    rows = load_rows(tmp_path)

    assert len(rows) == 3
    assert rows[0]["direction"] == "random"
    assert rows[0]["rate"] == "0.01"
    assert rows[0]["loss_delta"] == -0.25
    assert rows[0]["step_ratio"] == 1.5
    assert rows[0]["variance_ratio"] == 4.0

    markdown = render(rows)
    assert "| random | 0.01 | 0 | APOLLO |" in markdown
    assert markdown.count("| passed |") == 3
