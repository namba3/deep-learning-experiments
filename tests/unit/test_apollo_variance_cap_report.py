import json

import pytest

from verify.apollo_variance_cap_report import load_rows, render


def _write_result(path, *, loss, step, status="passed", capped=0):
    cases = {
        optimizer: {
            "optimizer": optimizer,
            "final_validation_loss": loss + offset,
            "host_seconds_per_optimizer_step": step,
            "persistent_state_bytes": 100 + offset,
            "peak_allocated_bytes": 1000 + offset,
            "update_norm_variance_capped_steps": capped,
            "update_norm_variance": 0.1 + offset,
            "status": status,
        }
        for optimizer, offset in (
            ("APOLLO", 0.0),
            ("APOLLO-CAME", 1.0),
            ("APOLLO-Mini", 2.0),
        )
    }
    path.write_text(json.dumps({"status": status, "cases": cases}))


def test_load_rows_compares_each_cap_to_same_seed_baseline(tmp_path):
    for cap in ("none", "0p001"):
        cell = tmp_path / f"cap-{cap}" / "seed-0"
        cell.mkdir(parents=True)
        _write_result(
            cell / "apollo-refresh-none-seed0.json",
            loss=1.0 if cap == "none" else 0.9,
            step=2.0 if cap == "none" else 3.0,
            capped=0 if cap == "none" else 2,
        )

    rows = load_rows(tmp_path)

    assert len(rows) == 6
    capped_row = next(row for row in rows if row["cap"] == "0.001")
    assert capped_row["loss_delta"] == pytest.approx(-0.1)
    assert capped_row["step_ratio"] == pytest.approx(1.5)
    assert capped_row["capped_steps"] == 2

    markdown = render(rows)
    assert "| 0.001 | 0 | APOLLO |" in markdown
    assert markdown.count("| passed |") == 6
