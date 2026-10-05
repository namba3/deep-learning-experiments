import json

from verify.text_lm_pretraining_report import collect_runs, summarize_runs


def test_text_lm_pretraining_report_collects_last_epoch_and_groups_seeds(tmp_path):
    run_dir = tmp_path / "runs" / "text_lm.train_tiny-seed-0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({
            "dataset_name": "roneneldan/TinyStories",
            "architecture": "naive",
            "max_seq_len": 128,
            "seed": 0,
        }),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        "\n".join([
            json.dumps({
                "event": "run_started",
                "config_path": str(run_dir / "config.json"),
            }),
            json.dumps({"event": "preflight", "device": "cpu", "dtype": "torch.float32"}),
            json.dumps({
                "event": "epoch", "epoch": 1, "model_parameters": 10,
                "trainable_parameters": 10, "train_loss": 4.0,
                "train_hard_loss": 3.0, "train_soft_loss": 2.0,
                "train_kl_divergence": 0.5, "eval_loss": 5.0,
                "eval_hard_loss": 4.0, "eval_kl_divergence": 0.25,
                "train_ppl": 20.0855, "eval_ppl": 54.5982,
                "steps_per_second": 2.0,
                "train_tokens": 100, "eval_tokens": 20,
            }),
            json.dumps({"event": "run_finished", "status": "completed"}),
        ]) + "\n",
        encoding="utf-8",
    )

    runs = collect_runs(tmp_path)
    summary = summarize_runs(runs)

    assert runs[0]["eval_loss"] == 5.0
    assert runs[0]["eval_ppl"] == 54.5982
    assert runs[0]["step_time_sec"] == 0.5
    assert summary[0]["eval_loss_mean"] == 5.0
    assert summary[0]["max_seq_len"] == 128
    assert summary[0]["run_count"] == 1
    assert summary[0]["seeds"] == [0]
