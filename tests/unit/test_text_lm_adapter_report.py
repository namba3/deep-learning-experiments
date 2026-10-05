import json

from verify.text_lm_adapter_report import collect_runs, summarize_runs


def test_text_lm_adapter_report_groups_context_and_adapter(tmp_path):
    run_dir = tmp_path / "runs" / "text_lm.train_adapter_lora-ctx-512-seed-0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({
            "dataset_name": "roneneldan/TinyStories",
            "adapter": "lora",
            "lora_rank": 16,
            "lora_alpha": 16.0,
            "max_seq_len": 512,
            "batch_size": 4,
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
                "event": "epoch", "epoch": 3, "model_parameters": 100,
                "trainable_parameters": 12, "train_loss": 4.0,
                "eval_loss": 3.0, "train_ppl": 54.6, "eval_ppl": 20.1,
                "steps_per_second": 2.0, "train_tokens": 1024,
                "eval_tokens": 256,
            }),
            json.dumps({"event": "run_finished", "status": "completed"}),
        ]) + "\n",
        encoding="utf-8",
    )

    runs = collect_runs(tmp_path)
    summary = summarize_runs(runs)

    assert runs[0]["adapter"] == "lora"
    assert runs[0]["rank"] == 16
    assert runs[0]["tokens_per_step"] == 2048
    assert summary[0]["rank"] == 16
    assert summary[0]["max_seq_len"] == 512
    assert summary[0]["tokens_per_step"] == 2048
    assert summary[0]["eval_loss_mean"] == 3.0
    assert summary[0]["run_count"] == 1
