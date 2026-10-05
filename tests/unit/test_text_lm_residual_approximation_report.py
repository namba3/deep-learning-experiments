import json

from verify.text_lm_residual_approximation_report import build_report


def test_residual_approximation_report_aggregates_methods(tmp_path):
    result = {
        "cases": [{
            "optimizer": "AdamW-SF",
            "seed": 0,
            "final_validation_loss": 12.0,
            "update_trajectory_pca": [{
                "parameter": "layer.weight",
                "rolling": {
                    "residual_compression_block_size": 128,
                    "residual_compression_scale_mode": "percentile_99_9",
                    "residual_approximation": {
                        "8": {
                            "samples": 4,
                            "low_rank_factor": {
                                "4": {
                                    "storage_ratio_to_target_bf16": 0.5,
                                    "update_cosine": 0.9,
                                    "update_norm_ratio": 1.0,
                                    "decode_milliseconds": 2.0,
                                },
                            },
                            "blockwise_int8": {
                                "storage_ratio_to_target_bf16": 0.51,
                                "update_cosine": 0.99,
                                "update_norm_ratio": 1.0,
                                "decode_milliseconds": 1.0,
                            },
                            "blockwise_int4": {
                                "storage_ratio_to_target_bf16": 0.26,
                                "update_cosine": 0.98,
                                "update_norm_ratio": 1.0,
                                "decode_milliseconds": 0.8,
                            },
                            "blockwise_int4_error_feedback": {
                                "storage_ratio_to_target_bf16": 0.26,
                                "feedback_storage_ratio_to_target_bf16": 2.0,
                                "update_cosine": 0.99,
                                "update_norm_ratio": 1.0,
                                "decode_milliseconds": 0.9,
                                "block_size": 128,
                                "scale_mode": "percentile_99_9",
                            },
                            "low_rank_plus_int8": {
                                "4": {
                                    "storage_ratio_to_target_bf16": 0.52,
                                    "update_cosine": 0.995,
                                    "update_norm_ratio": 1.0,
                                    "decode_milliseconds": 2.5,
                                },
                            },
                        },
                    },
                },
            }],
        }],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    report = build_report(path)

    assert "AdamW-SF" in report
    assert "low_rank_factor" in report
    assert "blockwise_int8" in report
    assert "blockwise_int4" in report
    assert "blockwise_int4_error_feedback" in report
    assert "low_rank_plus_int8" in report
    assert "0.9" in report
    assert "0.99" in report
    assert "| 128 |" in report
    assert "percentile_99_9" in report
    assert "feedback / target BF16" in report
    assert "| 2 |" in report
    assert "trajectory samples" in report
    assert "| 4 |" in report
