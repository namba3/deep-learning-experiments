import pytest

from benchmarks.benchmark_text_lm_architectures import parse_seeds, summarize_results


def test_parse_seeds_uses_fallback_and_rejects_duplicates():
    assert parse_seeds(None, 42) == (42,)
    assert parse_seeds("1, 3,5", 42) == (1, 3, 5)
    with pytest.raises(ValueError, match="duplicates"):
        parse_seeds("1,1", 42)


def test_summarize_results_handles_multiple_seeds_and_null_metrics():
    results = [
        {
            "architecture": "looped",
            "seed": 1,
            "parameters": 10,
            "forward_median_ms": 2.0,
            "cuda_peak_allocated_mib": None,
        },
        {
            "architecture": "looped",
            "seed": 2,
            "parameters": 10,
            "forward_median_ms": 4.0,
            "cuda_peak_allocated_mib": None,
        },
    ]

    summary = summarize_results(results, ("forward_median_ms", "cuda_peak_allocated_mib"))

    assert summary == [{
        "architecture": "looped",
        "seeds": [1, 2],
        "parameters": 10,
        "forward_median_ms_mean": 3.0,
        "forward_median_ms_std": pytest.approx(2 ** 0.5),
        "cuda_peak_allocated_mib_mean": None,
        "cuda_peak_allocated_mib_std": None,
    }]
