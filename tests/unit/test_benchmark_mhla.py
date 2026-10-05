import argparse

import pytest

from image_gen.benchmark_mhla import parse_patterns


def test_parse_patterns_accepts_training_pattern():
    assert parse_patterns("full,mhla3-full1") == ("full", "mhla3-full1")


def test_parse_patterns_accepts_all_supported_patterns():
    assert parse_patterns("full,mhla,mhla3-full1") == (
        "full", "mhla", "mhla3-full1",
    )


@pytest.mark.parametrize("value", ["", "unknown", "full,unknown"])
def test_parse_patterns_rejects_unknown_patterns(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_patterns(value)
