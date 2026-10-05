from pathlib import Path

import pytest


_TEST_ROOT = Path(__file__).resolve().parent
_SUITE_MARKERS = {
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration,
}


def pytest_collection_modifyitems(items):
    """Tag tests according to their top-level suite directory."""
    for item in items:
        relative_path = Path(str(item.path)).resolve().relative_to(_TEST_ROOT)
        if relative_path.parts:
            marker = _SUITE_MARKERS.get(relative_path.parts[0])
            if marker is not None:
                item.add_marker(marker)
