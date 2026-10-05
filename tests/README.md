# Test organization

The directory separates tests by scope:

- `unit/` covers individual modules and contracts with small tensors, synthetic inputs, or mocked dependencies. These tests should not require downloaded datasets or model weights.
- `integration/` covers interactions across training entrypoints, checkpoint/resume paths, and shared runtime components. Tests should use temporary directories and synthetic or patched data where possible.
- CUDA, Triton, and performance checks belong in `verify/` or `benchmarks/`; they are not part of the default CPU CI suite.

Within `unit/`, optimizer tests are grouped by optimizer family or backend. Text-LM optimizer probe tests are separated into CLI/model setup, optimizer state diagnostics, and trajectory diagnostics. VFP-DiT model tests are grouped into model primitives, training, and sampling/checkpoint behavior.

`tests/conftest.py` assigns the `unit` or `integration` pytest marker from the first directory under `tests/`. This keeps the filesystem layout and marker filters consistent without repeating a marker in every test module.

Run either suite or the full CPU suite from the repository root:

```bash
PYTHONPATH=. python3 -m pytest -q -m unit
PYTHONPATH=. python3 -m pytest -q -m integration
PYTHONPATH=. python3 -m pytest -q -m "(unit or integration) and not cuda and not triton and not benchmark"
```

The project-wide type checker currently includes `tests/unit` and excludes `tests/integration`, as configured in `pyrightconfig.json`.
