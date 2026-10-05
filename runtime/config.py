"""Configuration helpers shared by training entry points."""

import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from safetensors import safe_open

CONFIG_SCHEMA_VERSION = 1


def cli_option_provided(argv: Sequence[str], *options: str) -> bool:
    """Return whether one of ``options`` was explicitly present in ``argv``.

    Both ``--option value`` and ``--option=value`` forms are recognized.  The
    caller should include aliases such as ``--foo`` and ``--no-foo`` for
    boolean options when both forms are valid.
    """
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in argv
        for option in options
    )


def apply_saved_config(
    args: Any,
    saved_config: Mapping[str, Any] | None,
    option_names: Mapping[str, Sequence[str]] | None = None,
    *,
    argv: Sequence[str] | None = None,
    keys: Sequence[str] | None = None,
    overridden_keys: list[str] | None = None,
) -> list[str]:
    """Apply checkpoint settings unless the user explicitly overrides them.

    ``option_names`` maps config keys to their CLI spellings.  Unmapped keys
    use the conventional ``snake_case`` to ``--kebab-case`` conversion.  Only
    attributes already present on ``args`` are considered, so metadata from a
    newer training script remains forward-compatible with older scripts.
    """
    if not saved_config:
        return []
    argv = sys.argv[1:] if argv is None else argv
    option_names = option_names or {}
    selected_keys = saved_config.keys() if keys is None else keys
    restored: list[str] = []
    for key in selected_keys:
        if key not in saved_config or not hasattr(args, key):
            continue
        options = option_names.get(key, (f"--{key.replace('_', '-')}",))
        if cli_option_provided(argv, *options):
            if overridden_keys is not None:
                overridden_keys.append(key)
            continue
        setattr(args, key, saved_config[key])
        restored.append(key)
    return restored


def checkpoint_config_metadata(args: Any, key: str) -> dict[str, str]:
    """Serialize an argparse namespace as safetensors checkpoint metadata."""
    return {
        key: json.dumps(
            {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "args": vars(args),
            },
            sort_keys=True,
        ),
    }


def validate_config_schema(
    config: Mapping[str, Any], *, key: str, path: str,
) -> int:
    """Validate a config schema and return its version.

    Missing versions are treated as legacy version 0.  Newer versions are
    rejected instead of being partially applied with potentially wrong
    defaults.
    """
    version = config.get("config_schema_version", config.get("schema_version", 0))
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"Invalid schema version in {key} metadata in {path}")
    if version > CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{key} metadata in {path} uses unsupported schema version {version}; "
            f"current version is {CONFIG_SCHEMA_VERSION}. Upgrade this code or "
            "use a checkpoint saved by a compatible training script."
        )
    return version


def read_checkpoint_config(path: str, key: str) -> dict[str, Any] | None:
    """Read a JSON config metadata entry, returning ``None`` when absent."""
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    value = metadata.get(key)
    if not value:
        return None
    try:
        config = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {key} metadata in {path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Expected object in {key} metadata in {path}")
    if "args" not in config:
        # Checkpoints created before schema versioning stored argparse values
        # directly. Keep those checkpoints readable as legacy version 0.
        validate_config_schema(config, key=key, path=path)
        return config
    validate_config_schema(config, key=key, path=path)
    args = config["args"]
    if not isinstance(args, dict):
        raise ValueError(f"Expected args object in {key} metadata in {path}")
    return args
