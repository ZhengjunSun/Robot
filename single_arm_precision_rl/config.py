from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "single_arm_precision_alignment.yaml"


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    base_path = data.pop("extends", None)
    if base_path:
        base = load_config(base_path)
        base.pop("_config_path", None)
        base.pop("_base_config_path", None)
        data = _deep_merge(base, data)
        data["_base_config_path"] = str(resolve_project_path(base_path))
    data["_config_path"] = str(config_path)
    return data
