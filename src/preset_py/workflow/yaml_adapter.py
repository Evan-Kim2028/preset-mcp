from pathlib import Path
from typing import Any

import yaml


def read_yaml_dashboard(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dashboard YAML must contain a mapping: {path}")
    return payload


def write_yaml_dashboard(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
