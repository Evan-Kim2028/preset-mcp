import re
from pathlib import Path
from typing import Any

import yaml


_PRESET_ROOT = Path(__file__).resolve().parent.parent / "presets"
_PRESET_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"preset file must contain a mapping: {path}")
    return payload


def _validate_preset_name(name: str) -> str:
    if not _PRESET_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid preset name: {name}")
    return name


def load_layout_preset(name: str) -> dict[str, Any]:
    name = _validate_preset_name(name)
    return _load_yaml(_PRESET_ROOT / "layouts" / f"{name}.yaml")


def load_chart_preset(name: str) -> dict[str, Any]:
    name = _validate_preset_name(name)
    return _load_yaml(_PRESET_ROOT / "charts" / f"{name}.yaml")
