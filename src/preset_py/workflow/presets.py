from pathlib import Path
from typing import Any

import yaml


_PRESET_ROOT = Path(__file__).resolve().parent.parent / "presets"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def load_layout_preset(name: str) -> dict[str, Any]:
    return _load_yaml(_PRESET_ROOT / "layouts" / f"{name}.yaml")


def load_chart_preset(name: str) -> dict[str, Any]:
    return _load_yaml(_PRESET_ROOT / "charts" / f"{name}.yaml")
