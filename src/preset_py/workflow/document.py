from pathlib import Path
from typing import Any

import yaml

from preset_py.workflow.types import DashboardDocument


def load_dashboard_yaml_document(path: Path) -> DashboardDocument:
    payload: dict[str, Any] = yaml.safe_load(path.read_text())
    position = payload.get("position") or {}
    metadata = payload.get("metadata") or {}
    mode = (
        "tabbed"
        if any(
            isinstance(node, dict) and node.get("type") == "TABS"
            for node in position.values()
        )
        else "flat"
    )
    return DashboardDocument(
        source_path=path,
        dashboard_title=str(payload.get("dashboard_title") or ""),
        mode=mode,
        position=position,
        metadata=metadata,
    )
