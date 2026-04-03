from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DashboardDocument:
    source_path: Path
    dashboard_title: str
    mode: str
    position: dict[str, Any]
    metadata: dict[str, Any]
