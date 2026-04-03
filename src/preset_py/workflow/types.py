from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DashboardDocument:
    source_path: Path
    dashboard_title: str
    mode: str
    position: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class WorkflowPlan:
    status: str
    target: dict[str, Any]
    summary: str
    changes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
