"""Workspace snapshot: capture a full inventory of dashboards, charts, datasets, and databases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from preset_py.client import PresetWorkspace


class WorkspaceSnapshot(BaseModel):
    """Point-in-time inventory of a Preset workspace."""

    workspace_hostname: str
    snapshot_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dashboards: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    databases: list[dict[str, Any]] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def counts(self) -> dict[str, int]:
        return {
            "dashboards": len(self.dashboards),
            "charts": len(self.charts),
            "datasets": len(self.datasets),
            "databases": len(self.databases),
        }


def take_snapshot(ws: PresetWorkspace) -> WorkspaceSnapshot:
    """Capture a full workspace inventory."""
    return WorkspaceSnapshot(
        workspace_hostname=str(ws.workspace_url),
        dashboards=ws.dashboards(),
        charts=ws.charts(),
        datasets=ws.datasets(),
        databases=ws.databases(),
    )


def save_snapshot(snapshot: WorkspaceSnapshot, output_dir: str | Path) -> Path:
    """Write snapshot data to individual JSON files on disk.

    Creates:
        output_dir/snapshot_meta.json
        output_dir/dashboards.json
        output_dir/charts.json
        output_dir/datasets.json
        output_dir/databases.json

    Returns the output directory as a Path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "workspace_hostname": snapshot.workspace_hostname,
        "snapshot_time": snapshot.snapshot_time.isoformat(),
        "counts": snapshot.counts,
    }
    _write_json(out / "snapshot_meta.json", meta)
    _write_json(out / "dashboards.json", snapshot.dashboards)
    _write_json(out / "charts.json", snapshot.charts)
    _write_json(out / "datasets.json", snapshot.datasets)
    _write_json(out / "databases.json", snapshot.databases)

    return out


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
