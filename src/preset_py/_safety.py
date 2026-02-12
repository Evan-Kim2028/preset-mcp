"""Safety guardrails: audit journal, pre-mutation snapshots, dependency checks."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from preset_py.client import PresetWorkspace

_log = logging.getLogger("preset-mcp")

# ---------------------------------------------------------------------------
# Audit directory (env-configurable)
# ---------------------------------------------------------------------------

AUDIT_DIR = Path(
    os.environ.get("PRESET_MCP_AUDIT_DIR", "~/.preset-mcp/audit/")
).expanduser()

# ---------------------------------------------------------------------------
# MutationEntry model
# ---------------------------------------------------------------------------


class MutationEntry(BaseModel):
    """Single entry in the mutation audit journal."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str
    resource_type: str  # "dashboard", "chart", "dataset"
    resource_id: int | None = None
    action: str  # "create" or "update"
    fields_changed: list[str] = Field(default_factory=list)
    before_snapshot: dict[str, Any] | None = None
    after_summary: dict[str, Any] | None = None
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Audit journal
# ---------------------------------------------------------------------------


def record_mutation(entry: MutationEntry) -> None:
    """Append a JSONL line to the audit journal. Fire-and-forget."""
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        journal = AUDIT_DIR / "mutations.jsonl"
        line = entry.model_dump_json() + "\n"
        with journal.open("a") as f:
            f.write(line)
    except Exception as exc:
        _log.warning("audit journal write failed: %s", exc)


# ---------------------------------------------------------------------------
# Pre-mutation snapshots
# ---------------------------------------------------------------------------


def capture_before(
    ws: PresetWorkspace,
    resource_type: str,
    resource_id: int,
) -> dict[str, Any]:
    """Fetch the current state of a resource before mutation.

    Also writes a snapshot file for manual recovery.
    Returns the full dict; on failure returns ``{"_snapshot_error": ...}``.
    """
    try:
        data = ws.get_resource(resource_type, resource_id)

        # Write individual snapshot file
        try:
            snap_dir = AUDIT_DIR / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            snap_file = snap_dir / f"{resource_type}_{resource_id}_{ts}.json"
            snap_file.write_text(json.dumps(data, indent=2, default=str) + "\n")
        except Exception as exc:
            _log.warning("snapshot file write failed: %s", exc)

        return data
    except Exception as exc:
        _log.warning("capture_before failed for %s/%d: %s", resource_type, resource_id, exc)
        return {"_snapshot_error": str(exc)}


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def check_dataset_dependents(
    ws: PresetWorkspace,
    dataset_id: int,
) -> dict[str, Any]:
    """Find charts that depend on a given dataset.

    Returns advisory info — never blocks the mutation.
    """
    try:
        all_charts = ws.charts()
        affected = [
            {"id": ch.get("id"), "name": ch.get("slice_name")}
            for ch in all_charts
            if ch.get("datasource_id") == dataset_id
        ]
        return {
            "dataset_id": dataset_id,
            "affected_charts": affected,
            "chart_count": len(affected),
            "warning": (
                f"{len(affected)} chart(s) use this dataset and will be affected."
                if affected
                else "No charts depend on this dataset."
            ),
        }
    except Exception as exc:
        _log.warning("dependency check failed for dataset %d: %s", dataset_id, exc)
        return {
            "dataset_id": dataset_id,
            "affected_charts": [],
            "chart_count": 0,
            "warning": f"Could not check dependents: {exc}",
        }


# ---------------------------------------------------------------------------
# Params validation
# ---------------------------------------------------------------------------

_FORBIDDEN_PARAM_KEYS = frozenset({
    "datasource_id",
    "datasource_type",
    "database_id",
})


def validate_params_json(params_json: str) -> dict[str, Any]:
    """Parse and validate a params_json string for update_chart.

    Raises ``ValueError`` on:
      - Invalid JSON
      - Non-dict input
      - Keys that would rebind the chart's datasource
    """
    try:
        parsed = json.loads(params_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"params_json is not valid JSON: {exc}. "
            "Pass a JSON object string, e.g. '{\"metrics\": [\"count\"]}'"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"params_json must be a JSON object (dict), got {type(parsed).__name__}. "
            "Example: '{\"metrics\": [\"count\"]}'"
        )

    forbidden_found = _FORBIDDEN_PARAM_KEYS & set(parsed.keys())
    if forbidden_found:
        raise ValueError(
            f"params_json must not contain datasource-rebinding keys: "
            f"{sorted(forbidden_found)}. Use the dedicated tool parameters instead."
        )

    return parsed
