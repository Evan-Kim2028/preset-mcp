"""Safety guardrails: audit journal, pre-mutation snapshots, dependency checks."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

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
    action: Literal["create", "update", "delete", "restore"]
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
    "viz_type",
})

def _metric_column_name(metric: dict[str, Any]) -> str | None:
    """Extract a metric's referenced column name, if present."""
    column = metric.get("column")
    if isinstance(column, str):
        return column
    if isinstance(column, dict):
        name = column.get("column_name") or column.get("name")
        if isinstance(name, str):
            return name
    return None


def _validate_metric_object(metric: dict[str, Any], index: int) -> str | None:
    """Validate a metric object and return an optional referenced column."""
    expression_type = metric.get("expressionType")
    if expression_type is not None and not isinstance(expression_type, str):
        raise ValueError(
            f"metrics[{index}].expressionType must be a string when provided."
        )

    if expression_type == "SIMPLE":
        if not metric.get("aggregate"):
            raise ValueError(
                f"metrics[{index}] uses SIMPLE expressionType but has no aggregate."
            )
        col_name = _metric_column_name(metric)
        if not col_name:
            raise ValueError(
                f"metrics[{index}] uses SIMPLE expressionType but has no valid column."
            )
        return col_name

    if expression_type == "SQL":
        sql_expr = metric.get("sqlExpression")
        if not isinstance(sql_expr, str) or not sql_expr.strip():
            raise ValueError(
                f"metrics[{index}] uses SQL expressionType but sqlExpression is missing."
            )
        return None

    # Be permissive for legacy-style metric dicts so we don't reject valid
    # Superset payloads that omit expressionType.
    if expression_type is None:
        if metric.get("sqlExpression"):
            return None
        if metric.get("aggregate") and _metric_column_name(metric):
            return _metric_column_name(metric)
        if metric.get("metric_name") or metric.get("label") or metric.get("optionName"):
            return None
        raise ValueError(
            f"metrics[{index}] is a dict but missing metric structure. "
            "Expected saved metric reference or SIMPLE/SQL metric object."
        )

    raise ValueError(
        f"metrics[{index}] has unsupported expressionType={expression_type!r}."
    )


def _extract_filter_columns(filters: Any) -> set[str]:
    """Extract column references from filters/adhoc_filters payloads."""
    refs: set[str] = set()
    if not isinstance(filters, list):
        return refs
    for item in filters:
        if not isinstance(item, dict):
            continue
        for key in ("col", "subject", "column"):
            value = item.get(key)
            if isinstance(value, str) and value:
                refs.add(value)
        if isinstance(item.get("column"), dict):
            col_name = item["column"].get("column_name")
            if isinstance(col_name, str) and col_name:
                refs.add(col_name)
    return refs


def _extract_named_columns(value: Any) -> set[str]:
    """Extract string column names from list-like params entries."""
    refs: set[str] = set()
    if not isinstance(value, list):
        return refs
    for item in value:
        if isinstance(item, str) and item:
            refs.add(item)
    return refs


def validate_params_payload(
    params_json: str,
    *,
    dataset_columns: set[str] | None = None,
    dataset_metrics: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Parse + validate params_json and return advisory warnings.

    Raises ``ValueError`` on malformed JSON, forbidden keys, or invalid
    metric structures. Returns ``(parsed_params, warnings)`` on success.
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

    warnings: list[str] = []
    dataset_columns = dataset_columns or set()
    dataset_metrics = dataset_metrics or set()

    metrics = parsed.get("metrics")
    metric_column_refs: set[str] = set()
    if metrics is not None:
        if not isinstance(metrics, list):
            raise ValueError("params_json.metrics must be a list.")
        for idx, metric in enumerate(metrics):
            if isinstance(metric, str):
                if (
                    dataset_columns
                    and dataset_metrics
                    and metric not in dataset_metrics
                    and metric not in dataset_columns
                ):
                    warnings.append(
                        f"metrics[{idx}] references unknown metric/column {metric!r}."
                    )
                continue
            if isinstance(metric, dict):
                ref = _validate_metric_object(metric, idx)
                if ref:
                    metric_column_refs.add(ref)
                continue
            raise ValueError(
                f"metrics[{idx}] must be a string or metric object dict."
            )

    referenced_columns: set[str] = set()
    referenced_columns.update(_extract_named_columns(parsed.get("groupby")))
    referenced_columns.update(_extract_named_columns(parsed.get("columns")))
    referenced_columns.update(_extract_filter_columns(parsed.get("filters")))
    referenced_columns.update(_extract_filter_columns(parsed.get("adhoc_filters")))
    referenced_columns.update(metric_column_refs)

    if dataset_columns and referenced_columns:
        missing = sorted(col for col in referenced_columns if col not in dataset_columns)
        if missing:
            warnings.append(
                "params_json references unknown dataset columns: "
                f"{missing}."
            )

    return parsed, warnings


def validate_params_json(params_json: str) -> dict[str, Any]:
    """Backward-compatible validator returning only parsed params."""
    parsed, _ = validate_params_payload(params_json)
    return parsed


# ---------------------------------------------------------------------------
# Pre-delete export (mandatory, blocking)
# ---------------------------------------------------------------------------


def export_before_delete(
    ws: PresetWorkspace,
    resource_type: str,
    resource_id: int,
) -> Path:
    """Export a full ZIP backup before deletion. BLOCKING — raises on failure."""
    exports_dir = AUDIT_DIR / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_path = exports_dir / f"{resource_type}_{resource_id}_{ts}.zip"

    zip_bytes = ws.export_resource_zip(resource_type, [resource_id])
    export_path.write_bytes(zip_bytes)

    _log.info("pre-delete export saved: %s (%d bytes)", export_path, len(zip_bytes))
    return export_path
