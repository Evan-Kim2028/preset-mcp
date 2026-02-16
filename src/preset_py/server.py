"""MCP server exposing Preset workspace operations as tools.

Architecture
------------
Claude Code ──STDIO──▶ server.py (FastMCP) ──▶ PresetWorkspace ──▶ SupersetClient

Design (inspired by igloo-mcp):
  • Progressive disclosure – list tools accept ``response_mode``
    (compact / standard / full) so the LLM controls token budget.
  • Smart SQL truncation – ``run_sql`` defaults to a 5-row sample;
    full mode keeps head + tail when results exceed a threshold.
  • Structured errors – ToolError payloads carry ``error_type`` and
    ``hints[]`` so the LLM gets actionable recovery steps.
  • Structured logging – JSON lines on *stderr* (stdout is the STDIO
    transport).  Logs tool name, duration, status, and truncation info.
  • SQL read-only guard – AST-based validation via sqlglot; rejects
    write operations, multi-statement injection, and comment-wrapped
    bypasses before they reach Superset.
  • Workspace catalog – relationship-aware inventory showing how
    databases → datasets → charts → dashboards connect.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
from difflib import get_close_matches
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import sqlglot
from sqlglot import exp as E

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from preset_cli.api.operators import OneToMany

from preset_py.client import PresetWorkspace, connect
from preset_py._safety import (
    AUDIT_DIR,
    MutationEntry,
    capture_before,
    check_dataset_dependents,
    export_before_delete,
    record_mutation,
    validate_params_payload,
)

# ---------------------------------------------------------------------------
# Configuration (all overridable via PRESET_MCP_* env vars)
# ---------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


SQL_ROW_LIMIT: int = _env_int("PRESET_MCP_SQL_ROW_LIMIT", 1000)
SQL_SAMPLE_ROWS: int = _env_int("PRESET_MCP_SQL_SAMPLE_ROWS", 5)
TRUNCATION_THRESHOLD: int = _env_int("PRESET_MCP_TRUNCATION_THRESHOLD", 50)
TRUNCATION_TAIL: int = _env_int("PRESET_MCP_TRUNCATION_TAIL", 5)

# ---------------------------------------------------------------------------
# Logging  (stderr — stdout is the STDIO transport)
# ---------------------------------------------------------------------------

_log = logging.getLogger("preset-mcp")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(
    logging.Formatter(
        '{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
    )
)
_log.addHandler(_handler)
_log.setLevel(os.environ.get("PRESET_MCP_LOG_LEVEL", "INFO").upper())

# ---------------------------------------------------------------------------
# Progressive-disclosure field sets
# ---------------------------------------------------------------------------

ResponseMode = Literal["compact", "standard", "full"]

_COMPACT: dict[str, list[str]] = {
    "dashboard": ["id", "dashboard_title", "published"],
    "chart": ["id", "slice_name", "viz_type"],
    "dataset": ["id", "table_name", "schema"],
    "database": ["id", "database_name", "backend"],
}

_STANDARD: dict[str, list[str]] = {
    "dashboard": [
        "id", "dashboard_title", "status", "published",
        "changed_on", "owners", "url",
    ],
    "chart": [
        "id", "slice_name", "viz_type", "datasource_id",
        "datasource_name", "changed_on",
    ],
    "dataset": [
        "id", "table_name", "database", "schema", "sql",
        "columns", "changed_on",
    ],
    "database": [
        "id", "database_name", "backend", "expose_in_sqllab",
        "allow_dml",
    ],
}

# ---------------------------------------------------------------------------
# Detail-view field sets (for get_* single-resource tools)
# ---------------------------------------------------------------------------

_DETAIL_COMPACT: dict[str, list[str]] = {
    "dashboard": [
        "id", "dashboard_title", "published", "status",
        "orphaned_chart_refs",
    ],
    "chart": ["id", "slice_name", "viz_type", "datasource_id", "datasource_type"],
    "dataset": ["id", "table_name", "schema", "database", "kind"],
    "database": ["id", "database_name", "backend"],
}

_DETAIL_STANDARD: dict[str, list[str]] = {
    "dashboard": [
        "id", "dashboard_title", "published", "status",
        "url", "slug", "owners", "changed_on",
        "dashboard_health", "orphaned_chart_refs",
    ],
    "chart": [
        "id", "slice_name", "viz_type", "datasource_id",
        "datasource_type", "datasource_name_text",
        "params", "changed_on", "owners",
    ],
    "dataset": [
        "id", "table_name", "schema", "database", "kind", "sql",
        "columns", "metrics", "changed_on", "owners",
    ],
    "database": [
        "id", "database_name", "backend", "expose_in_sqllab",
        "allow_dml", "configuration_method",
    ],
}

# ---------------------------------------------------------------------------
# Server + lazy state
# ---------------------------------------------------------------------------

mcp = FastMCP("preset-mcp")

_ws: PresetWorkspace | None = None


def _get_ws() -> PresetWorkspace:
    global _ws
    if _ws is None:
        workspace = os.environ.get("PRESET_WORKSPACE")
        _ws = connect(workspace)
        _log.info("connected workspace=%s", workspace or "(none)")
    return _ws


# ---------------------------------------------------------------------------
# Internal helpers — progressive disclosure
# ---------------------------------------------------------------------------


def _pick(records: list[dict], fields: list[str]) -> list[dict]:
    """Extract *fields* from each record, skipping missing keys."""
    return [{k: r[k] for k in fields if k in r} for r in records]


def _format_list(
    records: list[dict],
    resource: str,
    mode: ResponseMode,
) -> str:
    """Apply progressive disclosure to a list of API records."""
    if mode == "compact":
        data = _pick(records, _COMPACT.get(resource, []))
    elif mode == "standard":
        data = _pick(records, _STANDARD.get(resource, []))
    else:
        data = records

    out: dict[str, Any] = {
        "count": len(records),
        "response_mode": mode,
        "data": data,
    }
    if mode != "full":
        out["hint"] = "Set response_mode='full' to see all fields."
    return json.dumps(out, indent=2, default=str)


def _format_detail(record: dict, resource: str, mode: ResponseMode) -> str:
    """Apply progressive disclosure to a single API record (detail view)."""
    if mode == "compact":
        data = {k: record[k] for k in _DETAIL_COMPACT.get(resource, []) if k in record}
    elif mode == "standard":
        data = {k: record[k] for k in _DETAIL_STANDARD.get(resource, []) if k in record}
    else:
        data = record
    out: dict[str, Any] = {"response_mode": mode, "data": data}
    if mode != "full":
        out["hint"] = "Set response_mode='full' to see all fields."
    return json.dumps(out, indent=2, default=str)


def _format_sql(
    records: list[dict],
    columns: list[str],
    mode: ResponseMode,
) -> str:
    """Apply progressive disclosure to SQL query results."""
    total = len(records)
    out: dict[str, Any] = {
        "rowcount": total,
        "columns": columns,
        "response_mode": mode,
    }

    if mode == "compact":
        out["hint"] = (
            "Schema only. Use response_mode='standard' for sample rows "
            "or 'full' for all rows."
        )
    elif mode == "standard":
        sample = records[:SQL_SAMPLE_ROWS]
        out["sample_rows"] = sample
        if total > SQL_SAMPLE_ROWS:
            out["truncated"] = True
            out["hint"] = (
                f"Showing {len(sample)}/{total} rows. "
                "Use response_mode='full' for all rows."
            )
    else:  # full
        if total > TRUNCATION_THRESHOLD:
            head_n = TRUNCATION_THRESHOLD - TRUNCATION_TAIL
            head = records[:head_n]
            tail = records[-TRUNCATION_TAIL:]
            omitted = total - head_n - TRUNCATION_TAIL
            out["rows"] = (
                head
                + [{"__truncated__": f"…{omitted} rows omitted…"}]
                + tail
            )
            out["truncated"] = True
        else:
            out["rows"] = records
            out["truncated"] = False

    return json.dumps(out, indent=2, default=str)


# ---------------------------------------------------------------------------
# Client-side name filter
# ---------------------------------------------------------------------------

_NAME_KEYS: dict[str, str] = {
    "dashboard": "dashboard_title",
    "chart": "slice_name",
    "dataset": "table_name",
    "database": "database_name",
}


def _filter_by_name(records: list[dict], resource: str, name_contains: str) -> list[dict]:
    """Case-insensitive substring filter on the resource's name field."""
    key = _NAME_KEYS.get(resource, "name")
    needle = name_contains.lower()
    return [r for r in records if needle in str(r.get(key, "")).lower()]


# ---------------------------------------------------------------------------
# Input coercion + validation helpers
# ---------------------------------------------------------------------------

_KNOWN_VIZ_TYPES = frozenset({
    "area",
    "big_number_total",
    "dist_bar",
    "echarts_area",
    "echarts_bar",
    "echarts_timeseries_area",
    "echarts_timeseries_bar",
    "echarts_timeseries_line",
    "histogram_v2",
    "line",
    "mixed_timeseries",
    "pie",
    "pivot_table_v2",
    "sankey_v2",
    "sunburst_v2",
    "table",
    "treemap_v2",
})

_DISCOURAGED_VIZ_TYPES: dict[str, str] = {
    # Legacy/deprecated aliases seen to fail in modern Preset workspaces.
    "echarts_bar": "echarts_timeseries_bar",
}


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _ensure_json_dict(value: Any, field_name: str) -> dict[str, Any]:
    """Parse dict-like JSON fields from dashboard payloads."""
    if isinstance(value, dict):
        return value
    if value in (None, "", "{}"):
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} is not valid JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"{field_name} must be a JSON object.")
    raise ValueError(f"{field_name} must be a dict or JSON object string.")


def _coerce_list_arg(
    value: list[Any] | tuple[Any, ...] | str | None,
    *,
    field_name: str,
    item_kind: Literal["str", "int"],
) -> list[Any] | None:
    """Accept native list or JSON-string list for MCP arguments."""
    if value is None:
        return None

    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{field_name} must be a list or JSON array string (e.g. "
                f"'[\"x\", \"y\"]'). Got {raw!r}."
            ) from exc
        raw = parsed

    if isinstance(raw, tuple):
        raw = list(raw)
    if not isinstance(raw, list):
        raise ValueError(
            f"{field_name} must be a list. Got {type(raw).__name__}."
        )

    normalized: list[Any] = []
    for idx, item in enumerate(raw):
        if item_kind == "str":
            if item is None:
                raise ValueError(f"{field_name}[{idx}] must not be null.")
            normalized.append(str(item))
            continue

        parsed_int = _to_int(item)
        if parsed_int is None:
            raise ValueError(
                f"{field_name}[{idx}] must be an integer. Got {item!r}."
            )
        normalized.append(parsed_int)
    return normalized


def _dataset_columns(dataset: dict[str, Any]) -> set[str]:
    columns = dataset.get("columns", [])
    if not isinstance(columns, list):
        return set()
    names: set[str] = set()
    for column in columns:
        if not isinstance(column, dict):
            continue
        name = column.get("column_name") or column.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _dataset_metrics(dataset: dict[str, Any]) -> set[str]:
    metrics = dataset.get("metrics", [])
    if not isinstance(metrics, list):
        return set()
    names: set[str] = set()
    for metric in metrics:
        if isinstance(metric, str):
            names.add(metric)
            continue
        if not isinstance(metric, dict):
            continue
        for key in ("metric_name", "label", "name"):
            value = metric.get(key)
            if isinstance(value, str) and value:
                names.add(value)
                break
    return names


def _require_dataset_exists(ws: PresetWorkspace, dataset_id: int) -> None:
    try:
        ws.dataset_detail(dataset_id)
    except Exception as exc:
        raise ValueError(
            f"Dataset {dataset_id} not found. Use list_datasets to find valid IDs."
        ) from exc


def _require_chart_exists(ws: PresetWorkspace, chart_id: int) -> None:
    try:
        ws.chart_detail(chart_id)
    except Exception as exc:
        raise ValueError(
            f"Chart {chart_id} not found. Use list_charts to find valid IDs."
        ) from exc


def _require_database_exists(ws: PresetWorkspace, database_id: int) -> None:
    try:
        ws.database_detail(database_id)
    except Exception as exc:
        raise ValueError(
            f"Database {database_id} not found. Use list_databases to find valid IDs."
        ) from exc


def _require_dashboards_exist(ws: PresetWorkspace, dashboard_ids: list[int]) -> None:
    for dashboard_id in dashboard_ids:
        try:
            ws.dashboard_detail(dashboard_id)
        except Exception as exc:
            raise ValueError(
                f"Dashboard {dashboard_id} not found. Use list_dashboards to find valid IDs."
            ) from exc


def _collect_workspace_viz_types(ws: PresetWorkspace) -> set[str]:
    viz_types = set(_KNOWN_VIZ_TYPES)
    try:
        for chart in ws.charts():
            viz = chart.get("viz_type")
            if isinstance(viz, str) and viz:
                viz_types.add(viz)
    except Exception:
        # Non-blocking fallback to curated defaults.
        pass
    return viz_types


def _validate_viz_type(ws: PresetWorkspace, viz_type: str) -> None:
    if viz_type in _DISCOURAGED_VIZ_TYPES:
        replacement = _DISCOURAGED_VIZ_TYPES[viz_type]
        raise ValueError(
            f"viz_type {viz_type!r} is discouraged/legacy. "
            f"Use {replacement!r} instead."
        )

    allowed = _collect_workspace_viz_types(ws)
    if viz_type in allowed:
        return

    suggestions = get_close_matches(viz_type, sorted(allowed), n=5, cutoff=0.55)
    msg = (
        f"Unsupported viz_type {viz_type!r}. "
        "Use list_charts(...) to inspect known viz types in this workspace."
    )
    if suggestions:
        msg += f" Suggestions: {suggestions}."
    raise ValueError(msg)


def _parse_chart_datasource_from_params(params_value: Any) -> tuple[int | None, str | None]:
    try:
        params = _ensure_json_dict(params_value, "chart params")
    except ValueError:
        return None, None
    datasource = params.get("datasource")
    if isinstance(datasource, str) and "__" in datasource:
        raw_id, ds_type = datasource.split("__", 1)
        ds_id = _to_int(raw_id)
        return ds_id, ds_type or "table"
    if isinstance(datasource, dict):
        return _to_int(datasource.get("id")), str(datasource.get("type", "table"))
    return None, None


def _parse_chart_datasource_from_query_context(value: Any) -> tuple[int | None, str | None]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None, None
    if not isinstance(value, dict):
        return None, None
    datasource = value.get("datasource")
    if isinstance(datasource, dict):
        return _to_int(datasource.get("id")), str(datasource.get("type", "table"))
    return None, None


def _enrich_chart_datasource_fields(ws: PresetWorkspace, chart: dict[str, Any]) -> dict[str, Any]:
    """Backfill datasource_* fields missing from chart detail endpoint."""
    enriched = dict(chart)
    ds_id = _to_int(enriched.get("datasource_id"))
    ds_type = enriched.get("datasource_type")
    ds_name = enriched.get("datasource_name_text") or enriched.get("datasource_name")

    if ds_id is None or not ds_type:
        id_from_params, type_from_params = _parse_chart_datasource_from_params(
            enriched.get("params", {})
        )
        if ds_id is None:
            ds_id = id_from_params
        if not ds_type:
            ds_type = type_from_params

    if ds_id is None or not ds_type:
        id_from_qc, type_from_qc = _parse_chart_datasource_from_query_context(
            enriched.get("query_context")
        )
        if ds_id is None:
            ds_id = id_from_qc
        if not ds_type:
            ds_type = type_from_qc

    if ds_name is None or ds_id is None or not ds_type:
        try:
            chart_id = _to_int(enriched.get("id"))
            if chart_id is not None:
                listing = next(
                    (item for item in ws.charts() if _to_int(item.get("id")) == chart_id),
                    None,
                )
                if isinstance(listing, dict):
                    ds_id = ds_id or _to_int(listing.get("datasource_id"))
                    ds_type = ds_type or listing.get("datasource_type")
                    ds_name = ds_name or listing.get("datasource_name_text") or listing.get("datasource_name")
        except Exception:
            pass

    if ds_name is None and ds_id is not None:
        try:
            dataset = ws.dataset_detail(ds_id)
            ds_name = dataset.get("table_name") or dataset.get("datasource_name_text")
        except Exception:
            pass

    if ds_id is not None:
        enriched["datasource_id"] = ds_id
    if ds_type:
        enriched["datasource_type"] = ds_type
    if ds_name:
        enriched["datasource_name_text"] = ds_name
    return enriched


def _extract_position_chart_refs(position_json: Any) -> dict[str, int]:
    """Return {layout_node_id: chart_id} from dashboard position_json."""
    payload = _ensure_json_dict(position_json, "position_json")
    refs: dict[str, int] = {}
    for node_id, node in payload.items():
        if not isinstance(node, dict):
            continue
        chart_id = _to_int(node.get("chartId"))
        if chart_id is None:
            meta = node.get("meta")
            if isinstance(meta, dict):
                chart_id = _to_int(meta.get("chartId"))
        if chart_id is not None:
            refs[str(node_id)] = chart_id
    return refs


def _extract_scope_chart_refs(json_metadata: Any) -> set[int]:
    payload = _ensure_json_dict(json_metadata, "json_metadata")
    charts_in_scope = payload.get("chartsInScope")
    if not isinstance(charts_in_scope, dict):
        return set()
    refs: set[int] = set()
    for raw_id in charts_in_scope.keys():
        parsed = _to_int(raw_id)
        if parsed is not None:
            refs.add(parsed)
    return refs


def _dashboard_orphan_refs(
    dashboard: dict[str, Any],
    existing_chart_ids: set[int],
) -> dict[str, Any]:
    pos_refs = _extract_position_chart_refs(dashboard.get("position_json", {}))
    position_ids = set(pos_refs.values())
    scope_ids = _extract_scope_chart_refs(dashboard.get("json_metadata", {}))
    referenced = position_ids | scope_ids
    orphaned = sorted(chart_id for chart_id in referenced if chart_id not in existing_chart_ids)
    return {
        "position_chart_ids": sorted(position_ids),
        "charts_in_scope_ids": sorted(scope_ids),
        "orphaned_chart_refs": orphaned,
        "orphaned_count": len(orphaned),
    }


def _layout_chart_ids(position_json: Any) -> set[int]:
    return set(_extract_position_chart_refs(position_json).values())


def _chart_title(entry: dict[str, Any]) -> str | None:
    for key in ("slice_name", "name", "chart_name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _repair_dashboard_refs(
    position_json: Any,
    json_metadata: Any,
    dashboard_charts: list[dict[str, Any]],
    *,
    strategy: Literal["replace_by_name", "remove_orphans"],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Repair stale chart refs in position_json and json_metadata."""
    position = _ensure_json_dict(position_json, "position_json")
    metadata = _ensure_json_dict(json_metadata, "json_metadata")

    valid_chart_ids = {
        cid for cid in (_to_int(chart.get("id")) for chart in dashboard_charts)
        if cid is not None
    }

    charts_by_name: dict[str, int] = {}
    for chart in dashboard_charts:
        title = _chart_title(chart)
        cid = _to_int(chart.get("id"))
        if title and cid is not None:
            charts_by_name.setdefault(title.lower(), cid)

    position_refs = _extract_position_chart_refs(position)
    replacement_map: dict[int, int] = {}
    removed_nodes: set[str] = set()
    stale_ids_before: set[int] = set()

    for node_id, chart_id in position_refs.items():
        if chart_id in valid_chart_ids:
            continue
        stale_ids_before.add(chart_id)
        node = position.get(node_id, {})
        replacement: int | None = None
        if strategy == "replace_by_name" and isinstance(node, dict):
            meta = node.get("meta", {})
            title = None
            if isinstance(meta, dict):
                value = meta.get("sliceName") or meta.get("chartName") or meta.get("text")
                if isinstance(value, str) and value.strip():
                    title = value.strip().lower()
            if title:
                replacement = charts_by_name.get(title)
        if replacement is not None:
            if isinstance(node, dict):
                if "chartId" in node:
                    node["chartId"] = replacement
                meta = node.get("meta")
                if isinstance(meta, dict):
                    meta["chartId"] = replacement
            replacement_map[chart_id] = replacement
            continue
        removed_nodes.add(node_id)

    for node_id in removed_nodes:
        position.pop(node_id, None)

    if removed_nodes:
        for node in position.values():
            if not isinstance(node, dict):
                continue
            children = node.get("children")
            if isinstance(children, list):
                node["children"] = [
                    child for child in children if str(child) not in removed_nodes
                ]

    charts_in_scope = metadata.get("chartsInScope")
    scope_before = set()
    if isinstance(charts_in_scope, dict):
        scope_before = _extract_scope_chart_refs(metadata)
        updated_scope: dict[str, Any] = {}
        for raw_id, value in charts_in_scope.items():
            cid = _to_int(raw_id)
            if cid is None:
                continue
            if cid in valid_chart_ids:
                updated_scope[str(cid)] = value
                continue
            replacement = replacement_map.get(cid)
            if replacement is not None:
                updated_scope[str(replacement)] = value
        metadata["chartsInScope"] = updated_scope

    repaired_refs = _extract_position_chart_refs(position)
    scope_after = _extract_scope_chart_refs(metadata)
    stale_after = {
        cid for cid in (set(repaired_refs.values()) | scope_after)
        if cid not in valid_chart_ids
    }

    summary = {
        "strategy": strategy,
        "valid_chart_ids": sorted(valid_chart_ids),
        "replacements": replacement_map,
        "removed_layout_nodes": sorted(removed_nodes),
        "orphaned_before": sorted(stale_ids_before | {cid for cid in scope_before if cid not in valid_chart_ids}),
        "orphaned_after": sorted(stale_after),
        "changed": bool(replacement_map or removed_nodes or stale_after != (stale_ids_before | {cid for cid in scope_before if cid not in valid_chart_ids})),
    }

    return position, metadata, summary


# ---------------------------------------------------------------------------
# SQL validation — AST-based via sqlglot
# ---------------------------------------------------------------------------

# Expression types that represent write/DDL operations
_WRITE_TYPES: tuple[type, ...] = (
    E.Insert,
    E.Update,
    E.Delete,
    E.Drop,
    E.Create,
    E.Alter,
    E.Merge,
    E.TruncateTable,
    E.Grant,
    E.Revoke,
)

# Command-text tokens that are blocked (parsed as sqlglot.exp.Command)
_BLOCKED_COMMANDS = frozenset({
    "TRUNCATE", "GRANT", "REVOKE", "CALL", "EXECUTE", "EXEC",
    "COPY", "UNLOAD", "VACUUM", "BEGIN", "COMMIT", "ROLLBACK",
})


def _validate_readonly(sql: str) -> None:
    """Reject write operations using AST-based analysis.

    Handles:
      • Comment injection  (``-- ...\\nDELETE``, ``/* */ DELETE``)
      • Multi-statement    (``SELECT 1; DROP TABLE x``)
      • CTE wrapping       (``WITH x AS (...) DELETE FROM y``)
      • Missing keywords   (MERGE, COPY, CALL, TRUNCATE, …)

    Raises :class:`ValueError` with a descriptive message on rejection.
    """
    if not sql or not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    # Parse with sqlglot — strips comments, builds AST
    try:
        stmts = sqlglot.parse(sql)
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"Could not parse SQL: {exc}") from exc

    # Filter out None entries (trailing semicolons produce these)
    real = [s for s in stmts if s is not None]

    if not real:
        raise ValueError("Empty SQL statement after parsing.")

    # Reject multi-statement queries (primary injection vector)
    if len(real) > 1:
        raise ValueError(
            "Multi-statement queries are not allowed. "
            "Submit one statement at a time."
        )

    stmt = real[0]

    # Block known write/DDL expression types
    if isinstance(stmt, _WRITE_TYPES):
        name = type(stmt).__name__.upper()
        raise ValueError(
            f"Write operation '{name}' is not allowed through MCP. "
            "Only SELECT, SHOW, DESCRIBE, and EXPLAIN are permitted."
        )

    # Block dangerous Command types (TRUNCATE, GRANT, CALL, etc.)
    if isinstance(stmt, E.Command):
        cmd = str(stmt.this).upper() if stmt.this else ""
        if cmd in _BLOCKED_COMMANDS:
            raise ValueError(
                f"Operation '{cmd}' is not allowed through MCP. "
                "Only SELECT, SHOW, DESCRIBE, and EXPLAIN are permitted."
            )


# ---------------------------------------------------------------------------
# Error handling (structured, with hints)
# ---------------------------------------------------------------------------


def _exception_to_tool_error(tool_name: str, exc: Exception) -> ToolError:
    """Map exception to structured ToolError with hints."""
    if isinstance(exc, KeyError):
        _log.warning("tool=%s error=missing_env key=%s", tool_name, exc)
        return ToolError(
            json.dumps({
                "error": f"Missing environment variable: {exc}",
                "error_type": "configuration",
                "hints": [
                    "Set PRESET_API_TOKEN and PRESET_API_SECRET.",
                    "Optionally set PRESET_WORKSPACE to auto-connect.",
                ],
            })
        )

    if isinstance(exc, RuntimeError):
        _log.warning("tool=%s error=no_workspace", tool_name)
        return ToolError(
            json.dumps({
                "error": str(exc),
                "error_type": "no_workspace",
                "hints": [
                    "Call list_workspaces to see available workspaces.",
                    "Then call use_workspace('Title') to select one.",
                ],
            })
        )

    if isinstance(exc, ValueError):
        _log.warning("tool=%s error=validation msg=%s", tool_name, exc)
        return ToolError(
            json.dumps({
                "error": str(exc),
                "error_type": "validation",
                "hints": [
                    "Check parameter values and try again.",
                ],
            })
        )

    _log.error("tool=%s error=api msg=%s", tool_name, exc)
    return ToolError(
        json.dumps({
            "error": f"Preset API error: {exc}",
            "error_type": "api_error",
            "hints": [
                "This may be transient — retry once.",
                "If the error mentions a resource ID, verify it "
                "exists with the corresponding list_* tool.",
            ],
        })
    )


def _handle_errors(fn):
    """Decorator: catch exceptions → structured ToolError with hints."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            _log.info(
                "tool=%s status=ok duration_ms=%.0f",
                fn.__name__, (time.monotonic() - t0) * 1000,
            )
            return result
        except ToolError:
            raise
        except Exception as exc:
            raise _exception_to_tool_error(fn.__name__, exc) from exc

    return wrapper


# ---------------------------------------------------------------------------
# Mutation helper — consolidates audit/dry-run/response boilerplate
# ---------------------------------------------------------------------------


def _do_mutation(
    *,
    tool_name: str,
    resource_type: str,
    action: Literal["create", "update", "delete"],
    fields_changed: list[str],
    dry_run: bool,
    execute: Callable[[], dict[str, Any]],
    resource_id: int | None = None,
    before: dict[str, Any] | None = None,
    export_path: Path | None = None,
    preview_extras: dict[str, Any] | None = None,
    result_extras: dict[str, Any] | None = None,
    after_extras: dict[str, Any] | None = None,
) -> str:
    """Handle audit/dry-run/response pattern for all mutation tools.

    Dry-run path: builds preview JSON, records audit, returns.
    Execute path: calls ``execute()``, builds response, records audit, returns.

    Extra dicts are merged into the appropriate output:
      - ``preview_extras`` → dry-run preview (e.g. ``{"values": {...}}``)
      - ``result_extras``  → execution response (e.g. ``{"_dependency_impact": ...}``)
      - ``after_extras``   → ``after_summary`` in audit entry (e.g. ``{"dashboard_title": ...}``)
    """
    # -- Dry-run path -------------------------------------------------------
    if dry_run:
        preview: dict[str, Any] = {
            "dry_run": True,
            "action": tool_name,
            "fields_to_change": fields_changed,
        }
        if resource_id is not None:
            preview[f"{resource_type}_id"] = resource_id
        if before is not None:
            preview["current_state"] = before
        if export_path is not None:
            preview["export_path"] = str(export_path)
        if preview_extras:
            preview.update(preview_extras)

        dry_after: dict[str, Any] | None = None
        if export_path is not None:
            dry_after = {"export_path": str(export_path)}
        if after_extras:
            dry_after = dry_after or {}
            dry_after.update(after_extras)

        record_mutation(MutationEntry(
            tool_name=tool_name,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            fields_changed=fields_changed,
            before_snapshot=before,
            after_summary=dry_after,
            dry_run=True,
        ))
        return json.dumps(preview, indent=2, default=str)

    # -- Execute path -------------------------------------------------------
    result = execute()

    if action == "create":
        rid = result.get("id") if result else None
        after_summary: dict[str, Any] = {"id": rid}
        resource_id = rid
        response = result
    elif action == "delete":
        after_summary = {}
        if export_path is not None:
            after_summary["export_path"] = str(export_path)
        response: dict[str, Any] = {
            "status": "deleted",
            f"{resource_type}_id": resource_id,
        }
        if export_path is not None:
            response["export_path"] = str(export_path)
    else:  # update
        after_summary = {"id": resource_id, "fields_updated": fields_changed}
        response = result if isinstance(result, dict) else {"result": result}

    if after_extras:
        after_summary.update(after_extras)
    if result_extras:
        response.update(result_extras)

    record_mutation(MutationEntry(
        tool_name=tool_name,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        fields_changed=fields_changed,
        before_snapshot=before,
        after_summary=after_summary,
    ))

    return json.dumps(response, indent=2, default=str)


# ===================================================================
# Tools — Workspace navigation
# ===================================================================


@mcp.tool()
@_handle_errors
def list_workspaces() -> str:
    """List all Preset workspaces you have access to.

    Call this FIRST to discover workspace titles, then pass one to
    use_workspace.
    """
    ws = _get_ws()
    return json.dumps(ws.list_workspaces(), indent=2)


@mcp.tool()
@_handle_errors
def use_workspace(workspace_title: str) -> str:
    """Switch to a different Preset workspace by its exact title.

    You must call this before any read/write tools if PRESET_WORKSPACE
    was not set at startup.  Use list_workspaces to discover titles.

    Args:
        workspace_title: Exact workspace title (e.g. "Mysten Labs--General")
    """
    global _ws
    ws = _get_ws()
    _ws = ws.use(workspace_title)
    _log.info("switched workspace=%s url=%s", workspace_title, _ws.workspace_url)
    return json.dumps({
        "status": "ok",
        "workspace": workspace_title,
        "url": str(_ws.workspace_url),
    })


# ===================================================================
# Tools — Read operations
# ===================================================================


@mcp.tool()
@_handle_errors
def list_dashboards(
    response_mode: ResponseMode = "standard",
    name_contains: str | None = None,
) -> str:
    """List dashboards in the current workspace.

    Start here to discover dashboard IDs, then use get_dashboard for
    detail on a specific one.

    Args:
        response_mode: 'compact' (id+title), 'standard' (key fields),
                       or 'full' (raw API response).  Default: standard.
        name_contains: Case-insensitive substring filter on dashboard_title.
    """
    ws = _get_ws()
    raw = ws.dashboards()
    if name_contains:
        raw = _filter_by_name(raw, "dashboard", name_contains)
    _log.info("list_dashboards count=%d mode=%s", len(raw), response_mode)
    return _format_list(raw, "dashboard", response_mode)


@mcp.tool()
@_handle_errors
def get_dashboard(
    dashboard_id: int,
    response_mode: ResponseMode = "full",
) -> str:
    """Get detail for a single dashboard.

    Use list_dashboards first to find valid IDs.  Use response_mode to
    control verbosity — dashboards can contain large position_json and
    json_metadata blobs.

    Args:
        dashboard_id: Numeric dashboard ID
        response_mode: 'compact' (id+title), 'standard' (key fields, no
                       position_json), or 'full' (raw API response).
                       Default: full.
    """
    ws = _get_ws()
    result = ws.dashboard_detail(dashboard_id)
    try:
        existing_chart_ids = {
            cid for cid in (_to_int(ch.get("id")) for ch in ws.charts())
            if cid is not None
        }
        health = _dashboard_orphan_refs(result, existing_chart_ids)
        result["dashboard_health"] = health
        result["orphaned_chart_refs"] = health["orphaned_chart_refs"]
    except Exception as exc:
        result["dashboard_health"] = {
            "status": "unavailable",
            "error": str(exc),
        }
    return _format_detail(result, "dashboard", response_mode)


@mcp.tool()
@_handle_errors
def get_chart(
    chart_id: int,
    response_mode: ResponseMode = "full",
) -> str:
    """Get detail for a single chart including params and query context.

    Returns the chart's visualization type, parameters, query_context,
    datasource info, and other metadata.  Use list_charts first to find
    valid IDs.

    Args:
        chart_id: Numeric chart ID
        response_mode: 'compact' (id+name+viz_type), 'standard' (key fields),
                       or 'full' (raw API response).  Default: full.
    """
    ws = _get_ws()
    result = _enrich_chart_datasource_fields(ws, ws.chart_detail(chart_id))
    return _format_detail(result, "chart", response_mode)


@mcp.tool()
@_handle_errors
def get_dataset(
    dataset_id: int,
    response_mode: ResponseMode = "full",
    refresh_columns: bool = False,
) -> str:
    """Get detail for a single dataset including columns, metrics, and SQL.

    Use list_datasets first to find valid IDs.  Set refresh_columns=True
    to query the source database for current column metadata (useful when
    the underlying table schema has changed).

    Args:
        dataset_id: Numeric dataset ID
        response_mode: 'compact' (id+name+schema), 'standard' (columns,
                       metrics, sql), or 'full' (raw API response).
                       Default: full.
        refresh_columns: If True, also fetch live column metadata from the
                         source database and include as 'external_columns'.
    """
    ws = _get_ws()
    result = ws.dataset_detail(dataset_id)
    if refresh_columns:
        result["external_columns"] = ws.refresh_dataset_columns(dataset_id)
    return _format_detail(result, "dataset", response_mode)


@mcp.tool()
@_handle_errors
def get_database(
    database_id: int,
    response_mode: ResponseMode = "full",
) -> str:
    """Get detail for a single database connection.

    Use list_databases first to find valid IDs.

    Args:
        database_id: Numeric database ID
        response_mode: 'compact' (id+name+backend), 'standard' (key fields),
                       or 'full' (raw API response).  Default: full.
    """
    ws = _get_ws()
    result = ws.database_detail(database_id)
    return _format_detail(result, "database", response_mode)


@mcp.tool()
@_handle_errors
def list_charts(
    response_mode: ResponseMode = "standard",
    name_contains: str | None = None,
    viz_type: str | None = None,
    dataset_id: int | None = None,
) -> str:
    """List charts in the current workspace.

    Use this to find chart IDs and see which viz types are in use.

    Args:
        response_mode: 'compact', 'standard', or 'full'.  Default: standard.
        name_contains: Case-insensitive substring filter on slice_name.
        viz_type: Exact-match filter on viz_type (e.g. "echarts_timeseries_bar").
        dataset_id: Filter to charts using this datasource_id.
    """
    ws = _get_ws()
    raw = ws.charts()
    if name_contains:
        raw = _filter_by_name(raw, "chart", name_contains)
    if viz_type:
        raw = [r for r in raw if r.get("viz_type") == viz_type]
    if dataset_id is not None:
        raw = [r for r in raw if r.get("datasource_id") == dataset_id]
    _log.info("list_charts count=%d mode=%s", len(raw), response_mode)
    return _format_list(raw, "chart", response_mode)


@mcp.tool()
@_handle_errors
def list_datasets(
    response_mode: ResponseMode = "standard",
    name_contains: str | None = None,
    schema: str | None = None,
    database_id: int | None = None,
) -> str:
    """List datasets (virtual tables) in the current workspace.

    Datasets are the data sources for charts.  Use list_databases to
    find connection IDs needed for creating new datasets.

    Args:
        response_mode: 'compact', 'standard', or 'full'.  Default: standard.
        name_contains: Case-insensitive substring filter on table_name.
        schema: Server-side exact-match filter on schema name.
        database_id: Server-side filter to datasets in this database.
    """
    ws = _get_ws()
    server_filters: dict = {}
    if schema:
        server_filters["schema"] = schema
    if database_id is not None:
        server_filters["database"] = OneToMany(database_id)
    raw = ws.datasets(**server_filters)
    if name_contains:
        raw = _filter_by_name(raw, "dataset", name_contains)
    _log.info("list_datasets count=%d mode=%s", len(raw), response_mode)
    return _format_list(raw, "dataset", response_mode)


@mcp.tool()
@_handle_errors
def list_databases(
    response_mode: ResponseMode = "standard",
    name_contains: str | None = None,
) -> str:
    """List database connections in the current workspace.

    Call this BEFORE run_sql or create_dataset to find a valid
    database_id.

    Args:
        response_mode: 'compact', 'standard', or 'full'.  Default: standard.
        name_contains: Case-insensitive substring filter on database_name.
    """
    ws = _get_ws()
    raw = ws.databases()
    if name_contains:
        raw = _filter_by_name(raw, "database", name_contains)
    _log.info("list_databases count=%d mode=%s", len(raw), response_mode)
    return _format_list(raw, "database", response_mode)


# ===================================================================
# Tools — Workspace catalog
# ===================================================================


@mcp.tool()
@_handle_errors
def workspace_catalog() -> str:
    """Build a relationship-aware catalog of the current workspace.

    Returns databases, datasets, charts, and dashboards with their
    dependency links (dataset→database, chart→dataset).  Use this to
    understand workspace topology BEFORE creating or updating resources.

    Much cheaper on tokens than snapshot_workspace — returns only the
    fields needed for navigation.
    """
    ws = _get_ws()
    databases = ws.databases()
    datasets = ws.datasets()
    charts = ws.charts()
    dashboards = ws.dashboards()

    def _db_ref(ds: dict) -> dict[str, Any]:
        """Extract database reference from a dataset record."""
        db = ds.get("database")
        if isinstance(db, dict):
            return {"id": db.get("id"), "name": db.get("database_name")}
        return {"id": db}

    catalog = {
        "counts": {
            "databases": len(databases),
            "datasets": len(datasets),
            "charts": len(charts),
            "dashboards": len(dashboards),
        },
        "databases": [
            {
                "id": db.get("id"),
                "name": db.get("database_name"),
                "backend": db.get("backend"),
            }
            for db in databases
        ],
        "datasets": [
            {
                "id": ds.get("id"),
                "name": ds.get("table_name"),
                "database": _db_ref(ds),
                "schema": ds.get("schema"),
            }
            for ds in datasets
        ],
        "charts": [
            {
                "id": ch.get("id"),
                "name": ch.get("slice_name"),
                "viz_type": ch.get("viz_type"),
                "dataset_id": ch.get("datasource_id"),
            }
            for ch in charts
        ],
        "dashboards": [
            {
                "id": d.get("id"),
                "title": d.get("dashboard_title"),
                "published": d.get("published"),
            }
            for d in dashboards
        ],
    }
    _log.info(
        "workspace_catalog counts=%s", catalog["counts"],
    )
    return json.dumps(catalog, indent=2, default=str)


# ===================================================================
# Tools — SQL
# ===================================================================


@mcp.tool()
@_handle_errors
def run_sql(
    sql: str,
    database_id: int,
    schema: str | None = None,
    limit: int = SQL_ROW_LIMIT,
    response_mode: ResponseMode = "standard",
) -> str:
    """Execute a read-only SQL query through a Preset database connection.

    This is a debugging/verification tool — use it to confirm a query
    works through Preset's connection before creating a dataset.  For
    primary SQL exploration, prefer your Snowflake MCP (igloo-mcp).

    Only SELECT, SHOW, DESCRIBE, and EXPLAIN are permitted.  Write
    operations and multi-statement queries are blocked.

    Args:
        sql: SQL query to execute (read-only, single statement)
        database_id: Database connection ID — use list_databases to find it
        schema: Optional schema name
        limit: Max rows to return (default 1000)
        response_mode: 'compact' (columns only), 'standard' (5 sample rows),
                       or 'full' (all rows with smart truncation).
                       Default: standard.
    """
    _validate_readonly(sql)
    ws = _get_ws()
    df = ws.run_sql(sql, database_id, schema=schema, limit=limit)
    records = df.to_dict("records")
    columns = df.columns.tolist()
    _log.info(
        "run_sql rows=%d cols=%d mode=%s truncated=%s",
        len(records),
        len(columns),
        response_mode,
        len(records) > SQL_SAMPLE_ROWS if response_mode == "standard" else "n/a",
    )
    return _format_sql(records, columns, response_mode)


@mcp.tool()
@_handle_errors
def query_dataset(
    dataset_id: int,
    metrics: list[str] | str,
    columns: list[str] | str | None = None,
    time_column: str | None = None,
    start: str | None = None,
    end: str | None = None,
    granularity: str | None = None,
    where: str = "",
    having: str = "",
    order_by: list[str] | str | None = None,
    order_desc: bool = True,
    row_limit: int = 10000,
    force: bool = False,
    response_mode: ResponseMode = "standard",
) -> str:
    """Query a dataset using Superset's metric/dimension abstraction.

    This executes the same query path that charts use — aggregating
    metrics over dimension columns — without needing raw SQL.  Use
    get_dataset to discover available metrics and columns first.

    For time-series queries, set time_column plus start/end (ISO-8601)
    and optionally granularity (e.g. "P1D", "P1W").

    Args:
        dataset_id: ID of the dataset to query
        metrics: Metric names to aggregate (e.g. ["count", "revenue"])
        columns: Dimension columns to group by
        time_column: Time column for time-filtered or time-series queries
        start: Start date/time in ISO-8601 format (e.g. "2025-01-01")
        end: End date/time in ISO-8601 format (e.g. "2025-06-01")
        granularity: Time grain (e.g. "P1D", "P1W", "P1M")
        where: SQL WHERE clause fragment for filtering
        having: SQL HAVING clause fragment for post-aggregation filtering
        order_by: Columns or metrics to order by
        order_desc: If True, sort descending (default: True)
        row_limit: Max rows to return (default: 10000)
        force: If True, bypass query cache
        response_mode: 'compact' (columns only), 'standard' (sample rows),
                       or 'full' (all rows).  Default: standard.
    """
    ws = _get_ws()
    metrics_list = _coerce_list_arg(
        metrics, field_name="metrics", item_kind="str"
    )
    if not metrics_list:
        raise ValueError("metrics must contain at least one metric.")
    columns_list = _coerce_list_arg(
        columns, field_name="columns", item_kind="str"
    )
    order_by_list = _coerce_list_arg(
        order_by, field_name="order_by", item_kind="str"
    )
    is_timeseries = bool(time_column and granularity)
    df = ws.query_dataset(
        dataset_id=dataset_id,
        metrics=metrics_list,
        columns=columns_list,
        order_by=order_by_list,
        order_desc=order_desc,
        is_timeseries=is_timeseries,
        time_column=time_column,
        start=start,
        end=end,
        granularity=granularity,
        where=where,
        having=having,
        row_limit=row_limit,
        force=force,
    )
    records = df.to_dict("records")
    col_names = df.columns.tolist()
    _log.info(
        "query_dataset dataset_id=%d rows=%d cols=%d mode=%s",
        dataset_id, len(records), len(col_names), response_mode,
    )
    return _format_sql(records, col_names, response_mode)


# ===================================================================
# Tools — Validation
# ===================================================================


@mcp.tool()
@_handle_errors
def validate_chart(
    chart_id: int,
    dashboard_id: int | None = None,
    row_limit: int = 10000,
    force: bool = False,
    response_mode: ResponseMode = "standard",
) -> str:
    """Run chart query validation and return render status.

    This validates the saved chart params against chart-data execution, returning
    actionable errors for missing metrics/columns/etc.

    Notes:
      - This does not take a screenshot; it executes the same query context
        used for chart rendering.
      - `dashboard_id` speeds lookup of chart form_data. If omitted, all
        dashboards are scanned.

    Args:
        chart_id: Chart ID to validate
        dashboard_id: Optional dashboard context for form_data lookup
        row_limit: Query row limit used for validation
        force: Whether to force recomputation (if supported by backend)
        response_mode: 'compact' (small summary), 'standard' (default),
                       or 'full' (raw payload + metadata)
    """
    ws = _get_ws()
    result = ws.validate_chart_data(
        chart_id,
        dashboard_id=dashboard_id,
        row_limit=row_limit,
        force=force,
    )
    if response_mode == "compact":
        status = result.get("status")
        return json.dumps({
            "chart_id": result.get("chart_id"),
            "slice_name": result.get("slice_name"),
            "dashboard_id": result.get("dashboard_id"),
            "status": status,
            "error": result.get("error"),
        }, indent=2, default=str)
    if response_mode == "standard":
        return json.dumps({
            "chart_id": result.get("chart_id"),
            "slice_name": result.get("slice_name"),
            "dashboard_id": result.get("dashboard_id"),
            "datasource": result.get("datasource"),
            "status": result.get("status"),
            "error": result.get("error"),
            "row_count": result.get("row_count"),
            "row_count_total": result.get("row_count_total"),
            "cache_key": result.get("cache_key"),
            "is_cached": result.get("is_cached"),
            "http_status": result.get("http_status"),
            "payload_source": result.get("payload_source"),
            "form_data_source": result.get("form_data_source"),
            "query_context_present": result.get("query_context_present"),
        }, indent=2, default=str)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
@_handle_errors
def validate_dashboard(
    dashboard_id: int,
    row_limit: int = 10000,
    force: bool = False,
    response_mode: ResponseMode = "standard",
) -> str:
    """Validate all charts on a dashboard and return aggregate statuses."""
    ws = _get_ws()
    result = ws.validate_dashboard_charts(
        dashboard_id,
        row_limit=row_limit,
        force=force,
    )

    if response_mode == "compact":
        errors = [
            r for r in result["results"]
            if r.get("error") is not None
            or (r.get("status") and r.get("status") != "success")
        ]
        return json.dumps({
            "dashboard_id": result["dashboard_id"],
            "chart_count": result["chart_count"],
            "validated": result["validated"],
            "broken_count": len(errors),
            "broken_charts": errors,
        }, indent=2, default=str)

    if response_mode == "standard":
        summary = {
            "dashboard_id": result["dashboard_id"],
            "chart_count": result["chart_count"],
            "validated": result["validated"],
            "results": [
                {
                    "chart_id": item.get("chart_id"),
                    "slice_name": item.get("slice_name"),
                    "status": item.get("status"),
                    "row_count": item.get("row_count"),
                    "cache_key": item.get("cache_key"),
                    "error": item.get("error"),
                }
                for item in result.get("results", [])
            ],
        }
        return json.dumps(summary, indent=2, default=str)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
@_handle_errors
def validate_chart_render(
    chart_id: int,
    timeout_ms: int = 45000,
    settle_ms: int = 2500,
    response_mode: ResponseMode = "standard",
) -> str:
    """Validate chart rendering with a headless browser (frontend-level probe)."""
    ws = _get_ws()
    result = ws.validate_chart_render(
        chart_id,
        timeout_ms=timeout_ms,
        settle_ms=settle_ms,
    )
    if response_mode == "compact":
        return json.dumps({
            "chart_id": result.get("chart_id"),
            "slice_name": result.get("slice_name"),
            "status": result.get("status"),
            "error": result.get("error"),
            "screenshot_path": result.get("screenshot_path"),
        }, indent=2, default=str)
    if response_mode == "standard":
        return json.dumps({
            "chart_id": result.get("chart_id"),
            "slice_name": result.get("slice_name"),
            "status": result.get("status"),
            "error": result.get("error"),
            "visible_errors": result.get("visible_errors"),
            "critical_page_errors": result.get("critical_page_errors"),
            "page_errors": result.get("page_errors"),
            "chart_data_failures": result.get("chart_data_failures"),
            "screenshot_path": result.get("screenshot_path"),
        }, indent=2, default=str)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
@_handle_errors
def validate_dashboard_render(
    dashboard_id: int,
    timeout_ms: int = 45000,
    settle_ms: int = 2500,
    response_mode: ResponseMode = "standard",
) -> str:
    """Validate frontend render status for all charts on a dashboard."""
    ws = _get_ws()
    result = ws.validate_dashboard_render(
        dashboard_id,
        timeout_ms=timeout_ms,
        settle_ms=settle_ms,
    )
    if response_mode == "compact":
        return json.dumps({
            "dashboard_id": result.get("dashboard_id"),
            "chart_count": result.get("chart_count"),
            "validated": result.get("validated"),
            "broken_count": result.get("broken_count"),
        }, indent=2, default=str)
    if response_mode == "standard":
        broken_summaries: list[dict[str, Any]] = []
        for item in result.get("broken_charts", []):
            if not isinstance(item, dict):
                continue
            broken_summaries.append({
                "chart_id": item.get("chart_id"),
                "slice_name": item.get("slice_name"),
                "status": item.get("status"),
                "error": item.get("error"),
                "visible_errors": item.get("visible_errors"),
                "critical_page_errors": item.get("critical_page_errors"),
                "page_errors": item.get("page_errors"),
                "chart_data_failures": item.get("chart_data_failures"),
                "screenshot_path": item.get("screenshot_path"),
            })
        return json.dumps({
            "dashboard_id": result.get("dashboard_id"),
            "chart_count": result.get("chart_count"),
            "validated": result.get("validated"),
            "broken_count": result.get("broken_count"),
            "broken_charts": broken_summaries,
        }, indent=2, default=str)
    return json.dumps(result, indent=2, default=str)


# ===================================================================
# Tools — Create operations
# ===================================================================


@mcp.tool()
@_handle_errors
def create_dashboard(
    dashboard_title: str,
    published: bool = False,
    dry_run: bool = False,
) -> str:
    """Create a new, empty dashboard.

    After creating, use create_chart with the dashboards parameter to
    add charts, or use update_dashboard to change properties.

    Args:
        dashboard_title: Display title for the dashboard
        published: Whether the dashboard is published (default: False)
        dry_run: If True, validate inputs and return a preview without
                 making any changes (default: False)
    """
    ws = _get_ws()
    return _do_mutation(
        tool_name="create_dashboard",
        resource_type="dashboard",
        action="create",
        fields_changed=["dashboard_title", "published"],
        dry_run=dry_run,
        execute=lambda: ws.create_dashboard(dashboard_title, published=published),
        preview_extras={"values": {"dashboard_title": dashboard_title, "published": published}},
        after_extras={"dashboard_title": dashboard_title},
    )


@mcp.tool()
@_handle_errors
def create_dataset(
    name: str,
    sql: str,
    database_id: int,
    schema: str | None = None,
    dry_run: bool = False,
) -> str:
    """Create a virtual (SQL-based) dataset for charting.

    This is the main entry point from analytics into visualization.
    Take a validated SQL query (e.g. from igloo-mcp) and register it
    as a Preset dataset.  Use list_databases to find database_id.

    Write SQL (INSERT, DROP, etc.) is blocked — only SELECT-style
    queries are permitted as dataset definitions.

    Args:
        name: Display name for the dataset
        sql: SQL query defining the dataset
        database_id: Database connection ID
        schema: Optional schema name
        dry_run: If True, validate inputs and return a preview without
                 making any changes (default: False)
    """
    _validate_readonly(sql)
    ws = _get_ws()
    _require_database_exists(ws, database_id)
    return _do_mutation(
        tool_name="create_dataset",
        resource_type="dataset",
        action="create",
        fields_changed=["name", "sql", "database_id", "schema"],
        dry_run=dry_run,
        execute=lambda: ws.create_dataset(name, sql, database_id, schema=schema),
        preview_extras={"values": {
            "name": name, "sql": sql,
            "database_id": database_id, "schema": schema,
        }},
        after_extras={"name": name},
    )


@mcp.tool()
@_handle_errors
def create_chart(
    dataset_id: int,
    title: str,
    viz_type: str,
    metrics: list[str] | str | None = None,
    groupby: list[str] | str | None = None,
    time_column: str | None = None,
    dashboards: list[int] | str | None = None,
    validate_after_create: bool = True,
    repair_dashboard_refs: bool = True,
    dry_run: bool = False,
) -> str:
    """Create a chart from an existing dataset.

    Recommended workflow: create_dataset → create_chart (with dashboards
    param to attach it).  Use list_datasets to find dataset_id and
    list_dashboards to find dashboard IDs.

    Args:
        dataset_id: ID of the dataset to visualize
        title: Chart title
        viz_type: Visualization type (e.g. "echarts_timeseries_bar",
                  "pie", "big_number_total", "table")
        metrics: Metric column names
        groupby: Columns to group by
        time_column: Time column for time-series charts
        dashboards: Dashboard IDs to attach this chart to
        validate_after_create: Run chart-data validation after create
        repair_dashboard_refs: Attempt to repair stale dashboard chart
                               references when dashboards are provided
        dry_run: If True, validate inputs and return a preview without
                 making any changes (default: False)
    """
    ws = _get_ws()
    _require_dataset_exists(ws, dataset_id)
    _validate_viz_type(ws, viz_type)

    metrics_list = _coerce_list_arg(
        metrics, field_name="metrics", item_kind="str"
    )
    groupby_list = _coerce_list_arg(
        groupby, field_name="groupby", item_kind="str"
    )
    dashboards_list = _coerce_list_arg(
        dashboards, field_name="dashboards", item_kind="int"
    )
    if dashboards_list:
        _require_dashboards_exist(ws, dashboards_list)

    fields = ["dataset_id", "title", "viz_type"]
    if metrics_list is not None:
        fields.append("metrics")
    if groupby_list is not None:
        fields.append("groupby")
    if time_column is not None:
        fields.append("time_column")
    if dashboards_list is not None:
        fields.append("dashboards")
    if validate_after_create:
        fields.append("validate_after_create")
    if repair_dashboard_refs and dashboards_list:
        fields.append("repair_dashboard_refs")

    raw = _do_mutation(
        tool_name="create_chart",
        resource_type="chart",
        action="create",
        fields_changed=fields,
        dry_run=dry_run,
        execute=lambda: ws.create_chart(
            dataset_id, title, viz_type,
            metrics=metrics_list, groupby=groupby_list,
            time_column=time_column, dashboards=dashboards_list,
        ),
        preview_extras={"values": {
            "dataset_id": dataset_id, "title": title, "viz_type": viz_type,
            "metrics": metrics_list, "groupby": groupby_list,
            "time_column": time_column, "dashboards": dashboards_list,
            "validate_after_create": validate_after_create,
            "repair_dashboard_refs": repair_dashboard_refs,
        }},
        after_extras={"title": title},
    )
    if dry_run:
        return raw

    payload = json.loads(raw)
    chart_id = _to_int(payload.get("id"))
    if chart_id is None:
        return raw

    if repair_dashboard_refs and dashboards_list:
        repairs: list[dict[str, Any]] = []
        for dashboard_id in dashboards_list:
            dashboard = ws.dashboard_detail(dashboard_id)
            charts = ws.dashboard_charts(dashboard_id)
            position, metadata, summary = _repair_dashboard_refs(
                dashboard.get("position_json", {}),
                dashboard.get("json_metadata", {}),
                charts,
                strategy="replace_by_name",
            )
            if summary.get("changed"):
                ws.update_dashboard(
                    dashboard_id,
                    position_json=json.dumps(position),
                    json_metadata=json.dumps(metadata),
                )
            summary["dashboard_id"] = dashboard_id
            repairs.append(summary)
        payload["_dashboard_repairs"] = repairs

    if validate_after_create:
        dashboard_id = dashboards_list[0] if dashboards_list else None
        payload["_validation"] = ws.validate_chart_data(
            chart_id,
            dashboard_id=dashboard_id,
        )
    return json.dumps(payload, indent=2, default=str)


# ===================================================================
# Tools — Update operations
# ===================================================================


@mcp.tool()
@_handle_errors
def update_dataset(
    dataset_id: int,
    sql: str | None = None,
    name: str | None = None,
    description: str | None = None,
    override_columns: bool = False,
    dry_run: bool = False,
) -> str:
    """Update an existing dataset's SQL, name, or description.

    Use list_datasets to find the dataset_id.  After updating the SQL,
    any charts built on this dataset will reflect the new data on their
    next refresh.

    Args:
        dataset_id: ID of the dataset to update
        sql: New SQL query (replaces existing)
        name: New display name
        description: New description
        override_columns: If True, refresh column metadata from the
                          new SQL (recommended when changing SQL)
        dry_run: If True, validate inputs, capture current state, and
                 check dependencies without making any changes
                 (default: False)
    """
    kwargs: dict[str, Any] = {}
    if sql is not None:
        _validate_readonly(sql)
        kwargs["sql"] = sql
    if name is not None:
        kwargs["table_name"] = name
    if description is not None:
        kwargs["description"] = description

    if not kwargs:
        raise ValueError(
            "Provide at least one field to update (sql, name, or description)."
        )

    ws = _get_ws()
    _require_dataset_exists(ws, dataset_id)
    before = capture_before(ws, "dataset", dataset_id)

    # Dependency check when SQL changes or columns are overridden
    dependency_impact: dict[str, Any] | None = None
    if sql is not None or override_columns:
        dependency_impact = check_dataset_dependents(ws, dataset_id)

    # Build result_extras for advisory warnings
    extras: dict[str, Any] = {}
    if dependency_impact:
        extras["_dependency_impact"] = dependency_impact
    if (
        override_columns
        and dependency_impact
        and dependency_impact.get("chart_count", 0) > 0
    ):
        extras["_override_columns_warning"] = (
            f"override_columns=True will refresh column metadata. "
            f"{dependency_impact['chart_count']} chart(s) may be affected: "
            f"{[c['name'] for c in dependency_impact.get('affected_charts', [])]}"
        )

    return _do_mutation(
        tool_name="update_dataset",
        resource_type="dataset",
        action="update",
        fields_changed=list(kwargs.keys()),
        dry_run=dry_run,
        execute=lambda: ws.update_dataset(
            dataset_id, override_columns=override_columns, **kwargs
        ),
        resource_id=dataset_id,
        before=before,
        preview_extras={"dependency_impact": dependency_impact} if dependency_impact else None,
        result_extras=extras or None,
    )


@mcp.tool()
@_handle_errors
def update_chart(
    chart_id: int,
    title: str | None = None,
    viz_type: str | None = None,
    params_json: str | None = None,
    dashboards: list[int] | str | None = None,
    validate_after_update: bool = True,
    dry_run: bool = False,
) -> str:
    """Update an existing chart's title, viz type, or parameters.

    Use list_charts to find the chart_id.  Pass params_json as a JSON
    string to override visualization parameters (metrics, groupby,
    filters, etc.).

    Args:
        chart_id: ID of the chart to update
        title: New chart title
        viz_type: New visualization type
        params_json: JSON string of chart parameters (advanced — use
                     get_chart to inspect existing chart params first)
        dashboards: Reassign chart to these dashboard IDs
        validate_after_update: Run chart-data validation after update
        dry_run: If True, validate inputs, capture current state, and
                 return a preview without making any changes
                 (default: False)
    """
    ws = _get_ws()
    _require_chart_exists(ws, chart_id)
    before = capture_before(ws, "chart", chart_id)

    chart_dataset_id = _to_int(before.get("datasource_id"))
    if chart_dataset_id is None:
        ds_from_params, _ = _parse_chart_datasource_from_params(before.get("params", {}))
        chart_dataset_id = ds_from_params

    dashboards_list = _coerce_list_arg(
        dashboards, field_name="dashboards", item_kind="int"
    )
    if dashboards_list:
        _require_dashboards_exist(ws, dashboards_list)

    if viz_type is not None:
        _validate_viz_type(ws, viz_type)

    params_warnings: list[str] = []
    if params_json is not None:
        dataset_columns: set[str] = set()
        dataset_metrics: set[str] = set()
        if chart_dataset_id is not None:
            try:
                dataset = ws.dataset_detail(chart_dataset_id)
                dataset_columns = _dataset_columns(dataset)
                dataset_metrics = _dataset_metrics(dataset)
            except Exception:
                dataset_columns = set()
                dataset_metrics = set()
        _, params_warnings = validate_params_payload(
            params_json,
            dataset_columns=dataset_columns,
            dataset_metrics=dataset_metrics,
        )

    kwargs: dict[str, Any] = {}
    if title is not None:
        kwargs["slice_name"] = title
    if viz_type is not None:
        kwargs["viz_type"] = viz_type
    if params_json is not None:
        kwargs["params"] = params_json
    if dashboards_list is not None:
        kwargs["dashboards"] = dashboards_list

    if not kwargs:
        raise ValueError(
            "Provide at least one field to update "
            "(title, viz_type, params_json, or dashboards)."
        )
    raw = _do_mutation(
        tool_name="update_chart",
        resource_type="chart",
        action="update",
        fields_changed=list(kwargs.keys()),
        dry_run=dry_run,
        execute=lambda: ws.update_chart(chart_id, **kwargs),
        resource_id=chart_id,
        before=before,
    )
    if dry_run:
        return raw

    payload = json.loads(raw)
    if params_warnings:
        payload["_params_warnings"] = params_warnings

    if validate_after_update:
        dashboard_id = dashboards_list[0] if dashboards_list else None
        payload["_validation"] = ws.validate_chart_data(
            chart_id,
            dashboard_id=dashboard_id,
        )
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
@_handle_errors
def update_dashboard(
    dashboard_id: int,
    dashboard_title: str | None = None,
    published: bool | None = None,
    position_json: str | None = None,
    json_metadata: str | None = None,
    allow_empty_layout: bool = False,
    dry_run: bool = False,
) -> str:
    """Update an existing dashboard's properties.

    Use list_dashboards to find the dashboard_id.
    Use get_dashboard to inspect the current position_json and json_metadata.

    Args:
        dashboard_id: ID of the dashboard to update
        dashboard_title: New dashboard title
        published: Set to True to publish, False to unpublish
        position_json: JSON string defining the dashboard layout (chart
            containers, rows, grid). Use this to add, remove, or
            rearrange chart containers — e.g. after deleting charts
            whose containers remain as orphaned placeholders.
        json_metadata: JSON string with dashboard metadata (cross-filter
            config, color schemes, label colors, refresh settings).
            Update this alongside position_json to keep chart references
            in sync.
        allow_empty_layout: If True, allow updates that remove all chart
            containers from the layout tree (default: False).
        dry_run: If True, validate inputs, capture current state, and
                 return a preview without making any changes
                 (default: False)
    """
    kwargs: dict[str, Any] = {}
    if dashboard_title is not None:
        kwargs["dashboard_title"] = dashboard_title
    if published is not None:
        kwargs["published"] = published
    if position_json is not None:
        kwargs["position_json"] = position_json
    if json_metadata is not None:
        kwargs["json_metadata"] = json_metadata

    if not kwargs:
        raise ValueError(
            "Provide at least one field to update "
            "(dashboard_title, published, position_json, or json_metadata)."
        )

    ws = _get_ws()
    before = capture_before(ws, "dashboard", dashboard_id)

    if position_json is not None:
        proposed_position = _ensure_json_dict(position_json, "position_json")
        proposed_chart_ids = _layout_chart_ids(proposed_position)

        existing_layout_ids = _layout_chart_ids(before.get("position_json", {}))
        attached_chart_ids: set[int] = set()
        if not existing_layout_ids:
            try:
                for chart in ws.dashboard_charts(dashboard_id):
                    chart_id = _to_int(chart.get("id"))
                    if chart_id is not None:
                        attached_chart_ids.add(chart_id)
            except Exception:
                attached_chart_ids = set()

        dashboard_has_charts = bool(existing_layout_ids or attached_chart_ids)
        if dashboard_has_charts and not proposed_chart_ids and not allow_empty_layout:
            raise ValueError(
                "Blocked potentially destructive layout update: position_json "
                "removes all chart containers from a dashboard that currently "
                "has charts. If this is intentional, re-run with "
                "allow_empty_layout=True."
            )

    return _do_mutation(
        tool_name="update_dashboard",
        resource_type="dashboard",
        action="update",
        fields_changed=list(kwargs.keys()),
        dry_run=dry_run,
        execute=lambda: ws.update_dashboard(dashboard_id, **kwargs),
        resource_id=dashboard_id,
        before=before,
    )


# ===================================================================
# Tools — Dashboard repair
# ===================================================================


@mcp.tool()
@_handle_errors
def repair_dashboard_chart_refs(
    dashboard_id: int,
    strategy: Literal["replace_by_name", "remove_orphans"] = "replace_by_name",
    dry_run: bool = False,
) -> str:
    """Repair stale chart IDs in a dashboard's layout metadata.

    This tool syncs `position_json` and `json_metadata.chartsInScope` by
    replacing or removing orphaned chart references.

    Args:
        dashboard_id: Dashboard ID to repair
        strategy: 'replace_by_name' (default) maps stale chart IDs to
                  currently attached charts with matching names; or
                  'remove_orphans' to drop stale references only.
        dry_run: If True, preview detected changes without updating.
    """
    ws = _get_ws()
    dashboard = ws.dashboard_detail(dashboard_id)
    charts = ws.dashboard_charts(dashboard_id)
    before = capture_before(ws, "dashboard", dashboard_id)

    position, metadata, summary = _repair_dashboard_refs(
        dashboard.get("position_json", {}),
        dashboard.get("json_metadata", {}),
        charts,
        strategy=strategy,
    )

    if not summary.get("changed"):
        payload = {
            "status": "noop",
            "dashboard_id": dashboard_id,
            "repair_summary": summary,
        }
        if dry_run:
            payload["dry_run"] = True
        return json.dumps(payload, indent=2, default=str)

    return _do_mutation(
        tool_name="repair_dashboard_chart_refs",
        resource_type="dashboard",
        action="update",
        fields_changed=["position_json", "json_metadata"],
        dry_run=dry_run,
        execute=lambda: ws.update_dashboard(
            dashboard_id,
            position_json=json.dumps(position),
            json_metadata=json.dumps(metadata),
        ),
        resource_id=dashboard_id,
        before=before,
        preview_extras={
            "repair_summary": summary,
            "strategy": strategy,
        },
        result_extras={
            "repair_summary": summary,
        },
    )


# ===================================================================
# Tools — Snapshot
# ===================================================================


@mcp.tool()
@_handle_errors
def snapshot_workspace() -> str:
    """Capture a full inventory of the current workspace.

    Returns counts and complete lists of dashboards, charts, datasets,
    and databases.  Heavier than workspace_catalog — use catalog for
    navigation, snapshot for full audit.
    """
    ws = _get_ws()
    snap = ws.snapshot()
    _log.info("snapshot counts=%s", snap.counts)
    return json.dumps(snap.model_dump(), indent=2, default=str)


# ===================================================================
# Tools — Delete operations (opt-in via PRESET_MCP_ENABLE_DELETE)
# ===================================================================

_DELETE_ENABLED = os.environ.get("PRESET_MCP_ENABLE_DELETE", "").lower() in (
    "true", "1", "yes",
)

if _DELETE_ENABLED:

    @mcp.tool()
    @_handle_errors
    def delete_dashboard(dashboard_id: int, dry_run: bool = False) -> str:
        """Delete a dashboard after exporting a full backup.

        A ZIP backup is saved to ~/.preset-mcp/audit/exports/ BEFORE the
        delete proceeds.  If the export fails, the dashboard is NOT deleted.

        Requires PRESET_MCP_ENABLE_DELETE=true to be available.

        Args:
            dashboard_id: ID of the dashboard to delete
            dry_run: If True, export backup and return preview without
                     actually deleting (default: False)
        """
        ws = _get_ws()
        before = capture_before(ws, "dashboard", dashboard_id)
        ep = export_before_delete(ws, "dashboard", dashboard_id)

        return _do_mutation(
            tool_name="delete_dashboard",
            resource_type="dashboard",
            action="delete",
            fields_changed=[],
            dry_run=dry_run,
            execute=lambda: ws.delete_resource("dashboard", dashboard_id) or {},
            resource_id=dashboard_id,
            before=before,
            export_path=ep,
        )

    @mcp.tool()
    @_handle_errors
    def delete_chart(chart_id: int, dry_run: bool = False) -> str:
        """Delete a chart after exporting a full backup.

        A ZIP backup is saved to ~/.preset-mcp/audit/exports/ BEFORE the
        delete proceeds.  If the export fails, the chart is NOT deleted.

        Requires PRESET_MCP_ENABLE_DELETE=true to be available.

        Args:
            chart_id: ID of the chart to delete
            dry_run: If True, export backup and return preview without
                     actually deleting (default: False)
        """
        ws = _get_ws()
        before = capture_before(ws, "chart", chart_id)
        ep = export_before_delete(ws, "chart", chart_id)

        return _do_mutation(
            tool_name="delete_chart",
            resource_type="chart",
            action="delete",
            fields_changed=[],
            dry_run=dry_run,
            execute=lambda: ws.delete_resource("chart", chart_id) or {},
            resource_id=chart_id,
            before=before,
            export_path=ep,
        )

    @mcp.tool()
    @_handle_errors
    def delete_dataset(
        dataset_id: int,
        force: bool = False,
        dry_run: bool = False,
    ) -> str:
        """Delete a dataset after exporting a full backup.

        If charts depend on this dataset, the delete is blocked unless
        force=True.  A ZIP backup is always saved before deletion.

        Requires PRESET_MCP_ENABLE_DELETE=true to be available.

        Args:
            dataset_id: ID of the dataset to delete
            force: If True, delete even if charts depend on this dataset
                   (default: False)
            dry_run: If True, export backup and return preview without
                     actually deleting (default: False)
        """
        ws = _get_ws()

        deps = check_dataset_dependents(ws, dataset_id)
        if deps["chart_count"] > 0 and not force:
            raise ValueError(
                f"Cannot delete dataset {dataset_id}: "
                f"{deps['chart_count']} chart(s) depend on it: "
                f"{[c['name'] for c in deps['affected_charts']]}. "
                f"Use force=True to delete anyway."
            )

        before = capture_before(ws, "dataset", dataset_id)
        ep = export_before_delete(ws, "dataset", dataset_id)

        return _do_mutation(
            tool_name="delete_dataset",
            resource_type="dataset",
            action="delete",
            fields_changed=[],
            dry_run=dry_run,
            execute=lambda: ws.delete_resource("dataset", dataset_id) or {},
            resource_id=dataset_id,
            before=before,
            export_path=ep,
            preview_extras={"dependency_info": deps},
            result_extras={"dependency_info": deps},
            after_extras={"dependency_info": deps},
        )

    @mcp.tool()
    @_handle_errors
    def list_deleted_backups(resource_type: str | None = None) -> str:
        """List ZIP backups from previous delete operations.

        Scans ~/.preset-mcp/audit/exports/ for backup files.  Use the
        export_path with restore_from_backup to undo a deletion.

        Requires PRESET_MCP_ENABLE_DELETE=true to be available.

        Args:
            resource_type: Filter by type ('dashboard', 'chart', 'dataset').
                           If None, shows all backups.
        """
        exports_dir = AUDIT_DIR / "exports"
        if not exports_dir.exists():
            return json.dumps({"backups": [], "count": 0}, indent=2)

        backups: list[dict[str, Any]] = []
        for f in sorted(exports_dir.glob("*.zip")):
            parts = f.stem.rsplit("_", 2)
            if len(parts) == 3:
                rtype, rid, ts = parts
            else:
                rtype, rid, ts = f.stem, "unknown", "unknown"

            if resource_type and rtype != resource_type:
                continue

            backups.append({
                "filename": f.name,
                "resource_type": rtype,
                "resource_id": rid,
                "timestamp": ts,
                "size_bytes": f.stat().st_size,
                "export_path": str(f),
            })

        return json.dumps({
            "backups": backups,
            "count": len(backups),
        }, indent=2)

    @mcp.tool()
    @_handle_errors
    def restore_from_backup(
        export_path: str,
        resource_type: str,
        overwrite: bool = True,
    ) -> str:
        """Restore a previously deleted resource from a ZIP backup.

        Use list_deleted_backups to find the export_path and resource_type.

        Requires PRESET_MCP_ENABLE_DELETE=true to be available.

        Args:
            export_path: Full path to the ZIP backup file
            resource_type: Resource type ('dashboard', 'chart', 'dataset')
            overwrite: If True, overwrite existing resources with same IDs
                       (default: True)
        """
        from pathlib import Path as _Path

        backup = _Path(export_path)
        if not backup.exists():
            raise ValueError(f"Backup file not found: {export_path}")

        ws = _get_ws()
        data = backup.read_bytes()
        result = ws.import_resource_zip(resource_type, data, overwrite=overwrite)
        _log.info(
            "restore_from_backup type=%s path=%s overwrite=%s",
            resource_type, export_path, overwrite,
        )

        record_mutation(MutationEntry(
            tool_name="restore_from_backup",
            resource_type=resource_type,
            action="restore",
            after_summary={
                "export_path": export_path,
                "overwrite": overwrite,
                "success": result,
            },
        ))

        return json.dumps({
            "status": "restored",
            "resource_type": resource_type,
            "export_path": export_path,
            "overwrite": overwrite,
            "success": result,
        }, indent=2)


# ===================================================================
# Entry point
# ===================================================================


def main():
    mcp.run()


if __name__ == "__main__":
    main()
