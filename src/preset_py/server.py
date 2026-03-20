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
import re
import sys
import time
from datetime import datetime, timezone
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
    item_kind: Literal["str", "int", "any"],
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
        if item_kind == "any":
            normalized.append(item)
            continue

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


def _is_timeout_exception(exc: Exception) -> bool:
    """Best-effort classification for timeout-shaped post-validation failures."""
    type_name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        isinstance(exc, TimeoutError)
        or "timeout" in type_name
        or "timed out" in message
        or "timeout" in message
    )


def _post_validation_result(
    ws: PresetWorkspace,
    *,
    chart_id: int,
    dashboard_id: int | None,
    operation_name: str,
    operation_past_tense: str,
) -> dict[str, Any]:
    """Run post-mutation validation without losing the mutated chart id."""
    try:
        return ws.validate_chart_data(
            chart_id,
            dashboard_id=dashboard_id,
            persist_synthetic=True,
        )
    except Exception as exc:
        timed_out = _is_timeout_exception(exc)
        status = "timeout" if timed_out else "post_validation_failed"
        outcome = "timed out" if timed_out else "failed"
        return {
            "chart_id": chart_id,
            "status": status,
            "error_type": type(exc).__name__,
            "error": (
                f"Post-{operation_name} validation {outcome} "
                f"({type(exc).__name__}: {exc}). "
                f"The chart was {operation_past_tense} successfully. "
                "Use get_chart to inspect it and rerun validate_chart when ready."
            ),
        }


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


def _fetch_dataset_or_raise(ws: PresetWorkspace, dataset_id: int) -> dict[str, Any]:
    """Fetch dataset detail, raising ValueError if not found."""
    try:
        return ws.dataset_detail(dataset_id)
    except Exception as exc:
        raise ValueError(
            f"Dataset {dataset_id} not found. Use list_datasets to find valid IDs."
        ) from exc


def _require_database_exists(ws: PresetWorkspace, database_id: int) -> None:
    try:
        ws.database_detail(database_id)
    except Exception as exc:
        raise ValueError(
            f"Database {database_id} not found. Use list_databases to find valid IDs."
        ) from exc


def _require_dashboards_exist(ws: PresetWorkspace, dashboard_ids: list[int]) -> None:
    """Validate that all dashboard IDs exist using a single listing call."""
    known_ids = {
        _to_int(d.get("id"))
        for d in ws.dashboards()
        if _to_int(d.get("id")) is not None
    }
    missing = [did for did in dashboard_ids if did not in known_ids]
    if missing:
        raise ValueError(
            f"Dashboard(s) {missing} not found. Use list_dashboards to find valid IDs."
        )


_viz_type_cache: dict[str, tuple[set[str], float]] = {}
_VIZ_TYPE_CACHE_TTL = 300  # seconds


def _collect_workspace_viz_types(ws: PresetWorkspace) -> set[str]:
    import time

    cache_key = str(getattr(ws, "workspace_url", None) or "default")
    cached = _viz_type_cache.get(cache_key)
    if cached is not None:
        viz_types, cached_at = cached
        if time.monotonic() - cached_at < _VIZ_TYPE_CACHE_TTL:
            return viz_types

    viz_types = set(_KNOWN_VIZ_TYPES)
    try:
        for chart in ws.charts():
            viz = chart.get("viz_type")
            if isinstance(viz, str) and viz:
                viz_types.add(viz)
    except Exception:
        # Non-blocking fallback to curated defaults.
        pass
    _viz_type_cache[cache_key] = (viz_types, time.monotonic())
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

    if ds_id is None or not ds_type:
        # Only scan the full chart listing when we're missing critical fields
        # (datasource_id or datasource_type). Skip if we only need the name.
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


def _find_duplicate_chart_placements(
    position_json: Any,
) -> list[dict[str, Any]]:
    """Find chart IDs placed in multiple layout nodes."""
    refs = _extract_position_chart_refs(position_json)
    chart_to_nodes: dict[int, list[str]] = {}
    for node_id, chart_id in refs.items():
        chart_to_nodes.setdefault(chart_id, []).append(node_id)
    return [
        {"chart_id": chart_id, "layout_nodes": sorted(nodes)}
        for chart_id, nodes in sorted(chart_to_nodes.items())
        if len(nodes) > 1
    ]


def _deduplicate_layout_containers(position_json: dict[str, Any]) -> dict[str, Any]:
    """Remove duplicate chart container nodes from a dashboard layout.

    After a ZIP import, Superset may duplicate ROW/CHART containers.
    This keeps the first occurrence of each chart_id (by sorted node key)
    and prunes the duplicate nodes plus their parent references.
    """
    if not position_json:
        return position_json

    refs = _extract_position_chart_refs(position_json)
    chart_to_nodes: dict[int, list[str]] = {}
    for node_id, chart_id in refs.items():
        chart_to_nodes.setdefault(chart_id, []).append(node_id)

    nodes_to_remove: set[str] = set()
    for chart_id, node_ids in chart_to_nodes.items():
        if len(node_ids) <= 1:
            continue
        # Keep the first (sorted) node, remove the rest
        for dup_node in sorted(node_ids)[1:]:
            nodes_to_remove.add(dup_node)

    if not nodes_to_remove:
        return position_json

    cleaned = dict(position_json)
    for node_id in nodes_to_remove:
        cleaned.pop(node_id, None)

    # Remove references to deleted nodes from parent children lists
    for node_id, node in cleaned.items():
        if not isinstance(node, dict):
            continue
        children = node.get("children")
        if isinstance(children, list):
            filtered = [c for c in children if str(c) not in nodes_to_remove]
            if len(filtered) != len(children):
                node["children"] = filtered

    # Prune empty wrapper rows that no longer have children
    empty_wrappers: set[str] = set()
    for node_id, node in list(cleaned.items()):
        if not isinstance(node, dict):
            continue
        ntype = node.get("type", "")
        if ntype in ("ROW", "COLUMN") and node.get("children") == []:
            # Only remove if the wrapper is not ROOT_ID or GRID_ID
            if node_id not in ("ROOT_ID", "GRID_ID"):
                empty_wrappers.add(node_id)

    for node_id in empty_wrappers:
        cleaned.pop(node_id, None)

    # Clean up parent references to removed wrappers
    for node_id, node in cleaned.items():
        if not isinstance(node, dict):
            continue
        children = node.get("children")
        if isinstance(children, list):
            filtered = [c for c in children if str(c) not in empty_wrappers]
            if len(filtered) != len(children):
                node["children"] = filtered

    return cleaned


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


def _validate_position_layout(position_json: dict[str, Any]) -> None:
    """Preflight-check dashboard layout structure for common invalid shapes."""
    if not position_json:
        return

    missing = [node for node in ("ROOT_ID", "GRID_ID") if node not in position_json]
    if missing:
        raise ValueError(
            "position_json is missing required nodes: "
            f"{missing}. Expected at least ROOT_ID and GRID_ID."
        )

    root = position_json.get("ROOT_ID")
    grid = position_json.get("GRID_ID")
    if not isinstance(root, dict) or not isinstance(grid, dict):
        raise ValueError("position_json ROOT_ID and GRID_ID must be JSON objects.")

    root_children = root.get("children")
    if not isinstance(root_children, list) or "GRID_ID" not in [
        str(child) for child in root_children
    ]:
        raise ValueError(
            "position_json ROOT_ID.children must include GRID_ID."
        )

    for node_id, node in position_json.items():
        if node_id == "DASHBOARD_VERSION_KEY":
            continue
        if not isinstance(node, dict):
            continue

        # Every layout node must carry an ``id`` that matches its key
        # and a ``type`` string so the Preset frontend can render it
        # (see GitHub issue #27).
        node_id_field = node.get("id")
        if node_id_field is None:
            raise ValueError(
                f"position_json[{node_id!r}] is missing required 'id' field. "
                "Every layout node must include an 'id' matching its key."
            )
        if str(node_id_field) != node_id:
            raise ValueError(
                f"position_json[{node_id!r}].id is {node_id_field!r} but must "
                f"match the node key {node_id!r}."
            )
        if not isinstance(node.get("type"), str) or not node.get("type"):
            raise ValueError(
                f"position_json[{node_id!r}] is missing required 'type' field."
            )

        children = node.get("children")
        if children is None:
            children = []
        if not isinstance(children, list):
            raise ValueError(
                f"position_json[{node_id!r}].children must be a list."
            )
        for child in children:
            child_id = str(child)
            if child_id not in position_json:
                raise ValueError(
                    "position_json has dangling child reference: "
                    f"{node_id!r} -> {child_id!r}."
                )

        parents = node.get("parents")
        if parents is not None and not isinstance(parents, list):
            raise ValueError(
                f"position_json[{node_id!r}].parents must be a list when provided."
            )


_TEMPLATE_DROP_COMMON_KEYS = frozenset({
    "slice_id",
    "slice_name",
    "cache_key",
    "cache_timeout",
    "changed_on",
    "changed_on_delta_humanized",
    "owners",
    "last_saved_at",
    "last_saved_by",
})

_TEMPLATE_DROP_PORTABLE_KEYS = frozenset({
    "datasource",
    "datasource_id",
    "datasource_type",
    "database_id",
    "dashboards",
    "uuid",
    "id",
})


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "dashboard"


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _sanitize_template_payload(
    value: Any,
    *,
    portable: bool,
) -> Any:
    """Normalize template payloads by dropping volatile runtime keys."""
    drop_keys = set(_TEMPLATE_DROP_COMMON_KEYS)
    if portable:
        drop_keys.update(_TEMPLATE_DROP_PORTABLE_KEYS)

    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in drop_keys:
                continue
            cleaned[key] = _sanitize_template_payload(item, portable=portable)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_template_payload(item, portable=portable) for item in value]
    return value


def _dashboard_structure_report(
    position_json: Any,
    json_metadata: Any,
    dashboard_charts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze dashboard layout graph and chart reference integrity."""
    position = _ensure_json_dict(position_json, "position_json")
    metadata = _ensure_json_dict(json_metadata, "json_metadata")
    nodes = {
        str(node_id): node
        for node_id, node in position.items()
        if isinstance(node, dict)
    }

    missing_required_nodes = [
        node_id for node_id in ("ROOT_ID", "GRID_ID")
        if node_id not in nodes
    ]
    missing_id_nodes: list[str] = []
    id_key_mismatches: list[dict[str, Any]] = []
    missing_type_nodes: list[str] = []
    invalid_children_nodes: list[str] = []
    invalid_parent_nodes: list[str] = []
    dangling_children: list[dict[str, str]] = []
    parent_mismatches: list[dict[str, Any]] = []

    for node_id, node in nodes.items():
        node_id_field = node.get("id")
        if node_id_field is None:
            missing_id_nodes.append(node_id)
        elif str(node_id_field) != node_id:
            id_key_mismatches.append({
                "node_key": node_id,
                "node_id": node_id_field,
            })
        if not isinstance(node.get("type"), str) or not node.get("type"):
            missing_type_nodes.append(node_id)
        children = node.get("children")
        if children is None:
            children = []
        if not isinstance(children, list):
            invalid_children_nodes.append(node_id)
            children = []
        for child in children:
            child_id = str(child)
            if child_id not in nodes:
                dangling_children.append({
                    "parent_node": node_id,
                    "child_node": child_id,
                })
                continue
            child_parents = nodes[child_id].get("parents")
            if child_parents is not None and not isinstance(child_parents, list):
                invalid_parent_nodes.append(child_id)
                continue
            if isinstance(child_parents, list):
                normalized_parents = {str(item) for item in child_parents}
                if node_id not in normalized_parents:
                    parent_mismatches.append({
                        "parent_node": node_id,
                        "child_node": child_id,
                        "child_parents": sorted(normalized_parents),
                    })

    reachable: set[str] = set()
    queue: list[str] = ["ROOT_ID"] if "ROOT_ID" in nodes else []
    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        children = nodes.get(current, {}).get("children")
        if not isinstance(children, list):
            continue
        for child in children:
            child_id = str(child)
            if child_id in nodes and child_id not in reachable:
                queue.append(child_id)

    unreachable_nodes = sorted(
        node_id for node_id in nodes.keys()
        if node_id not in reachable
    )

    attached_chart_ids = {
        cid
        for cid in (_to_int(chart.get("id")) for chart in dashboard_charts)
        if cid is not None
    }
    layout_chart_ids = set(_extract_position_chart_refs(position).values())
    scope_chart_ids = _extract_scope_chart_refs(metadata)
    duplicate_chart_placements = _find_duplicate_chart_placements(position)

    layout_orphans = sorted(layout_chart_ids - attached_chart_ids)
    scope_orphans = sorted(scope_chart_ids - attached_chart_ids)
    attached_missing_layout = sorted(attached_chart_ids - layout_chart_ids)

    status = "success"
    if (
        missing_required_nodes
        or invalid_children_nodes
        or dangling_children
        or missing_id_nodes
        or id_key_mismatches
        or missing_type_nodes
    ):
        status = "failed"
    elif (
        invalid_parent_nodes
        or parent_mismatches
        or unreachable_nodes
        or layout_orphans
        or scope_orphans
        or attached_missing_layout
        or duplicate_chart_placements
    ):
        status = "warning"

    return {
        "status": status,
        "node_count": len(nodes),
        "chart_count_attached": len(attached_chart_ids),
        "missing_required_nodes": missing_required_nodes,
        "missing_id_nodes": sorted(set(missing_id_nodes)),
        "id_key_mismatches": id_key_mismatches,
        "missing_type_nodes": sorted(set(missing_type_nodes)),
        "invalid_children_nodes": sorted(set(invalid_children_nodes)),
        "invalid_parent_nodes": sorted(set(invalid_parent_nodes)),
        "dangling_children": dangling_children,
        "parent_mismatches": parent_mismatches,
        "unreachable_nodes": unreachable_nodes,
        "layout_chart_ids": sorted(layout_chart_ids),
        "scope_chart_ids": sorted(scope_chart_ids),
        "layout_orphans": layout_orphans,
        "scope_orphans": scope_orphans,
        "attached_missing_layout": attached_missing_layout,
        "duplicate_chart_placements": duplicate_chart_placements,
    }


def _template_chart_specs(
    ws: PresetWorkspace,
    dashboard_charts: list[dict[str, Any]],
    *,
    portable: bool,
    include_query_context: bool,
    include_dataset_schema: bool,
) -> list[dict[str, Any]]:
    """Build reusable chart specs from a dashboard's chart definitions."""
    specs: list[dict[str, Any]] = []
    chart_detail_cache: dict[int, dict[str, Any]] = {}
    dataset_cache: dict[int, dict[str, Any]] = {}

    for chart in dashboard_charts:
        chart_id = _to_int(chart.get("id"))
        if chart_id is None:
            continue

        detail: dict[str, Any] | None = None
        if include_query_context or not isinstance(chart.get("form_data"), dict):
            try:
                detail = ws.chart_detail(chart_id)
            except Exception:
                detail = None
            if isinstance(detail, dict):
                chart_detail_cache[chart_id] = detail
        else:
            detail = chart_detail_cache.get(chart_id)

        viz_type = chart.get("viz_type")
        if not isinstance(viz_type, str) or not viz_type:
            if isinstance(detail, dict):
                raw_viz = detail.get("viz_type")
                if isinstance(raw_viz, str):
                    viz_type = raw_viz

        form_data = chart.get("form_data")
        if not isinstance(form_data, dict):
            if isinstance(detail, dict):
                form_data = _parse_json_object(detail.get("params"))
            if not isinstance(form_data, dict):
                form_data = {}

        datasource_id = _to_int(chart.get("datasource_id"))
        datasource_type = chart.get("datasource_type")
        if datasource_id is None or not isinstance(datasource_type, str):
            ds_from_form = form_data.get("datasource")
            if isinstance(ds_from_form, str) and "__" in ds_from_form:
                raw_id, raw_type = ds_from_form.split("__", 1)
                datasource_id = datasource_id or _to_int(raw_id)
                if not isinstance(datasource_type, str) or not datasource_type:
                    datasource_type = raw_type

        spec: dict[str, Any] = {
            "chart_id": chart_id,
            "title": chart.get("slice_name") or chart.get("chart_name"),
            "viz_type": viz_type,
            "datasource_id": datasource_id,
            "datasource_type": datasource_type,
            "form_data": _sanitize_template_payload(form_data, portable=portable),
        }

        if include_query_context:
            qc = _parse_json_object((detail or {}).get("query_context"))
            spec["query_context"] = _sanitize_template_payload(qc or {}, portable=portable)

        if include_dataset_schema and datasource_id is not None:
            dataset = dataset_cache.get(datasource_id)
            if dataset is None:
                try:
                    dataset = ws.dataset_detail(datasource_id)
                except Exception:
                    dataset = {}
                dataset_cache[datasource_id] = dataset
            if isinstance(dataset, dict):
                spec["dataset_schema"] = {
                    "table_name": dataset.get("table_name"),
                    "schema": dataset.get("schema"),
                    "columns": sorted(_dataset_columns(dataset)),
                    "metrics": sorted(_dataset_metrics(dataset)),
                }

        specs.append(spec)

    return specs


def _build_dashboard_template_payload(
    ws: PresetWorkspace,
    dashboard_id: int,
    *,
    portable: bool,
    include_query_context: bool,
    include_dataset_schema: bool,
) -> dict[str, Any]:
    dashboard = ws.dashboard_detail(dashboard_id)
    dashboard_charts = ws.dashboard_charts(dashboard_id)
    structure = _dashboard_structure_report(
        dashboard.get("position_json", {}),
        dashboard.get("json_metadata", {}),
        dashboard_charts,
    )

    return {
        "template_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portable": portable,
        "dashboard": {
            "id": _to_int(dashboard.get("id")) or dashboard_id,
            "title": dashboard.get("dashboard_title"),
            "slug": dashboard.get("slug"),
            "position_json": _sanitize_template_payload(
                _ensure_json_dict(dashboard.get("position_json", {}), "position_json"),
                portable=portable,
            ),
            "json_metadata": _sanitize_template_payload(
                _ensure_json_dict(dashboard.get("json_metadata", {}), "json_metadata"),
                portable=portable,
            ),
            "structure_report": structure,
        },
        "charts": _template_chart_specs(
            ws,
            dashboard_charts,
            portable=portable,
            include_query_context=include_query_context,
            include_dataset_schema=include_dataset_schema,
        ),
    }


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
    # A query is timeseries whenever a time column is provided; granularity is optional.
    is_timeseries = bool(time_column)
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
# Tools — Dashboard introspection
# ===================================================================


@mcp.tool()
@_handle_errors
def describe_dashboard(
    dashboard_id: int,
    include_lineage: bool = False,
    include_sql: bool = False,
    response_mode: ResponseMode = "standard",
) -> str:
    """Return a normalized dashboard summary in one call.

    Provides dashboard metadata, markdown blocks, chart inventory,
    unique dataset inventory, and optional source-table lineage —
    without requiring multiple follow-up tool calls.

    Args:
        dashboard_id: Numeric dashboard ID (or use get_dashboard to find it)
        include_lineage: Parse dataset SQL to extract upstream source tables
        include_sql: Include raw SQL for each virtual dataset
        response_mode: 'compact' (counts only), 'standard' (inventory),
                       or 'full' (all detail + lineage + warnings).
                       Default: standard.
    """
    from preset_py.client import _parse_source_tables

    ws = _get_ws()
    dashboard = ws.dashboard_detail(dashboard_id)
    dashboard_charts = ws.dashboard_charts(dashboard_id)

    # Extract markdown blocks from position_json
    position = _ensure_json_dict(dashboard.get("position_json", {}), "position_json")
    markdown_blocks: list[dict[str, str]] = []
    for node_id, node in position.items():
        if not isinstance(node, dict):
            continue
        node_type = node.get("type", "")
        if node_type != "MARKDOWN":
            continue
        meta = node.get("meta", {})
        if not isinstance(meta, dict):
            continue
        code = meta.get("code", "")
        if isinstance(code, str) and code.strip():
            markdown_blocks.append({
                "node_id": node_id,
                "text": code.strip(),
            })

    # Build chart inventory and collect unique dataset IDs
    chart_inventory: list[dict[str, Any]] = []
    seen_dataset_ids: set[int] = set()
    dataset_cache: dict[int, dict[str, Any]] = {}

    for chart in dashboard_charts:
        chart_id_val = _to_int(chart.get("id"))
        ds_id = _to_int(chart.get("datasource_id"))

        # Try to get datasource_id from form_data if not on chart
        if ds_id is None:
            form_data = chart.get("form_data", {})
            if isinstance(form_data, dict):
                raw_ds = form_data.get("datasource", "")
                if isinstance(raw_ds, str) and "__" in raw_ds:
                    ds_id = _to_int(raw_ds.split("__", 1)[0])

        entry: dict[str, Any] = {
            "chart_id": chart_id_val,
            "title": chart.get("slice_name") or chart.get("chart_name"),
            "viz_type": chart.get("viz_type"),
            "dataset_id": ds_id,
        }
        if ds_id is not None:
            seen_dataset_ids.add(ds_id)
        chart_inventory.append(entry)

    # Fetch dataset details
    dataset_inventory: list[dict[str, Any]] = []
    warnings: list[str] = []

    for ds_id in sorted(seen_dataset_ids):
        try:
            dataset = ws.dataset_detail(ds_id)
            dataset_cache[ds_id] = dataset
        except Exception:
            warnings.append(f"Could not fetch dataset {ds_id}.")
            continue

        ds_sql = dataset.get("sql")
        kind = dataset.get("kind") or ("virtual" if ds_sql else "physical")

        ds_entry: dict[str, Any] = {
            "dataset_id": ds_id,
            "name": dataset.get("table_name"),
            "database": None,
            "schema": dataset.get("schema"),
            "sql_kind": kind,
        }

        db_ref = dataset.get("database")
        if isinstance(db_ref, dict):
            ds_entry["database"] = db_ref.get("database_name") or db_ref.get("name")
        elif db_ref is not None:
            ds_entry["database"] = db_ref

        if include_sql and isinstance(ds_sql, str):
            ds_entry["sql"] = ds_sql

        if include_lineage or response_mode == "full":
            if isinstance(ds_sql, str) and ds_sql.strip():
                if "VALUES" in ds_sql.upper() and "SELECT" in ds_sql.upper():
                    warnings.append(f"dataset {ds_id} uses inline VALUES snapshot.")
                source_tables = _parse_source_tables(ds_sql)
                ds_entry["source_tables"] = source_tables
                if not source_tables:
                    warnings.append(f"dataset {ds_id}: could not parse source tables from SQL.")
            else:
                ds_entry["source_tables"] = []
                if kind == "virtual":
                    warnings.append(f"dataset {ds_id} is virtual but has no SQL.")

        # Enrich chart inventory with dataset names
        ds_name = dataset.get("table_name")
        for chart_entry in chart_inventory:
            if chart_entry.get("dataset_id") == ds_id and ds_name:
                chart_entry["dataset_name"] = ds_name

        dataset_inventory.append(ds_entry)

    # Assemble response
    dashboard_meta = {
        "id": _to_int(dashboard.get("id")) or dashboard_id,
        "title": dashboard.get("dashboard_title"),
        "published": dashboard.get("published"),
        "changed_on": dashboard.get("changed_on"),
    }
    owners = dashboard.get("owners")
    if isinstance(owners, list) and owners:
        first_owner = owners[0]
        if isinstance(first_owner, dict):
            dashboard_meta["owner"] = (
                first_owner.get("first_name", "")
                + " "
                + first_owner.get("last_name", "")
            ).strip() or first_owner.get("username")
        elif isinstance(first_owner, str):
            dashboard_meta["owner"] = first_owner

    if response_mode == "compact":
        return json.dumps({
            "dashboard": dashboard_meta,
            "chart_count": len(chart_inventory),
            "dataset_count": len(dataset_inventory),
            "markdown_block_count": len(markdown_blocks),
            "warning_count": len(warnings),
        }, indent=2, default=str)

    result: dict[str, Any] = {
        "dashboard": dashboard_meta,
        "markdown_blocks": markdown_blocks,
        "charts": chart_inventory,
        "datasets": dataset_inventory,
    }

    if response_mode == "full":
        result["warnings"] = warnings
    elif warnings:
        result["warning_count"] = len(warnings)
        result["warnings_preview"] = warnings[:3]

    return json.dumps(result, indent=2, default=str)


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


@mcp.tool()
@_handle_errors
def verify_chart_workflow(
    chart_id: int,
    dashboard_id: int | None = None,
    include_render: bool = True,
    row_limit: int = 10000,
    force: bool = False,
    timeout_ms: int = 45000,
    settle_ms: int = 2500,
    response_mode: ResponseMode = "standard",
) -> str:
    """Run end-to-end checks for chart query/render and optional dashboard context."""
    ws = _get_ws()
    chart_query = ws.validate_chart_data(
        chart_id,
        dashboard_id=dashboard_id,
        row_limit=row_limit,
        force=force,
    )
    chart_render = (
        ws.validate_chart_render(
            chart_id,
            timeout_ms=timeout_ms,
            settle_ms=settle_ms,
        )
        if include_render
        else None
    )

    dashboard_query = None
    dashboard_render = None
    if dashboard_id is not None:
        dashboard_query = ws.validate_dashboard_charts(
            dashboard_id,
            row_limit=row_limit,
            force=force,
        )
        if include_render:
            dashboard_render = ws.validate_dashboard_render(
                dashboard_id,
                timeout_ms=timeout_ms,
                settle_ms=settle_ms,
            )

    status = "success"
    if chart_query.get("status") != "success":
        status = "failed"
    if include_render and chart_render and chart_render.get("status") != "success":
        status = "failed"
    if isinstance(dashboard_query, dict):
        dq_results = dashboard_query.get("results", [])
        if any(
            isinstance(item, dict) and item.get("status") != "success"
            for item in dq_results
        ):
            status = "failed"
    if include_render and isinstance(dashboard_render, dict):
        if int(dashboard_render.get("broken_count") or 0) > 0:
            status = "failed"

    result = {
        "status": status,
        "chart_id": chart_id,
        "dashboard_id": dashboard_id,
        "include_render": include_render,
        "chart_query": chart_query,
        "chart_render": chart_render,
        "dashboard_query": dashboard_query,
        "dashboard_render": dashboard_render,
    }

    if response_mode == "compact":
        return json.dumps({
            "status": status,
            "chart_id": chart_id,
            "dashboard_id": dashboard_id,
            "chart_query_status": chart_query.get("status"),
            "chart_render_status": chart_render.get("status") if isinstance(chart_render, dict) else None,
            "dashboard_query_failures": (
                len([
                    item for item in (dashboard_query or {}).get("results", [])
                    if isinstance(item, dict) and item.get("status") != "success"
                ])
                if isinstance(dashboard_query, dict) else None
            ),
            "dashboard_render_broken_count": (
                int((dashboard_render or {}).get("broken_count") or 0)
                if isinstance(dashboard_render, dict) else None
            ),
        }, indent=2, default=str)

    if response_mode == "standard":
        return json.dumps({
            "status": status,
            "chart_id": chart_id,
            "dashboard_id": dashboard_id,
            "chart_query": {
                "status": chart_query.get("status"),
                "error": chart_query.get("error"),
                "payload_source": chart_query.get("payload_source"),
            },
            "chart_render": (
                {
                    "status": chart_render.get("status"),
                    "error": chart_render.get("error"),
                    "critical_page_errors": chart_render.get("critical_page_errors"),
                    "visible_errors": chart_render.get("visible_errors"),
                }
                if isinstance(chart_render, dict) else None
            ),
            "dashboard_query": (
                {
                    "dashboard_id": dashboard_query.get("dashboard_id"),
                    "chart_count": dashboard_query.get("chart_count"),
                    "validated": dashboard_query.get("validated"),
                    "failed_count": len([
                        item for item in dashboard_query.get("results", [])
                        if isinstance(item, dict) and item.get("status") != "success"
                    ]),
                }
                if isinstance(dashboard_query, dict) else None
            ),
            "dashboard_render": (
                {
                    "dashboard_id": dashboard_render.get("dashboard_id"),
                    "chart_count": dashboard_render.get("chart_count"),
                    "broken_count": dashboard_render.get("broken_count"),
                }
                if isinstance(dashboard_render, dict) else None
            ),
        }, indent=2, default=str)

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
@_handle_errors
def verify_dashboard_structure(
    dashboard_id: int,
    response_mode: ResponseMode = "standard",
) -> str:
    """Validate dashboard layout graph integrity and chart reference health."""
    ws = _get_ws()
    dashboard = ws.dashboard_detail(dashboard_id)
    charts = ws.dashboard_charts(dashboard_id)
    report = _dashboard_structure_report(
        dashboard.get("position_json", {}),
        dashboard.get("json_metadata", {}),
        charts,
    )

    if response_mode == "compact":
        return json.dumps({
            "dashboard_id": dashboard_id,
            "status": report.get("status"),
            "node_count": report.get("node_count"),
            "chart_count_attached": report.get("chart_count_attached"),
            "layout_orphan_count": len(report.get("layout_orphans", [])),
            "scope_orphan_count": len(report.get("scope_orphans", [])),
            "dangling_child_count": len(report.get("dangling_children", [])),
            "missing_id_count": len(report.get("missing_id_nodes", [])),
            "missing_type_count": len(report.get("missing_type_nodes", [])),
            "duplicate_chart_count": len(report.get("duplicate_chart_placements", [])),
        }, indent=2, default=str)

    if response_mode == "standard":
        return json.dumps({
            "dashboard_id": dashboard_id,
            "status": report.get("status"),
            "node_count": report.get("node_count"),
            "chart_count_attached": report.get("chart_count_attached"),
            "missing_required_nodes": report.get("missing_required_nodes"),
            "missing_id_nodes": report.get("missing_id_nodes"),
            "id_key_mismatches": report.get("id_key_mismatches"),
            "missing_type_nodes": report.get("missing_type_nodes"),
            "dangling_children": report.get("dangling_children"),
            "layout_orphans": report.get("layout_orphans"),
            "scope_orphans": report.get("scope_orphans"),
            "attached_missing_layout": report.get("attached_missing_layout"),
            "duplicate_chart_placements": report.get("duplicate_chart_placements"),
        }, indent=2, default=str)

    return json.dumps({
        "dashboard_id": dashboard_id,
        "dashboard_title": dashboard.get("dashboard_title"),
        "structure_report": report,
    }, indent=2, default=str)


@mcp.tool()
@_handle_errors
def verify_dashboard_workflow(
    dashboard_id: int,
    include_render: bool = True,
    row_limit: int = 10000,
    force: bool = False,
    timeout_ms: int = 45000,
    settle_ms: int = 2500,
    response_mode: ResponseMode = "standard",
) -> str:
    """Run structure + query + optional render verification for a dashboard."""
    ws = _get_ws()
    dashboard = ws.dashboard_detail(dashboard_id)
    charts = ws.dashboard_charts(dashboard_id)
    structure = _dashboard_structure_report(
        dashboard.get("position_json", {}),
        dashboard.get("json_metadata", {}),
        charts,
    )
    query_result = ws.validate_dashboard_charts(
        dashboard_id,
        row_limit=row_limit,
        force=force,
    )
    render_result = (
        ws.validate_dashboard_render(
            dashboard_id,
            timeout_ms=timeout_ms,
            settle_ms=settle_ms,
        )
        if include_render
        else None
    )

    query_failures = len([
        item for item in query_result.get("results", [])
        if isinstance(item, dict) and item.get("status") != "success"
    ])
    render_broken = int((render_result or {}).get("broken_count") or 0)

    status = "success"
    if structure.get("status") == "failed":
        status = "failed"
    elif structure.get("status") == "warning":
        status = "warning"
    if query_failures > 0:
        status = "failed"
    if include_render and render_broken > 0:
        status = "failed"

    if response_mode == "compact":
        return json.dumps({
            "dashboard_id": dashboard_id,
            "status": status,
            "structure_status": structure.get("status"),
            "query_failures": query_failures,
            "render_broken_count": render_broken if include_render else None,
        }, indent=2, default=str)

    if response_mode == "standard":
        return json.dumps({
            "dashboard_id": dashboard_id,
            "dashboard_title": dashboard.get("dashboard_title"),
            "status": status,
            "structure": {
                "status": structure.get("status"),
                "missing_required_nodes": structure.get("missing_required_nodes"),
                "layout_orphans": structure.get("layout_orphans"),
                "scope_orphans": structure.get("scope_orphans"),
            },
            "query_validation": {
                "chart_count": query_result.get("chart_count"),
                "validated": query_result.get("validated"),
                "failed_count": query_failures,
            },
            "render_validation": (
                {
                    "chart_count": render_result.get("chart_count"),
                    "broken_count": render_broken,
                }
                if include_render and isinstance(render_result, dict)
                else None
            ),
        }, indent=2, default=str)

    return json.dumps({
        "dashboard_id": dashboard_id,
        "dashboard_title": dashboard.get("dashboard_title"),
        "status": status,
        "structure_report": structure,
        "query_validation": query_result,
        "render_validation": render_result,
    }, indent=2, default=str)


@mcp.tool()
@_handle_errors
def capture_dashboard_template(
    dashboard_id: int,
    portable: bool = True,
    include_query_context: bool = False,
    include_dataset_schema: bool = False,
    output_path: str | None = None,
    response_mode: ResponseMode = "standard",
) -> str:
    """Capture a reusable dashboard+chart template from an existing dashboard."""
    ws = _get_ws()
    template = _build_dashboard_template_payload(
        ws,
        dashboard_id,
        portable=portable,
        include_query_context=include_query_context,
        include_dataset_schema=include_dataset_schema,
    )

    resolved_output: str | None = None
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template, indent=2, default=str) + "\n")
        resolved_output = str(path)

    chart_count = len(template.get("charts", []))
    structure = template.get("dashboard", {}).get("structure_report", {})

    if response_mode == "compact":
        return json.dumps({
            "dashboard_id": dashboard_id,
            "title": template.get("dashboard", {}).get("title"),
            "chart_count": chart_count,
            "structure_status": structure.get("status"),
            "output_path": resolved_output,
        }, indent=2, default=str)

    if response_mode == "standard":
        return json.dumps({
            "dashboard_id": dashboard_id,
            "title": template.get("dashboard", {}).get("title"),
            "chart_count": chart_count,
            "structure_status": structure.get("status"),
            "structure_summary": {
                "layout_orphans": structure.get("layout_orphans"),
                "scope_orphans": structure.get("scope_orphans"),
                "dangling_children": structure.get("dangling_children"),
            },
            "example_charts": template.get("charts", [])[:3],
            "output_path": resolved_output,
            "hint": "Use response_mode='full' to get the full template JSON inline.",
        }, indent=2, default=str)

    payload = dict(template)
    if resolved_output:
        payload["_saved_to"] = resolved_output
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
@_handle_errors
def capture_golden_templates(
    dashboard_ids: list[int] | str,
    output_dir: str = "~/.preset-mcp/golden-templates",
    portable: bool = True,
    include_query_context: bool = False,
    include_dataset_schema: bool = False,
    overwrite: bool = False,
    response_mode: ResponseMode = "standard",
) -> str:
    """Capture templates from one or more dashboards into a local folder."""
    ws = _get_ws()
    ids = _coerce_list_arg(
        dashboard_ids,
        field_name="dashboard_ids",
        item_kind="int",
    )
    if not ids:
        raise ValueError("dashboard_ids must include at least one dashboard ID.")

    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dashboard_id in ids:
        try:
            template = _build_dashboard_template_payload(
                ws,
                dashboard_id,
                portable=portable,
                include_query_context=include_query_context,
                include_dataset_schema=include_dataset_schema,
            )
            title = str(template.get("dashboard", {}).get("title") or f"dashboard-{dashboard_id}")
            filename = f"{dashboard_id}_{_slugify(title)}.json"
            path = target_dir / filename
            if path.exists() and not overwrite:
                failures.append({
                    "dashboard_id": dashboard_id,
                    "error": f"{path} already exists. Re-run with overwrite=True.",
                })
                continue
            path.write_text(json.dumps(template, indent=2, default=str) + "\n")
            saved.append({
                "dashboard_id": dashboard_id,
                "title": title,
                "chart_count": len(template.get("charts", [])),
                "structure_status": template.get("dashboard", {}).get("structure_report", {}).get("status"),
                "output_path": str(path),
            })
        except Exception as exc:
            failures.append({
                "dashboard_id": dashboard_id,
                "error": str(exc),
            })

    status = "success" if not failures else ("partial_success" if saved else "failed")
    payload = {
        "status": status,
        "output_dir": str(target_dir),
        "requested_count": len(ids),
        "saved_count": len(saved),
        "failed_count": len(failures),
        "saved": saved,
        "failures": failures,
    }

    if response_mode == "compact":
        return json.dumps({
            "status": status,
            "output_dir": str(target_dir),
            "saved_count": len(saved),
            "failed_count": len(failures),
        }, indent=2, default=str)

    if response_mode == "standard":
        return json.dumps(payload, indent=2, default=str)

    return json.dumps(payload, indent=2, default=str)


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
    metrics: list[Any] | str | None = None,
    groupby: list[str] | str | None = None,
    time_column: str | None = None,
    template: Literal["auto", "minimal"] = "auto",
    params_json: str | None = None,
    dashboards: list[int] | str | None = None,
    validate_after_create: bool = True,
    repair_dashboard_refs: bool = False,
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
        metrics: Metric names or ad-hoc metric objects
        groupby: Columns to group by
        time_column: Time column for time-series charts
        template: Defaulting strategy for missing chart fields
                  ('auto' or 'minimal')
        params_json: Optional JSON object to merge into chart params
        dashboards: Dashboard IDs to attach this chart to
        validate_after_create: Run chart-data validation after create
        repair_dashboard_refs: Attempt to repair stale dashboard chart
                               references when dashboards are provided.
                               Defaults to False so create_chart does not
                               mutate dashboard layouts unless explicitly
                               requested.
        dry_run: If True, validate inputs and return a preview without
                 making any changes (default: False)
    """
    ws = _get_ws()
    dataset = _fetch_dataset_or_raise(ws, dataset_id)
    _validate_viz_type(ws, viz_type)
    if template not in ("auto", "minimal"):
        raise ValueError("template must be one of: auto, minimal.")

    metrics_list = _coerce_list_arg(
        metrics, field_name="metrics", item_kind="any"
    )
    if metrics_list is not None:
        for idx, metric in enumerate(metrics_list):
            if not isinstance(metric, str | dict):
                raise ValueError(
                    f"metrics[{idx}] must be a string or metric object."
                )
    groupby_list = _coerce_list_arg(
        groupby, field_name="groupby", item_kind="str"
    )
    dashboards_list = _coerce_list_arg(
        dashboards, field_name="dashboards", item_kind="int"
    )
    if dashboards_list:
        _require_dashboards_exist(ws, dashboards_list)

    dataset_columns = _dataset_columns(dataset)
    dataset_metrics = _dataset_metrics(dataset)
    params_warnings: list[str] = []
    extra_params: dict[str, Any] = {}
    if params_json is not None:
        parsed_params, params_warnings = validate_params_payload(
            params_json,
            dataset_columns=dataset_columns,
            dataset_metrics=dataset_metrics,
            viz_type=viz_type,
            fallback_fields={
                "metrics": metrics_list,
                "groupby": groupby_list,
                "granularity_sqla": time_column,
                "time_column": time_column,
            },
        )
        extra_params = parsed_params

        if metrics_list is not None and "metrics" in parsed_params:
            raise ValueError(
                "Provide metrics in either the metrics argument or params_json.metrics, "
                "not both."
            )
        if groupby_list is not None and "groupby" in parsed_params:
            raise ValueError(
                "Provide groupby in either the groupby argument or params_json.groupby, "
                "not both."
            )
        if time_column is not None and "granularity_sqla" in parsed_params:
            raise ValueError(
                "Provide time_column directly or via params_json.granularity_sqla, not both."
            )

    fields = ["dataset_id", "title", "viz_type"]
    if metrics_list is not None:
        fields.append("metrics")
    if groupby_list is not None:
        fields.append("groupby")
    if time_column is not None:
        fields.append("time_column")
    if template != "auto":
        fields.append("template")
    if params_json is not None:
        fields.append("params_json")
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
            template=template,
            **extra_params,
        ),
        preview_extras={"values": {
            "dataset_id": dataset_id, "title": title, "viz_type": viz_type,
            "metrics": metrics_list, "groupby": groupby_list,
            "time_column": time_column, "template": template,
            "params_json": params_json,
            "dashboards": dashboards_list,
            "validate_after_create": validate_after_create,
            "repair_dashboard_refs": repair_dashboard_refs,
        }},
        after_extras={"title": title},
    )
    if dry_run:
        return raw

    payload = json.loads(raw)
    if params_warnings:
        payload["_params_warnings"] = params_warnings
    chart_id = _to_int(payload.get("id"))
    if chart_id is None:
        return json.dumps(payload, indent=2, default=str)

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
        payload["_validation"] = _post_validation_result(
            ws,
            chart_id=chart_id,
            dashboard_id=dashboard_id,
            operation_name="creation",
            operation_past_tense="created",
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
    before = capture_before(ws, "dataset", dataset_id)
    if "_snapshot_error" in before:
        raise ValueError(
            f"Dataset {dataset_id} not found. Use list_datasets to find valid IDs."
        )

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

    params_json uses strict semantics: when provided, it is treated as a
    full chart params payload for validation. For viz types with required
    fields (for example pie/timeseries), include those fields in the
    payload instead of sending partial patches.

    Args:
        chart_id: ID of the chart to update
        title: New chart title
        viz_type: New visualization type
        params_json: JSON string of chart parameters (advanced — use
                     get_chart to inspect existing chart params first).
                     Strict semantics: provide a complete params payload
                     compatible with the chart viz type.
        dashboards: Reassign chart to these dashboard IDs
        validate_after_update: Run chart-data validation after update
        dry_run: If True, validate inputs, capture current state, and
                 return a preview without making any changes
                 (default: False)
    """
    ws = _get_ws()
    before = capture_before(ws, "chart", chart_id)
    if "_snapshot_error" in before:
        raise ValueError(
            f"Chart {chart_id} not found. Use list_charts to find valid IDs."
        )

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
        resolved_viz_type = viz_type
        if resolved_viz_type is None:
            existing_viz = before.get("viz_type")
            if isinstance(existing_viz, str) and existing_viz:
                resolved_viz_type = existing_viz
        if chart_dataset_id is not None:
            try:
                dataset = ws.dataset_detail(chart_dataset_id)
                dataset_columns = _dataset_columns(dataset)
                dataset_metrics = _dataset_metrics(dataset)
            except Exception:
                dataset_columns = set()
                dataset_metrics = set()
        try:
            _, params_warnings = validate_params_payload(
                params_json,
                dataset_columns=dataset_columns,
                dataset_metrics=dataset_metrics,
                viz_type=resolved_viz_type,
            )
        except ValueError as exc:
            raise ValueError(
                f"{exc} Strict params semantics: update_chart.params_json must "
                "be a complete viz-compatible params payload, not a partial patch."
            ) from exc

        try:
            existing_params = (
                _ensure_json_dict(before.get("params", {}), "chart params")
                if before.get("params")
                else {}
            )
        except ValueError:
            existing_params = {}
            _log.debug(
                "update_chart: could not parse existing params for chart %s; "
                "skipping datasource metadata preservation",
                chart_id,
            )
        parsed_params = json.loads(params_json)
        if isinstance(parsed_params, dict):
            if (
                "datasource" not in parsed_params
                and "datasource" in existing_params
            ):
                parsed_params["datasource"] = existing_params["datasource"]
            resolved_viz_type = viz_type or before.get("viz_type")
            if "viz_type" not in parsed_params and resolved_viz_type:
                parsed_params["viz_type"] = resolved_viz_type
            params_json = json.dumps(parsed_params)

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
        payload["_validation"] = _post_validation_result(
            ws,
            chart_id=chart_id,
            dashboard_id=dashboard_id,
            operation_name="update",
            operation_past_tense="updated",
        )
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
@_handle_errors
def update_dashboard(
    dashboard_id: int,
    dashboard_title: str | None = None,
    published: bool | None = None,
    position_json: dict[str, Any] | str | None = None,
    json_metadata: dict[str, Any] | str | None = None,
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
        position_json: JSON object (or JSON string) defining the dashboard
            layout (chart containers, rows, grid). Use this to add,
            remove, or rearrange chart containers — e.g. after deleting
            charts whose containers remain as orphaned placeholders.
        json_metadata: JSON object (or JSON string) with dashboard metadata
            (cross-filter config, color schemes, label colors, refresh settings).
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

    ws = _get_ws()
    before = capture_before(ws, "dashboard", dashboard_id)

    if position_json is not None:
        proposed_position = _ensure_json_dict(position_json, "position_json")
        _validate_position_layout(proposed_position)
        kwargs["position_json"] = json.dumps(proposed_position)
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
    if json_metadata is not None:
        proposed_metadata = _ensure_json_dict(json_metadata, "json_metadata")
        kwargs["json_metadata"] = json.dumps(proposed_metadata)

    if not kwargs:
        raise ValueError(
            "Provide at least one field to update "
            "(dashboard_title, published, position_json, or json_metadata)."
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
    before = capture_before(ws, "dashboard", dashboard_id)
    if "_snapshot_error" in before:
        raise ValueError(
            f"Dashboard {dashboard_id} not found. Use list_dashboards to find valid IDs."
        )
    charts = ws.dashboard_charts(dashboard_id)

    position, metadata, summary = _repair_dashboard_refs(
        before.get("position_json", {}),
        before.get("json_metadata", {}),
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


@mcp.tool()
@_handle_errors
def repair_dashboard_layout_duplicates(
    dashboard_id: int,
    dry_run: bool = False,
) -> str:
    """Remove duplicate chart container nodes from a dashboard layout.

    After a ZIP import, Superset may duplicate ROW/CHART containers causing
    charts to render twice.  This tool detects and removes the duplicates.

    Args:
        dashboard_id: Dashboard ID to repair
        dry_run: If True, preview without mutating (default: False)
    """
    ws = _get_ws()
    before = capture_before(ws, "dashboard", dashboard_id)
    if "_snapshot_error" in before:
        raise ValueError(
            f"Dashboard {dashboard_id} not found. Use list_dashboards to find valid IDs."
        )

    position = _ensure_json_dict(before.get("position_json", {}), "position_json")
    dupes = _find_duplicate_chart_placements(position)

    if not dupes:
        return json.dumps({
            "status": "noop",
            "dashboard_id": dashboard_id,
            "duplicate_chart_placements": [],
            "message": "No duplicate chart placements found.",
        }, indent=2, default=str)

    cleaned = _deduplicate_layout_containers(position)

    return _do_mutation(
        tool_name="repair_dashboard_layout_duplicates",
        resource_type="dashboard",
        action="update",
        fields_changed=["position_json"],
        dry_run=dry_run,
        execute=lambda: ws.update_dashboard(
            dashboard_id,
            position_json=json.dumps(cleaned),
        ),
        resource_id=dashboard_id,
        before=before,
        preview_extras={
            "duplicate_chart_placements": dupes,
            "nodes_before": len(position),
            "nodes_after": len(cleaned),
        },
        result_extras={
            "duplicate_chart_placements_fixed": dupes,
            "nodes_before": len(position),
            "nodes_after": len(cleaned),
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
# Tools — Local audit visibility + dashboard recovery
# ===================================================================


def _read_mutation_entries(
    *,
    limit: int,
    resource_type: str | None = None,
    resource_id: int | None = None,
    tool_name: str | None = None,
) -> list[dict[str, Any]]:
    journal = AUDIT_DIR / "mutations.jsonl"
    if not journal.exists():
        return []

    entries: list[dict[str, Any]] = []
    for line in journal.read_text().splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if resource_type and entry.get("resource_type") != resource_type:
            continue
        if resource_id is not None and _to_int(entry.get("resource_id")) != resource_id:
            continue
        if tool_name and entry.get("tool_name") != tool_name:
            continue
        entries.append(entry)

    entries.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
    return entries[:limit]


@mcp.tool()
@_handle_errors
def list_mutations(
    resource_type: Literal["dashboard", "chart", "dataset"] | None = None,
    resource_id: int | None = None,
    tool_name: str | None = None,
    limit: int = 50,
) -> str:
    """List recent local mutation-journal entries for incident debugging."""
    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500.")

    entries = _read_mutation_entries(
        limit=limit,
        resource_type=resource_type,
        resource_id=resource_id,
        tool_name=tool_name,
    )
    return json.dumps({
        "count": len(entries),
        "limit": limit,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "tool_name": tool_name,
        "entries": entries,
    }, indent=2, default=str)


@mcp.tool()
@_handle_errors
def list_dashboard_snapshots(
    dashboard_id: int | None = None,
    limit: int = 50,
) -> str:
    """List locally saved dashboard snapshots captured before mutations."""
    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500.")

    snap_dir = AUDIT_DIR / "snapshots"
    if not snap_dir.exists():
        return json.dumps({
            "count": 0,
            "dashboard_id": dashboard_id,
            "snapshots": [],
        }, indent=2, default=str)

    pattern = "dashboard_*.json" if dashboard_id is None else f"dashboard_{dashboard_id}_*.json"
    files = sorted(
        snap_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    records: list[dict[str, Any]] = []
    for snap in files[:limit]:
        stem_parts = snap.stem.split("_")
        snap_dashboard_id = _to_int(stem_parts[1]) if len(stem_parts) >= 3 else None
        timestamp_token = stem_parts[2] if len(stem_parts) >= 3 else None
        records.append({
            "snapshot_path": str(snap),
            "dashboard_id": snap_dashboard_id,
            "timestamp_token": timestamp_token,
            "size_bytes": snap.stat().st_size,
            "modified_at": datetime.fromtimestamp(
                snap.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        })

    return json.dumps({
        "count": len(records),
        "dashboard_id": dashboard_id,
        "snapshots": records,
    }, indent=2, default=str)


@mcp.tool()
@_handle_errors
def restore_dashboard_snapshot(
    dashboard_id: int,
    snapshot_path: str,
    restore_json_metadata: bool = True,
    allow_id_mismatch: bool = False,
    dry_run: bool = False,
) -> str:
    """Restore dashboard layout/settings from a local snapshot JSON file.

    Args:
        dashboard_id: Target dashboard to restore into.
        snapshot_path: Path to snapshot JSON file.
        restore_json_metadata: Whether to restore json_metadata too (default: True).
        allow_id_mismatch: If True, allow restoring a snapshot taken from a
            different dashboard ID into *dashboard_id*.  Useful when a dashboard
            was deleted and re-imported with a new ID (default: False).
        dry_run: If True, preview without mutating (default: False).
    """
    ws = _get_ws()
    _require_dashboards_exist(ws, [dashboard_id])

    path = Path(snapshot_path).expanduser()
    if not path.exists():
        raise ValueError(f"Snapshot not found: {snapshot_path}")

    try:
        snapshot = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Snapshot file is not valid JSON: {exc}") from exc

    if not isinstance(snapshot, dict):
        raise ValueError("Snapshot file must contain a JSON object.")

    snap_dashboard_id = _to_int(snapshot.get("id"))
    if snap_dashboard_id is not None and snap_dashboard_id != dashboard_id:
        if not allow_id_mismatch:
            raise ValueError(
                f"Snapshot dashboard id {snap_dashboard_id} does not match requested "
                f"dashboard_id {dashboard_id}. Pass allow_id_mismatch=True to "
                f"restore this snapshot into a different dashboard."
            )

    restored_position = _ensure_json_dict(snapshot.get("position_json"), "position_json")
    _validate_position_layout(restored_position)
    kwargs: dict[str, Any] = {
        "position_json": json.dumps(restored_position),
    }
    if restore_json_metadata:
        restored_metadata = _ensure_json_dict(snapshot.get("json_metadata"), "json_metadata")
        kwargs["json_metadata"] = json.dumps(restored_metadata)

    before = capture_before(ws, "dashboard", dashboard_id)
    return _do_mutation(
        tool_name="restore_dashboard_snapshot",
        resource_type="dashboard",
        action="update",
        fields_changed=list(kwargs.keys()),
        dry_run=dry_run,
        execute=lambda: ws.update_dashboard(dashboard_id, **kwargs),
        resource_id=dashboard_id,
        before=before,
        preview_extras={
            "snapshot_path": str(path),
            "restore_json_metadata": restore_json_metadata,
        },
        result_extras={
            "_restored_from_snapshot": str(path),
        },
        after_extras={
            "snapshot_path": str(path),
            "restore_json_metadata": restore_json_metadata,
        },
    )


# ===================================================================
# Tools — Dashboard Export / Import (always available)
# ===================================================================


@mcp.tool()
@_handle_errors
def export_dashboard(
    dashboard_id: int,
    output_path: str | None = None,
) -> str:
    """Export a dashboard as a ZIP bundle for backup or migration.

    The ZIP contains the dashboard definition, its charts, and datasets.
    If *output_path* is provided the ZIP is written to disk; otherwise it
    is saved to the default audit exports directory.

    Args:
        dashboard_id: Dashboard to export
        output_path: Optional file path to save the ZIP (default: auto)
    """
    ws = _get_ws()
    _require_dashboards_exist(ws, [dashboard_id])

    zip_bytes = ws.export_resource_zip("dashboard", [dashboard_id])

    if output_path:
        dest = Path(output_path).expanduser()
    else:
        exports_dir = AUDIT_DIR / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = exports_dir / f"dashboard_{dashboard_id}_{ts}.zip"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(zip_bytes)

    record_mutation(MutationEntry(
        tool_name="export_dashboard",
        resource_type="dashboard",
        resource_id=dashboard_id,
        action="create",
        after_summary={
            "export_path": str(dest),
            "size_bytes": len(zip_bytes),
        },
    ))

    return json.dumps({
        "status": "exported",
        "dashboard_id": dashboard_id,
        "export_path": str(dest),
        "size_bytes": len(zip_bytes),
    }, indent=2, default=str)


@mcp.tool()
@_handle_errors
def import_dashboard(
    import_path: str,
    overwrite: bool = False,
    deduplicate_layout: bool = True,
) -> str:
    """Import a dashboard from a ZIP bundle.

    Args:
        import_path: Path to the ZIP file to import
        overwrite: If True, overwrite existing dashboards with same IDs
                   (default: False)
        deduplicate_layout: If True, automatically fix duplicate chart
                            containers that Superset may create during
                            import (default: True)
    """
    ws = _get_ws()
    path = Path(import_path).expanduser()
    if not path.exists():
        raise ValueError(f"Import file not found: {import_path}")

    data = path.read_bytes()
    result = ws.import_resource_zip("dashboard", data, overwrite=overwrite)

    dedup_summary: dict[str, Any] | None = None
    if deduplicate_layout and result:
        dashboards = ws.dashboards()
        for dash in dashboards:
            dash_id = _to_int(dash.get("id"))
            if dash_id is None:
                continue
            try:
                detail = ws.dashboard_detail(dash_id)
                pos = _ensure_json_dict(
                    detail.get("position_json", {}), "position_json",
                )
                dupes = _find_duplicate_chart_placements(pos)
                if dupes:
                    cleaned = _deduplicate_layout_containers(pos)
                    ws.update_dashboard(
                        dash_id,
                        position_json=json.dumps(cleaned),
                    )
                    dedup_summary = {
                        "dashboard_id": dash_id,
                        "duplicates_removed": len(dupes),
                    }
            except Exception:
                pass  # best-effort dedup

    record_mutation(MutationEntry(
        tool_name="import_dashboard",
        resource_type="dashboard",
        action="create",
        after_summary={
            "import_path": str(path),
            "overwrite": overwrite,
            "success": result,
            "dedup_applied": dedup_summary,
        },
    ))

    response: dict[str, Any] = {
        "status": "imported",
        "import_path": str(path),
        "overwrite": overwrite,
        "success": result,
    }
    if dedup_summary:
        response["layout_dedup"] = dedup_summary
    return json.dumps(response, indent=2, default=str)


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

        # Post-import deduplication for dashboards (fixes issue #39)
        dedup_summary: dict[str, Any] | None = None
        if resource_type == "dashboard" and result:
            dashboards = ws.dashboards()
            for dash in dashboards:
                dash_id = _to_int(dash.get("id"))
                if dash_id is None:
                    continue
                try:
                    detail = ws.dashboard_detail(dash_id)
                    pos = _ensure_json_dict(
                        detail.get("position_json", {}), "position_json",
                    )
                    dupes = _find_duplicate_chart_placements(pos)
                    if dupes:
                        cleaned = _deduplicate_layout_containers(pos)
                        ws.update_dashboard(
                            dash_id,
                            position_json=json.dumps(cleaned),
                        )
                        dedup_summary = {
                            "dashboard_id": dash_id,
                            "duplicates_removed": len(dupes),
                        }
                except Exception:
                    pass  # best-effort dedup

        record_mutation(MutationEntry(
            tool_name="restore_from_backup",
            resource_type=resource_type,
            action="restore",
            after_summary={
                "export_path": export_path,
                "overwrite": overwrite,
                "success": result,
                "dedup_applied": dedup_summary,
            },
        ))

        response: dict[str, Any] = {
            "status": "restored",
            "resource_type": resource_type,
            "export_path": export_path,
            "overwrite": overwrite,
            "success": result,
        }
        if dedup_summary:
            response["layout_dedup"] = dedup_summary
        return json.dumps(response, indent=2)


# ===================================================================
# Tools — Saved Queries
# ===================================================================


@mcp.tool()
@_handle_errors
def list_saved_queries(
    response_mode: ResponseMode = "standard",
    name_contains: str | None = None,
) -> str:
    """List saved SQL queries in the current workspace.

    Saved queries are reusable SQL snippets stored in SQL Lab.
    Use this to discover query IDs, then pass one to get_saved_query
    for full detail.

    Args:
        response_mode: 'compact' (id+label), 'standard' (key fields),
                       or 'full' (raw API response).  Default: standard.
        name_contains: Case-insensitive substring filter on the query label.
    """
    ws = _get_ws()
    records = ws.saved_queries()
    if name_contains:
        needle = name_contains.lower()
        records = [
            r for r in records
            if needle in str(r.get("label", "")).lower()
        ]
    if response_mode == "compact":
        data = _pick(records, ["id", "label", "db_id"])
    elif response_mode == "standard":
        data = _pick(records, [
            "id", "label", "db_id", "schema", "sql",
            "changed_on", "description",
        ])
    else:
        data = records
    out: dict[str, Any] = {
        "count": len(records),
        "response_mode": response_mode,
        "data": data,
    }
    if response_mode != "full":
        out["hint"] = "Set response_mode='full' to see all fields."
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
@_handle_errors
def get_saved_query(
    query_id: int,
    response_mode: ResponseMode = "standard",
) -> str:
    """Get detail for a single saved query.

    Args:
        query_id: The saved query ID.
        response_mode: 'compact', 'standard', or 'full'.
    """
    ws = _get_ws()
    record = ws.saved_query_detail(query_id)
    if response_mode == "compact":
        data = {k: record[k] for k in ["id", "label", "db_id", "schema"] if k in record}
    elif response_mode == "standard":
        data = {k: record[k] for k in [
            "id", "label", "db_id", "schema", "sql",
            "description", "changed_on",
        ] if k in record}
    else:
        data = record
    out: dict[str, Any] = {"response_mode": response_mode, "data": data}
    if response_mode != "full":
        out["hint"] = "Set response_mode='full' to see all fields."
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
@_handle_errors
def create_saved_query(
    label: str,
    sql: str,
    database_id: int,
    schema: str | None = None,
    description: str | None = None,
    dry_run: bool = False,
) -> str:
    """Create a new saved SQL query in SQL Lab.

    Saved queries persist reusable SQL snippets associated with a database.

    Args:
        label: Name / label for the saved query.
        sql: The SQL text to save.
        database_id: ID of the database this query targets.
        schema: Optional schema context (e.g. 'public').
        description: Optional description of the query.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    fields = ["label", "sql", "db_id"]
    if schema:
        fields.append("schema")
    if description:
        fields.append("description")

    return _do_mutation(
        tool_name="create_saved_query",
        resource_type="saved_query",
        action="create",
        fields_changed=fields,
        dry_run=dry_run,
        execute=lambda: ws.create_saved_query(
            label=label, sql=sql, database_id=database_id,
            schema=schema, description=description,
        ),
    )


@mcp.tool()
@_handle_errors
def update_saved_query(
    query_id: int,
    label: str | None = None,
    sql: str | None = None,
    description: str | None = None,
    schema: str | None = None,
    dry_run: bool = False,
) -> str:
    """Update an existing saved query.

    Args:
        query_id: The saved query ID to update.
        label: New label (name) for the query.
        sql: New SQL text.
        description: New description.
        schema: New schema context.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    kwargs: dict[str, Any] = {}
    if label is not None:
        kwargs["label"] = label
    if sql is not None:
        kwargs["sql"] = sql
    if description is not None:
        kwargs["description"] = description
    if schema is not None:
        kwargs["schema"] = schema

    if not kwargs:
        raise ValueError("No fields to update. Pass at least one of: label, sql, description, schema.")

    before = capture_before(ws, "saved_query", query_id)

    return _do_mutation(
        tool_name="update_saved_query",
        resource_type="saved_query",
        action="update",
        resource_id=query_id,
        fields_changed=list(kwargs.keys()),
        dry_run=dry_run,
        before=before,
        execute=lambda: ws.update_saved_query(query_id, **kwargs),
    )


@mcp.tool()
@_handle_errors
def delete_saved_query(
    query_id: int,
    dry_run: bool = False,
) -> str:
    """Delete a saved query.

    Args:
        query_id: The saved query ID to delete.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    before = capture_before(ws, "saved_query", query_id)

    return _do_mutation(
        tool_name="delete_saved_query",
        resource_type="saved_query",
        action="delete",
        resource_id=query_id,
        fields_changed=[],
        dry_run=dry_run,
        before=before,
        execute=lambda: ws.delete_saved_query(query_id) or {},
    )


# ===================================================================
# Tools — CSS Templates
# ===================================================================


@mcp.tool()
@_handle_errors
def list_css_templates(
    response_mode: ResponseMode = "standard",
    name_contains: str | None = None,
) -> str:
    """List CSS templates in the current workspace.

    CSS templates define reusable dashboard styling.  Use this to discover
    template IDs, then pass one to get_css_template for full detail.

    Args:
        response_mode: 'compact' (id+name), 'standard' (key fields),
                       or 'full' (raw API response).  Default: standard.
        name_contains: Case-insensitive substring filter on template_name.
    """
    ws = _get_ws()
    records = ws.css_templates()
    if name_contains:
        needle = name_contains.lower()
        records = [
            r for r in records
            if needle in str(r.get("template_name", "")).lower()
        ]
    if response_mode == "compact":
        data = _pick(records, ["id", "template_name"])
    elif response_mode == "standard":
        data = _pick(records, [
            "id", "template_name", "css", "changed_on",
        ])
    else:
        data = records
    out: dict[str, Any] = {
        "count": len(records),
        "response_mode": response_mode,
        "data": data,
    }
    if response_mode != "full":
        out["hint"] = "Set response_mode='full' to see all fields."
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
@_handle_errors
def get_css_template(
    template_id: int,
    response_mode: ResponseMode = "standard",
) -> str:
    """Get detail for a single CSS template.

    Args:
        template_id: The CSS template ID.
        response_mode: 'compact', 'standard', or 'full'.
    """
    ws = _get_ws()
    record = ws.css_template_detail(template_id)
    if response_mode == "compact":
        data = {k: record[k] for k in ["id", "template_name"] if k in record}
    elif response_mode == "standard":
        data = {k: record[k] for k in [
            "id", "template_name", "css", "changed_on",
        ] if k in record}
    else:
        data = record
    out: dict[str, Any] = {"response_mode": response_mode, "data": data}
    if response_mode != "full":
        out["hint"] = "Set response_mode='full' to see all fields."
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
@_handle_errors
def create_css_template(
    template_name: str,
    css: str,
    dry_run: bool = False,
) -> str:
    """Create a new CSS template for dashboard styling.

    Args:
        template_name: Name for the CSS template.
        css: The CSS stylesheet text.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()

    return _do_mutation(
        tool_name="create_css_template",
        resource_type="css_template",
        action="create",
        fields_changed=["template_name", "css"],
        dry_run=dry_run,
        execute=lambda: ws.create_css_template(
            template_name=template_name, css=css,
        ),
    )


@mcp.tool()
@_handle_errors
def update_css_template(
    template_id: int,
    template_name: str | None = None,
    css: str | None = None,
    dry_run: bool = False,
) -> str:
    """Update an existing CSS template.

    Args:
        template_id: The CSS template ID to update.
        template_name: New name for the template.
        css: New CSS stylesheet text.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    kwargs: dict[str, Any] = {}
    if template_name is not None:
        kwargs["template_name"] = template_name
    if css is not None:
        kwargs["css"] = css

    if not kwargs:
        raise ValueError("No fields to update. Pass at least one of: template_name, css.")

    before = capture_before(ws, "css_template", template_id)

    return _do_mutation(
        tool_name="update_css_template",
        resource_type="css_template",
        action="update",
        resource_id=template_id,
        fields_changed=list(kwargs.keys()),
        dry_run=dry_run,
        before=before,
        execute=lambda: ws.update_css_template(template_id, **kwargs),
    )


@mcp.tool()
@_handle_errors
def delete_css_template(
    template_id: int,
    dry_run: bool = False,
) -> str:
    """Delete a CSS template.

    Args:
        template_id: The CSS template ID to delete.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    before = capture_before(ws, "css_template", template_id)

    return _do_mutation(
        tool_name="delete_css_template",
        resource_type="css_template",
        action="delete",
        resource_id=template_id,
        fields_changed=[],
        dry_run=dry_run,
        before=before,
        execute=lambda: ws.delete_css_template(template_id) or {},
    )


# ===================================================================
# Tools — Annotation Layers
# ===================================================================


@mcp.tool()
@_handle_errors
def list_annotation_layers(
    response_mode: ResponseMode = "standard",
    name_contains: str | None = None,
) -> str:
    """List annotation layers in the current workspace.

    Annotation layers let you overlay time-based markers on charts
    (e.g. deploys, incidents).  Use this to discover layer IDs.

    Args:
        response_mode: 'compact' (id+name), 'standard' (key fields),
                       or 'full' (raw API response).  Default: standard.
        name_contains: Case-insensitive substring filter on layer name.
    """
    ws = _get_ws()
    records = ws.annotation_layers()
    if name_contains:
        needle = name_contains.lower()
        records = [
            r for r in records
            if needle in str(r.get("name", "")).lower()
        ]
    if response_mode == "compact":
        data = _pick(records, ["id", "name"])
    elif response_mode == "standard":
        data = _pick(records, [
            "id", "name", "descr", "changed_on",
        ])
    else:
        data = records
    out: dict[str, Any] = {
        "count": len(records),
        "response_mode": response_mode,
        "data": data,
    }
    if response_mode != "full":
        out["hint"] = "Set response_mode='full' to see all fields."
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
@_handle_errors
def get_annotation_layer(
    layer_id: int,
    response_mode: ResponseMode = "standard",
) -> str:
    """Get detail for a single annotation layer including its annotations.

    Args:
        layer_id: The annotation layer ID.
        response_mode: 'compact', 'standard', or 'full'.
    """
    ws = _get_ws()
    record = ws.annotation_layer_detail(layer_id)
    annotations = ws.annotation_layer_annotations(layer_id)

    if response_mode == "compact":
        data = {k: record[k] for k in ["id", "name"] if k in record}
        data["annotation_count"] = len(annotations)
    elif response_mode == "standard":
        data = {k: record[k] for k in [
            "id", "name", "descr", "changed_on",
        ] if k in record}
        data["annotations"] = [
            {k: a[k] for k in ["id", "short_descr", "start_dttm", "end_dttm"] if k in a}
            for a in annotations
        ]
    else:
        data = {**record, "annotations": annotations}
    out: dict[str, Any] = {"response_mode": response_mode, "data": data}
    if response_mode != "full":
        out["hint"] = "Set response_mode='full' to see all fields."
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
@_handle_errors
def create_annotation_layer(
    name: str,
    descr: str | None = None,
    dry_run: bool = False,
) -> str:
    """Create a new annotation layer.

    Annotation layers group time-based annotations that can be overlaid
    on time-series charts.  After creating a layer, use create_annotation
    to add individual annotations.

    Args:
        name: Name for the annotation layer.
        descr: Optional description.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    fields = ["name"]
    if descr:
        fields.append("descr")

    return _do_mutation(
        tool_name="create_annotation_layer",
        resource_type="annotation_layer",
        action="create",
        fields_changed=fields,
        dry_run=dry_run,
        execute=lambda: ws.create_annotation_layer(name=name, descr=descr),
    )


@mcp.tool()
@_handle_errors
def update_annotation_layer(
    layer_id: int,
    name: str | None = None,
    descr: str | None = None,
    dry_run: bool = False,
) -> str:
    """Update an existing annotation layer.

    Args:
        layer_id: The annotation layer ID to update.
        name: New name for the layer.
        descr: New description.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if descr is not None:
        kwargs["descr"] = descr

    if not kwargs:
        raise ValueError("No fields to update. Pass at least one of: name, descr.")

    before = capture_before(ws, "annotation_layer", layer_id)

    return _do_mutation(
        tool_name="update_annotation_layer",
        resource_type="annotation_layer",
        action="update",
        resource_id=layer_id,
        fields_changed=list(kwargs.keys()),
        dry_run=dry_run,
        before=before,
        execute=lambda: ws.update_annotation_layer(layer_id, **kwargs),
    )


@mcp.tool()
@_handle_errors
def delete_annotation_layer(
    layer_id: int,
    dry_run: bool = False,
) -> str:
    """Delete an annotation layer and all its annotations.

    Args:
        layer_id: The annotation layer ID to delete.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    before = capture_before(ws, "annotation_layer", layer_id)

    return _do_mutation(
        tool_name="delete_annotation_layer",
        resource_type="annotation_layer",
        action="delete",
        resource_id=layer_id,
        fields_changed=[],
        dry_run=dry_run,
        before=before,
        execute=lambda: ws.delete_annotation_layer(layer_id) or {},
    )


@mcp.tool()
@_handle_errors
def create_annotation(
    layer_id: int,
    short_descr: str,
    start_dttm: str,
    end_dttm: str,
    long_descr: str | None = None,
    dry_run: bool = False,
) -> str:
    """Add an annotation to an existing annotation layer.

    Annotations are time-based markers that overlay on time-series charts.
    Use list_annotation_layers or get_annotation_layer to find layer IDs.

    Args:
        layer_id: The annotation layer ID to add this annotation to.
        short_descr: Short description (label) shown on the chart overlay.
        start_dttm: Start datetime in ISO 8601 format (e.g. '2024-01-15T00:00:00').
        end_dttm: End datetime in ISO 8601 format (e.g. '2024-01-16T00:00:00').
        long_descr: Optional longer description with details.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    fields = ["short_descr", "start_dttm", "end_dttm"]
    if long_descr:
        fields.append("long_descr")

    return _do_mutation(
        tool_name="create_annotation",
        resource_type="annotation",
        action="create",
        fields_changed=fields,
        dry_run=dry_run,
        execute=lambda: ws.create_annotation(
            layer_id=layer_id,
            short_descr=short_descr,
            start_dttm=start_dttm,
            end_dttm=end_dttm,
            long_descr=long_descr,
        ),
    )


@mcp.tool()
@_handle_errors
def delete_annotation(
    layer_id: int,
    annotation_id: int,
    dry_run: bool = False,
) -> str:
    """Delete a specific annotation from an annotation layer.

    Args:
        layer_id: The annotation layer ID.
        annotation_id: The annotation ID to delete.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()

    return _do_mutation(
        tool_name="delete_annotation",
        resource_type="annotation",
        action="delete",
        resource_id=annotation_id,
        fields_changed=[],
        dry_run=dry_run,
        execute=lambda: ws.delete_annotation(layer_id, annotation_id) or {},
    )


# ===================================================================
# Tools — Async Query Results
# ===================================================================


@mcp.tool()
@_handle_errors
def get_async_query_result(
    query_id: str,
) -> str:
    """Fetch results of an async SQL query by its query ID (key).

    When SQL Lab runs a query asynchronously, it returns a query ID
    (also called a 'key').  Use this tool to poll for and retrieve
    the results once the query completes.

    Args:
        query_id: The query ID / key returned by an async SQL Lab query.
    """
    ws = _get_ws()
    result = ws.async_query_result(query_id)
    return json.dumps(result, indent=2, default=str)


# ===================================================================
# Tools — Embedded Dashboards
# ===================================================================


@mcp.tool()
@_handle_errors
def get_embedded_dashboard(
    dashboard_id: int,
) -> str:
    """Get the embedded configuration for a dashboard.

    Returns the embedding UUID and allowed domains if embedding is enabled,
    or indicates that embedding is not configured.

    Args:
        dashboard_id: The dashboard ID.
    """
    ws = _get_ws()
    result = ws.get_embedded_dashboard(dashboard_id)
    if result is None:
        return json.dumps({
            "dashboard_id": dashboard_id,
            "embedded": False,
            "hint": "Use enable_embedded_dashboard to enable embedding.",
        }, indent=2)
    return json.dumps({
        "dashboard_id": dashboard_id,
        "embedded": True,
        "data": result,
    }, indent=2, default=str)


@mcp.tool()
@_handle_errors
def enable_embedded_dashboard(
    dashboard_id: int,
    allowed_domains: list[str] | str | None = None,
    dry_run: bool = False,
) -> str:
    """Enable embedding for a dashboard and return its embed UUID.

    The returned UUID is used to embed the dashboard in external
    applications via the Superset embedded SDK.

    Args:
        dashboard_id: The dashboard ID to enable embedding for.
        allowed_domains: List of domains allowed to embed (e.g.
                         '["app.example.com"]').  Pass an empty list to
                         allow all origins.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()
    domains: list[str] = []
    if isinstance(allowed_domains, str):
        try:
            parsed = json.loads(allowed_domains)
            if isinstance(parsed, list):
                domains = [str(d) for d in parsed]
            else:
                domains = [str(allowed_domains)]
        except (json.JSONDecodeError, TypeError):
            domains = [d.strip() for d in allowed_domains.split(",") if d.strip()]
    elif isinstance(allowed_domains, list):
        domains = [str(d) for d in allowed_domains]

    return _do_mutation(
        tool_name="enable_embedded_dashboard",
        resource_type="dashboard",
        action="update",
        resource_id=dashboard_id,
        fields_changed=["embedded", "allowed_domains"],
        dry_run=dry_run,
        execute=lambda: ws.create_embedded_dashboard(
            dashboard_id=dashboard_id,
            allowed_domains=domains,
        ),
    )


@mcp.tool()
@_handle_errors
def disable_embedded_dashboard(
    dashboard_id: int,
    dry_run: bool = False,
) -> str:
    """Disable embedding for a dashboard.

    This revokes the embed UUID and prevents the dashboard from being
    embedded in external applications.

    Args:
        dashboard_id: The dashboard ID to disable embedding for.
        dry_run: If True, preview the action without executing.
    """
    ws = _get_ws()

    return _do_mutation(
        tool_name="disable_embedded_dashboard",
        resource_type="dashboard",
        action="update",
        resource_id=dashboard_id,
        fields_changed=["embedded"],
        dry_run=dry_run,
        execute=lambda: ws.delete_embedded_dashboard(dashboard_id) or {},
    )


# ===================================================================
# Entry point
# ===================================================================


def main():
    mcp.run()


if __name__ == "__main__":
    main()
