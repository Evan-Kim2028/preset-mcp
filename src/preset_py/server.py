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
from typing import Any, Literal

import sqlglot
from sqlglot import exp as E

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from preset_py.client import PresetWorkspace, connect
from preset_py._safety import (
    MutationEntry,
    capture_before,
    check_dataset_dependents,
    record_mutation,
    validate_params_json,
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


def _handle_errors(fn):
    """Decorator: catch exceptions → structured ToolError with hints."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        name = fn.__name__
        try:
            result = fn(*args, **kwargs)
            dur = (time.monotonic() - t0) * 1000
            _log.info("tool=%s status=ok duration_ms=%.0f", name, dur)
            return result

        except KeyError as exc:
            _log.warning("tool=%s error=missing_env key=%s", name, exc)
            raise ToolError(
                json.dumps({
                    "error": f"Missing environment variable: {exc}",
                    "error_type": "configuration",
                    "hints": [
                        "Set PRESET_API_TOKEN and PRESET_API_SECRET.",
                        "Optionally set PRESET_WORKSPACE to auto-connect.",
                    ],
                })
            ) from exc

        except RuntimeError as exc:
            _log.warning("tool=%s error=no_workspace", name)
            raise ToolError(
                json.dumps({
                    "error": str(exc),
                    "error_type": "no_workspace",
                    "hints": [
                        "Call list_workspaces to see available workspaces.",
                        "Then call use_workspace('Title') to select one.",
                    ],
                })
            ) from exc

        except ValueError as exc:
            _log.warning("tool=%s error=validation msg=%s", name, exc)
            raise ToolError(
                json.dumps({
                    "error": str(exc),
                    "error_type": "validation",
                    "hints": [
                        "Check parameter values and try again.",
                    ],
                })
            ) from exc

        except ToolError:
            raise

        except Exception as exc:
            dur = (time.monotonic() - t0) * 1000
            _log.error(
                "tool=%s error=api duration_ms=%.0f msg=%s", name, dur, exc,
            )
            raise ToolError(
                json.dumps({
                    "error": f"Preset API error: {exc}",
                    "error_type": "api_error",
                    "hints": [
                        "This may be transient — retry once.",
                        "If the error mentions a resource ID, verify it "
                        "exists with the corresponding list_* tool.",
                    ],
                })
            ) from exc

    return wrapper


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
def list_dashboards(response_mode: ResponseMode = "standard") -> str:
    """List dashboards in the current workspace.

    Start here to discover dashboard IDs, then use get_dashboard for
    detail on a specific one.

    Args:
        response_mode: 'compact' (id+title), 'standard' (key fields),
                       or 'full' (raw API response).  Default: standard.
    """
    ws = _get_ws()
    raw = ws.dashboards()
    _log.info("list_dashboards count=%d mode=%s", len(raw), response_mode)
    return _format_list(raw, "dashboard", response_mode)


@mcp.tool()
@_handle_errors
def get_dashboard(dashboard_id: int) -> str:
    """Get full detail for a single dashboard.

    Use list_dashboards first to find valid IDs.

    Args:
        dashboard_id: Numeric dashboard ID
    """
    ws = _get_ws()
    return json.dumps(ws.dashboard_detail(dashboard_id), indent=2, default=str)


@mcp.tool()
@_handle_errors
def list_charts(response_mode: ResponseMode = "standard") -> str:
    """List charts in the current workspace.

    Use this to find chart IDs and see which viz types are in use.

    Args:
        response_mode: 'compact', 'standard', or 'full'.  Default: standard.
    """
    ws = _get_ws()
    raw = ws.charts()
    _log.info("list_charts count=%d mode=%s", len(raw), response_mode)
    return _format_list(raw, "chart", response_mode)


@mcp.tool()
@_handle_errors
def list_datasets(response_mode: ResponseMode = "standard") -> str:
    """List datasets (virtual tables) in the current workspace.

    Datasets are the data sources for charts.  Use list_databases to
    find connection IDs needed for creating new datasets.

    Args:
        response_mode: 'compact', 'standard', or 'full'.  Default: standard.
    """
    ws = _get_ws()
    raw = ws.datasets()
    _log.info("list_datasets count=%d mode=%s", len(raw), response_mode)
    return _format_list(raw, "dataset", response_mode)


@mcp.tool()
@_handle_errors
def list_databases(response_mode: ResponseMode = "standard") -> str:
    """List database connections in the current workspace.

    Call this BEFORE run_sql or create_dataset to find a valid
    database_id.

    Args:
        response_mode: 'compact', 'standard', or 'full'.  Default: standard.
    """
    ws = _get_ws()
    raw = ws.databases()
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
    if dry_run:
        entry = MutationEntry(
            tool_name="create_dashboard",
            resource_type="dashboard",
            action="create",
            fields_changed=["dashboard_title", "published"],
            dry_run=True,
        )
        record_mutation(entry)
        return json.dumps({
            "dry_run": True,
            "action": "create_dashboard",
            "fields_to_change": ["dashboard_title", "published"],
            "values": {"dashboard_title": dashboard_title, "published": published},
        }, indent=2)

    ws = _get_ws()
    result = ws.create_dashboard(dashboard_title, published=published)
    _log.info("create_dashboard title=%s id=%s", dashboard_title, result.get("id"))

    record_mutation(MutationEntry(
        tool_name="create_dashboard",
        resource_type="dashboard",
        resource_id=result.get("id"),
        action="create",
        fields_changed=["dashboard_title", "published"],
        after_summary={"id": result.get("id"), "dashboard_title": dashboard_title},
    ))

    return json.dumps(result, indent=2, default=str)


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

    if dry_run:
        entry = MutationEntry(
            tool_name="create_dataset",
            resource_type="dataset",
            action="create",
            fields_changed=["name", "sql", "database_id", "schema"],
            dry_run=True,
        )
        record_mutation(entry)
        return json.dumps({
            "dry_run": True,
            "action": "create_dataset",
            "fields_to_change": ["name", "sql", "database_id", "schema"],
            "values": {
                "name": name,
                "sql": sql,
                "database_id": database_id,
                "schema": schema,
            },
        }, indent=2)

    ws = _get_ws()
    result = ws.create_dataset(name, sql, database_id, schema=schema)
    _log.info("create_dataset name=%s id=%s", name, result.get("id"))

    record_mutation(MutationEntry(
        tool_name="create_dataset",
        resource_type="dataset",
        resource_id=result.get("id"),
        action="create",
        fields_changed=["name", "sql", "database_id", "schema"],
        after_summary={"id": result.get("id"), "name": name},
    ))

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
@_handle_errors
def create_chart(
    dataset_id: int,
    title: str,
    viz_type: str,
    metrics: list[str] | None = None,
    groupby: list[str] | None = None,
    time_column: str | None = None,
    dashboards: list[int] | None = None,
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
        dry_run: If True, validate inputs and return a preview without
                 making any changes (default: False)
    """
    fields = ["dataset_id", "title", "viz_type"]
    if metrics is not None:
        fields.append("metrics")
    if groupby is not None:
        fields.append("groupby")
    if time_column is not None:
        fields.append("time_column")
    if dashboards is not None:
        fields.append("dashboards")

    if dry_run:
        entry = MutationEntry(
            tool_name="create_chart",
            resource_type="chart",
            action="create",
            fields_changed=fields,
            dry_run=True,
        )
        record_mutation(entry)
        return json.dumps({
            "dry_run": True,
            "action": "create_chart",
            "fields_to_change": fields,
            "values": {
                "dataset_id": dataset_id,
                "title": title,
                "viz_type": viz_type,
                "metrics": metrics,
                "groupby": groupby,
                "time_column": time_column,
                "dashboards": dashboards,
            },
        }, indent=2)

    ws = _get_ws()
    result = ws.create_chart(
        dataset_id,
        title,
        viz_type,
        metrics=metrics,
        groupby=groupby,
        time_column=time_column,
        dashboards=dashboards,
    )
    _log.info("create_chart title=%s id=%s", title, result.get("id"))

    record_mutation(MutationEntry(
        tool_name="create_chart",
        resource_type="chart",
        resource_id=result.get("id"),
        action="create",
        fields_changed=fields,
        after_summary={"id": result.get("id"), "title": title},
    ))

    return json.dumps(result, indent=2, default=str)


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

    # Pre-mutation snapshot
    before = capture_before(ws, "dataset", dataset_id)

    # Dependency check when SQL changes or columns are overridden
    dependency_impact: dict[str, Any] | None = None
    if sql is not None or override_columns:
        dependency_impact = check_dataset_dependents(ws, dataset_id)

    if dry_run:
        entry = MutationEntry(
            tool_name="update_dataset",
            resource_type="dataset",
            resource_id=dataset_id,
            action="update",
            fields_changed=list(kwargs.keys()),
            before_snapshot=before,
            dry_run=True,
        )
        record_mutation(entry)
        preview: dict[str, Any] = {
            "dry_run": True,
            "action": "update_dataset",
            "dataset_id": dataset_id,
            "fields_to_change": list(kwargs.keys()),
            "current_state": before,
        }
        if dependency_impact:
            preview["dependency_impact"] = dependency_impact
        return json.dumps(preview, indent=2, default=str)

    result = ws.update_dataset(
        dataset_id, override_columns=override_columns, **kwargs
    )
    _log.info("update_dataset id=%d fields=%s", dataset_id, list(kwargs.keys()))

    record_mutation(MutationEntry(
        tool_name="update_dataset",
        resource_type="dataset",
        resource_id=dataset_id,
        action="update",
        fields_changed=list(kwargs.keys()),
        before_snapshot=before,
        after_summary={"id": dataset_id, "fields_updated": list(kwargs.keys())},
    ))

    # Attach advisory info to the response
    response = result if isinstance(result, dict) else {"result": result}
    if dependency_impact:
        response["_dependency_impact"] = dependency_impact
    if override_columns and dependency_impact and dependency_impact.get("chart_count", 0) > 0:
        response["_override_columns_warning"] = (
            f"override_columns=True will refresh column metadata. "
            f"{dependency_impact['chart_count']} chart(s) may be affected: "
            f"{[c['name'] for c in dependency_impact.get('affected_charts', [])]}"
        )

    return json.dumps(response, indent=2, default=str)


@mcp.tool()
@_handle_errors
def update_chart(
    chart_id: int,
    title: str | None = None,
    viz_type: str | None = None,
    params_json: str | None = None,
    dashboards: list[int] | None = None,
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
                     get_dashboard to inspect existing chart params first)
        dashboards: Reassign chart to these dashboard IDs
        dry_run: If True, validate inputs, capture current state, and
                 return a preview without making any changes
                 (default: False)
    """
    # Validate params_json before building kwargs
    if params_json is not None:
        validate_params_json(params_json)

    kwargs: dict[str, Any] = {}
    if title is not None:
        kwargs["slice_name"] = title
    if viz_type is not None:
        kwargs["viz_type"] = viz_type
    if params_json is not None:
        kwargs["params"] = params_json
    if dashboards is not None:
        kwargs["dashboards"] = dashboards

    if not kwargs:
        raise ValueError(
            "Provide at least one field to update "
            "(title, viz_type, params_json, or dashboards)."
        )

    ws = _get_ws()

    # Pre-mutation snapshot
    before = capture_before(ws, "chart", chart_id)

    if dry_run:
        entry = MutationEntry(
            tool_name="update_chart",
            resource_type="chart",
            resource_id=chart_id,
            action="update",
            fields_changed=list(kwargs.keys()),
            before_snapshot=before,
            dry_run=True,
        )
        record_mutation(entry)
        return json.dumps({
            "dry_run": True,
            "action": "update_chart",
            "chart_id": chart_id,
            "fields_to_change": list(kwargs.keys()),
            "current_state": before,
        }, indent=2, default=str)

    result = ws.update_chart(chart_id, **kwargs)
    _log.info("update_chart id=%d fields=%s", chart_id, list(kwargs.keys()))

    record_mutation(MutationEntry(
        tool_name="update_chart",
        resource_type="chart",
        resource_id=chart_id,
        action="update",
        fields_changed=list(kwargs.keys()),
        before_snapshot=before,
        after_summary={"id": chart_id, "fields_updated": list(kwargs.keys())},
    ))

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
@_handle_errors
def update_dashboard(
    dashboard_id: int,
    dashboard_title: str | None = None,
    published: bool | None = None,
    dry_run: bool = False,
) -> str:
    """Update an existing dashboard's title or published status.

    Use list_dashboards to find the dashboard_id.

    Args:
        dashboard_id: ID of the dashboard to update
        dashboard_title: New dashboard title
        published: Set to True to publish, False to unpublish
        dry_run: If True, validate inputs, capture current state, and
                 return a preview without making any changes
                 (default: False)
    """
    kwargs: dict[str, Any] = {}
    if dashboard_title is not None:
        kwargs["dashboard_title"] = dashboard_title
    if published is not None:
        kwargs["published"] = published

    if not kwargs:
        raise ValueError(
            "Provide at least one field to update "
            "(dashboard_title or published)."
        )

    ws = _get_ws()

    # Pre-mutation snapshot
    before = capture_before(ws, "dashboard", dashboard_id)

    if dry_run:
        entry = MutationEntry(
            tool_name="update_dashboard",
            resource_type="dashboard",
            resource_id=dashboard_id,
            action="update",
            fields_changed=list(kwargs.keys()),
            before_snapshot=before,
            dry_run=True,
        )
        record_mutation(entry)
        return json.dumps({
            "dry_run": True,
            "action": "update_dashboard",
            "dashboard_id": dashboard_id,
            "fields_to_change": list(kwargs.keys()),
            "current_state": before,
        }, indent=2, default=str)

    result = ws.update_dashboard(dashboard_id, **kwargs)
    _log.info("update_dashboard id=%d fields=%s", dashboard_id, list(kwargs.keys()))

    record_mutation(MutationEntry(
        tool_name="update_dashboard",
        resource_type="dashboard",
        resource_id=dashboard_id,
        action="update",
        fields_changed=list(kwargs.keys()),
        before_snapshot=before,
        after_summary={"id": dashboard_id, "fields_updated": list(kwargs.keys())},
    ))

    return json.dumps(result, indent=2, default=str)


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
# Entry point
# ===================================================================


def main():
    mcp.run()


if __name__ == "__main__":
    main()
