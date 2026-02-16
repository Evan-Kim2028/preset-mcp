"""Core wrapper: connect() entry point and PresetWorkspace convenience class."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from yarl import URL

from preset_cli.auth.lib import get_access_token
from preset_cli.auth.jwt import JWTAuth
from preset_cli.api.clients.preset import PresetClient
from preset_cli.api.clients.superset import SupersetClient

from preset_py.snapshot import WorkspaceSnapshot, take_snapshot

PRESET_API_URL = "https://api.app.preset.io/"


# ---------------------------------------------------------------------------
# Dataset / chart creation helpers (moved from _helpers.py)
# ---------------------------------------------------------------------------


def _create_virtual_dataset(
    client: SupersetClient,
    name: str,
    sql: str,
    database_id: int,
    schema: str | None = None,
) -> dict[str, Any]:
    """Create a virtual (SQL-based) dataset and refresh its column metadata."""
    payload: dict[str, Any] = {
        "table_name": name,
        "sql": sql,
        "database": database_id,
    }
    if schema:
        payload["schema"] = schema

    result = client.create_dataset(**payload)

    # Refresh column metadata so the dataset is immediately usable
    dataset_id = result.get("id")
    if dataset_id:
        try:
            client.get_refreshed_dataset_columns(dataset_id)
        except Exception as exc:
            # Surface degraded-state warning instead of silently swallowing it.
            result["_warning"] = (
                "Dataset created but column refresh failed: "
                f"{exc}. Columns may not be available until first query."
            )

    return result


def _column_map(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build column_name -> column dict mapping from a dataset payload."""
    columns = dataset.get("columns", [])
    if not isinstance(columns, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for column in columns:
        if not isinstance(column, dict):
            continue
        name = column.get("column_name")
        if isinstance(name, str) and name:
            mapped[name] = column
    return mapped


def _column_type_name(column: dict[str, Any]) -> str:
    """Return a normalized type name from dataset column metadata."""
    for key in ("type", "type_generic", "python_date_format"):
        raw = column.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().upper()
    return ""


def _is_numeric_column(column: dict[str, Any]) -> bool:
    """Best-effort numeric column detection from Superset metadata."""
    type_name = _column_type_name(column)
    numeric_tokens = (
        "INT", "NUMBER", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE",
        "REAL", "BIGINT", "SMALLINT", "TINYINT",
    )
    return any(token in type_name for token in numeric_tokens)


def _simple_metric_from_column(column_name: str, column: dict[str, Any]) -> dict[str, Any]:
    """Build a renderable SIMPLE metric object from a dataset column."""
    aggregate = "SUM" if _is_numeric_column(column) else "COUNT"
    return {
        "expressionType": "SIMPLE",
        "column": {
            "column_name": column_name,
            "type": column.get("type"),
        },
        "aggregate": aggregate,
        "label": f"{aggregate}({column_name})",
        "optionName": f"metric_{aggregate.lower()}_{column_name}",
    }


def _normalize_create_metrics(
    metrics: list[Any],
    dataset: dict[str, Any],
) -> list[Any]:
    """Normalize string metrics to Superset-compatible metric definitions."""
    dataset_metric_names = _dataset_metric_names(dataset)
    columns = _column_map(dataset)

    normalized: list[Any] = []
    for metric in metrics:
        if isinstance(metric, dict):
            normalized.append(metric)
            continue

        metric_name = str(metric)
        if metric_name in dataset_metric_names:
            normalized.append(metric_name)
            continue

        column = columns.get(metric_name)
        if column is None:
            raise ValueError(
                f"Unknown metric {metric_name!r}. Provide a saved metric name "
                "or a dataset column name."
            )
        normalized.append(_simple_metric_from_column(metric_name, column))
    return normalized


def _validate_groupby_columns(groupby: list[str], dataset: dict[str, Any]) -> None:
    """Fail fast when groupby references missing dataset columns."""
    columns = set(_column_map(dataset).keys())
    missing = [col for col in groupby if col not in columns]
    if missing:
        raise ValueError(
            f"groupby contains unknown column(s): {missing}. "
            "Use get_dataset(..., response_mode='standard') to inspect valid columns."
        )


def _create_chart(
    client: SupersetClient,
    dataset_id: int,
    title: str,
    viz_type: str,
    *,
    metrics: list[Any] | None = None,
    groupby: list[str] | None = None,
    time_column: str | None = None,
    dashboards: list[int] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a chart with sensible defaults.

    ``extra_params`` is merged into the chart params dict, allowing callers
    to pass any viz-type-specific options.
    """
    dataset = client.get_dataset(dataset_id)

    params: dict[str, Any] = {
        "viz_type": viz_type,
        "datasource": f"{dataset_id}__table",
        "row_limit": 10000,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "adhoc_filters": [],
    }
    if metrics:
        params["metrics"] = _normalize_create_metrics(metrics, dataset)
    if groupby:
        _validate_groupby_columns(groupby, dataset)
        params["groupby"] = groupby
    if time_column:
        params["granularity_sqla"] = time_column

    if extra_params:
        params.update(extra_params)

    payload: dict[str, Any] = {
        "slice_name": title,
        "viz_type": viz_type,
        "datasource_id": dataset_id,
        "datasource_type": "table",
        "params": json.dumps(params),
    }
    if dashboards:
        payload["dashboards"] = dashboards

    return client.create_resource("chart", **payload)


def _coerce_list(value: Any) -> list[Any]:
    """Normalize scalar/list-like values to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_int(value: Any) -> int | None:
    """Convert values like numeric strings to integers."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _normalize_filters(filters: Any) -> list[Any]:
    """Normalize filters to a list."""
    if filters is None:
        return []
    if isinstance(filters, list):
        return filters
    return [filters]


def _safe_set(values: list[Any]) -> set[str]:
    """Return a string set for quick case-sensitive membership checks."""
    normalized: set[str] = set()
    for value in values:
        if isinstance(value, str):
            normalized.add(value)
        else:
            normalized.add(str(value))
    return normalized


def _dataset_column_names(dataset: dict[str, Any]) -> set[str]:
    """Extract column names from a dataset payload in flexible shapes."""
    raw_columns = dataset.get("columns")
    columns: list[Any] = []
    if isinstance(raw_columns, list):
        columns.extend(raw_columns)
    elif isinstance(raw_columns, dict):
        columns.extend(raw_columns.values())
    extracted: list[Any] = []
    for column in columns:
        if isinstance(column, dict):
            extracted.append(
                column.get("column_name")
                or column.get("name")
                or column.get("id")
            )
        else:
            extracted.append(column)
    return _safe_set(extracted)


def _dataset_metric_names(dataset: dict[str, Any]) -> set[str]:
    """Extract dataset metric names from a dataset payload in flexible shapes."""
    metrics = dataset.get("metrics", [])
    if not isinstance(metrics, list):
        return set()
    extracted: list[Any] = []
    for metric in metrics:
        if isinstance(metric, dict):
            extracted.append(
                metric.get("metric_name")
                or metric.get("label")
                or metric.get("name")
            )
        else:
            extracted.append(metric)
    return _safe_set(extracted)


def _metric_to_adhoc(column_name: str) -> dict[str, Any]:
    """Build a SQL metric payload from a dataset column name."""
    expression = f"SUM({column_name})"
    return {
        "expressionType": "SQL",
        "sqlExpression": expression,
        "label": expression,
        "optionName": f"metric_{column_name}",
    }


def _metric_label(metric: Any) -> str | None:
    """Get a stable label used by Superset for sorting and display."""
    if isinstance(metric, str):
        return metric
    if isinstance(metric, dict):
        label = metric.get("label") or metric.get("label_short")
        if label:
            return str(label)
        if metric.get("expressionType") == "SQL" and metric.get("sqlExpression"):
            return str(metric["sqlExpression"])
        if metric.get("column"):
            return str(metric["column"])
    return None


def _normalize_metrics(
    metrics: list[Any],
    dataset_columns: set[str],
    dataset_metrics: set[str],
) -> list[Any]:
    """Convert raw metric specs into chart-data-compatible specs."""
    normalized: list[Any] = []
    for metric in metrics:
        if metric is None or metric == "":
            continue
        if isinstance(metric, dict):
            normalized.append(metric)
            continue
        metric_name = str(metric)
        if metric_name in dataset_metrics or metric_name not in dataset_columns:
            normalized.append(metric_name)
        else:
            normalized.append(_metric_to_adhoc(metric_name))
    return normalized


def _normalize_orderby(
    orderby: Any,
    metrics: list[Any],
    dataset_columns: set[str] | None = None,
) -> list[list[Any]]:
    """Normalize chart orderby values into Superset chart-data shape."""
    dataset_columns = dataset_columns or set()
    metric_labels = {_metric_label(metric) for metric in metrics if _metric_label(metric)}
    normalized: list[list[Any]] = []
    if not isinstance(orderby, list):
        orderby = []
    for item in orderby:
        direction = False
        metric_key = None
        if isinstance(item, list | tuple) and len(item) >= 1:
            metric_key = _metric_label(item[0])
            if len(item) >= 2:
                direction = bool(item[1])
        elif isinstance(item, dict):
            metric_key = _metric_label(item)
            direction = False
        elif isinstance(item, str):
            metric_key = item
        if metric_key in dataset_columns and metric_key not in metric_labels:
            metric_key = f"SUM({metric_key})"
        if metric_key is not None:
            normalized.append([metric_key, direction])
    if not normalized and metrics:
        fallback = _metric_label(metrics[0]) or str(metrics[0])
        normalized = [[fallback, False]]
    return normalized


def _datasource_from_form_data(form_data: dict[str, Any]) -> tuple[int, str]:
    """Parse datasource definition from dashboard chart form_data."""
    datasource = form_data.get("datasource")
    if isinstance(datasource, dict):
        datasource_id = _as_int(datasource.get("id"))
        datasource_type = str(datasource.get("type", "table"))
        if datasource_id is None:
            raise ValueError("Invalid datasource id in form_data.")
        return datasource_id, datasource_type
    if not isinstance(datasource, str):
        raise ValueError("Datasource is missing or invalid in form_data.")
    if "__" not in datasource:
        raise ValueError(
            "Datasource format is invalid. Expected '<id>__<type>' "
            f"but got {datasource!r}."
        )
    raw_id, datasource_type = datasource.split("__", 1)
    datasource_id = _as_int(raw_id)
    if datasource_id is None:
        raise ValueError(f"Could not parse datasource id from {raw_id!r}.")
    return datasource_id, datasource_type or "table"


def _parse_json_dict(value: Any) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object payload."""
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


def _datasource_from_chart_payloads(
    chart: dict[str, Any],
    form_data: dict[str, Any] | None = None,
) -> tuple[int | None, str | None, str | None]:
    """Resolve datasource from available chart payloads in priority order."""
    candidates: list[tuple[dict[str, Any] | None, str]] = [
        (form_data, "dashboard.form_data"),
        (_parse_json_dict(chart.get("params")), "chart.params"),
        (_parse_json_dict(chart.get("query_context")), "chart.query_context"),
    ]
    for payload, source in candidates:
        if not isinstance(payload, dict):
            continue
        try:
            datasource_id, datasource_type = _datasource_from_form_data(payload)
            return datasource_id, datasource_type, source
        except ValueError:
            continue

    datasource_id = _as_int(chart.get("datasource_id"))
    datasource_type = chart.get("datasource_type")
    if datasource_id is not None:
        return datasource_id, str(datasource_type or "table"), "chart.detail"
    return None, None, None


def _critical_page_errors(page_errors: list[str]) -> list[str]:
    """Filter browser page errors down to chart-impacting signals."""
    ignored_page_error_substrings = (
        "failed to fetch",
        "uncaught (in promise) #<response>",
        "unexpected token '<'",
        "is not valid json",
    )
    critical: list[str] = []
    for page_error in page_errors:
        page_error_lower = page_error.lower()
        if any(token in page_error_lower for token in ignored_page_error_substrings):
            continue
        critical.append(page_error)
    return critical


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def connect(workspace: str | None = None) -> PresetWorkspace:
    """One-liner entry point. Reads ``PRESET_API_TOKEN`` / ``PRESET_API_SECRET``
    from the environment and returns a :class:`PresetWorkspace`.

    Parameters
    ----------
    workspace:
        Workspace title to connect to (e.g. ``"Mysten Labs--General"``).
        If *None*, the returned object can still list workspaces and then
        switch into one via :meth:`PresetWorkspace.use`.
    """
    api_token = os.environ["PRESET_API_TOKEN"]
    api_secret = os.environ["PRESET_API_SECRET"]

    jwt = get_access_token(PRESET_API_URL, api_token, api_secret)
    auth = JWTAuth(jwt)

    preset_client = PresetClient(PRESET_API_URL, auth)

    # Cache the workspace list (needed for title → hostname resolution)
    teams = preset_client.get_teams()
    all_workspaces: list[dict[str, Any]] = []
    for team in teams:
        all_workspaces.extend(preset_client.get_workspaces(team["name"]))

    workspace_url: URL | None = None
    if workspace:
        workspace_url = _resolve_workspace(all_workspaces, workspace)

    return PresetWorkspace(
        auth=auth,
        preset_client=preset_client,
        workspace_url=workspace_url,
        _workspaces=all_workspaces,
    )


def _resolve_workspace(workspaces: list[dict[str, Any]], title: str) -> URL:
    """Find workspace by title and return its URL."""
    for ws in workspaces:
        if ws.get("title") == title:
            hostname = ws["hostname"]
            return URL(f"https://{hostname}/")
    available = [ws.get("title", ws.get("hostname", "?")) for ws in workspaces]
    raise ValueError(
        f"Workspace {title!r} not found. Available: {available}"
    )


class PresetWorkspace:
    """Convenience wrapper around PresetClient + SupersetClient.

    All CRUD methods return plain Python dicts/lists/DataFrames so that
    callers never need to touch the underlying SDK objects.
    """

    def __init__(
        self,
        auth: JWTAuth,
        preset_client: PresetClient,
        workspace_url: URL | None = None,
        _workspaces: list[dict[str, Any]] | None = None,
    ) -> None:
        self._auth = auth
        self._preset_client = preset_client
        self._workspaces = _workspaces or []
        self.workspace_url = workspace_url

        self._superset: SupersetClient | None = None
        if workspace_url:
            self._superset = SupersetClient(str(workspace_url), auth)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @property
    def _client(self) -> SupersetClient:
        if self._superset is None:
            raise RuntimeError(
                "No workspace selected. Call .use('Workspace Title') first, "
                "or pass a workspace name to connect()."
            )
        return self._superset

    # ------------------------------------------------------------------
    # Workspace navigation
    # ------------------------------------------------------------------

    def list_workspaces(self) -> list[dict[str, Any]]:
        """Return cached list of workspaces (title + hostname)."""
        return [
            {"title": ws.get("title"), "hostname": ws.get("hostname")}
            for ws in self._workspaces
        ]

    def use(self, workspace_title: str) -> PresetWorkspace:
        """Switch to a different workspace and return a new PresetWorkspace."""
        url = _resolve_workspace(self._workspaces, workspace_title)
        return PresetWorkspace(
            auth=self._auth,
            preset_client=self._preset_client,
            workspace_url=url,
            _workspaces=self._workspaces,
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def dashboards(self, **filters: Any) -> list[dict[str, Any]]:
        return self._client.get_dashboards(**filters)

    def charts(self, **filters: Any) -> list[dict[str, Any]]:
        return self._client.get_charts(**filters)

    def datasets(self, **filters: Any) -> list[dict[str, Any]]:
        return self._client.get_datasets(**filters)

    def databases(self, **filters: Any) -> list[dict[str, Any]]:
        return self._client.get_databases(**filters)

    def get_resource(self, resource_type: str, resource_id: int) -> dict[str, Any]:
        """Fetch a single resource by type and ID."""
        return self._client.get_resource(resource_type, resource_id)

    def dashboard_detail(self, dashboard_id: int) -> dict[str, Any]:
        return self._client.get_dashboard(dashboard_id)

    def chart_detail(self, chart_id: int) -> dict[str, Any]:
        """Fetch full detail for a single chart (params, query_context, datasource)."""
        return self._client.get_chart(chart_id)

    def dataset_detail(self, dataset_id: int) -> dict[str, Any]:
        return self._client.get_dataset(dataset_id)

    def database_detail(self, database_id: int) -> dict[str, Any]:
        return self._client.get_database(database_id)

    def refresh_dataset_columns(self, dataset_id: int) -> list[Any]:
        return self._client.get_refreshed_dataset_columns(dataset_id)

    def dashboard_charts(self, dashboard_id: int) -> list[dict[str, Any]]:
        """Get chart definitions for one dashboard (includes form_data)."""
        response = self._client.session.get(
            str(self._client.baseurl / "api/v1" / "dashboard" / str(dashboard_id) / "charts")
        )
        payload = response.json()
        return payload.get("result", [])

    def chart_form_data(
        self,
        chart_id: int,
        dashboard_id: int | None = None,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """Return form_data for a chart from a specific dashboard or auto-scan all."""
        if dashboard_id is not None:
            charts = self.dashboard_charts(dashboard_id)
            for chart in charts:
                if chart.get("id") == chart_id:
                    return chart.get("form_data", {}), dashboard_id
            return None, dashboard_id

        for dashboard in self.dashboards():
            did = dashboard.get("id")
            if not isinstance(did, int):
                continue
            charts = self.dashboard_charts(did)
            for chart in charts:
                if chart.get("id") == chart_id:
                    return chart.get("form_data", {}), did
        return None, None

    def validate_chart_data(
        self,
        chart_id: int,
        dashboard_id: int | None = None,
        row_limit: int = 10000,
        force: bool = False,
    ) -> dict[str, Any]:
        """Execute chart query context and return render-status data.

        Returns a normalized dict with query status + errors. This validates chart
        parameter integrity (missing metrics/columns/filters) without waiting for
        full dashboard rendering.
        """
        chart = self.get_resource("chart", chart_id)
        sheet_name = chart.get("slice_name") or chart.get("name") or f"chart-{chart_id}"

        form_data, resolved_dashboard_id = self.chart_form_data(
            chart_id, dashboard_id=dashboard_id
        )
        form_data_source = "dashboard.form_data"
        if not isinstance(form_data, dict):
            form_data = _parse_json_dict(chart.get("params"))
            form_data_source = "chart.params" if isinstance(form_data, dict) else "missing"
        if not isinstance(form_data, dict):
            form_data = {}

        if resolved_dashboard_id is not None:
            resolved_dashboard_id = int(resolved_dashboard_id)
        resolved_dashboard_id = resolved_dashboard_id or dashboard_id

        datasource_id, datasource_type, datasource_source = _datasource_from_chart_payloads(
            chart,
            form_data=form_data,
        )
        if datasource_id is None or datasource_type is None:
            return {
                "chart_id": chart_id,
                "slice_name": sheet_name,
                "dashboard_id": resolved_dashboard_id,
                "status": "invalid_form_data",
                "error": (
                    "Could not resolve datasource from chart payloads "
                    "(dashboard form_data, chart.params, chart.query_context)."
                ),
                "form_data": form_data,
                "form_data_source": form_data_source,
            }

        query_context = _parse_json_dict(chart.get("query_context"))
        payload_source = "synthetic.form_data"
        if isinstance(query_context, dict) and isinstance(query_context.get("queries"), list):
            payload = copy.deepcopy(query_context)
            payload_source = "chart.query_context"
            payload["datasource"] = {
                "id": datasource_id,
                "type": datasource_type,
            }
            payload["force"] = force
            payload["result_format"] = "json"
            payload["result_type"] = "full"
            if isinstance(form_data, dict) and form_data:
                payload.setdefault("form_data", form_data)
            for query in payload.get("queries", []):
                if isinstance(query, dict):
                    query["row_limit"] = row_limit
        else:
            if not form_data:
                return {
                    "chart_id": chart_id,
                    "slice_name": sheet_name,
                    "dashboard_id": resolved_dashboard_id,
                    "status": "unsupported",
                    "error": (
                        "Could not locate chart query_context or form_data. "
                        "Pass dashboard_id explicitly if the chart is part of a "
                        "dashboard."
                    ),
                }

            raw_metrics = _coerce_list(form_data.get("metrics"))
            if not raw_metrics:
                raw_metric = form_data.get("metric")
                if raw_metric not in (None, "", []):
                    raw_metrics = _coerce_list(raw_metric)

            dataset: dict[str, Any] = {}
            try:
                dataset = self._client.get_dataset(datasource_id)
            except Exception:
                dataset = {}

            dataset_columns = _dataset_column_names(dataset)
            dataset_metrics = _dataset_metric_names(dataset)
            normalized_metrics = _normalize_metrics(
                raw_metrics,
                dataset_columns=dataset_columns,
                dataset_metrics=dataset_metrics,
            )

            columns: list[Any] = []
            columns.extend(_coerce_list(form_data.get("groupby")))
            columns.extend(_coerce_list(form_data.get("columns")))
            columns = list(dict.fromkeys(columns))

            filters = _normalize_filters(form_data.get("filters"))
            adhoc_filters = _normalize_filters(form_data.get("adhoc_filters"))

            query: dict[str, Any] = {
                "columns": columns,
                "metrics": normalized_metrics,
                "filters": filters,
                "adhoc_filters": adhoc_filters,
                "orderby": _normalize_orderby(
                    form_data.get("orderby"),
                    normalized_metrics,
                    dataset_columns,
                ),
                "is_timeseries": bool(
                    form_data.get("granularity_sqla")
                    or form_data.get("time_column")
                    or form_data.get("granularity")
                ),
                "extras": {
                    "where": "",
                    "having": "",
                },
                "annotation_layers": [],
                "custom_form_data": {},
                "custom_params": {},
                "row_limit": row_limit,
                "time_range": form_data.get("time_range", "No filter"),
                "url_params": {},
            }

            if form_data.get("granularity_sqla"):
                query["granularity_sqla"] = form_data["granularity_sqla"]
            if form_data.get("time_column"):
                query["granularity_sqla"] = form_data["time_column"]
            if form_data.get("granularity"):
                query.setdefault("extras", {})["time_grain_sqla"] = form_data["granularity"]

            payload: dict[str, Any] = {
                "datasource": {
                    "id": datasource_id,
                    "type": datasource_type,
                },
                "force": force,
                "queries": [query],
                "result_format": "json",
                "result_type": "full",
            }

        endpoint = str(self._client.baseurl / "api/v1" / "chart" / "data")
        response = self._client.session.post(endpoint, json=payload)

        try:
            body = response.json()
        except Exception:
            return {
                "chart_id": chart_id,
                "slice_name": sheet_name,
                "dashboard_id": resolved_dashboard_id,
                "status": "http_error",
                "error": "Non-JSON response from chart-data endpoint",
                "http_status": response.status_code,
                "raw_text": response.text[:400],
            }

        if response.status_code != 200:
            return {
                "chart_id": chart_id,
                "slice_name": sheet_name,
                "dashboard_id": resolved_dashboard_id,
                "status": "request_failed",
                "error": body.get("message", body),
                "http_status": response.status_code,
                "payload": payload,
                "payload_source": payload_source,
            }

        results = body.get("result", [])
        if not isinstance(results, list) or not results:
            return {
                "chart_id": chart_id,
                "slice_name": sheet_name,
                "dashboard_id": resolved_dashboard_id,
                "status": "malformed_response",
                "error": "Unexpected chart-data response shape",
                "payload": payload,
                "payload_source": payload_source,
            }

        first = results[0]
        query_results: list[dict[str, Any]] = []
        overall_status = "success"
        first_error: Any = None
        total_rows = 0
        for idx, item in enumerate(results):
            if not isinstance(item, dict):
                overall_status = "failed"
                malformed_error = f"query[{idx}] has malformed response entry."
                if first_error is None:
                    first_error = malformed_error
                query_results.append({
                    "index": idx,
                    "status": "malformed",
                    "error": malformed_error,
                    "row_count": None,
                })
                continue
            status = item.get("status")
            error = item.get("error")
            row_count = item.get("rowcount")
            if isinstance(row_count, int):
                total_rows += row_count
            query_results.append({
                "index": idx,
                "status": status,
                "error": error,
                "row_count": row_count,
            })
            if status != "success" or error:
                overall_status = "failed"
                if first_error is None:
                    first_error = error or f"query[{idx}] returned status {status!r}"

        return {
            "chart_id": chart_id,
            "slice_name": sheet_name,
            "dashboard_id": resolved_dashboard_id,
            "datasource": {
                "id": datasource_id,
                "type": datasource_type,
            },
            "status": overall_status if query_results else first.get("status"),
            "error": first_error,
            "row_count": first.get("rowcount"),
            "row_count_total": total_rows,
            "cache_key": first.get("cache_key"),
            "is_cached": first.get("is_cached"),
            "query": first.get("query"),
            "result_format": first.get("result_format"),
            "form_data": form_data,
            "form_data_source": form_data_source,
            "payload": payload,
            "payload_source": payload_source,
            "query_results": query_results,
            "query_context_present": bool(query_context),
            "datasource_source": datasource_source,
        }

    def validate_dashboard_charts(
        self,
        dashboard_id: int,
        row_limit: int = 10000,
        force: bool = False,
    ) -> dict[str, Any]:
        """Validate all charts on a dashboard using chart data endpoint."""
        charts = self.dashboard_charts(dashboard_id)
        results = []
        for chart in charts:
            cid = chart.get("id")
            if isinstance(cid, int):
                results.append(
                    self.validate_chart_data(cid, dashboard_id=dashboard_id, row_limit=row_limit, force=force)
                )
        return {
            "dashboard_id": dashboard_id,
            "chart_count": len(charts),
            "validated": len(results),
            "results": results,
        }

    def validate_chart_render(
        self,
        chart_id: int,
        *,
        timeout_ms: int = 45000,
        settle_ms: int = 2500,
        screenshot_on_failure: bool = True,
    ) -> dict[str, Any]:
        """Validate chart rendering in a headless browser session.

        This is stricter than query-only validation because it observes browser
        runtime failures (console/page errors and failed chart-data requests).
        """
        chart = self.get_resource("chart", chart_id)
        sheet_name = chart.get("slice_name") or chart.get("name") or f"chart-{chart_id}"

        auth_header = self._client.session.headers.get("Authorization")
        if not isinstance(auth_header, str) or not auth_header:
            return {
                "chart_id": chart_id,
                "slice_name": sheet_name,
                "status": "unsupported",
                "error": "Missing Authorization header for browser render validation.",
            }

        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            return {
                "chart_id": chart_id,
                "slice_name": sheet_name,
                "status": "unsupported",
                "error": (
                    "Playwright is not installed. Install it with "
                    "`pip install playwright` and run `playwright install chromium`."
                ),
                "exception": str(exc),
            }

        explore_url = str(
            self._client.baseurl.with_path("/explore/").with_query({"slice_id": chart_id})
        )
        workspace_origin = str(self._client.baseurl.origin())

        console_errors: list[str] = []
        page_errors: list[str] = []
        critical_page_errors: list[str] = []
        request_failures: list[dict[str, Any]] = []
        chart_data_failures: list[dict[str, Any]] = []
        visible_errors: list[str] = []
        screenshot_path: str | None = None
        navigation_error: str | None = None

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            def _on_console(msg):
                if msg.type == "error":
                    console_errors.append(msg.text)

            def _on_pageerror(exc):
                page_errors.append(str(exc))

            def _on_requestfailed(request):
                failure = request.failure
                if isinstance(failure, dict):
                    error_text = failure.get("errorText")
                elif isinstance(failure, str):
                    error_text = failure
                else:
                    error_text = getattr(failure, "error_text", None)
                request_failures.append(
                    {
                        "url": request.url,
                        "method": request.method,
                        "error_text": error_text,
                    }
                )
                if "/api/v1/chart/data" in request.url:
                    chart_data_failures.append(
                        {
                            "url": request.url,
                            "status": None,
                            "error_text": error_text,
                        }
                    )

            def _on_response(response):
                if "/api/v1/chart/data" not in response.url:
                    return
                if response.status < 400:
                    return
                body_preview = ""
                try:
                    body_preview = response.text()[:400]
                except Exception:
                    body_preview = ""
                chart_data_failures.append(
                    {
                        "url": response.url,
                        "status": response.status,
                        "body_preview": body_preview,
                    }
                )

            page.on("console", _on_console)
            page.on("pageerror", _on_pageerror)
            page.on("requestfailed", _on_requestfailed)
            page.on("response", _on_response)
            # Add bearer auth only for workspace-origin requests. Injecting it
            # cross-origin triggers CORS preflights that produce false failures.
            page.route(
                "**/*",
                lambda route, request: route.continue_(
                    headers={
                        **request.headers,
                        "Authorization": auth_header,
                    }
                )
                if request.url.startswith(workspace_origin)
                else route.continue_(),
            )

            try:
                page.goto(explore_url, wait_until="networkidle", timeout=timeout_ms)
                if settle_ms > 0:
                    page.wait_for_timeout(settle_ms)
            except Exception as exc:
                navigation_error = str(exc)
            critical_page_errors.extend(_critical_page_errors(page_errors))

            error_selectors = [
                ".alert-danger",
                ".ant-alert-error",
                ".Toastify__toast--error",
                "text=Unexpected error",
                "text=GENERIC_BACKEND_ERROR",
                "text=Issue 1011",
            ]
            for selector in error_selectors:
                locator = page.locator(selector)
                try:
                    count = locator.count()
                except Exception:
                    continue
                for idx in range(min(count, 3)):
                    item = locator.nth(idx)
                    try:
                        if not item.is_visible():
                            continue
                        text = item.inner_text(timeout=500).strip()
                        if text:
                            visible_errors.append(text[:400])
                    except Exception:
                        continue

            if screenshot_on_failure:
                has_failures = bool(
                    navigation_error
                    or critical_page_errors
                    or chart_data_failures
                    or visible_errors
                )
                if has_failures:
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    path = Path("/tmp") / f"preset_chart_render_{chart_id}_{ts}.png"
                    try:
                        page.screenshot(path=str(path), full_page=True)
                        screenshot_path = str(path)
                    except Exception:
                        screenshot_path = None

            browser.close()

        status = "success"
        if navigation_error or critical_page_errors or chart_data_failures or visible_errors:
            status = "failed"

        return {
            "chart_id": chart_id,
            "slice_name": sheet_name,
            "url": explore_url,
            "status": status,
            "error": navigation_error,
            "console_errors": console_errors,
            "page_errors": page_errors,
            "critical_page_errors": critical_page_errors,
            "request_failures": request_failures,
            "chart_data_failures": chart_data_failures,
            "visible_errors": visible_errors,
            "screenshot_path": screenshot_path,
        }

    def validate_dashboard_render(
        self,
        dashboard_id: int,
        *,
        timeout_ms: int = 45000,
        settle_ms: int = 2500,
    ) -> dict[str, Any]:
        """Run render validation on each chart referenced by a dashboard."""
        charts = self.dashboard_charts(dashboard_id)
        results: list[dict[str, Any]] = []
        for chart in charts:
            cid = chart.get("id")
            if not isinstance(cid, int):
                continue
            results.append(
                self.validate_chart_render(
                    cid,
                    timeout_ms=timeout_ms,
                    settle_ms=settle_ms,
                )
            )

        broken = [item for item in results if item.get("status") != "success"]
        return {
            "dashboard_id": dashboard_id,
            "chart_count": len(charts),
            "validated": len(results),
            "broken_count": len(broken),
            "broken_charts": broken,
            "results": results,
        }

    # ------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------

    def run_sql(
        self,
        sql: str,
        database_id: int,
        schema: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        return self._client.run_query(
            database_id=database_id, sql=sql, schema=schema, limit=limit
        )

    def query_dataset(
        self,
        dataset_id: int,
        metrics: list[str],
        columns: list[str] | None = None,
        order_by: list[str] | None = None,
        order_desc: bool = True,
        is_timeseries: bool = False,
        time_column: str | None = None,
        start: str | None = None,
        end: str | None = None,
        granularity: str | None = None,
        where: str = "",
        having: str = "",
        row_limit: int = 10000,
        force: bool = False,
    ) -> pd.DataFrame:
        from datetime import datetime

        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None
        return self._client.get_data(
            dataset_id=dataset_id,
            metrics=metrics,
            columns=columns or [],
            order_by=order_by,
            order_desc=order_desc,
            is_timeseries=is_timeseries,
            time_column=time_column,
            start=start_dt,
            end=end_dt,
            granularity=granularity,
            where=where,
            having=having,
            row_limit=row_limit,
            force=force,
        )

    # ------------------------------------------------------------------
    # Create helpers
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        sql: str,
        database_id: int,
        schema: str | None = None,
    ) -> dict[str, Any]:
        return _create_virtual_dataset(
            self._client, name, sql, database_id, schema
        )

    def create_chart(
        self,
        dataset_id: int,
        title: str,
        viz_type: str,
        *,
        metrics: list[Any] | None = None,
        groupby: list[str] | None = None,
        time_column: str | None = None,
        dashboards: list[int] | None = None,
        **extra_params: Any,
    ) -> dict[str, Any]:
        return _create_chart(
            self._client,
            dataset_id,
            title,
            viz_type,
            metrics=metrics,
            groupby=groupby,
            time_column=time_column,
            dashboards=dashboards,
            extra_params=extra_params or None,
        )

    # ------------------------------------------------------------------
    # Create helpers (dashboards)
    # ------------------------------------------------------------------

    def create_dashboard(
        self,
        dashboard_title: str,
        published: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.create_dashboard(
            dashboard_title=dashboard_title,
            published=published,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update_dataset(
        self,
        dataset_id: int,
        override_columns: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.update_dataset(
            dataset_id, override_columns=override_columns, **kwargs
        )

    def update_chart(self, chart_id: int, **kwargs: Any) -> dict[str, Any]:
        return self._client.update_chart(chart_id, **kwargs)

    def update_dashboard(
        self, dashboard_id: int, **kwargs: Any
    ) -> dict[str, Any]:
        return self._client.update_dashboard(dashboard_id, **kwargs)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_dashboards(self, ids: list[int]) -> bytes:
        """Export dashboards as a ZIP bundle (bytes)."""
        buf = self._client.export_zip("dashboard", ids)
        return buf.getvalue()

    def delete_resource(self, resource_type: str, resource_id: int) -> None:
        self._client.delete_resource(resource_type, resource_id)

    def export_resource_zip(self, resource_type: str, ids: list[int]) -> bytes:
        """Export any resource type as a ZIP bundle (bytes)."""
        buf = self._client.export_zip(resource_type, ids)
        return buf.getvalue()

    def import_resource_zip(self, resource_type: str, data: bytes, overwrite: bool = False) -> bool:
        from io import BytesIO
        return self._client.import_zip(resource_type, BytesIO(data), overwrite=overwrite)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> WorkspaceSnapshot:
        return take_snapshot(self)
