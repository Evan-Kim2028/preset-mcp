"""Dataset and chart creation helpers wrapping SupersetClient."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from preset_cli.api.clients.superset import SupersetClient


def create_virtual_dataset(
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
        except Exception:
            pass  # non-fatal; columns will refresh on first use

    return result


def create_chart(
    client: SupersetClient,
    dataset_id: int,
    title: str,
    viz_type: str,
    *,
    metrics: list[str] | None = None,
    groupby: list[str] | None = None,
    time_column: str | None = None,
    dashboards: list[int] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a chart with sensible defaults.

    ``extra_params`` is merged into the chart params dict, allowing callers
    to pass any viz-type-specific options.
    """
    params: dict[str, Any] = {
        "viz_type": viz_type,
    }
    if metrics:
        params["metrics"] = metrics
    if groupby:
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
