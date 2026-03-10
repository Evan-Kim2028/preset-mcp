"""Shared field-extraction and type-coercion helpers.

These small utilities are used by ``client.py``, ``server.py``, and
``_safety.py``.  Centralising them here eliminates subtle drift between
near-identical copies that previously lived in each module.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def to_int(value: Any) -> int | None:
    """Convert *value* to an ``int`` if possible, else ``None``.

    Booleans are explicitly rejected so that ``True``/``False`` are never
    silently converted to ``1``/``0``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def coerce_list(value: Any) -> list[Any]:
    """Normalize scalar/list-like values to a list (empty list on ``None``)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


# ---------------------------------------------------------------------------
# Column / metric name extraction
# ---------------------------------------------------------------------------


def column_name(column: dict[str, Any]) -> str | None:
    """Extract a normalised column name from a dataset column dict."""
    for key in ("column_name", "name"):
        value = column.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def column_type_name(column: dict[str, Any]) -> str:
    """Return a normalised upper-case type string from column metadata."""
    for key in ("type", "type_generic", "python_date_format"):
        raw = column.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().upper()
    return ""


def is_numeric_column(column: dict[str, Any]) -> bool:
    """Best-effort numeric detection from Superset column metadata."""
    type_name = column_type_name(column)
    _NUMERIC_TOKENS = (
        "INT", "NUMBER", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE",
        "REAL", "BIGINT", "SMALLINT", "TINYINT",
    )
    return any(token in type_name for token in _NUMERIC_TOKENS)


def is_temporal_column(column: dict[str, Any]) -> bool:
    """Best-effort temporal detection from Superset column metadata."""
    if column.get("is_dttm") is True:
        return True
    type_name = column_type_name(column)
    _TEMPORAL_TOKENS = ("DATE", "TIME", "TIMESTAMP", "DATETIME")
    return any(token in type_name for token in _TEMPORAL_TOKENS)


def metric_column_name(metric: dict[str, Any]) -> str | None:
    """Extract a metric's referenced column name, if present."""
    col = metric.get("column")
    if isinstance(col, str):
        return col
    if isinstance(col, dict):
        name = col.get("column_name") or col.get("name")
        if isinstance(name, str):
            return name
    return None


def metric_label(metric: Any) -> str | None:
    """Derive the effective metric label Superset uses for display/orderby."""
    if isinstance(metric, str):
        return metric
    if not isinstance(metric, dict):
        return None

    for key in ("label", "label_short", "metric_name", "name"):
        value = metric.get(key)
        if isinstance(value, str) and value:
            return value

    if metric.get("expressionType") == "SQL":
        sql_expr = metric.get("sqlExpression")
        if isinstance(sql_expr, str) and sql_expr.strip():
            return sql_expr.strip()

    col_name = metric_column_name(metric)
    aggregate = metric.get("aggregate")
    if isinstance(aggregate, str) and aggregate and col_name:
        return f"{aggregate}({col_name})"
    return col_name


def saved_metric_name(metric: Any) -> str | None:
    """Extract metric name from dataset metric payload shapes."""
    if isinstance(metric, str) and metric:
        return metric
    if not isinstance(metric, dict):
        return None
    for key in ("metric_name", "label", "name"):
        value = metric.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------


def dataset_column_names(dataset: dict[str, Any]) -> set[str]:
    """Extract column names from a dataset payload."""
    raw_columns = dataset.get("columns")
    if not isinstance(raw_columns, list):
        if isinstance(raw_columns, dict):
            raw_columns = list(raw_columns.values())
        else:
            return set()
    names: set[str] = set()
    for col in raw_columns:
        if isinstance(col, dict):
            name = column_name(col)
            if name:
                names.add(name)
        elif isinstance(col, str) and col:
            names.add(col)
    return names


def dataset_metric_names(dataset: dict[str, Any]) -> set[str]:
    """Extract metric names from a dataset payload."""
    metrics = dataset.get("metrics", [])
    if not isinstance(metrics, list):
        return set()
    names: set[str] = set()
    for m in metrics:
        name = saved_metric_name(m)
        if name:
            names.add(name)
    return names


def dataset_columns(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Return dataset columns as a normalised list of dicts."""
    columns = dataset.get("columns", [])
    if not isinstance(columns, list):
        return []
    return [c for c in columns if isinstance(c, dict)]
