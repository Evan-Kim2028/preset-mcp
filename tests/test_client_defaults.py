"""Tests for client.py default-inference helpers.

Covers metric normalization, dimension inference, temporal detection,
and default metric/dimension selection — all pure-logic functions that
run without an API connection.
"""

from __future__ import annotations

from typing import Any

import pytest

from preset_py.client import (
    _column_map,
    _simple_metric_from_column,
    _normalize_create_metrics,
    _default_metric,
    _default_dimension_column,
    _default_time_column,
    _validate_groupby_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DATASET_FULL: dict[str, Any] = {
    "id": 1,
    "columns": [
        {"column_name": "CATEGORY", "type": "VARCHAR"},
        {"column_name": "AMOUNT", "type": "FLOAT"},
        {"column_name": "QUANTITY", "type": "INTEGER"},
        {"column_name": "DAY", "type": "TIMESTAMP", "is_dttm": True},
        {"column_name": "NOTES", "type": "TEXT"},
    ],
    "metrics": [
        {"metric_name": "count", "label": "count"},
        {"metric_name": "total_amount", "label": "Total Amount"},
    ],
}

_DATASET_NO_METRICS: dict[str, Any] = {
    "id": 2,
    "columns": [
        {"column_name": "NAME", "type": "VARCHAR"},
        {"column_name": "PRICE", "type": "NUMERIC"},
    ],
    "metrics": [],
}

_DATASET_NO_TEMPORAL: dict[str, Any] = {
    "id": 3,
    "columns": [
        {"column_name": "REGION", "type": "VARCHAR"},
        {"column_name": "SCORE", "type": "DOUBLE PRECISION"},
    ],
    "metrics": [{"metric_name": "count"}],
}

_DATASET_ONLY_NUMERIC: dict[str, Any] = {
    "id": 4,
    "columns": [
        {"column_name": "VALUE1", "type": "INTEGER"},
        {"column_name": "VALUE2", "type": "BIGINT"},
    ],
    "metrics": [],
}

_DATASET_ONLY_TEMPORAL: dict[str, Any] = {
    "id": 5,
    "columns": [
        {"column_name": "CREATED_AT", "type": "TIMESTAMP", "is_dttm": True},
        {"column_name": "UPDATED_AT", "type": "DATE"},
    ],
    "metrics": [],
}

_DATASET_EMPTY: dict[str, Any] = {
    "id": 6,
    "columns": [],
    "metrics": [],
}


# ---------------------------------------------------------------------------
# _column_map
# ---------------------------------------------------------------------------


class TestColumnMap:
    def test_builds_map(self) -> None:
        result = _column_map(_DATASET_FULL)
        assert "CATEGORY" in result
        assert "AMOUNT" in result
        assert result["AMOUNT"]["type"] == "FLOAT"

    def test_empty_dataset(self) -> None:
        assert _column_map(_DATASET_EMPTY) == {}

    def test_non_list_columns(self) -> None:
        assert _column_map({"columns": "invalid"}) == {}


# ---------------------------------------------------------------------------
# _simple_metric_from_column
# ---------------------------------------------------------------------------


class TestSimpleMetricFromColumn:
    def test_numeric_column_uses_sum(self) -> None:
        result = _simple_metric_from_column(
            "AMOUNT", {"column_name": "AMOUNT", "type": "FLOAT"}
        )
        assert result["expressionType"] == "SIMPLE"
        assert result["aggregate"] == "SUM"
        assert result["column"]["column_name"] == "AMOUNT"
        assert result["label"] == "SUM(AMOUNT)"

    def test_non_numeric_column_uses_count(self) -> None:
        result = _simple_metric_from_column(
            "CATEGORY", {"column_name": "CATEGORY", "type": "VARCHAR"}
        )
        assert result["aggregate"] == "COUNT"
        assert result["label"] == "COUNT(CATEGORY)"


# ---------------------------------------------------------------------------
# _normalize_create_metrics
# ---------------------------------------------------------------------------


class TestNormalizeCreateMetrics:
    def test_saved_metric_name_preserved(self) -> None:
        result = _normalize_create_metrics(["count"], _DATASET_FULL)
        assert result == ["count"]

    def test_column_name_converted_to_simple_metric(self) -> None:
        result = _normalize_create_metrics(["AMOUNT"], _DATASET_FULL)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["expressionType"] == "SIMPLE"
        assert result[0]["aggregate"] == "SUM"

    def test_dict_metric_passed_through(self) -> None:
        adhoc = {"expressionType": "SQL", "sqlExpression": "AVG(AMOUNT)"}
        result = _normalize_create_metrics([adhoc], _DATASET_FULL)
        assert result == [adhoc]

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            _normalize_create_metrics(["NONEXISTENT"], _DATASET_FULL)

    def test_mixed_metrics(self) -> None:
        result = _normalize_create_metrics(
            ["count", "AMOUNT", {"expressionType": "SQL", "sqlExpression": "1"}],
            _DATASET_FULL,
        )
        assert result[0] == "count"
        assert isinstance(result[1], dict)
        assert result[1]["aggregate"] == "SUM"
        assert result[2]["expressionType"] == "SQL"


# ---------------------------------------------------------------------------
# _default_metric
# ---------------------------------------------------------------------------


class TestDefaultMetric:
    def test_prefers_saved_metric(self) -> None:
        result = _default_metric(_DATASET_FULL)
        assert result == "count"

    def test_falls_back_to_numeric_column(self) -> None:
        result = _default_metric(_DATASET_NO_METRICS)
        assert isinstance(result, dict)
        assert result["expressionType"] == "SIMPLE"
        assert result["aggregate"] == "SUM"

    def test_falls_back_to_count_star(self) -> None:
        result = _default_metric(_DATASET_EMPTY)
        assert isinstance(result, dict)
        assert result["expressionType"] == "SQL"
        assert result["sqlExpression"] == "COUNT(*)"


# ---------------------------------------------------------------------------
# _default_dimension_column
# ---------------------------------------------------------------------------


class TestDefaultDimensionColumn:
    def test_prefers_non_temporal_non_numeric(self) -> None:
        result = _default_dimension_column(_DATASET_FULL)
        assert result == "CATEGORY"

    def test_skips_temporal_columns(self) -> None:
        result = _default_dimension_column(_DATASET_FULL)
        assert result != "DAY"

    def test_falls_back_to_non_temporal(self) -> None:
        result = _default_dimension_column(_DATASET_ONLY_NUMERIC)
        assert result in ("VALUE1", "VALUE2")

    def test_falls_back_to_any_column(self) -> None:
        result = _default_dimension_column(_DATASET_ONLY_TEMPORAL)
        assert result in ("CREATED_AT", "UPDATED_AT")

    def test_empty_dataset_returns_none(self) -> None:
        assert _default_dimension_column(_DATASET_EMPTY) is None


# ---------------------------------------------------------------------------
# _default_time_column
# ---------------------------------------------------------------------------


class TestDefaultTimeColumn:
    def test_finds_temporal_column(self) -> None:
        result = _default_time_column(_DATASET_FULL)
        assert result == "DAY"

    def test_no_temporal_returns_none(self) -> None:
        assert _default_time_column(_DATASET_NO_TEMPORAL) is None

    def test_empty_dataset_returns_none(self) -> None:
        assert _default_time_column(_DATASET_EMPTY) is None

    def test_picks_first_temporal(self) -> None:
        result = _default_time_column(_DATASET_ONLY_TEMPORAL)
        assert result == "CREATED_AT"


# ---------------------------------------------------------------------------
# _validate_groupby_columns
# ---------------------------------------------------------------------------


class TestValidateGroupbyColumns:
    def test_valid_columns_pass(self) -> None:
        _validate_groupby_columns(["CATEGORY", "NOTES"], _DATASET_FULL)

    def test_unknown_column_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown column"):
            _validate_groupby_columns(["NOT_A_COLUMN"], _DATASET_FULL)

    def test_empty_groupby_passes(self) -> None:
        _validate_groupby_columns([], _DATASET_FULL)
