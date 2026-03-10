"""Tests for _helpers.py — shared field-extraction and type-coercion helpers."""

from __future__ import annotations

import pytest

from preset_py._helpers import (
    to_int,
    coerce_list,
    column_name,
    column_type_name,
    is_numeric_column,
    is_temporal_column,
    metric_column_name,
    metric_label,
    saved_metric_name,
    dataset_column_names,
    dataset_metric_names,
    dataset_columns,
)


# ---------------------------------------------------------------------------
# to_int
# ---------------------------------------------------------------------------


class TestToInt:
    def test_int_passthrough(self) -> None:
        assert to_int(42) == 42
        assert to_int(0) == 0

    def test_string_digits(self) -> None:
        assert to_int("42") == 42
        assert to_int(" 7 ") == 7

    def test_bool_rejected(self) -> None:
        assert to_int(True) is None
        assert to_int(False) is None

    def test_none(self) -> None:
        assert to_int(None) is None

    def test_non_digit_string(self) -> None:
        assert to_int("abc") is None
        assert to_int("12.5") is None
        assert to_int("") is None

    def test_float_rejected(self) -> None:
        assert to_int(3.14) is None


# ---------------------------------------------------------------------------
# coerce_list
# ---------------------------------------------------------------------------


class TestCoerceList:
    def test_none_returns_empty(self) -> None:
        assert coerce_list(None) == []

    def test_list_passthrough(self) -> None:
        assert coerce_list([1, 2]) == [1, 2]

    def test_tuple_converted(self) -> None:
        assert coerce_list((1, 2)) == [1, 2]

    def test_scalar_wrapped(self) -> None:
        assert coerce_list("hello") == ["hello"]
        assert coerce_list(42) == [42]


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


class TestColumnName:
    def test_column_name_key(self) -> None:
        assert column_name({"column_name": "CHAIN"}) == "CHAIN"

    def test_name_key(self) -> None:
        assert column_name({"name": "AMOUNT"}) == "AMOUNT"

    def test_column_name_takes_precedence(self) -> None:
        assert column_name({"column_name": "A", "name": "B"}) == "A"

    def test_empty_returns_none(self) -> None:
        assert column_name({}) is None
        assert column_name({"column_name": ""}) is None


class TestColumnTypeName:
    def test_type_key(self) -> None:
        assert column_type_name({"type": "varchar"}) == "VARCHAR"

    def test_type_generic(self) -> None:
        assert column_type_name({"type_generic": "numeric"}) == "NUMERIC"

    def test_empty(self) -> None:
        assert column_type_name({}) == ""

    def test_strips_whitespace(self) -> None:
        assert column_type_name({"type": " float "}) == "FLOAT"


class TestIsNumericColumn:
    @pytest.mark.parametrize("type_str", [
        "INTEGER", "BIGINT", "FLOAT", "DOUBLE PRECISION", "NUMERIC(10,2)",
        "DECIMAL", "SMALLINT", "TINYINT", "REAL", "NUMBER",
    ])
    def test_numeric_types(self, type_str: str) -> None:
        assert is_numeric_column({"type": type_str}) is True

    @pytest.mark.parametrize("type_str", [
        "VARCHAR", "TEXT", "BOOLEAN", "DATE", "TIMESTAMP",
    ])
    def test_non_numeric_types(self, type_str: str) -> None:
        assert is_numeric_column({"type": type_str}) is False


class TestIsTemporalColumn:
    @pytest.mark.parametrize("type_str", [
        "DATE", "TIMESTAMP", "DATETIME", "TIMESTAMP WITH TIME ZONE",
    ])
    def test_temporal_types(self, type_str: str) -> None:
        assert is_temporal_column({"type": type_str}) is True

    def test_is_dttm_flag(self) -> None:
        assert is_temporal_column({"is_dttm": True, "type": "VARCHAR"}) is True

    def test_non_temporal(self) -> None:
        assert is_temporal_column({"type": "VARCHAR"}) is False
        assert is_temporal_column({"type": "INTEGER"}) is False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestMetricColumnName:
    def test_string_column(self) -> None:
        assert metric_column_name({"column": "AMOUNT"}) == "AMOUNT"

    def test_dict_column(self) -> None:
        assert metric_column_name(
            {"column": {"column_name": "PRICE"}}
        ) == "PRICE"

    def test_dict_column_name_key(self) -> None:
        assert metric_column_name(
            {"column": {"name": "QTY"}}
        ) == "QTY"

    def test_missing_column(self) -> None:
        assert metric_column_name({}) is None

    def test_none_column(self) -> None:
        assert metric_column_name({"column": None}) is None


class TestMetricLabel:
    def test_string_metric(self) -> None:
        assert metric_label("count") == "count"

    def test_label_key(self) -> None:
        assert metric_label({"label": "Total Revenue"}) == "Total Revenue"

    def test_metric_name_key(self) -> None:
        assert metric_label({"metric_name": "revenue"}) == "revenue"

    def test_sql_expression(self) -> None:
        assert metric_label({
            "expressionType": "SQL",
            "sqlExpression": "SUM(amount)",
        }) == "SUM(amount)"

    def test_simple_aggregate(self) -> None:
        assert metric_label({
            "aggregate": "SUM",
            "column": {"column_name": "AMOUNT"},
        }) == "SUM(AMOUNT)"

    def test_none_returns_none(self) -> None:
        assert metric_label(None) is None

    def test_non_dict_returns_none(self) -> None:
        assert metric_label(42) is None


class TestSavedMetricName:
    def test_string(self) -> None:
        assert saved_metric_name("count") == "count"

    def test_dict_metric_name(self) -> None:
        assert saved_metric_name({"metric_name": "revenue"}) == "revenue"

    def test_dict_label(self) -> None:
        assert saved_metric_name({"label": "Total"}) == "Total"

    def test_empty_string(self) -> None:
        assert saved_metric_name("") is None

    def test_none(self) -> None:
        assert saved_metric_name(None) is None


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------


class TestDatasetColumnNames:
    def test_list_of_dicts(self) -> None:
        dataset = {
            "columns": [
                {"column_name": "A"},
                {"column_name": "B"},
            ]
        }
        assert dataset_column_names(dataset) == {"A", "B"}

    def test_dict_of_dicts(self) -> None:
        dataset = {
            "columns": {
                "0": {"column_name": "X"},
                "1": {"name": "Y"},
            }
        }
        assert dataset_column_names(dataset) == {"X", "Y"}

    def test_empty(self) -> None:
        assert dataset_column_names({}) == set()
        assert dataset_column_names({"columns": []}) == set()

    def test_mixed_types(self) -> None:
        dataset = {
            "columns": [
                {"column_name": "A"},
                "B",
                None,
                {"column_name": ""},
            ]
        }
        names = dataset_column_names(dataset)
        assert "A" in names
        assert "B" in names


class TestDatasetMetricNames:
    def test_dict_metrics(self) -> None:
        dataset = {
            "metrics": [
                {"metric_name": "count"},
                {"label": "revenue"},
            ]
        }
        assert dataset_metric_names(dataset) == {"count", "revenue"}

    def test_string_metrics(self) -> None:
        dataset = {"metrics": ["count", "total"]}
        assert dataset_metric_names(dataset) == {"count", "total"}

    def test_empty(self) -> None:
        assert dataset_metric_names({}) == set()

    def test_non_list_metrics(self) -> None:
        assert dataset_metric_names({"metrics": "not_a_list"}) == set()


class TestDatasetColumns:
    def test_returns_dicts_only(self) -> None:
        dataset = {
            "columns": [
                {"column_name": "A"},
                "not_a_dict",
                42,
                {"column_name": "B"},
            ]
        }
        result = dataset_columns(dataset)
        assert len(result) == 2
        assert all(isinstance(c, dict) for c in result)

    def test_empty(self) -> None:
        assert dataset_columns({}) == []
        assert dataset_columns({"columns": "not_a_list"}) == []
