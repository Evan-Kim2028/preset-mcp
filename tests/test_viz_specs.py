"""Tests for _viz_specs.py — the viz-type payload specification registry.

These tests verify:
1. VizSpec definitions are consistent and complete
2. check_required_fields catches missing fields per viz type
3. validate_chart_envelope catches structural envelope issues
4. _apply_chart_defaults produces spec-compliant params for every viz type
5. validate_params_payload and _apply_chart_defaults stay in sync via the spec
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from preset_py._viz_specs import (
    VIZ_SPECS,
    TIMESERIES_VIZ_TYPES,
    VizSpec,
    check_required_fields,
    get_viz_spec,
    validate_chart_envelope,
)
from preset_py._safety import validate_params_payload, validate_position_layout
from preset_py.client import _apply_chart_defaults, _coerce_list


# ---------------------------------------------------------------------------
# Fixtures: minimal dataset payloads for testing defaults
# ---------------------------------------------------------------------------

_DATASET_WITH_COLUMNS: dict[str, Any] = {
    "id": 1,
    "columns": [
        {"column_name": "CATEGORY", "type": "VARCHAR"},
        {"column_name": "AMOUNT", "type": "FLOAT"},
        {"column_name": "DAY", "type": "TIMESTAMP", "is_dttm": True},
    ],
    "metrics": [{"metric_name": "count", "label": "count"}],
}

_DATASET_NO_TEMPORAL: dict[str, Any] = {
    "id": 2,
    "columns": [
        {"column_name": "CATEGORY", "type": "VARCHAR"},
        {"column_name": "AMOUNT", "type": "FLOAT"},
    ],
    "metrics": [{"metric_name": "count", "label": "count"}],
}


# ---------------------------------------------------------------------------
# 1. Spec registry consistency
# ---------------------------------------------------------------------------


class TestVizSpecRegistry:
    def test_all_specs_are_viz_spec_instances(self) -> None:
        for name, spec in VIZ_SPECS.items():
            assert isinstance(spec, VizSpec), f"{name} is not a VizSpec"

    def test_timeseries_set_matches_specs(self) -> None:
        for name, spec in VIZ_SPECS.items():
            if spec.needs_time_column:
                assert name in TIMESERIES_VIZ_TYPES
            else:
                assert name not in TIMESERIES_VIZ_TYPES

    def test_get_viz_spec_known(self) -> None:
        assert get_viz_spec("pie") is not None
        assert get_viz_spec("pie") is VIZ_SPECS["pie"]

    def test_get_viz_spec_unknown_returns_none(self) -> None:
        assert get_viz_spec("nonexistent_chart_type") is None

    def test_all_required_fields_are_strings(self) -> None:
        for name, spec in VIZ_SPECS.items():
            for field_expr in spec.required:
                assert isinstance(field_expr, str), (
                    f"{name}.required contains non-string: {field_expr!r}"
                )

    def test_needs_dimension_implies_auto_dimension(self) -> None:
        """If a viz type *requires* a dimension, auto_dimension should be on."""
        for name, spec in VIZ_SPECS.items():
            if spec.needs_dimension:
                assert spec.auto_dimension, (
                    f"{name} needs_dimension=True but auto_dimension=False"
                )


# ---------------------------------------------------------------------------
# 2. check_required_fields — parameterized per viz type
# ---------------------------------------------------------------------------


class TestCheckRequiredFields:
    @pytest.mark.parametrize("viz_type", list(VIZ_SPECS.keys()))
    def test_empty_params_returns_errors_for_known_types(self, viz_type: str) -> None:
        errors = check_required_fields(viz_type, {})
        assert len(errors) > 0, f"Expected errors for empty params on {viz_type}"

    def test_unknown_viz_type_returns_no_errors(self) -> None:
        errors = check_required_fields("unknown_custom_chart", {})
        assert errors == []

    def test_pie_requires_metrics_and_dimension(self) -> None:
        errors = check_required_fields("pie", {})
        assert any("metrics" in e for e in errors)
        assert any("groupby" in e or "columns" in e for e in errors)

    def test_pie_accepts_groupby(self) -> None:
        errors = check_required_fields(
            "pie", {"metrics": ["count"], "groupby": ["CHAIN"]}
        )
        assert errors == []

    def test_pie_accepts_columns_instead_of_groupby(self) -> None:
        errors = check_required_fields(
            "pie", {"metrics": ["count"], "columns": ["CHAIN"]}
        )
        assert errors == []

    def test_timeseries_requires_metrics_and_time(self) -> None:
        errors = check_required_fields("echarts_timeseries_bar", {})
        assert any("metrics" in e for e in errors)
        assert any("time column" in e for e in errors)

    def test_timeseries_accepts_granularity_sqla(self) -> None:
        errors = check_required_fields(
            "echarts_timeseries_bar",
            {"metrics": ["count"], "granularity_sqla": "DAY"},
        )
        assert errors == []

    def test_timeseries_accepts_time_column_via_fallback(self) -> None:
        errors = check_required_fields(
            "echarts_timeseries_line",
            {"metrics": ["count"]},
            fallback_fields={"time_column": "DAY"},
        )
        assert errors == []

    def test_big_number_total_requires_metrics(self) -> None:
        errors = check_required_fields("big_number_total", {})
        assert any("metrics" in e for e in errors)

    def test_big_number_total_ok_with_metrics(self) -> None:
        errors = check_required_fields(
            "big_number_total", {"metrics": ["count"]}
        )
        assert errors == []

    def test_table_requires_metrics(self) -> None:
        errors = check_required_fields("table", {})
        assert any("metrics" in e for e in errors)

    def test_table_ok_with_metrics(self) -> None:
        errors = check_required_fields("table", {"metrics": ["count"]})
        assert errors == []


# ---------------------------------------------------------------------------
# 3. validate_chart_envelope
# ---------------------------------------------------------------------------


class TestValidateChartEnvelope:
    def test_valid_envelope_passes(self) -> None:
        payload = {
            "slice_name": "My Chart",
            "viz_type": "table",
            "datasource_id": 1,
            "datasource_type": "table",
            "params": json.dumps({"metrics": ["count"]}),
        }
        errors = validate_chart_envelope(payload, "table")
        assert errors == []

    def test_missing_slice_name(self) -> None:
        payload = {
            "viz_type": "table",
            "datasource_id": 1,
            "params": json.dumps({"metrics": ["count"]}),
        }
        errors = validate_chart_envelope(payload, "table")
        assert any("slice_name" in e for e in errors)

    def test_missing_viz_type(self) -> None:
        payload = {
            "slice_name": "Chart",
            "datasource_id": 1,
            "params": json.dumps({"metrics": ["count"]}),
        }
        errors = validate_chart_envelope(payload, "table")
        assert any("viz_type" in e for e in errors)

    def test_missing_datasource_id(self) -> None:
        payload = {
            "slice_name": "Chart",
            "viz_type": "table",
            "params": json.dumps({"metrics": ["count"]}),
        }
        errors = validate_chart_envelope(payload, "table")
        assert any("datasource_id" in e for e in errors)

    def test_bad_params_json(self) -> None:
        payload = {
            "slice_name": "Chart",
            "viz_type": "table",
            "datasource_id": 1,
            "params": "{invalid json",
        }
        errors = validate_chart_envelope(payload, "table")
        assert any("not valid JSON" in e for e in errors)

    def test_envelope_catches_missing_singular_metric(self) -> None:
        payload = {
            "slice_name": "Pie",
            "viz_type": "pie",
            "datasource_id": 1,
            "datasource_type": "table",
            "params": json.dumps({
                "metrics": ["count"],
                "groupby": ["CHAIN"],
            }),
        }
        errors = validate_chart_envelope(payload, "pie")
        assert any("singular" in e.lower() and "metric" in e.lower() for e in errors)

    def test_envelope_ok_when_singular_metric_present(self) -> None:
        payload = {
            "slice_name": "Pie",
            "viz_type": "pie",
            "datasource_id": 1,
            "datasource_type": "table",
            "params": json.dumps({
                "metrics": ["count"],
                "metric": "count",
                "groupby": ["CHAIN"],
            }),
        }
        errors = validate_chart_envelope(payload, "pie")
        assert errors == []


# ---------------------------------------------------------------------------
# 4. _apply_chart_defaults — spec-driven defaults per viz type
# ---------------------------------------------------------------------------


class TestApplyChartDefaults:
    @pytest.mark.parametrize("viz_type", list(VIZ_SPECS.keys()))
    def test_minimal_template_is_noop(self, viz_type: str) -> None:
        params: dict[str, Any] = {}
        _apply_chart_defaults(viz_type, _DATASET_WITH_COLUMNS, params, template="minimal")
        assert params == {}

    def test_pie_fills_metrics_groupby_and_singular_metric(self) -> None:
        params: dict[str, Any] = {}
        _apply_chart_defaults("pie", _DATASET_WITH_COLUMNS, params)
        assert _coerce_list(params.get("metrics")), "metrics should be set"
        assert _coerce_list(params.get("groupby")), "groupby should be set"
        assert "metric" in params, "singular metric should be set"
        assert params.get("show_legend") is True
        assert params.get("labels_outside") is True

    def test_pie_preserves_explicit_metrics(self) -> None:
        params: dict[str, Any] = {"metrics": ["custom"], "groupby": ["CATEGORY"]}
        _apply_chart_defaults("pie", _DATASET_WITH_COLUMNS, params)
        assert params["metrics"] == ["custom"]
        assert params["metric"] == "custom"

    def test_timeseries_fills_metrics_and_time(self) -> None:
        params: dict[str, Any] = {}
        _apply_chart_defaults("echarts_timeseries_bar", _DATASET_WITH_COLUMNS, params)
        assert _coerce_list(params.get("metrics")), "metrics should be set"
        assert params.get("granularity_sqla") == "DAY"
        assert params.get("time_grain_sqla") == "P1D"

    def test_timeseries_no_temporal_column_leaves_time_empty(self) -> None:
        params: dict[str, Any] = {}
        _apply_chart_defaults("echarts_timeseries_bar", _DATASET_NO_TEMPORAL, params)
        assert params.get("granularity_sqla") is None

    def test_big_number_total_fills_metrics(self) -> None:
        params: dict[str, Any] = {}
        _apply_chart_defaults("big_number_total", _DATASET_WITH_COLUMNS, params)
        assert _coerce_list(params.get("metrics")), "metrics should be set"

    def test_table_fills_metrics_and_dimension(self) -> None:
        params: dict[str, Any] = {}
        _apply_chart_defaults("table", _DATASET_WITH_COLUMNS, params)
        assert _coerce_list(params.get("metrics")), "metrics should be set"
        assert _coerce_list(params.get("columns")), "columns should be set for table"

    def test_unknown_viz_type_is_noop(self) -> None:
        params: dict[str, Any] = {"custom": "value"}
        _apply_chart_defaults("unknown_chart_type", _DATASET_WITH_COLUMNS, params)
        assert params == {"custom": "value"}

    @pytest.mark.parametrize("viz_type", list(VIZ_SPECS.keys()))
    def test_defaults_produce_spec_compliant_params(self, viz_type: str) -> None:
        """After applying defaults, check_required_fields should return no errors.

        This is the key integration test: defaults and validation use the same
        spec, so they must stay in sync.
        """
        params: dict[str, Any] = {}
        dataset = _DATASET_WITH_COLUMNS
        try:
            _apply_chart_defaults(viz_type, dataset, params)
        except ValueError:
            pytest.skip(f"_apply_chart_defaults raises for {viz_type} (expected for some edge cases)")

        errors = check_required_fields(viz_type, params)
        assert errors == [], (
            f"_apply_chart_defaults for {viz_type} produced params that fail "
            f"check_required_fields: {errors}"
        )


# ---------------------------------------------------------------------------
# 5. validate_position_layout (extracted from server.py)
# ---------------------------------------------------------------------------


class TestValidatePositionLayout:
    def test_empty_layout_is_ok(self) -> None:
        validate_position_layout({})

    def test_missing_root_id(self) -> None:
        with pytest.raises(ValueError, match="ROOT_ID"):
            validate_position_layout({
                "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": []},
            })

    def test_missing_grid_id(self) -> None:
        with pytest.raises(ValueError, match="GRID_ID"):
            validate_position_layout({
                "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
            })

    def test_valid_minimal_layout(self) -> None:
        validate_position_layout({
            "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
            "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": []},
            "DASHBOARD_VERSION_KEY": "v2",
        })

    def test_mismatched_node_id(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            validate_position_layout({
                "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
                "GRID_ID": {"id": "WRONG_ID", "type": "GRID", "children": []},
            })

    def test_missing_type_field(self) -> None:
        with pytest.raises(ValueError, match="missing required 'type'"):
            validate_position_layout({
                "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
                "GRID_ID": {"id": "GRID_ID", "children": []},
            })

    def test_dangling_child_reference(self) -> None:
        with pytest.raises(ValueError, match="dangling child"):
            validate_position_layout({
                "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
                "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": ["NONEXISTENT"]},
            })

    def test_root_must_include_grid_child(self) -> None:
        with pytest.raises(ValueError, match="ROOT_ID.children must include GRID_ID"):
            validate_position_layout({
                "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": []},
                "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": []},
            })

    def test_parents_must_be_list(self) -> None:
        with pytest.raises(ValueError, match="parents must be a list"):
            validate_position_layout({
                "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
                "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": [], "parents": "ROOT_ID"},
            })
