import pytest

from preset_py._safety import validate_params_payload


def test_params_payload_blocks_viz_type_override() -> None:
    with pytest.raises(ValueError, match="viz_type"):
        validate_params_payload('{"viz_type":"table"}')


def test_params_payload_rejects_invalid_metric_shape() -> None:
    with pytest.raises(ValueError, match="aggregate"):
        validate_params_payload(
            '{"metrics":[{"expressionType":"SIMPLE","column":{"column_name":"value"}}]}'
        )


def test_params_payload_rejects_unknown_columns() -> None:
    with pytest.raises(ValueError, match="unknown dataset columns"):
        validate_params_payload(
            '{"groupby":["known_col"],"filters":[{"col":"missing_col"}]}',
            dataset_columns={"known_col"},
            dataset_metrics={"saved_metric"},
        )


def test_params_payload_rejects_unknown_axis_column() -> None:
    with pytest.raises(ValueError, match="unknown dataset columns"):
        validate_params_payload(
            '{"x_axis":"BAD_COLUMN","groupby":["known_col"]}',
            dataset_columns={"known_col"},
            dataset_metrics={"saved_metric"},
        )


def test_params_payload_rejects_duplicate_dimension_labels() -> None:
    with pytest.raises(ValueError, match="duplicate dimension labels"):
        validate_params_payload(
            '{"x_axis":"CHAIN","groupby":["CHAIN"]}',
            dataset_columns={"CHAIN"},
            dataset_metrics={"count"},
        )


def test_params_payload_rejects_metric_dimension_label_collision() -> None:
    with pytest.raises(ValueError, match="metric labels collide"):
        validate_params_payload(
            '{"groupby":["CHAIN"],"metrics":["CHAIN"]}',
            dataset_columns={"CHAIN"},
            dataset_metrics={"count"},
        )


def test_params_payload_enforces_pie_required_fields() -> None:
    with pytest.raises(ValueError, match="pie.*requires.*metrics"):
        validate_params_payload(
            '{"groupby":["CHAIN"]}',
            dataset_columns={"CHAIN"},
            dataset_metrics={"count"},
            viz_type="pie",
        )

    with pytest.raises(ValueError, match="pie.*requires.*groupby.*columns"):
        validate_params_payload(
            '{"metrics":["count"]}',
            dataset_columns={"CHAIN"},
            dataset_metrics={"count"},
            viz_type="pie",
        )


def test_params_payload_pie_auto_sets_singular_metric() -> None:
    """Pie chart params_json should auto-derive metric from metrics."""
    parsed, warnings = validate_params_payload(
        '{"metrics":["count"],"groupby":["CHAIN"]}',
        dataset_columns={"CHAIN"},
        dataset_metrics={"count"},
        viz_type="pie",
    )
    assert parsed["metric"] == "count"
    assert any("Auto-set" in w and "pie" in w for w in warnings)


def test_params_payload_pie_preserves_explicit_metric() -> None:
    """If metric is already set, don't override it."""
    parsed, warnings = validate_params_payload(
        '{"metrics":["count"],"metric":"explicit_metric","groupby":["CHAIN"]}',
        dataset_columns={"CHAIN"},
        dataset_metrics={"count", "explicit_metric"},
        viz_type="pie",
    )
    assert parsed["metric"] == "explicit_metric"
    assert not any("Auto-set" in w for w in warnings)


def test_params_payload_enforces_timeseries_required_fields() -> None:
    with pytest.raises(ValueError, match="echarts_timeseries_line.*requires.*metrics"):
        validate_params_payload(
            '{"granularity_sqla":"DAY"}',
            dataset_columns={"DAY"},
            dataset_metrics={"count"},
            viz_type="echarts_timeseries_line",
        )

    with pytest.raises(ValueError, match="echarts_timeseries_line.*requires a time column"):
        validate_params_payload(
            '{"metrics":["count"]}',
            dataset_columns={"DAY"},
            dataset_metrics={"count"},
            viz_type="echarts_timeseries_line",
        )
