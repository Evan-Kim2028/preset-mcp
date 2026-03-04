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
