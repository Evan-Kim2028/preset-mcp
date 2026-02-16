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


def test_params_payload_warns_on_unknown_columns() -> None:
    _, warnings = validate_params_payload(
        '{"groupby":["known_col"],"filters":[{"col":"missing_col"}]}',
        dataset_columns={"known_col"},
        dataset_metrics={"saved_metric"},
    )
    assert warnings
    assert "unknown dataset columns" in warnings[0]
