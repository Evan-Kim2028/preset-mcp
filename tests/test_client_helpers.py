import json

import pytest

from preset_py.client import _create_chart, _create_virtual_dataset


class _DatasetClient:
    def __init__(self) -> None:
        self.created_dataset_payload: dict | None = None
        self.created_chart_payload: dict | None = None

    def create_dataset(self, **payload):
        self.created_dataset_payload = payload
        return {"id": 123, "table_name": payload["table_name"]}

    def get_refreshed_dataset_columns(self, dataset_id: int):
        raise RuntimeError(f"refresh failed for {dataset_id}")

    def get_dataset(self, dataset_id: int):
        return {
            "id": dataset_id,
            "columns": [
                {"column_name": "VOLUME_M", "type": "DOUBLE PRECISION"},
                {"column_name": "BRIDGE", "type": "VARCHAR"},
            ],
            "metrics": [],
        }

    def create_resource(self, resource_type: str, **payload):
        assert resource_type == "chart"
        self.created_chart_payload = payload
        return {"id": 77, **payload}


def test_create_virtual_dataset_surfaces_column_refresh_warning() -> None:
    client = _DatasetClient()
    result = _create_virtual_dataset(
        client,
        name="daily_flows",
        sql="SELECT 1",
        database_id=9,
    )
    assert result["id"] == 123
    assert "_warning" in result
    assert "column refresh failed" in result["_warning"]


def test_create_chart_normalizes_metric_and_sets_defaults() -> None:
    client = _DatasetClient()
    result = _create_chart(
        client,
        dataset_id=10,
        title="Bridge Volume",
        viz_type="echarts_timeseries_bar",
        metrics=["VOLUME_M"],
        groupby=["BRIDGE"],
    )
    assert result["id"] == 77
    assert client.created_chart_payload is not None

    params = json.loads(client.created_chart_payload["params"])
    assert params["datasource"] == "10__table"
    assert params["row_limit"] == 10000
    assert params["color_scheme"] == "supersetColors"
    assert isinstance(params["metrics"], list)
    assert isinstance(params["metrics"][0], dict)
    assert params["metrics"][0]["expressionType"] == "SIMPLE"
    assert params["metrics"][0]["aggregate"] == "SUM"


def test_create_chart_rejects_unknown_groupby_columns() -> None:
    client = _DatasetClient()
    with pytest.raises(ValueError, match="unknown column"):
        _create_chart(
            client,
            dataset_id=10,
            title="Bad Groupby",
            viz_type="table",
            metrics=["VOLUME_M"],
            groupby=["NOT_A_REAL_COLUMN"],
        )
