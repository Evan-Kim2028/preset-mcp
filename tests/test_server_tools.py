import json

import pytest
from fastmcp.exceptions import ToolError

import preset_py.server as server


class _WorkspaceBase:
    def charts(self):
        return [{"id": 1, "viz_type": "table"}]


def _validation_payload(exc: ToolError) -> dict:
    return json.loads(str(exc))


def test_update_dataset_blocks_destructive_sql() -> None:
    with pytest.raises(ToolError) as exc:
        server.update_dataset.fn(dataset_id=1, sql="DROP TABLE important_data")
    payload = _validation_payload(exc.value)
    assert payload["error_type"] == "validation"
    assert "Write operation" in payload["error"]


def test_create_dataset_checks_database_precondition(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def database_detail(self, database_id: int):
            raise RuntimeError("not found")

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    with pytest.raises(ToolError) as exc:
        server.create_dataset.fn(
            name="my_dataset",
            sql="SELECT 1",
            database_id=999,
            dry_run=True,
        )
    payload = _validation_payload(exc.value)
    assert "Database 999 not found" in payload["error"]


def test_create_chart_accepts_json_array_strings_and_validates(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def __init__(self) -> None:
            self.create_kwargs = None

        def dataset_detail(self, dataset_id: int):
            return {"id": dataset_id}

        def create_chart(self, dataset_id: int, title: str, viz_type: str, **kwargs):
            self.create_kwargs = kwargs
            return {"id": 77, "slice_name": title, "viz_type": viz_type}

        def validate_chart_data(self, chart_id: int, dashboard_id=None, row_limit=10000, force=False):
            return {
                "chart_id": chart_id,
                "dashboard_id": dashboard_id,
                "status": "success",
                "error": None,
            }

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.create_chart.fn(
        dataset_id=10,
        title="My Chart",
        viz_type="table",
        metrics='["VOLUME_M"]',
        groupby='["BRIDGE"]',
        validate_after_create=True,
        repair_dashboard_refs=False,
        dry_run=False,
    )
    payload = json.loads(raw)
    assert ws.create_kwargs is not None
    assert ws.create_kwargs["metrics"] == ["VOLUME_M"]
    assert ws.create_kwargs["groupby"] == ["BRIDGE"]
    assert payload["_validation"]["status"] == "success"


def test_create_chart_rejects_invalid_dashboard_id(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dataset_detail(self, dataset_id: int):
            return {"id": dataset_id}

        def dashboard_detail(self, dashboard_id: int):
            raise RuntimeError("not found")

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    with pytest.raises(ToolError) as exc:
        server.create_chart.fn(
            dataset_id=10,
            title="My Chart",
            viz_type="table",
            dashboards="[123]",
            dry_run=True,
        )
    payload = _validation_payload(exc.value)
    assert "Dashboard 123 not found" in payload["error"]


def test_get_chart_enriches_datasource_fields(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def chart_detail(self, chart_id: int):
            return {
                "id": chart_id,
                "slice_name": "Flows",
                "viz_type": "sankey_v2",
                "params": '{"datasource":"745__table"}',
            }

        def dataset_detail(self, dataset_id: int):
            return {"id": dataset_id, "table_name": "stablecoin_flows"}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.get_chart.fn(chart_id=1192, response_mode="standard")
    payload = json.loads(raw)
    data = payload["data"]
    assert data["datasource_id"] == 745
    assert data["datasource_type"] == "table"
    assert data["datasource_name_text"] == "stablecoin_flows"


def test_repair_dashboard_refs_replaces_orphans_by_name() -> None:
    position, metadata, summary = server._repair_dashboard_refs(
        position_json={
            "CHART-1": {"meta": {"chartId": 1, "sliceName": "Revenue"}, "children": []},
            "ROOT_ID": {"children": ["CHART-1"]},
        },
        json_metadata={"chartsInScope": {"1": {"scope": ["root"]}}},
        dashboard_charts=[
            {"id": 9, "slice_name": "Revenue"},
        ],
        strategy="replace_by_name",
    )
    chart_ref = position["CHART-1"]["meta"]["chartId"]
    assert chart_ref == 9
    assert metadata["chartsInScope"] == {"9": {"scope": ["root"]}}
    assert summary["replacements"] == {1: 9}
    assert summary["orphaned_after"] == []


def test_coerce_list_arg_handles_json_arrays() -> None:
    assert server._coerce_list_arg('["a","b"]', field_name="metrics", item_kind="str") == ["a", "b"]
    assert server._coerce_list_arg("[1,2,3]", field_name="dashboards", item_kind="int") == [1, 2, 3]
