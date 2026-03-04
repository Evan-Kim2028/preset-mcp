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


def test_create_chart_accepts_adhoc_metric_and_params_json(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def __init__(self) -> None:
            self.create_kwargs = None

        def dataset_detail(self, dataset_id: int):
            return {
                "id": dataset_id,
                "columns": [
                    {"column_name": "CATEGORY"},
                    {"column_name": "SOURCE_NAME"},
                    {"column_name": "TOKEN_SYMBOL"},
                    {"column_name": "AMOUNT_USD"},
                ],
                "metrics": [{"metric_name": "count"}],
            }

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
        title="USDSUI Distribution",
        viz_type="pie",
        metrics='[{"expressionType":"SQL","sqlExpression":"AVG(AMOUNT_USD)","label":"AVG(AMOUNT_USD)"}]',
        groupby='["CATEGORY","SOURCE_NAME"]',
        params_json='{"adhoc_filters":[{"col":"TOKEN_SYMBOL","op":"==","val":"USDSUI"}]}',
        validate_after_create=False,
        repair_dashboard_refs=False,
    )
    payload = json.loads(raw)
    assert payload["id"] == 77
    assert ws.create_kwargs is not None
    assert ws.create_kwargs["metrics"][0]["expressionType"] == "SQL"
    assert ws.create_kwargs["adhoc_filters"][0]["col"] == "TOKEN_SYMBOL"
    assert ws.create_kwargs["template"] == "auto"


def test_create_chart_rejects_invalid_template(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dataset_detail(self, dataset_id: int):
            return {"id": dataset_id, "columns": [], "metrics": []}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    with pytest.raises(ToolError) as exc:
        server.create_chart.fn(
            dataset_id=10,
            title="Bad Template",
            viz_type="table",
            template="foobar",
            dry_run=True,
        )
    payload = _validation_payload(exc.value)
    assert "template must be one of" in payload["error"]


def test_create_chart_rejects_duplicate_metrics_inputs(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dataset_detail(self, dataset_id: int):
            return {
                "id": dataset_id,
                "columns": [{"column_name": "CHAIN"}],
                "metrics": [{"metric_name": "count"}],
            }

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError) as exc:
        server.create_chart.fn(
            dataset_id=10,
            title="Bad Inputs",
            viz_type="table",
            metrics='["count"]',
            params_json='{"metrics":["count"]}',
            dry_run=True,
        )
    payload = _validation_payload(exc.value)
    assert "either the metrics argument or params_json.metrics" in payload["error"]


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


def test_validate_chart_render_tool_passes_through(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def validate_chart_render(self, chart_id: int, **kwargs):
            return {
                "chart_id": chart_id,
                "slice_name": "Render Check",
                "status": "success",
                "error": None,
                "screenshot_path": None,
                "kwargs": kwargs,
            }

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.validate_chart_render.fn(
        chart_id=10,
        timeout_ms=12345,
        settle_ms=777,
        response_mode="standard",
    )
    payload = json.loads(raw)
    assert payload["chart_id"] == 10
    assert payload["status"] == "success"


def test_validate_dashboard_render_tool_compact(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def validate_dashboard_render(self, dashboard_id: int, **kwargs):
            return {
                "dashboard_id": dashboard_id,
                "chart_count": 2,
                "validated": 2,
                "broken_count": 1,
                "broken_charts": [
                    {"chart_id": 99, "status": "failed"},
                ],
                "kwargs": kwargs,
            }

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.validate_dashboard_render.fn(
        dashboard_id=77,
        response_mode="compact",
    )
    payload = json.loads(raw)
    assert payload["dashboard_id"] == 77
    assert payload["broken_count"] == 1


def test_update_dashboard_blocks_empty_layout_wipe(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "position_json": {
                    "ROOT_ID": {"children": ["GRID_ID"]},
                    "GRID_ID": {"children": ["CHART-1"]},
                    "CHART-1": {"meta": {"chartId": 101}},
                },
            }

        def dashboard_charts(self, dashboard_id: int):
            return [{"id": 101}]

        def update_dashboard(self, dashboard_id: int, **kwargs):
            raise AssertionError("update_dashboard should have been blocked")

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError) as exc:
        server.update_dashboard.fn(
            dashboard_id=80,
            position_json='{"DASHBOARD_VERSION_KEY":"v2","ROOT_ID":{"children":["GRID_ID"]},"GRID_ID":{"children":[]}}',
        )
    payload = _validation_payload(exc.value)
    assert payload["error_type"] == "validation"
    assert "Blocked potentially destructive layout update" in payload["error"]


def test_update_dashboard_allows_empty_layout_with_override(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def __init__(self) -> None:
            self.updated = False

        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "position_json": {
                    "ROOT_ID": {"children": ["GRID_ID"]},
                    "GRID_ID": {"children": ["CHART-1"]},
                    "CHART-1": {"meta": {"chartId": 101}},
                },
            }

        def dashboard_charts(self, dashboard_id: int):
            return [{"id": 101}]

        def update_dashboard(self, dashboard_id: int, **kwargs):
            self.updated = True
            return {"id": dashboard_id, **kwargs}

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.update_dashboard.fn(
        dashboard_id=80,
        position_json='{"DASHBOARD_VERSION_KEY":"v2","ROOT_ID":{"children":["GRID_ID"]},"GRID_ID":{"children":[]}}',
        allow_empty_layout=True,
    )
    payload = json.loads(raw)
    assert ws.updated is True
    assert payload["id"] == 80


def test_update_dashboard_accepts_dict_payload_and_serializes(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def __init__(self) -> None:
            self.kwargs = None

        def dashboard_charts(self, dashboard_id: int):
            return []

        def update_dashboard(self, dashboard_id: int, **kwargs):
            self.kwargs = kwargs
            return {"id": dashboard_id, **kwargs}

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.update_dashboard.fn(
        dashboard_id=80,
        position_json={
            "DASHBOARD_VERSION_KEY": "v2",
            "ROOT_ID": {"children": ["GRID_ID"]},
            "GRID_ID": {"children": ["ROW-1"], "parents": ["ROOT_ID"]},
            "ROW-1": {"children": [], "parents": ["ROOT_ID", "GRID_ID"]},
        },
    )
    payload = json.loads(raw)
    assert payload["id"] == 80
    assert ws.kwargs is not None
    assert isinstance(ws.kwargs["position_json"], str)
    parsed = json.loads(ws.kwargs["position_json"])
    assert parsed["GRID_ID"]["children"] == ["ROW-1"]


def test_update_dashboard_rejects_dangling_layout_children(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dashboard_charts(self, dashboard_id: int):
            return []

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError) as exc:
        server.update_dashboard.fn(
            dashboard_id=80,
            position_json={
                "DASHBOARD_VERSION_KEY": "v2",
                "ROOT_ID": {"children": ["GRID_ID"]},
                "GRID_ID": {"children": ["ROW-404"]},
            },
        )
    payload = _validation_payload(exc.value)
    assert "dangling child reference" in payload["error"]


def test_list_mutations_reads_local_journal(monkeypatch, tmp_path) -> None:
    journal_dir = tmp_path
    journal_file = journal_dir / "mutations.jsonl"
    journal_file.write_text(
        "\n".join(
            [
                '{"timestamp":"2026-03-04T10:00:00Z","tool_name":"update_dashboard","resource_type":"dashboard","resource_id":80}',
                '{"timestamp":"2026-03-04T11:00:00Z","tool_name":"update_chart","resource_type":"chart","resource_id":12}',
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(server, "AUDIT_DIR", journal_dir)

    raw = server.list_mutations.fn(resource_type="dashboard", limit=10)
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["entries"][0]["resource_id"] == 80


def test_restore_dashboard_snapshot_updates_layout(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "dashboard_80_20260304T120000Z.json"
    snapshot.write_text(
        json.dumps(
            {
                "id": 80,
                "position_json": {
                    "DASHBOARD_VERSION_KEY": "v2",
                    "ROOT_ID": {"children": ["GRID_ID"]},
                    "GRID_ID": {"children": [], "parents": ["ROOT_ID"]},
                },
                "json_metadata": {"chartsInScope": {}},
            }
        )
    )

    class _WS(_WorkspaceBase):
        def __init__(self) -> None:
            self.updated = None

        def dashboard_detail(self, dashboard_id: int):
            return {"id": dashboard_id}

        def update_dashboard(self, dashboard_id: int, **kwargs):
            self.updated = kwargs
            return {"id": dashboard_id, **kwargs}

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.restore_dashboard_snapshot.fn(
        dashboard_id=80,
        snapshot_path=str(snapshot),
        restore_json_metadata=True,
    )
    payload = json.loads(raw)
    assert payload["id"] == 80
    assert ws.updated is not None
    assert "position_json" in ws.updated
    assert "json_metadata" in ws.updated


def test_list_dashboard_snapshots_filters_by_dashboard(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    (snapshots / "dashboard_80_20260304T120000Z.json").write_text("{}")
    (snapshots / "dashboard_81_20260304T120100Z.json").write_text("{}")

    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.list_dashboard_snapshots.fn(dashboard_id=80, limit=10)
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["snapshots"][0]["dashboard_id"] == 80


def test_verify_chart_workflow_compact(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def validate_chart_data(self, chart_id: int, dashboard_id=None, row_limit=10000, force=False):
            return {"status": "success", "chart_id": chart_id}

        def validate_chart_render(self, chart_id: int, **kwargs):
            return {"status": "success", "chart_id": chart_id}

        def validate_dashboard_charts(self, dashboard_id: int, row_limit=10000, force=False):
            return {
                "dashboard_id": dashboard_id,
                "results": [{"status": "success"}],
                "chart_count": 1,
                "validated": 1,
            }

        def validate_dashboard_render(self, dashboard_id: int, **kwargs):
            return {"dashboard_id": dashboard_id, "broken_count": 0, "chart_count": 1}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    raw = server.verify_chart_workflow.fn(
        chart_id=10,
        dashboard_id=80,
        include_render=True,
        response_mode="compact",
    )
    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload["dashboard_render_broken_count"] == 0
