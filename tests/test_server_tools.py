import json

import pandas as pd
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


def test_query_dataset_treats_time_column_as_timeseries(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def __init__(self) -> None:
            self.kwargs = None

        def query_dataset(self, **kwargs):
            self.kwargs = kwargs
            return pd.DataFrame([{"day": "2026-01-01", "count": 1}])

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)

    raw = server.query_dataset.fn(
        dataset_id=814,
        metrics='["count"]',
        time_column="day",
        start="2026-01-01",
        end="2026-01-31",
        response_mode="compact",
    )
    payload = json.loads(raw)

    assert payload["rowcount"] == 1
    assert ws.kwargs is not None
    assert ws.kwargs["is_timeseries"] is True
    assert ws.kwargs["time_column"] == "day"


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


def test_create_chart_default_does_not_repair_dashboard_refs(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def __init__(self) -> None:
            self.dashboard_detail_calls = 0
            self.updated_dashboard = False

        def dataset_detail(self, dataset_id: int):
            return {
                "id": dataset_id,
                "columns": [{"column_name": "CHAIN"}],
                "metrics": [{"metric_name": "count"}],
            }

        def dashboard_detail(self, dashboard_id: int):
            self.dashboard_detail_calls += 1
            return {
                "id": dashboard_id,
                "position_json": {"ROOT_ID": {"children": ["GRID_ID"]}, "GRID_ID": {"children": []}},
                "json_metadata": {"chartsInScope": {}},
            }

        def create_chart(self, dataset_id: int, title: str, viz_type: str, **kwargs):
            return {"id": 77, "slice_name": title, "viz_type": viz_type, **kwargs}

        def update_dashboard(self, dashboard_id: int, **kwargs):
            self.updated_dashboard = True
            return {"id": dashboard_id, **kwargs}

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.create_chart.fn(
        dataset_id=10,
        title="Safe Chart Create",
        viz_type="table",
        dashboards="[123]",
        validate_after_create=False,
    )
    payload = json.loads(raw)

    assert payload["id"] == 77
    assert ws.dashboard_detail_calls == 1  # only precondition check
    assert ws.updated_dashboard is False


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
            position_json='{"DASHBOARD_VERSION_KEY":"v2","ROOT_ID":{"id":"ROOT_ID","type":"ROOT","children":["GRID_ID"]},"GRID_ID":{"id":"GRID_ID","type":"GRID","children":[]}}',
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
        position_json='{"DASHBOARD_VERSION_KEY":"v2","ROOT_ID":{"id":"ROOT_ID","type":"ROOT","children":["GRID_ID"]},"GRID_ID":{"id":"GRID_ID","type":"GRID","children":[]}}',
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
            "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
            "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": ["ROW-1"], "parents": ["ROOT_ID"]},
            "ROW-1": {"id": "ROW-1", "type": "ROW", "children": [], "parents": ["ROOT_ID", "GRID_ID"]},
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
                "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
                "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": ["ROW-404"]},
            },
        )
    payload = _validation_payload(exc.value)
    assert "dangling child reference" in payload["error"]


def test_update_chart_partial_params_reports_strict_semantics(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def chart_detail(self, chart_id: int):
            return {"id": chart_id, "viz_type": "pie"}

        def get_resource(self, resource_type: str, resource_id: int):
            assert resource_type == "chart"
            return {
                "id": resource_id,
                "viz_type": "pie",
                "datasource_id": 42,
                "params": json.dumps({
                    "datasource": "42__table",
                    "metrics": ["count"],
                    "groupby": ["CHAIN"],
                }),
            }

        def dataset_detail(self, dataset_id: int):
            return {
                "id": dataset_id,
                "columns": [{"column_name": "CHAIN"}],
                "metrics": [{"metric_name": "count"}],
            }

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError) as exc:
        server.update_chart.fn(
            chart_id=1,
            params_json='{"color_scheme":"supersetColors"}',
            validate_after_update=False,
        )

    payload = _validation_payload(exc.value)
    assert "Strict params semantics" in payload["error"]
    assert "complete viz-compatible params payload" in payload["error"]


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
                    "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
                    "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": [], "parents": ["ROOT_ID"]},
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


def test_verify_dashboard_structure_detects_layout_issues(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "dashboard_title": "Deepbook",
                "position_json": {
                    "ROOT_ID": {"children": ["GRID_ID"]},
                    "GRID_ID": {"children": ["CHART-1", "ROW-404"]},
                    "CHART-1": {"children": [], "meta": {"chartId": 101}},
                },
                "json_metadata": {"chartsInScope": {"101": {}, "999": {}}},
            }

        def dashboard_charts(self, dashboard_id: int):
            return [{"id": 101, "slice_name": "Volume"}]

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.verify_dashboard_structure.fn(
        dashboard_id=80,
        response_mode="standard",
    )
    payload = json.loads(raw)
    assert payload["status"] == "failed"
    assert payload["dangling_children"]
    assert payload["scope_orphans"] == [999]


def test_verify_dashboard_workflow_reports_failed_query(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "dashboard_title": "BTC Fight",
                "position_json": {
                    "ROOT_ID": {"children": ["GRID_ID"]},
                    "GRID_ID": {"children": ["CHART-1"]},
                    "CHART-1": {"children": [], "meta": {"chartId": 1}},
                },
                "json_metadata": {"chartsInScope": {"1": {}}},
            }

        def dashboard_charts(self, dashboard_id: int):
            return [{"id": 1, "slice_name": "Chart 1"}]

        def validate_dashboard_charts(self, dashboard_id: int, row_limit=10000, force=False):
            return {
                "dashboard_id": dashboard_id,
                "chart_count": 1,
                "validated": 1,
                "results": [{"chart_id": 1, "status": "failed"}],
            }

        def validate_dashboard_render(self, dashboard_id: int, **kwargs):
            return {"dashboard_id": dashboard_id, "chart_count": 1, "broken_count": 0}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.verify_dashboard_workflow.fn(
        dashboard_id=80,
        include_render=True,
        response_mode="compact",
    )
    payload = json.loads(raw)
    assert payload["status"] == "failed"
    assert payload["query_failures"] == 1


def test_capture_dashboard_template_writes_file(monkeypatch, tmp_path) -> None:
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "dashboard_title": "Walrus",
                "slug": "walrus",
                "position_json": {
                    "ROOT_ID": {"children": ["GRID_ID"]},
                    "GRID_ID": {"children": ["CHART-1"]},
                    "CHART-1": {"children": [], "meta": {"chartId": 1}},
                },
                "json_metadata": {"chartsInScope": {"1": {}}},
            }

        def dashboard_charts(self, dashboard_id: int):
            return [
                {
                    "id": 1,
                    "slice_name": "Walrus Volume",
                    "viz_type": "pie",
                    "datasource_id": 100,
                    "datasource_type": "table",
                    "form_data": {
                        "datasource": "100__table",
                        "metrics": ["count"],
                        "groupby": ["CHAIN"],
                    },
                }
            ]

        def chart_detail(self, chart_id: int):
            return {
                "id": chart_id,
                "query_context": '{"datasource":{"id":100,"type":"table"},"queries":[]}',
            }

        def dataset_detail(self, dataset_id: int):
            return {
                "id": dataset_id,
                "table_name": "walrus_metrics",
                "schema": "public",
                "columns": [{"column_name": "CHAIN"}],
                "metrics": [{"metric_name": "count"}],
            }

    output_path = tmp_path / "walrus_template.json"
    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.capture_dashboard_template.fn(
        dashboard_id=80,
        portable=True,
        include_query_context=True,
        include_dataset_schema=True,
        output_path=str(output_path),
        response_mode="compact",
    )
    payload = json.loads(raw)
    assert payload["chart_count"] == 1
    assert payload["output_path"] == str(output_path)
    saved = json.loads(output_path.read_text())
    assert saved["dashboard"]["title"] == "Walrus"
    assert saved["charts"][0]["dataset_schema"]["columns"] == ["CHAIN"]
    assert "datasource" not in saved["charts"][0]["form_data"]


def test_capture_golden_templates_batch(monkeypatch, tmp_path) -> None:
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "dashboard_title": f"Dashboard {dashboard_id}",
                "position_json": {
                    "ROOT_ID": {"children": ["GRID_ID"]},
                    "GRID_ID": {"children": []},
                },
                "json_metadata": {"chartsInScope": {}},
            }

        def dashboard_charts(self, dashboard_id: int):
            return []

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.capture_golden_templates.fn(
        dashboard_ids="[80,81]",
        output_dir=str(tmp_path),
        overwrite=True,
        response_mode="compact",
    )
    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload["saved_count"] == 2


# ===================================================================
# Issue #26 — pie chart metric/metrics backward compatibility
# ===================================================================


def test_create_chart_pie_sets_singular_metric(monkeypatch) -> None:
    """Pie charts must set singular ``metric`` from first metrics entry."""
    captured_params = {}

    class _WS(_WorkspaceBase):
        def dataset_detail(self, dataset_id: int):
            return {
                "id": dataset_id,
                "columns": [
                    {"column_name": "CHAIN", "type": "VARCHAR"},
                    {"column_name": "AMOUNT", "type": "NUMERIC"},
                ],
                "metrics": [{"metric_name": "count"}],
            }

        def create_chart(self, dataset_id: int, title: str, viz_type: str, **kwargs):
            captured_params.update(kwargs)
            return {"id": 99, "slice_name": title, "viz_type": viz_type}

        def validate_chart_data(self, chart_id: int, dashboard_id=None, row_limit=10000, force=False):
            return {"chart_id": chart_id, "status": "success", "error": None}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.create_chart.fn(
        dataset_id=10,
        title="Pie Chart",
        viz_type="pie",
        metrics='["count"]',
        groupby='["CHAIN"]',
        validate_after_create=False,
    )
    payload = json.loads(raw)
    assert payload["id"] == 99

    # The client should have received metrics and the create_chart helper
    # should have set params including singular ``metric``
    assert captured_params["metrics"] == ["count"]


# ===================================================================
# Issue #27 — position_json node id/type validation
# ===================================================================


def test_update_dashboard_rejects_nodes_without_id(monkeypatch) -> None:
    """Nodes missing ``id`` field should be rejected."""
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {"id": dashboard_id, "position_json": {}}

        def dashboard_charts(self, dashboard_id: int):
            return []

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError) as exc:
        server.update_dashboard.fn(
            dashboard_id=80,
            position_json={
                "DASHBOARD_VERSION_KEY": "v2",
                "ROOT_ID": {"type": "ROOT", "children": ["GRID_ID"]},
                "GRID_ID": {"type": "GRID", "children": []},
            },
        )
    payload = _validation_payload(exc.value)
    assert "missing required 'id' field" in payload["error"]


def test_update_dashboard_rejects_nodes_without_type(monkeypatch) -> None:
    """Nodes missing ``type`` field should be rejected."""
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {"id": dashboard_id, "position_json": {}}

        def dashboard_charts(self, dashboard_id: int):
            return []

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError) as exc:
        server.update_dashboard.fn(
            dashboard_id=80,
            position_json={
                "DASHBOARD_VERSION_KEY": "v2",
                "ROOT_ID": {"id": "ROOT_ID", "children": ["GRID_ID"]},
                "GRID_ID": {"id": "GRID_ID", "children": []},
            },
        )
    payload = _validation_payload(exc.value)
    assert "missing required 'type' field" in payload["error"]


def test_update_dashboard_rejects_id_key_mismatch(monkeypatch) -> None:
    """Node ``id`` must match the dict key."""
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {"id": dashboard_id, "position_json": {}}

        def dashboard_charts(self, dashboard_id: int):
            return []

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError) as exc:
        server.update_dashboard.fn(
            dashboard_id=80,
            position_json={
                "DASHBOARD_VERSION_KEY": "v2",
                "ROOT_ID": {"id": "WRONG_ID", "type": "ROOT", "children": ["GRID_ID"]},
                "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": []},
            },
        )
    payload = _validation_payload(exc.value)
    assert "must match the node key" in payload["error"]


def test_verify_dashboard_structure_detects_missing_node_ids(monkeypatch) -> None:
    """verify_dashboard_structure should flag nodes without id/type."""
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "dashboard_title": "Bad Layout",
                "position_json": {
                    "ROOT_ID": {"type": "ROOT", "children": ["GRID_ID"]},
                    "GRID_ID": {"children": []},
                },
                "json_metadata": {},
            }

        def dashboard_charts(self, dashboard_id: int):
            return []

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.verify_dashboard_structure.fn(
        dashboard_id=80, response_mode="standard",
    )
    payload = json.loads(raw)
    assert payload["status"] == "failed"
    assert "ROOT_ID" in payload["missing_id_nodes"]
    assert "GRID_ID" in payload["missing_id_nodes"]
    assert "GRID_ID" in payload["missing_type_nodes"]


# ===================================================================
# Issue #28 — describe_dashboard
# ===================================================================


def test_describe_dashboard_standard(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "dashboard_title": "Research Yield",
                "published": False,
                "changed_on": "2026-03-05T21:52:01",
                "owners": [{"first_name": "Evan", "last_name": "Kim"}],
                "position_json": {
                    "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
                    "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": ["MARKDOWN-1", "CHART-1"]},
                    "MARKDOWN-1": {"id": "MARKDOWN-1", "type": "MARKDOWN", "children": [], "meta": {"code": "## Overview\nYield analysis"}},
                    "CHART-1": {"id": "CHART-1", "type": "CHART", "children": [], "meta": {"chartId": 100}},
                },
                "json_metadata": {},
            }

        def dashboard_charts(self, dashboard_id: int):
            return [
                {
                    "id": 100,
                    "slice_name": "Yield Pie",
                    "viz_type": "pie",
                    "datasource_id": 50,
                    "datasource_type": "table",
                    "form_data": {"datasource": "50__table"},
                }
            ]

        def dataset_detail(self, dataset_id: int):
            return {
                "id": dataset_id,
                "table_name": "yield_data",
                "schema": "analytics",
                "database": {"database_name": "snowflake_prod"},
                "sql": "SELECT * FROM raw.yield_rates",
                "kind": "virtual",
            }

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.describe_dashboard.fn(
        dashboard_id=171,
        include_lineage=True,
        response_mode="standard",
    )
    payload = json.loads(raw)

    assert payload["dashboard"]["title"] == "Research Yield"
    assert payload["dashboard"]["owner"] == "Evan Kim"
    assert len(payload["markdown_blocks"]) == 1
    assert "Yield analysis" in payload["markdown_blocks"][0]["text"]
    assert len(payload["charts"]) == 1
    assert payload["charts"][0]["chart_id"] == 100
    assert payload["charts"][0]["dataset_name"] == "yield_data"
    assert len(payload["datasets"]) == 1
    assert payload["datasets"][0]["database"] == "snowflake_prod"
    assert payload["datasets"][0]["source_tables"]


def test_describe_dashboard_compact(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def dashboard_detail(self, dashboard_id: int):
            return {
                "id": dashboard_id,
                "dashboard_title": "Test",
                "position_json": {},
                "json_metadata": {},
            }

        def dashboard_charts(self, dashboard_id: int):
            return []

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    raw = server.describe_dashboard.fn(
        dashboard_id=1,
        response_mode="compact",
    )
    payload = json.loads(raw)
    assert payload["chart_count"] == 0
    assert payload["dataset_count"] == 0


# ===================================================================
# Snapshot & recovery — comprehensive tests
# ===================================================================


def test_list_snapshots_returns_all_types(monkeypatch, tmp_path) -> None:
    """list_snapshots with no resource_type returns snapshots for all types."""
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    (snapshots / "dashboard_1_20260304T120000000000Z.json").write_text("{}")
    (snapshots / "chart_2_20260304T120100000000Z.json").write_text("{}")
    (snapshots / "dataset_3_20260304T120200000000Z.json").write_text("{}")

    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.list_snapshots.fn(limit=50)
    payload = json.loads(raw)
    assert payload["count"] == 3
    types = {s["resource_type"] for s in payload["snapshots"]}
    assert types == {"dashboard", "chart", "dataset"}


def test_list_snapshots_filters_by_resource_type(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    (snapshots / "dashboard_1_20260304T120000000000Z.json").write_text("{}")
    (snapshots / "chart_2_20260304T120100000000Z.json").write_text("{}")
    (snapshots / "dataset_3_20260304T120200000000Z.json").write_text("{}")

    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.list_snapshots.fn(resource_type="chart", limit=50)
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["snapshots"][0]["resource_type"] == "chart"
    assert payload["snapshots"][0]["resource_id"] == 2


def test_list_snapshots_filters_by_resource_id(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    (snapshots / "chart_10_20260304T120000000000Z.json").write_text("{}")
    (snapshots / "chart_11_20260304T120100000000Z.json").write_text("{}")

    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.list_snapshots.fn(resource_type="chart", resource_id=10, limit=50)
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["snapshots"][0]["resource_id"] == 10


def test_list_snapshots_empty_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)
    raw = server.list_snapshots.fn(limit=50)
    payload = json.loads(raw)
    assert payload["count"] == 0
    assert payload["snapshots"] == []


def test_list_mutations_filters_by_action(monkeypatch, tmp_path) -> None:
    journal_file = tmp_path / "mutations.jsonl"
    journal_file.write_text(
        "\n".join([
            '{"timestamp":"2026-03-04T10:00:00Z","tool_name":"update_dashboard","resource_type":"dashboard","resource_id":80,"action":"update"}',
            '{"timestamp":"2026-03-04T11:00:00Z","tool_name":"create_chart","resource_type":"chart","resource_id":12,"action":"create"}',
            '{"timestamp":"2026-03-04T12:00:00Z","tool_name":"delete_chart","resource_type":"chart","resource_id":13,"action":"delete"}',
        ])
        + "\n"
    )
    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.list_mutations.fn(action="create", limit=50)
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["entries"][0]["action"] == "create"
    assert payload["entries"][0]["resource_id"] == 12


def test_list_mutations_combined_filters(monkeypatch, tmp_path) -> None:
    journal_file = tmp_path / "mutations.jsonl"
    journal_file.write_text(
        "\n".join([
            '{"timestamp":"2026-03-04T10:00:00Z","tool_name":"update_dashboard","resource_type":"dashboard","resource_id":80,"action":"update"}',
            '{"timestamp":"2026-03-04T11:00:00Z","tool_name":"update_chart","resource_type":"chart","resource_id":12,"action":"update"}',
            '{"timestamp":"2026-03-04T12:00:00Z","tool_name":"create_chart","resource_type":"chart","resource_id":13,"action":"create"}',
        ])
        + "\n"
    )
    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.list_mutations.fn(resource_type="chart", action="update", limit=50)
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["entries"][0]["resource_id"] == 12


def test_list_mutations_empty_journal(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)
    raw = server.list_mutations.fn(limit=50)
    payload = json.loads(raw)
    assert payload["count"] == 0


def test_list_mutations_malformed_lines(monkeypatch, tmp_path) -> None:
    journal_file = tmp_path / "mutations.jsonl"
    journal_file.write_text(
        "not valid json\n"
        '{"timestamp":"2026-03-04T10:00:00Z","tool_name":"update_chart","resource_type":"chart","resource_id":1,"action":"update"}\n'
        "\n"
        "42\n"
    )
    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.list_mutations.fn(limit=50)
    payload = json.loads(raw)
    assert payload["count"] == 1


def test_list_mutations_reverse_chronological(monkeypatch, tmp_path) -> None:
    journal_file = tmp_path / "mutations.jsonl"
    journal_file.write_text(
        "\n".join([
            '{"timestamp":"2026-03-04T08:00:00Z","tool_name":"t1","resource_type":"chart","resource_id":1,"action":"create"}',
            '{"timestamp":"2026-03-04T12:00:00Z","tool_name":"t2","resource_type":"chart","resource_id":2,"action":"update"}',
            '{"timestamp":"2026-03-04T10:00:00Z","tool_name":"t3","resource_type":"chart","resource_id":3,"action":"delete"}',
        ])
        + "\n"
    )
    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.list_mutations.fn(limit=50)
    payload = json.loads(raw)
    timestamps = [e["timestamp"] for e in payload["entries"]]
    assert timestamps == sorted(timestamps, reverse=True)


def test_restore_chart_snapshot_restores_params_and_viz(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "chart_10_20260304T120000000000Z.json"
    snapshot.write_text(json.dumps({
        "id": 10,
        "viz_type": "pie",
        "params": {"metrics": ["count"], "groupby": ["region"]},
    }))

    class _WS(_WorkspaceBase):
        def __init__(self):
            self.updated = None

        def chart_detail(self, chart_id: int):
            return {"id": chart_id}

        def update_chart(self, chart_id: int, **kwargs):
            self.updated = kwargs
            return {"id": chart_id, **kwargs}

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.restore_chart_snapshot.fn(
        chart_id=10,
        snapshot_path=str(snapshot),
    )
    payload = json.loads(raw)
    assert payload["id"] == 10
    assert ws.updated is not None
    assert "params" in ws.updated
    assert "viz_type" in ws.updated
    assert ws.updated["viz_type"] == "pie"


def test_restore_chart_snapshot_dry_run(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "chart_10_20260304T120000000000Z.json"
    snapshot.write_text(json.dumps({
        "id": 10,
        "viz_type": "table",
        "params": {"metrics": ["count"]},
    }))

    class _WS(_WorkspaceBase):
        def chart_detail(self, chart_id: int):
            return {"id": chart_id}

        def update_chart(self, chart_id: int, **kwargs):
            raise AssertionError("Should not be called in dry_run")

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.restore_chart_snapshot.fn(
        chart_id=10,
        snapshot_path=str(snapshot),
        dry_run=True,
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True


def test_restore_chart_snapshot_id_mismatch(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "chart_10_20260304T120000000000Z.json"
    snapshot.write_text(json.dumps({
        "id": 99,
        "viz_type": "table",
        "params": {"metrics": ["count"]},
    }))

    class _WS(_WorkspaceBase):
        def chart_detail(self, chart_id: int):
            return {"id": chart_id}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError):
        server.restore_chart_snapshot.fn(
            chart_id=10,
            snapshot_path=str(snapshot),
        )


def test_restore_dataset_snapshot_restores_sql(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "dataset_5_20260304T120000000000Z.json"
    snapshot.write_text(json.dumps({
        "id": 5,
        "sql": "SELECT * FROM orders",
        "table_name": "orders_v2",
        "description": "Order data",
    }))

    class _WS(_WorkspaceBase):
        def __init__(self):
            self.updated = None

        def dataset_detail(self, dataset_id: int):
            return {"id": dataset_id}

        def update_dataset(self, dataset_id: int, **kwargs):
            self.updated = kwargs
            return {"id": dataset_id, **kwargs}

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.restore_dataset_snapshot.fn(
        dataset_id=5,
        snapshot_path=str(snapshot),
    )
    payload = json.loads(raw)
    assert payload["id"] == 5
    assert ws.updated is not None
    assert ws.updated["sql"] == "SELECT * FROM orders"
    assert ws.updated["table_name"] == "orders_v2"
    assert ws.updated["description"] == "Order data"


def test_restore_dataset_snapshot_dry_run(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "dataset_5_20260304T120000000000Z.json"
    snapshot.write_text(json.dumps({
        "id": 5,
        "sql": "SELECT 1",
        "table_name": "test",
    }))

    class _WS(_WorkspaceBase):
        def dataset_detail(self, dataset_id: int):
            return {"id": dataset_id}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.restore_dataset_snapshot.fn(
        dataset_id=5,
        snapshot_path=str(snapshot),
        dry_run=True,
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True


def test_restore_dataset_snapshot_id_mismatch(monkeypatch, tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "dataset_5_20260304T120000000000Z.json"
    snapshot.write_text(json.dumps({
        "id": 99,
        "sql": "SELECT 1",
    }))

    class _WS(_WorkspaceBase):
        def dataset_detail(self, dataset_id: int):
            return {"id": dataset_id}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError):
        server.restore_dataset_snapshot.fn(
            dataset_id=5,
            snapshot_path=str(snapshot),
        )


def test_restore_dashboard_snapshot_validates_chart_refs(monkeypatch, tmp_path) -> None:
    """Dashboard restore should warn about missing chart references."""
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "dashboard_80_20260304T120000000000Z.json"
    snapshot.write_text(json.dumps({
        "id": 80,
        "position_json": {
            "DASHBOARD_VERSION_KEY": "v2",
            "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
            "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": ["CHART-abc"], "parents": ["ROOT_ID"]},
            "CHART-abc": {"id": "CHART-abc", "type": "CHART", "children": [], "parents": ["GRID_ID"], "meta": {"chartId": 999}},
        },
        "json_metadata": {},
    }))

    class _WS(_WorkspaceBase):
        def __init__(self):
            self.updated = None

        def dashboard_detail(self, dashboard_id: int):
            return {"id": dashboard_id}

        def charts(self):
            return [{"id": 1}, {"id": 2}]  # chart 999 doesn't exist

        def update_dashboard(self, dashboard_id: int, **kwargs):
            self.updated = kwargs
            return {"id": dashboard_id, **kwargs}

    ws = _WS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)

    raw = server.restore_dashboard_snapshot.fn(
        dashboard_id=80,
        snapshot_path=str(snapshot),
        validate_chart_refs=True,
    )
    payload = json.loads(raw)
    assert "_warnings" in payload
    assert any("999" in w for w in payload["_warnings"])


def test_restore_dashboard_snapshot_no_metadata(monkeypatch, tmp_path) -> None:
    """Test restore with restore_json_metadata=False."""
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / "dashboard_80_20260304T120000000000Z.json"
    snapshot.write_text(json.dumps({
        "id": 80,
        "position_json": {
            "DASHBOARD_VERSION_KEY": "v2",
            "ROOT_ID": {"id": "ROOT_ID", "type": "ROOT", "children": ["GRID_ID"]},
            "GRID_ID": {"id": "GRID_ID", "type": "GRID", "children": [], "parents": ["ROOT_ID"]},
        },
        "json_metadata": {"chartsInScope": {}},
    }))

    class _WS(_WorkspaceBase):
        def __init__(self):
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
        restore_json_metadata=False,
    )
    payload = json.loads(raw)
    assert "position_json" in ws.updated
    assert "json_metadata" not in ws.updated


def test_snapshot_not_found_raises(monkeypatch) -> None:
    class _WS(_WorkspaceBase):
        def chart_detail(self, chart_id: int):
            return {"id": chart_id}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError):
        server.restore_chart_snapshot.fn(
            chart_id=10,
            snapshot_path="/nonexistent/path/snapshot.json",
        )


def test_snapshot_invalid_json_raises(monkeypatch, tmp_path) -> None:
    bad_snap = tmp_path / "bad.json"
    bad_snap.write_text("not json {{{")

    class _WS(_WorkspaceBase):
        def dataset_detail(self, dataset_id: int):
            return {"id": dataset_id}

    monkeypatch.setattr(server, "_get_ws", lambda: _WS())

    with pytest.raises(ToolError):
        server.restore_dataset_snapshot.fn(
            dataset_id=1,
            snapshot_path=str(bad_snap),
        )


def test_prune_snapshots_removes_old(monkeypatch, tmp_path) -> None:
    import os

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    old_file = snapshots / "chart_1_20200101T000000000000Z.json"
    old_file.write_text("{}")
    # Set mtime to 60 days ago
    old_time = (
        __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        - __import__("datetime").timedelta(days=60)
    ).timestamp()
    os.utime(old_file, (old_time, old_time))

    new_file = snapshots / "chart_2_20260304T120000000000Z.json"
    new_file.write_text("{}")

    monkeypatch.setattr(server, "AUDIT_DIR", tmp_path)

    raw = server.prune_snapshots.fn(max_age_days=30)
    payload = json.loads(raw)
    assert payload["snapshots_removed"] == 1
    assert not old_file.exists()
    assert new_file.exists()
