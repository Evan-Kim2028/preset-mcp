import json

import pytest
from yarl import URL

from preset_py.client import PresetWorkspace, _critical_page_errors


class _FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_payload = None

    def post(self, _endpoint: str, json=None):
        self.last_payload = json
        return self._response


class _FakeSupersetClient:
    def __init__(self, response: _FakeResponse):
        self.baseurl = URL("https://example.preset.io")
        self.session = _FakeSession(response)
        self.update_chart_calls: list[dict] = []

    def get_dataset(self, _dataset_id: int):
        return {
            "columns": [
                {"column_name": "QC_COL", "type": "VARCHAR"},
                {"column_name": "FD_COL", "type": "VARCHAR"},
                {"column_name": "TOKEN", "type": "VARCHAR"},
                {"column_name": "VOLUME_M", "type": "FLOAT"},
            ],
            "metrics": [],
        }

    def update_chart(self, chart_id: int, **kwargs):
        self.update_chart_calls.append({"chart_id": chart_id, **kwargs})
        return {"id": chart_id, "result": "ok"}


def _workspace_with(
    *,
    chart: dict,
    form_data: dict | None,
    response: _FakeResponse,
    dashboard_id: int | None = 162,
) -> tuple[PresetWorkspace, _FakeSupersetClient]:
    client = _FakeSupersetClient(response)
    ws = PresetWorkspace.__new__(PresetWorkspace)
    ws._superset = client
    ws.get_resource = lambda _rtype, _rid: chart
    ws.chart_form_data = lambda _cid, dashboard_id=None: (form_data, dashboard_id or 162) if form_data is not None else (None, dashboard_id)
    return ws, client


def test_validate_chart_uses_saved_query_context_and_aggregates_results() -> None:
    chart = {
        "id": 1,
        "slice_name": "QC Chart",
        "query_context": json.dumps(
            {
                "datasource": {"id": 42, "type": "table"},
                "queries": [
                    {"columns": ["QC_COL"], "metrics": [], "row_limit": 500},
                    {"columns": ["QC_COL"], "metrics": [], "row_limit": 500},
                ],
            }
        ),
        "params": json.dumps({"datasource": "42__table"}),
    }
    form_data = {
        "datasource": "42__table",
        "groupby": ["FD_COL"],
        "metrics": ["VOLUME_M"],
    }
    response = _FakeResponse(
        200,
        {
            "result": [
                {"status": "success", "error": None, "rowcount": 10, "cache_key": "k1"},
                {"status": "failed", "error": "boom", "rowcount": 0, "cache_key": "k2"},
            ]
        },
    )
    ws, client = _workspace_with(chart=chart, form_data=form_data, response=response)

    result = ws.validate_chart_data(1, dashboard_id=162, row_limit=100, force=True)

    assert client.session.last_payload is not None
    assert client.session.last_payload["queries"][0]["columns"] == ["QC_COL"]
    assert client.session.last_payload["queries"][0]["row_limit"] == 100
    assert result["payload_source"] == "chart.query_context"
    assert result["status"] == "failed"
    assert result["error"] == "boom"
    assert result["row_count"] == 10
    assert result["row_count_total"] == 10
    assert len(result["query_results"]) == 2


def test_validate_chart_falls_back_to_chart_params_when_dashboard_form_data_missing() -> None:
    chart = {
        "id": 2,
        "slice_name": "Params Fallback",
        "query_context": None,
        "datasource_id": 42,
        "datasource_type": "table",
        "params": json.dumps(
            {
                "datasource": "42__table",
                "groupby": ["TOKEN"],
                "metrics": [
                    {
                        "expressionType": "SIMPLE",
                        "column": {"column_name": "VOLUME_M"},
                        "aggregate": "SUM",
                        "label": "Volume",
                    }
                ],
            }
        ),
    }
    response = _FakeResponse(
        200,
        {"result": [{"status": "success", "error": None, "rowcount": 5, "cache_key": "ok"}]},
    )
    ws, client = _workspace_with(chart=chart, form_data=None, response=response)

    result = ws.validate_chart_data(2, dashboard_id=None, row_limit=100, force=False)

    assert result["status"] == "success"
    assert result["form_data_source"] == "chart.params"
    assert result["payload_source"] == "synthetic.form_data"
    assert client.session.last_payload is not None
    assert client.session.last_payload["queries"][0]["columns"] == ["TOKEN"]


def test_validate_chart_reports_unsupported_when_no_query_context_or_form_data() -> None:
    chart = {
        "id": 3,
        "slice_name": "Missing Payloads",
        "query_context": None,
        "params": None,
        "datasource_id": 42,
        "datasource_type": "table",
    }
    response = _FakeResponse(200, {"result": [{"status": "success"}]})
    ws, _client = _workspace_with(chart=chart, form_data=None, response=response)

    result = ws.validate_chart_data(3, dashboard_id=None)

    assert result["status"] == "unsupported"
    assert "query_context or form_data" in result["error"]


def test_critical_page_errors_filters_known_noise() -> None:
    page_errors = [
        "Failed to fetch",
        "Uncaught (in promise) #<Response>",
        "Unexpected token '<', \"<!DOCTYPE \"... is not valid JSON",
        "TypeError: Cannot read properties of undefined (reading 'foo')",
    ]
    critical = _critical_page_errors(page_errors)
    assert critical == ["TypeError: Cannot read properties of undefined (reading 'foo')"]


def test_query_dataset_works_without_datetime_column() -> None:
    response = _FakeResponse(
        200,
        {
            "result": [
                {
                    "status": "success",
                    "error": None,
                    "data": [{"x": 1}],
                }
            ]
        },
    )
    client = _FakeSupersetClient(response)
    ws = PresetWorkspace.__new__(PresetWorkspace)
    ws._superset = client

    df = ws.query_dataset(
        dataset_id=814,
        metrics=["count"],
        columns=["TOKEN"],
        is_timeseries=False,
    )

    assert client.session.last_payload is not None
    query = client.session.last_payload["queries"][0]
    assert query["is_timeseries"] is False
    assert query["time_range"] == "No filter"
    assert df.to_dict("records") == [{"x": 1}]


def test_query_dataset_surfaces_empty_api_error() -> None:
    response = _FakeResponse(400, {})
    client = _FakeSupersetClient(response)
    ws = PresetWorkspace.__new__(PresetWorkspace)
    ws._superset = client

    with pytest.raises(ValueError, match="empty API error payload"):
        ws.query_dataset(
            dataset_id=814,
            metrics=["count"],
            columns=[],
        )


def test_validate_chart_translates_orderby_null_errors() -> None:
    chart = {
        "id": 4,
        "slice_name": "Orderby Error",
        "query_context": None,
        "datasource_id": 42,
        "datasource_type": "table",
        "params": json.dumps(
            {
                "datasource": "42__table",
                "groupby": ["TOKEN"],
                "metrics": ["count"],
            }
        ),
    }
    response = _FakeResponse(
        400,
        {
            "message": {
                "queries": {
                    "0": {
                        "orderby": {
                            "0": {"0": ["Field may not be null."]},
                        },
                    }
                }
            }
        },
    )
    ws, _client = _workspace_with(chart=chart, form_data=None, response=response)

    result = ws.validate_chart_data(4, dashboard_id=None)

    assert result["status"] == "request_failed"
    assert "invalid metric key" in str(result["error"])


def test_validate_chart_persists_synthetic_query_context_when_requested() -> None:
    """Issue #35: synthetic query_context should be written back to the chart."""
    chart = {
        "id": 5,
        "slice_name": "Persist QC Chart",
        "query_context": None,
        "datasource_id": 42,
        "datasource_type": "table",
        "params": json.dumps(
            {
                "datasource": "42__table",
                "groupby": ["TOKEN"],
                "metrics": ["count"],
            }
        ),
    }
    response = _FakeResponse(
        200,
        {"result": [{"status": "success", "error": None, "rowcount": 3, "cache_key": "x"}]},
    )
    ws, client = _workspace_with(chart=chart, form_data=None, response=response)

    result = ws.validate_chart_data(5, dashboard_id=None, persist_synthetic=True)

    assert result["status"] == "success"
    assert result["payload_source"] == "synthetic.form_data"
    assert len(client.update_chart_calls) == 1
    call = client.update_chart_calls[0]
    assert call["chart_id"] == 5
    persisted = json.loads(call["query_context"])
    assert "datasource" in persisted
    assert persisted["datasource"]["id"] == 42
    assert "queries" in persisted
    assert isinstance(persisted["queries"], list)
    assert len(persisted["queries"]) == 1


def test_validate_chart_does_not_persist_when_persist_synthetic_false() -> None:
    """Default behavior: no update_chart call when persist_synthetic is False."""
    chart = {
        "id": 6,
        "slice_name": "No Persist Chart",
        "query_context": None,
        "datasource_id": 42,
        "datasource_type": "table",
        "params": json.dumps({"datasource": "42__table", "groupby": ["TOKEN"], "metrics": ["count"]}),
    }
    response = _FakeResponse(
        200,
        {"result": [{"status": "success", "error": None, "rowcount": 1, "cache_key": "y"}]},
    )
    ws, client = _workspace_with(chart=chart, form_data=None, response=response)

    ws.validate_chart_data(6, dashboard_id=None, persist_synthetic=False)

    assert client.update_chart_calls == []


def test_validate_chart_does_not_persist_when_existing_query_context_used() -> None:
    """When chart already has query_context, no update_chart call should be made."""
    chart = {
        "id": 7,
        "slice_name": "Real QC Chart",
        "query_context": json.dumps(
            {
                "datasource": {"id": 42, "type": "table"},
                "queries": [{"columns": ["TOKEN"], "metrics": [], "row_limit": 1000}],
            }
        ),
        "params": json.dumps({"datasource": "42__table"}),
    }
    response = _FakeResponse(
        200,
        {"result": [{"status": "success", "error": None, "rowcount": 2, "cache_key": "z"}]},
    )
    ws, client = _workspace_with(chart=chart, form_data=None, response=response)

    ws.validate_chart_data(7, dashboard_id=None, persist_synthetic=True)

    assert client.update_chart_calls == []
