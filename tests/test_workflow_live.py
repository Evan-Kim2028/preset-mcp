from preset_py.workflow.live_adapter import apply_live_chart_preset


class _WS:
    def __init__(self) -> None:
        self.updated = None

    def chart_detail(self, chart_id: int) -> dict:
        return {
            "id": chart_id,
            "viz_type": "echarts_timeseries_line",
            "datasource_id": 10,
            "params": '{"datasource":"10__table","viz_type":"echarts_timeseries_line","metrics":["count"],"groupby":["CHAIN"]}',
        }

    def get_resource(self, resource_type: str, resource_id: int) -> dict:
        assert resource_type == "chart"
        return self.chart_detail(resource_id)

    def dataset_detail(self, dataset_id: int) -> dict:
        return {
            "id": dataset_id,
            "columns": [{"column_name": "CHAIN"}],
            "metrics": [{"metric_name": "count"}],
        }

    def update_chart(self, chart_id: int, **kwargs) -> dict:
        self.updated = kwargs
        return {"id": chart_id, **kwargs}


def test_apply_live_chart_preset_compiles_to_update_chart() -> None:
    ws = _WS()

    result = apply_live_chart_preset(ws, chart_id=5, preset_name="timeseries_default")

    assert result["id"] == 5
    assert ws.updated is not None
    assert '"x_axis_title_margin": 30' in ws.updated["params"]
