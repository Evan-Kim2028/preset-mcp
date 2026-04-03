import json
from typing import Any

from preset_py.workflow.presets import load_chart_preset


def apply_live_chart_preset(
    ws: Any,
    *,
    chart_id: int,
    preset_name: str,
) -> dict[str, Any]:
    chart = ws.get_resource("chart", chart_id)
    params = json.loads(chart.get("params") or "{}")
    if not isinstance(params, dict):
        raise ValueError(f"chart params must decode to a mapping for chart {chart_id}")

    params.update(load_chart_preset(preset_name))
    params.setdefault("datasource", f'{chart["datasource_id"]}__table')
    params.setdefault("viz_type", chart["viz_type"])
    return ws.update_chart(chart_id, params=json.dumps(params))
