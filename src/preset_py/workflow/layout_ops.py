from collections import defaultdict
from typing import Any


def list_tab_ids(position: dict[str, Any]) -> list[str]:
    return sorted(
        node_id
        for node_id, node in position.items()
        if isinstance(node, dict) and node.get("type") == "TAB"
    )


def find_chart_node_ids(position: dict[str, Any]) -> dict[int, list[str]]:
    found: dict[int, list[str]] = defaultdict(list)
    for node_id, node in position.items():
        if not isinstance(node, dict):
            continue
        meta = node.get("meta") or {}
        chart_id = meta.get("chartId")
        if isinstance(chart_id, int):
            found[chart_id].append(node_id)
    return dict(found)
