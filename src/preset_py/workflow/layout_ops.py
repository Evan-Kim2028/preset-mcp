from collections import defaultdict
from typing import Any


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


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
        chart_id = _to_int(node.get("chartId"))
        if chart_id is None:
            meta = node.get("meta")
            if isinstance(meta, dict):
                chart_id = _to_int(meta.get("chartId"))
        if chart_id is not None:
            found[chart_id].append(node_id)
    return dict(found)


def has_tab_container(position: dict[str, Any]) -> bool:
    return any(
        isinstance(node, dict) and node.get("type") == "TABS"
        for node in position.values()
    )
