from pathlib import Path
from typing import Any

from preset_py.workflow.document import load_dashboard_yaml_document
from preset_py.workflow.layout_ops import has_tab_container, list_tab_ids
from preset_py.workflow.types import WorkflowPlan
from preset_py.workflow.yaml_adapter import read_yaml_dashboard, write_yaml_dashboard


def plan_add_tab(*, yaml_path: Path, tab_name: str) -> dict[str, Any]:
    document = load_dashboard_yaml_document(Path(yaml_path))
    changes: list[dict[str, Any]] = []
    summary = f"Plan add tab '{tab_name}' on {document.mode} dashboard"
    if document.mode != "tabbed":
        changes.append(
            {"kind": "mode_transition", "from": document.mode, "to": "tabbed"}
        )
        summary = f"Plan add tab '{tab_name}' ({document.mode} -> tabbed)"
    changes.append({"kind": "add_tab", "tab_name": tab_name})
    plan = WorkflowPlan(
        status="planned",
        target={"mode": "yaml", "path": str(yaml_path)},
        summary=summary,
        changes=changes,
    )
    return plan.__dict__


def plan_create_chart_with_preset(
    *,
    yaml_path: Path,
    chart_title: str,
    viz_type: str,
    chart_preset: str,
) -> dict[str, Any]:
    document = load_dashboard_yaml_document(Path(yaml_path))
    plan = WorkflowPlan(
        status="planned",
        target={"mode": "yaml", "path": str(yaml_path)},
        summary=f"Plan create chart '{chart_title}' on {document.mode} dashboard",
        changes=[
            {
                "kind": "create_chart",
                "title": chart_title,
                "viz_type": viz_type,
                "chart_preset": chart_preset,
            }
        ],
    )
    return plan.__dict__


def _tab_meta(text: str) -> dict[str, str]:
    return {
        "text": text,
        "defaultText": "Tab title",
        "placeholder": "Tab title",
    }


def _prepend_parents(
    position: dict[str, Any],
    *,
    node_id: str,
    root_id: str,
    extra_parents: list[str],
) -> None:
    node = position.get(node_id)
    if not isinstance(node, dict):
        return

    parents = node.get("parents")
    if isinstance(parents, list):
        cleaned = [str(parent) for parent in parents]
        if cleaned and cleaned[0] == root_id:
            node["parents"] = [root_id, *extra_parents, *cleaned[1:]]
        else:
            node["parents"] = [*extra_parents, *cleaned]

    for child_id in node.get("children", []):
        if isinstance(child_id, str):
            _prepend_parents(
                position,
                node_id=child_id,
                root_id=root_id,
                extra_parents=extra_parents,
            )


def _apply_add_tab(position: dict[str, Any], tab_name: str) -> None:
    root_id = "ROOT_ID"
    root = position.get(root_id)
    if not isinstance(root, dict):
        raise ValueError("dashboard position is missing ROOT_ID")

    if has_tab_container(position):
        tabs_id = next(
            node_id
            for node_id, node in position.items()
            if isinstance(node, dict) and node.get("type") == "TABS"
        )
        next_tab_number = len(list_tab_ids(position)) + 1
        tab_id = f"TAB-{next_tab_number}"
        grid_id = f"GRID-{next_tab_number}"
        tabs_node = position[tabs_id]
        tabs_node.setdefault("children", []).append(tab_id)
        position[tab_id] = {
            "id": tab_id,
            "type": "TAB",
            "parents": [root_id, tabs_id],
            "children": [grid_id],
            "meta": _tab_meta(tab_name),
        }
        position[grid_id] = {
            "id": grid_id,
            "type": "GRID",
            "parents": [root_id, tabs_id, tab_id],
            "children": [],
        }
        return

    grid_id = next(
        (
            child_id
            for child_id in root.get("children", [])
            if isinstance(child_id, str)
            and isinstance(position.get(child_id), dict)
            and position[child_id].get("type") == "GRID"
        ),
        None,
    )
    if grid_id is None:
        raise ValueError("flat dashboard must have a root grid before adding a tab")

    tabs_id = "TABS_ID"
    overview_tab_id = "TAB-1"
    new_tab_id = "TAB-2"
    new_grid_id = "GRID-2"

    root["children"] = [tabs_id]
    position[tabs_id] = {
        "id": tabs_id,
        "type": "TABS",
        "parents": [root_id],
        "children": [overview_tab_id, new_tab_id],
    }
    position[overview_tab_id] = {
        "id": overview_tab_id,
        "type": "TAB",
        "parents": [root_id, tabs_id],
        "children": [grid_id],
        "meta": _tab_meta("Overview"),
    }
    position[new_tab_id] = {
        "id": new_tab_id,
        "type": "TAB",
        "parents": [root_id, tabs_id],
        "children": [new_grid_id],
        "meta": _tab_meta(tab_name),
    }
    _prepend_parents(
        position,
        node_id=grid_id,
        root_id=root_id,
        extra_parents=[tabs_id, overview_tab_id],
    )
    position[new_grid_id] = {
        "id": new_grid_id,
        "type": "GRID",
        "parents": [root_id, tabs_id, new_tab_id],
        "children": [],
    }


def apply_workflow_plan(plan: dict[str, Any]) -> dict[str, Any]:
    target = plan.get("target") or {}
    if target.get("mode") != "yaml":
        raise ValueError("only YAML workflow targets are supported")

    path = Path(str(target["path"]))
    payload = read_yaml_dashboard(path)
    position = payload.setdefault("position", {})
    if not isinstance(position, dict):
        raise ValueError("dashboard position must be a mapping")

    for change in plan.get("changes", []):
        if change.get("kind") == "add_tab":
            _apply_add_tab(position, str(change["tab_name"]))

    write_yaml_dashboard(path, payload)
    return {
        "status": "applied",
        "target": target,
        "applied_changes": plan.get("changes", []),
    }
