from pathlib import Path
from typing import Any

from preset_py.workflow.document import load_dashboard_yaml_document
from preset_py.workflow.types import WorkflowPlan


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
