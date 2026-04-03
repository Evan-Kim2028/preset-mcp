from pathlib import Path

import pytest

from preset_py.workflow.document import load_dashboard_yaml_document
from preset_py.workflow.layout_ops import find_chart_node_ids, list_tab_ids
from preset_py.workflow.planner import (
    apply_workflow_plan,
    plan_add_tab,
    plan_create_chart_with_preset,
)
from preset_py.workflow.presets import load_chart_preset, load_layout_preset


def test_load_dashboard_yaml_document_reads_flat_fixture() -> None:
    fixture = Path("tests/fixtures/workflow/research_yield_simple.yaml")

    document = load_dashboard_yaml_document(fixture)

    assert document.dashboard_title == "[Research] Yield Simple"
    assert document.mode == "flat"
    assert "ROOT_ID" in document.position


def test_load_dashboard_yaml_document_reads_tabbed_fixture() -> None:
    fixture = Path("tests/fixtures/workflow/tabbed_dashboard.yaml")

    document = load_dashboard_yaml_document(fixture)

    assert document.dashboard_title == "[Research] Tabbed Dashboard"
    assert document.mode == "tabbed"
    assert "TABS_ID" in document.position


def test_load_dashboard_yaml_document_marks_empty_tabs_container_as_tabbed(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "authoring_tabbed.yaml"
    fixture.write_text(
        """
dashboard_title: "[Research] Empty Tabs Dashboard"
position:
  ROOT_ID:
    id: ROOT_ID
    type: ROOT
    children: [TABS_ID]
  TABS_ID:
    id: TABS_ID
    type: TABS
    parents: [ROOT_ID]
    children: []
metadata: {}
""".strip()
    )

    document = load_dashboard_yaml_document(fixture)

    assert document.mode == "tabbed"


def test_list_tab_ids_returns_empty_for_flat_dashboard() -> None:
    document = load_dashboard_yaml_document(
        Path("tests/fixtures/workflow/research_yield_simple.yaml")
    )

    assert list_tab_ids(document.position) == []


def test_find_chart_node_ids_returns_chart_nodes() -> None:
    document = load_dashboard_yaml_document(
        Path("tests/fixtures/workflow/research_yield_simple.yaml")
    )

    assert find_chart_node_ids(document.position) == {1384: ["CHART-1384"]}


def test_find_chart_node_ids_recognizes_top_level_chart_id() -> None:
    position = {
        "CHART-1": {
            "id": "CHART-1",
            "type": "CHART",
            "chartId": 101,
            "children": [],
        }
    }

    assert find_chart_node_ids(position) == {101: ["CHART-1"]}


def test_load_layout_preset_reads_flat_two_up() -> None:
    preset = load_layout_preset("flat_two_up")

    assert preset["mode"] == "flat"
    assert preset["charts_per_row"] == 2


def test_load_chart_preset_reads_timeseries_defaults() -> None:
    preset = load_chart_preset("timeseries_default")

    assert preset["x_axis_title_margin"] == 30
    assert preset["y_axis_title_margin"] == 50
    assert preset["legend_orientation"] == "top"


def test_load_chart_preset_reads_pie_defaults() -> None:
    preset = load_chart_preset("pie_default")

    assert preset["legend_orientation"] == "top"
    assert preset["labels_outside"] is True
    assert preset["number_format"] == "SMART_NUMBER"


def test_load_chart_preset_reads_table_defaults() -> None:
    preset = load_chart_preset("table_default")

    assert preset["row_limit"] == 1000
    assert preset["order_by_cols"] == []
    assert preset["order_desc"] is True


def test_plan_add_tab_returns_structural_change_without_mutating_file() -> None:
    fixture = Path("tests/fixtures/workflow/research_yield_simple.yaml")
    original = fixture.read_text(encoding="utf-8")

    plan = plan_add_tab(yaml_path=fixture, tab_name="Sandbox")

    assert plan["status"] == "planned"
    assert plan["target"]["mode"] == "yaml"
    assert any(change["kind"] == "mode_transition" for change in plan["changes"])
    assert "flat" in plan["summary"]
    assert fixture.read_text(encoding="utf-8") == original


def test_plan_add_tab_on_tabbed_dashboard_avoids_fake_mode_transition() -> None:
    fixture = Path("tests/fixtures/workflow/tabbed_dashboard.yaml")

    plan = plan_add_tab(yaml_path=fixture, tab_name="Sandbox")

    assert plan["status"] == "planned"
    assert not any(change["kind"] == "mode_transition" for change in plan["changes"])
    assert plan["changes"] == [{"kind": "add_tab", "tab_name": "Sandbox"}]


def test_apply_workflow_plan_writes_tabbed_yaml(tmp_path: Path) -> None:
    target = tmp_path / "dashboard.yaml"
    fixture = Path("tests/fixtures/workflow/research_yield_simple.yaml")
    target.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    plan = plan_add_tab(yaml_path=target, tab_name="Sandbox")
    result = apply_workflow_plan(plan)

    assert result["status"] == "applied"
    written = target.read_text(encoding="utf-8")
    assert "type: TABS" in written
    assert "text: Sandbox" in written


def test_plan_create_chart_with_preset_keeps_flat_dashboard_flat(tmp_path: Path) -> None:
    target = tmp_path / "dashboard.yaml"
    fixture = Path("tests/fixtures/workflow/research_yield_simple.yaml")
    target.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    plan = plan_create_chart_with_preset(
        yaml_path=target,
        chart_title="New Chart",
        viz_type="table",
        chart_preset="table_default",
    )

    assert not any(change["kind"] == "mode_transition" for change in plan["changes"])


def test_load_layout_preset_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="invalid preset name"):
        load_layout_preset("../secrets")


def test_load_chart_preset_requires_mapping_payload(tmp_path: Path, monkeypatch) -> None:
    preset_root = tmp_path / "presets"
    (preset_root / "charts").mkdir(parents=True)
    (preset_root / "charts" / "broken.yaml").write_text("- not-a-mapping\n", encoding="utf-8")

    monkeypatch.setattr("preset_py.workflow.presets._PRESET_ROOT", preset_root)

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_chart_preset("broken")
