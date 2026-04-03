from pathlib import Path

from preset_py.workflow.document import load_dashboard_yaml_document
from preset_py.workflow.layout_ops import find_chart_node_ids, list_tab_ids


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
