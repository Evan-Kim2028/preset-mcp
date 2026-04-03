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
