from pathlib import Path

from preset_py.workflow.document import load_dashboard_yaml_document


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
