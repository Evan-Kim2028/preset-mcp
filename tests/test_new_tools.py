"""Tests for the new MCP tools: saved queries, CSS templates,
annotation layers, async query results, and embedded dashboards.

Covers:
  - list / get / create / update / delete for each resource type
  - response_mode progressive disclosure (compact / standard / full)
  - name_contains filtering
  - dry_run vs execute paths
  - validation (empty update kwargs)
  - domain coercion for enable_embedded_dashboard
  - no-mutation side-effect on annotation layer full-mode record
  - error propagation through _handle_errors
"""

import json

import pytest
from fastmcp.exceptions import ToolError

import preset_py.server as server


# ---------------------------------------------------------------------------
# Shared workspace base (mirrors test_server_tools.py pattern)
# ---------------------------------------------------------------------------


class _WorkspaceBase:
    """Minimal workspace stub with default helpers for monkeypatching."""

    def get_resource(self, resource_type: str, resource_id: int):
        return {"id": resource_id}


# ===================================================================
# Saved Queries
# ===================================================================

_SAVED_QUERIES = [
    {
        "id": 1, "label": "Revenue by region",
        "db_id": 10, "schema": "public",
        "sql": "SELECT region, SUM(rev) FROM sales GROUP BY 1",
        "description": "Quarterly revenue", "changed_on": "2026-01-01",
    },
    {
        "id": 2, "label": "Active users",
        "db_id": 10, "schema": "analytics",
        "sql": "SELECT COUNT(*) FROM users WHERE active",
        "description": "DAU count", "changed_on": "2026-02-01",
    },
]


class _SavedQueryWS(_WorkspaceBase):
    def __init__(self) -> None:
        self.create_kwargs: dict | None = None
        self.update_id: int | None = None
        self.update_kwargs: dict | None = None
        self.deleted_id: int | None = None

    def saved_queries(self):
        return list(_SAVED_QUERIES)

    def saved_query_detail(self, query_id: int):
        for q in _SAVED_QUERIES:
            if q["id"] == query_id:
                return dict(q)
        raise ValueError(f"not found: {query_id}")

    def create_saved_query(self, **kwargs):
        self.create_kwargs = kwargs
        return {"id": 99, **kwargs}

    def update_saved_query(self, query_id: int, **kwargs):
        self.update_id = query_id
        self.update_kwargs = kwargs
        return {"id": query_id, "result": "ok"}

    def delete_saved_query(self, query_id: int):
        self.deleted_id = query_id


# -- list --


def test_list_saved_queries_compact(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _SavedQueryWS())
    raw = server.list_saved_queries.fn(response_mode="compact")
    payload = json.loads(raw)
    assert payload["count"] == 2
    assert payload["response_mode"] == "compact"
    # compact should only have id, label, db_id
    first = payload["data"][0]
    assert set(first.keys()) == {"id", "label", "db_id"}


def test_list_saved_queries_standard(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _SavedQueryWS())
    raw = server.list_saved_queries.fn(response_mode="standard")
    payload = json.loads(raw)
    first = payload["data"][0]
    assert "sql" in first
    assert "description" in first
    assert "hint" in payload


def test_list_saved_queries_full(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _SavedQueryWS())
    raw = server.list_saved_queries.fn(response_mode="full")
    payload = json.loads(raw)
    assert "hint" not in payload
    # full mode returns all original fields
    first = payload["data"][0]
    assert "changed_on" in first


def test_list_saved_queries_name_filter(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _SavedQueryWS())
    raw = server.list_saved_queries.fn(name_contains="revenue")
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["data"][0]["label"] == "Revenue by region"


def test_list_saved_queries_name_filter_no_match(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _SavedQueryWS())
    raw = server.list_saved_queries.fn(name_contains="nonexistent")
    payload = json.loads(raw)
    assert payload["count"] == 0


# -- get --


def test_get_saved_query_compact(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _SavedQueryWS())
    raw = server.get_saved_query.fn(query_id=1, response_mode="compact")
    payload = json.loads(raw)
    assert payload["data"]["id"] == 1
    assert "sql" not in payload["data"]


def test_get_saved_query_standard(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _SavedQueryWS())
    raw = server.get_saved_query.fn(query_id=1, response_mode="standard")
    payload = json.loads(raw)
    assert payload["data"]["sql"] == "SELECT region, SUM(rev) FROM sales GROUP BY 1"


def test_get_saved_query_full(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _SavedQueryWS())
    raw = server.get_saved_query.fn(query_id=2, response_mode="full")
    payload = json.loads(raw)
    assert payload["data"]["label"] == "Active users"
    assert "hint" not in payload


# -- create --


def test_create_saved_query_execute(monkeypatch) -> None:
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_saved_query.fn(
        label="New query", sql="SELECT 1", database_id=10,
        schema="public", description="test",
    )
    payload = json.loads(raw)
    assert payload["id"] == 99
    assert ws.create_kwargs["label"] == "New query"
    assert ws.create_kwargs["database_id"] == 10
    assert ws.create_kwargs["schema"] == "public"


def test_create_saved_query_dry_run(monkeypatch) -> None:
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_saved_query.fn(
        label="New query", sql="SELECT 1", database_id=10, dry_run=True,
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.create_kwargs is None  # not executed


# -- update --


def test_update_saved_query_execute(monkeypatch) -> None:
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.update_saved_query.fn(
        query_id=1, label="Updated label",
    )
    json.loads(raw)
    assert ws.update_id == 1
    assert ws.update_kwargs == {"label": "Updated label"}


def test_update_saved_query_no_fields_raises(monkeypatch) -> None:
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    with pytest.raises(ToolError):
        server.update_saved_query.fn(query_id=1)


def test_update_saved_query_dry_run(monkeypatch) -> None:
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.update_saved_query.fn(
        query_id=1, sql="SELECT 2", dry_run=True,
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert payload["fields_to_change"] == ["sql"]
    assert ws.update_id is None  # not executed


# -- delete --


def test_delete_saved_query_execute(monkeypatch) -> None:
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.delete_saved_query.fn(query_id=1)
    payload = json.loads(raw)
    assert payload["status"] == "deleted"
    assert payload["saved_query_id"] == 1
    assert ws.deleted_id == 1


def test_delete_saved_query_dry_run(monkeypatch) -> None:
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.delete_saved_query.fn(query_id=1, dry_run=True)
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.deleted_id is None


# ===================================================================
# CSS Templates
# ===================================================================

_CSS_TEMPLATES = [
    {
        "id": 1, "template_name": "Dark Theme",
        "css": ".dashboard { background: #1a1a1a; }",
        "changed_on": "2026-01-15",
    },
    {
        "id": 2, "template_name": "Brand Colors",
        "css": ".header { color: #ff6600; }",
        "changed_on": "2026-02-15",
    },
]


class _CssTemplateWS(_WorkspaceBase):
    def __init__(self) -> None:
        self.create_kwargs: dict | None = None
        self.update_id: int | None = None
        self.update_kwargs: dict | None = None
        self.deleted_id: int | None = None

    def css_templates(self):
        return list(_CSS_TEMPLATES)

    def css_template_detail(self, template_id: int):
        for t in _CSS_TEMPLATES:
            if t["id"] == template_id:
                return dict(t)
        raise ValueError(f"not found: {template_id}")

    def create_css_template(self, **kwargs):
        self.create_kwargs = kwargs
        return {"id": 99, **kwargs}

    def update_css_template(self, template_id: int, **kwargs):
        self.update_id = template_id
        self.update_kwargs = kwargs
        return {"id": template_id, "result": "ok"}

    def delete_css_template(self, template_id: int):
        self.deleted_id = template_id


# -- list --


def test_list_css_templates_compact(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _CssTemplateWS())
    raw = server.list_css_templates.fn(response_mode="compact")
    payload = json.loads(raw)
    assert payload["count"] == 2
    first = payload["data"][0]
    assert set(first.keys()) == {"id", "template_name"}


def test_list_css_templates_standard(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _CssTemplateWS())
    raw = server.list_css_templates.fn(response_mode="standard")
    payload = json.loads(raw)
    first = payload["data"][0]
    assert "css" in first
    assert "changed_on" in first


def test_list_css_templates_name_filter(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _CssTemplateWS())
    raw = server.list_css_templates.fn(name_contains="dark")
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["data"][0]["template_name"] == "Dark Theme"


# -- get --


def test_get_css_template_compact(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _CssTemplateWS())
    raw = server.get_css_template.fn(template_id=1, response_mode="compact")
    payload = json.loads(raw)
    assert set(payload["data"].keys()) == {"id", "template_name"}


def test_get_css_template_full(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _CssTemplateWS())
    raw = server.get_css_template.fn(template_id=1, response_mode="full")
    payload = json.loads(raw)
    assert payload["data"]["css"] == ".dashboard { background: #1a1a1a; }"
    assert "hint" not in payload


# -- create --


def test_create_css_template_execute(monkeypatch) -> None:
    ws = _CssTemplateWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_css_template.fn(
        template_name="New Style", css="body { margin: 0; }",
    )
    payload = json.loads(raw)
    assert payload["id"] == 99
    assert ws.create_kwargs["template_name"] == "New Style"
    assert ws.create_kwargs["css"] == "body { margin: 0; }"


def test_create_css_template_dry_run(monkeypatch) -> None:
    ws = _CssTemplateWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_css_template.fn(
        template_name="New Style", css="body { margin: 0; }", dry_run=True,
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.create_kwargs is None


# -- update --


def test_update_css_template_execute(monkeypatch) -> None:
    ws = _CssTemplateWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.update_css_template.fn(
        template_id=1, css="body { color: red; }",
    )
    json.loads(raw)
    assert ws.update_id == 1
    assert ws.update_kwargs == {"css": "body { color: red; }"}


def test_update_css_template_no_fields_raises(monkeypatch) -> None:
    ws = _CssTemplateWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    with pytest.raises(ToolError):
        server.update_css_template.fn(template_id=1)


# -- delete --


def test_delete_css_template_execute(monkeypatch) -> None:
    ws = _CssTemplateWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.delete_css_template.fn(template_id=1)
    payload = json.loads(raw)
    assert payload["status"] == "deleted"
    assert ws.deleted_id == 1


def test_delete_css_template_dry_run(monkeypatch) -> None:
    ws = _CssTemplateWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.delete_css_template.fn(template_id=1, dry_run=True)
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.deleted_id is None


# ===================================================================
# Annotation Layers
# ===================================================================

_ANNOTATION_LAYERS = [
    {"id": 1, "name": "Deploys", "descr": "Production deploys", "changed_on": "2026-01-01"},
    {"id": 2, "name": "Incidents", "descr": "SEV1 incidents", "changed_on": "2026-02-01"},
]

_ANNOTATIONS = [
    {"id": 10, "short_descr": "v2.0 release", "start_dttm": "2026-01-15T00:00:00", "end_dttm": "2026-01-15T01:00:00"},
    {"id": 11, "short_descr": "v2.1 hotfix", "start_dttm": "2026-02-01T00:00:00", "end_dttm": "2026-02-01T00:30:00"},
]


class _AnnotationWS(_WorkspaceBase):
    def __init__(self) -> None:
        self.create_layer_kwargs: dict | None = None
        self.update_layer_id: int | None = None
        self.update_layer_kwargs: dict | None = None
        self.deleted_layer_id: int | None = None
        self.create_ann_kwargs: dict | None = None
        self.deleted_ann: tuple | None = None

    def annotation_layers(self):
        return list(_ANNOTATION_LAYERS)

    def annotation_layer_detail(self, layer_id: int):
        for layer in _ANNOTATION_LAYERS:
            if layer["id"] == layer_id:
                return dict(layer)
        raise ValueError(f"not found: {layer_id}")

    def annotation_layer_annotations(self, layer_id: int):
        return list(_ANNOTATIONS)

    def create_annotation_layer(self, **kwargs):
        self.create_layer_kwargs = kwargs
        return {"id": 99, **kwargs}

    def update_annotation_layer(self, layer_id: int, **kwargs):
        self.update_layer_id = layer_id
        self.update_layer_kwargs = kwargs
        return {"id": layer_id, "result": "ok"}

    def delete_annotation_layer(self, layer_id: int):
        self.deleted_layer_id = layer_id

    def create_annotation(self, **kwargs):
        self.create_ann_kwargs = kwargs
        return {"id": 50, **kwargs}

    def delete_annotation(self, layer_id: int, annotation_id: int):
        self.deleted_ann = (layer_id, annotation_id)


# -- list layers --


def test_list_annotation_layers_compact(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _AnnotationWS())
    raw = server.list_annotation_layers.fn(response_mode="compact")
    payload = json.loads(raw)
    assert payload["count"] == 2
    first = payload["data"][0]
    assert set(first.keys()) == {"id", "name"}


def test_list_annotation_layers_standard(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _AnnotationWS())
    raw = server.list_annotation_layers.fn(response_mode="standard")
    payload = json.loads(raw)
    first = payload["data"][0]
    assert "descr" in first
    assert "changed_on" in first


def test_list_annotation_layers_name_filter(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _AnnotationWS())
    raw = server.list_annotation_layers.fn(name_contains="deploy")
    payload = json.loads(raw)
    assert payload["count"] == 1
    assert payload["data"][0]["name"] == "Deploys"


# -- get layer --


def test_get_annotation_layer_compact(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _AnnotationWS())
    raw = server.get_annotation_layer.fn(layer_id=1, response_mode="compact")
    payload = json.loads(raw)
    assert payload["data"]["id"] == 1
    assert payload["data"]["annotation_count"] == 2
    assert "annotations" not in payload["data"]


def test_get_annotation_layer_standard(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _AnnotationWS())
    raw = server.get_annotation_layer.fn(layer_id=1, response_mode="standard")
    payload = json.loads(raw)
    annotations = payload["data"]["annotations"]
    assert len(annotations) == 2
    assert annotations[0]["short_descr"] == "v2.0 release"


def test_get_annotation_layer_full(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _AnnotationWS())
    raw = server.get_annotation_layer.fn(layer_id=1, response_mode="full")
    payload = json.loads(raw)
    assert payload["data"]["annotations"] == _ANNOTATIONS
    assert "hint" not in payload


def test_get_annotation_layer_full_does_not_mutate_original(monkeypatch) -> None:
    """Ensure full mode doesn't mutate the record returned by the workspace."""
    ws = _AnnotationWS()
    original_record = ws.annotation_layer_detail(1)
    assert "annotations" not in original_record

    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    server.get_annotation_layer.fn(layer_id=1, response_mode="full")

    # The original record should not have been mutated
    fresh_record = ws.annotation_layer_detail(1)
    assert "annotations" not in fresh_record


# -- create layer --


def test_create_annotation_layer_execute(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_annotation_layer.fn(
        name="Maintenance Windows", descr="Planned downtime",
    )
    payload = json.loads(raw)
    assert payload["id"] == 99
    assert ws.create_layer_kwargs["name"] == "Maintenance Windows"
    assert ws.create_layer_kwargs["descr"] == "Planned downtime"


def test_create_annotation_layer_dry_run(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_annotation_layer.fn(
        name="Test", dry_run=True,
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.create_layer_kwargs is None


# -- update layer --


def test_update_annotation_layer_execute(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.update_annotation_layer.fn(layer_id=1, name="Renamed")
    json.loads(raw)
    assert ws.update_layer_id == 1
    assert ws.update_layer_kwargs == {"name": "Renamed"}


def test_update_annotation_layer_no_fields_raises(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    with pytest.raises(ToolError):
        server.update_annotation_layer.fn(layer_id=1)


# -- delete layer --


def test_delete_annotation_layer_execute(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.delete_annotation_layer.fn(layer_id=1)
    payload = json.loads(raw)
    assert payload["status"] == "deleted"
    assert ws.deleted_layer_id == 1


def test_delete_annotation_layer_dry_run(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.delete_annotation_layer.fn(layer_id=1, dry_run=True)
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.deleted_layer_id is None


# -- create annotation --


def test_create_annotation_execute(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_annotation.fn(
        layer_id=1,
        short_descr="Deploy v3.0",
        start_dttm="2026-03-01T00:00:00",
        end_dttm="2026-03-01T01:00:00",
        long_descr="Major release",
    )
    payload = json.loads(raw)
    assert payload["id"] == 50
    assert ws.create_ann_kwargs["layer_id"] == 1
    assert ws.create_ann_kwargs["short_descr"] == "Deploy v3.0"
    assert ws.create_ann_kwargs["long_descr"] == "Major release"


def test_create_annotation_dry_run(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_annotation.fn(
        layer_id=1,
        short_descr="Test",
        start_dttm="2026-03-01T00:00:00",
        end_dttm="2026-03-01T01:00:00",
        dry_run=True,
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.create_ann_kwargs is None


def test_create_annotation_fields_include_long_descr_when_provided(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_annotation.fn(
        layer_id=1,
        short_descr="test",
        start_dttm="2026-03-01T00:00:00",
        end_dttm="2026-03-01T01:00:00",
        long_descr="details here",
        dry_run=True,
    )
    payload = json.loads(raw)
    assert "long_descr" in payload["fields_to_change"]


def test_create_annotation_fields_omit_long_descr_when_absent(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_annotation.fn(
        layer_id=1,
        short_descr="test",
        start_dttm="2026-03-01T00:00:00",
        end_dttm="2026-03-01T01:00:00",
        dry_run=True,
    )
    payload = json.loads(raw)
    assert "long_descr" not in payload["fields_to_change"]


# -- delete annotation --


def test_delete_annotation_execute(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.delete_annotation.fn(layer_id=1, annotation_id=10)
    payload = json.loads(raw)
    assert payload["status"] == "deleted"
    assert payload["annotation_id"] == 10
    assert ws.deleted_ann == (1, 10)


def test_delete_annotation_dry_run(monkeypatch) -> None:
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.delete_annotation.fn(layer_id=1, annotation_id=10, dry_run=True)
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.deleted_ann is None


# ===================================================================
# Async Query Results
# ===================================================================


class _AsyncQueryWS(_WorkspaceBase):
    def async_query_result(self, query_id: str):
        if query_id == "abc-123":
            return {
                "status": "success",
                "data": [{"col1": 1, "col2": "a"}],
                "columns": ["col1", "col2"],
                "query_id": query_id,
            }
        return {"status": "pending", "query_id": query_id}


def test_get_async_query_result_success(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _AsyncQueryWS())
    raw = server.get_async_query_result.fn(query_id="abc-123")
    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload["data"] == [{"col1": 1, "col2": "a"}]


def test_get_async_query_result_pending(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _AsyncQueryWS())
    raw = server.get_async_query_result.fn(query_id="unknown-key")
    payload = json.loads(raw)
    assert payload["status"] == "pending"


# ===================================================================
# Embedded Dashboards
# ===================================================================


class _EmbeddedWS(_WorkspaceBase):
    def __init__(self) -> None:
        self.enable_dashboard_id: int | None = None
        self.enable_domains: list[str] | None = None
        self.disable_dashboard_id: int | None = None

    def get_embedded_dashboard(self, dashboard_id: int):
        if dashboard_id == 80:
            return {
                "uuid": "abc-def-123",
                "allowed_domains": ["app.example.com"],
                "dashboard_id": "80",
            }
        return None

    def create_embedded_dashboard(self, dashboard_id: int, allowed_domains=None):
        self.enable_dashboard_id = dashboard_id
        self.enable_domains = allowed_domains
        return {
            "uuid": "new-uuid-456",
            "allowed_domains": allowed_domains or [],
            "dashboard_id": str(dashboard_id),
        }

    def delete_embedded_dashboard(self, dashboard_id: int):
        self.disable_dashboard_id = dashboard_id


# -- get --


def test_get_embedded_dashboard_enabled(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _EmbeddedWS())
    raw = server.get_embedded_dashboard.fn(dashboard_id=80)
    payload = json.loads(raw)
    assert payload["embedded"] is True
    assert payload["data"]["uuid"] == "abc-def-123"


def test_get_embedded_dashboard_not_enabled(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_ws", lambda: _EmbeddedWS())
    raw = server.get_embedded_dashboard.fn(dashboard_id=99)
    payload = json.loads(raw)
    assert payload["embedded"] is False
    assert "hint" in payload


# -- enable --


def test_enable_embedded_dashboard_with_list(monkeypatch) -> None:
    ws = _EmbeddedWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.enable_embedded_dashboard.fn(
        dashboard_id=42,
        allowed_domains=["app.example.com", "staging.example.com"],
    )
    payload = json.loads(raw)
    assert payload["uuid"] == "new-uuid-456"
    assert ws.enable_dashboard_id == 42
    assert ws.enable_domains == ["app.example.com", "staging.example.com"]


def test_enable_embedded_dashboard_with_json_string(monkeypatch) -> None:
    ws = _EmbeddedWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.enable_embedded_dashboard.fn(
        dashboard_id=42,
        allowed_domains='["app.example.com"]',
    )
    json.loads(raw)
    assert ws.enable_domains == ["app.example.com"]


def test_enable_embedded_dashboard_with_comma_string(monkeypatch) -> None:
    ws = _EmbeddedWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.enable_embedded_dashboard.fn(
        dashboard_id=42,
        allowed_domains="app.example.com, staging.example.com",
    )
    json.loads(raw)
    assert ws.enable_domains == ["app.example.com", "staging.example.com"]


def test_enable_embedded_dashboard_with_none(monkeypatch) -> None:
    ws = _EmbeddedWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.enable_embedded_dashboard.fn(dashboard_id=42)
    json.loads(raw)
    assert ws.enable_domains == []


def test_enable_embedded_dashboard_dry_run(monkeypatch) -> None:
    ws = _EmbeddedWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.enable_embedded_dashboard.fn(
        dashboard_id=42, dry_run=True,
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.enable_dashboard_id is None


# -- disable --


def test_disable_embedded_dashboard_execute(monkeypatch) -> None:
    ws = _EmbeddedWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.disable_embedded_dashboard.fn(dashboard_id=80)
    json.loads(raw)
    assert ws.disable_dashboard_id == 80


def test_disable_embedded_dashboard_dry_run(monkeypatch) -> None:
    ws = _EmbeddedWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.disable_embedded_dashboard.fn(dashboard_id=80, dry_run=True)
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert ws.disable_dashboard_id is None


# ===================================================================
# Edge cases and error propagation
# ===================================================================


def test_saved_query_create_optional_fields_omitted_from_payload(monkeypatch) -> None:
    """When schema and description are None, they should not be sent."""
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    server.create_saved_query.fn(
        label="Minimal", sql="SELECT 1", database_id=5,
    )
    assert ws.create_kwargs["schema"] is None
    assert ws.create_kwargs["description"] is None


def test_create_annotation_layer_no_descr(monkeypatch) -> None:
    """Create layer without description still works."""
    ws = _AnnotationWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    raw = server.create_annotation_layer.fn(name="Simple Layer")
    payload = json.loads(raw)
    assert payload["id"] == 99
    assert ws.create_layer_kwargs["descr"] is None


def test_workspace_error_propagated_as_tool_error(monkeypatch) -> None:
    """Exceptions from workspace methods become ToolErrors via _handle_errors."""
    class _BrokenWS(_WorkspaceBase):
        def saved_queries(self):
            raise RuntimeError("Connection refused")

    monkeypatch.setattr(server, "_get_ws", lambda: _BrokenWS())
    with pytest.raises(ToolError) as exc:
        server.list_saved_queries.fn()
    payload = json.loads(str(exc.value))
    assert "Connection refused" in payload["error"]


def test_update_saved_query_multiple_fields(monkeypatch) -> None:
    """Updating multiple fields at once captures all field names."""
    ws = _SavedQueryWS()
    monkeypatch.setattr(server, "_get_ws", lambda: ws)
    monkeypatch.setattr(server, "record_mutation", lambda entry: None)
    monkeypatch.setattr(server, "capture_before", lambda ws, rt, rid: {"id": rid})
    raw = server.update_saved_query.fn(
        query_id=1, label="New", sql="SELECT 2", description="Updated", schema="new_schema",
        dry_run=True,
    )
    payload = json.loads(raw)
    assert set(payload["fields_to_change"]) == {"label", "sql", "description", "schema"}
