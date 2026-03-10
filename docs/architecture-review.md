# Preset MCP — Architecture Review & Root Cause Analysis

**Date:** 2026-03-10
**Scope:** Full repository review — architecture, development history, bug patterns, and recommendations

---

## Executive Summary

This repository has gone through 7 releases in 24 days (Feb 11 → Mar 6), growing from 3,900 lines to ~7,500 lines across 10 source/test files. Each release has followed a **"ship then patch"** cycle where real usage immediately surfaces bugs that destroy charts and dashboards. The recurring failures are not random — they stem from **three systemic root causes**:

1. **The Superset API is a minefield of implicit contracts that the `preset-cli` SDK does not abstract away**
2. **Tests mock the wrong layer**, so bugs pass unit tests but fail against real Preset
3. **The MCP tool surface area grew faster than the validation layer could cover**

---

## Repository Architecture

```
Claude Code (STDIO) → server.py (FastMCP, 34 tools) → client.py (PresetWorkspace) → preset-cli SDK → Superset REST API
                              ↓
                        _safety.py (audit, validation)
                        snapshot.py (workspace snapshots)
```

**Key files:**
| File | Lines | Role |
|------|-------|------|
| `server.py` | 3,755 | MCP tool definitions, input coercion, validation, progressive disclosure |
| `client.py` | 1,627 | PresetWorkspace wrapper, chart/dataset creation helpers, defaults |
| `_safety.py` | 491 | Audit journal, params validation, SQL read-only guard |
| `snapshot.py` | 78 | Workspace inventory snapshots |
| `test_server_tools.py` | 1,032 | Primary test suite (34 tests) |

---

## Development Timeline & Bug Progression

### v0.1.0 (Feb 11) — Initial Ship
- 16 MCP tools, monolithic first commit
- **No tests shipped**
- No input validation beyond basic type checks

### v0.2.0 (Feb 12) — Safety Guardrails
- Added audit journal, pre-mutation snapshots, dry_run mode
- **Critical miss:** `update_dataset` had no SQL read-only validation (security bug)

### v0.3.1 (Feb 12) — Security Fix
- Fixed the SQL injection gap on `update_dataset`
- Consolidated audit boilerplate

### v0.5.0 (Feb 16) — The Big Patch (11 issues)
- Addressed issues #2, #5-#14 in one batch
- Added first real test suite
- Fixed: FK preconditions, JSON array coercion, params validation, orphan detection, chart defaults

### v0.5.2 (Feb 16) — Validation Parity
- `validate_chart` was trusting only the first query result
- Dashboard form_data fallback was broken
- Added frontend render validation (headless browser)

### v0.6.0 (Mar 4) — Dashboard Reliability (8 issues)
- `query_dataset` crashed on non-timeseries datasets (assumed datetime column always exists)
- `update_dashboard` was silently mangling `position_json` (dict → invalid string)
- Empty layout writes could wipe entire dashboards
- Pie charts couldn't use ad-hoc metrics
- Duplicate dimension labels not caught

### v0.7.0 (Mar 6) — Pie + Layout Fixes (3 issues)
- Pie charts had null `orderby` because `metric` (singular) was never set
- Layout nodes missing `id`/`type` fields passed validation

---

## Root Cause Analysis: Why Bugs Keep Destroying Charts and Dashboards

### Root Cause #1: Superset API Has Implicit Contracts That `preset-cli` Doesn't Abstract

The Superset REST API is not a well-documented, stable API. It has **implicit field dependencies** that are nowhere in the docs:

| Bug | Implicit Contract |
|-----|-------------------|
| #26: Pie chart null orderby | `metric` (singular) must be set alongside `metrics` (plural) — Superset derives `orderby` from the singular form |
| #23: query_dataset crash | `preset-cli`'s `get_data()` assumes every dataset has a datetime column |
| #20: validate_chart failure | Chart validation requires `query_context` OR `form_data`, but the fallback chain was incomplete |
| #17: position_json mangling | `position_json` must be a JSON **string**, not a dict — but `json_metadata` can be either |
| #27: Malformed layout nodes | Every layout node must have `id` matching its dict key AND a `type` string — undocumented |

**This is the fundamental problem.** The `preset-cli` SDK provides low-level HTTP client wrappers but does NOT validate payloads against Superset's actual expectations. Every chart type (pie, timeseries, big_number, table, sankey, etc.) has its own undocumented required-field matrix. The MCP server is discovering these requirements one bug at a time through production failures.

### Root Cause #2: Tests Mock the Wrong Layer

Every test in the suite mocks `PresetWorkspace` methods:

```python
class _WS(_WorkspaceBase):
    def create_chart(self, dataset_id, title, viz_type, **kwargs):
        self.create_kwargs = kwargs
        return {"id": 77, "slice_name": title}  # ← Always succeeds
```

This means:
- **Tests never hit the actual Superset API contract** — the mock always returns success
- **Tests never validate the actual payload shape** sent to Superset
- **Tests never catch implicit field dependencies** like `metric` vs `metrics`
- **Tests validate MCP input coercion** (JSON string → list) but NOT output correctness

The live smoke test (`test_live_dashboard_smoke.py`) exists but is opt-in and only covers 2 operations (template capture + query validation). It does NOT cover chart creation, dashboard updates, or the failure modes that keep breaking things.

### Root Cause #3: Reactive Validation — Always One Step Behind

The validation layer has been growing reactively:

| Version | What was validated | What was missed |
|---------|-------------------|-----------------|
| v0.1.0 | SQL read-only | Everything else |
| v0.2.0 | params_json basic shape | update_dataset SQL, FK refs |
| v0.5.0 | FK preconditions, metric shape | viz-specific required fields, layout structure |
| v0.5.2 | Query context fallback | Non-timeseries datasets, ad-hoc metrics |
| v0.6.0 | Layout preflight, viz-axis fields | singular vs plural metric, node id/type |
| v0.7.0 | Node id/type, singular metric | ??? (next bug) |

Each release fixes the last release's blind spots while introducing new surface area with new blind spots. **There is no comprehensive spec** of what a valid chart payload looks like for each viz type.

---

## Specific Code Issues

### 1. `server.py` is a 3,755-line God File

`server.py` contains:
- Tool definitions (34 tools)
- Input coercion logic
- Validation logic
- Progressive disclosure formatting
- Dashboard layout validation
- Template sanitization
- SQL safety checks

This makes it extremely hard to reason about what validation a given tool path goes through. The validation for `create_chart` alone spans ~200 lines across `server.py`, `client.py`, and `_safety.py`.

### 2. `_apply_chart_defaults` Is a Growing Switch Statement

```python
def _apply_chart_defaults(viz_type, dataset, params, *, template="auto"):
    if viz_type == "pie": ...
    if viz_type in {"echarts_timeseries_bar", ...}: ...
    if viz_type == "big_number_total": ...
    if viz_type == "table": ...
```

Every new viz type needs a new branch. Every branch has its own implicit required-field knowledge. This is where bugs like #26 (missing singular `metric`) hide — the pie branch set `metrics` but not `metric`.

### 3. Params Validation Has No Canonical Spec

`_safety.py:validate_params_payload` checks for:
- Forbidden keys (datasource_id, viz_type)
- Metric shape (SIMPLE/SQL expressionType)
- Column references against dataset
- Label collisions
- Viz-specific required fields

But the viz-specific checks are minimal:
```python
if viz_type == "pie":
    if not metrics_list:
        errors.append("Pie chart requires at least one metric in 'metrics'.")
    if not groupby_list and not columns_list:
        errors.append("Pie chart requires at least one dimension ...")
```

There's no validation of `metric` (singular), no validation of `color_scheme` compatibility, no validation of `adhoc_filters` structure, etc.

### 4. `position_json` Handling Is Fragile

Dashboard layout (`position_json`) is the single most dangerous field in the Superset API. It's a nested tree structure where:
- Every node must have `id` matching its dict key
- Every node must have a `type` string
- `ROOT_ID` and `GRID_ID` must exist
- Children must reference existing keys
- `CHART-*` nodes must reference valid chart IDs
- `DASHBOARD_VERSION_KEY` must be `"v2"`

The MCP now validates most of this, but it took 3 releases and 5+ bugs to get here. And it's still possible to create subtly invalid layouts that render incorrectly.

---

## Why the Current Test Strategy Fails

### What Tests Currently Cover (Well)
- MCP input coercion (JSON strings → lists)
- FK precondition checks (invalid database_id, dataset_id, dashboard_id)
- SQL read-only enforcement
- Layout structure validation (dangling children, empty wipes)
- Audit journal read/write
- Progressive disclosure formatting

### What Tests Do NOT Cover (The Gap)
- **Actual Superset API payload correctness** — no test verifies that the payload sent to Superset will actually produce a working chart
- **End-to-end chart creation** — no test creates a chart and then validates it renders
- **Viz-type payload matrices** — no test systematically verifies required fields per viz type
- **Serialization round-trips** — the `position_json` string↔dict issue (#17) was not caught because mocks never serialize
- **Multi-step workflows** — no test creates a dataset → chart → dashboard → validates the whole pipeline

The test strategy is: "mock the workspace, test the MCP layer." But the MCP layer is mostly input coercion and formatting. **The bugs live in the payload construction and API interaction layer**, which is exactly what the mocks skip.

---

## Recommendations

### 1. Build a Viz-Type Payload Spec (Highest Priority)

Create a canonical reference of required fields per viz type, derived from working Superset dashboards:

```python
VIZ_REQUIRED_FIELDS = {
    "pie": {
        "required": ["metrics", "metric", "groupby"],
        "defaults": {"show_legend": True, "labels_outside": True},
    },
    "echarts_timeseries_bar": {
        "required": ["metrics", "granularity_sqla", "time_grain_sqla"],
        "conditional": {"x_axis": "when not using granularity_sqla"},
    },
    ...
}
```

Use this spec to:
- Drive `_apply_chart_defaults` (eliminate ad-hoc if/else chains)
- Drive `validate_params_payload` (systematic, not reactive)
- Drive tests (parameterized tests per viz type)

### 2. Add Integration-Level Tests with Recorded API Fixtures

Instead of mocking `PresetWorkspace`, record actual Superset API responses and replay them:

```python
# Record once from a real workspace:
# response = superset_client.create_resource("chart", **payload)
# save_fixture("create_pie_chart", request=payload, response=response)

def test_create_pie_chart_payload_matches_superset_expectations():
    fixture = load_fixture("create_pie_chart")
    # Verify our payload builder produces the same shape as the recorded request
    actual_payload = _create_chart(mock_client, dataset_id=10, title="Pie", viz_type="pie", ...)
    assert_payload_compatible(actual_payload, fixture["request"])
```

This catches serialization issues, missing fields, and implicit contract violations **without needing a live Superset instance for every test run**.

### 3. Add a Pre-Mutation Payload Validator

Before any `create_resource` or `update_resource` call, validate the payload against the known Superset schema:

```python
def _validate_chart_payload(payload: dict, viz_type: str) -> list[str]:
    """Validate a chart creation/update payload before sending to Superset."""
    errors = []
    params = json.loads(payload.get("params", "{}"))

    spec = VIZ_REQUIRED_FIELDS.get(viz_type, {})
    for field in spec.get("required", []):
        if not params.get(field):
            errors.append(f"viz_type={viz_type} requires '{field}' in params")

    # Validate metric/metrics consistency
    if "metrics" in params and "metric" not in params:
        if viz_type in NEEDS_SINGULAR_METRIC:
            errors.append(f"viz_type={viz_type} requires singular 'metric' alongside 'metrics'")

    return errors
```

### 4. Expand Live Smoke Tests

The existing `test_live_dashboard_smoke.py` is a good start but only covers 2 operations. Expand it to cover the full create→validate→destroy lifecycle for each major viz type:

```python
@pytest.mark.live
@pytest.mark.parametrize("viz_type", ["pie", "echarts_timeseries_bar", "big_number_total", "table"])
def test_chart_lifecycle(preset_workspace, viz_type):
    dataset = preset_workspace.create_dataset(...)
    chart = preset_workspace.create_chart(dataset_id=dataset["id"], viz_type=viz_type, ...)
    validation = preset_workspace.validate_chart_data(chart["id"])
    assert validation["status"] == "success"
    # cleanup
```

### 5. Split `server.py` Into Modules

At 3,755 lines, `server.py` is unsustainable. Suggested split:

```
server/
  __init__.py          # FastMCP app + main()
  tools_read.py        # list/get tools (9 tools)
  tools_create.py      # create tools (3 tools)
  tools_update.py      # update tools (3 tools)
  tools_validate.py    # validation tools (15+ tools)
  tools_query.py       # SQL/query tools (2 tools)
  tools_nav.py         # workspace navigation (2 tools)
  _coercion.py         # input coercion helpers
  _formatting.py       # progressive disclosure
  _layout.py           # position_json validation
```

### 6. Capture Golden Payloads from Working Dashboards

Use the existing `capture_dashboard_template` tool to export working chart configurations, then use these as **test fixtures** and **default templates**:

```python
# From a known-working pie chart in production:
GOLDEN_PIE_PARAMS = {
    "viz_type": "pie",
    "metrics": [{"expressionType": "SIMPLE", ...}],
    "metric": {"expressionType": "SIMPLE", ...},  # <-- singular!
    "groupby": ["CHAIN"],
    "show_legend": True,
    ...
}
```

This converts tribal knowledge ("pie charts need singular `metric`") into tested, version-controlled artifacts.

---

## Summary of Systemic Issues

| Issue | Severity | Status |
|-------|----------|--------|
| No viz-type payload spec | Critical | Not addressed |
| Tests mock wrong layer | Critical | Partially addressed (live smoke test exists but minimal) |
| `preset-cli` doesn't validate payloads | High | Worked around with `_safety.py` |
| `server.py` god file | Medium | Not addressed |
| Reactive validation pattern | High | Ongoing — each release adds checks for last release's bugs |
| No integration-level test fixtures | High | Not addressed |
| Dashboard `position_json` fragility | High | Partially addressed (v0.6.0 + v0.7.0 added checks) |

**The core answer to "why do we keep finding bugs that destroy charts and dashboards":** The Superset API has dozens of undocumented implicit contracts per viz type. The `preset-cli` SDK provides no protection against violating them. The test suite validates MCP input handling but not Superset payload correctness. Each release discovers 3-8 new implicit contracts through production failures and patches them reactively, but the fundamental approach of guessing the API contract instead of testing against it remains unchanged.

The path forward is: **capture what works, spec it, test against it, and validate before every mutation.**
