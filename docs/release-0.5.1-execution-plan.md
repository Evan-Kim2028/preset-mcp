# preset-mcp 0.5.1 Validation Parity + Dashboard Reliability Plan

## Goal

Close all current open issues in `Evan-Kim2028/preset-mcp` (`#16`-`#20`) with one patch release that improves frontend-parity validation and dashboard mutation reliability.

Target release: `0.5.1`.

## Open Issue Inventory (5)

1. [#16](https://github.com/Evan-Kim2028/preset-mcp/issues/16) frontend duplicate-label errors not caught (`x_axis` + `groupby`)
2. [#17](https://github.com/Evan-Kim2028/preset-mcp/issues/17) `update_dashboard` fails on non-empty `position_json`
3. [#18](https://github.com/Evan-Kim2028/preset-mcp/issues/18) `params_json` validation misses viz-axis/series fields
4. [#19](https://github.com/Evan-Kim2028/preset-mcp/issues/19) `validate_chart` only checks first chart-data result
5. [#20](https://github.com/Evan-Kim2028/preset-mcp/issues/20) `validate_chart` lacks fallback when dashboard form_data is unavailable

## Root Cause Map

| Issue | Probable root cause | Primary files |
|---|---|---|
| #20 | `chart_form_data()` only scans dashboard charts and returns `(None, None)` when chart is standalone/stale | `src/preset_py/client.py` |
| #19 | `validate_chart_data()` returns only `result[0]` from `/api/v1/chart/data` | `src/preset_py/client.py` |
| #18 | `validate_params_payload()` only extracts columns from `groupby`/`columns`/filters; ignores viz-specific axis/series references | `src/preset_py/_safety.py`, `src/preset_py/server.py` |
| #16 | label-collision validation focuses on metrics and misses dimension/viz-field collisions | `src/preset_py/_safety.py` |
| #17 | `update_dashboard` path does not normalize/serialize JSON-like fields; dict payloads can become invalid non-JSON strings in upstream validation | `src/preset_py/server.py`, `src/preset_py/client.py` |

## Execution Plan

### Phase 1: Chart validation data-source fallback (`#20`)

1. Add chart-level fallback in `PresetWorkspace.chart_form_data()`:
   - when dashboard scan misses, load chart detail (`get_resource("chart", chart_id)`) and parse `params` and/or `query_context` for form-data-like fields.
   - keep explicit `dashboard_id` context behavior, but allow optional fallback to chart-level payload for validation paths.
2. Add helper functions for safe JSON parsing of `params` / `query_context` payloads.
3. Add tests for standalone charts and stale dashboard references.

### Phase 2: Multi-query validation aggregation (`#19`)

1. Change `validate_chart_data()` to evaluate all entries in `body["result"]`.
2. Derive `status` as:
   - `failed` if any query is failed/error,
   - `success` only if all are success,
   - fallback status for mixed/unknown states.
3. Return per-query summaries (`index`, `status`, `error`, `rowcount`) while preserving existing top-level fields for backward compatibility.
4. Add regression tests for mixed success/failure arrays.

### Phase 3: Frontend parity for `params_json` validation (`#16`, `#18`)

1. Expand referenced-column extraction in `validate_params_payload()` to include viz-specific keys:
   - scalar keys like `x_axis`, `y_axis`, `left_axis`, `right_axis`, `series`, `time_column`, `granularity_sqla`
   - list/object variants where applicable.
2. Add dimension+metric label collision checks:
   - fail when labels collide (for example `x_axis="CHAIN"` with `groupby=["CHAIN"]`).
   - fail when metric labels collide with dimension labels.
3. Keep warnings for soft issues, but hard-fail on deterministic invalid references and label collisions.
4. Add tests in `tests/test_safety.py` for:
   - invalid axis columns,
   - `x_axis == groupby[i]`,
   - metric-vs-dimension collisions.

### Phase 4: Dashboard layout update reliability (`#17`)

1. Update `update_dashboard` tool inputs to accept either dicts or JSON strings for `position_json` and `json_metadata`.
2. Normalize inputs with `_ensure_json_dict(...)` and always serialize to valid JSON strings via `json.dumps(...)` before `ws.update_dashboard(...)`.
3. Add preflight validation for minimally valid layout shape:
   - required root/grid nodes,
   - parent/children references are structurally consistent.
4. Improve API-error diagnostics for dashboard updates by surfacing response body excerpts when available.
5. Add tests for:
   - dict payload round-trip serialization,
   - non-empty layout accepted by preflight normalizer,
   - malformed layout rejected before API call.

### Phase 5: QA and release

1. Add/extend tests:
   - `tests/test_client_validation.py` for `#19` and `#20`.
   - `tests/test_safety.py` for `#16` and `#18`.
   - `tests/test_server_tools.py` for `#17` input normalization.
2. Run full test suite and update changelog with concrete fix notes.
3. Release `0.5.1` and close issues with evidence links to tests/commits.

## Definition of Done

1. `#20`: standalone chart validation returns a real validation status, not `unsupported`.
2. `#19`: a mixed result array with any failing query produces a failing overall validation status.
3. `#18`: invalid viz-axis column references are caught before update/create mutation reaches Superset.
4. `#16`: duplicate label scenarios (`x_axis`/`groupby`, metric-vs-dimension) are blocked preflight.
5. `#17`: non-empty dashboard layout payloads are normalized to valid JSON and no longer fail due to serialization mismatch.
