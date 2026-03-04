# preset-mcp 0.6.0 Reliability + Contract Upgrade Plan

## Goal

Ship a minor release that closes all currently open issues in
`Evan-Kim2028/preset-mcp`, hardens dashboard safety, and upgrades chart/query
contracts so MCP behavior matches real Superset frontend behavior.

Target release: `0.6.0`.

## Open Issue Inventory (8)

1. [#24](https://github.com/Evan-Kim2028/preset-mcp/issues/24) pie chart creation fails for ad-hoc metrics; opaque `orderby` errors; docs gap
2. [#23](https://github.com/Evan-Kim2028/preset-mcp/issues/23) `query_dataset` fails on valid datasets (`SELECT 1`) with generic API errors
3. [#22](https://github.com/Evan-Kim2028/preset-mcp/issues/22) critical dashboard empty-layout wipe + weak recovery visibility
4. [#20](https://github.com/Evan-Kim2028/preset-mcp/issues/20) `validate_chart` fallback to chart params when dashboard context missing
5. [#19](https://github.com/Evan-Kim2028/preset-mcp/issues/19) `validate_chart` only checking first chart-data result
6. [#18](https://github.com/Evan-Kim2028/preset-mcp/issues/18) `update_chart` validation misses viz-axis fields
7. [#17](https://github.com/Evan-Kim2028/preset-mcp/issues/17) `update_dashboard` fails for non-empty layout payloads
8. [#16](https://github.com/Evan-Kim2028/preset-mcp/issues/16) duplicate frontend labels (`x_axis` + `groupby`) not caught preflight

## Triage Notes

- `#19` and `#20` appear implemented in `0.5.2`:
  - `validate_chart_data()` now aggregates all query results.
  - validation now falls back to `chart.params` when dashboard form_data is unavailable.
  - covered by tests in `tests/test_client_validation.py`.
- Action: verify against live workspace once, then close `#19` and `#20` as shipped in `0.5.2`.
- `#16`, `#17`, `#18`, `#22`, `#23`, and `#24` are the active `0.6.0` scope.

## Root Cause Map

| Issue | Probable root cause | Primary files |
|---|---|---|
| #23 | `query_dataset` delegates to `preset-cli` `get_data()` which assumes an auto-detected datetime column and can crash on datasets without one (`time_columns[0]`) | `src/preset_py/client.py`, `src/preset_py/server.py` |
| #24 | `create_chart` tool contract only accepts string metrics, while real workflows require ad-hoc metric objects and richer params; error mapping is too opaque | `src/preset_py/server.py`, `src/preset_py/client.py`, `src/preset_py/_safety.py` |
| #18 | `validate_params_payload()` misses viz-specific axis/series references | `src/preset_py/_safety.py` |
| #16 | no dimension-label collision lint across viz fields (`x_axis`, `groupby`, `columns`, etc.) | `src/preset_py/_safety.py` |
| #17 | dashboard layout update contract is under-validated; weak normalization/diagnostics around `position_json` | `src/preset_py/server.py` |
| #22 | destructive layout writes are now partially blocked, but operators still lack first-class snapshot/audit retrieval tools for rapid recovery | `src/preset_py/_safety.py`, `src/preset_py/server.py` |

## Execution Plan

### Phase 0: Issue Hygiene and Milestone Setup

1. Create milestone `v0.6.0`.
2. Assign all open issues to milestone.
3. Add labels for planning clarity:
   - `area:query`, `area:chart-contract`, `area:validation`,
     `area:dashboard-safety`, `release:v0.6.0`.
4. Post verification comments on `#19` and `#20` and close them after live check.

### Phase 1: Query Path Reliability (`#23`)

1. Stop routing MCP `query_dataset` through `preset-cli` `get_data()` for non-time-series cases.
2. Build chart-data payload directly (same endpoint used by chart validation):
   - no implicit datetime requirement when `is_timeseries=False`
   - explicit, validated handling for metrics, columns, filters, `order_by`.
3. Add error translation for common backend failures (empty API messages, malformed result shape).
4. Add regression tests:
   - dataset with no datetime columns (`SELECT 1 AS x`) succeeds
   - mixed metric types still work
   - invalid metric references fail with actionable errors.

### Phase 2: Chart Contract Upgrade + Frontend-Parity Validation (`#24`, `#18`, `#16`)

1. Expand `create_chart` contract:
   - allow ad-hoc metric objects in addition to metric names
   - add optional richer chart params input (`params_json`) for advanced flows.
2. Expand `validate_params_payload()` coverage to include viz fields:
   - scalar: `x_axis`, `y_axis`, `left_axis`, `right_axis`, `series`,
     `time_column`, `granularity_sqla`
   - list/object shapes where supported.
3. Add label-collision lint across dimensions and metrics:
   - block `x_axis == groupby[i]`
   - block metric labels colliding with dimension labels.
4. Improve semantic `dry_run` for chart mutations:
   - verify references against dataset metadata
   - return remediation hints instead of raw backend `orderby` payload errors.
5. Add docs and examples:
   - explicit pie-chart recipe with ad-hoc metric + filter
   - one-call and two-step workflows (`create_chart` then `update_chart`)
   - forbidden `params_json` keys and correct alternatives.

### Phase 3: Dashboard Mutation Safety and Recovery (`#22`, `#17`)

1. Normalize dashboard mutation inputs:
   - accept dict or JSON string for `position_json` and `json_metadata`
   - normalize and serialize consistently before API calls.
2. Add stronger preflight layout validation:
   - required `ROOT_ID` / `GRID_ID`
   - child/parent reference integrity
   - chart-node references are consistent.
3. Keep existing empty-layout guard and upgrade diagnostics with layout diff summary.
4. Expose operator recovery tools over local audit artifacts:
   - list recent mutations
   - list dashboard snapshots
   - restore dashboard layout from snapshot.
5. Add optional Preset team audit-log reader tool if API access is reliable in all target workspaces.

### Phase 4: QA, Docs, and Release

1. Add/expand tests:
   - `tests/test_client_validation.py` for query + validation parity
   - `tests/test_safety.py` for viz-field and collision detection
   - `tests/test_server_tools.py` for dashboard input normalization/recovery tools.
2. Update docs:
   - `README.md` tool list/count and advanced chart recipes
   - `docs/setup-guide.md` recovery workflow references.
3. Bump version to `0.6.0`.
4. Add `CHANGELOG.md` entry with issue-level mapping.
5. Run full test suite and release checklist before tag/publication.

## Definition of Done

1. `#23`: `query_dataset` works on datasets with no datetime column (including `SELECT 1`) and returns clear errors on invalid inputs.
2. `#24`: pie charts with ad-hoc metrics can be created/updated through MCP without UI fallback.
3. `#18`: viz-axis and related field references are validated before chart update.
4. `#16`: duplicate dimension/metric label collisions are blocked preflight.
5. `#17`: valid non-empty dashboard layouts can be persisted through MCP with actionable failure diagnostics.
6. `#22`: destructive layout writes are guarded and operators can list/restore snapshots via MCP tools.
7. `#19` and `#20`: verified in live workspace and closed with links to shipped tests/commits.

## Risks and Mitigations

- Risk: stricter validation may break permissive legacy workflows.
  - Mitigation: keep explicit warnings for soft checks and reserve hard-fail for deterministic invalid references/collisions.
- Risk: Superset/Preset response shapes vary by workspace version.
  - Mitigation: add defensive parsing and response-shape tests with malformed fixtures.
- Risk: audit-log endpoint availability differs by org permissions.
  - Mitigation: treat team audit API as optional; always support local mutation journal + snapshots.

## Suggested Release Sequence

1. Land `#23` first (query reliability is foundational for tool confidence).
2. Land chart contract + validation parity (`#24`, `#18`, `#16`) second.
3. Land dashboard safety/recovery (`#22`, `#17`) third.
4. Close verified `#19`/`#20`, finalize docs/changelog, tag `0.6.0`.
