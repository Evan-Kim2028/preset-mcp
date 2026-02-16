# preset-mcp 0.5.0 Cleanup + Reliability Release Plan

## Goal

Ship a single `0.x.0` release that closes all currently open issues in `Evan-Kim2028/preset-mcp`, hardens safety/validation, and improves chart/dashboard reliability for MCP agents.

Target release: `0.5.0` (current PyPI latest observed: `0.4.0`).

## Open Issue Inventory (11)

1. [#2](https://github.com/Evan-Kim2028/preset-mcp/issues/2) `get_chart` read-tool gap
2. [#5](https://github.com/Evan-Kim2028/preset-mcp/issues/5) `get_chart` missing datasource fields in detail response
3. [#6](https://github.com/Evan-Kim2028/preset-mcp/issues/6) list param serialization bug in MCP tool calls
4. [#7](https://github.com/Evan-Kim2028/preset-mcp/issues/7) `create_chart` creates non-rendering charts with minimal params
5. [#8](https://github.com/Evan-Kim2028/preset-mcp/issues/8) dashboard layout refs stale chart IDs
6. [#9](https://github.com/Evan-Kim2028/preset-mcp/issues/9) no `viz_type` validation
7. [#10](https://github.com/Evan-Kim2028/preset-mcp/issues/10) no reliable post-create/update chart validation flow
8. [#11](https://github.com/Evan-Kim2028/preset-mcp/issues/11) critical read-only SQL validation gap on `update_dataset`
9. [#12](https://github.com/Evan-Kim2028/preset-mcp/issues/12) swallowed dataset column refresh failure
10. [#13](https://github.com/Evan-Kim2028/preset-mcp/issues/13) missing FK-style precondition validation
11. [#14](https://github.com/Evan-Kim2028/preset-mcp/issues/14) weak `params_json` validation (metric/viz consistency)

## Branch + Integration Strategy

1. Work branch: `chore/0-x-0-cleanup-master-plan`.
2. First integrate existing local commits (already done locally but not on `origin/main`):
   - `7cf8409` (`v0.3.1` refactor + safety)
   - `0fcefb4` (chart validation tools)
   - `8ac41c5` (`get_chart` tool)
3. Push branch and open one tracking PR for `0.5.0`.
4. Use issue-closing commits referencing `Fixes #...` to auto-close on merge.

## Execution Phases

### Phase 1: Security and Input Validation First (Blocker)

Scope:
- #11, #13, #14, #6, #12

Tasks:
1. Enforce read-only SQL in every SQL mutation path (`create_dataset`, `update_dataset`) and add regression tests for destructive SQL payloads.
2. Add precondition checks before mutations:
   - `create_chart`: verify `dataset_id` and `dashboards` exist.
   - `create_dataset`: verify `database_id` exists.
   - `update_chart`: verify dashboards exist when provided.
3. Strengthen `params_json` validation:
   - disallow `viz_type` in `params_json` (single source of truth)
   - validate metric shape (string metric names or structured metric objects only)
   - validate referenced columns/filters against dataset metadata (warning or error policy defined explicitly)
4. Add robust list coercion for MCP tool args:
   - accept true lists or JSON-stringified lists
   - central parser for `metrics`, `groupby`, `dashboards`, and other list params.
5. Return explicit warning metadata when dataset column refresh fails in `_create_virtual_dataset` (no silent swallow).

Deliverables:
- Central validation/coercion helper module
- Structured warning responses for degraded states
- Unit tests for coercion + validation failures

### Phase 2: Chart Creation Reliability

Scope:
- #7, #9, #10

Tasks:
1. Add `viz_type` guardrail:
   - maintain allowlist or workspace-derived supported set
   - return actionable suggestions on invalid/low-confidence type.
2. Make `create_chart` produce renderable defaults:
   - normalize bare metrics into valid Superset metric objects where possible
   - set required datasource/default params for common viz families
   - fail fast with clear error when required params cannot be inferred.
3. Integrate validation flow:
   - optional `validate_after_create` / `validate_after_update` flag (default true for safer behavior)
   - execute `validate_chart` and include result/warnings in mutation response.

Deliverables:
- Create/update path returns `validation_status` payload
- Known-bad `viz_type` and malformed metric payloads blocked early
- Regression tests for non-rendering chart scenarios

### Phase 3: Dashboard Reference Integrity

Scope:
- #8

Tasks:
1. Add orphan detection in `get_dashboard`:
   - compare `position_json` chart IDs vs actual chart resources
   - return `orphaned_chart_refs` warning block.
2. Add repair utility/tool:
   - `repair_dashboard_chart_refs(dashboard_id, strategy=...)`
   - update `position_json` + `json_metadata.chartsInScope` consistently.
3. Ensure chart-dashboard association updates layout state, not only metadata association.

Deliverables:
- Deterministic dashboard health output
- One-command repair path for stale chart IDs
- Integration tests with recreate/delete chart lifecycle

### Phase 4: Read API Completeness and Consistency

Scope:
- #2, #5

Tasks:
1. Keep `get_chart` as first-class read tool.
2. Backfill datasource fields in `get_chart` detail standard mode:
   - parse from `params` / `query_context` when detail endpoint omits
   - fallback to list endpoint lookup when needed.
3. Ensure `get_chart`, `list_charts`, and standard/detail modes are schema-consistent.

Deliverables:
- Stable `get_chart(response_mode=\"standard\")` datasource fields
- Tests for detail endpoint missing fields and fallback behavior

### Phase 5: QA, Docs, and Release Hardening

Scope:
- Entire release

Tasks:
1. Add `tests/` with pytest coverage for:
   - validation guards
   - list coercion
   - chart creation defaults
   - dashboard reference repair
   - read fallback behavior.
2. Add smoke test script to run against a real test workspace (non-production).
3. Update docs:
   - new/changed tool signatures
   - warnings and validation semantics
   - recommended chart creation workflow.
4. Produce migration notes for any breaking behavior changes.

Deliverables:
- `CHANGELOG` entry for `0.5.0`
- Updated `README.md` and `docs/setup-guide.md`
- Reproducible release checklist

## Definition of Done Per Issue

1. #11: destructive SQL rejected in `update_dataset`; test proves block.
2. #13: invalid resource IDs fail with clear validation errors before mutation.
3. #12: dataset creation response includes warning if column refresh fails.
4. #14: `params_json` rejects conflicting/invalid metric/viz combinations.
5. #6: list args accepted as list or JSON string across affected tools.
6. #9: unsupported `viz_type` blocked or strongly warned with alternatives.
7. #7: minimal chart create path results in renderable chart or explicit error.
8. #10: create/update can run chart validation and return health status.
9. #8: stale dashboard chart refs detected and repairable.
10. #2: `get_chart` available/documented and tested.
11. #5: `get_chart` standard mode includes datasource fields consistently.

## Release Plan (`0.5.0`)

1. Versioning:
   - bump `pyproject.toml` to `0.5.0`
   - update lock/build metadata.
2. Verification:
   - run lint + tests
   - run local MCP smoke flow with real workspace.
3. Build artifacts:
   - `uv build`
   - `uv run twine check dist/*`
4. Publish to PyPI:
   - `uv publish` (token/trusted publishing configured)
5. Post-publish:
   - tag `v0.5.0`
   - push tag
   - publish GitHub release notes
   - close all linked issues with validation evidence.

## Rollout and Risk Controls

1. Roll out in two PRs if needed:
   - PR-A: safety/validation/coercion
   - PR-B: chart/dashboard reliability + repair
2. Keep new strict checks behind explicit errors with actionable hints (avoid silent failures).
3. Validate against one staging workspace before production use.
4. If repair tooling touches dashboard layout JSON, require dry-run preview first.
