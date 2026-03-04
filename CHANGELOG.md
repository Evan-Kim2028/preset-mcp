# Changelog

## 0.6.0 - 2026-03-04

### Fixed
- `query_dataset` now uses a direct chart-data payload path instead of relying
  on `preset-cli` `get_data()` assumptions that could fail on non-time-series
  datasets (for example `SELECT 1` datasets with no datetime column).
- `create_chart` now accepts ad-hoc metric objects and optional `params_json`
  for richer chart creation workflows (including pie charts with ad-hoc filters).
- Improved chart-data API error translation for opaque `orderby` failures with
  actionable remediation text.
- `update_dashboard` now accepts dict or JSON string payloads for
  `position_json` and `json_metadata`, normalizes them, and preflight-validates
  layout structure (required root/grid nodes + reference integrity).
- `validate_params_payload` now checks viz-axis/dimension fields and fails fast
  on unknown referenced columns.
- Added duplicate-label linting in params validation:
  - duplicate dimension labels across fields (for example `x_axis` + `groupby`)
  - metric/dimension label collisions.

### Added
- Local audit/recovery MCP tools:
  - `list_mutations`
  - `list_dashboard_snapshots`
  - `restore_dashboard_snapshot`
- `verify_chart_workflow` tool for one-shot end-to-end chart/dashboard
  verification (query + optional render checks).
- Golden-template and structure-verification tools:
  - `verify_dashboard_structure`
  - `verify_dashboard_workflow`
  - `capture_dashboard_template`
  - `capture_golden_templates`
- Added `scripts/export_golden_templates.py` for repeatable live template
  harvesting from known dashboards.

### Improved
- `create_chart` now supports `template` mode (`auto` or `minimal`).
  `auto` fills missing, viz-specific defaults to improve first-pass
  chart renderability.
- Viz-aware params validation now enforces required fields for common chart
  types (`pie`, timeseries ECharts, `big_number_total`), reducing incomplete
  chart configs that pass mutation but fail at render time.
- Added portable template sanitization and dashboard-graph integrity analysis
  to improve reuse of proven dashboard patterns (for example BTC Fight,
  Walrus, DeepBook-style templates).
- Added opt-in live smoke test (`tests/test_live_dashboard_smoke.py`) for
  real dashboard verification + template export.

### Quality
- Added regression tests covering:
  - non-time-series `query_dataset` reliability and empty API errors
  - orderby-null error translation in chart validation
  - viz-field validation and label-collision failures
  - ad-hoc metric + `params_json` create-chart path
  - dict-based dashboard layout update serialization and layout preflight checks
  - local mutation journal and dashboard snapshot restore tools

## 0.5.2 - 2026-02-16

### Fixed
- `validate_chart` now prefers saved `query_context` and falls back to
  `chart.params` when dashboard `form_data` is unavailable.
- Validation now aggregates all query results from `/api/v1/chart/data`
  instead of trusting only the first query result.
- Added richer validation diagnostics: `payload_source`,
  `form_data_source`, `query_context_present`, `row_count_total`, and
  per-query statuses.
- Added frontend render validation tools:
  - `validate_chart_render`
  - `validate_dashboard_render`
- Fixed render-validation false failures by injecting `Authorization`
  only for workspace-origin browser requests.
- Added page-error filtering so known non-chart browser noise does not
  mark a chart as broken.

### Quality
- Added client-side tests for:
  - query-context-first validation path
  - chart.params fallback path
  - unsupported path when both query context and form data are missing
  - render page-error classification filtering
- Added server-tool tests for new render validation MCP tools.

### Docs
- Added execution plan for the 0.5.1 release scope in
  `docs/release-0.5.1-execution-plan.md`.

## 0.5.0 - 2026-02-16

### Fixed
- Added strict read-only SQL enforcement coverage for dataset updates.
- Added precondition validation for foreign-key-like IDs:
  - `create_dataset(database_id)`
  - `create_chart(dataset_id, dashboards)`
  - `update_chart(dashboards)`
- Added list-argument coercion for MCP JSON-string arrays on:
  - `create_chart(metrics, groupby, dashboards)`
  - `update_chart(dashboards)`
  - `query_dataset(metrics, columns, order_by)`
- Added stronger `params_json` validation:
  - blocks `viz_type` inside `params_json`
  - validates metric object structure
  - warns for unknown metric/column references against dataset metadata
- Added datasource enrichment fallback in `get_chart` detail responses.
- Added dashboard orphan detection in `get_dashboard`.
- Added `repair_dashboard_chart_refs` tool to repair stale chart references in
  `position_json` and `json_metadata.chartsInScope`.
- Added optional post-mutation chart validation payloads:
  - `create_chart(validate_after_create=True)`
  - `update_chart(validate_after_update=True)`
- Improved chart creation defaults to produce renderable params from minimal
  metric/groupby inputs.
- `create_dataset` now surfaces column refresh failures via response warning
  instead of silently swallowing exceptions.

### Changed
- Default MCP tool count increased to 23.

### Quality
- Added pytest suite covering safety validation, chart creation helpers,
  server coercion/preconditions, datasource enrichment, and dashboard repair.
