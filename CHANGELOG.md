# Changelog

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
