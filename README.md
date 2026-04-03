# preset-mcp

MCP server for [Preset](https://preset.io) (managed Apache Superset). Manage dashboards, charts, and datasets from Claude Code and other LLM agents.

```
Claude Code ──STDIO──> preset-mcp ──> Preset API
```

## Setup for Claude Code

### 1. Get your Preset API credentials

1. Log in to [app.preset.io](https://app.preset.io)
2. Go to **Settings > API Keys**
3. Create a new token/secret pair
4. Copy both the **token** and **secret**

### 2. Install from PyPI

```bash
uv tool install preset-mcp --with preset-cli --with fastmcp --with sqlglot --with pydantic
```

### 3. Register with Claude Code

```bash
claude mcp add --scope user -e PRESET_API_TOKEN=<your-token> \
  -e PRESET_API_SECRET=<your-secret> \
  preset-mcp -- preset-mcp
```

To auto-connect to a specific workspace on startup:

```bash
claude mcp add --scope user -e PRESET_API_TOKEN=<your-token> \
  -e PRESET_API_SECRET=<your-secret> \
  -e PRESET_WORKSPACE="Your Workspace Title" \
  preset-mcp -- preset-mcp
```

### 4. Verify

```bash
claude mcp list
# Should show: preset-mcp  ... 68 tools
```

Then in a Claude Code session, try:

```
> list my preset workspaces
```

### Alternative: Install from source

```bash
git clone https://github.com/Evan-Kim2028/preset-mcp.git
cd preset-mcp
uv sync

claude mcp add --scope user -e PRESET_API_TOKEN=<your-token> \
  -e PRESET_API_SECRET=<your-secret> \
  preset-mcp -- uv run --directory /path/to/preset-mcp preset-mcp
```

## Tools (68)

### Workspace Navigation

| Tool | Purpose |
|------|---------|
| `list_workspaces` | List all workspaces you have access to |
| `use_workspace` | Switch to a workspace by title |

### Read

| Tool | Purpose |
|------|---------|
| `list_dashboards` | List dashboards (with progressive disclosure) |
| `get_dashboard` | Get detail for a single dashboard (supports `response_mode`) |
| `list_charts` | List charts |
| `get_chart` | Get detail for a single chart (supports `response_mode`) |
| `list_datasets` | List datasets |
| `get_dataset` | Get detail for a single dataset (columns, metrics, SQL) |
| `list_databases` | List database connections |
| `get_database` | Get detail for a single database connection |
| `list_logs` | List official audit logs when the workspace exposes them |
| `get_log` | Get detail for a single official audit-log record |
| `get_dashboard_history` | Combine official dashboard audit logs with local MCP history |
| `workspace_catalog` | Relationship-aware topology map |

### Create

| Tool | Purpose |
|------|---------|
| `create_dashboard` | Create a new empty dashboard |
| `create_dataset` | Register a SQL query as a virtual dataset |
| `create_chart` | Build a chart from a dataset |

### Update

| Tool | Purpose |
|------|---------|
| `update_dataset` | Change a dataset's SQL, name, or description |
| `update_chart` | Change a chart's title, viz type, or parameters |
| `update_dashboard` | Rename or publish/unpublish a dashboard |

### Workflow

| Tool | Purpose |
|------|---------|
| `plan_dashboard_changes` | Return a plan-first workflow diff for supported dashboard edits |
| `apply_dashboard_plan` | Apply a previously planned workflow change set |

### Dashboard Lifecycle

| Tool | Purpose |
|------|---------|
| `export_dashboard` | Export a dashboard ZIP bundle for backup or migration |
| `import_dashboard` | Import a dashboard ZIP bundle and report affected dashboard IDs |
| `delete_dashboard` | Delete a dashboard after exporting a backup ZIP |

### SQL & Query

| Tool | Purpose |
|------|---------|
| `run_sql` | Execute a read-only SQL query through Preset's connection |
| `query_dataset` | Query a dataset using Superset's metric/dimension abstraction |

### Validation & Audit

| Tool | Purpose |
|------|---------|
| `validate_chart` | Validate a single chart via chart-data execution |
| `validate_dashboard` | Validate all charts on a dashboard |
| `validate_chart_render` | Validate chart rendering via headless browser probe |
| `validate_dashboard_render` | Validate render status across dashboard charts |
| `verify_chart_workflow` | One-shot chart→dashboard query/render verification |
| `verify_dashboard_structure` | Validate dashboard layout graph and chart references |
| `verify_dashboard_workflow` | One-shot dashboard structure/query/render verification |
| `repair_dashboard_chart_refs` | Repair stale dashboard chart ID references |
| `list_logs` | Read official audit logs from `/api/v1/log/` when available |
| `get_log` | Inspect a single official audit-log record |
| `get_dashboard_history` | Best-effort dashboard history from official logs + local snapshots |
| `list_mutations` | Inspect local mutation audit journal entries |
| `list_dashboard_snapshots` | List local pre-mutation dashboard snapshots |
| `restore_dashboard_snapshot` | Restore dashboard layout/settings from local snapshot |
| `capture_dashboard_template` | Capture reusable dashboard+chart template JSON |
| `capture_golden_templates` | Batch-export templates from dashboard IDs |
| `snapshot_workspace` | Full inventory dump for auditing |

## Typical Workflow

The intended workflow pairs preset-mcp with a data warehouse MCP (like [igloo-mcp](https://github.com/Evan-Kim2028/igloo-mcp) for Snowflake):

```
1. Explore data in Snowflake          (igloo-mcp)
2. Write and validate your SQL         (igloo-mcp)
3. workspace_catalog                   (preset-mcp) — understand what exists
4. list_databases                      (preset-mcp) — find the database_id
5. create_dataset                      (preset-mcp) — register the SQL
6. create_chart + create_dashboard     (preset-mcp) — build the viz
7. update_dataset / update_chart       (preset-mcp) — iterate
```

## Workflow Layer

The workflow layer is additive and plan-first. It does not replace the existing low-level Preset tools.

1. Call `plan_dashboard_changes` or a future convenience `plan_*` workflow helper.
2. Inspect the returned plan payload and structural changes.
3. Call `apply_dashboard_plan` to mutate the target.

Local mode uses exported YAML as the canonical authoring artifact. Flat dashboards remain first-class, and tabs are only created through explicit tab operations or tabbed layout presets.

## Features

### Progressive Disclosure

All list and detail tools accept a `response_mode` parameter to control token usage:

- **`compact`** — IDs and names only (~80% fewer tokens)
- **`standard`** — Key metadata fields (default for list tools)
- **`full`** — Raw API response (default for detail tools)

```
list_dashboards(response_mode="compact")
→ {"count": 42, "data": [{"id": 1, "dashboard_title": "Revenue"}, ...]}

get_dashboard(dashboard_id=80, response_mode="standard")
→ key fields only, no position_json or json_metadata blobs
```

Detail tools (`get_dashboard`, `get_chart`, `get_dataset`, `get_database`) default to `full` for backward compatibility. Use `standard` or `compact` to avoid large payloads — dashboards with 20+ charts can return 50-100K chars in full mode.

### SQL Safety

`run_sql` uses [sqlglot](https://github.com/tobymao/sqlglot) for AST-based validation:

- Blocks write operations (INSERT, UPDATE, DELETE, DROP, ALTER, MERGE, TRUNCATE, GRANT, REVOKE)
- Detects multi-statement injection (`SELECT 1; DROP TABLE x`)
- Handles comment-wrapped bypasses (`-- comment\nDELETE FROM x`)
- Catches CTE-wrapped writes (`WITH x AS (...) DELETE FROM y`)

### Structured Errors

Errors include `error_type` and `hints[]` so the LLM can self-recover:

```json
{
  "error": "No workspace selected.",
  "error_type": "no_workspace",
  "hints": [
    "Call list_workspaces to see available workspaces.",
    "Then call use_workspace('Title') to select one."
  ]
}
```

### Structured Logging

JSON logs on stderr (stdout is reserved for the STDIO transport):

```json
{"ts":"2025-02-11 12:00:00","level":"INFO","msg":"tool=list_dashboards status=ok duration_ms=234"}
```

### Official Audit Logs

`list_logs`, `get_log`, and `get_dashboard_history` wrap Superset's
official `/api/v1/log/` endpoint when the workspace exposes it. Some
Preset deployments return `404 Not found` for that route or restrict it
to specific plans/roles. In those environments:

- `list_logs` / `get_log` return a structured error
- `get_dashboard_history` reports official-log unavailability but still
  returns MCP-local mutation journal and snapshot history when present

## Configuration

All settings are overridable via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PRESET_API_TOKEN` | (required) | Preset API token |
| `PRESET_API_SECRET` | (required) | Preset API secret |
| `PRESET_WORKSPACE` | (optional) | Auto-connect to this workspace |
| `PRESET_MCP_SQL_ROW_LIMIT` | `1000` | Max rows from SQL queries |
| `PRESET_MCP_SQL_SAMPLE_ROWS` | `5` | Rows shown in standard mode |
| `PRESET_MCP_TRUNCATION_THRESHOLD` | `50` | Full-mode truncation cutoff |
| `PRESET_MCP_TRUNCATION_TAIL` | `5` | Tail rows kept when truncating |
| `PRESET_MCP_LOG_LEVEL` | `INFO` | Logging verbosity |

## Python Library

preset-mcp also works as a standalone Python library (no MCP required):

```python
from preset_py import connect

ws = connect("My Workspace")
dashboards = ws.dashboards()
df = ws.run_sql("SELECT * FROM revenue LIMIT 10", database_id=1)

ws.create_dataset("daily_revenue", "SELECT ...", database_id=1)
ws.create_chart(dataset_id=5, title="Revenue", viz_type="echarts_timeseries_bar")
```

## Advanced Recipe: Pie Chart with Ad-hoc Metric

Use `params_json` for advanced chart params such as ad-hoc filters.

```json
{
  "dataset_id": 868,
  "title": "USDSUI Distribution",
  "viz_type": "pie",
  "metrics": "[{\"expressionType\":\"SQL\",\"sqlExpression\":\"AVG(AMOUNT_USD)\",\"label\":\"AVG(AMOUNT_USD)\"}]",
  "groupby": "[\"CATEGORY\",\"SOURCE_NAME\"]",
  "params_json": "{\"adhoc_filters\":[{\"col\":\"TOKEN_SYMBOL\",\"op\":\"==\",\"val\":\"USDSUI\"}]}"
}
```

Notes:
- `create_chart.metrics` accepts saved metric names or ad-hoc metric objects.
- `create_chart.template="auto"` applies viz-specific defaults for missing fields.
- `params_json` is validated preflight against dataset columns/metrics.
- `params_json` cannot include datasource-rebinding keys like `viz_type` or `datasource_id`.
- `create_chart.repair_dashboard_refs` defaults to `false` so chart creation does not mutate dashboard layouts unless explicitly requested.

## Strict Params Semantics

- `update_chart(params_json=...)` uses strict validation semantics and treats `params_json` as a full viz-compatible params payload.
- For viz types with required fields (for example `pie` and timeseries charts), partial payloads like only `{"color_scheme":"..."}` are rejected.
- Use `get_chart(chart_id=<id>, response_mode="full")` to copy/edit the existing params JSON when you need precise updates.

## Golden Template Workflow

Use proven dashboards (for example BTC Fight, Walrus, DeepBook) as template sources:

1. Find dashboard IDs:
```text
list_dashboards(response_mode="compact")
```
2. Verify layout/query/render health before templating:
```text
verify_dashboard_workflow(dashboard_id=<id>, include_render=true, response_mode="standard")
```
3. Export a single reusable template:
```text
capture_dashboard_template(
  dashboard_id=<id>,
  portable=true,
  include_query_context=false,
  include_dataset_schema=true,
  output_path="~/.preset-mcp/golden-templates/<name>.json"
)
```
4. Export multiple dashboards in one run:
```text
capture_golden_templates(
  dashboard_ids="[80,97,162]",
  output_dir="~/.preset-mcp/golden-templates",
  portable=true,
  include_dataset_schema=true
)
```

CLI alternative:

```bash
uv run scripts/export_golden_templates.py \
  --workspace "Mysten Labs--General" \
  --dashboard-ids 80,103,102 \
  --output-dir ~/.preset-mcp/golden-templates \
  --overwrite
```

Optional live smoke test (skipped by default):

```bash
PRESET_MCP_ENABLE_LIVE_TESTS=1 \
PRESET_MCP_LIVE_DASHBOARD_IDS=80,103,102 \
uv run --with pytest pytest -q tests/test_live_dashboard_smoke.py
```

## License

MIT
