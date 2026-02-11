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

### 2. Install and register (one command)

The simplest setup — `uv run` handles installation automatically:

```bash
claude mcp add --scope user preset-mcp -- \
  uv run --from preset-mcp \
    --with preset-cli --with fastmcp --with sqlglot --with pydantic \
    env PRESET_API_TOKEN=<your-token> PRESET_API_SECRET=<your-secret> \
    preset-mcp
```

To auto-connect to a specific workspace on startup, add the `PRESET_WORKSPACE` variable:

```bash
claude mcp add --scope user preset-mcp -- \
  uv run --from preset-mcp \
    --with preset-cli --with fastmcp --with sqlglot --with pydantic \
    env PRESET_API_TOKEN=<your-token> \
        PRESET_API_SECRET=<your-secret> \
        PRESET_WORKSPACE="Your Workspace Title" \
    preset-mcp
```

### 3. Verify

```bash
claude mcp list
# Should show: preset-mcp  ... 16 tools
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

claude mcp add --scope user preset-mcp -- \
  uv run --directory /path/to/preset-mcp \
    env PRESET_API_TOKEN=<your-token> PRESET_API_SECRET=<your-secret> \
    preset-mcp
```

## Tools (16)

### Workspace Navigation

| Tool | Purpose |
|------|---------|
| `list_workspaces` | List all workspaces you have access to |
| `use_workspace` | Switch to a workspace by title |

### Read

| Tool | Purpose |
|------|---------|
| `list_dashboards` | List dashboards (with progressive disclosure) |
| `get_dashboard` | Get full detail for a single dashboard |
| `list_charts` | List charts |
| `list_datasets` | List datasets |
| `list_databases` | List database connections |
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

### SQL & Audit

| Tool | Purpose |
|------|---------|
| `run_sql` | Execute a read-only query through Preset's connection |
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

## Features

### Progressive Disclosure

All list tools accept a `response_mode` parameter to control token usage:

- **`compact`** — IDs and names only (~80% fewer tokens)
- **`standard`** (default) — Key metadata fields
- **`full`** — Raw API response

```
list_dashboards(response_mode="compact")
→ {"count": 42, "data": [{"id": 1, "dashboard_title": "Revenue"}, ...]}
```

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

## License

MIT
