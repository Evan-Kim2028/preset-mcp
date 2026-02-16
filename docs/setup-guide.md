# preset-mcp: End-to-End Setup Guide

Complete walkthrough for setting up preset-mcp with Claude Code — from creating API credentials to your first tool call.

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed
- [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A [Preset](https://preset.io) account with at least one workspace

## Step 1: Create Preset API Credentials

1. Open [app.preset.io](https://app.preset.io) and log in
2. Click your profile icon (bottom-left) > **Settings**
3. Navigate to **API Keys**
4. Click **Create API Key**
5. Give it a name (e.g. `claude-code`)
6. Copy both values:
   - **API Token** — looks like `abc123def456...`
   - **API Secret** — looks like `xyz789...`

> Keep these safe. You'll need both in the next step.

## Step 2: Install preset-mcp from PyPI

```bash
uv tool install preset-mcp \
  --with preset-cli \
  --with fastmcp \
  --with sqlglot \
  --with pydantic
```

This installs `preset-mcp` as a standalone CLI tool. Verify it installed:

```bash
which preset-mcp
# Should print: ~/.local/bin/preset-mcp (or similar)
```

## Step 3: Register with Claude Code

Run this command, replacing `<your-token>` and `<your-secret>` with the values from Step 1:

```bash
claude mcp add --scope user \
  -e PRESET_API_TOKEN=<your-token> \
  -e PRESET_API_SECRET=<your-secret> \
  preset-mcp -- preset-mcp
```

### Auto-connect to a workspace (recommended)

If you want to skip the `use_workspace` step every session, specify your workspace title:

```bash
claude mcp add --scope user \
  -e PRESET_API_TOKEN=<your-token> \
  -e PRESET_API_SECRET=<your-secret> \
  -e PRESET_WORKSPACE="Your Workspace Title" \
  preset-mcp -- preset-mcp
```

> To find your workspace title, log in to [app.preset.io](https://app.preset.io) — it's shown at the top of the workspace selector.

## Step 4: Verify the Connection

Restart Claude Code (or open a new session), then check:

```bash
claude mcp list
```

You should see:

```
preset-mcp · 23 tools
```

If it shows `failed`, check:
- Your API token/secret are correct (no extra spaces or quotes)
- `preset-mcp` is on your PATH (`which preset-mcp`)
- Try running `preset-mcp` directly in your terminal to see error output

## Step 5: Test It

In a Claude Code session, try these prompts:

```
> list my preset workspaces
```

```
> show me all dashboards in compact mode
```

```
> what databases are connected to this workspace?
```

If you set `PRESET_WORKSPACE`, you'll get results immediately. Otherwise, Claude will call `list_workspaces` first, then `use_workspace` to connect.

## What You Can Do

Once connected, you can ask Claude to:

| Task | Example prompt |
|------|----------------|
| Browse dashboards | "list all published dashboards" |
| Inspect a dashboard | "get details for dashboard 80" |
| See what exists | "show me the workspace catalog" |
| Create a dataset | "create a dataset from this SQL: SELECT ..." |
| Build a chart | "create a bar chart from dataset 5" |
| Create a dashboard | "create a new dashboard called Revenue" |
| Update things | "rename dashboard 80 to Deepbook V3 Analytics" |
| Run diagnostic SQL | "run this query on database 3: SELECT count(*) FROM ..." |

## Typical Data Workflow

preset-mcp is designed to pair with a data warehouse MCP like [igloo-mcp](https://github.com/Evan-Kim2028/igloo-mcp) (Snowflake):

```
 igloo-mcp (Snowflake)              preset-mcp (Preset)
 ─────────────────────              ───────────────────
 1. Explore tables/schemas
 2. Write & validate SQL
 3. Finalize the query
                          ──────>   4. workspace_catalog  (see what exists)
                          ──────>   5. list_databases     (find database_id)
                          ──────>   6. create_dataset     (register the SQL)
                          ──────>   7. create_chart       (build the viz)
                          ──────>   8. create_dashboard   (group charts)
                          ──────>   9. update_* tools     (iterate)
```

## Updating preset-mcp

To upgrade to a newer version:

```bash
uv tool upgrade preset-mcp
```

No need to re-register with Claude Code — the binary path stays the same.

## Uninstalling

```bash
claude mcp remove --scope user preset-mcp
uv tool uninstall preset-mcp
```

## Troubleshooting

### "failed" status in `claude mcp list`

1. Run `preset-mcp` directly in your terminal to see the error
2. Most common cause: missing or invalid API credentials
3. Re-add with correct credentials:
   ```bash
   claude mcp remove --scope user preset-mcp
   claude mcp add --scope user \
     -e PRESET_API_TOKEN=<your-token> \
     -e PRESET_API_SECRET=<your-secret> \
     preset-mcp -- preset-mcp
   ```

### "No workspace selected" errors

Either set `PRESET_WORKSPACE` in your config (Step 3) or tell Claude:
```
> use workspace "Your Workspace Title"
```

### Tools not appearing

Restart Claude Code after adding the MCP. Check `claude mcp list` shows 23 tools.
