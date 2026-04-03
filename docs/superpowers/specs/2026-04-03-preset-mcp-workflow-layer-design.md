# Preset MCP Workflow Layer Design

Date: 2026-04-03
Status: Proposed
Scope: Additive workflow layer for dashboard authoring in `preset-mcp`

## Goal

Add a higher-level workflow layer on top of the existing `preset-mcp` tool surface so dashboard authoring is safer and more ergonomic, especially for:

- tab management
- chart placement and movement
- reusable layout presets
- reusable chart presets
- local/offline editing against exported Preset YAML

The workflow layer must preserve the current low-level tools and safety behavior. It should not replace `create_chart`, `update_chart`, `update_dashboard`, repair tools, or validation tools.

## Problem Statement

`preset-mcp` already has strong low-level primitives:

- chart, dataset, and dashboard CRUD
- layout validation
- repair tools for broken dashboard references
- chart and dashboard validation
- mutation snapshots and audit history
- reusable dashboard template capture

The current gap is authoring ergonomics.

Today:

- tabs are only manipulable through raw `position_json` / `json_metadata`
- chart defaults are helpful but too limited to serve as a reusable preset system
- live editing and local export editing are separate mental models
- common dashboard tasks require understanding Superset layout internals instead of expressing intent

The result is that `preset-mcp` is safe and capable, but cumbersome for higher-level dashboard-building workflows.

## Design Priorities

Priority order:

1. Safety
2. Portability
3. Full dashboard authoring coverage

Additional requirements:

- additive only: existing tool behavior must remain unchanged
- offline/local mode must be first-class
- flat dashboards must work as well as tabbed dashboards
- plan/apply should be mandatory by default for high-level workflow tools

## Non-Goals

This design does not propose:

- replacing the current low-level tool surface
- making tabs mandatory
- inventing a brand new dashboard spec format
- weakening strict validation behavior in existing low-level chart tools
- changing the current return contracts of existing tools

## Current State Summary

The current architecture has three effective layers:

1. Thin client wrappers over Preset/Superset APIs
2. Server-side workflow and safety semantics
3. Safety and audit helpers

Important existing strengths:

- `create_chart` already applies sensible defaults for some chart types
- `update_chart` is intentionally strict about `params_json`
- `update_dashboard` validates layout structure and blocks destructive wipes by default
- repair and verification tools already exist for layout and chart integrity
- snapshots and audit logs already exist for pre-mutation safety and recovery

Important existing weaknesses:

- no first-class tab operations
- no first-class chart/layout preset registry
- no unified live/offline workflow authoring layer
- no intent-based planning surface for common dashboard changes

## Source Of Truth Model

The workflow layer should be YAML-first.

### Canonical authoring artifact

Exported Preset YAML is the canonical offline artifact.

Reasons:

- it matches the files already stored in the repo
- it is easier to review in git than large JSON blobs
- it avoids introducing another local dashboard representation for users to learn

### Minimal normalization

The implementation may use minimal internal normalization as a private helper, but not a heavyweight typed domain model that becomes a second conceptual source of truth.

Minimal normalization should only cover unsafe or repetitive operations such as:

- locating layout nodes
- resolving chart references
- managing `TAB` / `TABS` nodes
- syncing layout and metadata chart references
- expanding chart presets into valid params payloads

This keeps the user-facing model simple while still reducing brittle string-level YAML manipulation inside the implementation.

### Live mode

In live mode, the workflow layer should fetch the current dashboard/chart state from Preset and convert it into a YAML-shaped working document for planning and apply.

This preserves one planning model:

- local YAML path
- live dashboard converted into YAML-shaped working state

## High-Level Architecture

The workflow layer should sit above the current low-level tools.

### Existing low-level tools remain intact

The following stay unchanged and continue to be usable directly:

- chart create/update tools
- dashboard update tools
- repair tools
- verification tools
- snapshot and audit behavior

### New workflow planner

Add a planner/reconciler responsible for:

- loading a target dashboard from YAML or live Preset
- computing a structured plan for requested operations
- validating structural implications before apply
- applying the approved plan through the appropriate adapter

### Adapters

Add two thin adapters:

- `LivePresetAdapter`
  - reads from `PresetWorkspace`
  - applies changes using existing low-level tools and helpers

- `YamlExportAdapter`
  - reads exported dashboard/chart/dataset YAML
  - writes YAML patches back to those files

### Preset registries

Store presets as YAML:

- layout presets
- chart presets

These should live in repo-managed files so they are reviewable and portable.

## Dashboard Mode Invariant

The workflow layer must support both dashboard modes as first-class:

- flat dashboards
- tabbed dashboards

### Rules

- flat dashboards are not a degraded case
- tabs must never be created implicitly
- a chart operation on a flat dashboard must remain flat unless the user explicitly requested tab creation or a tabbed preset
- mode transitions must be explicit in plan output:
  - `flat -> tabbed`
  - `tabbed -> flat`

### Layout preset mode

Each layout preset should declare one of:

- `flat`
- `tabbed`
- `adaptive`

`adaptive` means preserve the current dashboard mode unless the requested workflow explicitly changes it.

## Workflow Surface

V1 should expose a small additive set of plan-first tools.

### Core workflow tools

- `load_dashboard_document`
- `plan_dashboard_changes`
- `apply_dashboard_plan`

These form the foundation. Convenience tools should compile into the same plan shape.

### Convenience planning tools

- `plan_add_tab`
- `plan_rename_tab`
- `plan_remove_tab`
- `plan_reorder_tabs`
- `plan_move_chart_to_tab`
- `plan_move_chart`
- `plan_add_chart_to_dashboard`
- `plan_remove_chart_from_dashboard`
- `plan_apply_layout_preset`
- `plan_create_chart_with_preset`
- `plan_update_chart_with_preset`

These tools should not mutate by default. They should return plans that are later applied through `apply_dashboard_plan`.

## Plan / Apply Model

High-level workflow operations should default to mandatory plan-then-apply.

### Plan

The plan result should include:

- target mode: live or YAML path
- operation summary
- structural changes
- file or resource targets
- warnings
- explicit mode transition notices
- post-apply validation steps

### Apply

`apply_dashboard_plan` should:

- verify the plan still matches the current target state closely enough to apply safely
- execute through the live or YAML adapter
- reuse existing snapshot and audit mechanisms for live mutation
- return an applied summary plus validation results

## Tab Management Semantics

Tabs should become a first-class concept in the workflow layer.

### Add tab

`plan_add_tab` should:

- create `TABS` if explicitly required and missing
- create a `TAB` node
- attach rows under the new tab
- update any affected chart/layout bookkeeping

### Rename tab

`plan_rename_tab` should:

- modify only the tab label metadata
- avoid unrelated layout churn

### Remove tab

`plan_remove_tab` should require a strategy when the tab is not empty:

- block
- move contents to another tab
- flatten contents into the root grid
- remove the tab and prune now-unreachable nodes

### Move chart to tab

`plan_move_chart_to_tab` should:

- locate the chart’s current layout placement
- remove it from the old placement
- insert it into the target tab using layout preset rules
- update chart scope bookkeeping as needed

### Reorder tabs

`plan_reorder_tabs` should only reorder tab children and leave contained rows/charts untouched.

## Layout Presets

Layout presets should define placement behavior and structure rules, not hardcode specific charts.

Expected concerns:

- dashboard mode
- charts per row
- chart width allocation
- row generation behavior
- insertion order
- spacing rules
- empty-tab behavior

### Example

`flat_two_up`

- mode: `flat`
- max charts per row: `2`
- chart width: `6`
- insertion order: left-to-right, top-to-bottom

`tabbed_two_up`

- mode: `tabbed`
- same placement rules as above, but scoped inside each tab

## Chart Presets

Chart presets should reduce repetitive `params_json` authoring for common styles.

They should expand into valid full params payloads before invoking existing chart tools.

### Example starter presets

`timeseries_default`

- x-axis title margin: `30`
- y-axis title margin: `50`
- legend at top
- rich tooltip enabled

`pie_default`

- legend at top
- labels outside
- standard smart number formatting

`table_default`

- standard row limit
- common ordering defaults

### Important constraint

Existing low-level strictness stays in place.

The workflow layer should make common edits easier by generating complete valid chart params, not by weakening strict validation semantics in `update_chart`.

## Local / Offline Workflow

Offline workflow should operate directly on exported YAML.

### Primary behavior

- load dashboard YAML
- compute plan against YAML
- write YAML patches on apply

### Benefits

- reviewable in git
- works before touching live Preset
- supports experimentation in repo-backed “sandbox” dashboards

### Design consequence

Plan output for offline mode should reference YAML file paths and changed sections, not abstract hidden identifiers whenever possible.

## Live Workflow

Live apply should compile into the current low-level tools rather than bypass them.

Examples:

- chart creation flows into `create_chart`
- chart preset updates flow into `update_chart`
- layout changes flow into `update_dashboard`
- post-apply verification reuses existing dashboard/chart validation tools

This preserves the current safety posture and reduces the risk of introducing a second mutation stack.

## Safety Rules

The workflow layer must preserve or improve the current safety guarantees.

### Required rules

- no direct mutation in high-level workflow tools without an explicit apply step
- no implicit tab creation
- no implicit flattening of tabbed dashboards
- no destructive layout wipe without explicit override
- layout and metadata updates must stay synchronized
- plans must surface major structural changes prominently

### Existing safety mechanisms to reuse

- pre-mutation snapshots
- mutation audit journal
- layout validation
- repair tools
- dashboard verification tools

## Compatibility Rules

Compatibility requirements are strict.

- existing tools keep their current signatures and semantics
- existing responses remain unchanged
- the workflow layer is additive only
- low-level tools remain the escape hatch for advanced users

The workflow layer must not create a devx regression for users who already know the current tool surface.

## Testing Strategy

Testing should cover both safety and portability.

### Offline tests

- YAML fixture-based plan generation
- YAML apply roundtrip tests
- no-op stability tests
- flat dashboard fixture coverage
- tabbed dashboard fixture coverage

### Live-like tests

- plan compilation into existing low-level tool calls
- mode transition planning
- chart movement planning
- chart preset expansion into valid full params

### Regression coverage

Preserve and extend coverage around:

- destructive empty-layout wipes
- dangling layout children
- duplicate chart placements
- stale chart references
- datasource preservation during chart updates
- validation timeouts and partial success behavior

## Rollout Plan

### Phase 1

Add workflow planning substrate only:

- YAML loaders/writers
- minimal normalization helpers
- plan generation without mutation

### Phase 2

Add offline/local plan/apply:

- tab operations
- chart movement
- layout presets
- chart presets

### Phase 3

Add live apply through existing primitives:

- compile workflow plan into current chart/dashboard tools
- reuse snapshots, audit, repair, and validation

### Phase 4

Ship initial preset registry:

- `flat_two_up`
- `tabbed_two_up`
- `timeseries_default`
- `pie_default`
- `table_default`

### Phase 5

Add convenience MCP tools over the planner.

## Suggested File Layout

Suggested additions:

- `src/preset_py/workflow/`
- `src/preset_py/workflow/planner.py`
- `src/preset_py/workflow/yaml_adapter.py`
- `src/preset_py/workflow/live_adapter.py`
- `src/preset_py/workflow/layout_ops.py`
- `src/preset_py/workflow/chart_presets.py`
- `src/preset_py/workflow/layout_presets.py`

Suggested tests:

- `tests/test_workflow_yaml.py`
- `tests/test_workflow_live.py`
- `tests/fixtures/workflow/`

Suggested preset storage:

- `src/preset_py/presets/layouts/*.yaml`
- `src/preset_py/presets/charts/*.yaml`

## Success Criteria

V1 is successful if:

- flat dashboards work without any tab assumptions
- tabbed dashboards can be managed without hand-editing raw `position_json`
- offline/local YAML editing is first-class
- live workflow apply reuses existing safety rails
- chart and layout presets materially reduce repetitive parameter authoring
- existing low-level users do not lose any current capability

## Final Recommendation

Proceed with an additive, YAML-first workflow layer built around mandatory plan/apply, first-class support for both flat and tabbed dashboards, and preset-driven dashboard authoring that compiles into the current safe mutation primitives.
