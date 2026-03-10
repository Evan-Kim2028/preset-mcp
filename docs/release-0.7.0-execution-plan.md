# preset-mcp 0.7.0 Release Plan

## Goal

Cut the missing `0.7.0` release from the code already merged to `main`,
including the post-`0.6.0` chart/layout fixes and the follow-up fix for
issue `#30`.

Current state on March 10, 2026:
- `pyproject.toml` already declares `0.7.0`
- latest published GitHub release/tag is `v0.6.0`
- open issue scope for the release is just `#30`

## Release Scope

### Included

1. Pie-chart orderby compatibility fix (`metric` + `metrics`)
2. Hardened `position_json` node validation
3. `describe_dashboard` tool
4. Issue `#30` fix:
   - preserve datasource/viz metadata during strict `update_chart(params_json=...)`
   - return partial-success validation payloads after successful chart mutation

### Deferred

1. PR `#32` (`docs/architecture-review.md`)
   - docs-only; safe to merge independently
2. PR `#33` (viz-spec / validator refactor)
   - large refactor; not required to close the current open issue
3. PR `#34` (snapshot parity expansion)
   - explicitly out of `0.7.0` scope until its failing tests and restore-safety
     regressions are fixed

## Pre-Release Checklist

1. Confirm issue `#30` is fixed on the release branch.
2. Run the full test suite:
   - `uv run pytest -q`
3. Review `CHANGELOG.md` for final wording.
4. Build artifacts locally:
   - `uv build`
5. Smoke-test the wheel or tool install path if release infra changed:
   - `uv tool install --from dist/preset_mcp-0.7.0-py3-none-any.whl preset-mcp`

## Release Steps

1. Merge the release branch to `main`.
2. Close GitHub issue `#30`.
3. Tag the release:
   - `git tag v0.7.0`
   - `git push origin v0.7.0`
4. Publish release notes on GitHub using the `0.7.0` changelog summary.
5. Publish to PyPI if packaging credentials are configured:
   - `uv publish`

## Post-Release Verification

1. Verify GitHub shows `v0.7.0` as the latest release.
2. Confirm `uv tool install preset-mcp` resolves to `0.7.0`.
3. Run one live smoke test against a Preset workspace:
   - create/update a standalone chart
   - verify post-mutation validation output
   - confirm dashboard description still works
