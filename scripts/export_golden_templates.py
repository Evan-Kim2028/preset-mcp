#!/usr/bin/env python3
"""Export reusable golden templates from live Preset dashboards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preset_py import connect
import preset_py.server as server


def _parse_dashboard_ids(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid dashboard id: {token!r}")
        values.append(int(token))
    if not values:
        raise ValueError("At least one dashboard id is required.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export golden templates from Preset dashboards."
    )
    parser.add_argument(
        "--workspace",
        default="Mysten Labs--General",
        help="Preset workspace title (default: Mysten Labs--General).",
    )
    parser.add_argument(
        "--dashboard-ids",
        required=True,
        help="Comma-separated dashboard IDs (e.g. 80,103,102).",
    )
    parser.add_argument(
        "--output-dir",
        default="~/.preset-mcp/golden-templates",
        help="Directory for exported templates.",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        default=True,
        help="Strip datasource/runtime-specific keys for portability (default: true).",
    )
    parser.add_argument(
        "--include-query-context",
        action="store_true",
        default=False,
        help="Include chart query_context in templates.",
    )
    parser.add_argument(
        "--include-dataset-schema",
        action="store_true",
        default=True,
        help="Include dataset column/metric names in template specs (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    dashboard_ids = _parse_dashboard_ids(args.dashboard_ids)

    # Reuse server tool implementation so script and MCP path stay identical.
    server._ws = connect(args.workspace)
    raw = server.capture_golden_templates.fn(
        dashboard_ids=dashboard_ids,
        output_dir=str(Path(args.output_dir).expanduser()),
        portable=args.portable,
        include_query_context=args.include_query_context,
        include_dataset_schema=args.include_dataset_schema,
        overwrite=args.overwrite,
        response_mode="standard",
    )
    payload = json.loads(raw)
    print(json.dumps(payload, indent=2, default=str))

    status = payload.get("status")
    if status == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
