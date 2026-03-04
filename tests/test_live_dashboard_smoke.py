from __future__ import annotations

import json
import os

import pytest

from preset_py import connect
import preset_py.server as server


def test_live_dashboard_template_capture_smoke(tmp_path) -> None:
    """Optional live smoke test for template export + query verification.

    This test is skipped unless PRESET_MCP_ENABLE_LIVE_TESTS=1.
    """
    if os.getenv("PRESET_MCP_ENABLE_LIVE_TESTS") != "1":
        pytest.skip("Set PRESET_MCP_ENABLE_LIVE_TESTS=1 to run live smoke tests.")

    workspace = os.getenv("PRESET_WORKSPACE", "Mysten Labs--General")
    raw_ids = os.getenv("PRESET_MCP_LIVE_DASHBOARD_IDS", "")
    if not raw_ids.strip():
        pytest.skip("Set PRESET_MCP_LIVE_DASHBOARD_IDS (e.g. '80,103,102').")

    dashboard_ids: list[int] = []
    for token in raw_ids.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise AssertionError(f"Invalid dashboard id in PRESET_MCP_LIVE_DASHBOARD_IDS: {token!r}")
        dashboard_ids.append(int(token))
    if not dashboard_ids:
        pytest.skip("No valid dashboard IDs parsed.")

    server._ws = connect(workspace)

    for dashboard_id in dashboard_ids:
        verify_raw = server.verify_dashboard_workflow.fn(
            dashboard_id=dashboard_id,
            include_render=False,
            response_mode="compact",
        )
        verify_payload = json.loads(verify_raw)
        assert verify_payload["query_failures"] == 0

    out_dir = tmp_path / "golden"
    capture_raw = server.capture_golden_templates.fn(
        dashboard_ids=dashboard_ids,
        output_dir=str(out_dir),
        portable=True,
        include_query_context=False,
        include_dataset_schema=True,
        overwrite=True,
        response_mode="standard",
    )
    capture_payload = json.loads(capture_raw)
    assert capture_payload["saved_count"] == len(dashboard_ids)
    for item in capture_payload["saved"]:
        assert os.path.exists(item["output_path"])
