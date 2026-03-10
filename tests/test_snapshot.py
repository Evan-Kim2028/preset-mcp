"""Tests for snapshot.py — WorkspaceSnapshot serialization round-trips."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from preset_py.snapshot import WorkspaceSnapshot, save_snapshot


# ---------------------------------------------------------------------------
# Construction & computed fields
# ---------------------------------------------------------------------------


class TestWorkspaceSnapshotModel:
    def test_empty_snapshot(self) -> None:
        snap = WorkspaceSnapshot(workspace_hostname="example.preset.io")
        assert snap.workspace_hostname == "example.preset.io"
        assert snap.dashboards == []
        assert snap.charts == []
        assert snap.datasets == []
        assert snap.databases == []
        assert snap.counts == {
            "dashboards": 0,
            "charts": 0,
            "datasets": 0,
            "databases": 0,
        }

    def test_counts_reflect_data(self) -> None:
        snap = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            dashboards=[{"id": 1}],
            charts=[{"id": 10}, {"id": 11}],
            datasets=[{"id": 100}],
            databases=[{"id": 200}, {"id": 201}, {"id": 202}],
        )
        assert snap.counts == {
            "dashboards": 1,
            "charts": 2,
            "datasets": 1,
            "databases": 3,
        }

    def test_snapshot_time_is_utc(self) -> None:
        snap = WorkspaceSnapshot(workspace_hostname="ws.preset.io")
        assert snap.snapshot_time.tzinfo == timezone.utc

    def test_explicit_snapshot_time(self) -> None:
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        snap = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            snapshot_time=ts,
        )
        assert snap.snapshot_time == ts


# ---------------------------------------------------------------------------
# JSON serialization round-trips
# ---------------------------------------------------------------------------


class TestSnapshotSerialization:
    def test_model_dump_json_round_trip(self) -> None:
        snap = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            dashboards=[{"id": 1, "title": "Dashboard 1"}],
            charts=[{"id": 10, "viz_type": "pie"}],
            datasets=[{"id": 100, "table_name": "orders"}],
            databases=[{"id": 200, "database_name": "production"}],
        )
        dumped = snap.model_dump_json()
        parsed = json.loads(dumped)

        assert parsed["workspace_hostname"] == "ws.preset.io"
        assert len(parsed["dashboards"]) == 1
        assert len(parsed["charts"]) == 1
        assert len(parsed["datasets"]) == 1
        assert len(parsed["databases"]) == 1
        assert parsed["counts"]["dashboards"] == 1

    def test_model_validate_json_round_trip(self) -> None:
        original = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            dashboards=[{"id": 1}],
            charts=[{"id": 10}],
            datasets=[],
            databases=[],
        )
        dumped = original.model_dump_json()
        restored = WorkspaceSnapshot.model_validate_json(dumped)

        assert restored.workspace_hostname == original.workspace_hostname
        assert restored.dashboards == original.dashboards
        assert restored.charts == original.charts
        assert restored.counts == original.counts

    def test_model_dump_dict_round_trip(self) -> None:
        original = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            charts=[{"id": 1, "slice_name": "Volume"}],
        )
        data = original.model_dump()
        restored = WorkspaceSnapshot.model_validate(data)
        assert restored.charts == original.charts


# ---------------------------------------------------------------------------
# save_snapshot disk I/O
# ---------------------------------------------------------------------------


class TestSaveSnapshot:
    def test_creates_expected_files(self, tmp_path: Path) -> None:
        snap = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            dashboards=[{"id": 1}],
            charts=[{"id": 10}],
            datasets=[{"id": 100}],
            databases=[{"id": 200}],
        )
        result_dir = save_snapshot(snap, tmp_path / "output")
        assert result_dir.exists()
        expected_files = [
            "snapshot_meta.json",
            "dashboards.json",
            "charts.json",
            "datasets.json",
            "databases.json",
        ]
        for filename in expected_files:
            path = result_dir / filename
            assert path.exists(), f"Missing {filename}"
            content = json.loads(path.read_text())
            assert content is not None

    def test_meta_file_contents(self, tmp_path: Path) -> None:
        ts = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        snap = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            snapshot_time=ts,
            dashboards=[{"id": 1}, {"id": 2}],
        )
        save_snapshot(snap, tmp_path)
        meta = json.loads((tmp_path / "snapshot_meta.json").read_text())
        assert meta["workspace_hostname"] == "ws.preset.io"
        assert meta["counts"]["dashboards"] == 2
        assert "2026-03-10" in meta["snapshot_time"]

    def test_data_files_match_snapshot(self, tmp_path: Path) -> None:
        snap = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            dashboards=[{"id": 1, "title": "D1"}],
            charts=[{"id": 10, "viz_type": "pie"}],
            datasets=[],
            databases=[],
        )
        save_snapshot(snap, tmp_path)

        dashboards = json.loads((tmp_path / "dashboards.json").read_text())
        assert dashboards == [{"id": 1, "title": "D1"}]

        charts = json.loads((tmp_path / "charts.json").read_text())
        assert charts == [{"id": 10, "viz_type": "pie"}]

        datasets = json.loads((tmp_path / "datasets.json").read_text())
        assert datasets == []

    def test_nested_output_dir_created(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c"
        snap = WorkspaceSnapshot(workspace_hostname="ws.preset.io")
        result = save_snapshot(snap, deep_path)
        assert result.exists()
        assert (result / "snapshot_meta.json").exists()

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        snap1 = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            dashboards=[{"id": 1}],
        )
        snap2 = WorkspaceSnapshot(
            workspace_hostname="ws.preset.io",
            dashboards=[{"id": 1}, {"id": 2}],
        )
        save_snapshot(snap1, tmp_path)
        save_snapshot(snap2, tmp_path)

        dashboards = json.loads((tmp_path / "dashboards.json").read_text())
        assert len(dashboards) == 2
