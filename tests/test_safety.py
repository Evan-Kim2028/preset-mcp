import pytest

from preset_py._safety import validate_params_payload


def test_params_payload_blocks_viz_type_override() -> None:
    with pytest.raises(ValueError, match="viz_type"):
        validate_params_payload('{"viz_type":"table"}')


def test_params_payload_rejects_invalid_metric_shape() -> None:
    with pytest.raises(ValueError, match="aggregate"):
        validate_params_payload(
            '{"metrics":[{"expressionType":"SIMPLE","column":{"column_name":"value"}}]}'
        )


def test_params_payload_rejects_unknown_columns() -> None:
    with pytest.raises(ValueError, match="unknown dataset columns"):
        validate_params_payload(
            '{"groupby":["known_col"],"filters":[{"col":"missing_col"}]}',
            dataset_columns={"known_col"},
            dataset_metrics={"saved_metric"},
        )


def test_params_payload_rejects_unknown_axis_column() -> None:
    with pytest.raises(ValueError, match="unknown dataset columns"):
        validate_params_payload(
            '{"x_axis":"BAD_COLUMN","groupby":["known_col"]}',
            dataset_columns={"known_col"},
            dataset_metrics={"saved_metric"},
        )


def test_params_payload_rejects_duplicate_dimension_labels() -> None:
    with pytest.raises(ValueError, match="duplicate dimension labels"):
        validate_params_payload(
            '{"x_axis":"CHAIN","groupby":["CHAIN"]}',
            dataset_columns={"CHAIN"},
            dataset_metrics={"count"},
        )


def test_params_payload_rejects_metric_dimension_label_collision() -> None:
    with pytest.raises(ValueError, match="metric labels collide"):
        validate_params_payload(
            '{"groupby":["CHAIN"],"metrics":["CHAIN"]}',
            dataset_columns={"CHAIN"},
            dataset_metrics={"count"},
        )


def test_params_payload_enforces_pie_required_fields() -> None:
    with pytest.raises(ValueError, match="Pie charts require params_json.metrics"):
        validate_params_payload(
            '{"groupby":["CHAIN"]}',
            dataset_columns={"CHAIN"},
            dataset_metrics={"count"},
            viz_type="pie",
        )

    with pytest.raises(ValueError, match="Pie charts require params_json.groupby"):
        validate_params_payload(
            '{"metrics":["count"]}',
            dataset_columns={"CHAIN"},
            dataset_metrics={"count"},
            viz_type="pie",
        )


def test_params_payload_pie_auto_sets_singular_metric() -> None:
    """Pie chart params_json should auto-derive metric from metrics."""
    parsed, warnings = validate_params_payload(
        '{"metrics":["count"],"groupby":["CHAIN"]}',
        dataset_columns={"CHAIN"},
        dataset_metrics={"count"},
        viz_type="pie",
    )
    assert parsed["metric"] == "count"
    assert any("Auto-set" in w for w in warnings)


def test_params_payload_pie_preserves_explicit_metric() -> None:
    """If metric is already set, don't override it."""
    parsed, warnings = validate_params_payload(
        '{"metrics":["count"],"metric":"explicit_metric","groupby":["CHAIN"]}',
        dataset_columns={"CHAIN"},
        dataset_metrics={"count", "explicit_metric"},
        viz_type="pie",
    )
    assert parsed["metric"] == "explicit_metric"
    assert not any("Auto-set" in w for w in warnings)


def test_params_payload_enforces_timeseries_required_fields() -> None:
    with pytest.raises(ValueError, match="requires params_json.metrics"):
        validate_params_payload(
            '{"granularity_sqla":"DAY"}',
            dataset_columns={"DAY"},
            dataset_metrics={"count"},
            viz_type="echarts_timeseries_line",
        )

    with pytest.raises(ValueError, match="requires a time column"):
        validate_params_payload(
            '{"metrics":["count"]}',
            dataset_columns={"DAY"},
            dataset_metrics={"count"},
            viz_type="echarts_timeseries_line",
        )


# ===================================================================
# Additional viz-type validation tests
# ===================================================================


def test_params_payload_enforces_big_number_metrics() -> None:
    with pytest.raises(ValueError, match="big_number requires"):
        validate_params_payload('{}', viz_type="big_number")


def test_params_payload_enforces_table_fields() -> None:
    with pytest.raises(ValueError, match="table requires"):
        validate_params_payload('{}', viz_type="table")

    # Should pass if metrics OR groupby is present
    parsed, _ = validate_params_payload(
        '{"metrics":["count"]}',
        dataset_metrics={"count"},
        viz_type="table",
    )
    assert "metrics" in parsed

    parsed2, _ = validate_params_payload(
        '{"groupby":["region"]}',
        dataset_columns={"region"},
        viz_type="table",
    )
    assert "groupby" in parsed2


def test_params_payload_enforces_heatmap_fields() -> None:
    with pytest.raises(ValueError, match="heatmap requires params_json.metrics"):
        validate_params_payload(
            '{"x_axis":"col1","y_axis":"col2"}',
            dataset_columns={"col1", "col2"},
            viz_type="heatmap",
        )

    with pytest.raises(ValueError, match="heatmap requires params_json.x_axis"):
        validate_params_payload(
            '{"metrics":["count"],"y_axis":"col2"}',
            dataset_columns={"col2"},
            dataset_metrics={"count"},
            viz_type="heatmap",
        )

    with pytest.raises(ValueError, match="heatmap requires params_json.y_axis"):
        validate_params_payload(
            '{"metrics":["count"],"x_axis":"col1"}',
            dataset_columns={"col1"},
            dataset_metrics={"count"},
            viz_type="heatmap",
        )


def test_params_payload_enforces_bar_chart_fields() -> None:
    with pytest.raises(ValueError, match="dist_bar requires params_json.metrics"):
        validate_params_payload(
            '{"groupby":["region"]}',
            dataset_columns={"region"},
            viz_type="dist_bar",
        )

    with pytest.raises(ValueError, match="dist_bar requires params_json.groupby"):
        validate_params_payload(
            '{"metrics":["count"]}',
            dataset_metrics={"count"},
            viz_type="dist_bar",
        )


def test_params_payload_enforces_treemap_fields() -> None:
    with pytest.raises(ValueError, match="treemap requires params_json.metrics"):
        validate_params_payload('{"groupby":["r"]}', dataset_columns={"r"}, viz_type="treemap")

    with pytest.raises(ValueError, match="treemap requires params_json.groupby"):
        validate_params_payload('{"metrics":["c"]}', dataset_metrics={"c"}, viz_type="treemap")


def test_params_payload_enforces_sunburst_fields() -> None:
    with pytest.raises(ValueError, match="sunburst requires params_json.metrics"):
        validate_params_payload('{"groupby":["r"]}', dataset_columns={"r"}, viz_type="sunburst")

    with pytest.raises(ValueError, match="sunburst requires params_json.groupby"):
        validate_params_payload('{"metrics":["c"]}', dataset_metrics={"c"}, viz_type="sunburst")


def test_params_payload_enforces_funnel_fields() -> None:
    with pytest.raises(ValueError, match="funnel requires params_json.metrics"):
        validate_params_payload('{"groupby":["r"]}', dataset_columns={"r"}, viz_type="funnel")

    with pytest.raises(ValueError, match="funnel requires params_json.groupby"):
        validate_params_payload('{"metrics":["c"]}', dataset_metrics={"c"}, viz_type="funnel")


def test_params_payload_enforces_gauge_metrics() -> None:
    with pytest.raises(ValueError, match="gauge_chart requires"):
        validate_params_payload('{}', viz_type="gauge_chart")


def test_params_payload_enforces_bubble_fields() -> None:
    with pytest.raises(ValueError, match="bubble requires params_json.x"):
        validate_params_payload('{"y":"col","size":"col2"}', viz_type="bubble")

    with pytest.raises(ValueError, match="bubble requires params_json.y"):
        validate_params_payload('{"x":"col","size":"col2"}', viz_type="bubble")

    with pytest.raises(ValueError, match="bubble requires params_json.size"):
        validate_params_payload('{"x":"col","y":"col2"}', viz_type="bubble")


def test_params_payload_enforces_box_plot_fields() -> None:
    with pytest.raises(ValueError, match="box_plot requires params_json.metrics"):
        validate_params_payload('{"groupby":["r"]}', dataset_columns={"r"}, viz_type="box_plot")

    with pytest.raises(ValueError, match="box_plot requires params_json.groupby"):
        validate_params_payload('{"metrics":["c"]}', dataset_metrics={"c"}, viz_type="box_plot")


# ===================================================================
# Safety module unit tests (record_mutation, capture_before, prune)
# ===================================================================


def test_record_mutation_writes_jsonl(tmp_path, monkeypatch) -> None:
    import preset_py._safety as safety

    monkeypatch.setattr(safety, "AUDIT_DIR", tmp_path)

    from preset_py._safety import MutationEntry, record_mutation

    entry = MutationEntry(
        tool_name="update_chart",
        resource_type="chart",
        resource_id=42,
        action="update",
        fields_changed=["params"],
    )
    record_mutation(entry)

    journal = tmp_path / "mutations.jsonl"
    assert journal.exists()
    lines = [l for l in journal.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    import json
    parsed = json.loads(lines[0])
    assert parsed["tool_name"] == "update_chart"
    assert parsed["resource_id"] == 42


def test_record_mutation_creates_directory(tmp_path, monkeypatch) -> None:
    import preset_py._safety as safety

    audit_dir = tmp_path / "nested" / "audit"
    monkeypatch.setattr(safety, "AUDIT_DIR", audit_dir)

    from preset_py._safety import MutationEntry, record_mutation

    entry = MutationEntry(
        tool_name="create_dashboard",
        resource_type="dashboard",
        action="create",
    )
    record_mutation(entry)
    assert (audit_dir / "mutations.jsonl").exists()


def test_prune_old_snapshots_removes_expired(tmp_path, monkeypatch) -> None:
    import os
    import preset_py._safety as safety
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(safety, "AUDIT_DIR", tmp_path)

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()

    old = snap_dir / "chart_1_old.json"
    old.write_text("{}")
    old_time = (datetime.now(timezone.utc) - timedelta(days=60)).timestamp()
    os.utime(old, (old_time, old_time))

    recent = snap_dir / "chart_2_recent.json"
    recent.write_text("{}")

    from preset_py._safety import prune_old_snapshots
    removed = prune_old_snapshots(max_age_days=30)
    assert removed == 1
    assert not old.exists()
    assert recent.exists()


def test_prune_old_snapshots_no_directory(tmp_path, monkeypatch) -> None:
    import preset_py._safety as safety

    monkeypatch.setattr(safety, "AUDIT_DIR", tmp_path)

    from preset_py._safety import prune_old_snapshots
    removed = prune_old_snapshots(max_age_days=30)
    assert removed == 0
