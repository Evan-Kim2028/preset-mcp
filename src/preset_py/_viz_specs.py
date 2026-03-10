"""Viz-type payload specifications for Superset chart types.

Single source of truth for per-viz-type required fields, defaults, and
validation rules.  Consumed by:

- ``client._apply_chart_defaults()`` — to fill in missing fields
- ``_safety.validate_params_payload()`` — to reject incomplete payloads
- Tests — parameterized validation per viz type

Adding a new viz type
---------------------
1. Add a ``VizSpec`` entry to ``VIZ_SPECS``.
2. Existing defaults/validation logic picks it up automatically.
3. Add a parameterized test case in ``test_viz_specs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VizSpec:
    """Specification for a single Superset viz type.

    Parameters
    ----------
    required:
        Field names that *must* be present (non-empty) in the chart params
        for the chart to render.  Use ``"field_a|field_b"`` to express
        "at least one of" (OR-group).
    defaults:
        Static default values applied when the field is absent.
    needs_singular_metric:
        If ``True``, a singular ``metric`` field is derived from the first
        entry in ``metrics`` (needed by Superset builds that derive orderby
        from the singular form — see GitHub issue #26).
    needs_time_column:
        If ``True``, a temporal axis (``granularity_sqla`` or ``time_column``)
        is required.
    needs_dimension:
        If ``True``, a dimension field (``groupby`` or ``columns``) is
        required.
    auto_time_grain:
        When a time column is present and this is set, ``time_grain_sqla``
        is defaulted to this value.
    auto_metrics:
        If ``True``, a default metric is inferred from the dataset when
        ``metrics`` is absent.
    auto_dimension:
        If ``True``, a default dimension column is inferred from the
        dataset when no dimension field is present.
    """

    required: tuple[str, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    needs_singular_metric: bool = False
    needs_time_column: bool = False
    needs_dimension: bool = False
    auto_time_grain: str | None = None
    auto_metrics: bool = True
    auto_dimension: bool = False


# ---- Canonical viz-type registry ----------------------------------------

VIZ_SPECS: dict[str, VizSpec] = {
    "pie": VizSpec(
        required=("metrics", "groupby|columns"),
        defaults={"show_legend": True, "labels_outside": True},
        needs_singular_metric=True,
        needs_dimension=True,
        auto_dimension=True,
    ),
    "echarts_timeseries_bar": VizSpec(
        required=("metrics",),
        needs_time_column=True,
        auto_time_grain="P1D",
    ),
    "echarts_timeseries_line": VizSpec(
        required=("metrics",),
        needs_time_column=True,
        auto_time_grain="P1D",
    ),
    "echarts_timeseries_area": VizSpec(
        required=("metrics",),
        needs_time_column=True,
        auto_time_grain="P1D",
    ),
    "big_number_total": VizSpec(
        required=("metrics",),
    ),
    "table": VizSpec(
        required=("metrics",),
        auto_dimension=True,
    ),
}

TIMESERIES_VIZ_TYPES = frozenset(
    name for name, spec in VIZ_SPECS.items() if spec.needs_time_column
)

# Curated set of viz types known to work in modern Preset workspaces.
# Supplemented at runtime by dynamically-discovered types from charts.
KNOWN_VIZ_TYPES: frozenset[str] = frozenset({
    "area",
    "big_number_total",
    "dist_bar",
    "echarts_area",
    "echarts_bar",
    "echarts_timeseries_area",
    "echarts_timeseries_bar",
    "echarts_timeseries_line",
    "histogram_v2",
    "line",
    "mixed_timeseries",
    "pie",
    "pivot_table_v2",
    "sankey_v2",
    "sunburst_v2",
    "table",
    "treemap_v2",
})

# Legacy/deprecated aliases that should be replaced.
DISCOURAGED_VIZ_TYPES: dict[str, str] = {
    "echarts_bar": "echarts_timeseries_bar",
}


def get_viz_spec(viz_type: str) -> VizSpec | None:
    """Look up the spec for *viz_type*, returning ``None`` if unknown."""
    return VIZ_SPECS.get(viz_type)


# ---- Validation helpers -------------------------------------------------


def check_required_fields(
    viz_type: str,
    params: dict[str, Any],
    *,
    fallback_fields: dict[str, Any] | None = None,
) -> list[str]:
    """Return a list of human-readable error strings for missing required fields.

    ``fallback_fields`` supplies values that the caller (e.g. tool-level
    coercion) has resolved but not yet merged into *params* (e.g.
    ``granularity_sqla`` set from the ``time_column`` tool parameter).
    """
    spec = VIZ_SPECS.get(viz_type)
    if spec is None:
        return []

    fallback = fallback_fields or {}
    errors: list[str] = []

    def _present(key: str) -> bool:
        value = params.get(key) if key in params else fallback.get(key)
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) > 0
        return True

    for field_expr in spec.required:
        if "|" in field_expr:
            alternatives = [f.strip() for f in field_expr.split("|")]
            if not any(_present(alt) for alt in alternatives):
                errors.append(
                    f"{viz_type} requires at least one of "
                    f"{alternatives} in params."
                )
        else:
            if not _present(field_expr):
                errors.append(
                    f"{viz_type} requires '{field_expr}' in params."
                )

    if spec.needs_time_column:
        if not _present("granularity_sqla") and not _present("time_column"):
            errors.append(
                f"{viz_type} requires a time column "
                "(granularity_sqla or time_column)."
            )

    return errors


def validate_chart_envelope(
    payload: dict[str, Any],
    viz_type: str,
) -> list[str]:
    """Validate the outer chart creation/update payload before sending to Superset.

    Checks structural requirements of the API envelope — the fields that
    wrap around the inner ``params`` JSON.  This intentionally does NOT
    re-validate viz-type required fields (that's ``check_required_fields``
    and ``validate_params_payload``'s job) because the caller may be
    constructing params with dataset-dependent defaults that can't always
    satisfy every requirement (e.g. timeseries charts when no temporal
    column exists in the dataset).
    """
    errors: list[str] = []

    if not payload.get("slice_name"):
        errors.append("Chart payload requires a non-empty 'slice_name'.")

    if not payload.get("viz_type"):
        errors.append("Chart payload requires 'viz_type'.")

    if payload.get("datasource_id") is None:
        errors.append("Chart payload requires 'datasource_id'.")

    ds_type = payload.get("datasource_type")
    if ds_type is not None and ds_type != "table":
        errors.append(
            f"Unexpected datasource_type={ds_type!r}; expected 'table'."
        )

    raw_params = payload.get("params")
    if raw_params is not None:
        import json as _json

        if isinstance(raw_params, str):
            try:
                params = _json.loads(raw_params)
            except _json.JSONDecodeError as exc:
                errors.append(f"Chart payload 'params' is not valid JSON: {exc}")
                return errors
        elif isinstance(raw_params, dict):
            params = raw_params
        else:
            errors.append("Chart payload 'params' must be a JSON string or dict.")
            return errors

        # Singular metric consistency — this is a structural invariant that
        # _apply_chart_defaults should always satisfy, unlike dataset-dependent
        # required fields.
        spec = VIZ_SPECS.get(viz_type)
        if spec and spec.needs_singular_metric:
            metrics = params.get("metrics")
            if isinstance(metrics, list) and metrics and not params.get("metric"):
                errors.append(
                    f"{viz_type} requires singular 'metric' alongside 'metrics'. "
                    "Set params.metric to the first entry in params.metrics."
                )

    return errors
