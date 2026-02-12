"""Core wrapper: connect() entry point and PresetWorkspace convenience class."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from yarl import URL

from preset_cli.auth.lib import get_access_token
from preset_cli.auth.jwt import JWTAuth
from preset_cli.api.clients.preset import PresetClient
from preset_cli.api.clients.superset import SupersetClient

from preset_py._helpers import create_chart as _create_chart
from preset_py._helpers import create_virtual_dataset
from preset_py.snapshot import WorkspaceSnapshot, take_snapshot

PRESET_API_URL = "https://api.app.preset.io/"


def connect(workspace: str | None = None) -> PresetWorkspace:
    """One-liner entry point. Reads ``PRESET_API_TOKEN`` / ``PRESET_API_SECRET``
    from the environment and returns a :class:`PresetWorkspace`.

    Parameters
    ----------
    workspace:
        Workspace title to connect to (e.g. ``"Mysten Labs--General"``).
        If *None*, the returned object can still list workspaces and then
        switch into one via :meth:`PresetWorkspace.use`.
    """
    api_token = os.environ["PRESET_API_TOKEN"]
    api_secret = os.environ["PRESET_API_SECRET"]

    jwt = get_access_token(PRESET_API_URL, api_token, api_secret)
    auth = JWTAuth(jwt)

    preset_client = PresetClient(PRESET_API_URL, auth)

    # Cache the workspace list (needed for title → hostname resolution)
    teams = preset_client.get_teams()
    all_workspaces: list[dict[str, Any]] = []
    for team in teams:
        all_workspaces.extend(preset_client.get_workspaces(team["name"]))

    workspace_url: URL | None = None
    if workspace:
        workspace_url = _resolve_workspace(all_workspaces, workspace)

    return PresetWorkspace(
        auth=auth,
        preset_client=preset_client,
        workspace_url=workspace_url,
        _workspaces=all_workspaces,
    )


def _resolve_workspace(workspaces: list[dict[str, Any]], title: str) -> URL:
    """Find workspace by title and return its URL."""
    for ws in workspaces:
        if ws.get("title") == title:
            hostname = ws["hostname"]
            return URL(f"https://{hostname}/")
    available = [ws.get("title", ws.get("hostname", "?")) for ws in workspaces]
    raise ValueError(
        f"Workspace {title!r} not found. Available: {available}"
    )


class PresetWorkspace:
    """Convenience wrapper around PresetClient + SupersetClient.

    All CRUD methods return plain Python dicts/lists/DataFrames so that
    callers never need to touch the underlying SDK objects.
    """

    def __init__(
        self,
        auth: JWTAuth,
        preset_client: PresetClient,
        workspace_url: URL | None = None,
        _workspaces: list[dict[str, Any]] | None = None,
    ) -> None:
        self._auth = auth
        self._preset_client = preset_client
        self._workspaces = _workspaces or []
        self.workspace_url = workspace_url

        self._superset: SupersetClient | None = None
        if workspace_url:
            self._superset = SupersetClient(str(workspace_url), auth)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @property
    def _client(self) -> SupersetClient:
        if self._superset is None:
            raise RuntimeError(
                "No workspace selected. Call .use('Workspace Title') first, "
                "or pass a workspace name to connect()."
            )
        return self._superset

    # ------------------------------------------------------------------
    # Workspace navigation
    # ------------------------------------------------------------------

    def list_workspaces(self) -> list[dict[str, Any]]:
        """Return cached list of workspaces (title + hostname)."""
        return [
            {"title": ws.get("title"), "hostname": ws.get("hostname")}
            for ws in self._workspaces
        ]

    def use(self, workspace_title: str) -> PresetWorkspace:
        """Switch to a different workspace and return a new PresetWorkspace."""
        url = _resolve_workspace(self._workspaces, workspace_title)
        return PresetWorkspace(
            auth=self._auth,
            preset_client=self._preset_client,
            workspace_url=url,
            _workspaces=self._workspaces,
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def dashboards(self, **filters: Any) -> list[dict[str, Any]]:
        return self._client.get_dashboards(**filters)

    def charts(self, **filters: Any) -> list[dict[str, Any]]:
        return self._client.get_charts(**filters)

    def datasets(self, **filters: Any) -> list[dict[str, Any]]:
        return self._client.get_datasets(**filters)

    def databases(self, **filters: Any) -> list[dict[str, Any]]:
        return self._client.get_databases(**filters)

    def get_resource(self, resource_type: str, resource_id: int) -> dict[str, Any]:
        """Fetch a single resource by type and ID."""
        return self._client.get_resource(resource_type, resource_id)

    def dashboard_detail(self, dashboard_id: int) -> dict[str, Any]:
        return self._client.get_dashboard(dashboard_id)

    # ------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------

    def run_sql(
        self,
        sql: str,
        database_id: int,
        schema: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        return self._client.run_query(
            database_id=database_id, sql=sql, schema=schema, limit=limit
        )

    # ------------------------------------------------------------------
    # Create helpers
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        sql: str,
        database_id: int,
        schema: str | None = None,
    ) -> dict[str, Any]:
        return create_virtual_dataset(
            self._client, name, sql, database_id, schema
        )

    def create_chart(
        self,
        dataset_id: int,
        title: str,
        viz_type: str,
        *,
        metrics: list[str] | None = None,
        groupby: list[str] | None = None,
        time_column: str | None = None,
        dashboards: list[int] | None = None,
        **extra_params: Any,
    ) -> dict[str, Any]:
        return _create_chart(
            self._client,
            dataset_id,
            title,
            viz_type,
            metrics=metrics,
            groupby=groupby,
            time_column=time_column,
            dashboards=dashboards,
            extra_params=extra_params or None,
        )

    # ------------------------------------------------------------------
    # Create helpers (dashboards)
    # ------------------------------------------------------------------

    def create_dashboard(
        self,
        dashboard_title: str,
        published: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.create_dashboard(
            dashboard_title=dashboard_title,
            published=published,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update_dataset(
        self,
        dataset_id: int,
        override_columns: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.update_dataset(
            dataset_id, override_columns=override_columns, **kwargs
        )

    def update_chart(self, chart_id: int, **kwargs: Any) -> dict[str, Any]:
        return self._client.update_chart(chart_id, **kwargs)

    def update_dashboard(
        self, dashboard_id: int, **kwargs: Any
    ) -> dict[str, Any]:
        return self._client.update_dashboard(dashboard_id, **kwargs)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_dashboards(self, ids: list[int]) -> bytes:
        """Export dashboards as a ZIP bundle (bytes)."""
        buf = self._client.export_zip("dashboard", ids)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> WorkspaceSnapshot:
        return take_snapshot(self)
