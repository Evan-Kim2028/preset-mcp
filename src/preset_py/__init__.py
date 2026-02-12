"""preset_py — Thin agentic wrapper around preset-cli."""

from preset_py.client import PresetWorkspace, connect
from preset_py.snapshot import WorkspaceSnapshot, save_snapshot, take_snapshot
from preset_py._safety import MutationEntry

__all__ = [
    "connect",
    "MutationEntry",
    "PresetWorkspace",
    "take_snapshot",
    "save_snapshot",
    "WorkspaceSnapshot",
]
