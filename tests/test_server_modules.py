"""The server package separates its public API from private runtime helpers."""

from __future__ import annotations

from pathlib import Path

import comfy_mcp.server as public_server
from comfy_mcp.server import _internal, tools


def test_public_server_exports_only_application_and_tool_api():
    expected = {"main", "mcp", *tools.__all__}

    assert set(public_server.__all__) == expected
    assert len(tools.__all__) == 39
    assert "_run_comfy" not in public_server.__all__
    assert not hasattr(public_server, "_run_comfy")


def test_public_exports_are_the_objects_registered_by_the_private_runtime():
    assert public_server.mcp is _internal.mcp
    assert public_server.main is _internal.main

    for name in tools.__all__:
        assert getattr(public_server, name) is getattr(tools, name)
        assert getattr(tools, name) is getattr(_internal, name)


def test_legacy_flat_server_and_entry_modules_are_not_in_the_source_layout():
    package_dir = Path(public_server.__file__).resolve().parent

    assert not package_dir.with_suffix(".py").exists()
    for name in ("cli", "config", "instructions", "mcp_app", "remote"):
        assert not (package_dir.parent / f"{name}.py").exists()
