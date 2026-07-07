"""Shared pytest fixtures.

The comfy-cli version guard (`server._check_comfy_version`) shells out to
`comfy --version` once per process from inside `_run_comfy`. The unit tests stub
`subprocess.run` to emit canned envelopes and assert the exact argv — a stray
`comfy --version` call would consume that stub and pollute those assertions. So
by default we mark the guard "already checked" for every test; the dedicated
guard tests (`test_wrapper.py`) re-enable it explicitly and mock `--version`.
"""

from __future__ import annotations

import pytest

from comfy_local_mcp import server


@pytest.fixture(autouse=True)
def _skip_version_guard(monkeypatch):
    """Neutralize the once-per-process comfy-cli version guard for unit tests."""
    monkeypatch.setattr(server, "_version_checked", True)
