"""Live contract checks against a real installed comfy-cli binary.

The unit suite mocks every spawn, so the argv/envelope contract it protects
is otherwise never exercised against the real, unpinned comfy-cli. These
tests need the binary only — no running ComfyUI — and run in the scheduled
cli-contract workflow (and locally via `pytest -m cli_contract` when comfy
is on PATH). A failure here means comfy-cli moved, not that this repo broke.
"""

from __future__ import annotations

import shutil

import pytest

from comfy_mcp import server

pytestmark = pytest.mark.cli_contract


@pytest.fixture(scope="module", autouse=True)
def _needs_binary():
    if shutil.which(server.COMFY_BIN) is None:
        pytest.skip(f"`{server.COMFY_BIN}` not on PATH")


def test_env_returns_a_current_major_envelope_with_a_dict_payload():
    data = server._run_comfy("env", timeout=60.0)
    assert isinstance(data, dict), f"comfy env payload drifted: {type(data)}"
    assert "server" in data or "workspace" in data, (
        f"comfy env payload keys drifted: {sorted(data)[:10]}"
    )


def test_templates_ls_payload_has_the_rows_contract():
    data = server._run_comfy("templates", "ls", timeout=60.0)
    assert isinstance(data, dict) and isinstance(data.get("rows"), list), (
        "templates ls payload drifted — search_templates parses "
        f"{{rows: [...]}}, got: {type(data)}"
    )
    if data["rows"]:
        row = data["rows"][0]
        missing = {"name", "title", "output_type", "tags"} - set(row)
        assert not missing, f"template row lost fields the projection relays: {missing}"
