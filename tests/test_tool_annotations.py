"""Every ``@mcp.tool()`` must carry explicit ``ToolAnnotations``.

MCP's spec defaults an un-annotated tool to ``readOnlyHint=false`` /
``destructiveHint=true`` / ``openWorldHint=true`` — so a host that confirms on
those hints prompts the user before a harmless local read like ``server_info``
or ``nodes``. Setting the four hints explicitly on each tool is what stops that.

The table below is the contract: the first test fails the moment a NEW tool ships
with ``annotations is None`` (the acceptance criterion), and the per-tool
assertions pin the classification so a careless edit to one tool's hints is
caught here rather than by a confused host. ``openWorldHint`` is set from what
comfy-cli actually reaches: the open internet / Comfy Cloud / a partner API / a
registry, NOT the local ComfyUI process or the local filesystem — so
``search_models``, which reads the LOCAL install's model files, is closed-world
while ``download_model`` (fetches an arbitrary URL) is open-world.
"""

from __future__ import annotations

import pytest

from comfy_mcp import server

# name -> (readOnlyHint, destructiveHint, idempotentHint, openWorldHint, title)
_EXPECTED = {
    "server_info": (True, False, True, False, None),
    "auth_status": (True, False, True, True, None),
    "billing_status": (True, False, True, True, None),
    "auth_login": (False, False, False, True, None),
    "run_workflow": (False, False, False, True, None),
    "generate_image": (False, False, False, True, None),
    "list_partner_models": (True, False, True, False, None),
    "partner_model_schema": (True, False, True, False, None),
    "partner_generate": (False, True, False, True, None),
    "emit_partner_workflow": (False, True, True, False, None),
    "run_template": (False, False, False, True, None),
    "job": (False, True, False, False, None),
    "system_stats": (True, False, True, False, None),
    "free_memory": (False, True, True, False, None),
    "fetch_outputs": (False, False, False, False, None),
    "launch_comfyui": (False, False, False, False, None),
    "stop_comfyui": (False, True, True, False, None),
    "restart_comfyui": (False, True, False, False, None),
    "update_comfyui": (False, True, False, True, None),
    "switch_comfyui_version": (False, True, False, True, None),
    "install_node": (False, True, False, True, None),
    "preview_media": (False, True, True, False, None),
    "get_logs": (True, False, True, False, None),
    "discover": (True, False, True, False, "Discover comfy-cli commands"),
    "which": (True, False, True, False, "Which ComfyUI is targeted"),
    "project": (False, False, True, False, None),
    "search_templates": (True, False, True, True, None),
    "get_template": (True, False, True, True, None),
    "fetch_template": (False, True, True, True, None),
    "nodes": (True, False, True, False, "Search the local node catalog"),
    "node_dependencies": (True, False, True, True, None),
    "workflow_deps": (True, False, True, True, None),
    "search_models": (True, False, True, False, None),
    "download_model": (False, True, False, True, None),
    "download": (False, True, False, False, None),
    "upload_file": (False, True, False, False, None),
    "validate_workflow": (True, False, True, False, None),
    "list_workflow_slots": (True, False, True, False, None),
    "list_workflow_notes": (True, False, True, False, None),
    "set_workflow_slot": (False, True, False, False, None),
    "vary_workflow": (False, True, True, False, None),
}


def _tools():
    return server.mcp._tool_manager.list_tools()


def _tool_names():
    return sorted(t.name for t in _tools())


def test_every_tool_has_explicit_annotations():
    """The acceptance criterion: no registered tool may ship un-annotated.

    A new ``@mcp.tool()`` added without ``annotations=...`` inherits the MCP
    spec's destructive/open-world defaults — this fails until it is classified.
    """
    missing = sorted(t.name for t in _tools() if t.annotations is None)
    assert missing == [], f"tools missing ToolAnnotations: {missing}"


def test_expected_table_matches_registered_tools():
    """The pinned table and the live tool set stay in lockstep.

    If a tool is added or removed, this fails so whoever did it has to classify
    the new tool (or drop the stale row) rather than leave the table silently
    partial.
    """
    assert _tool_names() == sorted(_EXPECTED)


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_tool_annotation_hints(name):
    ro, destructive, idempotent, open_world, title = _EXPECTED[name]
    ann = server.mcp._tool_manager.get_tool(name).annotations
    assert ann is not None, f"{name} has no annotations"
    assert ann.read_only_hint is ro, name
    assert ann.destructive_hint is destructive, name
    assert ann.idempotent_hint is idempotent, name
    assert ann.open_world_hint is open_world, name
    assert ann.title == title, name


def test_read_only_tools_are_never_destructive():
    """Internal consistency: a read-only tool cannot also be destructive."""
    for tool in _tools():
        ann = tool.annotations
        if ann is not None and ann.read_only_hint:
            assert ann.destructive_hint is False, tool.name


def test_acceptance_pins():
    """The exact hints the ticket calls out, asserted on their own."""
    get = server.mcp._tool_manager.get_tool
    si = get("server_info").annotations
    assert (si.read_only_hint, si.destructive_hint, si.open_world_hint) == (
        True,
        False,
        False,
    )
    for name in ("install_node", "free_memory"):
        ann = get(name).annotations
        assert ann.destructive_hint is True, name
        assert ann.read_only_hint is False, name
