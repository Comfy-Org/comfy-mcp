"""Every public tool carries intentional MCP annotations on ``tools/list``.

The classification lives in ``tool_annotations``. This file restates the
read-only and destructive sets so a quiet reclassification fails here, and
checks the serialized camelCase hints a client actually receives.
"""

from __future__ import annotations

import asyncio

from comfy_mcp import server, tool_annotations

# Discovery, metadata, inspection, and status. Grouped tools that can also
# mutate (`job`, `download`, `project`) are intentionally absent.
READ_ONLY = frozenset(
    {
        "server_info",
        "auth_status",
        "billing_status",
        "list_partner_models",
        "partner_model_schema",
        "system_stats",
        "get_logs",
        "discover",
        "which",
        "search_templates",
        "get_template",
        "nodes",
        "node_dependencies",
        "workflow_deps",
        "search_models",
        "validate_workflow",
        "list_workflow_slots",
        "list_workflow_notes",
    }
)

# State-changing tools whose effect can replace, stop, delete, or reinstall.
# Generation that only adds a job is not in this set. Uploads are, because
# `overwrite=True` can replace an existing asset.
DESTRUCTIVE = frozenset(
    {
        "emit_partner_workflow",
        "job",
        "fetch_outputs",
        "stop_comfyui",
        "restart_comfyui",
        "update_comfyui",
        "switch_comfyui_version",
        "install_node",
        "fetch_template",
        "download_model",
        "download",
        "upload_file",
        "init_upload",
        "complete_upload",
        "set_workflow_slot",
        "vary_workflow",
    }
)


def _listed() -> dict[str, dict]:
    tools = asyncio.run(server.mcp.list_tools())
    return {tool.name: tool.model_dump(by_alias=True) for tool in tools}


def test_every_listed_tool_has_serialized_annotations():
    listed = _listed()
    assert set(listed) == set(tool_annotations.TOOL_ANNOTATIONS)
    for name, payload in listed.items():
        hints = payload["annotations"]
        assert payload["title"]
        assert hints["title"] == payload["title"]
        for key in (
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        ):
            assert isinstance(hints[key], bool), f"{name} {key}"
        spec = tool_annotations.TOOL_ANNOTATIONS[name]
        assert hints["readOnlyHint"] is spec.read_only_hint
        assert hints["destructiveHint"] is spec.destructive_hint
        assert hints["idempotentHint"] is spec.idempotent_hint
        assert hints["openWorldHint"] is spec.open_world_hint


def test_read_only_and_destructive_sets_match_real_behavior():
    listed = _listed()
    read_only = {
        name
        for name, payload in listed.items()
        if payload["annotations"]["readOnlyHint"]
    }
    destructive = {
        name
        for name, payload in listed.items()
        if payload["annotations"]["destructiveHint"]
    }
    assert read_only == READ_ONLY
    assert destructive == DESTRUCTIVE
    assert read_only.isdisjoint(destructive)
    # A mutating tool is either conservative-destructive or explicitly additive.
    assert read_only | destructive | {
        "auth_login",
        "run_workflow",
        "generate_image",
        "partner_generate",
        "run_template",
        "free_memory",
        "launch_comfyui",
        "project",
    } == set(listed)


def test_generation_is_not_read_only_and_uploads_are_conservative():
    listed = _listed()
    for name in ("run_workflow", "generate_image", "run_template", "partner_generate"):
        hints = listed[name]["annotations"]
        assert hints["readOnlyHint"] is False
        assert hints["destructiveHint"] is False
    for name in ("upload_file", "init_upload", "complete_upload"):
        hints = listed[name]["annotations"]
        assert hints["readOnlyHint"] is False
        assert hints["destructiveHint"] is True
