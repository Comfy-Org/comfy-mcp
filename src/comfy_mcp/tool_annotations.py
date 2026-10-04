"""Semantic hints for every public MCP tool.

Leaf module: no import of ``server``. ``server`` applies :data:`TOOL_ANNOTATIONS`
once, after the ``@mcp.tool`` decorators have registered the live set, so a
new tool that is not classified here fails at import instead of shipping
without a hint.

These are MCP ``ToolAnnotations`` — hints a client may display. They are not
authorization, not a spend gate, and not a second copy of the elicitation
rules. A grouped tool (``job``, ``download``, ``project``, ``nodes``) is one
``tools/list`` entry, so it is classified by the most mutating action that
entry can perform.

Ambiguous calls, stated so a later edit does not "correct" them by accident:

- ``job`` and ``download`` also poll. Cancel is why neither is read-only, and
  why ``destructiveHint`` is true.
- ``project`` also reports status. ``init`` creates a project and refuses
  when one exists, so the tool is not read-only and not destructive.
- ``set_workflow_slot`` defaults to ``stdout=True`` (no write). It can still
  replace slot values in the file, so ``destructiveHint`` is true.
- ``free_memory`` unloads cached weights. That is not additive, but it does
  not delete user files or interrupt a job, so ``destructiveHint`` stays false.
- ``generate_image`` is a local free template. ``openWorldHint`` stays false
  even though ``COMFY_T2I_TEMPLATE`` could be pointed at a partner graph.
- ``partner_generate`` / ``fetch_outputs`` / ``emit_partner_workflow`` can
  replace a caller-chosen path. Generation itself is not classified
  destructive; a tool whose job is to write that path is.
- ``server_info`` and ``search_templates`` may contact the network (freshness,
  a cold gallery). They stay read-only.
- ``init_upload`` may fetch a host-minted HTTPS URL. That is the open-world
  half; ``overwrite=True`` is the destructive half.
- ``launch_comfyui`` / ``restart_comfyui`` can publish ComfyUI, but their
  domain is still this machine, so ``openWorldHint`` is false.
"""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations


def _hint(
    title: str,
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool,
    open_world: bool,
) -> ToolAnnotations:
    return ToolAnnotations(
        title=title,
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=open_world,
    )


def _read(title: str, *, open_world: bool = False) -> ToolAnnotations:
    return _hint(
        title,
        read_only=True,
        destructive=False,
        idempotent=True,
        open_world=open_world,
    )


# Keys are the public tool names. Every registered tool must appear here.
TOOL_ANNOTATIONS: dict[str, ToolAnnotations] = {
    "server_info": _read("ComfyUI environment", open_world=True),
    "auth_status": _read("Cloud sign-in status", open_world=True),
    "billing_status": _read("Credit balance", open_world=True),
    "auth_login": _hint(
        "Start cloud sign-in",
        read_only=False,
        destructive=False,
        idempotent=False,
        open_world=True,
    ),
    "run_workflow": _hint(
        "Run a workflow",
        read_only=False,
        destructive=False,
        idempotent=False,
        open_world=True,
    ),
    "generate_image": _hint(
        "Generate an image",
        read_only=False,
        destructive=False,
        idempotent=False,
        open_world=False,
    ),
    "list_partner_models": _read("List partner models"),
    "partner_model_schema": _read("Partner model parameters"),
    "partner_generate": _hint(
        "Run a paid partner model",
        read_only=False,
        destructive=False,
        idempotent=False,
        open_world=True,
    ),
    "emit_partner_workflow": _hint(
        "Write a partner workflow",
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=False,
    ),
    "run_template": _hint(
        "Run a gallery template",
        read_only=False,
        destructive=False,
        idempotent=False,
        open_world=True,
    ),
    "job": _hint(
        "Inspect or cancel a job",
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=False,
    ),
    "system_stats": _read("Read VRAM and RAM"),
    "free_memory": _hint(
        "Unload cached models",
        read_only=False,
        destructive=False,
        idempotent=True,
        open_world=False,
    ),
    "fetch_outputs": _hint(
        "Save job outputs",
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=False,
    ),
    "launch_comfyui": _hint(
        "Start ComfyUI",
        read_only=False,
        destructive=False,
        idempotent=False,
        open_world=False,
    ),
    "stop_comfyui": _hint(
        "Stop ComfyUI",
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=False,
    ),
    "restart_comfyui": _hint(
        "Restart ComfyUI",
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=False,
    ),
    "update_comfyui": _hint(
        "Update the install",
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "switch_comfyui_version": _hint(
        "Switch ComfyUI version",
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "install_node": _hint(
        "Install a node pack",
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "get_logs": _read("Read ComfyUI logs"),
    "discover": _read("Describe comfy-cli"),
    "which": _read("Show the active install"),
    "project": _hint(
        "Project status or init",
        read_only=False,
        destructive=False,
        idempotent=True,
        open_world=False,
    ),
    "search_templates": _read("Search templates", open_world=True),
    "get_template": _read("Show a template", open_world=True),
    "fetch_template": _hint(
        "Save a template",
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=True,
    ),
    "nodes": _read("Inspect installed nodes"),
    "node_dependencies": _read("Check pack dependencies", open_world=True),
    "workflow_deps": _read("Workflow node packs"),
    "search_models": _read("Search installed models"),
    "download_model": _hint(
        "Download a model",
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "download": _hint(
        "Check or cancel a download",
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=False,
    ),
    "upload_file": _hint(
        "Upload a local file",
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=False,
    ),
    "init_upload": _hint(
        "Start an upload",
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "complete_upload": _hint(
        "Finish an upload",
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=False,
    ),
    "validate_workflow": _read("Check a workflow"),
    "list_workflow_slots": _read("List workflow slots"),
    "list_workflow_notes": _read("Read workflow notes"),
    "set_workflow_slot": _hint(
        "Edit workflow slots",
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=False,
    ),
    "vary_workflow": _hint(
        "Fan out workflow variants",
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=False,
    ),
}


def apply(mcp: Any) -> None:
    """Stamp :data:`TOOL_ANNOTATIONS` onto the tools already registered on ``mcp``.

    Mutates the tool manager's own records. ``list_tools`` returns copies, so
    writing those would not change what ``tools/list`` serializes.
    """
    tools = mcp._tool_manager.list_tools()
    names = {tool.name for tool in tools}
    missing = sorted(names - set(TOOL_ANNOTATIONS))
    extra = sorted(set(TOOL_ANNOTATIONS) - names)
    if missing or extra:
        raise RuntimeError(
            "tool annotations out of sync with registered tools: "
            f"missing={missing} extra={extra}"
        )
    for tool in tools:
        hint = TOOL_ANNOTATIONS[tool.name]
        tool.annotations = hint
        tool.title = hint.title
