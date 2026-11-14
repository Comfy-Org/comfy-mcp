"""Structured output (`outputSchema` + `structuredContent`) for the data tools.

A handful of tools answer with a single JSON object a host wants to read as data,
not scrape out of a text block: ``job``, ``fetch_outputs``, ``server_info``,
``run_workflow``/``run_template``/``partner_generate`` and ``download``/
``download_model``. Each is annotated ``-> dict[str, Any]`` so the MCP SDK
derives an object ``outputSchema`` (published in ``tools/list``) and, on every
call, returns the payload as ``structuredContent`` ALONGSIDE the existing JSON
text block — the shape ``2026-07-28`` asks for.

These tests pin that boundary behaviour. The per-tool argv/passthrough tests live
in their own files and still compare the tools' raw ``dict`` return values
directly (the ``@mcp.tool()`` decorator leaves the function returning that dict);
this file is the one that drives the tools THROUGH ``MCPServer`` and inspects the
``CallToolResult`` a real client receives.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest
from conftest import envelope

from comfy_mcp import server

# The tools that now publish an object output schema. The first two are the
# acceptance-named pair; the rest are the full set the change covers.
_STRUCTURED_TOOLS = (
    "job",
    "fetch_outputs",
    "server_info",
    "run_workflow",
    "run_template",
    "partner_generate",
    "download",
    "download_model",
)


def _tools_by_name():
    return {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}


def test_tools_list_publishes_output_schema_for_the_data_tools():
    """``tools/list`` advertises an object ``outputSchema`` for each data tool.

    A client reads this schema to know a tool answers with structured data; the
    acceptance names ``job`` and ``fetch_outputs`` explicitly, and the rest of the
    set is held to the same bar here so none silently reverts to ``-> Any``.
    """
    tools = _tools_by_name()
    for name in _STRUCTURED_TOOLS:
        schema = tools[name].output_schema
        assert schema is not None, f"{name} lost its outputSchema"
        assert schema.get("type") == "object", f"{name} schema is not an object"


def test_a_long_tail_catalog_tool_stays_text_only():
    """The change is scoped: a catalog/discovery tool keeps NO output schema.

    ``search_templates`` is one of the long-tail tools the ticket leaves as text,
    so its absence of an ``outputSchema`` is the control that proves the structured
    set was opted in deliberately rather than applied server-wide.

    What "no output schema" looks like depends on the interpreter, not on this
    server: on Python 3.11+ ``typing.Any`` is a class with no type hints, so the
    SDK derives nothing for ``-> Any``; on 3.10 it is not a class, so the SDK
    falls through to its generic branch and wraps it as ``{"result": Any}``.
    Either way the tool carries none of the opted-in object payload schema.
    """
    schema = _tools_by_name()["search_templates"].output_schema
    if sys.version_info >= (3, 11):
        assert schema is None
    else:
        assert schema == {
            "properties": {"result": {"title": "Result"}},
            "required": ["result"],
            "title": "search_templatesOutput",
            "type": "object",
        }


def test_job_call_returns_structured_content_equal_to_the_payload(patched_run):
    """``structuredContent`` equals the old payload, and the text block still carries it.

    Drives ``job(action="status")`` through ``MCPServer.call_tool`` — the path a
    real client takes — and checks all three halves of the contract: the result is
    not an error, ``structured_content`` is byte-for-byte the comfy-cli ``data``
    dict, and ``content`` still carries that same object serialized as JSON text.
    """
    data = {"status": "completed", "outputs": {"9": ["gen.png"]}}
    patched_run(envelope(data=data))

    result = asyncio.run(server.mcp.call_tool("job", {"prompt_id": "p1"}))

    assert not result.is_error
    assert result.structured_content == data  # equals the old payload exactly
    text_blocks = [c for c in result.content if c.type == "text"]
    assert text_blocks, "the JSON text block must still be present"
    assert json.loads(text_blocks[0].text) == data  # text is the same object


def test_fetch_outputs_call_returns_structured_content(patched_run, tmp_path):
    """``fetch_outputs`` publishes its download metadata as ``structuredContent``."""
    data = {"downloaded": ["gen.png"]}
    patched_run(envelope(data=data))

    result = asyncio.run(
        server.mcp.call_tool(
            "fetch_outputs", {"prompt_id": "p1", "out_dir": str(tmp_path)}
        )
    )

    assert not result.is_error
    assert result.structured_content == data
    text_blocks = [c for c in result.content if c.type == "text"]
    assert json.loads(text_blocks[0].text) == data


def test_fetch_outputs_inline_images_call_keeps_structured_metadata_and_images(
    patched_run, tmp_path
):
    """Inlined images ride in ``content`` while the metadata stays structured.

    The ``inline_images`` branch hands back a hand-built ``CallToolResult`` so an
    object schema can coexist with image content blocks — ``structuredContent`` is
    the metadata plus the ``media`` manifest, the PNG is an ``image`` block
    alongside the metadata's JSON text.
    """
    png = b"\x89PNG\r\n\x1a\nfake"
    (tmp_path / "gen.png").write_bytes(png)
    data = {"downloaded": ["gen.png"]}
    patched_run(envelope(data=data))

    result = asyncio.run(
        server.mcp.call_tool(
            "fetch_outputs",
            {"prompt_id": "p1", "out_dir": str(tmp_path), "inline_images": True},
        )
    )

    assert not result.is_error
    media = result.structured_content.pop("media")
    assert result.structured_content == data
    assert [r["inline"] for r in media] == [True]
    text_blocks = [c for c in result.content if c.type == "text"]
    assert json.loads(text_blocks[0].text) == data  # text keeps the metadata alone
    images = [c for c in result.content if c.type == "image"]
    assert len(images) == 1
    import base64

    assert base64.b64decode(images[0].data) == png


@pytest.mark.parametrize("payload", [[{"prompt_id": "a"}], None, "queued"])
def test_a_non_object_payload_is_kept_under_data_not_failed(patched_run, payload):
    """A list/``null``/scalar ``data`` still yields a valid structured reply.

    The SDK validates the return value against the object ``outputSchema`` only
    after comfy-cli ran, so a bare non-object payload would turn a call that
    already happened (a cancel, a paid generation) into a tool error. It is kept
    verbatim under ``data`` instead, in ``structuredContent`` and the text block.
    """
    patched_run(envelope(data=payload))

    result = asyncio.run(server.mcp.call_tool("job", {"action": "queue"}))

    assert not result.is_error
    assert result.structured_content == {"data": payload}
    text_blocks = [c for c in result.content if c.type == "text"]
    assert json.loads(text_blocks[0].text) == {"data": payload}


def test_every_structured_tool_wraps_a_non_object_reply():
    """Each schema'd tool is wrapped, so none can return a bare non-object.

    Guards the set against a tool gaining ``-> dict[str, Any]`` without the
    wrapper: ``functools.wraps`` leaves ``__wrapped__`` on exactly those.
    """
    for name in _STRUCTURED_TOOLS:
        assert hasattr(getattr(server, name), "__wrapped__"), name


def test_fetch_outputs_inline_images_wraps_a_non_object_payload(patched_run, tmp_path):
    """The hand-built ``CallToolResult`` branch wraps a non-object payload too."""
    patched_run(envelope(data=None))

    result = asyncio.run(
        server.mcp.call_tool(
            "fetch_outputs",
            {"prompt_id": "p1", "out_dir": str(tmp_path), "inline_images": True},
        )
    )

    assert not result.is_error
    assert result.structured_content == {"data": None, "media": []}
