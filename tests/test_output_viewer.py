"""The MCP Apps inline output viewer (``ui://comfy-mcp/output-viewer``).

These run in-process against the real :class:`MCPServer` — no host, no iframe.
They pin the two wire-visible facts an Apps host reads (the ``ui://`` resource
and the ``_meta.ui.resourceUri`` pointer), the self-contained-document
invariants that keep the viewer inside the sandbox's default CSP (no ``fetch``,
no ``innerHTML``, no out-of-sandbox URLs), and the graceful-degradation
guarantee that the two result tools still return their existing text payload so
a non-Apps client is unaffected.

The live render path (a real host dropping the HTML into an iframe and
delivering a tool result over ``postMessage``) cannot be exercised here; it is a
manual check, noted in the PR.
"""

from __future__ import annotations

import asyncio
import json

from conftest import envelope

from comfy_mcp import output_viewer, server

# Any bytes exercise the inline round-trip; the tool does not validate pixels.
_FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-pixels"

# Substrings whose ABSENCE is the contract: the viewer opens no socket, calls no
# tool, assigns no untrusted markup, and points at no origin the ext-apps
# default CSP forbids. A regression that reintroduces one is what this guards.
_FORBIDDEN_HTML_TOKENS = (
    "fetch(",
    "innerHTML",
    "tools/call",
    "file://",
    "http://127.0.0.1",
    "http://localhost",
)


def _list_tools() -> list:
    return asyncio.run(server.mcp.list_tools())


def _read_resource(uri: str) -> list:
    return list(asyncio.run(server.mcp.read_resource(uri)))


def test_resources_read_returns_the_mime_type_and_html():
    """``resources/read`` serves the viewer as the MCP Apps MIME, HTML intact."""
    contents = _read_resource(output_viewer.OUTPUT_VIEWER_URI)

    assert len(contents) == 1
    served = contents[0]
    assert served.mime_type == "text/html;profile=mcp-app"
    assert served.mime_type == output_viewer.OUTPUT_VIEWER_MIME
    # The bytes the host drops into the iframe are the module's document verbatim.
    assert served.content == output_viewer.OUTPUT_VIEWER_HTML
    assert served.content.lstrip().lower().startswith("<!doctype html>")


def test_viewer_is_listed_as_an_apps_resource():
    """The ``ui://`` resource is advertised on ``resources/list`` unconditionally."""
    resources = asyncio.run(server.mcp.list_resources())
    viewer = [r for r in resources if str(r.uri) == output_viewer.OUTPUT_VIEWER_URI]

    assert len(viewer) == 1
    assert viewer[0].mime_type == "text/html;profile=mcp-app"


def test_html_has_no_network_or_unsafe_dom_access():
    """The document stays inside the sandbox: no fetch, no innerHTML, no outside URLs."""
    html = output_viewer.OUTPUT_VIEWER_HTML
    present = [token for token in _FORBIDDEN_HTML_TOKENS if token in html]
    assert not present, f"viewer HTML must not contain {present}"


def test_initialize_request_carries_the_required_bridge_fields():
    """``ui/initialize`` sends ``appInfo`` + ``appCapabilities``, not just the version.

    The 2026-01-26 bridge contract requires all three; a schema-validating host
    rejects a request missing either and never delivers the tool result.
    """
    html = output_viewer.OUTPUT_VIEWER_HTML
    info = output_viewer._APP_INFO
    assert set(info) == {"name", "version"} and all(info.values())
    assert f"var APP_INFO = {json.dumps(info)};" in html
    init = html[html.index('method: "ui/initialize"') :]
    params = init[: init.index("});")]
    for field in ("appInfo: APP_INFO", "appCapabilities: {}", "protocolVersion"):
        assert field in params


def test_bridge_follows_handshake_order_and_trusts_only_the_host():
    """``initialized`` waits for the id-matched response; other frames are ignored."""
    html = output_viewer.OUTPUT_VIEWER_HTML
    kickoff = html[html.index("// Kick off the handshake.") :]
    assert "ui/notifications/initialized" not in kickoff
    assert "msg.id === reqId" in html
    assert "event.source !== window.parent" in html


def test_bridge_acknowledges_teardown_and_reports_late_resizes():
    """Teardown gets a JSON-RPC reply; media load / ResizeObserver re-report height."""
    html = output_viewer.OUTPUT_VIEWER_HTML
    teardown = html[html.index('method === "ui/resource-teardown"') :]
    assert "reply(msg.id" in teardown[: teardown.index("return;")]
    assert "new ResizeObserver(reportSize).observe(document.body)" in html
    # The root's scrollHeight is floored at the viewport, so a frame the host
    # grew from it could never shrink; the measurement is the body's own box.
    assert "document.body.getBoundingClientRect().height" in html
    assert "scrollHeight" not in html
    assert (
        'addEventListener(tag === "img" ? "load" : "loadedmetadata", reportSize)'
        in html
    )


def test_meta_ui_is_set_on_exactly_the_two_result_tools():
    """``tools/list`` points only ``fetch_outputs`` and ``preview_media`` at the viewer."""
    tools = _list_tools()
    pointed = {
        tool.name
        for tool in tools
        if (tool.meta or {}).get("ui", {}).get("resourceUri")
    }

    assert pointed == {"fetch_outputs", "preview_media"}
    for tool in tools:
        if tool.name in pointed:
            assert tool.meta["ui"]["resourceUri"] == output_viewer.OUTPUT_VIEWER_URI


def test_a_non_result_tool_carries_no_apps_pointer():
    """A representative other tool (``run_workflow``) advertises no ``_meta.ui``."""
    tools = {tool.name: tool for tool in _list_tools()}
    assert not (tools["run_workflow"].meta or {}).get("ui")


def test_fetch_outputs_still_returns_its_text_payload_alongside_images(
    patched_run, tmp_path
):
    """Graceful degradation: the metadata payload is preserved as content[0]."""
    (tmp_path / "gen.png").write_bytes(_FAKE_PNG)
    patched_run(envelope(data={"downloaded": ["gen.png"]}))

    result = server.fetch_outputs("pid", str(tmp_path), inline_images=True)

    assert isinstance(result, server.types.CallToolResult)
    first, *rest = result.content
    assert first.type == "text"
    assert json.loads(first.text) == {"downloaded": ["gen.png"]}  # the payload, first
    assert any(item.type == "image" for item in rest)


def test_preview_media_still_returns_its_text_payload(patched_run):
    """``preview_media`` without an inline image is unchanged by the Apps pointer."""
    metadata = {"kind": "audio", "preview": "track.preview.png"}
    patched_run(envelope(data=metadata))

    assert server.preview_media("track.wav", inline_image=False) == metadata
