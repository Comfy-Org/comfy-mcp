"""Client setup docs keep stdio and Streamable HTTP equally discoverable."""

from pathlib import Path

_README = Path(__file__).resolve().parent.parent / "README.md"


def _client_section() -> str:
    body = _README.read_text(encoding="utf-8")
    start = body.index("## Configure your AI client")
    end = body.index("## Comfy Cloud MCP", start)
    return body[start:end]


def test_remote_http_precedes_client_configuration():
    body = _README.read_text(encoding="utf-8")

    assert body.index("## Remote Streamable HTTP") < body.index(
        "## Configure your AI client"
    )


def test_client_setup_documents_both_transports_and_environment_ownership():
    section = _client_section()

    assert "all 39 tools" in section
    assert "the same 39 tools" in section
    assert "two separately running" in section
    assert "same application contract" in section
    assert "ComfyUI continues on its own port (normally `8188`)" in section
    assert "COMFY_BIN=/path/to/venv/bin/comfy" in section
    assert "COMFY_API_KEY=YOUR_COMFY_API_KEY" in section
    assert "COMFY_LOCAL_URL=http://127.0.0.1:8188" in section
    assert "comfy-mcp serve --host 127.0.0.1 --port 9000" in section
    assert "belong to the `comfy-mcp serve` process" in section


def test_claude_code_documents_http_command_and_project_json():
    section = _client_section()

    assert (
        "claude mcp add --transport http comfy-mcp http://127.0.0.1:9000/mcp"
    ) in section
    assert '"type": "http"' in section
    assert '"url": "${COMFY_MCP_URL:-http://127.0.0.1:9000/mcp}"' in section


def test_desktop_and_cursor_document_their_distinct_http_configuration():
    section = _client_section()

    assert "Settings → Connectors → Add custom connector" in section
    assert "does not load remote servers from" in section
    assert '"url": "http://127.0.0.1:9000/mcp"' in section
