"""Required operational guidance stays compatible with both MCP transports."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_README = (_ROOT / "README.md").read_text(encoding="utf-8")
_CONTRIBUTING = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")


def _readme_section(heading: str, next_heading: str) -> str:
    start = _README.index(heading)
    end = _README.index(next_heading, start)
    return _README[start:end]


def test_failure_log_docs_cover_stdio_http_and_server_side_ownership():
    section = _readme_section("## Failure log (opt-in)", "## Smoke test")

    assert "`spawn_failed`" in section
    assert "**stdio:**" in section
    assert "**Streamable HTTP:**" in section
    assert "comfy-mcp serve --host 127.0.0.1 --port 9000" in section
    assert "MCP server host" in section
    assert "COMFY_MCP_DEBUG_LOG=1" in section
    assert "immutable failure event" in section
    assert "JSONL writer observes those events" in section
    assert "HTTP upload request bytes" in section
    assert "never enter the failure event" in section


def test_smoke_docs_require_transport_business_flow_and_live_engine_stages():
    section = _readme_section("## Smoke test", "## Contributing")

    assert "tests/test_stdio_business_flow.py" in section
    assert "tests/test_remote_http.py" in section
    assert "tests/test_fastmcp_app.py" in section
    assert "tests/test_file_transfer.py" in section
    assert "real loopback Streamable HTTP/ASGI server" in section
    assert "`server_info` → workflow submission → job status" in section
    assert "same 40-tool application" in section
    assert "temporary signed URL from the same listener" in section
    assert "upload URL is single-use" in section
    assert "./scripts/smoke.sh" in section
    assert "COMFY_LOCAL_URL" in section
    assert "SD1.5" in section


def test_contributing_docs_pin_framework_architecture_and_required_checks():
    readme_section = _readme_section("## Contributing", "## License")
    normalized = " ".join(readme_section.split())

    assert "FastMCP 4" in normalized
    assert "transport guardrails" in normalized
    assert "mandatory targeted + full test gates" in normalized
    assert "fastmcp==4.0.0b3" in _CONTRIBUTING
    assert "mcp==2.0.0" in _CONTRIBUTING
    assert "src/comfy_mcp/client/" in _CONTRIBUTING
    assert "Do not add legacy SSE" in _CONTRIBUTING
    assert "private session monkeypatches" in _CONTRIBUTING
    assert "McpApplicationBuilder" in _CONTRIBUTING
    assert "comfy_mcp.server._internal" in _CONTRIBUTING
    assert "40 tool callables" in _CONTRIBUTING
    assert "immutable `_FailureEvent`" in _CONTRIBUTING
    assert "tests/test_failure_log.py" in _CONTRIBUTING
    assert "## Remote file-transfer changes" in _CONTRIBUTING
    assert "same MCP listener" in _CONTRIBUTING
    assert "upload → submit → poll → fetch" in _CONTRIBUTING
    assert "COMFY_LOCAL_URL" in _CONTRIBUTING
