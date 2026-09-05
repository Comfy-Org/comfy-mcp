"""HTTP listener configuration and the shared-application adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from comfy_mcp.server import _internal as server
from comfy_mcp.server import config, remote
from comfy_mcp.server.mcp_app import McpApplicationBuilder


@pytest.fixture(autouse=True)
def _clear_remote_listener_env(monkeypatch):
    for name in (
        config.MCP_HOST_ENV,
        config.MCP_PORT_ENV,
        config.MCP_TRANSPORT_ENV,
        config.MCP_PATH_ENV,
        config.MCP_LOG_LEVEL_ENV,
        config.MCP_ALLOWED_HOSTS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_builder_owns_the_shared_application_identity():
    builder = McpApplicationBuilder(
        name="test-mcp",
        version="1.2.3",
        instructions="shared instructions",
    )

    app = builder.build()

    assert app.name == "test-mcp"
    assert app.version == "1.2.3"
    assert app.instructions == "shared instructions"
    assert server.mcp.name == server.MCP_APPLICATION_BUILDER.name
    assert server.mcp.version == server.MCP_APPLICATION_BUILDER.version


def test_remote_adapter_contains_no_second_fastmcp_constructor():
    source = Path(remote.__file__).read_text(encoding="utf-8")

    assert "FastMCP(" not in source
    assert not hasattr(remote, "create_server")


def test_http_app_is_built_from_the_exact_shared_application(patched_http_app):
    settings = config.RemoteServerConfig()
    sentinel = object()
    calls = patched_http_app(sentinel)

    assert remote.create_http_app(settings, server.mcp) is sentinel
    assert calls == [
        {
            "path": "/mcp",
            "json_response": True,
            "stateless_http": True,
            "transport": "http",
            "host_origin_protection": "auto",
            "allowed_hosts": None,
            "allowed_origins": [],
        }
    ]


def test_listener_defaults_are_distinct_from_comfyui_target():
    settings = config.RemoteServerConfig.from_env()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.endpoint_url == "http://127.0.0.1:8000/mcp"
    assert "COMFYUI" not in settings.model_dump_json()


def test_non_loopback_listener_requires_an_explicit_host_policy():
    with pytest.raises(ValidationError, match="non-loopback MCP bind"):
        config.RemoteServerConfig(host="0.0.0.0")

    settings = config.RemoteServerConfig(
        host="0.0.0.0", allowed_hosts=("mcp.example.test:*",)
    )
    assert remote._allowed_hosts(settings) == ["mcp.example.test:*"]


def test_listener_environment_is_validated(monkeypatch):
    monkeypatch.setenv(config.MCP_HOST_ENV, "0.0.0.0")
    monkeypatch.setenv(config.MCP_PORT_ENV, "9000")
    monkeypatch.setenv(
        config.MCP_ALLOWED_HOSTS_ENV,
        "mcp.example.test:*,127.0.0.1:*",
    )

    settings = config.RemoteServerConfig.from_env()

    assert settings.port == 9000
    assert settings.allowed_hosts == ("mcp.example.test:*", "127.0.0.1:*")


def test_serve_treats_clean_sdk_keyboard_interrupt_as_normal(monkeypatch):
    calls = []

    class InterruptedServer:
        def run(self, **kwargs):
            calls.append(kwargs)
            raise KeyboardInterrupt

    monkeypatch.setattr(
        remote,
        "create_uvicorn_server",
        lambda settings, app: InterruptedServer(),
    )

    remote.serve(config.RemoteServerConfig(), server.mcp)

    assert calls == [{}]
