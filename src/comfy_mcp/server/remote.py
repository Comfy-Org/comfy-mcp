"""Streamable HTTP adapter for the shared FastMCP application.

The application is built once in this package. This module only
adapts that already-registered resource to FastMCP's public ASGI surface and
uvicorn; it owns no tools, backend client, JSON-RPC, or session implementation.
"""

from __future__ import annotations

import uvicorn
from fastmcp import FastMCP

from .config import RemoteServerConfig


def _allowed_hosts(config: RemoteServerConfig) -> list[str] | None:
    """Return the explicit Host allowlist; FastMCP protects localhost by default."""

    return list(config.allowed_hosts) if config.allowed_hosts else None


def create_http_app(config: RemoteServerConfig, mcp: FastMCP) -> object:
    """Adapt the shared application to stateless Streamable HTTP.

    Stateless JSON mode prevents a pre-initialize or stale session ID from
    selecting half-initialized process state. FastMCP still owns JSON-RPC,
    validation, protocol negotiation, request contexts, and ASGI lifespan.
    """

    return mcp.http_app(
        path=config.path,
        json_response=True,
        stateless_http=True,
        transport="http",
        host_origin_protection="auto",
        allowed_hosts=_allowed_hosts(config),
        allowed_origins=[],
    )


def create_uvicorn_server(
    config: RemoteServerConfig,
    mcp: FastMCP,
) -> uvicorn.Server:
    """Compose the shared application's ASGI adapter with uvicorn."""

    app = create_http_app(config, mcp)
    return uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.host,
            port=config.port,
            log_level=config.log_level.value.lower(),
            # Keep stdout predictable across transports. Uvicorn lifecycle and
            # error logs remain on stderr; child stdout remains envelope data.
            access_log=False,
        )
    )


def serve(config: RemoteServerConfig, mcp: FastMCP) -> None:
    """Serve the shared application until SIGINT/SIGTERM."""

    http_server = create_uvicorn_server(config, mcp)
    try:
        http_server.run()
    except KeyboardInterrupt:
        # Uvicorn has already unwound the ASGI lifespan. Treat an operator
        # interrupt as a normal CLI exit rather than printing a traceback.
        return
