"""Configuration for the long-running Remote MCP listener entry point.

The MCP listener and the ComfyUI target are deliberately separate concerns:
this module reads only ``COMFY_MCP_*`` server settings.  ``target.py`` remains
the single owner of ``COMFYUI_URL`` / ``COMFYUI_HOST`` / ``COMFYUI_PORT``.
"""

from __future__ import annotations

import ipaddress
import os
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MCP_HOST_ENV = "COMFY_MCP_HOST"
MCP_PORT_ENV = "COMFY_MCP_PORT"
MCP_TRANSPORT_ENV = "COMFY_MCP_TRANSPORT"
MCP_PATH_ENV = "COMFY_MCP_PATH"
MCP_LOG_LEVEL_ENV = "COMFY_MCP_LOG_LEVEL"
MCP_ALLOWED_HOSTS_ENV = "COMFY_MCP_ALLOWED_HOSTS"


class RemoteTransport(str, Enum):
    """The SDK transport supported by the Remote entry point."""

    STREAMABLE_HTTP = "streamable-http"


class LogLevel(str, Enum):
    """Log levels accepted by uvicorn."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def _is_loopback_host(host: str) -> bool:
    """Whether ``host`` unambiguously names only this machine."""

    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized.strip("[]")).is_loopback
    except ValueError:
        return False


class RemoteServerConfig(BaseModel):
    """Validated configuration for ``comfy-mcp serve``.

    Without this boundary, listener and ComfyUI addresses are both loose
    environment strings and can be accidentally interchanged.  A non-loopback
    bind is also refused until the operator supplies explicit Host-header
    patterns for the SDK's DNS-rebinding protection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: RemoteTransport = RemoteTransport.STREAMABLE_HTTP
    host: str = Field(default="127.0.0.1", min_length=1, max_length=255)
    port: int = Field(default=8000, ge=1, le=65535)
    path: str = Field(default="/mcp", min_length=1, max_length=255)
    log_level: LogLevel = LogLevel.INFO
    allowed_hosts: tuple[str, ...] = ()

    @field_validator("host")
    @classmethod
    def _validate_host(cls, value: str) -> str:
        value = value.strip()
        if not value or any(char.isspace() for char in value):
            raise ValueError("host must be a non-empty hostname or IP address")
        return value

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if not value.startswith("/") or "?" in value or "#" in value:
            raise ValueError(
                "path must start with '/' and contain no query or fragment"
            )
        return value

    @field_validator("allowed_hosts")
    @classmethod
    def _validate_allowed_hosts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(
            not value or any(char.isspace() for char in value) for value in normalized
        ):
            raise ValueError(
                "allowed host patterns must be non-empty and contain no spaces"
            )
        return normalized

    @model_validator(mode="after")
    def _require_remote_host_policy(self) -> RemoteServerConfig:
        if not _is_loopback_host(self.host) and not self.allowed_hosts:
            raise ValueError(
                "a non-loopback MCP bind requires at least one allowed Host pattern "
                "(--allowed-host or COMFY_MCP_ALLOWED_HOSTS)"
            )
        return self

    @classmethod
    def from_env(cls) -> RemoteServerConfig:
        """Read the HTTP listener settings from ``COMFY_MCP_*`` variables."""

        allowed = tuple(
            part.strip()
            for part in os.environ.get(MCP_ALLOWED_HOSTS_ENV, "").split(",")
            if part.strip()
        )
        return cls(
            transport=os.environ.get(
                MCP_TRANSPORT_ENV, RemoteTransport.STREAMABLE_HTTP.value
            ),
            host=os.environ.get(MCP_HOST_ENV, "127.0.0.1"),
            port=os.environ.get(MCP_PORT_ENV, "8000"),
            path=os.environ.get(MCP_PATH_ENV, "/mcp"),
            log_level=os.environ.get(MCP_LOG_LEVEL_ENV, "INFO").upper(),
            allowed_hosts=allowed,
        )

    @property
    def endpoint_url(self) -> str:
        """The configured cleartext MCP endpoint for clients and diagnostics."""

        host = (
            f"[{self.host}]"
            if ":" in self.host and not self.host.startswith("[")
            else self.host
        )
        return f"http://{host}:{self.port}{self.path}"
