"""Outbound client boundary for the local comfy-cli engine.

The MCP transports depend on this package; this package never imports an MCP
server or transport adapter.  The concrete client is composed with the
existing, heavily tested subprocess runners at the application root.
"""

from .protocols import ComfyCliClient, RawComfyResult
from .subprocess_client import SubprocessComfyCliClient

__all__ = ("ComfyCliClient", "RawComfyResult", "SubprocessComfyCliClient")
