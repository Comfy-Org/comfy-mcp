"""Public comfy-mcp server API.

Only the shared FastMCP application, its console entry point, and registered
tool callables are exported here.  Runtime helpers deliberately live in
``comfy_mcp.server._internal`` and are not part of this module's public API.
"""

from __future__ import annotations

from ._internal import main as main
from ._internal import mcp as mcp
from .tools import *  # noqa: F403 - tools.__all__ is the explicit public list
from .tools import __all__ as _tool_exports

__all__ = ("main", "mcp", *_tool_exports)
