"""Minimal builder for the server's one FastMCP application resource."""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp import FastMCP


@dataclass(frozen=True, slots=True)
class McpApplicationBuilder:
    """Own the application identity and construct its single FastMCP instance.

    Transport is deliberately absent: stdio and Streamable HTTP are adapters
    selected after the application has been built and all tools registered.
    """

    name: str
    version: str
    instructions: str

    def build(self) -> FastMCP:
        """Create the application that both transport adapters will serve."""

        return FastMCP(
            self.name,
            version=self.version,
            instructions=self.instructions,
        )
