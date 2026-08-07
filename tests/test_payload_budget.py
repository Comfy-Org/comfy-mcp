"""Ceiling on the LLM-facing payload: tool docstrings + handshake instructions.

Every MCP session ships every tool description plus INSTRUCTIONS before any
work happens. That payload was measured at roughly 39k tokens on 2026-08-06
— a fixed tax on every conversation. This test freezes a ceiling just above
the measured size so growth is a deliberate decision (bump the constant in
the same PR, with justification), and the planned docstring diet ratchets
the constant DOWN as it lands.

Token estimate: len(chars) / 4 — crude but stable, and only deltas matter.
Docstrings are collected via AST rather than SDK introspection so the test
does not depend on MCP SDK internals.
"""

from __future__ import annotations

import ast
from pathlib import Path

from comfy_mcp import instructions

_SERVER_SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_mcp" / "server.py"

# Ceiling, in estimated tokens (chars/4). Measured 2026-08-06 after diet pass
# 3 (INSTRUCTIONS deduped against the now-slimmed docstrings, on top of pass
# 1's 12 largest docstrings and pass 2's remaining 40): ~15.1k.
# Set ~10% above the measurement so ordinary edits never trip it while real
# growth does. Ratchet DOWN as the docstring diet lands; never bump without
# stating why in the same PR.
_BUDGET_TOKENS = 17_000


def _is_tool_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        # matches @mcp.tool() and @mcp.tool
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "tool"
            and isinstance(target.value, ast.Name)
            and target.value.id == "mcp"
        ):
            return True
    return False


def test_llm_payload_within_budget():
    tree = ast.parse(_SERVER_SRC.read_text(encoding="utf-8"))
    doc_chars = 0
    tool_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_tool_decorated(node):
                tool_count += 1
                doc_chars += len(ast.get_docstring(node) or "")
    assert tool_count > 40, f"tool discovery broke: found {tool_count} tools"

    total_chars = doc_chars + len(instructions.INSTRUCTIONS)
    est_tokens = total_chars // 4
    assert est_tokens <= _BUDGET_TOKENS, (
        f"LLM payload grew to ~{est_tokens} est. tokens "
        f"({tool_count} tool docstrings {doc_chars} chars + "
        f"INSTRUCTIONS {len(instructions.INSTRUCTIONS)} chars) — budget is "
        f"{_BUDGET_TOKENS}. Trim the docstring (agent contract only; "
        "maintainer rationale goes in comments) or, if growth is deliberate, "
        "bump _BUDGET_TOKENS in this same PR and say why."
    )
