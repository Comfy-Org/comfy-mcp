"""Ceiling on the LLM-facing payload: tool docstrings + handshake instructions.

Every MCP session ships every tool description plus INSTRUCTIONS before any
work happens. After a three-pass docstring diet, the payload is approximately
15.1k estimated tokens — a fixed tax on every conversation. This test freezes
a ceiling just above the measured size so growth is a deliberate decision (bump
the constant in the same PR, with justification), and the ceiling ratchets DOWN
as further trims land.

Token estimate: len(chars) / 4 — crude but stable, and only deltas matter.
Docstrings are collected via AST rather than SDK introspection so the test
does not depend on MCP SDK internals.
"""

from __future__ import annotations

import ast
from pathlib import Path

from comfy_mcp import instructions

_SERVER_SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_mcp" / "server.py"

# Ceiling, in estimated tokens (chars/4). Measured 2026-08-07 after the
# tool-consolidation series' third and final commit (`nodes(action=...)`,
# 47 -> 39 tools): ~13.84k (39 tool docstrings, 40248 doc chars + INSTRUCTIONS
# 15126 chars). Ratcheted 17,000 -> 15,000 accordingly — comfortably above the
# measurement so ordinary edits never trip it while real growth does. Ratchet
# DOWN as the docstring diet lands; never bump without stating why in the same
# PR.
#
# 15,000 -> 15,250 when `nodes` documented `is_api_node` / `exclude_api`: that
# paragraph is ~130 tokens and the tree was ALREADY at 14,984, so the slack the
# ratchet was set with had been spent by intervening docstring growth rather
# than by this change. Measured ~15,112 after it. Deliberately tight — the
# point of this ceiling is that the next growth is a decision too, not that
# there is room for one.
#
# 15,250 -> 15,500 when `billing_status` was added. This one is a
# whole new TOOL, not docstring growth on an existing one: its description is
# ~1,110 chars, which is the mean length of the 39 that were already here, so
# there was no version of it that fit the 552 chars of slack left. The ceiling
# moves by what the tool costs (~277 tokens) and no further — measured ~15,390
# after it, so the next growth is still a decision.
_BUDGET_TOKENS = 15_500


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
    # Was `> 40`: the tool-consolidation series (six job tools into one grouped
    # `job(action=...)` here, `download`/`nodes` in later commits) walks the
    # live count from 53 down to 39. Relaxed rather than pinned exactly, so
    # this stays a discovery sanity check, not a second copy of
    # `test_the_readme_tool_count_matches_live_tool_set`.
    assert tool_count >= 35, f"tool discovery broke: found {tool_count} tools"

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
