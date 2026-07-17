"""Parity check: the README tool table must list exactly the registered tools.

Several open PRs each add more tools, so the ``## Tools`` table drifts out of
sync silently. These tests enumerate the registered tools via FastMCP's
in-process registry (``mcp.list_tools()`` — no ``comfy`` binary needed, same
zero-dependency footing as the other mocked tests) and cross-check them against
the README:

- the set of tool names in the table's first column must equal the registered
  set (deliberately removing a table row, or adding a tool without a row, fails);
- the tool count stated in prose (the Status blockquote and the ``## Tools``
  header) must match ``len(mcp.list_tools())``.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from comfy_local_mcp import server

README = Path(__file__).resolve().parent.parent / "README.md"

# First column of a tool-table row: `| `tool_name(...)` | ... |`.
_ROW_RE = re.compile(r"^\s*\|\s*`([a-z_]+)\(")

# Spelled-out English numbers, matching the README's prose voice ("Thirty-one
# tools"). Covers 0..99 — well past any realistic tool count.
_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def _int_to_words(n: int) -> str:
    """English words for 0..99, e.g. 31 -> ``"thirty-one"``."""
    if not 0 <= n <= 99:
        raise ValueError(f"unsupported count for prose check: {n}")
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def _registered_tool_names() -> set[str]:
    return {tool.name for tool in asyncio.run(server.mcp.list_tools())}


def _readme_table_tool_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        match = _ROW_RE.match(line)
        if match:
            names.add(match.group(1))
    return names


def test_readme_table_matches_registered_tools():
    """Every registered tool has exactly one table row, and vice versa."""
    registered = _registered_tool_names()
    table = _readme_table_tool_names(README.read_text(encoding="utf-8"))

    missing = registered - table
    extra = table - registered
    assert not missing, (
        f"tools registered but absent from the README table: {sorted(missing)}"
    )
    assert not extra, f"README table rows with no registered tool: {sorted(extra)}"


def test_readme_stated_count_matches_registered():
    """The prose count (both stated locations) matches the registered tool count."""
    count = len(_registered_tool_names())
    phrase = f"{_int_to_words(count).capitalize()} tools"
    occurrences = README.read_text(encoding="utf-8").count(phrase)
    # The Status blockquote and the `## Tools` header both state the count; both
    # must agree with the registry. Exactly two mentions keeps them in lockstep.
    assert occurrences == 2, (
        f"expected {phrase!r} stated in exactly 2 places (Status line + Tools "
        f"header), found {occurrences}; update the count to match "
        f"{count} registered tools"
    )
