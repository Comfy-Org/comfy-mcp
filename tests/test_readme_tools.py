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
# The name class includes digits so a tool like ``nodes_v2`` is still captured.
_ROW_RE = re.compile(r"^\s*\|\s*`([a-z0-9_]+)\(")

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


def _readme_table_tool_names(text: str) -> list[str]:
    """Tool names from the table's first column, in document order.

    Returns a list (not a set) so callers can detect duplicate rows for the
    same tool, which the one-row-per-tool contract forbids.
    """
    names: list[str] = []
    for line in text.splitlines():
        match = _ROW_RE.match(line)
        if match:
            names.append(match.group(1))
    return names


def test_readme_table_matches_registered_tools():
    """Every registered tool has exactly one table row, and vice versa."""
    registered = _registered_tool_names()
    table = _readme_table_tool_names(README.read_text(encoding="utf-8"))

    duplicates = sorted({name for name in table if table.count(name) > 1})
    assert not duplicates, f"README table has more than one row for: {duplicates}"

    table_set = set(table)
    missing = registered - table_set
    extra = table_set - registered
    assert not missing, (
        f"tools registered but absent from the README table: {sorted(missing)}"
    )
    assert not extra, f"README table rows with no registered tool: {sorted(extra)}"


def test_readme_stated_count_matches_registered():
    """The prose count (both stated locations) matches the registered tool count."""
    count = len(_registered_tool_names())
    noun = "tool" if count == 1 else "tools"
    phrase = f"{_int_to_words(count).capitalize()} {noun}"

    text = README.read_text(encoding="utf-8")
    # Both intended locations must state the count independently, so drift in
    # one is caught even if a stray mention elsewhere keeps a bare total right.
    status_line = next(
        (line for line in text.splitlines() if "**Status:**" in line), ""
    )
    tools_section = text.split("## Tools", 1)[1] if "## Tools" in text else ""

    assert phrase in status_line, (
        f"Status blockquote must state {phrase!r}; update it to match "
        f"{count} registered tools (found: {status_line.strip()!r})"
    )
    assert phrase in tools_section, (
        f"the `## Tools` section must state {phrase!r}; update it to match "
        f"{count} registered tools"
    )
