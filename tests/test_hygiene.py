"""Destined-public hygiene guards (AGENTS.md).

The repo ships to people who cannot read the internal tracker, so AGENTS.md
bans tracker references outright. Comments must carry their rationale
inline; a ticket id is a pointer to nowhere the day the repo is public.
This test is the enforcement the rule previously lacked.
"""

from __future__ import annotations

import re
from pathlib import Path

# The internal tracker's issue-key shape. Kept as a fragment assembled at
# runtime so this file never contains a match for its own pattern.
_TRACKER_RE = re.compile(r"\b" + "BE" + r"-\d+")

_ROOT = Path(__file__).resolve().parents[1]


def test_no_internal_tracker_references():
    scanned = [
        _ROOT / "README.md",
        _ROOT / "AGENTS.md",
        *sorted((_ROOT / "src").rglob("*.py")),
        *sorted((_ROOT / "tests").rglob("*.py")),
    ]
    offenders = []
    for path in scanned:
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _TRACKER_RE.search(line):
                offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "internal-tracker references found (AGENTS.md destined-public rule); "
        "state the rationale inline instead:\n" + "\n".join(offenders)
    )
