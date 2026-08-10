"""The packaged version lives in two files; pin that they agree.

`publish.yml`'s `check-version` job already compares both against the release
tag — but it runs on the `release: created` event, so it fires only AFTER the
tag and the GitHub Release exist. A mismatch caught there costs a deleted
release and tag, and a PyPI version number can never be reused. This pins the
same invariant at PR time, where the fix is an edit instead.

Read from the FILES rather than `importlib.metadata`: an editable install
carries the version recorded when it was installed, so a stale local env would
compare the wrong string in both directions. The files are the source of truth
the release checks against.
"""

import re
import sys
from pathlib import Path

import comfy_mcp

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"
_INIT = _ROOT / "src" / "comfy_mcp" / "__init__.py"


def _pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text)["project"]["version"]
    # 3.10 has no `tomllib`, and one field does not justify a TOML dependency.
    # `^version` is anchored, so `target-version` under [tool.ruff] cannot match.
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, 'no `version = "..."` line in pyproject.toml'
    return match.group(1)


def _dunder_version_literal() -> str:
    """Read `__version__` as TEXT, mirroring what `check-version` parses."""
    match = re.search(
        r'^__version__\s*=\s*"([^"]+)"', _INIT.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert match is not None, "no `__version__` assignment in src/comfy_mcp/__init__.py"
    return match.group(1)


def test_pyproject_and_dunder_version_agree():
    """A release whose two versions disagree fails `check-version` after tagging."""
    assert _pyproject_version() == _dunder_version_literal(), (
        "pyproject.toml and src/comfy_mcp/__init__.py disagree on the version. "
        "Both must equal the release tag or publish.yml's check-version job "
        "fails after the release has already been created."
    )


def test_imported_dunder_matches_its_own_source():
    """Guard the parse above: the literal read must match the imported value."""
    assert comfy_mcp.__version__ == _dunder_version_literal()
