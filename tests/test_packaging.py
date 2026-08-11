"""Pin the packaging metadata that only PyPI, not this repo, ever renders.

The version lives in two files; pin that they agree.

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


# The links PyPI renders in its sidebar. Nothing in this repo displays them, so
# their absence is invisible here and only shows up on a published page — which
# is how 0.10.0 shipped with `"project_urls": null` and a blank `Home-page`.
_REQUIRED_URL_LABELS = ("Homepage", "Repository", "Documentation", "Issues")
_REPO_URL = "https://github.com/Comfy-Org/comfy-mcp"


def _project_urls() -> dict[str, str]:
    text = _PYPROJECT.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text).get("project", {}).get("urls", {})
    # 3.10 has no `tomllib` (see `_pyproject_version`). Slice the `[project.urls]`
    # table out by hand: from its header to the next table header, so a `Key =
    # "..."` line under some LATER table can never be read as a project URL.
    match = re.search(
        r"^\[project\.urls\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.S
    )
    if match is None:
        return {}
    return dict(re.findall(r'^(\w+)\s*=\s*"([^"]+)"', match.group(1), re.MULTILINE))


def test_pyproject_declares_the_pypi_sidebar_links():
    """No `[project.urls]` means a PyPI page with no route back to the project."""
    urls = _project_urls()
    missing = [label for label in _REQUIRED_URL_LABELS if label not in urls]
    assert not missing, (
        f"pyproject.toml `[project.urls]` is missing {missing}. Without them the "
        "PyPI page links nowhere: no repository, no docs, no issue tracker."
    )
    assert all(url.startswith("https://") for url in urls.values()), urls


def test_project_urls_point_at_this_repository():
    """A copy-paste from another project would publish someone else's links."""
    urls = _project_urls()
    assert urls["Repository"] == _REPO_URL
    assert urls["Issues"].startswith(_REPO_URL + "/")
