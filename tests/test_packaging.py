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


# The 3.10 leg's hand-rolled key/value line. Accepting a quoted key and a
# single-quoted value costs one alternation each and buys agreement with the
# 3.11+ `tomllib` leg: a TOML-legal `"Bug Tracker" = '...'` that tomllib parses
# but a bare-word/double-quote-only pattern DROPS would leave the entry
# unchecked on 3.10 while 3.11 checks it, and the two CI legs would disagree
# about the same file.
_URL_ENTRY_RE = re.compile(
    r"""^[^\S\n]*(?:(?P<bare>\w[\w.-]*)|"(?P<dkey>[^"]*)"|'(?P<skey>[^']*)')"""
    r"""[^\S\n]*=[^\S\n]*(?:"(?P<dval>[^"]*)"|'(?P<sval>[^']*)')""",
    re.MULTILINE,
)


def _urls_from_toml_text(text: str) -> dict[str, str]:
    """Read `[project.urls]` without `tomllib` — what the 3.10 leg must do.

    Takes its TEXT rather than reading `_PYPROJECT` so the 3.11+ leg can pin it
    against `tomllib` on a sample (below); otherwise this whole branch is dead
    code on 3.14 and only the 3.10 job would ever notice it drifting.
    """
    # 3.10 has no `tomllib` (see `_pyproject_version`). Slice the `[project.urls]`
    # table out by hand: from its header to the next table header, so a `Key =
    # "..."` line under some LATER table can never be read as a project URL. The
    # header tolerates a trailing comment — `[project.urls]  # PyPI sidebar` is
    # legal TOML, and rejecting it would report every label missing on the 3.10
    # leg while the table sat there correct, pointing the fix at the wrong file.
    match = re.search(
        r"^\[project\.urls\][^\S\n]*(?:#[^\n]*)?$(.*?)(?=^\[|\Z)",
        text,
        re.MULTILINE | re.S,
    )
    if match is None:
        return {}
    urls: dict[str, str] = {}
    for entry in _URL_ENTRY_RE.finditer(match.group(1)):
        key = next(
            group
            for group in (entry["bare"], entry["dkey"], entry["skey"])
            if group is not None
        )
        urls[key] = entry["dval"] if entry["dval"] is not None else entry["sval"]
    return urls


def _project_urls() -> dict[str, str]:
    text = _PYPROJECT.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text).get("project", {}).get("urls", {})
    return _urls_from_toml_text(text)


# Everything TOML allows here that a bare-word/double-quote-only pattern drops:
# a commented header, a quoted key, a single-quoted value, a hyphenated key —
# plus a same-named key under a LATER table, which must not leak in.
_SAMPLE_URLS_TOML = """\
[project]
name = "sample"

[project.urls]  # PyPI sidebar
Homepage = "https://example.invalid/home"
"Bug Tracker" = 'https://example.invalid/bugs'
Source-Code = "https://example.invalid/src"

[tool.sample]
Homepage = "https://example.invalid/not-a-project-url"
"""


def test_fallback_toml_parse_agrees_with_tomllib():
    """The 3.10 leg parses by regex; pin that it reads what `tomllib` reads."""
    parsed = _urls_from_toml_text(_SAMPLE_URLS_TOML)
    assert parsed == {
        "Homepage": "https://example.invalid/home",
        "Bug Tracker": "https://example.invalid/bugs",
        "Source-Code": "https://example.invalid/src",
    }
    if sys.version_info >= (3, 11):
        import tomllib

        assert parsed == tomllib.loads(_SAMPLE_URLS_TOML)["project"]["urls"]


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
    # `Homepage` too, not just `Repository`: it is the field PyPI headlines and
    # the one that backfills the legacy `Home-page` that `pip show` prints, so a
    # stale copy-paste there is the most visible version of this bug. Read every
    # label through `.get` — a parse that came back `{}` should fail saying WHICH
    # label is wrong, not raise a bare `KeyError` naming no file.
    for label in ("Homepage", "Repository"):
        assert urls.get(label) == _REPO_URL, (
            f"pyproject.toml `[project.urls]` has {label}={urls.get(label)!r}, "
            f"expected {_REPO_URL}. PyPI would link at another project."
        )
    assert urls.get("Issues", "").startswith(_REPO_URL + "/"), (
        f"pyproject.toml `[project.urls]` has Issues={urls.get('Issues')!r}, "
        f"expected an issue tracker under {_REPO_URL}."
    )
