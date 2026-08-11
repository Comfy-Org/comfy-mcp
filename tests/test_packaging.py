"""Packaging invariants: the two version files, and the undeclared engine.

The packaged version lives in two files; pin that they agree.

`publish.yml`'s `check-version` job already compares both against the release
tag — but it runs on the `release: created` event, so it fires only AFTER the
tag and the GitHub Release exist. A mismatch caught there costs a deleted
release and tag, and a PyPI version number can never be reused. This pins the
same invariant at PR time, where the fix is an edit instead.

Read from the FILES rather than `importlib.metadata`: an editable install
carries the version recorded when it was installed, so a stale local env would
compare the wrong string in both directions. The files are the source of truth
the release checks against.

The second invariant is the deliberately UNDECLARED engine. comfy-cli is not a
dependency of this distribution — the binary that matters is whichever one
`PATH`/`COMFY_BIN` resolves to, so a pinned wheel would install a second copy
the tools may never call — and the price is that a bare `pip install comfy-mcp`
yields a server whose every tool errors until the user installs comfy-cli
themselves. That price is paid in the README, which installs both in one
command. The two halves are therefore one decision, and these tests pin them as
a pair: undeclared HERE requires documented THERE, so a future "fix" to either
half fails loudly instead of quietly leaving new users with a broken install (or
leaving the docs telling them to install something pip already did).
"""

import re
import sys
from pathlib import Path

import pytest

import comfy_mcp
from comfy_mcp import server

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"
_INIT = _ROOT / "src" / "comfy_mcp" / "__init__.py"
_README = _ROOT / "README.md"


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


def _requirement_name(requirement: str) -> str:
    """The PEP 503-normalized distribution name in a PEP 508 requirement.

    Normalized rather than compared literally, so `comfy_cli` / `Comfy.CLI` —
    the same distribution to pip — cannot slip past the assertion below. The
    strip comes first because pip accepts a padded requirement (`" comfy-cli"`)
    and both parsers below hand it over verbatim: matched unstripped, that
    entry would normalize to `""` and slip past the absence assertion. An entry
    with no leading name at all returns `""` rather than raising, so a
    malformed array fails on the assertion that reads it, not inside here.
    """
    match = re.match(r"[A-Za-z0-9._-]+", requirement.strip())
    return re.sub(r"[-_.]+", "-", match.group(0)).lower() if match else ""


def _dependencies_without_tomllib(text: str) -> list[str]:
    """`[project] dependencies`, parsed the way the Python 3.10 leg must.

    3.10 has no `tomllib`, and one field does not justify a TOML dependency
    (same call as `_pyproject_version` above). Slice the array literal — its
    terminator is anchored at column 0, which no line INSIDE the array is, so a
    `]` inside one of its comments cannot end the slice early — then SCAN the
    slice for quoted entries rather than matching one per line. TOML allows a
    literal string (`'comfy-cli>=1.14.0'`), several entries on one line, and a
    trailing `#` comment, none of which a line-anchored `^\\s*"..."` pattern
    sees; and an entry this parser silently drops is an entry
    `test_comfy_cli_is_not_a_declared_dependency` never gets to check.
    `test_the_fallback_parse_matches_tomllib` pins it against the real parser.
    """
    match = re.search(
        r"^dependencies\s*=\s*\[\n(.*?)^\]", text, re.MULTILINE | re.DOTALL
    )
    assert match is not None, "no `[project] dependencies` array in pyproject.toml"
    body = match.group(1)
    entries: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == "#":  # a comment runs to end of line
            newline = body.find("\n", index)
            index = len(body) if newline == -1 else newline + 1
        elif char in "\"'":
            end = index + 1
            while end < len(body) and body[end] != char:
                # Only a basic string can escape its own quote; `'…'` is literal.
                end += 2 if char == '"' and body[end] == "\\" else 1
            assert end < len(body), f"unterminated {char} string in dependencies"
            entries.append(body[index + 1 : end])
            index = end + 1
        else:
            index += 1
    return entries


def _runtime_dependencies() -> list[str]:
    """The requirement strings in `[project] dependencies`."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text)["project"]["dependencies"]
    return _dependencies_without_tomllib(text)


def test_runtime_dependencies_parse_as_expected():
    """Guard the parse above: it must find the deps that ARE declared.

    Without this, a regex that silently matched nothing (a reformatted array, a
    trailing-comment style change) would make the negative assertion below pass
    vacuously — the one failure mode a "this name is absent" test has.
    """
    names = {_requirement_name(req) for req in _runtime_dependencies()}
    assert {"mcp", "pydantic", "anyio"} <= names, names


@pytest.mark.skipif(sys.version_info < (3, 11), reason="no tomllib to compare against")
def test_the_fallback_parse_matches_tomllib():
    """Pin the 3.10 fallback against a real TOML parser, on the legs that have one.

    The check above only proves the fallback finds the three names it is told
    to look for — it stays green if a fourth entry is dropped, which is exactly
    how the absence assertion goes vacuous. Comparing the whole list catches a
    dropped entry whatever form it takes; the 3.14 leg is where a mis-parse the
    3.10 leg would swallow gets caught.
    """
    import tomllib

    text = _PYPROJECT.read_text(encoding="utf-8")
    assert (
        _dependencies_without_tomllib(text)
        == tomllib.loads(text)["project"]["dependencies"]
    )


@pytest.mark.parametrize(
    "array",
    [
        pytest.param('    "mcp>=1.0",\n    "comfy-cli>=1.14.0",\n', id="basic-string"),
        pytest.param("    'comfy-cli>=1.14.0',\n", id="literal-string"),
        pytest.param('    "mcp>=1.0", "comfy-cli>=1.14.0",\n', id="two-on-one-line"),
        pytest.param('    " comfy-cli>=1.14.0",\n', id="leading-whitespace"),
        pytest.param('    "comfy-cli>=1.14.0",  # why\n', id="trailing-comment"),
        pytest.param('    "comfy_cli>=1.14.0",\n', id="underscored-name"),
    ],
)
def test_a_declared_comfy_cli_is_visible_in_every_toml_spelling(array):
    """The absence assertion must SEE a comfy-cli however it is written.

    A test that says "this name is absent" is only as good as the parser
    underneath it: a spelling the parser skips reads as absent, and the
    tripwire silently stops being one. Each form here is valid TOML that pip
    accepts, and each was invisible to the line-anchored pattern this parser
    replaced. Run against the helper (not the real file, which declares no
    comfy-cli) — that is the only way to exercise the miss.
    """
    parsed = _dependencies_without_tomllib(f"dependencies = [\n{array}]\n")
    assert "comfy-cli" in {_requirement_name(req) for req in parsed}, parsed


def test_comfy_cli_is_not_a_declared_dependency():
    """comfy-cli stays undeclared — see this module's docstring for why.

    Not a style rule: the server invokes whatever `comfy` binary
    `PATH`/`COMFY_BIN` names, and enforces its floor at runtime against THAT
    binary. Declaring it here would install a copy into this environment that
    the tools may never call. Reversing the decision is allowed — it just has
    to be a decision, taken together with the README half below.
    """
    declared = [
        req for req in _runtime_dependencies() if _requirement_name(req) == "comfy-cli"
    ]
    assert not declared, (
        f"comfy-cli is declared as a runtime dependency ({declared}) — it is "
        "deliberately NOT one (pyproject.toml's comment, AGENTS.md 'Toolchain'). "
        "If that decision is being reversed on purpose, update this test, the "
        "pyproject comment, AGENTS.md, and the README quickstart together."
    )


def test_readme_quickstart_installs_the_engine_alongside():
    """The docs half of the decision above: install both, in one command.

    Keyed on the exact command a user copies, with the floor read from
    `_MIN_COMFY_CLI_STR` rather than hardcoded — so raising the floor in
    `server.py` fails here until the README says the new number, the same way
    the runtime error message does.
    """
    readme = _README.read_text(encoding="utf-8")
    command = f'pip install comfy-mcp "comfy-cli>={server._MIN_COMFY_CLI_STR}"'
    # Anchored to the Quickstart section (up to the next H2) — a mention buried
    # further down is not what a new user follows. `find`, not `index`: a
    # renamed heading should fail as this test's own message, not as a bare
    # `ValueError`, and Quickstart being the LAST H2 is a section to the end of
    # the file, not an error.
    start = readme.find("\n## Quickstart\n")
    assert start != -1, "README has no `## Quickstart` section to anchor this test to"
    end = readme.find("\n## ", start + 1)
    quickstart = readme[start : end if end != -1 else len(readme)]
    assert command in quickstart, (
        f"the README quickstart must install the engine alongside the server "
        f"(`{command}`): comfy-cli is not a declared dependency, so nothing at "
        "install time tells the user they need it."
    )


def test_no_document_pins_a_stale_comfy_cli_floor():
    """Every documented `comfy-cli >= …` names the CURRENT floor, not just the one above.

    The quickstart is keyed to `_MIN_COMFY_CLI_STR`, but the same pin is
    written out again in the README prerequisites and in `pyproject.toml`'s
    rationale. Without this, raising the floor in `server.py` fails on the one
    copy the test above reads and leaves the rest silently stale — telling a
    user to install a comfy-cli the runtime gate will then reject.

    Scoped to the DOCS. `server.py`'s own prose is left out on purpose: a
    per-tool minimum there may legitimately sit above the global floor.
    """
    floor = server._MIN_COMFY_CLI_STR
    pattern = r"comfy-cli\s*(?:>=|≥)\s*v?([0-9][0-9A-Za-z.+!-]*)"
    stale = {}
    for name in ("README.md", "AGENTS.md", "pyproject.toml"):
        pinned = set(re.findall(pattern, (_ROOT / name).read_text(encoding="utf-8")))
        if pinned - {floor}:
            stale[name] = sorted(pinned - {floor})
    assert not stale, (
        f"these docs pin a comfy-cli floor other than `{floor}` (the value of "
        f"`server._MIN_COMFY_CLI_STR`): {stale}. Raising the floor means "
        "updating every copy — the README quickstart AND prerequisites, and "
        "pyproject.toml's rationale — or the docs send users to a version the "
        "runtime gate rejects."
    )
