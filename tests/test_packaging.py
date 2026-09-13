"""Packaging invariants: the two version files, the undeclared engine, and
metadata that only PyPI, not this repo, ever renders.

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

The third invariant is `[project.urls]`, the links PyPI renders in its
sidebar. Nothing in this repo displays them, so their absence is invisible
here and only shows up on a published page — which is how 0.10.0 shipped with
`"project_urls": null` and a blank `Home-page`.

The fourth is `server.json`, the listing published to the official MCP
Registry. It is the same shape of problem one step further out: nothing in this
repo reads that file, and its consumer is a PUBLIC directory reached only from
`publish.yml`'s `publish-mcp-registry` job, on a `release: created` event, after
PyPI has already been written to irreversibly. A mistake there is discovered at
the worst possible moment, so the checks that can be made statically are made
here instead.
"""

import json
import re
import sys
from pathlib import Path

import pytest

import comfy_mcp
from comfy_mcp.server import _internal as server

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"
_INIT = _ROOT / "src" / "comfy_mcp" / "__init__.py"
_README = _ROOT / "README.md"
_SERVER_JSON = _ROOT / "server.json"


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
    assert {"fastmcp", "mcp", "pydantic", "anyio", "uvicorn"} <= names, names


def test_fastmcp_beta_and_protocol_engine_are_exact_pins():
    """A beta framework/protocol pair must not drift under an unrelated install."""

    dependencies = _runtime_dependencies()
    assert "fastmcp==4.0.0b3" in dependencies
    assert "mcp==2.0.0" in dependencies


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
    `server/_internal.py` fails here until the README says the new number, the same way
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
    rationale. Without this, raising the floor in `server/_internal.py` fails on the one
    copy the test above reads and leaves the rest silently stale — telling a
    user to install a comfy-cli the runtime gate will then reject.

    Scoped to the DOCS. `server/_internal.py`'s own prose is left out on purpose: a
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


# ---------------------------------------------------------------------------
# server.json — the official MCP Registry listing.
#
# `publish.yml`'s `publish-mcp-registry` job rewrites the two version fields
# from the release tag and hands the file to `mcp-publisher`. Everything ELSE in
# it ships exactly as committed, to a public directory, under the org's
# identity — and the failure modes are all silent here and loud there. So the
# invariants below are the ones the registry itself enforces at publish time,
# pinned at PR time where the fix is an edit.
# ---------------------------------------------------------------------------

# The namespace the org's registry grant covers is `io.github.Comfy-Org/*` and
# it is CASE-SENSITIVE — a publish as `io.github.comfy-org/...` is rejected 403.
_SERVER_NAME = "io.github.Comfy-Org/comfy-mcp"

# https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json:
# `description` is `maxLength: 100`, and the publisher rejects a longer one
# rather than truncating it.
_MAX_REGISTRY_DESCRIPTION = 100

# The registry accepts exactly this base URL for `registryType: pypi`
# (`model.RegistryURLPyPI`); anything else, including a trailing slash or the
# JSON API path, fails validation as a registry/type mismatch.
_PYPI_BASE_URL = "https://pypi.org"


def _server_json() -> dict:
    return json.loads(_SERVER_JSON.read_text(encoding="utf-8"))


def test_server_json_is_valid_json_naming_this_org_namespace():
    """The name is the identity the registry authorizes, and its case matters.

    The sibling cloud server's first publish attempt died on exactly this:
    `403 … You have permission to publish: io.github.Comfy-Org/*. Attempting to
    publish: io.github.comfy-org/comfy-cloud-mcp-server`. A lowercased org here
    would fail the same way — after the release is cut and PyPI is written.
    """
    assert _server_json().get("name") == _SERVER_NAME


def test_server_json_declares_the_pypi_package_and_no_remotes():
    """This is a LOCAL stdio server: a package to install, not a URL to call.

    The cloud MCP's listing is the opposite shape — a `remotes` entry pointing
    at its hosted endpoint — and it is the nearest file to copy from. A
    `remotes` block here would advertise this project as something a client can
    connect to over the network, which it is not.
    """
    server_json = _server_json()
    assert "remotes" not in server_json, (
        "server.json declares `remotes`. This server is stdio: the client "
        "launches it as a subprocess from the installed PyPI package. A "
        "`remotes` entry belongs to the cloud MCP, not to this one."
    )
    packages = server_json.get("packages")
    assert isinstance(packages, list) and len(packages) == 1, packages
    package = packages[0]
    assert package.get("registryType") == "pypi", package
    assert package.get("identifier") == "comfy-mcp", package
    assert package.get("registryBaseUrl") == _PYPI_BASE_URL, package
    assert package.get("transport") == {"type": "stdio"}, package


def test_server_json_versions_seed_from_the_packaged_version():
    """Both version fields, kept honest even though the job overwrites them.

    `publish-mcp-registry` rewrites `.version` AND `.packages[].version` from
    the tag, so a stale committed value never reaches the registry. It is still
    wrong to leave one behind: the file is what a reader (and `mcp-publisher
    validate`) sees, and a `packages[].version` that names a release which does
    not exist on PyPI is indistinguishable from a real defect.
    """
    server_json = _server_json()
    version = _pyproject_version()
    assert server_json.get("version") == version, (
        f"server.json version {server_json.get('version')!r} != packaged "
        f"version {version!r}. Both version bumps happen in one commit."
    )
    assert [package.get("version") for package in server_json["packages"]] == [
        version
    ], server_json["packages"]


def test_server_json_description_fits_the_registry_limit():
    """`maxLength: 100` in the schema, enforced by the publisher, not trimmed."""
    description = _server_json().get("description", "")
    assert 0 < len(description) <= _MAX_REGISTRY_DESCRIPTION, (
        f"server.json description is {len(description)} characters; the "
        f"registry schema caps it at {_MAX_REGISTRY_DESCRIPTION}."
    )


def test_server_json_links_point_at_this_repository():
    """A copy from the cloud server's listing would advertise the wrong project.

    `repository.url` is checked against the same constant `[project.urls]` uses,
    so the two cannot drift; `websiteUrl` is only required not to be the cloud
    server's page, because which docs page is right is an editorial call this
    test has no business making.
    """
    server_json = _server_json()
    assert server_json.get("repository", {}).get("url") == _REPO_URL, server_json
    assert server_json.get("repository", {}).get("source") == "github", server_json
    website = server_json.get("websiteUrl", "")
    assert website.startswith("https://"), website
    assert "/agent-tools/cloud" not in website, (
        f"server.json websiteUrl is {website!r}, the Comfy Cloud MCP's page. "
        "This listing is for the local server."
    )


# The registry's own boundary rule for the ownership token, from
# `internal/validators/registries/mcpname.go`: the matched name must be followed
# by end-of-content, a character that cannot continue a server name, or an HTML
# comment close. Anything else means the token was read as a PREFIX of some
# longer name and the publish is rejected — which is a real trap here, because
# the documented place to hide the token is an HTML comment.
_SERVER_NAME_CHARS = re.compile(r"[A-Za-z0-9._/-]")


def _token_has_boundary(rest: str) -> bool:
    if not rest:
        return True
    if not _SERVER_NAME_CHARS.fullmatch(rest[0]):
        return True
    return rest.startswith("-->") or rest.startswith("--!>")


@pytest.mark.parametrize(
    ("rest", "expected"),
    [
        pytest.param("", True, id="end-of-content"),
        pytest.param("\n", True, id="newline"),
        pytest.param(" -->", True, id="spaced-comment-close"),
        pytest.param("-->", True, id="glued-comment-close"),
        pytest.param("--!>", True, id="glued-bogus-comment-close"),
        pytest.param("<br>", True, id="html-tag"),
        pytest.param("-pro", False, id="longer-name"),
        pytest.param("/extra", False, id="longer-path"),
        pytest.param(".dev", False, id="longer-dotted"),
    ],
)
def test_the_boundary_helper_matches_the_registry_rule(rest, expected):
    """Pin the helper against the cases the registry's own matcher distinguishes.

    Without this, a helper that returned `True` unconditionally would make the
    README check below pass on a token the registry rejects — the usual failure
    mode of a test that asserts a string is "present and well-formed".
    """
    assert _token_has_boundary(rest) is expected


def test_readme_carries_the_registry_ownership_token():
    """The registry proves PyPI ownership by finding this string in the README.

    It fetches `https://pypi.org/pypi/comfy-mcp/<version>/json` and scans
    `info.description` — which IS this file, because `[project] readme` points
    here — for `mcp-name: <the server.json name>`. No token, no publish. Nothing
    else in the repo reads it, so deleting it as stray markup is easy and its
    cost lands on the next release rather than on the PR that removed it.
    """
    readme = _README.read_text(encoding="utf-8")
    name = _server_json()["name"]
    token = f"mcp-name: {name}"
    occurrences = [match.end() for match in re.finditer(re.escape(token), readme)]
    assert occurrences, (
        f"README.md does not contain the registry ownership token `{token}`. "
        "The MCP Registry reads it out of the PUBLISHED PyPI description to "
        "prove this repo owns the `comfy-mcp` distribution; without it "
        "publish.yml's registry job fails after the release is already cut."
    )
    assert any(_token_has_boundary(readme[end:]) for end in occurrences), (
        f"README.md contains `{token}` but every occurrence is glued to a "
        "following name character, so the registry reads it as a prefix of a "
        "longer name and rejects it. Put it on its own line."
    )
