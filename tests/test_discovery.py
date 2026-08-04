"""Mocked tests for the discovery tools (search_nodes / get_node / search_models).

These assert the exact ``comfy`` argv each tool composes (the thin-passthrough
contract) against a stubbed ``subprocess.run``, plus one error-envelope path
(local server not running) that must surface as ``ComfyCliError``.
"""

from __future__ import annotations

import pytest
from conftest import envelope

from comfy_mcp import server


def test_search_nodes_argv(patched_run):
    calls = patched_run(envelope(data=[{"name": "KSampler"}]))
    assert server.search_nodes("sampler") == [{"name": "KSampler"}]
    # global flags first, then the subcommand + query verbatim
    assert calls[0]["cmd"] == [
        server.COMFY_BIN,
        "--json",
        "--where",
        "local",
        "nodes",
        "search",
        "sampler",
    ]


# `search_nodes` / `get_node` leading-dash + NUL rejection is covered for the
# whole `nodes` family by `test_node_tools_reject_option_like_positionals` and
# `test_node_tools_reject_embedded_nul` below (landed on main in #86).


def test_get_node_argv(patched_run):
    calls = patched_run(envelope(data={"name": "KSampler", "inputs": {}}))
    assert server.get_node("KSampler") == {"name": "KSampler", "inputs": {}}
    assert calls[0]["cmd"][4:] == ["nodes", "show", "KSampler"]


def test_list_nodes_no_filters_bare_ls(patched_run):
    calls = patched_run(envelope(data=[{"name": "KSampler"}]))
    assert server.list_nodes() == [{"name": "KSampler"}]
    # no filters -> a bare `nodes ls`, no stray flags
    assert calls[0]["cmd"][4:] == ["nodes", "ls"]


def test_list_nodes_all_filters_in_order(patched_run):
    calls = patched_run(envelope(data=[]))
    server.list_nodes(
        produces="IMAGE",
        accepts="MODEL",
        category="loaders",
        pack="was-suite",
        label="Load",
    )
    assert calls[0]["cmd"][4:] == [
        "nodes",
        "ls",
        "--produces",
        "IMAGE",
        "--accepts",
        "MODEL",
        "--category",
        "loaders",
        "--pack",
        "was-suite",
        "--label",
        "Load",
    ]


def test_list_nodes_omits_empty_filters(patched_run):
    calls = patched_run(envelope(data=[]))
    server.list_nodes(produces="IMAGE", category="sampling")
    # only the two non-empty filters are passed, in declared order
    assert calls[0]["cmd"][4:] == [
        "nodes",
        "ls",
        "--produces",
        "IMAGE",
        "--category",
        "sampling",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"produces": "--help"},
        {"accepts": "-x"},
        {"category": "--pack"},
        {"pack": "-p"},
        {"label": "--label"},
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_list_nodes_rejects_leading_dash_filter_values(patched_run, kwargs):
    """A filter value starting with '-' is rejected before it reaches comfy-cli argv."""
    calls = patched_run(envelope(data=[]))
    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.list_nodes(**kwargs)
    # refused before the spawn, not after
    assert calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"produces": "a\0"},
        {"accepts": "a\0"},
        {"category": "a\0"},
        {"pack": "a\0"},
        {"label": "a\0"},
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_list_nodes_rejects_embedded_nul_filter_values(patched_run, kwargs):
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    calls = patched_run(envelope(data=[]))
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.list_nodes(**kwargs)
    assert calls == []


def test_list_nodes_still_passes_a_normal_filter_value(patched_run):
    """The guards are value-shape only — an ordinary filter still reaches argv."""
    calls = patched_run(envelope(data=[]))
    server.list_nodes(produces="IMAGE")
    assert calls[0]["cmd"][4:] == ["nodes", "ls", "--produces", "IMAGE"]


def test_nodes_upstream_without_limit(patched_run):
    calls = patched_run(envelope(data=[{"name": "CheckpointLoaderSimple"}]))
    assert server.nodes_upstream("KSampler") == [{"name": "CheckpointLoaderSimple"}]
    assert calls[0]["cmd"][4:] == ["nodes", "upstream", "KSampler"]


def test_nodes_upstream_with_limit(patched_run):
    calls = patched_run(envelope(data=[]))
    server.nodes_upstream("KSampler", limit=5)
    assert calls[0]["cmd"][4:] == ["nodes", "upstream", "KSampler", "--limit", "5"]


def test_nodes_downstream_without_limit(patched_run):
    calls = patched_run(envelope(data=[{"name": "VAEDecode"}]))
    assert server.nodes_downstream("KSampler") == [{"name": "VAEDecode"}]
    assert calls[0]["cmd"][4:] == ["nodes", "downstream", "KSampler"]


def test_nodes_downstream_with_limit(patched_run):
    calls = patched_run(envelope(data=[]))
    server.nodes_downstream("KSampler", limit=3)
    assert calls[0]["cmd"][4:] == ["nodes", "downstream", "KSampler", "--limit", "3"]


def test_nodes_path_defaults(patched_run):
    calls = patched_run(envelope(data=[]))
    server.nodes_path("MODEL", "IMAGE")
    # concrete int defaults are always passed, so the argv is deterministic
    assert calls[0]["cmd"][4:] == [
        "nodes",
        "path",
        "MODEL",
        "IMAGE",
        "--max-depth",
        "6",
        "--max-paths",
        "10",
    ]


def test_nodes_path_overrides(patched_run):
    calls = patched_run(envelope(data=[]))
    server.nodes_path("LATENT", "IMAGE", max_depth=3, max_paths=2)
    assert calls[0]["cmd"][4:] == [
        "nodes",
        "path",
        "LATENT",
        "IMAGE",
        "--max-depth",
        "3",
        "--max-paths",
        "2",
    ]


def test_nodes_types_argv(patched_run):
    calls = patched_run(envelope(data=["MODEL", "IMAGE", "LATENT"]))
    assert server.nodes_types() == ["MODEL", "IMAGE", "LATENT"]
    assert calls[0]["cmd"][4:] == ["nodes", "types"]


def test_nodes_categories_argv(patched_run):
    calls = patched_run(envelope(data={"loaders": {}}))
    assert server.nodes_categories() == {"loaders": {}}
    assert calls[0]["cmd"][4:] == ["nodes", "categories"]


def test_node_dependencies_bare_argv(patched_run):
    """No arguments -> a bare `node deps`, which reports every installed pack."""
    data = {"workspace": "/ws", "python": "/ws/.venv/bin/python", "packs": []}
    calls = patched_run(envelope(data=data))
    assert server.node_dependencies() == data
    # `node` (pack-level), NOT the `nodes` class-introspection family
    assert calls[0]["cmd"][4:] == ["node", "deps"]
    assert calls[0]["timeout"] == 60.0


def test_node_dependencies_pack_is_a_positional(patched_run):
    calls = patched_run(envelope(data={"packs": []}))
    server.node_dependencies(pack="comfyui-impact-pack")
    assert calls[0]["cmd"][4:] == ["node", "deps", "comfyui-impact-pack"]


def test_node_dependencies_registry_id_uses_the_flag(patched_run):
    calls = patched_run(envelope(data={"packs": []}))
    server.node_dependencies(registry_id="comfyui-impact-pack")
    assert calls[0]["cmd"][4:] == ["node", "deps", "--registry", "comfyui-impact-pack"]


def test_node_dependencies_both_are_additive(patched_run):
    """The two are not exclusive — comfy-cli emits a row for each.

    Naming the same id both ways is the deliberate "installed vs published"
    comparison, so the positional must stay ahead of the option rather than
    either one replacing the other.
    """
    calls = patched_run(envelope(data={"packs": []}))
    server.node_dependencies(pack="was-suite", registry_id="was-suite")
    assert calls[0]["cmd"][4:] == [
        "node",
        "deps",
        "was-suite",
        "--registry",
        "was-suite",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [{"pack": "--help"}, {"pack": "-p"}, {"registry_id": "--registry"}],
    ids=["pack-long", "pack-short", "registry_id"],
)
def test_node_dependencies_rejects_leading_dash(patched_run, kwargs):
    """Both values are refused before the spawn, `get_node`/`list_nodes`-style."""
    calls = patched_run(envelope(data={"packs": []}))
    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.node_dependencies(**kwargs)
    assert calls == []


@pytest.mark.parametrize(
    "kwargs",
    [{"pack": "was\0suite"}, {"registry_id": "was\0suite"}],
    ids=lambda kw: next(iter(kw)),
)
def test_node_dependencies_rejects_embedded_nul(patched_run, kwargs):
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    calls = patched_run(envelope(data={"packs": []}))
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.node_dependencies(**kwargs)
    assert calls == []


def test_node_dependencies_returns_the_envelope_payload(patched_run):
    """Thin passthrough: comfy-cli's `data` is returned verbatim, unreshaped."""
    data = {
        "workspace": "/ws",
        "python": "/ws/.venv/bin/python",
        "compiled_lock": {"present": False, "path": None},
        "packs": [
            {
                "pack": "demo-pack",
                "path": "custom_nodes/demo-pack",
                "status": "installed",
                "requirement_files": ["requirements.txt"],
                "requirements": [
                    {
                        "raw": "numpy==1.26.4",
                        "name": "numpy",
                        "specifier": "==1.26.4",
                        "installed": None,
                        "status": "missing",
                        "source": "requirements.txt",
                    }
                ],
                "summary": {
                    "satisfied": 0,
                    "mismatch": 0,
                    "missing": 1,
                    "unparseable": 0,
                    "unknown": 0,
                },
            }
        ],
        "warnings": [],
    }
    patched_run(envelope(data=data))
    assert server.node_dependencies() == data


def test_node_dependencies_degrades_without_the_verb(patched_run):
    """A comfy-cli predating `comfy node deps` reads as a version gap, not a break.

    The verb ships in releases AFTER the `_MIN_COMFY_CLI` floor, so an install
    that satisfies the version guard can still lack it — the common path today.
    Verified against the released 1.13.0: a missing SUBcommand of `node` exits 2
    with no envelope and Click's `No such command 'deps'.` on stderr, exactly the
    shape `_is_missing_verb_error` already matches for a top-level verb.
    """
    patched_run(
        "",
        returncode=2,
        stderr="Usage: comfy node [OPTIONS] COMMAND\nNo such command 'deps'.",
    )

    result = server.node_dependencies()

    assert result["unsupported"] is True
    assert "node_dependencies unavailable" in result["error"]
    assert "comfy node deps" in result["error"]
    # None of the raw wrapper/CLI text leaks through.
    assert "No such command" not in result["error"]
    assert "Usage: comfy" not in result["error"]
    assert "returned no JSON" not in result["error"]


def test_node_dependencies_degrades_through_a_rich_panel(patched_run):
    """Typer wraps the same error in a bordered, width-wrapped rich panel.

    `_normalize_cli_text` folds the box glyphs and the wrap away, so the degrade
    must not depend on the terminal width the child happened to render at. This
    is the stderr the real 1.13.0 emits, box characters and all.
    """
    patched_run(
        "",
        returncode=2,
        stderr=(
            "Usage: comfy node [OPTIONS] COMMAND [ARGS]...\n"
            "Try 'comfy node --help' for help.\n"
            "╭─ Error ─────────────────────╮\n"
            "│ No such command\n"
            "│ 'deps'.                     │\n"
            "╰─────────────────────────────╯\n"
        ),
    )

    assert server.node_dependencies()["unsupported"] is True


def test_node_dependencies_keeps_a_real_error_raw(patched_run):
    """A verb comfy-cli DID dispatch must never be waved through as a gap.

    No workspace is the case that matters: comfy-cli answers with an
    `not_in_workspace` envelope, and the agent has to see it to know the fix is
    `comfy install`, not a comfy-cli upgrade.
    """
    patched_run(
        envelope(
            ok=False,
            error={
                "code": "not_in_workspace",
                "message": "ComfyUI workspace not found.",
            },
        )
    )

    with pytest.raises(server.ComfyCliError, match="not_in_workspace"):
        server.node_dependencies()


def test_node_dependencies_relayed_phrase_is_not_unsupported(patched_run):
    """A failure that merely QUOTES the phrase, inside an envelope, stays raw.

    `_is_missing_verb_error` requires the no-envelope + usage-exit pair exactly
    so a nested error relaying "No such command 'deps'" from somewhere else — a
    pack's own build hook, a pip/git call comfy-cli shelled out to — cannot be
    mistaken for the verb itself being absent.
    """
    patched_run(
        envelope(
            ok=False,
            error={
                "code": "pack_scan_failed",
                "message": "a pack hook failed: No such command 'deps'.",
            },
        ),
        returncode=2,
    )

    with pytest.raises(server.ComfyCliError, match="pack_scan_failed"):
        server.node_dependencies()


def test_node_dependencies_different_verb_is_not_unsupported(patched_run):
    """A "no such command" naming a DIFFERENT verb is not this tool's gap.

    A CLI new enough to have `node deps` can still reject something the verb
    shells out to; degrading on that would assert nothing is broken.
    """
    patched_run(
        "",
        returncode=2,
        stderr="Error: No such command 'deps-in-workflow'.",
    )

    with pytest.raises(server.ComfyCliError):
        server.node_dependencies()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pack": "no such command 'deps'"},
        {"registry_id": "no such command 'deps'"},
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_node_dependencies_echoed_phrase_is_not_unsupported(patched_run, kwargs):
    """A caller cannot forge the version gap through its own argument.

    Click echoes an offending value verbatim in a usage error, on the same exit 2
    with no envelope `_is_missing_verb_error` reads — the one route to a false
    `unsupported` its two conditions cannot close. A real failure must stay a
    real failure rather than becoming "your comfy-cli is just too old".
    """
    value = next(iter(kwargs.values()))
    patched_run(
        "",
        returncode=2,
        stderr=(
            "Usage: comfy node deps [OPTIONS] [PACK]\n"
            f"Error: Invalid value for '[PACK]': {value!r} is not an installed pack."
        ),
    )

    with pytest.raises(server.ComfyCliError):
        server.node_dependencies(**kwargs)


def test_node_dependencies_degrades_with_a_pack_argument(patched_run):
    """The echoed-input check must not cost the genuine degrade.

    Discounting the caller's own text is subtraction, not a veto: an ordinary
    `pack` shares no wording with Click's message, so the parser's own phrase
    survives and the version gap still reports as one.
    """
    patched_run(
        "",
        returncode=2,
        stderr=("Usage: comfy node [OPTIONS] COMMAND\nError: No such command 'deps'."),
    )

    assert server.node_dependencies(pack="comfyui-impact-pack")["unsupported"] is True


def test_node_dependencies_degrades_without_the_registry_option(patched_run):
    """The OPTION half of the version gap, the way `download_model` covers it.

    A comfy-cli that HAS `node deps` but not `--registry` never matches the verb
    pattern, so without this it falls through as the raw usage dump the degrade
    exists to replace.
    """
    patched_run(
        "",
        returncode=2,
        stderr=(
            "Usage: comfy node deps [OPTIONS] [PACK]\n"
            "Try 'comfy node deps --help' for help.\n"
            "Error: No such option: --registry"
        ),
    )

    result = server.node_dependencies(registry_id="comfyui-impact-pack")

    assert result["unsupported"] is True
    assert "--registry" in result["error"]
    # Points at the half that still works rather than dead-ending.
    assert "registry_id empty" in result["error"]
    assert "No such option" not in result["error"]


def test_node_dependencies_registry_option_gap_needs_the_argument(patched_run):
    """With `registry_id` empty the flag is never on the command line.

    So a "no such option: --registry" can only have been relayed from elsewhere,
    and degrading on it would assert this call is fine when it is not.
    """
    patched_run(
        "",
        returncode=2,
        stderr="Error: No such option: --registry",
    )

    with pytest.raises(server.ComfyCliError):
        server.node_dependencies(pack="comfyui-impact-pack")


@pytest.mark.parametrize("label", ["pack", "registry_id"])
def test_node_dependencies_rejects_an_oversized_id(patched_run, label):
    """An id past `ARG_MAX` fails `execve`, not the lookup — refuse it first.

    And the error must report the SIZE: `_reject_option_like` would otherwise
    echo a megabyte-long value back through the response and the failure log.
    """
    calls = patched_run(envelope(data={"packs": []}))
    oversized = "-" + "x" * server._MAX_NODE_PACK_ID_LEN

    with pytest.raises(server.ComfyCliError, match="exceeds the") as excinfo:
        server.node_dependencies(**{label: oversized})

    assert oversized not in str(excinfo.value)
    assert calls == []


def test_search_models_query_uses_search(patched_run):
    # BE-2952: comfy-cli 1.12's `models search` takes the query as `--text`,
    # not a positional — a positional exits 2 ("returned no JSON (exit 2)").
    calls = patched_run(envelope(data=["sd_xl_base.safetensors"]))
    assert server.search_models(query="xl") == ["sd_xl_base.safetensors"]
    assert calls[0]["cmd"][4:] == ["models", "search", "--text", "xl"]


def test_search_models_folder_uses_list_folder(patched_run):
    calls = patched_run(envelope(data=["model.ckpt"]))
    assert server.search_models(folder="checkpoints") == ["model.ckpt"]
    assert calls[0]["cmd"][4:] == ["models", "list-folder", "checkpoints"]


def test_search_models_empty_lists_folders(patched_run):
    calls = patched_run(envelope(data=["checkpoints", "loras"]))
    assert server.search_models() == ["checkpoints", "loras"]
    assert calls[0]["cmd"][4:] == ["models", "list-folders"]


def test_search_models_query_takes_precedence_over_folder(patched_run):
    calls = patched_run(envelope(data=[]))
    server.search_models(query="xl", folder="checkpoints")
    assert calls[0]["cmd"][4:] == ["models", "search", "--text", "xl"]


@pytest.mark.parametrize("folder", ["--pack", "-c"])
def test_search_models_rejects_leading_dash_folder(patched_run, folder):
    """`folder` is a bare positional — comfy-cli reads a leading dash as an option."""
    calls = patched_run(envelope(data=[]))
    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.search_models(folder=folder)
    # refused before the spawn, not after
    assert calls == []


def test_search_models_rejects_an_oversized_folder(no_spawn):
    """An oversized `folder` is refused before it can reach argv.

    `folder` is a models-SUBFOLDER positional — path-shaped, so it takes the
    same ceiling as `download_model`'s `relative_path` and for the same reason:
    an oversized argv string is rejected by the OS with an `OSError` (`E2BIG`)
    `_run_comfy_raw` never converts, because its `try` wraps only
    `communicate()` and not the `Popen(...)` that raises.
    """
    oversized = "checkpoints/" + "f" * server._MAX_PATH_ARG_LEN

    with pytest.raises(server.ComfyCliError, match="exceeds") as excinfo:
        server.search_models(folder=oversized)

    # Length-not-value: the size check runs ahead of both value guards, whose
    # echoes would name the value instead of its size.
    assert oversized not in str(excinfo.value)


def test_search_models_allows_a_folder_at_the_ceiling(patched_run):
    """The boundary value itself rides through as the `list-folder` positional."""
    calls = patched_run(envelope(data=[]))
    prefix = "checkpoints/"
    at_ceiling = prefix + "f" * (server._MAX_PATH_ARG_LEN - len(prefix))
    assert len(at_ceiling) == server._MAX_PATH_ARG_LEN

    server.search_models(folder=at_ceiling)

    assert calls[0]["cmd"][4:] == ["models", "list-folder", at_ceiling]


def test_search_models_leaves_the_free_form_query_uncapped(patched_run):
    """The `--text` query is deliberately NOT capped — it is prompt-shaped.

    The counterpart to the `folder` cap above, and the reason the sweep is
    scoped to path-shaped values: a path-sized ceiling on a free-form search
    term would refuse a legitimately long query while buying nothing a caller
    cannot already spend by sending more of them. See the SCOPE note at
    `_MAX_URL_LEN`.
    """
    calls = patched_run(envelope(data=[]))
    long_query = "q" * (server._MAX_PATH_ARG_LEN + 1)

    server.search_models(query=long_query)

    assert calls[0]["cmd"][4:] == ["models", "search", "--text", long_query]


@pytest.mark.parametrize("query", ["-fp16", "-fp8-e4m3fn", "--help"])
def test_search_models_allows_leading_dash_query(patched_run, query):
    """`--text` is free-form filename matching: a leading dash is data, not a flag.

    Click takes the token after a value-taking option verbatim, so comfy-cli
    receives these as the search term — and `-fp16` / `-fp8` are ordinary model
    filename substrings with no other spelling. Guarding here would refuse a
    working search. Contrast `folder` above, which really is a positional.
    """
    calls = patched_run(envelope(data=[]))
    server.search_models(query=query)
    assert calls[0]["cmd"][4:] == ["models", "search", "--text", query]


@pytest.mark.parametrize(
    "kwargs",
    [{"query": "x\0l"}, {"folder": "check\0points"}],
    ids=lambda kw: next(iter(kw)),
)
def test_search_models_rejects_embedded_nul(patched_run, kwargs):
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    calls = patched_run(envelope(data=[]))
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.search_models(**kwargs)
    assert calls == []


def test_search_models_empty_values_still_select_the_mode(patched_run):
    """Empty stays the 'mode not selected' signal — the guards must not reject it."""
    calls = patched_run(envelope(data=["checkpoints"]))
    # empty query falls through to the folder mode, empty folder to list-folders
    server.search_models(query="", folder="checkpoints")
    server.search_models(query="", folder="")
    assert calls[0]["cmd"][4:] == ["models", "list-folder", "checkpoints"]
    assert calls[1]["cmd"][4:] == ["models", "list-folders"]


def test_discover_defaults_to_schemas_only(patched_run):
    """The default MUST forward `--schemas-only`.

    The full command surface is ~5x the schemas bundle and ~1.8x the 25,000
    tokens Claude Code's `MAX_MCP_OUTPUT_TOKENS` defaults to (caps are set by the
    client, not by MCP) — and that cap truncates rather than rejects, so the full
    mode hands back JSON cut mid-structure.
    Defaulting to the slim mode is what keeps this tool's response parseable.
    """
    calls = patched_run(envelope(data={"schemas": {}}))
    assert server.discover() == {"schemas": {}}
    assert calls[0]["cmd"] == [
        server.COMFY_BIN,
        "--json",
        "--where",
        "local",
        "discover",
        "--schemas-only",
    ]


def test_discover_schemas_only_false_omits_the_flag(patched_run):
    """`schemas_only=False` still reaches the full surface — flag simply absent."""
    calls = patched_run(envelope(data={"commands": {}}))
    assert server.discover(schemas_only=False) == {"commands": {}}
    assert calls[0]["cmd"][4:] == ["discover"]


def test_discovery_surfaces_error_envelope(patched_run):
    """A local server-not-running envelope must raise, not return silently."""
    patched_run(
        envelope(
            ok=False,
            error={
                "code": "server_not_running",
                "message": "local ComfyUI not running",
            },
        )
    )
    with pytest.raises(server.ComfyCliError, match="server_not_running"):
        server.search_nodes("sampler")


def test_node_tools_reject_option_like_positionals(monkeypatch):
    """Every bare positional on the `nodes` verbs refuses a leading-dash value.

    `search_nodes`/`get_node`/`nodes_upstream`/`nodes_downstream`/`nodes_path`
    all splat their caller string in as a positional, so comfy-cli reads a
    dash-leading value as an option — sharpest on `upstream`/`downstream`
    (beside their own `--limit`) and on `path`, where consuming the first type
    as a flag shifts the second into its slot.
    """

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    for call in (
        lambda: server.search_nodes("--help"),
        lambda: server.get_node("--help"),
        lambda: server.nodes_upstream("--help"),
        lambda: server.nodes_downstream("--help"),
        lambda: server.nodes_path("--help", "IMAGE"),
        lambda: server.nodes_path("MODEL", "--help"),
    ):
        with pytest.raises(server.ComfyCliError, match="leading '-'"):
            call()


def test_nodes_path_still_passes_negative_bounds_through(patched_run):
    """The guard covers the two types only — the int bounds are untouched.

    They ride behind `--max-depth`/`--max-paths` as option values (Click takes
    those verbatim), so even a negative bound is comfy-cli's to reject, not the
    wrapper's to refuse for looking dash-leading.
    """
    calls = patched_run(envelope(data=[]))
    server.nodes_path("MODEL", "IMAGE", max_depth=-1, max_paths=-2)
    assert calls[0]["cmd"][4:] == [
        "nodes",
        "path",
        "MODEL",
        "IMAGE",
        "--max-depth",
        "-1",
        "--max-paths",
        "-2",
    ]


def test_node_tools_reject_embedded_nul(monkeypatch):
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError.

    Orthogonal to the leading-dash guard: `subprocess` cannot carry a NUL in
    argv at all, so it is refused wherever the value rides.
    """

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    for call in (
        lambda: server.search_nodes("samp\0ler"),
        lambda: server.get_node("KSamp\0ler"),
        lambda: server.nodes_upstream("KSamp\0ler"),
        lambda: server.nodes_downstream("KSamp\0ler"),
        lambda: server.nodes_path("MOD\0EL", "IMAGE"),
        lambda: server.nodes_path("MODEL", "IMA\0GE"),
    ):
        with pytest.raises(server.ComfyCliError, match="embedded NUL"):
            call()
