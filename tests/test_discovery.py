"""Mocked tests for the discovery tools (search_nodes / get_node / search_models).

These assert the exact ``comfy`` argv each tool composes (the thin-passthrough
contract) against a stubbed ``subprocess.run``, plus one error-envelope path
(local server not running) that must surface as ``ComfyCliError``.
"""

from __future__ import annotations

import pytest
from conftest import envelope

from comfy_local_mcp import server


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
