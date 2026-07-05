"""Mocked tests for the discovery tools (search_nodes / get_node / search_models).

These assert the exact ``comfy`` argv each tool composes (the thin-passthrough
contract) against a stubbed ``subprocess.run``, plus one error-envelope path
(local server not running) that must surface as ``ComfyCliError``.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from comfy_local_mcp import server


def _stub_comfy(monkeypatch, envelope: dict):
    """Stub shutil.which + subprocess.run; return the list that records each argv."""
    calls: list[list[str]] = []

    def fake(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(envelope), stderr=""
        )

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)
    return calls


def _ok(data):
    return {"type": "envelope", "ok": True, "data": data}


def test_search_nodes_argv(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([{"name": "KSampler"}]))
    assert server.search_nodes("sampler") == [{"name": "KSampler"}]
    # global flags first, then the subcommand + query verbatim
    assert calls[0] == [
        server.COMFY_BIN,
        "--json",
        "--where",
        "local",
        "nodes",
        "search",
        "sampler",
    ]


def test_get_node_argv(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok({"name": "KSampler", "inputs": {}}))
    assert server.get_node("KSampler") == {"name": "KSampler", "inputs": {}}
    assert calls[0][4:] == ["nodes", "show", "KSampler"]


def test_list_nodes_no_filters_bare_ls(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([{"name": "KSampler"}]))
    assert server.list_nodes() == [{"name": "KSampler"}]
    # no filters -> a bare `nodes ls`, no stray flags
    assert calls[0][4:] == ["nodes", "ls"]


def test_list_nodes_all_filters_in_order(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([]))
    server.list_nodes(
        produces="IMAGE",
        accepts="MODEL",
        category="loaders",
        pack="was-suite",
        label="Load",
    )
    assert calls[0][4:] == [
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


def test_list_nodes_omits_empty_filters(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([]))
    server.list_nodes(produces="IMAGE", category="sampling")
    # only the two non-empty filters are passed, in declared order
    assert calls[0][4:] == [
        "nodes",
        "ls",
        "--produces",
        "IMAGE",
        "--category",
        "sampling",
    ]


def test_nodes_upstream_without_limit(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([{"name": "CheckpointLoaderSimple"}]))
    assert server.nodes_upstream("KSampler") == [{"name": "CheckpointLoaderSimple"}]
    assert calls[0][4:] == ["nodes", "upstream", "KSampler"]


def test_nodes_upstream_with_limit(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([]))
    server.nodes_upstream("KSampler", limit=5)
    assert calls[0][4:] == ["nodes", "upstream", "KSampler", "--limit", "5"]


def test_nodes_downstream_without_limit(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([{"name": "VAEDecode"}]))
    assert server.nodes_downstream("KSampler") == [{"name": "VAEDecode"}]
    assert calls[0][4:] == ["nodes", "downstream", "KSampler"]


def test_nodes_downstream_with_limit(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([]))
    server.nodes_downstream("KSampler", limit=3)
    assert calls[0][4:] == ["nodes", "downstream", "KSampler", "--limit", "3"]


def test_nodes_path_defaults(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([]))
    server.nodes_path("MODEL", "IMAGE")
    # concrete int defaults are always passed, so the argv is deterministic
    assert calls[0][4:] == [
        "nodes",
        "path",
        "MODEL",
        "IMAGE",
        "--max-depth",
        "6",
        "--max-paths",
        "10",
    ]


def test_nodes_path_overrides(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([]))
    server.nodes_path("LATENT", "IMAGE", max_depth=3, max_paths=2)
    assert calls[0][4:] == [
        "nodes",
        "path",
        "LATENT",
        "IMAGE",
        "--max-depth",
        "3",
        "--max-paths",
        "2",
    ]


def test_nodes_types_argv(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok(["MODEL", "IMAGE", "LATENT"]))
    assert server.nodes_types() == ["MODEL", "IMAGE", "LATENT"]
    assert calls[0][4:] == ["nodes", "types"]


def test_nodes_categories_argv(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok({"loaders": {}}))
    assert server.nodes_categories() == {"loaders": {}}
    assert calls[0][4:] == ["nodes", "categories"]


def test_search_models_query_uses_search(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok(["sd_xl_base.safetensors"]))
    assert server.search_models(query="xl") == ["sd_xl_base.safetensors"]
    assert calls[0][4:] == ["models", "search", "xl"]


def test_search_models_folder_uses_list_folder(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok(["model.ckpt"]))
    assert server.search_models(folder="checkpoints") == ["model.ckpt"]
    assert calls[0][4:] == ["models", "list-folder", "checkpoints"]


def test_search_models_empty_lists_folders(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok(["checkpoints", "loras"]))
    assert server.search_models() == ["checkpoints", "loras"]
    assert calls[0][4:] == ["models", "list-folders"]


def test_search_models_query_takes_precedence_over_folder(monkeypatch):
    calls = _stub_comfy(monkeypatch, _ok([]))
    server.search_models(query="xl", folder="checkpoints")
    assert calls[0][4:] == ["models", "search", "xl"]


def test_discovery_surfaces_error_envelope(monkeypatch):
    """A local server-not-running envelope must raise, not return silently."""
    _stub_comfy(
        monkeypatch,
        {
            "type": "envelope",
            "ok": False,
            "error": {
                "code": "server_not_running",
                "message": "local ComfyUI not running",
            },
        },
    )
    with pytest.raises(server.ComfyCliError, match="server_not_running"):
        server.search_nodes("sampler")
