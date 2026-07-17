"""Tests for the template tools — the search -> fetch -> run_workflow on-ramp.

These lock in the passthrough argv (global flags before the subcommand, same
rule the wrapper enforces) and the two behaviors the tools own on top of the
passthrough:
1. ``search_templates`` filters ``comfy templates ls`` client-side, since the
   CLI has no server-side filter arg.
2. ``fetch_template`` returns the ABSOLUTE output path so it can be handed
   straight to ``run_workflow``.
"""

from __future__ import annotations

import json
import os
import subprocess

from comfy_local_mcp import server


def _fake_run(envelope: dict):
    """Return a subprocess.run stand-in that captures the call and emits an envelope."""
    calls: list[dict] = []

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
        calls.append({"cmd": cmd, "env": env})
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(envelope), stderr=""
        )

    return fake, calls


def test_search_templates_argv(monkeypatch):
    """Passthrough: `comfy --json --where local templates ls`, empty query = full list."""
    data = ["flux_dev", "sd15_basic"]
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": data})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server.search_templates() == data

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["templates", "ls"]  # subcommand strictly after


def test_search_templates_filters_list_client_side(monkeypatch):
    """Non-empty query narrows the `ls` output (CLI has no filter arg)."""
    data = ["flux_dev", "flux_schnell", "sd15_basic"]
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: data)

    assert server.search_templates("flux") == ["flux_dev", "flux_schnell"]
    assert server.search_templates("FLUX") == [
        "flux_dev",
        "flux_schnell",
    ]  # case-insensitive
    assert server.search_templates("nope") == []


def test_search_templates_filters_dict_entries(monkeypatch):
    """Matches across any text field of a dict-shaped template entry."""
    data = [
        {"name": "flux_dev", "description": "Flux dev text-to-image"},
        {"name": "sd15_basic", "description": "Stable Diffusion 1.5"},
    ]
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: data)

    # Hit on description text, not just the name.
    assert server.search_templates("stable diffusion") == [data[1]]
    assert server.search_templates("text-to-image") == [data[0]]


def test_search_templates_unknown_shape_returned_unfiltered(monkeypatch):
    """An unexpected data shape is returned as-is rather than silently dropped."""
    data = {"count": 2}
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: data)
    assert server.search_templates("flux") == data


def test_get_template_argv(monkeypatch):
    """Passthrough: `comfy --json --where local templates show <name>`."""
    fake, calls = _fake_run(
        {"type": "envelope", "ok": True, "data": {"name": "flux_dev", "nodes": 12}}
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server.get_template("flux_dev") == {"name": "flux_dev", "nodes": 12}
    assert calls[0]["cmd"][4:] == ["templates", "show", "flux_dev"]


def test_fetch_template_argv_and_returns_abspath(monkeypatch, tmp_path):
    """Passthrough argv is `templates fetch <name> --out <path>`; returns the abs path."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": None})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    out = tmp_path / "flux.json"
    result = server.fetch_template("flux_dev", str(out))

    assert calls[0]["cmd"][4:] == ["templates", "fetch", "flux_dev", "--out", str(out)]
    assert result == str(out)  # tmp_path is already absolute
    assert os.path.isabs(result)


def test_fetch_template_resolves_relative_path(monkeypatch):
    """A relative out_path is returned as an absolute path (ready for run_workflow)."""
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: None)
    result = server.fetch_template("flux_dev", "flux.json")
    assert result == os.path.abspath("flux.json")
    assert os.path.isabs(result)
