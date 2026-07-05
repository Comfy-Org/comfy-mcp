"""Tests for the workflow slot-editing tools — list / set-slot / vary.

These lock in the passthrough argv (global flags before the subcommand, same
rule the wrapper enforces) for the three tools that let an agent parameterize a
fetched template — the ``fetch_template`` -> ``set_workflow_slot`` ->
``run_workflow`` loop — without hand-editing raw workflow JSON. The behaviors
they own on top of the passthrough:
1. ``set_workflow_slot`` passes each override as a positional ``ADDR=VALUE`` and
   defaults to ``--stdout`` (non-destructive), togglable off.
2. ``vary_workflow`` repeats ``--slot`` per address and forwards ``--out-dir``
   only when given.
"""

from __future__ import annotations

import json
import subprocess

from comfy_local_mcp import server


def _fake_run(envelope: dict):
    """Return a subprocess.run stand-in that captures the call and emits an envelope."""
    calls: list[dict] = []

    def fake(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
        calls.append({"cmd": cmd, "env": env})
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(envelope), stderr=""
        )

    return fake, calls


def test_list_workflow_slots_argv(monkeypatch):
    """Passthrough: `comfy --json --where local workflow slots <path>`."""
    data = [{"addr": "6.text", "value": "a cat"}, {"addr": "3.seed", "value": 42}]
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": data})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server.list_workflow_slots("/tmp/flux.json") == data

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["workflow", "slots", "/tmp/flux.json"]  # subcommand after


def test_set_workflow_slot_argv_default_stdout(monkeypatch):
    """Default: positional ADDR=VALUE overrides + trailing --stdout (non-destructive)."""
    fake, calls = _fake_run(
        {"type": "envelope", "ok": True, "data": {"modified": True}}
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    result = server.set_workflow_slot(
        "/tmp/flux.json", ["6.text=a red bicycle", "3.seed=42"]
    )
    assert result == {"modified": True}

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "/tmp/flux.json",
        "6.text=a red bicycle",  # each override passed as a positional ADDR=VALUE
        "3.seed=42",
        "--stdout",  # default: return the workflow, don't mutate the file
    ]


def test_set_workflow_slot_stdout_false_writes_in_place(monkeypatch):
    """stdout=False drops --stdout so comfy-cli writes the change back to the file."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": None})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    server.set_workflow_slot("/tmp/flux.json", ["3.seed=7"], stdout=False)

    cmd = calls[0]["cmd"]
    assert cmd[4:] == ["workflow", "set-slot", "/tmp/flux.json", "3.seed=7"]
    assert "--stdout" not in cmd


def test_vary_workflow_argv_repeats_slot_flag(monkeypatch):
    """Each address becomes its own `--slot "ADDR=[...]"`; no --out-dir when unset."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": {"variants": 3}})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    result = server.vary_workflow(
        "/tmp/flux.json", ["3.seed=[1,2,3]", "6.text=[cat,dog,fish]"]
    )
    assert result == {"variants": 3}

    cmd = calls[0]["cmd"]
    assert cmd[4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        "3.seed=[1,2,3]",
        "--slot",
        "6.text=[cat,dog,fish]",
    ]
    assert "--out-dir" not in cmd  # stdout NDJSON mode when out_dir is unset


def test_vary_workflow_forwards_out_dir(monkeypatch, tmp_path):
    """out_dir appends `--out-dir <dir>` so variants are written to files."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": None})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    out = tmp_path / "variants"
    server.vary_workflow("/tmp/flux.json", ["3.seed=[1,2]"], out_dir=str(out))

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        "3.seed=[1,2]",
        "--out-dir",
        str(out),
    ]
