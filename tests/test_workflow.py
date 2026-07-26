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

import pytest
from conftest import envelope

from comfy_local_mcp import server


def test_list_workflow_slots_argv(patched_run):
    """Passthrough: `comfy --json --where local workflow slots <path>`."""
    data = [{"addr": "6.text", "value": "a cat"}, {"addr": "3.seed", "value": 42}]
    calls = patched_run(envelope(data=data))

    assert server.list_workflow_slots("/tmp/flux.json") == data

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["workflow", "slots", "/tmp/flux.json"]  # subcommand after


def test_set_workflow_slot_argv_default_stdout(patched_run):
    """Default: positional ADDR=VALUE overrides + trailing --stdout (non-destructive)."""
    calls = patched_run(envelope(data={"modified": True}))

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


def test_set_workflow_slot_stdout_false_writes_in_place(patched_run):
    """stdout=False drops --stdout so comfy-cli writes the change back to the file."""
    calls = patched_run(envelope(data=None))

    server.set_workflow_slot("/tmp/flux.json", ["3.seed=7"], stdout=False)

    cmd = calls[0]["cmd"]
    assert cmd[4:] == ["workflow", "set-slot", "/tmp/flux.json", "3.seed=7"]
    assert "--stdout" not in cmd


def test_set_workflow_slot_rejects_option_like_override(monkeypatch):
    """A leading-dash override is refused before any child spawns.

    Splatted in as a positional it would BE the flag — `"--stdout"` would flip
    the in-place-write behavior the ``stdout`` argument owns.
    """

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.set_workflow_slot("/tmp/flux.json", ["--stdout"])

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.set_workflow_slot("/tmp/flux.json", ["6.text=x", "--stdout"])


def test_workflow_path_positional_rejects_option_like(monkeypatch):
    """The sibling `workflow_path` positional is guarded too, not just the overrides.

    All three tools splat the path in bare, so a leading-dash path is read as a
    flag: for `set-slot` that shifts the first override into the path slot,
    which is the very injection the override guard exists to stop. The error
    names the escape hatch — a genuinely dash-leading filename works as `./-x`.
    """

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    with pytest.raises(server.ComfyCliError, match=r"leading '-'.*\./"):
        server.set_workflow_slot("--stdout", ["6.text=x"])

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.list_workflow_slots("--stdout")

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.vary_workflow("--out-dir", ["3.seed=[1,2]"])


def test_workflow_path_guard_allows_dot_slash_dash_name(patched_run):
    """The documented escape hatch actually works: `./-flux.json` is not refused."""
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot("./-flux.json", ["6.text=x"])

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "./-flux.json",
        "6.text=x",
        "--stdout",
    ]


def test_workflow_tools_reject_embedded_nul(monkeypatch):
    """A NUL anywhere surfaces as ComfyCliError, not subprocess's bare ValueError.

    Orthogonal to the leading-dash guard: `subprocess` cannot carry a NUL in
    argv at all, so it is refused on option values (`--slot`, `--out-dir`) too,
    not just on the bare positionals.
    """

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    for call in (
        lambda: server.list_workflow_slots("/tmp/f\0.json"),
        lambda: server.set_workflow_slot("/tmp/f\0.json", ["6.text=x"]),
        lambda: server.set_workflow_slot("/tmp/f.json", ["6.text=\0"]),
        lambda: server.vary_workflow("/tmp/f\0.json", ["3.seed=[1,2]"]),
        lambda: server.vary_workflow("/tmp/f.json", ["3.seed=\0"]),
        lambda: server.vary_workflow("/tmp/f.json", ["3.seed=[1,2]"], out_dir="/o\0"),
    ):
        with pytest.raises(server.ComfyCliError, match="embedded NUL"):
            call()


def test_vary_workflow_option_values_are_not_guarded(patched_run):
    """Option VALUES stay unguarded on purpose — only bare positionals are injectable.

    comfy-cli is Click-backed and Click takes the token after a value-taking
    option verbatim, so `--out-dir --slot` parses as `out_dir="--slot"` rather
    than shifting anything. `slots`/`out_dir` therefore ride through untouched;
    the guard above them is for `workflow_path`, which IS a bare positional.
    """
    calls = patched_run(envelope(data={"variants": 2}))

    server.vary_workflow("/tmp/flux.json", ["-3.seed=[1,2]"], out_dir="-out")

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        "-3.seed=[1,2]",
        "--out-dir",
        "-out",
    ]


def test_set_workflow_slot_guard_leaves_valid_overrides_alone(patched_run):
    """The guard reads the override's FIRST character only: `-` inside a VALUE is fine."""
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot("/tmp/flux.json", ["6.text=x", "4.ckpt=sd-xl --turbo"])

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "/tmp/flux.json",
        "6.text=x",
        "4.ckpt=sd-xl --turbo",
        "--stdout",
    ]


def test_vary_workflow_argv_repeats_slot_flag(patched_run):
    """Each address becomes its own `--slot "ADDR=[...]"`; no --out-dir when unset."""
    calls = patched_run(envelope(data={"variants": 3}))

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


def test_vary_workflow_forwards_out_dir(patched_run, tmp_path):
    """out_dir appends `--out-dir <dir>` so variants are written to files."""
    calls = patched_run(envelope(data=None))

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
