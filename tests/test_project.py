"""Project anchoring: `COMFY_PROJECT` -> every comfy-cli spawn's `cwd=`, and
the `project` status/init tool.

comfy-cli 1.15.0's `project/1` convention (`comfy project init` / `status`)
resolves the GOVERNING project by walking up from its OWN process cwd only —
no `--project` flag, no env var it reads itself (verified: `COMFY_PROJECT` set
with an ungoverned cwd still returns `project_not_found`). An MCP client's cwd
is arbitrary and unrelated to any project the user has in mind, so this server
anchors every spawn from the OUTSIDE instead: `cwd=` on its own subprocess
calls, resolved by `server._project_root` from `COMFY_PROJECT`. These lock in:

1. `_project_root` resolution/caching/validation on its own, including that a
   RELATIVE value is rejected exactly like a missing/non-directory one — it
   would otherwise resolve against this server's own client-assigned cwd,
   reintroducing the non-determinism anchoring exists to remove.
2. Unset -> every spawn path (plain / async / streaming) passes no `cwd`,
   byte-identical to before this feature existed.
3. Set + valid -> every spawn path passes it.
4. Set + invalid (missing, non-directory, or relative) -> the first comfy-cli
   call raises, before anything spawns.
5. `project(action=...)`'s exact argv, default, and bad-action rejection.
6. The tool docstring's own token budget.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
from pathlib import Path

import pytest
from conftest import _OK_STREAM, envelope

from comfy_mcp.server import _internal as server

_SERVER_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "comfy_mcp"
    / "server"
    / "_internal.py"
)


def _boom(*args, **kwargs):
    raise AssertionError("no comfy-cli child may be spawned")


# --- _project_root: resolution, caching, validation -------------------------


def test_project_root_none_when_unset():
    assert server._project_root() is None


def test_project_root_returns_the_configured_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path))
    assert server._project_root() == str(tmp_path)


def test_project_root_reads_the_env_var_once_and_caches(tmp_path, monkeypatch):
    """A value written AFTER the first read is not picked up mid-process."""
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path))
    assert server._project_root() == str(tmp_path)

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("COMFY_PROJECT", str(other))
    assert server._project_root() == str(tmp_path)  # still the FIRST value


def test_project_root_raises_on_a_missing_directory(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("COMFY_PROJECT", str(missing))

    with pytest.raises(server.ComfyCliError, match="COMFY_PROJECT"):
        server._project_root()


def test_project_root_raises_on_a_file_not_a_directory(tmp_path, monkeypatch):
    not_a_dir = tmp_path / "a-file"
    not_a_dir.write_text("x")
    monkeypatch.setenv("COMFY_PROJECT", str(not_a_dir))

    with pytest.raises(server.ComfyCliError, match="not a directory"):
        server._project_root()


def test_project_root_error_names_the_value_and_the_fix(tmp_path, monkeypatch):
    missing = tmp_path / "gone"
    monkeypatch.setenv("COMFY_PROJECT", str(missing))

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._project_root()

    message = str(excinfo.value)
    assert str(missing) in message
    assert "unset COMFY_PROJECT" in message
    assert "mkdir" in message


def test_project_root_does_not_latch_an_invalid_verdict(tmp_path, monkeypatch):
    """Unlike the env READ, a bad root is re-checked every call: mkdir-and-retry
    works mid-process, mirroring `_check_comfy_version`'s too-old verdict."""
    not_yet = tmp_path / "not-yet"
    monkeypatch.setenv("COMFY_PROJECT", str(not_yet))

    with pytest.raises(server.ComfyCliError):
        server._project_root()

    not_yet.mkdir()
    assert server._project_root() == str(not_yet)  # the retry sees it


def test_project_root_raises_on_a_relative_path(monkeypatch):
    """A relative COMFY_PROJECT must not silently resolve against this
    server's own (client-assigned, arbitrary) cwd — that would be exactly
    the non-determinism anchoring exists to remove."""
    monkeypatch.setenv("COMFY_PROJECT", "relative/project/dir")

    with pytest.raises(server.ComfyCliError, match="absolute"):
        server._project_root()


def test_project_root_relative_error_names_the_value(monkeypatch):
    monkeypatch.setenv("COMFY_PROJECT", "some/relative/path")

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._project_root()

    assert "some/relative/path" in str(excinfo.value)


def test_relative_project_root_refuses_before_any_spawn(monkeypatch):
    """The relative check fails closed at the same point the directory check
    does: before the real comfy-cli subcommand ever spawns."""
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", _boom)
    monkeypatch.setenv("COMFY_PROJECT", "relative/dir")

    with pytest.raises(server.ComfyCliError, match="absolute"):
        server._run_comfy("env")


# --- unset: every spawn path carries no cwd (byte-identical to before) ------


def test_plain_spawn_carries_no_cwd_when_unset(patched_run):
    calls = patched_run(envelope(data={"x": 1}))
    server._run_comfy("env")
    assert calls[0]["cwd"] is None


def test_async_spawn_carries_no_cwd_when_unset(patched_async_run):
    procs = patched_async_run(envelope(data={"x": 1}))
    asyncio.run(server._run_comfy_async("jobs", "status", "abc"))
    assert procs[0].cwd is None


def test_streaming_spawn_carries_no_cwd_when_unset(patched_stream):
    procs = patched_stream(_OK_STREAM)
    asyncio.run(server._run_comfy_streaming("run", "wf.json"))
    assert procs[0].cwd is None


def test_version_probe_carries_no_cwd_when_unset(monkeypatch):
    seen: dict[str, object] = {}

    def fake(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake)
    server._spawn_comfy_version()
    assert seen["cwd"] is None


# --- set + valid: every spawn path carries it --------------------------------


def test_plain_spawn_carries_the_configured_cwd(tmp_path, monkeypatch, patched_run):
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path))
    calls = patched_run(envelope(data={"x": 1}))
    server._run_comfy("env")
    assert calls[0]["cwd"] == str(tmp_path)


def test_async_spawn_carries_the_configured_cwd(
    tmp_path, monkeypatch, patched_async_run
):
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path))
    procs = patched_async_run(envelope(data={"x": 1}))
    asyncio.run(server._run_comfy_async("jobs", "status", "abc"))
    assert procs[0].cwd == str(tmp_path)


def test_streaming_spawn_carries_the_configured_cwd(
    tmp_path, monkeypatch, patched_stream
):
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path))
    procs = patched_stream(_OK_STREAM)
    asyncio.run(server._run_comfy_streaming("run", "wf.json"))
    assert procs[0].cwd == str(tmp_path)


def test_version_probe_carries_the_configured_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path))
    seen: dict[str, object] = {}

    def fake(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="comfy-cli, version 1.14.0", stderr=""
        )

    monkeypatch.setattr(server.subprocess, "run", fake)
    server._spawn_comfy_version()
    assert seen["cwd"] == str(tmp_path)


# --- set + invalid: fails closed, nothing spawns -----------------------------


def test_plain_spawn_refuses_on_invalid_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", _boom)
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path / "missing"))

    with pytest.raises(server.ComfyCliError, match="COMFY_PROJECT"):
        server._run_comfy("env")


def test_async_spawn_refuses_on_invalid_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", _boom)
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path / "missing"))

    with pytest.raises(server.ComfyCliError, match="COMFY_PROJECT"):
        asyncio.run(server._run_comfy_async("jobs", "status", "abc"))


def test_streaming_spawn_refuses_on_invalid_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", _boom)
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path / "missing"))

    with pytest.raises(server.ComfyCliError, match="COMFY_PROJECT"):
        asyncio.run(server._run_comfy_streaming("run", "wf.json"))


def test_version_probe_refuses_on_invalid_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(server.subprocess, "run", _boom)
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path / "missing"))

    with pytest.raises(server.ComfyCliError, match="COMFY_PROJECT"):
        server._spawn_comfy_version()


# --- the `project` tool: argv, default, bad action ---------------------------


def test_project_status_argv(patched_run):
    calls = patched_run(
        envelope(data={"root": "/x", "schema": "project/1", "recent_runs": []})
    )
    server.project(action="status")

    cmd = calls[0]["cmd"]
    assert cmd[:4] == [server.COMFY_BIN, "--json", "--where", "local"]
    assert cmd[4:] == ["project", "status"]


def test_project_init_argv(patched_run):
    calls = patched_run(
        envelope(data={"root": "/x", "created": [], "where_default": "cloud"})
    )
    server.project(action="init")

    assert calls[0]["cmd"][4:] == ["project", "init"]


def test_project_default_action_is_status(patched_run):
    calls = patched_run(envelope(data={}))
    server.project()
    assert calls[0]["cmd"][4:] == ["project", "status"]


def test_project_rejects_an_unknown_action(no_spawn):
    with pytest.raises(server.ComfyCliError, match="invalid project action"):
        server.project(action="switch")


def test_project_bad_action_error_names_the_valid_ones(no_spawn):
    with pytest.raises(server.ComfyCliError, match=r"'status'.*'init'"):
        server.project(action="bogus")


# --- tool docstring budget ---------------------------------------------------


def test_project_tool_docstring_within_its_own_token_budget():
    """Not the whole-payload ceiling (`test_payload_budget.py`) — this pins
    THIS tool's own share (brief: <=150 est. tokens) so a future edit notices
    growth immediately, at the point it happens."""
    tree = ast.parse(_SERVER_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "project":
            doc = ast.get_docstring(node)
            assert doc is not None
            est_tokens = len(doc) // 4
            assert est_tokens <= 150, f"project() docstring ~{est_tokens} est. tokens"
            return
    pytest.fail("project() tool not found in server/_internal.py")
