"""Regression tests for the comfy-cli invocation and output collection.

These lock in the two behaviors that actually broke during development:
1. comfy-cli's global flags (``--json``, ``--where``) MUST precede the
   subcommand — a trailing ``--json`` errors with "No such option".
2. ``fetch_outputs`` is a thin passthrough to
   ``comfy download <prompt_id> --where local -o <dir>`` — comfy-cli owns the
   local download, so the tool only maps its argv (no hand-rolled HTTP client).

Plus the streaming ``run_workflow(wait=True)`` path: it drives
``comfy --json-stream … run --wait`` via Popen, forwards NDJSON run events as
MCP progress notifications, and still returns the final envelope's data.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest
from conftest import _OK_STREAM, _FakeProc, _RecordingCtx, envelope

from comfy_local_mcp import server, textutil


def test_global_flags_precede_subcommand(patched_run):
    """Regression: `comfy run … --json` errors; it must be `comfy --json … run`."""
    calls = patched_run(envelope(data={"x": 1}))

    assert server._run_comfy("jobs", "status", "abc") == {"x": 1}

    cmd = calls[0]["cmd"]
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["jobs", "status", "abc"]  # subcommand strictly after
    assert calls[0]["env"]["COMFY_WHERE"] == "local"  # belt-and-suspenders pin


def test_run_comfy_sets_no_watch_env(patched_run):
    """Agentic caller: comfy-cli's file watcher is suppressed via COMFY_NO_WATCH."""
    calls = patched_run(envelope(data={"x": 1}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["env"]["COMFY_NO_WATCH"] == "1"


def test_run_comfy_forces_utf8_env(patched_run):
    """Windows cp1252 fix: the child env forces UTF-8 so catalog output can't crash.

    On a default Windows console (cp1252) comfy-cli raises UnicodeEncodeError
    printing the UTF-8 catalog and wedges, so discovery tools present as a 60s
    timeout. Forcing UTF-8 on the child prevents the crash (no-op on POSIX).
    """
    calls = patched_run(envelope(data={"x": 1}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["env"]["PYTHONUTF8"] == "1"
    assert calls[0]["env"]["PYTHONIOENCODING"] == "utf-8"


def test_run_comfy_pins_parent_decode_to_utf8(patched_run):
    """The parent-side read is pinned to UTF-8 to match the child's forced output.

    Without an explicit ``encoding``, ``text=True`` decodes the pipe with the
    system locale (cp1252 on a default Windows console), so the non-ASCII catalog
    output raises UnicodeDecodeError/mojibake before ``_unwrap_envelope`` — the
    same crash, just moved from the child's write to the parent's read.
    """
    calls = patched_run(envelope(data={"x": 1}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["encoding"] == "utf-8"


def test_run_comfy_utf8_env_overrides_inherited(patched_run, monkeypatch):
    """The injected UTF-8 vars win over any conflicting value in the parent env."""
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    calls = patched_run(envelope(data={"x": 1}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["env"]["PYTHONUTF8"] == "1"
    assert calls[0]["env"]["PYTHONIOENCODING"] == "utf-8"


def test_run_comfy_closes_child_stdin(patched_run):
    """The child never inherits the stdio transport's stdin.

    This is an MCP **stdio** server: the parent's stdin carries JSON-RPC
    requests. A child that inherits it (the subprocess default) can read those
    bytes out from under the client, silently corrupting the session, or block
    on a prompt nobody can answer. No comfy-cli call here is interactive.
    """
    calls = patched_run(envelope(data={"x": 1}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["stdin"] == subprocess.DEVNULL


def test_run_comfy_sets_non_interactive_child_env(patched_run):
    """git/pip are told never to prompt, so a would-be prompt fails fast.

    With stdin closed an interactive prompt could not be answered anyway; these
    turn "block invisibly until the timeout" into an immediate, legible error —
    which matters most for `update_comfyui`'s 30-minute ceiling.
    """
    calls = patched_run(envelope(data={"x": 1}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert calls[0]["env"]["PIP_NO_INPUT"] == "1"


def test_run_comfy_leaves_askpass_alone(patched_run, monkeypatch):
    """A GUI/keychain credential helper still works — it does not use stdin.

    Overriding `GIT_ASKPASS` would break private-remote updates that succeed
    today, so the non-interactive pins deliberately stop at the terminal prompt.
    """
    monkeypatch.setenv("GIT_ASKPASS", "/usr/local/bin/my-helper")
    calls = patched_run(envelope(data={"x": 1}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["env"]["GIT_ASKPASS"] == "/usr/local/bin/my-helper"


# --- COMFY_BIN's directory is guaranteed first on the child PATH -------------
#
# `comfy launch --background` re-invokes `comfy` by BARE NAME via PATH to spawn
# the detached process, so an absolute COMFY_BIN whose directory is not on the
# inherited PATH (an MCP server started by a GUI client + a venv-installed
# comfy-cli) crashed the child with FileNotFoundError before ComfyUI was ever
# started. See `_comfy_env` (BE-4735 / BE-3780).

_needs_posix_exec = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX exec bit; which() needs a PATHEXT suffix on Windows",
)

# The spawn-site fixtures stub `shutil.which` to a fixed "/fake/comfy" (they
# patch the shared module attribute), which would mask the real resolution the
# two end-to-end tests below are checking. Captured at import, before any test
# has had a chance to patch it, so they can put the genuine one back.
_real_which = shutil.which


def _dummy_comfy(tmp_path, dirname="venv-bin", name="comfy"):
    """A real, executable file on disk standing in for the comfy-cli binary.

    `shutil.which` checks the exec bit, so a bare `tmp_path.touch()` would
    resolve to None and quietly exercise the skip path instead of the prepend.
    """
    bin_dir = tmp_path / dirname
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / name
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    return exe


@_needs_posix_exec
def test_comfy_env_prepends_absolute_comfy_bin_dir(tmp_path, monkeypatch):
    """The QA case: absolute COMFY_BIN in a venv, minimal PATH that excludes it."""
    exe = _dummy_comfy(tmp_path)
    bin_dir = str(exe.parent)
    inherited = os.pathsep.join(["/usr/bin", "/bin"])
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", inherited)

    path = server._comfy_env()["PATH"]

    assert path.startswith(bin_dir + os.pathsep)
    # The inherited PATH is preserved behind it, not replaced.
    assert path == bin_dir + os.pathsep + inherited


@_needs_posix_exec
def test_comfy_env_absolutizes_a_relative_comfy_bin(tmp_path, monkeypatch):
    """comfy-cli chdirs to the workspace before re-resolving, so a relative entry
    would point somewhere else by then — the prepended entry must be absolute."""
    exe = _dummy_comfy(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "COMFY_BIN", os.path.join(".", "venv-bin", "comfy"))
    monkeypatch.setenv("PATH", "/usr/bin")

    first = server._comfy_env()["PATH"].split(os.pathsep)[0]

    assert os.path.isabs(first)
    assert os.path.realpath(first) == os.path.realpath(str(exe.parent))


@_needs_posix_exec
def test_comfy_env_does_not_duplicate_bin_dir_already_first(tmp_path, monkeypatch):
    """Already first: PATH is left exactly as inherited — no repeated prepend."""
    exe = _dummy_comfy(tmp_path)
    bin_dir = str(exe.parent)
    inherited = os.pathsep.join([bin_dir, "/usr/bin"])
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", inherited)

    path = server._comfy_env()["PATH"]

    assert path == inherited
    assert path.split(os.pathsep).count(bin_dir) == 1


@_needs_posix_exec
def test_comfy_env_prepends_even_when_bin_dir_is_later_on_path(tmp_path, monkeypatch):
    """Present but shadowed: prepend anyway so an earlier stale comfy can't win.

    Prepending (rather than appending, or skipping on "already present") is the
    deliberate half of this: inside the child, `comfy` must resolve to the SAME
    install this server was pointed at, not to whichever one happens to be
    earlier on the user's PATH.
    """
    exe = _dummy_comfy(tmp_path)
    bin_dir = str(exe.parent)
    stale = tmp_path / "stale-bin"
    stale.mkdir()
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", os.pathsep.join([str(stale), bin_dir]))

    path = server._comfy_env()["PATH"]

    assert path.split(os.pathsep)[0] == bin_dir


@_needs_posix_exec
def test_comfy_env_prepends_dir_of_bare_name_resolved_on_path(tmp_path, monkeypatch):
    """A bare COMFY_BIN is resolved through PATH, and ITS directory is hoisted."""
    exe = _dummy_comfy(tmp_path)
    bin_dir = str(exe.parent)
    other = tmp_path / "other-bin"
    other.mkdir()
    monkeypatch.setattr(server, "COMFY_BIN", "comfy")
    monkeypatch.setenv("PATH", os.pathsep.join([str(other), bin_dir]))

    # Sanity: the bare name really does resolve into bin_dir under this PATH.
    assert shutil.which("comfy") == str(exe)

    assert server._comfy_env()["PATH"].split(os.pathsep)[0] == bin_dir


def test_comfy_env_leaves_path_alone_when_comfy_bin_unresolvable(tmp_path, monkeypatch):
    """Unresolvable COMFY_BIN: skip silently, never raise, never touch PATH.

    `_require_comfy_bin` already raises the curated missing-binary error before
    any spawn, so `_comfy_env` must not add a second failure mode here.
    """
    inherited = os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")])
    monkeypatch.setattr(server, "COMFY_BIN", str(tmp_path / "nope" / "comfy"))
    monkeypatch.setenv("PATH", inherited)

    assert server._comfy_env()["PATH"] == inherited


def test_comfy_env_skips_a_non_executable_comfy_bin(tmp_path, monkeypatch):
    """A file that exists but is not executable is not a resolution — skip it."""
    exe = tmp_path / "not-exec" / "comfy"
    exe.parent.mkdir()
    exe.write_text("")
    exe.chmod(0o644)
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", "/usr/bin")

    assert server._comfy_env()["PATH"] == "/usr/bin"


def test_comfy_env_unresolvable_with_no_inherited_path(tmp_path, monkeypatch):
    """No PATH at all and nothing to resolve: PATH stays absent, no KeyError."""
    monkeypatch.setattr(server, "COMFY_BIN", str(tmp_path / "nope" / "comfy"))
    monkeypatch.delenv("PATH", raising=False)

    assert "PATH" not in server._comfy_env()


@_needs_posix_exec
def test_comfy_env_sets_bare_path_when_none_inherited(tmp_path, monkeypatch):
    """An empty/absent inherited PATH yields the bin dir alone — no stray separator."""
    exe = _dummy_comfy(tmp_path)
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", "")

    assert server._comfy_env()["PATH"] == str(exe.parent)


@_needs_posix_exec
def test_comfy_env_falls_back_to_defpath_when_no_path_inherited(tmp_path, monkeypatch):
    """No PATH inherited: keep the platform default behind the bin dir.

    With no PATH in the environment CPython resolves a child's bare-name exec
    against `os.defpath` (`os.get_exec_path`). Writing the bin dir ALONE would
    replace that implicit default, leaving the child unable to find the `git` /
    `python` / `uv` helpers comfy-cli shells out to — a strict regression on the
    pre-prepend behavior. The prepend has to stay additive here too.
    """
    exe = _dummy_comfy(tmp_path)
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.delenv("PATH", raising=False)

    entries = server._comfy_env()["PATH"].split(os.pathsep)

    assert entries[0] == str(exe.parent)
    assert entries[1:] == os.defpath.split(os.pathsep)


@_needs_posix_exec
def test_comfy_env_skips_prepend_when_bin_dir_contains_pathsep(tmp_path, monkeypatch):
    """A bin dir containing `os.pathsep` is unrepresentable — leave PATH alone.

    There is no escape for the separator in PATH syntax, so writing such a
    directory splits it into fragments: the intended entry is destroyed AND its
    tail becomes a RELATIVE entry, which the child resolves against the
    workspace comfy-cli chdir'd into. Skipping preserves the inherited PATH;
    corrupting it would be strictly worse than not helping.
    """
    exe = _dummy_comfy(tmp_path, dirname="we" + os.pathsep + "ird")
    inherited = os.pathsep.join(["/usr/bin", "/bin"])
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", inherited)

    path = server._comfy_env()["PATH"]

    assert path == inherited
    # Nothing relative leaked in — that is the half with security weight.
    assert all(os.path.isabs(entry) for entry in path.split(os.pathsep))


@_needs_posix_exec
def test_comfy_env_path_prepend_keeps_the_existing_pins(tmp_path, monkeypatch):
    """The PATH rewrite is additive — every previously-injected pin still holds."""
    exe = _dummy_comfy(tmp_path)
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", "/usr/bin")

    env = server._comfy_env()

    assert env["COMFY_WHERE"] == "local"
    assert env["COMFY_NO_WATCH"] == "1"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["PIP_NO_INPUT"] == "1"
    assert env["PATH"].split(os.pathsep)[0] == str(exe.parent)


@_needs_posix_exec
def test_run_comfy_child_env_carries_the_bin_dir_on_path(
    tmp_path, monkeypatch, patched_run
):
    """The plain `--json` spawn site really hands the child the rewritten PATH."""
    exe = _dummy_comfy(tmp_path)
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", "/usr/bin")
    calls = patched_run(envelope(data={"x": 1}))
    monkeypatch.setattr(server.shutil, "which", _real_which)

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["env"]["PATH"].split(os.pathsep)[0] == str(exe.parent)


def test_error_envelope_raises_with_code(patched_run):
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "ComfyUI not running"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="server_not_running"):
        server._run_comfy("env")


def test_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    with pytest.raises(server.ComfyCliError, match="not found on PATH"):
        server._run_comfy("env")


def test_error_envelope_carries_structured_code(patched_run):
    """The raised ComfyCliError exposes the envelope's error.code (not just text)."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "down"},
        }
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env")

    assert excinfo.value.code == "server_not_running"


# --- comfy-cli version guard -------------------------------------------------


def _fake_version(stdout: str, *, stderr: str = "", raises: Exception | None = None):
    """A `subprocess.run` stand-in for `comfy --version`; records each call."""
    calls: list[list[str]] = []

    def fake(cmd, capture_output, text, timeout, check, errors=None):  # noqa: ARG001
        calls.append(cmd)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=stderr)

    return fake, calls


def test_version_guard_raises_on_old_comfy_cli(monkeypatch):
    """A comfy-cli below the floor raises an actionable upgrade error."""
    monkeypatch.setattr(server, "_version_checked", False)
    fake, calls = _fake_version("comfy-cli, version 1.11.0")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match=r"too old.*1\.12\.0"):
        server._check_comfy_version()

    assert calls[0] == [server.COMFY_BIN, "--version"]
    # A too-old verdict is NOT memoized, so a later retry re-checks.
    assert server._version_checked is False


def test_version_guard_blocks_run_comfy_on_old_cli(monkeypatch):
    """The guard fires from inside `_run_comfy`, before any real subcommand runs."""
    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    fake, _ = _fake_version("comfy version 1.9.9")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match="too old"):
        server._run_comfy("logs", "--tail", "10")


def test_version_guard_allows_new_comfy_cli_and_memoizes(monkeypatch):
    """A comfy-cli at/above the floor passes and shells out only once."""
    monkeypatch.setattr(server, "_version_checked", False)
    fake, calls = _fake_version("comfy-cli, version 1.12.0")
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()
    assert server._version_checked is True

    server._check_comfy_version()  # memoized: no second `comfy --version`
    assert len(calls) == 1


def test_version_guard_fails_open_on_unparseable_version(monkeypatch):
    """An unreadable `--version` must not block an otherwise-working install."""
    monkeypatch.setattr(server, "_version_checked", False)
    fake, _ = _fake_version("comfy-cli (dev build, no tag)")
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()  # no raise
    assert server._version_checked is True


def test_version_guard_fails_open_when_version_errors(monkeypatch):
    """A `comfy --version` that can't be spawned fails OPEN, not closed —
    and a transient spawn error is NOT latched, so a later call re-checks."""
    monkeypatch.setattr(server, "_version_checked", False)
    fake, _ = _fake_version("", raises=OSError("boom"))
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()  # no raise
    assert server._version_checked is False  # transient error not latched


def test_version_guard_latches_on_timeout(monkeypatch):
    """A hung `--version` (timeout) fails OPEN and IS latched, so a later call
    doesn't re-block on the same 30s wait."""
    monkeypatch.setattr(server, "_version_checked", False)
    fake, _ = _fake_version(
        "", raises=subprocess.TimeoutExpired(cmd="comfy --version", timeout=30.0)
    )
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()  # no raise
    assert server._version_checked is True


def test_spawn_comfy_version_keeps_the_bounded_decode_safe_invocation(monkeypatch):
    """The one shared spawn site: right argv, bounded, and decode-safe."""
    seen: dict[str, object] = {}

    def fake(cmd, capture_output, text, errors, timeout, check):
        seen.update(
            cmd=cmd,
            capture_output=capture_output,
            text=text,
            errors=errors,
            timeout=timeout,
            check=check,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake)
    server._spawn_comfy_version()

    assert seen == {
        "cmd": [server.COMFY_BIN, "--version"],
        "capture_output": True,
        "text": True,
        "errors": "replace",
        "timeout": 30.0,
        "check": False,
    }


def test_both_version_probes_go_through_the_shared_spawn(monkeypatch):
    """The floor guard and the best-effort detector share ONE spawn site, so the
    invocation can never drift between them (they keep their own policies)."""
    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    calls: list[str] = []

    def fake_spawn():
        calls.append("spawn")
        return subprocess.CompletedProcess(
            [server.COMFY_BIN, "--version"],
            0,
            stdout="comfy-cli, version 1.12.0\n",
            stderr="",
        )

    monkeypatch.setattr(server, "_spawn_comfy_version", fake_spawn)

    server._check_comfy_version()  # no raise: at the floor
    assert server._detect_comfy_cli_version() == "1.12.0"
    assert len(calls) == 2


def test_parse_version_reads_two_and_three_part_and_none():
    assert server._parse_version("comfy-cli, version 1.12") == (1, 12, 0)
    assert server._parse_version("v2.0.5 extra") == (2, 0, 5)
    assert server._parse_version("no version token here") is None


def test_parse_version_prefers_token_after_version_keyword():
    """A leading dotted token (a Python version) must not be mistaken for the
    comfy-cli version when a `version X.Y.Z` token is present."""
    assert server._parse_version("Python 3.10.2\ncomfy-cli, version 1.12.0") == (
        1,
        12,
        0,
    )


# --- get_logs ---------------------------------------------------------------


def test_get_logs_maps_command_and_returns_data(patched_run):
    """get_logs wraps `comfy logs --tail <n>` and returns the envelope data."""
    payload = {
        "lines": ["boot ok", "listening on :8188"],
        "path": "/ws/user/comfyui_8188.log",
        "truncated": False,
    }
    calls = patched_run(envelope(data=payload))

    assert server.get_logs() == payload

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["logs", "--tail", "200"]  # default tail, stringified


def test_get_logs_forwards_custom_tail(patched_run):
    """A custom tail is stringified and forwarded to `--tail`."""
    calls = patched_run(envelope(data={"lines": []}))

    server.get_logs(tail=50)

    assert calls[0]["cmd"][4:] == ["logs", "--tail", "50"]


def test_get_logs_clamps_negative_and_huge_tail(patched_run):
    """A negative tail can't forward a malformed `--tail -N`, and an absurd tail
    is capped so comfy-cli isn't asked for an enormous slice."""
    calls = patched_run(envelope(data={"lines": []}))

    server.get_logs(tail=-5)
    assert calls[-1]["cmd"][4:] == ["logs", "--tail", str(server._MIN_LOG_TAIL)]

    server.get_logs(tail=10**9)
    assert calls[-1]["cmd"][4:] == ["logs", "--tail", str(server._MAX_LOG_TAIL)]


def test_get_logs_no_log_file_returned_not_raised(patched_run):
    """A `no_log_file` error envelope is returned as data, not raised."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "no_log_file", "message": "no log file for local yet"},
        }
    )

    result = server.get_logs()

    assert result["error"] == "no_log_file"
    assert "no log file" in result["message"]


def test_get_logs_other_error_still_raises(patched_run):
    """Any other error code keeps raising — only `no_log_file` is swallowed."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "ComfyUI not running"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="server_not_running"):
        server.get_logs()


def test_cancel_job_maps_command_and_returns_data(patched_run):
    """cancel_job wraps `comfy jobs cancel <id>` and returns the envelope data."""
    calls = patched_run(envelope(data={"cancelled": "abc"}))

    assert server.cancel_job("abc") == {"cancelled": "abc"}
    assert calls[0]["cmd"][4:] == ["jobs", "cancel", "abc"]  # mapped subcommand


def test_cancel_job_unknown_id_raises_error_envelope(patched_run):
    """Cancelling an unknown prompt_id surfaces comfy-cli's error envelope."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "not_found", "message": "no such job: nope"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="not_found"):
        server.cancel_job("nope")


# `_run_comfy` builds argv with no `--` separator, so a leading-dash positional
# reaches comfy-cli as an option rather than a job id — `fetch_outputs` most
# sharply, since there the id sits beside a real `-o`. An embedded NUL is a legal
# JSON (so MCP) string that `subprocess.run` refuses with a bare ValueError, and
# an empty id can only be a caller mistake. `_guard_prompt_id` rejects all three
# for every tool that takes one; the synchronous ones are checked here as a
# family (`watch_job` and `get_execution_error` are covered separately).
@pytest.mark.parametrize(
    ("tool", "extra_args"),
    [
        ("job_status", ()),
        ("cancel_job", ()),
        ("wait_for_job", ()),
        ("fetch_outputs", ("/tmp/out",)),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
@pytest.mark.parametrize("bad_id", ["--help", "-o", "", "p\x001"])
def test_job_tools_reject_an_unusable_prompt_id(monkeypatch, tool, extra_args, bad_id):
    """A dash-led / empty / NUL-bearing prompt_id is refused before any spawn."""

    def fake_run(*args, **kwargs):
        raise AssertionError(f"{tool} spawned comfy-cli with {bad_id!r}")

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="prompt_id"):
        getattr(server, tool)(bad_id, *extra_args)


def test_get_queue_maps_command_and_returns_data(patched_run):
    """get_queue wraps `comfy jobs ls` and returns the merged job list."""
    jobs = {
        "jobs": [
            {"prompt_id": "a", "status": "running"},
            {"prompt_id": "b", "status": "completed"},
        ]
    }
    calls = patched_run(envelope(data=jobs))

    assert server.get_queue() == jobs
    assert calls[0]["cmd"][4:] == ["jobs", "ls"]  # no positional args


def test_get_queue_error_envelope_raises(patched_run):
    """A failing `comfy jobs ls` (e.g. server unreachable) raises ComfyCliError."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "ComfyUI not running"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="server_not_running"):
        server.get_queue()


def test_get_queue_drops_cloud_rows(patched_run):
    """Cloud-tracked rows are filtered out — this server is local-only.

    comfy-cli merges its on-disk job state into `jobs ls` without scoping it to
    the `--where local` this server always passes, so a listing can carry rows
    from a prior cloud run.
    """
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {
                "host": "127.0.0.1",
                "port": 8188,
                "where": "local",
                "count": 3,
                "jobs": [
                    {"prompt_id": "a", "status": "running", "where": "local"},
                    {"prompt_id": "b", "status": "completed", "where": "cloud"},
                    {"prompt_id": "c", "status": "completed", "where": "local"},
                ],
            },
        }
    )

    result = server.get_queue()

    assert [job["prompt_id"] for job in result["jobs"]] == ["a", "c"]
    assert result["count"] == 2
    assert result["where"] == "local"  # rest of the payload is untouched


def test_get_queue_keeps_rows_without_a_where(patched_run):
    """A row with no `where` is a legacy LOCAL row — kept, not dropped."""
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {
                "count": 2,
                "jobs": [
                    {"prompt_id": "a", "status": "running"},
                    {"prompt_id": "b", "status": "completed", "where": None},
                ],
            },
        }
    )

    assert [job["prompt_id"] for job in server.get_queue()["jobs"]] == ["a", "b"]


def test_get_queue_keeps_rows_that_are_not_dicts(patched_run):
    """An unknown ROW shape passes through — only positively-cloud rows drop."""
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {
                "count": 3,
                "jobs": ["a-bare-id", None, {"prompt_id": "b", "where": "cloud"}],
            },
        }
    )

    result = server.get_queue()

    assert result["jobs"] == ["a-bare-id", None]
    assert result["count"] == 2


def test_get_queue_passes_through_foreign_payload_shapes(patched_run):
    """An unrecognized payload (older/newer comfy-cli) is returned untouched."""
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": [{"prompt_id": "a", "where": "cloud"}],  # a bare list
        }
    )

    assert server.get_queue() == [{"prompt_id": "a", "where": "cloud"}]


def test_get_queue_passes_through_payload_without_jobs(patched_run):
    """A dict with no `jobs` list is returned untouched (no `count` invented)."""
    patched_run(
        {"type": "envelope", "ok": True, "data": {"host": "127.0.0.1", "port": 8188}}
    )

    assert server.get_queue() == {"host": "127.0.0.1", "port": 8188}


def test_upload_file_passes_paths_and_overwrite(patched_run):
    """upload_file forwards every path and appends --overwrite when asked."""
    calls = patched_run(envelope(data={"uploaded": 2}))

    assert server.upload_file(["a.png", "b.png"], overwrite=True) == {"uploaded": 2}

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["upload", "a.png", "b.png", "--overwrite"]


def test_upload_file_omits_overwrite_by_default(patched_run):
    """Without overwrite the flag must be absent, not passed as False."""
    calls = patched_run(envelope(data={"uploaded": 1}))

    server.upload_file(["only.png"])

    assert calls[0]["cmd"][4:] == ["upload", "only.png"]
    assert "--overwrite" not in calls[0]["cmd"]


def test_upload_file_rejects_option_like_path():
    """A leading-dash path is refused: splatted in, it would BE the flag."""
    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.upload_file(["--overwrite"])


def test_upload_file_rejects_option_like_path_among_valid_ones():
    """The guard scans every path, not just the first (argument injection)."""
    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.upload_file(["/tmp/a.png", "--overwrite"])


def test_upload_file_rejects_embedded_nul_path():
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.upload_file(["/tmp/a\0.png"])

    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.upload_file(["/tmp/a.png", "/tmp/b\0.png"])


def test_download_model_url_only(patched_run):
    """download_model wraps the SINGULAR `model download --url` and returns data."""
    calls = patched_run(
        {"type": "envelope", "ok": True, "data": {"saved": "/models/x.safetensors"}}
    )

    assert server.download_model("https://hf.co/x.safetensors") == {
        "saved": "/models/x.safetensors"
    }

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    # SINGULAR `model` verb group (download engine), not the plural catalog.
    assert cmd[4:] == ["model", "download", "--url", "https://hf.co/x.safetensors"]


def test_download_model_threads_relative_path(patched_run):
    """--relative-path is appended only when provided."""
    calls = patched_run(envelope(data={}))

    server.download_model("https://hf.co/l.safetensors", relative_path="models/loras")

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/l.safetensors",
        "--relative-path",
        "models/loras",
    ]


def test_download_model_threads_filename(patched_run):
    """--filename is appended only when provided."""
    calls = patched_run(envelope(data={}))

    server.download_model("https://hf.co/c.safetensors", filename="renamed.safetensors")

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/c.safetensors",
        "--filename",
        "renamed.safetensors",
    ]


def test_download_model_threads_all_optionals(patched_run):
    """Both optional args thread through together, in order, only when set."""
    calls = patched_run(envelope(data={}))

    server.download_model(
        "https://civitai.com/api/download/models/42",
        relative_path="models/checkpoints",
        filename="sd.safetensors",
    )

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://civitai.com/api/download/models/42",
        "--relative-path",
        "models/checkpoints",
        "--filename",
        "sd.safetensors",
    ]


def test_download_model_omits_absent_optionals(patched_run):
    """Neither optional flag is emitted when the argument is left unset."""
    calls = patched_run(envelope(data={}))

    server.download_model("https://hf.co/x.safetensors")

    cmd = calls[0]["cmd"]
    assert "--relative-path" not in cmd
    assert "--filename" not in cmd


def test_download_model_rejects_option_like_url():
    """A leading-dash url is refused so comfy-cli can't parse it as a flag."""
    with pytest.raises(server.ComfyCliError, match="invalid url"):
        server.download_model("--config")


def test_download_model_rejects_option_like_relative_path():
    """A leading-dash relative_path is refused (argument injection guard)."""
    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        server.download_model("https://hf.co/x.safetensors", relative_path="-rf")


def test_download_model_rejects_option_like_filename():
    """A leading-dash filename is refused (argument injection guard)."""
    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        server.download_model("https://hf.co/x.safetensors", filename="--evil")


@pytest.mark.parametrize(
    "bad_url", ["file:///etc/passwd", "ftp://host/x", "/etc/passwd"]
)
def test_download_model_rejects_non_http_scheme(bad_url):
    """Only http(s) URLs are allowed; file://, ftp:// and bare paths are refused."""
    with pytest.raises(server.ComfyCliError, match="invalid url"):
        server.download_model(bad_url)


@pytest.mark.parametrize(
    "bad_path",
    ["../../etc", "models/../../etc", "/abs/models", "..\\..\\etc", "C:evil"],
)
def test_download_model_rejects_traversal_relative_path(bad_path):
    """relative_path must stay within the models dir: no `..`, absolute paths,
    or a drive prefix (``C:evil`` has no separator but is drive-relative on
    Windows)."""
    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        server.download_model("https://hf.co/x.safetensors", relative_path=bad_path)


# --- Windows-shaped escapes are refused on EVERY host, including Linux CI ----
#
# The guard runs wherever the MCP server runs; the write happens wherever
# comfy-cli runs. Judging these from the string's own shape plus
# `ntpath.splitdrive` — never the host's `os.path` — is what makes them fail on a
# POSIX box too, so these cases are the regression pins for that: to `posixpath`
# on Linux CI every string below is an ordinary relative name.


@pytest.mark.parametrize(
    "bad_path",
    [
        "C:evil",  # drive-RELATIVE: no separator, resolves against C:'s cwd
        "C:/Windows",  # drive-absolute with a forward slash
        "C:\\Windows",  # drive-absolute with a backslash
        "\\\\server\\share",  # UNC root — no `:` at all, so `:`-checks miss it
        "\\\\server\\share\\evil",  # UNC with a trailing path
        "\\evil",  # root-relative on the current drive
    ],
)
def test_download_model_rejects_windows_drive_relative_path(bad_path, patched_run):
    """A Windows drive/UNC/root-relative ``relative_path`` is refused before any
    child spawns — on Linux CI as much as on Windows."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        server.download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


# --- A `..` wearing trailing spaces/periods is still a `..` ------------------
#
# Windows strips trailing spaces and periods from every path component at
# syscall time, so `".. "` and `"..."` reach the filesystem as `..` while an
# equality check against `".."` says they are something else. `normpath` does
# not collapse them either, which is why the guard has to match the dot RUN
# rather than the exact string. Like the drive cases above, these have to be
# refused from a POSIX host too — Linux keeps `".. "` as a literal directory
# name, so nothing here fires locally on CI unless the string check does it.


@pytest.mark.parametrize(
    "bad_path",
    [
        "models/.. /evil",  # trailing space: normalizes back to `..` on Windows
        "models/.../evil",  # dot run: strips to `..`
        "models/... /evil",  # both
        ".. ",  # the whole value is a disguised `..`
        "models/. /evil",  # a disguised `.` — not an escape, but not a real name
    ],
)
def test_download_model_rejects_dot_run_relative_path(bad_path, patched_run):
    """A `..` disguised by trailing spaces/periods is refused before any child
    spawns — on Linux CI as much as on Windows."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        server.download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


@pytest.mark.parametrize(
    "good_path",
    [
        "models/.hidden",  # a leading dot is an ordinary name, not a dot run
        "models/v1.5",  # interior periods survive `strip(" .")`
        "models/loras/",  # trailing slash: an EMPTY segment, still accepted
        "models//loras",  # doubled slash, likewise
    ],
)
def test_download_model_accepts_dotted_but_ordinary_relative_path(
    good_path, patched_run
):
    """The dot-run check must not widen into ordinary names or empty segments —
    only a component that is *nothing but* dots and spaces is a disguised `..`."""
    calls = patched_run(envelope(data={}))

    server.download_model("https://hf.co/x.safetensors", relative_path=good_path)

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--relative-path") + 1] == good_path


@pytest.mark.parametrize("bad_name", [".. ", "...", ". ", "... ", " "])
def test_download_model_rejects_dot_run_filename(bad_name, patched_run):
    """Same disguise on the bare-name side: `".. "` is a `..`, not a filename."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        server.download_model("https://hf.co/x.safetensors", filename=bad_name)

    assert calls == []


@pytest.mark.parametrize("good_name", [".gitkeep", "v1.5.safetensors", "model."])
def test_download_model_accepts_dotted_but_ordinary_filename(good_name, patched_run):
    """Regression pin for the dot-run filename check: a name that merely
    *contains* dots still has something left after `strip(" .")`."""
    calls = patched_run(envelope(data={}))

    server.download_model("https://hf.co/x.safetensors", filename=good_name)

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--filename") + 1] == good_name


@pytest.mark.parametrize(
    "bad_name", ["../evil", "sub/dir.safetensors", "..", "a\\b", "C:evil.dll"]
)
def test_download_model_rejects_pathy_filename(bad_name):
    """filename must be a bare name: no separators, `..`, or a drive prefix to
    escape the dir (``C:evil.dll`` has no separator but is drive-relative on
    Windows)."""
    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        server.download_model("https://hf.co/x.safetensors", filename=bad_name)


@pytest.mark.parametrize(
    "bad_name",
    [
        "C:evil.exe",  # drive-RELATIVE: a bare-name check alone lets this pass
        "C:/evil.exe",
        "\\\\server\\share\\evil.exe",  # UNC
        "\\evil.exe",  # root-relative on the current drive
    ],
)
def test_download_model_rejects_windows_drive_relative_filename(bad_name, patched_run):
    """A Windows drive/UNC/root-relative ``filename`` is refused before any child
    spawns — on Linux CI as much as on Windows."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        server.download_model("https://hf.co/x.safetensors", filename=bad_name)

    assert calls == []


def test_download_model_still_accepts_ordinary_path_and_name(patched_run):
    """Regression pin: the Windows-shaped rejections above must not catch the
    ordinary values these tools are actually called with."""
    calls = patched_run(envelope(data={}))

    server.download_model(
        "https://hf.co/x.safetensors",
        relative_path="models/loras",
        filename="x.safetensors",
    )

    cmd = calls[0]["cmd"]
    assert "--relative-path" in cmd and cmd[cmd.index("--relative-path") + 1] == (
        "models/loras"
    )
    assert "--filename" in cmd and cmd[cmd.index("--filename") + 1] == "x.safetensors"


def test_download_model_rejects_embedded_nul_url():
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.download_model("https://hf.co/x\0.safetensors")


def test_download_model_rejects_embedded_nul_relative_path():
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.download_model("https://hf.co/x.safetensors", relative_path="models/\0")


def test_download_model_rejects_embedded_nul_filename():
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.download_model("https://hf.co/x.safetensors", filename="x\0.safetensors")


def test_download_model_omits_empty_string_optionals(patched_run):
    """Explicit empty-string optionals are treated as unset, not forwarded as ``""``."""
    calls = patched_run(envelope(data={}))

    server.download_model("https://hf.co/x.safetensors", relative_path="", filename="")

    cmd = calls[0]["cmd"]
    assert "--relative-path" not in cmd
    assert "--filename" not in cmd


def test_validate_workflow_returns_results_for_valid(patched_run):
    """A valid workflow returns comfy-cli's validation data unwrapped."""
    calls = patched_run(
        {"type": "envelope", "ok": True, "data": {"valid": True, "nodes": 7}}
    )

    assert server.validate_workflow("wf.json") == {"valid": True, "nodes": 7}
    assert calls[0]["cmd"][4:] == ["validate", "--workflow", "wf.json"]


def test_validate_workflow_raises_with_error_code(patched_run):
    """An invalid workflow surfaces comfy-cli's structured error code, not a swallow."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {
                "code": "workflow_unknown_nodes",
                "message": "Unknown node type: FooSampler",
            },
        }
    )

    with pytest.raises(server.ComfyCliError, match="workflow_unknown_nodes"):
        server.validate_workflow("broken.json")


def test_wait_for_job_returns_terminal_status(monkeypatch):
    """wait_for_job polls until a terminal status and returns that final payload."""
    statuses = iter(
        [
            {"status": "running"},
            {"status": "running"},
            {"status": "completed", "outputs": ["/tmp/gen.png"]},
        ]
    )
    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return next(statuses)

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    result = server.wait_for_job("pid", timeout_seconds=25.0)

    assert result == {"status": "completed", "outputs": ["/tmp/gen.png"]}
    assert calls[0] == ("jobs", "status", "pid")  # polls `comfy jobs status <id>`
    assert len(calls) == 3  # kept polling past the two `running` responses


def test_wait_for_job_times_out_cleanly(monkeypatch):
    """A job that never finishes within the bound returns a clean timed-out payload."""
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"status": "running"})
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # Fake a monotonic clock that jumps 10s each read, so the 25s bound expires.
    clock = {"t": 0.0}

    def fake_monotonic():
        now = clock["t"]
        clock["t"] += 10.0
        return now

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = server.wait_for_job("pid", timeout_seconds=25.0)

    assert result == {"timed_out": True, "status": {"status": "running"}}


@pytest.mark.parametrize("oversized", [float("inf"), 86_400.0])
def test_wait_for_job_clamps_an_oversized_timeout(monkeypatch, oversized):
    """An oversized bound is clamped to the ceiling, so the poll loop terminates.

    Left raw, `deadline = monotonic() + inf` keeps `remaining` positive forever
    and the tool re-spawns `comfy jobs status` until the client gives up; a
    day-long finite bound outlives any client's patience just as surely.
    """
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"status": "running"})
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # A clock that jumps just past the ceiling once the first poll is done:
    # clamped, the deadline has passed and the tool returns; unclamped, it would
    # sleep and poll on, exhausting the iterator (failing loudly rather than
    # hanging the suite). Reads are (deadline, pre-poll, post-poll).
    reads = iter([0.0, 1.0, server._MAX_WATCH_TIMEOUT + 1.0])

    def fake_monotonic():
        try:
            return next(reads)
        except StopIteration:
            pytest.fail("wait_for_job kept polling past the clamped ceiling")

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = server.wait_for_job("pid", timeout_seconds=oversized)

    assert result == {"timed_out": True, "status": {"status": "running"}}


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_wait_for_job_rejects_a_non_positive_or_nan_timeout(monkeypatch, bad):
    """NaN/0/negative are refused before the first poll.

    With NaN every comparison is False, so `remaining <= 0` never fires and
    `min(2.0, nan)` returns 2.0 — the same forever-loop as `inf`.
    """
    polled = False

    # Raise rather than return a status: with a NaN bound left unclamped this
    # loop never exits (`remaining <= 0` is False and `sleep` is stubbed out),
    # so a regression must fail the test rather than hang the suite.
    def fake_run(*args, **kwargs):
        nonlocal polled
        polled = True
        raise AssertionError("wait_for_job polled with an invalid timeout")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        server.wait_for_job("pid", timeout_seconds=bad)

    assert polled is False


def test_wait_for_job_always_polls_at_least_once(monkeypatch):
    """A bound that expires before the first poll still reports a real status.

    The per-poll cap re-checks the deadline at the top of the loop; that check
    must not short-circuit the very first poll and return the degenerate
    `{"timed_out": True, "status": None}` with nothing ever asked of comfy-cli.
    """
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"status": "running"}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    # A clock that is already past the deadline by the loop's first read: the
    # bound is set at t=0 and every later read is t=1.
    reads = iter([0.0])

    def fake_monotonic():
        return next(reads, 1.0)

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = server.wait_for_job("pid", timeout_seconds=1e-9)

    assert calls == 1
    assert result == {"timed_out": True, "status": {"status": "running"}}


def test_wait_for_job_caps_each_poll_to_the_remaining_bound(monkeypatch):
    """A single poll never gets a longer subprocess budget than the wait itself.

    Each `comfy jobs status` used a fixed 60s timeout, so a short wait was only
    bounded *between* polls: one wedged status call could hold a
    `timeout_seconds=1` wait open for a full minute.
    """
    seen: list[float] = []

    def fake_run(*args, timeout=None, **kwargs):
        seen.append(timeout)
        return {"status": "running"}

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # A clock that advances 1s per read, so the 5s bound expires after two polls.
    clock = {"t": 0.0}

    def fake_monotonic():
        now = clock["t"]
        clock["t"] += 1.0
        return now

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = server.wait_for_job("pid", timeout_seconds=5.0)

    assert result == {"timed_out": True, "status": {"status": "running"}}
    assert seen == [4.0, 2.0]  # each poll gets only what is left of the 5s bound


def test_wait_for_job_gives_a_long_wait_the_full_poll_budget(monkeypatch):
    """The per-poll cap only bites when it is *below* the normal poll budget."""
    seen: list[float] = []

    def fake_run(*args, timeout=None, **kwargs):
        seen.append(timeout)
        return {"status": "completed"}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    assert server.wait_for_job("pid", timeout_seconds=600.0) == {"status": "completed"}
    assert seen == [server._JOB_STATUS_POLL_TIMEOUT]


def test_wait_for_job_reports_a_deadline_poll_timeout_as_timed_out(monkeypatch):
    """A poll killed by the caller's own bound returns `timed_out`, not an error.

    Capping each poll to the time left makes the poll's deadline double as the
    caller's, so a slow-but-healthy `comfy jobs status` near the bound now raises
    where the old fixed 60s budget let it finish. That is this call expiring, and
    it must not discard the last real status behind a `ComfyCliError`.
    """
    statuses = iter([{"status": "running"}])

    def fake_run(*args, **kwargs):
        try:
            return next(statuses)
        except StopIteration:
            raise server.ComfyCliError("comfy-cli timed out after 1.0s", timed_out=True)

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # 10s bound: the first poll succeeds, the second is granted the remainder and
    # dies at it — by which time the clock is past the deadline.
    reads = iter([0.0, 1.0, 2.0, 3.0, 11.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(reads))

    result = server.wait_for_job("pid", timeout_seconds=10.0)

    assert result == {"timed_out": True, "status": {"status": "running"}}


def test_wait_for_job_reraises_a_wedged_poll_with_budget_left(monkeypatch):
    """A poll that burns the FULL budget with time to spare is a real failure.

    Only the caller's bound expiring earns the `timed_out` envelope. A poll that
    exhausted `_JOB_STATUS_POLL_TIMEOUT` while the wait still has time left means
    comfy-cli is wedged — which raised before the per-poll cap existed too.
    """

    def fake_run(*args, **kwargs):
        raise server.ComfyCliError("comfy-cli timed out after 60.0s", timed_out=True)

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # A 600s bound with the clock barely moving: the deadline is nowhere near.
    reads = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(reads))

    with pytest.raises(server.ComfyCliError, match="timed out"):
        server.wait_for_job("pid", timeout_seconds=600.0)


def test_wait_for_job_reraises_when_no_status_was_ever_read(monkeypatch):
    """With no status in hand, the error beats a contentless `{"status": None}`.

    `{"timed_out": True, "status": None}` would bury the real diagnosis (and the
    budget that produced it) under an envelope the caller can do nothing with.
    """

    def fake_run(*args, **kwargs):
        raise server.ComfyCliError("comfy-cli timed out after 1.0s", timed_out=True)

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    reads = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(reads))

    with pytest.raises(server.ComfyCliError, match="timed out"):
        server.wait_for_job("pid", timeout_seconds=1.0)


def test_wait_for_job_reraises_a_non_timeout_poll_failure(monkeypatch):
    """An ordinary comfy-cli error still propagates, deadline or not."""

    def fake_run(*args, **kwargs):
        raise server.ComfyCliError("job not found", code="not_found")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    reads = iter([0.0, 1.0, 99.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(reads))

    with pytest.raises(server.ComfyCliError, match="job not found"):
        server.wait_for_job("pid", timeout_seconds=1.0)


def test_run_comfy_marks_a_subprocess_timeout(patched_run):
    """The `timed_out` flag is set only where the child was killed at our budget.

    `wait_for_job` branches on it to tell its own deadline from a comfy-cli
    failure, so the flag has to come from the raise site rather than the message.
    """
    calls = patched_run(
        raises=subprocess.TimeoutExpired([server.COMFY_BIN, "jobs"], 1.0)
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("jobs", "status", "pid", timeout=1.0)

    assert excinfo.value.timed_out is True
    assert calls[0]["timeout"] == 1.0  # the caller's budget bounded the wait


def test_prompt_id_guard_rejects_an_oversized_id(monkeypatch):
    """An id far past any real one is refused before it can reach argv.

    An oversized argv is rejected by the OS with an `OSError` no caller converts
    (and echoed back whole in the error), rather than failing as a clean
    `ComfyCliError` like every other bad input.
    """
    oversized = "p" * (server._MAX_PROMPT_ID_LEN + 1)

    def fake_run(*args, **kwargs):
        raise AssertionError("spawned comfy-cli with an oversized prompt_id")

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="exceeds"):
        server.job_status(oversized)

    # The cap is generous enough that a real (UUID-shaped) id is untouched.
    assert server._guard_prompt_id("p" * server._MAX_PROMPT_ID_LEN)


def test_fetch_outputs_rejects_a_nul_in_out_dir(monkeypatch):
    """`out_dir` rides the same argv as the id and needs the same NUL refusal.

    `_run_comfy_raw` converts only `TimeoutExpired`, so `subprocess.run`'s bare
    "embedded null byte" ValueError would escape as an internal error.
    """

    def fake_run(*args, **kwargs):
        raise AssertionError("spawned comfy-cli with a NUL-bearing out_dir")

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="out_dir"):
        server.fetch_outputs("pid", "/tmp/o\x00ut")


def test_fetch_outputs_wraps_comfy_download(patched_run):
    """fetch_outputs is a thin `comfy download … --where local -o <dir>` passthrough."""
    calls = patched_run(
        {"type": "envelope", "ok": True, "data": {"downloaded": ["img.png"]}}
    )

    assert server.fetch_outputs("pid", "/tmp/out") == {"downloaded": ["img.png"]}

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global --where pins local
    assert cmd[4:] == ["download", "pid", "-o", "/tmp/out"]  # mapped subcommand
    assert calls[0]["env"]["COMFY_WHERE"] == "local"


def test_fetch_outputs_url_only_appends_flag(patched_run):
    """url_only=True adds --url-only so comfy emits URLs without downloading bytes."""
    calls = patched_run(envelope(data={"urls": []}))

    server.fetch_outputs("pid", "/tmp/out", url_only=True)

    assert calls[0]["cmd"][4:] == ["download", "pid", "-o", "/tmp/out", "--url-only"]


def test_fetch_outputs_omits_url_only_by_default(patched_run):
    """Without url_only the flag must be absent, not passed as False."""
    calls = patched_run(envelope(data={}))

    server.fetch_outputs("pid", "/tmp/out")

    assert "--url-only" not in calls[0]["cmd"]


def test_server_instructions_cover_canonical_flows():
    """Instructions ride the handshake and teach submit->poll->fetch + templates."""
    instructions = server.mcp.instructions
    assert instructions  # present on the FastMCP instance

    # Call-server-info-first + the async submit -> poll -> fetch generation loop.
    for tool in ("server_info", "run_workflow", "wait_for_job", "fetch_outputs"):
        assert tool in instructions

    # The template on-ramp.
    for tool in ("search_templates", "fetch_template"):
        assert tool in instructions


def test_auth_status_maps_command_and_passes_payload_through(patched_run, monkeypatch):
    """auth_status wraps `comfy --json --where local cloud whoami`, payload unchanged."""
    whoami = {
        "signed_in": True,
        "auth_method": "oauth",
        "api_key_source": "store",
        "base_url": "https://api.comfy.example",
        "expired": False,
    }
    calls = patched_run(envelope(data=whoami))
    # No registration env key set -> the added flag is False, everything else as-is.
    monkeypatch.delenv("COMFY_API_KEY", raising=False)

    result = server.auth_status()

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == [
        "cloud",
        "whoami",
    ]  # whoami is not target-routed; --where local is harmless
    # Every whoami field passes through unchanged; only the local flag is added.
    for key, value in whoami.items():
        assert result[key] == value
    assert result["registration_env_key_present"] is False


def test_auth_status_signed_out_payload_passes_through(patched_run, monkeypatch):
    """A signed-out whoami (signed_in false, auth_method null) passes through cleanly."""
    whoami = {
        "signed_in": False,
        "auth_method": None,
        "api_key_source": None,
        "base_url": "https://api.comfy.example",
    }
    patched_run(envelope(data=whoami))
    monkeypatch.delenv("COMFY_API_KEY", raising=False)

    result = server.auth_status()

    assert result["signed_in"] is False
    assert result["auth_method"] is None
    assert result["registration_env_key_present"] is False


def test_auth_status_reports_registration_env_presence_not_value(
    patched_run, monkeypatch
):
    """The COMFY_API_KEY registration-env blind spot is reported as presence only."""
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {"signed_in": False, "auth_method": None},
        }
    )
    monkeypatch.setenv("COMFY_API_KEY", "sk-super-secret-value")

    result = server.auth_status()

    assert result["registration_env_key_present"] is True
    # Presence only — the actual key material must never appear in the payload.
    assert "sk-super-secret-value" not in json.dumps(result)


def test_auth_status_reports_flag_even_for_non_dict_payload(patched_run, monkeypatch):
    """A non-dict whoami payload still carries registration_env_key_present."""
    # Defensive: whoami normally returns an object, but if it ever hands back a
    # non-dict (null / list), the flag must still be present per the docstring.
    patched_run(envelope(data=None))
    monkeypatch.setenv("COMFY_API_KEY", "sk-super-secret-value")

    result = server.auth_status()

    assert result["registration_env_key_present"] is True
    assert result["whoami"] is None
    assert "sk-super-secret-value" not in json.dumps(result)


def test_auth_status_never_returns_key_material(patched_run, monkeypatch):
    """A realistic redacted whoami stays redacted — no key/token material leaks out."""
    # comfy-cli pre-redacts secrets; the tool must pass them through, never re-derive.
    whoami = {
        "signed_in": True,
        "auth_method": "api_key",
        "api_key_source": "env",
        "base_url": "https://api.comfy.example",
        "expired": False,
        "session": {"api_key": "***REDACTED***", "token": "***REDACTED***"},
        "stale_base_url": False,
    }
    patched_run(envelope(data=whoami))
    monkeypatch.delenv("COMFY_API_KEY", raising=False)

    dumped = json.dumps(server.auth_status())

    assert "REDACTED" in dumped  # the redaction placeholder survives unchanged
    # No un-redacted-looking key/token material in the returned dict.
    assert "sk-" not in dumped
    for token_word in ("secret", "bearer"):
        assert token_word not in dumped.lower()


def test_auth_status_error_envelope_raises(patched_run):
    """A failing `comfy cloud whoami` surfaces comfy-cli's error envelope."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "not_signed_in", "message": "no credentials"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="not_signed_in"):
        server.auth_status()


def test_server_instructions_cover_credential_steering():
    """Instructions steer an agent to auth_status + the three-step credential order."""
    instructions = server.mcp.instructions

    assert "auth_status" in instructions
    # The three credential steps must appear, in the canonical order.
    login = instructions.index("comfy cloud login")
    env_key = instructions.index("COMFY_API_KEY")
    set_key = instructions.index("comfy auth set comfy-cloud-api-key")
    assert login < env_key < set_key  # (1) login, (2) registration env, (3) set-key


def test_launch_comfyui_passes_background_flag(patched_run):
    """launch_comfyui must run `comfy … launch --background` (detached start)."""
    calls = patched_run(envelope(data={"pid": 42}))

    assert server.launch_comfyui() == {"pid": 42}

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert cmd[4:] == ["launch", "--background"]  # no extras -> no `--` separator


def test_launch_comfyui_forwards_extra_args_after_separator(patched_run):
    """Extra args are forwarded to ComfyUI after a `--` separator."""
    calls = patched_run(envelope(data={}))

    server.launch_comfyui(["--port", "8189"])

    assert calls[0]["cmd"][4:] == ["launch", "--background", "--", "--port", "8189"]


def test_stop_comfyui_surfaces_no_recorded_server_error(patched_run):
    """stop only targets comfy-cli's own pid; no recorded server -> clean error."""
    calls = patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {
                "code": "no_recorded_server",
                "message": "no ComfyUI server was launched by comfy-cli",
            },
        }
    )

    with pytest.raises(server.ComfyCliError, match="no_recorded_server"):
        server.stop_comfyui()

    assert calls[0]["cmd"][4:] == ["stop"]


# --- lifecycle commands with no JSON envelope (BE-2953) --------------------


def test_stop_comfyui_synthesizes_success_on_plain_exit(patched_plain_run):
    """`comfy stop` prints text + exits 0 with no envelope -> synthesized success."""
    patched_plain_run(0, stderr="Stopped ComfyUI server (pid 42).")

    result = server.stop_comfyui()

    assert result["ok"] is True
    assert result["action"] == "stop"
    assert "Stopped ComfyUI server" in result["message"]


def test_launch_comfyui_synthesizes_success_on_plain_exit(patched_plain_run):
    """`comfy launch --background` exits 0 with no envelope -> synthesized success."""
    patched_plain_run(0, stdout="Launched ComfyUI in the background.")

    result = server.launch_comfyui()

    assert result["ok"] is True
    assert result["action"] == "launch"
    assert "Launched ComfyUI" in result["message"]


def test_launch_comfyui_nonzero_exit_still_raises(patched_plain_run):
    """A real launch failure (non-zero exit, no envelope) must still raise."""
    patched_plain_run(1, stderr="Address already in use: port 8188")

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        server.launch_comfyui()


def test_plain_ok_synthesizes_despite_stray_non_envelope_json(patched_plain_run):
    """A stray non-envelope JSON line on a clean lifecycle exit is still success.

    `_last_json_object` returns any JSON object (not just `type==envelope`), so a
    diagnostic line that happens to parse must NOT be mistaken for a result
    envelope and unwrapped into a spurious failure (BE-2953 edge case).
    """
    patched_plain_run(
        0,
        stdout='{"level": "info", "msg": "bound port 8188"}\n',
        stderr="Launched ComfyUI in the background.",
    )

    result = server.launch_comfyui()

    assert result["ok"] is True
    assert result["action"] == "launch"
    assert "Launched ComfyUI" in result["message"]


def test_plain_ok_does_not_leak_to_other_commands(patched_plain_run):
    """Without plain_ok, an exit-0 command with no JSON still raises (unchanged)."""
    patched_plain_run(0, stdout="not json")

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        server._run_comfy("env")


def test_non_plain_ok_rejects_stray_non_envelope_json(patched_plain_run):
    """A stray non-envelope JSON on the NORMAL path must NOT satisfy the contract.

    Without `plain_ok`, an incidental object like `{"ok": true, "data": ...}`
    (no `type==envelope`) must not be mis-unwrapped as a valid response — the
    normal path passes `real_envelope`, so a stray diagnostic line raises the
    "returned no JSON" error like any other missing envelope (CodeRabbit review).
    """
    patched_plain_run(0, stdout='{"ok": true, "data": {"x": 1}}\n')

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        server._run_comfy("env")


def test_plain_ok_still_honors_a_real_envelope(patched_run):
    """When comfy-cli DOES emit an envelope, plain_ok unwraps it normally."""
    patched_run(envelope(data={"pid": 7}))

    assert server._run_comfy("launch", "--background", plain_ok=True) == {"pid": 7}


# --- envelope-failure messages must carry BOTH stream tails -----------------
# A comfy-cli that dies without emitting an envelope used to raise a message
# that showed only the HEAD of stderr and nothing at all about stdout — so a
# plain-text diagnostic printed to stdout was lost, a traceback was clipped
# right before the exception that explains it, and an empty stderr rendered as
# a dangling `stderr: ` indistinguishable from truncation.


def test_stream_tail_marks_empty_and_truncation():
    """`_stream_tail` marks a blank capture and a clipped one, and keeps the END."""
    assert textutil._stream_tail("") == "<empty>"
    assert textutil._stream_tail(None) == "<empty>"
    assert textutil._stream_tail("   \n  ") == "<empty>"  # whitespace-only is empty
    # A capture that fits the bound is passed through verbatim — no marker.
    assert textutil._stream_tail("  short  ") == "short"
    assert textutil._stream_tail("x" * 500, limit=500) == "x" * 500  # exactly at bound
    # One char over the bound: clipped, marked, and it is the TAIL that survives.
    clipped = textutil._stream_tail("HEAD" + "x" * 600 + "FINAL_ERROR_LINE", limit=500)
    assert clipped.startswith("...")
    assert clipped.endswith("FINAL_ERROR_LINE")
    assert "HEAD" not in clipped
    assert len(clipped) == 503  # bounded: the marker plus exactly `limit` chars
    # bytes are decoded defensively, same as `_tail`
    assert textutil._stream_tail(b"cafe\xff") == "cafe\ufffd"
    # A non-positive bound yields no tail rather than defeating itself: the
    # single-pass implementation asks `_tail` for `limit + 1`, so a 0 would slip
    # past its guard and `[-0:]` would return the whole capture unbounded.
    assert textutil._stream_tail("x" * 100, limit=0) == "<empty>"
    assert textutil._stream_tail("x" * 100, limit=-1) == "<empty>"


def test_tail_of_bytes_survives_a_long_trailing_whitespace_run():
    """A whitespace flood at the end must not swallow the error before it.

    The bytes branch windows the last ``4 * limit`` bytes before decoding (to
    avoid decoding a huge capture). If that window lands wholly inside trailing
    padding — a progress bar's, say — the strip empties it and the real error,
    sitting just before, is lost. The str branch strips first and never has this
    problem, so the two must agree.
    """
    padded = b"REAL_ERROR_LINE" + b" " * 5000

    assert textutil._tail(padded, limit=100) == "REAL_ERROR_LINE"
    assert textutil._stream_tail(padded, limit=100) == "REAL_ERROR_LINE"
    # Same answer as the equivalent str capture, which strips before slicing.
    assert textutil._tail(padded.decode(), limit=100) == textutil._tail(
        padded, limit=100
    )
    # The fast path is unchanged: a capture with no trailing flood is windowed,
    # not fully stripped, and still yields the TAIL.
    assert textutil._tail(b"HEAD" + b"y" * 600, limit=100) == "y" * 100


def test_no_envelope_error_marks_both_empty_streams(patched_plain_run):
    """Nothing on either stream still renders explicitly, never a dangling colon."""
    patched_plain_run(1, stdout="", stderr="")

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env")

    msg = str(excinfo.value)
    assert "exit 1" in msg
    assert "stderr: <empty>" in msg
    assert "stdout: <empty>" in msg


def test_no_envelope_error_surfaces_plain_stdout_diagnostic(patched_plain_run):
    """comfy-cli's plain-text stdout diagnosis survives `_last_json_object`."""
    patched_plain_run(
        1,
        stdout="Traceback (most recent call last):\n  ...\nValueError: boom\n",
        stderr="",
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env")

    msg = str(excinfo.value)
    assert "ValueError: boom" in msg  # the whole point: stdout is no longer dropped
    assert "stderr: <empty>" in msg


def test_no_envelope_error_keeps_the_stderr_tail_not_the_head(patched_plain_run):
    """A long traceback is clipped from the FRONT — the exception is at the END."""
    patched_plain_run(1, stderr="HEAD_MARKER" + "x" * 600 + "FINAL_ERROR_LINE")

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env")

    msg = str(excinfo.value)
    assert "FINAL_ERROR_LINE" in msg  # the useful end survived
    assert "HEAD_MARKER" not in msg  # the useless start did not
    assert "stderr: ..." in msg  # truncation is visibly marked


def test_error_envelope_falls_back_to_marked_empty_not_bare_colon():
    """`ok: false` with no message and no stderr must not end in a bare colon."""
    envelope = {"type": "envelope", "ok": False, "error": {"code": "x"}}

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._unwrap_envelope(envelope, ("env",), 1, "")

    assert str(excinfo.value) == "comfy env failed [x]: <empty>"


def test_error_envelope_stderr_fallback_keeps_its_tail():
    """The stderr fallback on `ok: false` is bounded from the END, marker intact."""
    stderr = "HEAD_MARKER" + "x" * 900 + "FINAL_ERROR_LINE"
    envelope = {"type": "envelope", "ok": False, "error": {"code": "x"}}

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._unwrap_envelope(envelope, ("env",), 1, stderr)

    msg = str(excinfo.value)
    assert msg.endswith("FINAL_ERROR_LINE")  # the cap must not re-clip the tail off
    assert "HEAD_MARKER" not in msg
    assert "[x]: ..." in msg


def test_streaming_no_envelope_error_includes_both_stream_tails(monkeypatch):
    """The streaming EOF path produces the same enriched message as `_run_comfy`."""
    procs: list[_FakeProc] = []

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _FakeProc(
            cmd,
            "starting run...\ncomfy-cli crashed before emitting a result\n",
            stderr_text="Traceback (most recent call last):\nRuntimeError: nope",
            env=env,
            encoding=encoding,
        )
        proc.returncode = 1
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(
            server._run_comfy_streaming("run", "--workflow", "wf.json", timeout=5.0)
        )

    msg = str(excinfo.value)
    assert "returned no JSON (exit 1)" in msg
    assert "RuntimeError: nope" in msg  # stderr tail
    assert "comfy-cli crashed before emitting a result" in msg  # stdout tail


def test_streaming_eof_ignores_a_trailing_progress_event(monkeypatch):
    """A crash whose last stdout line is a progress EVENT still reports both tails.

    The streaming path reads NDJSON, so at EOF `_last_json_object`'s fallback is
    typically the run's last progress/custom-node event — not a result. Unwrapping
    it would swallow the diagnostics this branch exists to surface (and an event
    carrying `ok: true` would be read as a successful run on a spend-capable
    tool), so it is filtered to None and the no-envelope branch fires instead.
    """

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _FakeProc(
            cmd,
            # An event that both parses AND claims success — the worst case.
            '{"type": "progress", "ok": true, "data": {"value": 3}}\n'
            "comfy-cli crashed mid-run\n",
            stderr_text="RuntimeError: nope",
            env=env,
            encoding=encoding,
        )
        proc.returncode = 1
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(
            server._run_comfy_streaming("run", "--workflow", "wf.json", timeout=5.0)
        )

    msg = str(excinfo.value)
    assert "returned no JSON (exit 1)" in msg  # not silently "succeeded"
    assert "RuntimeError: nope" in msg
    assert "comfy-cli crashed mid-run" in msg


def test_streaming_eof_still_refuses_an_incompatible_envelope(monkeypatch):
    """The filter keeps REAL envelopes: a bad `schema` still raises the version error.

    `_pump` only stops on `envelope/1`, so an envelope declaring another major
    reaches EOF. It is a genuine envelope, so it must flow through and be refused
    with the incompatible-version message — not demoted to "returned no JSON".
    """

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _FakeProc(
            cmd,
            '{"schema": "envelope/2", "type": "envelope", "ok": true, "data": {}}\n',
            env=env,
            encoding=encoding,
        )
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    with pytest.raises(server.ComfyCliError, match="incompatible comfy-cli envelope"):
        asyncio.run(
            server._run_comfy_streaming("run", "--workflow", "wf.json", timeout=5.0)
        )


def test_error_envelope_whitespace_message_falls_back_to_stderr():
    """A whitespace-only `error.message` is truthy but renders as a dangling colon."""
    envelope = {
        "type": "envelope",
        "ok": False,
        "error": {"code": "x", "message": "   "},
    }

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._unwrap_envelope(envelope, ("env",), 1, "the real reason")

    assert str(excinfo.value) == "comfy env failed [x]: the real reason"

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._unwrap_envelope(envelope, ("env",), 1, "")

    assert str(excinfo.value) == "comfy env failed [x]: <empty>"


# --- download_model: no JSON envelope on a successful fetch (BE-3345) -------


def test_download_model_synthesizes_success_on_plain_exit(patched_plain_run):
    """`comfy model download` streams text + exits 0, no envelope -> success.

    The download landed on disk (exit 0), so instead of raising the
    "returned no JSON" false negative — which would invite a bandwidth-expensive
    retry of a multi-GB fetch — a success payload is synthesized carrying the
    CLI's printed tail (where "Done in …" and the saved path live).
    """
    patched_plain_run(
        0,
        stderr="Downloading x.safetensors...\nDone in 55.8s. Saved to /models/x.safetensors",
    )

    result = server.download_model("https://hf.co/x.safetensors")

    assert result["ok"] is True
    assert result["action"] == "model download"
    assert "Done in 55.8s" in result["message"]


def test_download_model_nonzero_exit_still_raises(patched_plain_run):
    """A real download failure (non-zero exit, no envelope) must still raise."""
    patched_plain_run(1, stderr="HTTP 404: model not found")

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        server.download_model("https://hf.co/missing.safetensors")


def test_download_model_synthesizes_despite_stray_non_envelope_json(patched_plain_run):
    """A stray non-envelope JSON line on a clean download exit is still success.

    `_last_json_object` returns any JSON object (not just `type==envelope`), so a
    diagnostic line that happens to parse must NOT be mistaken for a result
    envelope and unwrapped into a spurious failure (BE-3345 edge case).
    """
    patched_plain_run(
        0,
        stdout='{"level": "info", "msg": "connection reused"}\n',
        stderr="Done in 12.3s. Saved to /models/x.safetensors",
    )

    result = server.download_model("https://hf.co/x.safetensors")

    assert result["ok"] is True
    assert result["action"] == "model download"
    assert "Done in 12.3s" in result["message"]


def test_download_model_envelope_then_diagnostic_keeps_envelope_data(patched_plain_run):
    """A real envelope FOLLOWED BY a diagnostic JSON line still wins (BE-3345).

    `_last_json_object` prefers a `type==envelope` object over a later plain JSON
    line, so a trailing diagnostic must NOT null out `real_envelope` and demote a
    genuine success into the synthesized fast-path — the envelope's `data` (the
    real saved-path metadata) must be returned, not the printed-text stopgap.
    """
    patched_plain_run(
        0,
        stdout=(
            '{"type": "envelope", "ok": true, "data": {"saved": "/models/x"}}\n'
            '{"level": "info", "msg": "cleanup done"}\n'
        ),
    )

    result = server.download_model("https://hf.co/x.safetensors")

    assert result == {"saved": "/models/x"}


def test_download_model_error_envelope_then_diagnostic_still_raises(patched_plain_run):
    """An error envelope followed by a diagnostic line still raises, not synthesized.

    Even on exit 0, a trailing diagnostic JSON line must not mask an earlier
    error envelope as a synthesized success — the error envelope is preferred and
    propagates its code (BE-3345).
    """
    patched_plain_run(
        0,
        stdout=(
            '{"type": "envelope", "ok": false, '
            '"error": {"code": "download_failed", "message": "checksum mismatch"}}\n'
            '{"level": "info", "msg": "cleanup done"}\n'
        ),
    )

    with pytest.raises(server.ComfyCliError, match=r"download_failed"):
        server.download_model("https://hf.co/x.safetensors")


def test_download_model_still_honors_a_real_error_envelope(patched_run):
    """A real error envelope on download still raises with its code (not synthesized)."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "download_failed", "message": "checksum mismatch"},
        }
    )

    with pytest.raises(server.ComfyCliError, match=r"download_failed"):
        server.download_model("https://hf.co/x.safetensors")


def test_download_model_keeps_saved_path_tail_on_verbose_output(patched_plain_run):
    """A verbose multi-GB fetch caps to the TAIL so the saved-path survives.

    `comfy model download` streams progress noise ahead of the `Done in …` /
    saved-path tail that the synthesized payload exists to surface. A front-slice
    cap would drop that tail as noise, so the message must keep the last chars.
    """
    tail = "Done in 903.4s. Saved to /models/checkpoints/big.safetensors"
    noise = "\n".join(f"Downloading... {i}% ({i * 40} MiB)" for i in range(100))
    patched_plain_run(0, stderr=f"{noise}\n{tail}")

    result = server.download_model("https://hf.co/big.safetensors")

    assert result["ok"] is True
    assert len(result["message"]) <= 1000
    assert "Saved to /models/checkpoints/big.safetensors" in result["message"]


def test_download_model_fallback_omits_url_when_no_output(patched_plain_run):
    """The no-output fallback message must not echo the (possibly signed) URL.

    A `model download` URL can carry a token / userinfo in its query string; the
    synthesized fallback lands in the tool response and host logs, so it reports
    only the flag-free `model download` action, never the raw args.
    """
    patched_plain_run(0)  # exit 0 with no stdout/stderr -> fallback message

    result = server.download_model("https://hf.co/x.safetensors?sig=SECRETTOKEN")

    assert result["ok"] is True
    assert "SECRETTOKEN" not in result["message"]
    assert "hf.co" not in result["message"]
    assert "model download" in result["message"]


def test_discover_maps_command_and_returns_data(patched_run):
    """discover wraps `comfy discover` and returns the envelope data verbatim."""
    surface = {"commands": ["run", "env"], "error_codes": ["server_not_running"]}
    calls = patched_run(envelope(data=surface))

    assert server.discover() == surface
    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["discover"]  # bare subcommand, no positional args


def test_which_maps_command_and_returns_data(patched_run):
    """which wraps `comfy which` and returns the selected-workspace data."""
    calls = patched_run(
        {"type": "envelope", "ok": True, "data": {"workspace": "/home/me/ComfyUI"}}
    )

    assert server.which() == {"workspace": "/home/me/ComfyUI"}
    assert calls[0]["cmd"][4:] == ["which"]  # no positional args


# --- streaming run_workflow(wait=True) -------------------------------------


def test_run_workflow_streams_progress_and_returns_data(patched_stream):
    """wait=True drives --json-stream, emits progress, returns the envelope data."""
    procs = patched_stream(_OK_STREAM)
    ctx = _RecordingCtx()

    result = asyncio.run(server.run_workflow("wf.json", wait=True, ctx=ctx))

    assert result == {"outputs": ["/x.png"]}  # final envelope's data
    assert len(ctx.calls) >= 1  # acceptance: >=1 progress notification

    cmd = procs[0].cmd
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json-stream", "--where", "local"]  # global flags first
    assert cmd[4:] == ["run", "--workflow", "wf.json", "--wait"]

    # The 2-node manifest becomes the progress total, and the value never drops.
    assert all(c["total"] == 2.0 for c in ctx.calls if c["total"] is not None)
    values = [c["progress"] for c in ctx.calls]
    assert values == sorted(values)  # monotonically non-decreasing
    assert values[-1] == 2.0  # both nodes finished


def test_run_workflow_stream_sets_no_watch_env(patched_stream):
    """The streaming path also suppresses comfy-cli's watcher via COMFY_NO_WATCH."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.run_workflow("wf.json", wait=True))

    assert procs[0].env["COMFY_WHERE"] == "local"
    assert procs[0].env["COMFY_NO_WATCH"] == "1"


def test_run_workflow_stream_forces_utf8_env(patched_stream):
    """The streaming (Popen) spawn path also forces UTF-8 for the Windows fix."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.run_workflow("wf.json", wait=True))

    assert procs[0].env["PYTHONUTF8"] == "1"
    assert procs[0].env["PYTHONIOENCODING"] == "utf-8"
    # And the parent-side stream read is pinned to UTF-8 to match (readline()/
    # stderr.read() would otherwise decode with the cp1252 parent locale).
    assert procs[0].encoding == "utf-8"


@_needs_posix_exec
def test_run_workflow_stream_carries_the_bin_dir_on_path(
    tmp_path, monkeypatch, patched_stream
):
    """The streaming (Popen) spawn site hands the child the rewritten PATH too."""
    exe = _dummy_comfy(tmp_path)
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setenv("PATH", "/usr/bin")
    procs = patched_stream(_OK_STREAM)
    monkeypatch.setattr(server.shutil, "which", _real_which)

    asyncio.run(server.run_workflow("wf.json", wait=True))

    assert procs[0].env["PATH"].split(os.pathsep)[0] == str(exe.parent)


def test_run_workflow_stream_closes_child_stdin(patched_stream):
    """The streaming spawn also refuses to hand the child our JSON-RPC stdin."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.run_workflow("wf.json", wait=True))

    assert procs[0].stdin_arg == subprocess.DEVNULL


def test_run_workflow_stream_error_envelope_raises_with_code(patched_stream):
    """An error envelope on the final line raises ComfyCliError with its code."""
    stream = (
        "\n".join(
            json.dumps(evt)
            for evt in [
                {"type": "queued", "nodes": [{"node_id": "1"}]},
                {"type": "progress", "node": "1", "completed": 1, "total": 4},
                {
                    "schema": "envelope/1",
                    "type": "envelope",
                    "ok": False,
                    "error": {"code": "execution_error", "message": "boom"},
                },
            ]
        )
        + "\n"
    )
    patched_stream(stream)
    ctx = _RecordingCtx()

    with pytest.raises(server.ComfyCliError, match="execution_error"):
        asyncio.run(server.run_workflow("wf.json", wait=True, ctx=ctx))


def test_run_workflow_stream_works_without_ctx(patched_stream):
    """A direct wait=True call with no Context still returns the final data."""
    patched_stream(_OK_STREAM)

    result = asyncio.run(server.run_workflow("wf.json", wait=True))

    assert result == {"outputs": ["/x.png"]}


def test_watch_job_streams_progress_and_returns_data(patched_stream):
    """watch_job tails `comfy jobs watch <id>`, emits progress, returns the data."""
    procs = patched_stream(_OK_STREAM)
    ctx = _RecordingCtx()

    result = asyncio.run(server.watch_job("pid", ctx=ctx))

    assert result == {"outputs": ["/x.png"]}  # final envelope's data unwrapped
    assert len(ctx.calls) >= 1  # progress ticks forwarded

    cmd = procs[0].cmd
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json-stream", "--where", "local"]  # global flags first
    assert cmd[4:] == ["jobs", "watch", "pid"]  # subcommand strictly after

    # Same overall bar as run_workflow: 2-node manifest -> total, never drops.
    assert all(c["total"] == 2.0 for c in ctx.calls if c["total"] is not None)
    values = [c["progress"] for c in ctx.calls]
    assert values == sorted(values)  # monotonically non-decreasing
    assert values[-1] == 2.0  # both nodes finished


def test_watch_job_stream_error_envelope_raises_with_code(patched_stream):
    """A watch that ends on an error envelope raises ComfyCliError with its code."""
    stream = (
        "\n".join(
            json.dumps(evt)
            for evt in [
                {"type": "queued", "nodes": [{"node_id": "1"}]},
                {
                    "schema": "envelope/1",
                    "type": "envelope",
                    "ok": False,
                    "error": {"code": "execution_error", "message": "boom"},
                },
            ]
        )
        + "\n"
    )
    patched_stream(stream)

    with pytest.raises(server.ComfyCliError, match="execution_error"):
        asyncio.run(server.watch_job("pid"))


class _BlockingProc:
    """A fake Popen whose stdout yields ``first_lines`` then blocks (forces a timeout)."""

    def __init__(self, cmd, first_lines):
        self.cmd = cmd
        self._lines = list(first_lines)
        self.stdout = self  # readline lives on the proc itself
        self.stderr = io.StringIO("")
        self.returncode = 0
        self.killed = False
        self._alive = True

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        time.sleep(1.0)  # outlives the test's tiny timeout, never yields the envelope
        return ""

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self, timeout=None):  # noqa: ARG002
        self._alive = False
        return self.returncode

    def kill(self):
        self.killed = True
        self._alive = False


def test_watch_job_times_out_returns_payload(monkeypatch):
    """A watch bounded by timeout_seconds returns a timed-out payload, not an error."""
    queued = json.dumps(
        {"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]}
    )
    procs: list[_BlockingProc] = []

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _BlockingProc(cmd, [queued + "\n"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    # MCP always injects a ctx, so the tracker has advanced by the time we expire.
    result = asyncio.run(
        server.watch_job("pid", timeout_seconds=0.25, ctx=_RecordingCtx())
    )

    # Consistent with wait_for_job: a {"timed_out": True, ...} marker, no raise.
    assert result["timed_out"] is True
    assert result["status"]["total"] == 2.0  # queued manifest was seen first
    assert result["status"]["nodes_done"] == 0  # never reached a completion
    assert procs[0].killed  # the child was cleaned up on timeout


def test_watch_job_times_out_reports_progress_without_ctx(monkeypatch):
    """A ctx-less watch still advances the tracker, so its timed-out snapshot is real."""
    queued = json.dumps(
        {"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]}
    )
    executed = json.dumps({"type": "executed", "node": "1"})
    procs: list[_BlockingProc] = []

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _BlockingProc(cmd, [queued + "\n", executed + "\n"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    # No ctx: the notification is a no-op, but tracker state must still advance
    # so the snapshot reports the node that actually finished (not all zeros).
    result = asyncio.run(server.watch_job("pid", timeout_seconds=0.25))

    assert result["timed_out"] is True
    assert result["status"]["total"] == 2.0
    assert result["status"]["nodes_done"] == 1  # the `executed` event was tracked
    assert result["status"]["progress"] == 1.0  # not None / not zero
    assert procs[0].killed


@pytest.mark.parametrize("bad_id", ["--help", "", "p\x001"])
def test_watch_job_rejects_an_unusable_prompt_id(bad_id):
    """watch_job shares the family guard: no dash-led, empty, or NUL-bearing id."""
    with pytest.raises(server.ComfyCliError, match="prompt_id"):
        asyncio.run(server.watch_job(bad_id))


def test_watch_job_rejects_embedded_nul_prompt_id():
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        asyncio.run(server.watch_job("pid\0"))


def test_watch_job_clamps_oversized_timeout(monkeypatch):
    """timeout_seconds is clamped to the module ceiling, not passed through raw."""
    seen: dict = {}

    async def fake_stream(*args, ctx=None, timeout=None, raise_on_timeout=True):
        seen["timeout"] = timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    asyncio.run(server.watch_job("pid", timeout_seconds=float("inf")))

    assert seen["timeout"] == server._MAX_WATCH_TIMEOUT


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_watch_job_rejects_a_non_positive_or_nan_timeout(monkeypatch, bad):
    """NaN slips through `min(max(...))` — it must not reach the child at all.

    Shares `_bounded_timeout` with `partner_generate`: `max(nan, 0.0)` is `nan`,
    which would land as `timeout=nan` and raise a bare ValueError out of the
    selector instead of a ComfyCliError.
    """
    started = False

    async def fake_stream(*args, ctx=None, timeout=None, raise_on_timeout=True):
        nonlocal started
        started = True
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        asyncio.run(server.watch_job("pid", timeout_seconds=bad))

    assert started is False


def test_run_workflow_wait_false_uses_plain_json_no_stream(monkeypatch):
    """wait=False keeps the plain --json _run_comfy path (no streaming, no --wait)."""
    seen: dict = {}

    def fake_run_comfy(*args, timeout=None):
        seen["args"] = args
        seen["timeout"] = timeout
        return {"prompt_id": "p1"}

    # If streaming were (wrongly) taken, this would blow up instead of returning.
    def boom(*a, **k):
        raise AssertionError("wait=False must not stream")

    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)
    monkeypatch.setattr(server, "_run_comfy_streaming", boom)

    result = asyncio.run(server.run_workflow("wf.json", wait=False))

    assert result == {"prompt_id": "p1"}
    assert seen["args"] == ("run", "--workflow", "wf.json")  # no --wait
    assert seen["timeout"] == 60.0


@pytest.mark.parametrize("oversized", [float("inf"), 86_400.0])
def test_run_workflow_clamps_oversized_timeout(monkeypatch, oversized):
    """timeout_seconds is clamped to the module ceiling, not passed through raw.

    Raw, `inf` reaches `asyncio.wait_for` and the `comfy run --wait` child is
    never given up on; a day-long finite bound is the same problem, slower.
    """
    seen: dict = {}

    async def fake_stream(*args, ctx=None, timeout=None, **kwargs):
        seen["timeout"] = timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    asyncio.run(server.run_workflow("wf.json", wait=True, timeout_seconds=oversized))

    assert seen["timeout"] == server._MAX_RUN_WORKFLOW_TIMEOUT


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_run_workflow_rejects_a_non_positive_or_nan_timeout(monkeypatch, bad):
    """NaN/0/negative are refused before the streaming child is spawned."""
    started = False

    async def fake_stream(*args, ctx=None, timeout=None, **kwargs):
        nonlocal started
        started = True
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        asyncio.run(server.run_workflow("wf.json", wait=True, timeout_seconds=bad))

    assert started is False


@pytest.mark.parametrize("ignored", [float("nan"), 0.0, float("inf")])
def test_run_workflow_wait_false_still_submits_with_an_odd_timeout(
    monkeypatch, ignored
):
    """`wait=False` never reads timeout_seconds, so the clamp must not reject it.

    The submit runs on its own fixed 60s budget; hardening the waiting path must
    not turn a working fire-and-return call into an error.
    """
    seen: dict = {}

    def fake_run_comfy(*args, timeout=None):
        seen["timeout"] = timeout
        return {"prompt_id": "p1"}

    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)

    result = asyncio.run(
        server.run_workflow("wf.json", wait=False, timeout_seconds=ignored)
    )

    assert result == {"prompt_id": "p1"}
    assert seen["timeout"] == 60.0


# --- BE-3343: timeout errors must surface the captured stdout/stderr tails ---
# A crashed-and-wedged comfy-cli (e.g. the BE-3328 Windows UnicodeEncodeError)
# wrote its diagnosis on stderr before being killed; the timeout handler used to
# discard it, so the failure looked identical to a genuinely slow run.


def test_tail_bounds_and_decodes():
    """`_tail` hard-bounds length and decodes bytes defensively (never raises)."""
    assert textutil._tail(None) == ""
    assert textutil._tail("") == ""
    assert textutil._tail(b"") == ""
    assert textutil._tail("  hello  ") == "hello"  # stripped
    # bytes decoded, invalid utf-8 replaced rather than raising
    assert textutil._tail(b"cafe\xff") == "cafe\ufffd"
    # a chatty child cannot inflate the payload past the bound (str and bytes)
    assert textutil._tail("x" * 999, limit=500) == "x" * 500
    assert len(textutil._tail(b"y" * 5000)) == 500
    # limit<=0 must NOT return the whole string (the `[-0:]` trap), it means "none"
    assert textutil._tail("anything", limit=0) == ""
    assert textutil._tail(b"anything", limit=0) == ""
    # slicing the raw bytes before decoding still yields the true tail
    assert textutil._tail(b"z" * 100 + b"tail-marker", limit=20) == (
        "z" * 9 + "tail-marker"
    )


def _timeout(stderr, stdout):
    """The ``TimeoutExpired`` a killed ``comfy discover`` child would raise.

    The captures are what ``subprocess.run(capture_output=True)`` attaches to the
    exception before re-raising; the message the tests below read is built by
    ``_run_comfy`` from its OWN cmd/timeout, so these two only have to be
    plausible.
    """
    return subprocess.TimeoutExpired(
        [server.COMFY_BIN, "discover"], 60.0, output=stdout, stderr=stderr
    )


def test_sync_timeout_surfaces_bytes_stderr(patched_run):
    """POSIX shape: TimeoutExpired carries bytes; the stderr traceback reaches the message."""
    patched_run(
        raises=_timeout(
            stderr=b"Traceback (most recent call last):\nUnicodeEncodeError: 'charmap'",
            stdout=b"partial stdout",
        )
    )
    with pytest.raises(server.ComfyCliError) as exc:
        server._run_comfy("discover", timeout=60.0)
    msg = str(exc.value)
    assert "comfy-cli timed out after 60.0s" in msg  # prefix preserved verbatim
    assert "UnicodeEncodeError" in msg  # the diagnosis is no longer discarded
    assert "partial stdout" in msg


def test_sync_timeout_surfaces_str_stderr(patched_run):
    """Windows shape: run() re-communicates after the kill and returns str captures."""
    patched_run(raises=_timeout(stderr="boom on stderr", stdout="half a line"))
    with pytest.raises(server.ComfyCliError) as exc:
        server._run_comfy("discover", timeout=60.0)
    msg = str(exc.value)
    assert "boom on stderr" in msg
    assert "half a line" in msg


def test_sync_timeout_with_no_captures_is_sane(patched_run):
    """None captures (nothing written before the kill) must not crash and read sanely."""
    patched_run(raises=_timeout(stderr=None, stdout=None))
    with pytest.raises(server.ComfyCliError) as exc:
        server._run_comfy("discover", timeout=60.0)
    msg = str(exc.value)
    assert "comfy-cli timed out after 60.0s" in msg
    assert "stderr tail: <empty>" in msg
    assert "stdout tail: <empty>" in msg


def test_plain_spawn_leads_its_own_process_group(patched_run):
    """The plain path spawns with `start_new_session=True`, like the streaming one.

    Prerequisite for the group kill below: without it the child sits in the MCP
    server's OWN process group, so `os.getpgid(child)` would resolve to the
    server and the timeout handler would SIGKILL itself.
    """
    calls = patched_run()

    server._run_comfy("env", timeout=5.0)

    assert calls[0]["start_new_session"] is True


class _GroupKillProc:
    """A `Popen` fake with a pid, for the process-group assertions below.

    conftest's `_FakeRunProc` deliberately carries no `pid` so `_kill_proc_tree`
    takes its single-child fallback rather than calling `os.killpg` on a made-up
    one. This fake has a pid, and the test stubs `os.killpg`, so the group path
    is the one under test.

    `exited` models the case the group kill exists for: the direct `comfy` child
    is already gone (a non-None `poll()`) while a forked grandchild is still
    running and holding the pipes — which is why `communicate()` timed out.
    """

    def __init__(self, cmd, exc, *, exited=False):
        self.args = cmd
        self.pid = 424242
        self.stdout = None
        self.stderr = None
        self.returncode = -1 if exited else None
        self.killed = False
        self._exc = exc
        self._communicates = 0

    def communicate(self, timeout=None):  # noqa: ARG002
        self._communicates += 1
        if self._communicates == 1:
            raise self._exc
        return "drained stdout", "drained stderr"  # the post-kill drain

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_sync_timeout_kills_the_whole_process_group(monkeypatch):
    """A timed-out plain spawn reaps comfy-cli's DESCENDANTS, not just comfy-cli.

    comfy-cli's long verbs fork real work — `comfy update` runs `git pull` and
    then a multi-GB `pip install -r requirements.txt`, `comfy model download`
    streams a large file — and both ride a 1800s ceiling. `subprocess.run` (what
    this path used to be) kills only the direct child at the deadline, so those
    grandchildren kept mutating the ComfyUI workspace and Python environment
    long after the tool had reported failure, or died mid-write. Assert on the
    signal that actually reaches the group.
    """
    proc = _GroupKillProc(
        [server.COMFY_BIN, "update", "all"], _timeout(stderr=None, stdout=None)
    )
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: proc)  # noqa: ARG005
    monkeypatch.setattr(
        server.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig))
    )

    with pytest.raises(server.ComfyCliError) as exc:
        server._run_comfy("update", "all", timeout=1800.0)

    # `start_new_session=True` makes the child its own group leader, so its pid
    # IS the pgid — signalled directly, without an `os.getpgid` lookup that
    # would raise (and so skip the kill) on an already-reaped leader.
    assert signalled == [(proc.pid, signal.SIGKILL)]  # the GROUP, not the child
    assert proc.killed is False  # so the single-child fallback never had to fire
    assert exc.value.timed_out is True
    # The drain after the kill is what recovers the partial output that
    # `subprocess.run` used to attach to the `TimeoutExpired` itself.
    assert "drained stderr" in str(exc.value)


def test_group_kill_is_not_gated_on_the_leader_still_running(monkeypatch):
    """A dead `comfy` does not mean a dead tree — kill the group regardless.

    The case this whole change exists for is a `comfy update` whose `git pull` /
    `pip install` (or a `model download`) outlives its parent: `comfy` itself has
    exited, the grandchild is still running and still holding the pipe open —
    which is exactly WHY `communicate()` blew its deadline. Gating the group kill
    on `proc.poll() is None` would read the exited leader and skip the kill in
    precisely that case, leaving the descendant to keep mutating the workspace.
    """
    proc = _GroupKillProc(
        [server.COMFY_BIN, "update", "all"],
        _timeout(stderr=None, stdout=None),
        exited=True,  # the leader is already gone; its group is not
    )
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: proc)  # noqa: ARG005
    monkeypatch.setattr(
        server.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig))
    )

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("update", "all", timeout=1800.0)

    assert signalled == [(proc.pid, signal.SIGKILL)]  # the survivors still die


def test_drain_second_timeout_keeps_the_longer_capture(monkeypatch):
    """A drain that times out too still reports what it managed to read.

    `communicate()` resumes the same accumulation buffers, so the capture on the
    drain's own `TimeoutExpired` is a superset of the first one's. Falling back
    to the first would silently under-report the diagnostics in the timeout
    message and the failure log.
    """

    class _DoubleTimeoutProc(_GroupKillProc):
        def communicate(self, timeout=None):  # noqa: ARG002
            self._communicates += 1
            if self._communicates == 1:
                raise self._exc
            raise subprocess.TimeoutExpired(
                self.args, 5.0, output=b"first chunk + more", stderr=b"traceback tail"
            )

    proc = _DoubleTimeoutProc(
        [server.COMFY_BIN, "update", "all"],
        _timeout(stderr=b"trace", stdout=b"first"),
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: proc)  # noqa: ARG005
    monkeypatch.setattr(server.os, "killpg", lambda pgid, sig: None)  # noqa: ARG005

    with pytest.raises(server.ComfyCliError) as exc:
        server._run_comfy("update", "all", timeout=1800.0)

    msg = str(exc.value)
    assert "first chunk + more" in msg  # the drain's longer read, not `first`
    assert "traceback tail" in msg


def test_drain_non_timeout_failure_falls_back_to_the_first_capture(monkeypatch):
    """A drain that dies on a decode error has nothing better than the first read."""

    class _DecodeFailProc(_GroupKillProc):
        def communicate(self, timeout=None):  # noqa: ARG002
            self._communicates += 1
            if self._communicates == 1:
                raise self._exc
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    proc = _DecodeFailProc(
        [server.COMFY_BIN, "update", "all"],
        _timeout(stderr="partial trace", stdout="partial out"),
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: proc)  # noqa: ARG005
    monkeypatch.setattr(server.os, "killpg", lambda pgid, sig: None)  # noqa: ARG005

    with pytest.raises(server.ComfyCliError) as exc:
        server._run_comfy("update", "all", timeout=1800.0)

    assert "partial trace" in str(exc.value)


def test_sync_non_timeout_failure_still_reaps_the_child(patched_run):
    """A decode error mid-drain must not leak the child either.

    `subprocess.run` wrapped every call in `with Popen(...)` and had a bare
    `except` that killed the process before re-raising; `_run_comfy_raw` owns
    the process by hand now, so it has to do that itself — and it kills the
    whole group rather than just the child.
    """
    calls = patched_run(
        raises=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    )

    with pytest.raises(UnicodeDecodeError):
        server._run_comfy("env", timeout=5.0)

    assert calls[0]["proc"].killed is True


class _BlockingProcWithStderr:
    """A blocking Popen fake that also carries buffered stderr (a crashed child's traceback)."""

    def __init__(self, cmd, first_lines, stderr_text):
        self.cmd = cmd
        self._lines = list(first_lines)
        self.stdout = self  # readline lives on the proc itself
        self.stderr = io.StringIO(stderr_text)
        self.returncode = 0
        self.killed = False
        self._alive = True

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        time.sleep(1.0)  # outlives the test's tiny timeout, never yields the envelope
        return ""

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self, timeout=None):  # noqa: ARG002
        self._alive = False
        return self.returncode

    def kill(self):
        self.killed = True
        self._alive = False


def test_streaming_timeout_surfaces_stdout_and_stderr_tails(monkeypatch):
    """A raising streaming timeout appends the NDJSON stdout tail and the child's stderr tail."""
    queued = json.dumps({"type": "queued", "nodes": [{"node_id": "1"}]})
    procs: list[_BlockingProcWithStderr] = []

    def fake_popen(cmd, stdout, stderr, text, env, **kwargs):  # noqa: ARG001
        proc = _BlockingProcWithStderr(
            cmd, [queued + "\n"], "Traceback ...\nUnicodeEncodeError: boom"
        )
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(
            server._run_comfy_streaming(
                "run", "--workflow", "wf.json", timeout=0.25, ctx=_RecordingCtx()
            )
        )
    msg = str(exc.value)
    assert "comfy-cli timed out after 0.25s" in msg  # prefix preserved
    assert "UnicodeEncodeError" in msg  # stderr tail surfaced after the kill
    assert queued in msg  # the one NDJSON line the child emitted (stdout tail)
    assert procs[0].killed  # child cleaned up


def test_streaming_timeout_stdout_tail_is_bounded(monkeypatch):
    """Even a chatty streaming child cannot inflate the raised message past the tail bound."""
    noisy = [("x" * 100 + "\n") for _ in range(50)]  # 5000+ chars of NDJSON
    procs: list[_BlockingProcWithStderr] = []

    def fake_popen(cmd, stdout, stderr, text, env, **kwargs):  # noqa: ARG001
        proc = _BlockingProcWithStderr(cmd, noisy, "")
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(
            server._run_comfy_streaming(
                "run", "--workflow", "wf.json", timeout=0.25, ctx=_RecordingCtx()
            )
        )
    msg = str(exc.value)
    # only the bounded (<=500 char) tail of the 5000+ char stream is embedded
    expected_tail = textutil._tail("".join(noisy))
    assert len(expected_tail) == 500
    assert expected_tail in msg
    assert "".join(noisy).strip() not in msg  # the full blob never made it in


class _StderrBlockingProc:
    """stdout hits EOF fast; ``stderr.read()`` blocks past the timeout.

    Reproduces the cancellation path: ``_drain`` finishes ``_pump`` + ``wait``
    and then suspends at ``await stderr_future``, so the outer ``wait_for``
    timeout cancels the future. Awaiting it in the handler re-raises
    ``CancelledError`` (a ``BaseException``) — which must NOT escape and mask the
    intended ``ComfyCliError``.
    """

    def __init__(self, cmd, stdout_lines):
        self.cmd = cmd
        self._lines = list(stdout_lines)
        self.stdout = self  # readline lives on the proc
        self.stderr = self  # read lives on the proc
        self.returncode = 0
        self.killed = False
        self._alive = True

    def readline(self):
        return self._lines.pop(0) if self._lines else ""  # EOF -> _pump breaks

    def read(self, size=-1):  # noqa: ARG002 — size ignored; models a blocking pipe
        time.sleep(1.0)  # outlives the tiny timeout; suspends _drain at stderr_future
        return ""

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self, timeout=None):  # noqa: ARG002
        self._alive = False
        return self.returncode

    def kill(self):
        self.killed = True
        self._alive = False


def test_streaming_timeout_stderr_cancel_still_raises(monkeypatch):
    """A timeout that cancels the stderr read must still raise ComfyCliError, not CancelledError."""
    queued = json.dumps({"type": "queued", "nodes": [{"node_id": "1"}]})

    def fake_popen(cmd, stdout, stderr, text, env, **kwargs):  # noqa: ARG001
        return _StderrBlockingProc(cmd, [queued + "\n"])

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(
            server._run_comfy_streaming(
                "run", "--workflow", "wf.json", timeout=0.25, ctx=_RecordingCtx()
            )
        )
    msg = str(exc.value)
    assert (
        "comfy-cli timed out after 0.25s" in msg
    )  # the real error, not a masked cancel
    assert queued in msg  # stdout tail still surfaced
    assert (
        "stderr tail: <empty>" in msg
    )  # tail gathering was cancelled -> empty, best-effort


# --- restart_comfyui (stop -> launch composition) --------------------------


def test_restart_comfyui_runs_stop_then_launch(patched_run):
    """restart is a thin `comfy stop` then `comfy launch --background` composition."""
    calls = patched_run(envelope(data={"pid": 7}))

    # patched_run's fake emits the same envelope for every call, so launch's
    # data ({"pid": 7}) is what restart returns.
    assert server.restart_comfyui() == {"pid": 7}

    assert len(calls) == 2  # exactly stop then launch, nothing else
    assert calls[0]["cmd"][4:] == ["stop"]
    assert calls[1]["cmd"][4:] == ["launch", "--background"]


def test_restart_comfyui_forwards_extra_args_to_launch(patched_run):
    """extra_args ride the launch step after the `--` separator, not the stop."""
    calls = patched_run(envelope(data={}))

    server.restart_comfyui(["--port", "8189"])

    assert calls[0]["cmd"][4:] == ["stop"]  # stop takes no extras
    assert calls[1]["cmd"][4:] == ["launch", "--background", "--", "--port", "8189"]


def test_restart_comfyui_tolerates_no_recorded_server(monkeypatch):
    """A failed stop (nothing recorded to stop) is swallowed; launch still runs."""
    launched: list = []

    def fake_stop():
        raise server.ComfyCliError("comfy stop failed [no_recorded_server]: none")

    def fake_launch(extra_args=None):
        launched.append(extra_args)
        return {"pid": 1}

    monkeypatch.setattr(server, "stop_comfyui", fake_stop)
    monkeypatch.setattr(server, "launch_comfyui", fake_launch)

    assert server.restart_comfyui(["--cpu"]) == {"pid": 1}
    assert launched == [["--cpu"]]  # launch happened despite the stop error


def test_restart_comfyui_reraises_genuine_stop_failure(monkeypatch):
    """A stop failure that ISN'T 'no recorded server' propagates — launch is skipped."""
    launched: list = []

    def fake_stop():
        raise server.ComfyCliError(
            "comfy stop failed [permission_denied]: cannot kill pid 7",
            code="permission_denied",
        )

    monkeypatch.setattr(server, "stop_comfyui", fake_stop)
    monkeypatch.setattr(
        server, "launch_comfyui", lambda extra_args=None: launched.append(extra_args)
    )

    with pytest.raises(server.ComfyCliError, match="permission_denied"):
        server.restart_comfyui()
    assert launched == []  # genuine failure is not masked by a relaunch


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_restart_comfyui_tolerates_plain_no_comfyui_running_text(
    patched_plain_run, monkeypatch, stream
):
    """The REAL shape of "nothing to stop": non-zero exit, no envelope, human text.

    comfy-cli (1.12.0 `cmdline.stop`) does not emit a `no_recorded_server`
    envelope when it has no background server recorded — it prints "No ComfyUI is
    running in the background." and exits 1, which carries neither a structured
    code nor the literal marker string. That is the common real-world case (a
    foreground `comfy launch`, the desktop app, `python main.py`, or nothing
    running), and it used to abort the restart before it ever reached the launch
    step.

    Parametrized over the stream because comfy-cli prints this one through Rich
    (stdout) while the QA report captured it on stderr — the match must not care
    which, so neither does this test.
    """
    calls = patched_plain_run(1, **{stream: "No ComfyUI is running in the background."})
    launched: list = []

    def fake_launch(extra_args=None):
        launched.append(extra_args)
        return {"pid": 3}

    monkeypatch.setattr(server, "launch_comfyui", fake_launch)

    assert server.restart_comfyui(["--cpu"]) == {"pid": 3}

    assert calls[0]["cmd"][4:] == ["stop"]  # the stop really was attempted
    assert launched == [["--cpu"]]  # and the relaunch still happened


def test_stop_comfyui_plain_no_comfyui_running_carries_no_structured_code(
    patched_plain_run,
):
    """Pin the gap this fix closes: that failure has no code and no envelope."""
    patched_plain_run(1, stdout="No ComfyUI is running in the background.")

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.stop_comfyui()

    assert excinfo.value.code is None  # nothing structured to branch on
    assert excinfo.value.no_envelope is True
    assert server._NO_RECORDED_SERVER_CODE not in str(excinfo.value)
    assert server._is_no_recorded_server(excinfo.value)  # matched on the text


@pytest.mark.parametrize(
    "message",
    [
        # comfy-cli 1.12.0, as the wrapper renders it — on either stream.
        "comfy-cli returned no JSON (exit 1). stderr: <empty> | stdout: No "
        "ComfyUI is running in the background.",
        "comfy-cli returned no JSON (exit 1). stderr: No ComfyUI is running in "
        "the background. | stdout: <empty>",
        "No ComfyUI is running in the background.",
        "no comfyui is running in the background",  # casing drift
        "No ComfyUI server is running in the background.",  # inserted word
        "No ComfyUI running in the background",  # dropped copula
        # Rich soft-wrapping the sentence at a narrow width: a line break inside
        # the phrase is a wrap, not a clause break.
        "No ComfyUI is running in the\nbackground.",
        # A clipped capture: `textutil._stream_tail` prefixes `...`, which can
        # land directly against the sentence. That marker opens a field.
        "comfy-cli returned no JSON (exit 1). stderr: <empty> | stdout: ...No "
        "ComfyUI is running in the background.",
        "comfy stop failed [no_recorded_server]: none",  # the pre-existing marker
    ],
)
def test_is_no_recorded_server_matches_wording_drift(message):
    """The benign case is matched on the stable phrase, not one exact sentence."""
    assert server._is_no_recorded_server(server.ComfyCliError(message))


@pytest.mark.parametrize(
    "message",
    [
        "comfy stop failed [permission_denied]: cannot kill pid 7",
        # comfy-cli's OTHER stop message, verbatim: it DID have a server
        # recorded and could not kill it. Nothing benign about that one.
        "comfy-cli returned no JSON (exit 1). stderr: <empty> | stdout: Failed "
        "to stop ComfyUI in the background.",
        "comfy-cli returned no JSON (exit 1). stderr: Failed to stop ComfyUI "
        "running in the background: operation not permitted | stdout: <empty>",
        "comfy-cli returned no JSON (exit 2). stderr: Traceback (most recent "
        "call last): RuntimeError | stdout: <empty>",
        # Two unrelated sentences: the match must not span the sentence break.
        "No ComfyUI workspace is configured. Something is running in the background.",
        # A failed stop whose clauses are separated by a semicolon, not a period:
        # this one says the server IS still running, the opposite of benign.
        "No ComfyUI process could be stopped; it is still running in the background.",
        # The two halves living in DIFFERENT streams of the wrapper's rendering —
        # the match must not stitch across the ` | ` delimiter.
        "comfy-cli returned no JSON (exit 1). stderr: No ComfyUI workspace "
        "configured | stdout: a server is running in the background",
        # ...nor across a field label on its own line.
        "comfy stop failed\nstderr: No ComfyUI workspace configured\n"
        "stdout: a server is running in the background",
        # The phrase as ADVICE inside another failure's hint, not as a report
        # that nothing was recorded: it does not open a message, line, or field.
        "comfy stop failed: cannot kill pid 7 (operation not permitted); "
        "check permissions and ensure no ComfyUI is running in the background",
        # Opposite-meaning reports joined WITHOUT punctuation — a conjunction or
        # a dash. The halves must be joined by the sentence's grammar, not just
        # sit near each other.
        "No ComfyUI process was stopped and remains running in the background",
        "No ComfyUI process could be stopped — it is still running in the background",
        "No ComfyUI was stopped so something else is running in the background",
        # A longer, unrelated structured code that merely starts with the marker.
        "comfy stop failed [no_recorded_server_pid]: none",
    ],
)
def test_is_no_recorded_server_rejects_other_failures(message):
    """Every OTHER stop failure stays outside the benign net."""
    assert not server._is_no_recorded_server(server.ComfyCliError(message))


def test_is_no_recorded_server_lets_a_structured_code_outrank_the_text():
    """comfy-cli said structurally what broke; stray prose does not overrule it."""
    exc = server.ComfyCliError(
        "No ComfyUI is running in the background.", code="permission_denied"
    )

    assert not server._is_no_recorded_server(exc)


def test_is_no_recorded_server_rejects_a_stop_we_timed_out():
    """A stop killed at OUR deadline never finished, so it reported nothing."""
    exc = server.ComfyCliError(
        "comfy stop timed out after 60s. stdout: No ComfyUI is running in the "
        "background.",
        timed_out=True,
    )

    assert not server._is_no_recorded_server(exc)


@pytest.mark.parametrize(
    "exc",
    [
        # The gate has to sit above BOTH text reads, not between them: a message
        # quoting the literal marker must not slip past a timeout or a different
        # structured code.
        server.ComfyCliError(
            "comfy stop timed out after 60s. stdout: comfy stop failed "
            "[no_recorded_server]: none",
            timed_out=True,
        ),
        server.ComfyCliError(
            "comfy stop failed [permission_denied]: cannot kill pid 7 "
            "(previous run reported no_recorded_server)",
            code="permission_denied",
        ),
    ],
)
def test_is_no_recorded_server_gate_outranks_the_literal_marker_too(exc):
    """A quoted marker does not make a timeout or a coded failure benign."""
    assert not server._is_no_recorded_server(exc)


def test_is_no_recorded_server_accepts_an_envelope_without_a_code():
    """An envelope carrying the sentence but no `error.code` is the same case."""
    exc = server.ComfyCliError(
        "comfy stop failed: No ComfyUI is running in the background."
    )

    assert exc.no_envelope is False  # not gated on provenance, only on code
    assert server._is_no_recorded_server(exc)


def test_restart_comfyui_reraises_a_timed_out_stop_that_printed_the_phrase(monkeypatch):
    """The gate is load-bearing: a timed-out stop must not be relaunched over."""

    def fake_stop():
        raise server.ComfyCliError(
            "comfy stop timed out after 60s. stdout: No ComfyUI is running in "
            "the background.",
            timed_out=True,
        )

    launched: list = []
    monkeypatch.setattr(server, "stop_comfyui", fake_stop)
    monkeypatch.setattr(
        server, "launch_comfyui", lambda extra_args=None: launched.append(extra_args)
    )

    with pytest.raises(server.ComfyCliError, match="timed out"):
        server.restart_comfyui()
    assert launched == []


def test_restart_comfyui_reraises_unrelated_plain_stop_failure(
    patched_plain_run, monkeypatch
):
    """A plain non-zero stop whose text ISN'T the benign phrase still aborts."""
    patched_plain_run(1, stderr="Failed to kill pid 7: operation not permitted")
    launched: list = []
    monkeypatch.setattr(
        server, "launch_comfyui", lambda extra_args=None: launched.append(extra_args)
    )

    with pytest.raises(server.ComfyCliError, match="operation not permitted"):
        server.restart_comfyui()
    assert launched == []


def test_restart_comfyui_explains_port_clash_after_nothing_to_stop(monkeypatch):
    """Nothing to stop + port taken = a running server comfy-cli never launched."""

    def fake_stop():
        raise server.ComfyCliError("No ComfyUI is running in the background.")

    def fake_launch(extra_args=None):  # noqa: ARG001
        raise server.ComfyCliError(
            "comfy-cli returned no JSON (exit 1). stderr: The 8188 port is "
            "already in use. | stdout: <empty>",
            no_envelope=True,
            returncode=1,
        )

    monkeypatch.setattr(server, "stop_comfyui", fake_stop)
    monkeypatch.setattr(server, "launch_comfyui", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.restart_comfyui()

    message = str(excinfo.value)
    assert "The 8188 port is already in use." in message  # original kept verbatim
    assert "comfy-cli has no record of launching it" in message
    assert "restart_comfyui" in message  # the different-port way out
    # Provenance of the underlying failure survives the re-wrap.
    assert excinfo.value.no_envelope is True
    assert excinfo.value.returncode == 1


def test_restart_comfyui_leaves_port_clash_alone_after_a_real_stop(monkeypatch):
    """A port clash after a stop that DID kill comfy-cli's server is a different bug."""
    monkeypatch.setattr(server, "stop_comfyui", lambda: {"ok": True})

    def fake_launch(extra_args=None):  # noqa: ARG001
        raise server.ComfyCliError("The 8188 port is already in use.")

    monkeypatch.setattr(server, "launch_comfyui", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.restart_comfyui()

    assert str(excinfo.value) == "The 8188 port is already in use."


def test_restart_comfyui_leaves_non_port_launch_failure_alone(monkeypatch):
    """A launch failure that isn't a port clash keeps its own message."""
    monkeypatch.setattr(
        server,
        "stop_comfyui",
        lambda: (_ for _ in ()).throw(
            server.ComfyCliError("No ComfyUI is running in the background.")
        ),
    )

    def fake_launch(extra_args=None):  # noqa: ARG001
        raise server.ComfyCliError("ComfyUI exited during startup: missing torch")

    monkeypatch.setattr(server, "launch_comfyui", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.restart_comfyui()

    assert str(excinfo.value) == "ComfyUI exited during startup: missing torch"


@pytest.mark.parametrize(
    "message",
    [
        "The 8188 port is already in use.",  # comfy-cli's own preflight
        "OSError: [Errno 48] Address already in use",  # the socket bind under it
        "error while attempting to bind on address ('0.0.0.0', 8188): "
        "address already in use",
        "Port 8188 is already in use",
    ],
)
def test_port_in_use_matches_the_real_phrasings(message):
    """Both layers word the port clash differently; both must be recognized."""
    assert server._PORT_IN_USE_TEXT_RE.search(message)


@pytest.mark.parametrize(
    "message",
    [
        # "already in use" with a subject that is NOT a port: appending the port
        # guidance here would assert something plainly false about the failure.
        "Cannot load model: the file is already in use by another process",
        "CUDA device 0 is already in use",
        "The output directory is already in use",
        # A `--port` echoed back from the command in one stream must not be
        # stitched to an unrelated "already in use" in the other.
        "comfy-cli returned no JSON (exit 1). stderr: launch --background -- "
        "--port 8188 | stdout: CUDA device 0 is already in use",
        "comfy-cli returned no JSON (exit 1). stderr: could not bind port | "
        "stdout: the model file is already in use",
    ],
)
def test_port_in_use_ignores_non_port_conflicts(message):
    """A busy file or GPU is not a port clash, so it gets no port guidance."""
    assert not server._PORT_IN_USE_TEXT_RE.search(message)


def test_restart_comfyui_leaves_a_non_port_resource_clash_alone(monkeypatch):
    """End to end: a busy model file after nothing-to-stop keeps its own message."""
    monkeypatch.setattr(
        server,
        "stop_comfyui",
        lambda: (_ for _ in ()).throw(
            server.ComfyCliError("No ComfyUI is running in the background.")
        ),
    )

    def fake_launch(extra_args=None):  # noqa: ARG001
        raise server.ComfyCliError("Cannot load model: the file is already in use")

    monkeypatch.setattr(server, "launch_comfyui", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.restart_comfyui()

    assert str(excinfo.value) == "Cannot load model: the file is already in use"


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (None, None),
        ([], None),
        (["--cpu"], None),
        (["--port", "8189"], 8189),
        (["--port=8189"], 8189),
        (["--cpu", "--port", "8300", "--lowvram"], 8300),
        (["--port", "8189", "--port", "8300"], 8300),  # last one wins
        (["--port"], None),  # dangling flag, no value
        (["--port", "not-a-number"], None),
        (["--port", "0"], None),  # out of range
        (["--port", "70000"], None),
        (["--portable"], None),  # a different flag that merely shares the prefix
        # A trailing unparseable --port supersedes an earlier good one: it means
        # we do not know the requested port, not that 8189 still stands.
        (["--port", "8189", "--port", "bad"], None),
        (["--port", "8189", "--port", "99999"], None),
    ],
)
def test_requested_port_reads_the_forwarded_port(extra_args, expected):
    """Best-effort read of the caller's `--port`; anything odd yields None."""
    assert server._requested_port(extra_args) == expected


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (None, ["--port", "8189"]),
        ([], ["--port", "8189"]),
        (["--cpu"], ["--cpu", "--port", "8189"]),  # other flags survive
        (["--cpu", "--port", "8188"], ["--cpu", "--port", "8189"]),
        (["--port=8188", "--lowvram"], ["--lowvram", "--port", "8189"]),
        (["--port"], ["--port", "8189"]),  # dangling flag dropped, not doubled
    ],
)
def test_suggested_relaunch_args_swaps_the_port_and_keeps_the_rest(
    extra_args, expected
):
    """The pasteable suggestion must not silently drop the caller's own flags."""
    assert server._suggested_relaunch_args(extra_args, 8189) == expected


def test_untracked_server_guidance_keeps_the_callers_other_flags():
    """A user pasting the suggestion should not lose `--cpu` along the way."""
    guidance = server._untracked_server_guidance(["--cpu", "--port", "8188"])

    assert 'extra_args=["--cpu", "--port", "8189"]' in guidance


def test_untracked_server_guidance_falls_back_when_the_args_are_long():
    """Past the cap the suggestion degrades to the bare port rather than noise."""
    guidance = server._untracked_server_guidance(["--some-very-long-flag"] * 12)

    assert 'extra_args=["--port", "8189"]' in guidance
    assert "--some-very-long-flag" not in guidance


def test_untracked_server_guidance_avoids_suggesting_the_port_that_just_failed():
    """Suggesting the exact launch that just lost the race is useless advice."""
    default = server._untracked_server_guidance(["--cpu"])
    assert f'"--port", "{server._ALT_PORT_SUGGESTION}"' in default

    collided = server._untracked_server_guidance(
        ["--port", str(server._ALT_PORT_SUGGESTION)]
    )
    assert f'"--port", "{server._ALT_PORT_SUGGESTION}"' not in collided
    assert f'"--port", "{server._ALT_PORT_FALLBACK}"' in collided


def test_restart_comfyui_port_guidance_reflects_the_requested_port(monkeypatch):
    """The suggestion the caller actually sees is derived from their own args."""

    def fake_stop():
        raise server.ComfyCliError("No ComfyUI is running in the background.")

    def fake_launch(extra_args=None):  # noqa: ARG001
        raise server.ComfyCliError("The 8189 port is already in use.")

    monkeypatch.setattr(server, "stop_comfyui", fake_stop)
    monkeypatch.setattr(server, "launch_comfyui", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.restart_comfyui(["--port", str(server._ALT_PORT_SUGGESTION)])

    message = str(excinfo.value)
    assert "The 8189 port is already in use." in message  # original kept verbatim
    assert f'"--port", "{server._ALT_PORT_FALLBACK}"' in message


def test_error_envelope_populates_structured_code(patched_run):
    """ComfyCliError from an error envelope carries the code as an attribute, not just text."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "ComfyUI not running"},
        }
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env")
    assert excinfo.value.code == "server_not_running"


def test_restart_comfyui_returns_new_server_status(monkeypatch):
    """restart returns launch_comfyui's data (the fresh server status), not stop's."""
    monkeypatch.setattr(server, "stop_comfyui", lambda: {"stopped": True})
    monkeypatch.setattr(
        server, "launch_comfyui", lambda extra_args=None: {"pid": 42, "port": 8188}
    )

    assert server.restart_comfyui() == {"pid": 42, "port": 8188}


# --- update_comfyui (`comfy update [all|comfy|cli]`) ------------------------


def test_update_comfyui_defaults_to_core_target(patched_plain_run):
    """Bare call updates ComfyUI core: `comfy … update comfy`, plain-exit success.

    `comfy update` never emits an envelope — it prints through comfy-cli's
    rprint shim (which routes to stderr in `--json` mode) and exits 0 — so this
    rides the same `plain_ok` synthesis as launch/stop.
    """
    calls = patched_plain_run(
        0, stderr="Updating ComfyUI in /ws...\nAlready up to date."
    )

    result = server.update_comfyui()

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert cmd[4:] == ["update", "comfy"]
    assert result["ok"] is True
    assert result["action"] == "update comfy"
    assert "Already up to date" in result["message"]


@pytest.mark.parametrize("target", ["all", "comfy", "cli"])
def test_update_comfyui_forwards_each_accepted_target(patched_plain_run, target):
    """Every target comfy-cli accepts is forwarded verbatim."""
    calls = patched_plain_run(0, stderr="done")

    server.update_comfyui(target)

    assert calls[0]["cmd"][4:] == ["update", target]


def test_update_comfyui_nonzero_exit_raises(patched_plain_run):
    """A failed update (non-zero exit, no envelope) must still raise, not synthesize."""
    calls = patched_plain_run(
        1, stderr="error: Your local changes would be overwritten"
    )

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        server.update_comfyui()

    assert len(calls) == 1  # it really did run, and the failure was not swallowed


def test_update_comfyui_surfaces_error_envelope(patched_run):
    """If comfy-cli does emit an error envelope, its code still propagates."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "workspace_not_found", "message": "no ComfyUI path"},
        }
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.update_comfyui("comfy")
    assert excinfo.value.code == "workspace_not_found"


@pytest.mark.parametrize(
    "target",
    ["nodes", "", "  ", "comfy; rm -rf /", "--help", "core"],
)
def test_update_comfyui_rejects_unknown_target_before_spawning(patched_run, target):
    """An unaccepted target is named and refused BEFORE any subprocess runs."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="invalid update target"):
        server.update_comfyui(target)

    assert calls == []  # nothing was forwarded to comfy-cli


def test_update_comfyui_error_names_the_allowed_targets(patched_run):
    """The rejection is explicit about what IS accepted, not a bare refusal."""
    patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.update_comfyui("everything")
    message = str(excinfo.value)
    assert "'everything'" in message  # the offending value, echoed back
    for allowed in ("'all'", "'comfy'", "'cli'"):
        assert allowed in message


def test_update_comfyui_normalizes_case_and_whitespace(patched_plain_run):
    """`" Comfy "` resolves to the canonical target; the raw string never hits argv."""
    calls = patched_plain_run(0, stderr="done")

    server.update_comfyui("  COMFY ")

    assert calls[0]["cmd"][4:] == ["update", "comfy"]


def test_update_comfyui_timeout_is_generous(patched_plain_run):
    """The update timeout must comfortably exceed launch_comfyui's 180s boot."""
    calls = patched_plain_run(0, stderr="done")

    server.update_comfyui()

    assert calls[0]["timeout"] >= 180.0
    assert calls[0]["timeout"] == server._UPDATE_TIMEOUT


def test_update_comfyui_is_non_interactive(patched_plain_run):
    """An update must never stop to ask git/pip a question it cannot be answered.

    The 30-minute ceiling makes a silent credential prompt the worst case: with
    stdin inherited it would both eat JSON-RPC bytes and hang for half an hour.
    """
    calls = patched_plain_run(0, stderr="done")

    server.update_comfyui()

    assert calls[0]["stdin"] == subprocess.DEVNULL
    assert calls[0]["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert calls[0]["env"]["PIP_NO_INPUT"] == "1"


def test_update_comfyui_refuses_a_concurrent_update(monkeypatch, patched_plain_run):
    """A second update while one is in flight is refused, not run in parallel.

    FastMCP dispatches sync tools onto a worker thread pool, so two calls really
    can overlap — and both would then drive git/pip against the same checkout and
    Python environment. A real second thread here, pinned inside the first
    update's subprocess (i.e. while the lock is held).
    """
    calls = patched_plain_run(0, stderr="done")
    inside = threading.Event()
    release = threading.Event()
    fixture_popen = server.subprocess.Popen

    def blocking_popen(*args, **kwargs):
        inside.set()  # the first update now holds the lock...
        release.wait(5)  # ...and keeps holding it until this test lets go
        return fixture_popen(*args, **kwargs)

    monkeypatch.setattr(server.subprocess, "Popen", blocking_popen)
    worker = threading.Thread(target=server.update_comfyui, args=("comfy",))
    worker.start()
    try:
        assert inside.wait(5), "the first update never reached its subprocess"
        with pytest.raises(server.ComfyCliError, match="already running"):
            server.update_comfyui("all")
    finally:
        release.set()
        worker.join(5)

    # The refused call never reached comfy-cli; only the first update spawned.
    assert [c["cmd"][4:] for c in calls] == [["update", "comfy"]]


def test_update_comfyui_lock_is_released_after_failure(patched_plain_run):
    """A failed update must not wedge every later update behind a held lock."""
    patched_plain_run(1, stderr="error: local changes would be overwritten")

    with pytest.raises(server.ComfyCliError):
        server.update_comfyui()

    # The lock is free again, so a retry proceeds instead of being refused.
    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_update_comfyui_lock_is_released_after_success(patched_plain_run):
    """The happy path releases too — two sequential updates are always allowed."""
    calls = patched_plain_run(0, stderr="done")

    server.update_comfyui("comfy")
    server.update_comfyui("cli")

    assert [c["cmd"][4:] for c in calls] == [["update", "comfy"], ["update", "cli"]]


def test_update_comfyui_invalid_target_does_not_take_the_lock(patched_run):
    """A rejected target leaves the lock untouched, so a good call still works."""
    patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="invalid update target"):
        server.update_comfyui("nope")

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


# --- fetch_outputs inline image return -------------------------------------

# A few bytes standing in for a PNG — Image just base64-encodes the file, it
# does not decode/validate the pixels, so any bytes exercise the round-trip.
_FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-pixels"


def test_fetch_outputs_inline_images_returns_image_content(patched_run, tmp_path):
    """inline_images=True returns [metadata, Image...] for each downloaded image."""
    (tmp_path / "gen.png").write_bytes(_FAKE_PNG)
    patched_run(envelope(data={"downloaded": ["gen.png"]}))

    result = server.fetch_outputs("pid", str(tmp_path), inline_images=True)

    assert isinstance(result, list)
    assert result[0] == {"downloaded": ["gen.png"]}  # metadata preserved first
    images = [r for r in result[1:] if isinstance(r, server.Image)]
    assert len(images) == 1

    content = images[0].to_image_content()
    assert content.type == "image"
    assert content.mimeType == "image/png"
    import base64

    assert base64.b64decode(content.data) == _FAKE_PNG  # the real file bytes


def test_fetch_outputs_inline_images_resolves_nested_absolute_paths(
    patched_run, tmp_path
):
    """Absolute image paths nested anywhere in the data (under out_dir) are found."""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"jpeg-bytes")
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {"files": [{"path": str(img), "node": "SaveImage"}]},
        }
    )

    # The absolute path the data carries lives inside out_dir (comfy download -o
    # writes there), so it is inlined.
    result = server.fetch_outputs("pid", str(tmp_path), inline_images=True)

    images = [r for r in result if isinstance(r, server.Image)]
    assert len(images) == 1
    assert images[0].to_image_content().mimeType == "image/jpeg"


def test_fetch_outputs_inline_images_rejects_paths_outside_out_dir(
    patched_run, tmp_path
):
    """A real image referenced by the data but living OUTSIDE out_dir is not inlined.

    comfy download only writes into out_dir, so an absolute/`..` path escaping it
    is an input reference or a traversal, never a file this job produced.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(_FAKE_PNG)  # a real image, but outside out_dir

    patched_run(
        {
            "type": "envelope",
            "ok": True,
            # both an absolute escape and a ../ traversal that resolve outside
            "data": {"abs": str(outside), "rel": "../elsewhere.png"},
        }
    )

    result = server.fetch_outputs("pid", str(out_dir), inline_images=True)

    assert not any(isinstance(r, server.Image) for r in result)


def test_fetch_outputs_inline_images_prefers_out_dir_over_cwd(
    patched_run, tmp_path, monkeypatch
):
    """A bare filename binds to the copy in out_dir, not a same-named file in the CWD."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "gen.png").write_bytes(_FAKE_PNG)  # the real output

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "gen.png").write_bytes(b"cwd-decoy-not-this-one")  # a CWD shadow
    monkeypatch.chdir(cwd)

    patched_run(envelope(data={"downloaded": ["gen.png"]}))

    result = server.fetch_outputs("pid", str(out_dir), inline_images=True)

    images = [r for r in result if isinstance(r, server.Image)]
    assert len(images) == 1
    import base64

    # the out_dir copy, never the CWD decoy
    assert base64.b64decode(images[0].to_image_content().data) == _FAKE_PNG


def test_fetch_outputs_url_only_returns_no_inline_images(patched_run, tmp_path):
    """url_only downloads no bytes, so inline_images can't surface stale out_dir files."""
    # A stale image from a prior run whose basename the emitted URL echoes.
    (tmp_path / "old.png").write_bytes(_FAKE_PNG)
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {"urls": ["http://localhost:8188/view/old.png"]},
        }
    )

    result = server.fetch_outputs(
        "pid", str(tmp_path), url_only=True, inline_images=True
    )

    # Bare envelope data — no [data, Image...] wrapping, no stale file inlined.
    assert result == {"urls": ["http://localhost:8188/view/old.png"]}


def test_fetch_outputs_inline_images_caps_count(patched_run, tmp_path):
    """No more than _INLINE_IMAGE_MAX_COUNT images are inlined, however many exist."""
    names = [f"g{i}.png" for i in range(server._INLINE_IMAGE_MAX_COUNT + 5)]
    for name in names:
        (tmp_path / name).write_bytes(_FAKE_PNG)
    patched_run(envelope(data={"downloaded": names}))

    result = server.fetch_outputs("pid", str(tmp_path), inline_images=True)

    images = [r for r in result if isinstance(r, server.Image)]
    assert len(images) == server._INLINE_IMAGE_MAX_COUNT


def test_fetch_outputs_inline_images_caps_aggregate_bytes(
    patched_run, tmp_path, monkeypatch
):
    """Inlining stops once the aggregate byte budget would be exceeded."""
    monkeypatch.setattr(server, "_INLINE_IMAGE_MAX_BYTES", 10)
    for name in ("a.png", "b.png", "c.png"):
        (tmp_path / name).write_bytes(b"1234567")  # 7 bytes each
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {"downloaded": ["a.png", "b.png", "c.png"]},
        }
    )

    result = server.fetch_outputs("pid", str(tmp_path), inline_images=True)

    images = [r for r in result if isinstance(r, server.Image)]
    # first (7B) fits; second would push to 14B > 10B budget -> stop.
    assert len(images) == 1


def test_fetch_outputs_inline_images_skips_non_image_and_missing(patched_run, tmp_path):
    """Non-image outputs and dangling references yield no inline images."""
    (tmp_path / "notes.txt").write_bytes(b"hello")
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {"downloaded": ["notes.txt", "ghost.png"]},
        }
    )

    result = server.fetch_outputs("pid", str(tmp_path), inline_images=True)

    assert result[0] == {"downloaded": ["notes.txt", "ghost.png"]}
    assert not any(isinstance(r, server.Image) for r in result)  # neither qualifies


def test_fetch_outputs_inline_images_still_downloads(patched_run, tmp_path):
    """Inline return is additive: the `comfy download -o` copy command is unchanged."""
    (tmp_path / "gen.png").write_bytes(_FAKE_PNG)
    calls = patched_run(
        {"type": "envelope", "ok": True, "data": {"downloaded": ["gen.png"]}}
    )

    server.fetch_outputs("pid", str(tmp_path), inline_images=True)

    # Same passthrough argv as the plain path — no --url-only, still copies bytes.
    assert calls[0]["cmd"][4:] == ["download", "pid", "-o", str(tmp_path)]


def test_fetch_outputs_default_return_is_unchanged(patched_run, tmp_path):
    """Without inline_images the bare envelope data is returned (no list wrapping)."""
    (tmp_path / "gen.png").write_bytes(_FAKE_PNG)
    patched_run(envelope(data={"downloaded": ["gen.png"]}))

    result = server.fetch_outputs("pid", str(tmp_path))

    assert result == {"downloaded": ["gen.png"]}  # dict, not a [data, ...] list


# --- envelope-terminated read: don't wait on a lingering child ----------------


class _LingeringProc:
    """Fake Popen that emits ``lines`` (incl. the terminal envelope) then lingers.

    Once the canned lines are exhausted ``stdout.readline`` blocks until the
    child is killed, and ``wait()`` blocks the same way — modeling comfy-cli
    outliving its own ``--json-stream`` envelope under a pipe. ``stderr.read``
    blocks too, so a test also proves the envelope path never awaits stderr. The
    block is bounded (10s) only so a buggy test can't hang the suite; the real
    exit is ``kill()``.
    """

    def __init__(self, cmd, lines):
        self.cmd = cmd
        self._lines = list(lines)
        self.stdout = self  # readline lives on the proc itself
        self.stderr = self  # read() blocks the same way -> proves we don't await it
        # None until reaped, mirroring real Popen: while the child lingers past
        # its envelope its returncode is unknown, and _unwrap_envelope must
        # tolerate that (it ignores returncode whenever an envelope is present).
        self.returncode = None
        self.killed = False
        self._dead = threading.Event()

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._dead.wait(timeout=10.0)  # stdout never EOFs on its own
        return ""

    def read(self, size=-1):  # noqa: ARG002 — size ignored; models a blocking pipe
        self._dead.wait(timeout=10.0)  # stderr never EOFs while the child lives
        return ""

    def poll(self):
        return self.returncode if self._dead.is_set() else None

    def wait(self, timeout=None):  # noqa: ARG002
        self._dead.wait(timeout=10.0)  # a child that lingers past its envelope
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9  # reaped by SIGKILL
        self._dead.set()


def _lingering_popen(procs, stream_lines):
    """A ``subprocess.Popen`` stand-in minting a fresh ``_LingeringProc`` per call."""

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _LingeringProc(cmd, list(stream_lines))
        procs.append(proc)
        return proc

    return fake_popen


# Run events (2-node manifest + one completion) then the terminal envelope; the
# child then lingers instead of closing stdout.
_ENVELOPE_THEN_LINGER = [
    json.dumps({"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]})
    + "\n",
    json.dumps({"type": "executed", "node": "1", "title": "Load"}) + "\n",
    json.dumps(
        {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": True,
            "data": {"outputs": ["/x.png"]},
        }
    )
    + "\n",
]

_ERROR_ENVELOPE_THEN_LINGER = [
    json.dumps({"type": "queued", "nodes": [{"node_id": "1"}]}) + "\n",
    json.dumps(
        {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": False,
            "error": {"code": "execution_error", "message": "boom"},
        }
    )
    + "\n",
]


def test_run_workflow_returns_on_envelope_despite_lingering_child(monkeypatch):
    """Child emits its envelope then never closes stdout: return promptly, reap it.

    The core fix — the read loop terminates on the ``envelope`` line, so a fast
    run does not sit in ``readline`` waiting for a child that outlives its own
    envelope. The result comes back well under the tool timeout.
    """
    procs: list[_LingeringProc] = []
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess, "Popen", _lingering_popen(procs, _ENVELOPE_THEN_LINGER)
    )
    # Tiny post-envelope grace so the lingering child is reaped fast in-test.
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.1)

    start = time.monotonic()
    result = asyncio.run(
        server.run_workflow(
            "wf.json", wait=True, timeout_seconds=30.0, ctx=_RecordingCtx()
        )
    )
    elapsed = time.monotonic() - start

    assert result == {"outputs": ["/x.png"]}  # the envelope's data, unwrapped
    assert elapsed < 5.0  # did NOT block on the lingering child
    assert procs[0].killed  # the child was killed / reaped on the way out


def test_run_workflow_error_envelope_with_open_pipe_raises(monkeypatch):
    """An error envelope followed by an open pipe still raises with its error code."""
    procs: list[_LingeringProc] = []
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess, "Popen", _lingering_popen(procs, _ERROR_ENVELOPE_THEN_LINGER)
    )
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.1)

    with pytest.raises(server.ComfyCliError, match="execution_error"):
        asyncio.run(server.run_workflow("wf.json", wait=True, timeout_seconds=30.0))

    assert procs[0].killed  # still reaped on the error path


def test_two_overlapping_run_workflow_calls_complete_independently(monkeypatch):
    """Two concurrent wait=True runs against independent children both finish.

    There is no shared state in the wrapper (each call spawns its own child), so
    the reported "second concurrent call hung" is the same envelope-vs-EOF wait —
    with the envelope-terminated read, both return.
    """
    procs: list[_LingeringProc] = []
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess, "Popen", _lingering_popen(procs, _ENVELOPE_THEN_LINGER)
    )
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.1)

    async def _both():
        return await asyncio.gather(
            server.run_workflow("a.json", wait=True, timeout_seconds=30.0),
            server.run_workflow("b.json", wait=True, timeout_seconds=30.0),
        )

    results = asyncio.run(_both())

    assert results == [{"outputs": ["/x.png"]}, {"outputs": ["/x.png"]}]
    assert len(procs) == 2  # two independent children spawned
    assert all(p.killed for p in procs)  # both cleaned up


def test_run_workflow_timeout_error_includes_snapshot_and_hint(monkeypatch):
    """A genuine timeout surfaces the progress snapshot + a job_status/wait=False hint."""
    queued = json.dumps(
        {"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]}
    )
    procs: list[_BlockingProc] = []

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _BlockingProc(cmd, [queued + "\n"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(
            server.run_workflow(
                "wf.json", wait=True, timeout_seconds=0.25, ctx=_RecordingCtx()
            )
        )

    msg = str(excinfo.value)
    assert "timed out" in msg
    assert "nodes_done" in msg  # tracker.snapshot() dict is embedded
    assert "job_status" in msg  # actionable next-step hints
    assert "wait=False" in msg
    assert procs[0].killed  # the child was still cleaned up on timeout


# Non-terminal ``type == "envelope"`` line (relayed custom-node output, schema
# ``event/1``) followed by the REAL terminal ``schema == "envelope/1"`` result.
_SPURIOUS_ENVELOPE_THEN_REAL = [
    json.dumps({"type": "queued", "nodes": [{"node_id": "1"}]}) + "\n",
    json.dumps(
        {
            "schema": "event/1",
            "type": "envelope",
            "ok": True,
            "data": {"spurious": "not the result"},
        }
    )
    + "\n",
    json.dumps({"type": "executed", "node": "1", "title": "Load"}) + "\n",
    json.dumps(
        {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": True,
            "data": {"outputs": ["/real.png"]},
        }
    )
    + "\n",
]


def test_run_workflow_returns_envelope_after_deadline_during_reap(monkeypatch):
    """An envelope that lands before the deadline is returned even if reaping the
    lingering child would overrun it.

    Boundary case for the envelope-vs-deadline race: the authoritative result is
    read within ``timeout``, but the post-envelope reap grace is LONGER than the
    remaining budget. Because the reap runs off the client budget, the result
    comes back instead of being discarded as a spurious timeout.
    """
    procs: list[_LingeringProc] = []
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess, "Popen", _lingering_popen(procs, _ENVELOPE_THEN_LINGER)
    )
    # Grace (0.5s) deliberately exceeds the client timeout (0.2s): the OLD code
    # ran the reap inside the client budget and raised; the fix returns the data.
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.5)

    start = time.monotonic()
    result = asyncio.run(
        server.run_workflow(
            "wf.json", wait=True, timeout_seconds=0.2, ctx=_RecordingCtx()
        )
    )
    elapsed = time.monotonic() - start

    assert result == {"outputs": ["/x.png"]}  # envelope data, not a timeout error
    assert elapsed < 5.0  # reaped within the grace, not the 10s mock block
    assert procs[0].killed  # lingering child still cleaned up


def test_run_workflow_ignores_non_terminal_envelope_typed_line(monkeypatch):
    """A relayed ``type == "envelope"`` line that isn't ``schema == "envelope/1"``
    must not abort the read and return its payload as the result."""
    procs: list[_LingeringProc] = []
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        _lingering_popen(procs, _SPURIOUS_ENVELOPE_THEN_REAL),
    )
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.1)

    result = asyncio.run(
        server.run_workflow(
            "wf.json", wait=True, timeout_seconds=30.0, ctx=_RecordingCtx()
        )
    )

    assert result == {
        "outputs": ["/real.png"]
    }  # the terminal envelope, not the spurious one
    assert procs[0].killed


class _ExitedErrorProc:
    """Emits an error envelope with an empty message, then exits with stderr text.

    Models an error run whose envelope carries no ``error.message`` but whose
    stderr does — the envelope path must still surface that stderr text.
    """

    def __init__(self, cmd, lines, stderr_text):
        self.cmd = cmd
        self._lines = list(lines)
        self.stdout = self
        self.stderr = io.StringIO(stderr_text)
        self.returncode = 1  # already exited by the time we reap
        self.killed = False

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode

    def kill(self):
        self.killed = True


def test_run_workflow_error_envelope_empty_message_falls_back_to_stderr(monkeypatch):
    """A message-less error envelope on the streaming path still reports stderr."""
    lines = [
        json.dumps({"type": "queued", "nodes": [{"node_id": "1"}]}) + "\n",
        json.dumps(
            {
                "schema": "envelope/1",
                "type": "envelope",
                "ok": False,
                "error": {"code": "execution_error", "message": ""},
            }
        )
        + "\n",
    ]
    stderr_text = "Traceback (most recent call last):\nRuntimeError: kaboom"

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        return _ExitedErrorProc(cmd, lines, stderr_text)

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.1)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server.run_workflow("wf.json", wait=True, timeout_seconds=30.0))

    msg = str(excinfo.value)
    assert "execution_error" in msg  # the envelope's error code
    assert "kaboom" in msg  # the stderr fallback filled the empty message


# --- stderr drain is bounded in both memory and time --------------------------


class _ChunkedStream:
    """Fake pipe that hands back ``data`` in fixed ``chunk`` slices, then EOF.

    Models a real pipe delivering stderr in many small reads (ignoring the
    requested size), so a test can prove `_drain_capped` keeps reading to EOF
    and retains the tail across chunk boundaries — not just within one read.
    """

    def __init__(self, data, chunk):
        self._data = data
        self._chunk = chunk
        self._pos = 0

    def read(self, size=-1):  # noqa: ARG002 — models a pipe: own chunk, not size
        piece = self._data[self._pos : self._pos + self._chunk]
        self._pos += len(piece)
        return piece


def test_drain_capped_retains_only_the_tail_across_chunks():
    """`_drain_capped` drains to EOF but keeps at most ``limit`` trailing chars.

    A verbose child must be fully drained (so it can't wedge on a full stderr
    pipe) without letting its output drive unbounded allocation here — only the
    tail, where the actual error/traceback lands, is retained.
    """
    data = "".join(f"line{i}\n" for i in range(1000))  # ~7 KB across many chunks
    out = server._drain_capped(_ChunkedStream(data, chunk=64), limit=100)
    assert out == data[-100:]
    assert len(out) == 100
    # Under the cap: returned whole. Empty: drains to "".
    assert server._drain_capped(_ChunkedStream("short", chunk=64), 100) == "short"
    assert server._drain_capped(_ChunkedStream("", chunk=64), 100) == ""


class _StderrNeverEOFProc:
    """Envelope-then-linger child whose stderr pipe never EOFs, even after kill.

    Models comfy-cli leaving a descendant that inherited the stderr write fd:
    ``kill()`` reaps the direct child (unblocking ``readline`` / ``wait``) but
    the stderr ``read()`` never returns. The bounded ``finally`` join must
    detach the parked reader instead of hanging the tool call forever.
    """

    def __init__(self, cmd, lines):
        self.cmd = cmd
        self._lines = list(lines)
        self.stdout = self
        self.stderr = self
        self.returncode = None
        self.killed = False
        self._child_dead = threading.Event()  # set by kill(): the direct child
        self._stderr_eof = threading.Event()  # NEVER set: descendant holds the fd

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._child_dead.wait(timeout=10.0)
        return ""

    def read(self, size=-1):  # noqa: ARG002 — parks past kill(); never EOFs
        self._stderr_eof.wait(timeout=10.0)
        return ""

    def poll(self):
        return self.returncode if self._child_dead.is_set() else None

    def wait(self, timeout=None):  # noqa: ARG002
        self._child_dead.wait(timeout=10.0)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
        self._child_dead.set()  # reaps the direct child, NOT the stderr holder


def test_envelope_path_does_not_hang_when_stderr_pipe_never_eofs(monkeypatch):
    """A descendant holding the stderr write fd can't wedge cleanup forever.

    ``proc.kill()`` reaps only the direct child; if a descendant keeps the
    stderr pipe open the reader never EOFs. The ``finally`` join is bounded, so
    the tool call still returns its already-read envelope instead of hanging.
    """
    lines = [
        json.dumps({"type": "queued", "nodes": [{"node_id": "1"}]}) + "\n",
        json.dumps(
            {
                "schema": "envelope/1",
                "type": "envelope",
                "ok": True,
                "data": {"outputs": ["/out.png"]},
            }
        )
        + "\n",
    ]
    proc = _StderrNeverEOFProc(cmd=["comfy"], lines=lines)

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.05)
    monkeypatch.setattr(server, "_STDERR_JOIN_GRACE", 0.05)

    async def _drive():
        # A regression (unbounded join) would blow this guard, not park for 30s.
        return await asyncio.wait_for(
            server.run_workflow("wf.json", wait=True, timeout_seconds=30.0),
            timeout=5.0,
        )

    result = asyncio.run(_drive())
    assert result == {"outputs": ["/out.png"]}
    assert proc.killed  # the direct child was reaped even though stderr never EOF'd


# --- bounded pipe-read pool: threads don't accumulate on the default executor -


def test_pipe_executor_is_dedicated_and_bounded():
    """The subprocess pipe reads run on their own bounded pool, not the default.

    A dedicated `ThreadPoolExecutor` with a finite `max_workers` is what keeps
    parked pipe-reader/waiter threads off asyncio's shared default executor, so
    they can never starve unrelated `to_thread` work.
    """
    from concurrent.futures import ThreadPoolExecutor

    assert isinstance(server._PIPE_EXECUTOR, ThreadPoolExecutor)
    assert server._PIPE_EXECUTOR._max_workers == server._PIPE_POOL_MAX_WORKERS
    assert server._PIPE_POOL_MAX_WORKERS >= 1  # bounded, non-zero


def test_overlapping_streaming_runs_confine_and_release_pipe_threads(monkeypatch):
    """N overlapping streaming runs keep their blocking pipe reads on the
    dedicated pool and drain back to baseline once each run returns.

    Each fake child emits its envelope then lingers with a blocked stderr pipe
    (``read()`` never EOFs until the child is killed) — the exact shape that
    parks a reader thread. This asserts the two halves of the fix together:

    1. Isolation — every blocking ``readline`` / ``read`` / ``wait`` runs on a
       ``_PIPE_EXECUTOR`` worker (``comfy-pipe`` prefix), NOT the loop's shared
       default executor (whose threads are named ``asyncio_*``). On the old
       ``asyncio.to_thread`` code these would land on the default pool and the
       name assertion fails.
    2. Baseline — after every run returns, no reader/waiter thread is still
       parked: the ``finally`` joins the stderr reader once the child is dead
       instead of leaving it parked on a cancelled future.
    """
    from concurrent.futures import ThreadPoolExecutor

    n_runs = 6
    parked = 0  # threads currently blocked in a fake pipe read / wait
    peak_parked = 0
    seen_thread_names: set[str] = set()
    lock = threading.Lock()

    # A generously sized dedicated pool so the test never deadlocks on a
    # low-core host (a persistent stderr reader holds a slot for the whole run);
    # the `comfy-pipe` prefix mirrors the real `_PIPE_EXECUTOR`.
    test_pool = ThreadPoolExecutor(
        max_workers=4 * n_runs, thread_name_prefix="comfy-pipe"
    )
    monkeypatch.setattr(server, "_PIPE_EXECUTOR", test_pool)

    class _InstrumentedProc:
        def __init__(self, cmd, lines):
            self.cmd = cmd
            self._lines = list(lines)
            self.stdout = self
            self.stderr = self
            self.returncode = None
            self.killed = False
            self._dead = threading.Event()

        def _record_thread(self):
            with lock:
                seen_thread_names.add(threading.current_thread().name)

        def _block_until_dead(self):
            nonlocal parked, peak_parked
            self._record_thread()
            with lock:
                parked += 1
                peak_parked = max(peak_parked, parked)
            try:
                self._dead.wait(timeout=10.0)  # real exit is kill(); bound as a guard
            finally:
                with lock:
                    parked -= 1

        def readline(self):
            self._record_thread()
            if self._lines:
                return self._lines.pop(0)
            self._block_until_dead()
            return ""

        def read(self, size=-1):  # noqa: ARG002 — size ignored; parks until killed
            self._block_until_dead()
            return ""

        def poll(self):
            return self.returncode if self._dead.is_set() else None

        def wait(self, timeout=None):  # noqa: ARG002
            self._block_until_dead()
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9
            self._dead.set()

    procs: list[_InstrumentedProc] = []

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _InstrumentedProc(cmd, list(_ENVELOPE_THEN_LINGER))
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.2)

    async def _run_many():
        return await asyncio.gather(
            *[
                server.run_workflow(f"wf{i}.json", wait=True, timeout_seconds=30.0)
                for i in range(n_runs)
            ]
        )

    try:
        results = asyncio.run(_run_many())

        assert results == [{"outputs": ["/x.png"]}] * n_runs
        assert len(procs) == n_runs
        assert all(p.killed for p in procs)  # every child reaped

        # The runs genuinely overlapped: at least the N stderr readers were
        # parked at once (proves this isn't trivially serialized).
        assert peak_parked >= n_runs

        # (1) Isolation: everything ran on the dedicated pool, never the default.
        assert seen_thread_names, "expected pipe reads to run on worker threads"
        assert all(name.startswith("comfy-pipe") for name in seen_thread_names), (
            seen_thread_names
        )

        # (2) Baseline: no pipe thread stays parked after the runs return. The
        # only laggard is each timed-out reap `wait`, released by kill(); poll
        # briefly so a sub-millisecond unwind race isn't read as a leak.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                if parked == 0:
                    break
            time.sleep(0.01)
        with lock:
            assert parked == 0, f"{parked} pipe thread(s) still parked at baseline"
    finally:
        test_pool.shutdown(wait=True)


def test_get_logs_caps_oversized_line(patched_run):
    """A single pathological (megabyte) log line is truncated; normal lines pass through."""
    blob = "x" * (server._MAX_LOG_LINE_CHARS + 5_000)
    payload = {
        "lines": ["startup ok\n", blob, "loaded checkpoint\n"],
        "path": "/ws/user/comfyui_8188.log",
        "truncated": False,
    }
    patched_run(envelope(data=payload))

    result = server.get_logs()
    lines = result["lines"]

    assert lines[0] == "startup ok\n"  # short line untouched
    assert lines[2] == "loaded checkpoint\n"
    # TOTAL length (content + marker) never exceeds the hard cap.
    assert len(lines[1]) <= server._MAX_LOG_LINE_CHARS
    assert lines[1].endswith(server._TRACEBACK_TRUNCATION_MARKER)


def test_validate_workflow_rejects_option_like_path(monkeypatch):
    """A leading-dash `--workflow` value is refused before any child spawns."""

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    with pytest.raises(server.ComfyCliError, match=r"leading '-'.*\./"):
        server.validate_workflow("--help")


@pytest.mark.parametrize("wait", [True, False])
def test_run_workflow_rejects_option_like_path_on_both_paths(monkeypatch, wait):
    """The guard sits at function entry, so it covers submit AND streaming.

    `wait=False` goes through `_run_comfy` and `wait=True` through
    `_run_comfy_streaming`; guarding once up front covers both, and fails before
    the credential retry loop can re-raise it.
    """

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)
    monkeypatch.setattr(server, "_run_comfy_streaming", boom)

    with pytest.raises(server.ComfyCliError, match=r"leading '-'.*\./"):
        asyncio.run(server.run_workflow("--help", wait=wait))


def test_validate_workflow_argv_unchanged_by_the_guard(patched_run):
    """Happy-path argv is untouched: `validate --workflow <path>`."""
    calls = patched_run(envelope(data={"valid": True}))

    server.validate_workflow("./-wf.json")

    assert calls[0]["cmd"][4:] == ["validate", "--workflow", "./-wf.json"]


def test_run_and_validate_workflow_reject_embedded_nul(monkeypatch):
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)
    monkeypatch.setattr(server, "_run_comfy_streaming", boom)

    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        server.validate_workflow("/tmp/w\0f.json")

    for wait in (True, False):
        with pytest.raises(server.ComfyCliError, match="embedded NUL"):
            asyncio.run(server.run_workflow("/tmp/w\0f.json", wait=wait))
