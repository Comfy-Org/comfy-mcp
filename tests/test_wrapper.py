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
import enum
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import PurePosixPath

import pytest
from conftest import _OK_STREAM, _FakeProc, _RecordingCtx, envelope, stream_reader

from comfy_mcp import failure_log, server, tcc, textutil


def _download_model(*args, **kwargs):
    """Drive the async ``download_model`` from these synchronous tests.

    ``download_model`` went async when it started submitting to a background
    worker and polling for it (a blocking poll loop on the event loop would
    stall every other concurrent MCP request). Its contract is otherwise
    unchanged, and an inline ``asyncio.run`` at each of the ~30 call sites below
    — several of them multi-line — would bury what each test asserts.
    """
    return asyncio.run(server.download_model(*args, **kwargs))


def _launch(*args, **kwargs):
    """Drive the async ``launch_comfyui`` from these synchronous tests.

    Both lifecycle tools went async when ``extra_args`` gained the
    network-exposure consent gate (see ``test_network_exposure.py``): an
    elicitation can only be awaited. Nothing else about them changed — the
    spawn itself still runs off the event loop, on a worker thread.
    """
    return asyncio.run(server.launch_comfyui(*args, **kwargs))


def _restart(*args, **kwargs):
    """Drive the async ``restart_comfyui`` from these synchronous tests."""
    return asyncio.run(server.restart_comfyui(*args, **kwargs))


# What Click prints when a comfy-cli that predates the background download is
# handed `--background`: `NoSuchOption`, i.e. a `UsageError` (exit 2) raised
# while PARSING, so no envelope is ever emitted. The message body is Click's own
# `format_message()` verbatim; the borders, colour, and the mid-phrase wrap are
# the rich panel Typer renders it inside — every part `_normalize_cli_text`
# exists to fold away. Wrapped the way `_unwrap_envelope` wraps it.
_NO_SUCH_OPTION_STDERR = (
    "\x1b[31m╭─\x1b[0m Error \x1b[31m─╮\x1b[0m\n"
    "│ No such option    │\n"
    "│ '--background'.   │\n"
    "╰───────────────────╯"
)


def _missing_background_error() -> server.ComfyCliError:
    """The `ComfyCliError` an old comfy-cli's `--background` rejection produces."""
    return server.ComfyCliError(
        f"comfy-cli returned no JSON (exit 2). stderr: {_NO_SUCH_OPTION_STDERR} "
        "| stdout: <empty>",
        no_envelope=True,
        returncode=2,
    )


def _reject_background(*args, **kwargs):
    """Stand in for ``_run_comfy`` on a CLI too old to know ``--background``.

    The whole-of-``_run_comfy`` counterpart to :func:`legacy_comfy_cli`, for the
    tests that only care about what the FALLBACK is handed and stub
    ``_run_comfy_async`` themselves — there is no second call to pass through, so
    a per-argv dispatch would be ceremony.
    """
    raise _missing_background_error()


@pytest.fixture
def legacy_comfy_cli(monkeypatch):
    """Simulate a comfy-cli too old to know ``model download --background``.

    Answers only the ``--background`` submit with the Click usage error such a
    CLI produces and passes every other call through to the REAL ``_run_comfy``.
    The fallback itself runs on ``_run_comfy_async``, which this leaves entirely
    alone, so it still exercises the genuine plain-exit machinery underneath
    (``patched_async_run`` + ``_synthesize_plain_result``) rather than a stub of
    it. This is a per-ARGV reply rather than a second spawn shape, which is why
    it lives here instead of in ``conftest``.
    """
    real = server._run_comfy

    def dispatch(*args, **kwargs):
        if "--background" in args:
            raise _missing_background_error()
        return real(*args, **kwargs)

    monkeypatch.setattr(server, "_run_comfy", dispatch)


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


def test_run_comfy_self_attributes_via_user_agent_env(patched_run):
    """Every spawn labels itself `comfy-mcp` through comfy-cli's caller hook.

    The attribution the engine needs to tell usage that originated in THIS
    server apart from a human running the same subcommand — partner-node calls
    above all, since those are the ones that cost money.
    """
    calls = patched_run(envelope(data={"x": 1}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["env"]["COMFY_USER_AGENT"] == "comfy-mcp"


def test_comfy_env_user_agent_wins_over_an_inherited_value(monkeypatch):
    """An inherited `COMFY_USER_AGENT` is overwritten, not deferred to.

    `_comfy_env` forwards `os.environ` wholesale, so a value in the user's shell
    or the MCP client's `env` block reaches the child — and comfy-cli reads
    `COMFY_USER_AGENT` ahead of every other caller signal. Deferring to it would
    file calls this server made under someone else's label, which is exactly the
    question the label exists to answer.
    """
    monkeypatch.setenv("COMFY_USER_AGENT", "some-other-harness")

    assert server._comfy_env()["COMFY_USER_AGENT"] == "comfy-mcp"


def test_mcp_user_agent_matches_the_distribution_name():
    """The label is the distribution name, bare — it is matched on exactly.

    A tripwire for the rename half of the sync note on `_MCP_USER_AGENT`: the
    value is consumed downstream as an identifier, so a package rename that left
    this behind would silently strand every attributed call under a name nothing
    matches (and a version suffix would defeat the match outright).
    """
    pyproject = (
        pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    ).read_text(encoding="utf-8")
    name = re.search(r'(?m)^name\s*=\s*"([^"]+)"', pyproject)

    assert name is not None, "pyproject.toml has no [project] name"
    assert server._MCP_USER_AGENT == name.group(1)


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


def test_missing_binary_install_advice_names_the_version_floor(monkeypatch):
    """The not-found message's `pip install` must already satisfy the floor.

    A bare ``pip install comfy-cli`` can land a release below
    ``_MIN_COMFY_CLI``, which drops the user straight into
    ``_check_comfy_version``'s separate "too old" error — so the first install
    advice a fresh machine gets pins the floor.

    It pins it with DOUBLE quotes, which is the only form that survives a
    copy-paste into every shell an MCP client might be launched from: `>` needs
    quoting everywhere, but cmd.exe does not quote with `'`.
    """
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    monkeypatch.setattr(tcc, "_is_macos", lambda: False)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env")

    message = str(excinfo.value)
    assert f'pip install "comfy-cli>={server._MIN_COMFY_CLI_STR}"' in message
    # The bare form is what left users one call away from the "too old" error.
    assert "pip install comfy-cli`" not in message
    # The single-quoted form leaves cmd.exe users with a stray `=1.13.0'` file.
    assert "'comfy-cli" not in message


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


def test_error_envelope_carries_its_data_payload(patched_run):
    """A negative verdict that IS a structured report reaches the caller intact.

    ``comfy validate`` emits its full report as the envelope's ``data`` and sets
    ``ok`` to the verdict, so "this workflow does not fit the install" arrives as
    an error whose payload is the actual answer. Dropping it would leave a caller
    unable to tell a real verdict from a check that never ran — which is exactly
    what the template ``local_check`` branches on.
    """
    report = {"valid": False, "errors": [{"node_id": "3", "message": "nope"}]}
    patched_run({"type": "envelope", "ok": False, "data": report})

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("validate", "--workflow", "wf.json")

    assert excinfo.value.data == report
    # Failures with nothing structured to carry keep `data` at None.
    patched_run({"type": "envelope", "ok": False, "error": {"code": "boom"}})
    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env")
    assert excinfo.value.data is None


# --- comfy-cli version guard -------------------------------------------------


def _fake_version(stdout: str, *, stderr: str = "", raises: Exception | None = None):
    """A `subprocess.run` stand-in for `comfy --version`; records each call."""
    calls: list[list[str]] = []

    def fake(cmd, capture_output, text, timeout, check, errors=None):
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

    with pytest.raises(server.ComfyCliError, match=r"too old.*1\.13\.0"):
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
    fake, calls = _fake_version("comfy-cli, version 1.13.0")
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()
    assert server._version_checked is True

    server._check_comfy_version()  # memoized: no second `comfy --version`
    assert len(calls) == 1


def test_version_guard_floor_is_exactly_the_login_url_release(monkeypatch):
    """Pin the boundary: 1.12.0 is rejected, 1.13.0 is accepted.

    1.13.0 is the first published comfy-cli carrying the machine-readable
    `login_url` event `auth_login` blocks on (1.12.0 has no `login_url` anywhere
    in the wheel). Without this floor, a 1.12.0 install sails past the guard and
    only learns it cannot log in after `auth_login` burns its whole
    `_LOGIN_URL_WAIT_S` budget spawning and reaping a child process. Pinning
    both sides of the boundary is what stops the constant drifting back to the
    version that gives that slow, late "no".
    """
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    # The immediately-prior release: rejected, up front, with the upgrade line.
    monkeypatch.setattr(server, "_version_checked", False)
    prior, _ = _fake_version("comfy-cli, version 1.12.0")
    monkeypatch.setattr(server.subprocess, "run", prior)
    with pytest.raises(server.ComfyCliError) as excinfo:
        server._check_comfy_version()
    assert "1.12.0 is too old" in str(excinfo.value)
    assert "comfy-cli>=1.13.0" in str(excinfo.value)

    # The floor itself: accepted.
    monkeypatch.setattr(server, "_version_checked", False)
    floor, _ = _fake_version("comfy-cli, version 1.13.0")
    monkeypatch.setattr(server.subprocess, "run", floor)
    server._check_comfy_version()  # no raise
    assert server._version_checked is True

    assert server._MIN_COMFY_CLI == (1, 13, 0)
    assert server._MIN_COMFY_CLI_STR == "1.13.0"


def test_auth_login_is_refused_up_front_on_a_cli_without_login_url(monkeypatch):
    """`auth_login` below the floor fails at the guard, not after the 15s wait.

    The whole point of raising the floor: on a comfy-cli with no `login_url`
    event the old behavior was to spawn `comfy cloud login`, wait the full
    `_LOGIN_URL_WAIT_S` budget, reap the child, and only then say "upgrade
    comfy-cli". The guard now answers on the first tool call — so no child is
    ever spawned.
    """
    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_login_child", None)  # no pending flow to resume
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    fake, _ = _fake_version("comfy-cli, version 1.12.0")
    monkeypatch.setattr(server.subprocess, "run", fake)

    spawned: list = []

    async def never(*args, **kwargs):  # pragma: no cover - must not be reached
        spawned.append(args)
        raise AssertionError("auth_login spawned a child below the version floor")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", never)

    with pytest.raises(server.ComfyCliError, match=r"too old.*1\.13\.0"):
        asyncio.run(server.auth_login())

    assert spawned == []  # refused before any `comfy cloud login` spawn


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
            stdout="comfy-cli, version 1.13.0\n",
            stderr="",
        )

    monkeypatch.setattr(server, "_spawn_comfy_version", fake_spawn)

    server._check_comfy_version()  # no raise: at the floor
    assert server._detect_comfy_cli_version() == "1.13.0"
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


def test_get_logs_forwards_port_hint(patched_run):
    """An explicit `port` is appended as `--port <n>`, after the tail."""
    calls = patched_run(envelope(data={"lines": []}))

    server.get_logs(port=8189)

    assert calls[0]["cmd"][4:] == ["logs", "--tail", "200", "--port", "8189"]


def test_get_logs_without_port_leaves_argv_unchanged(patched_run):
    """`port=None` (the default) is byte-identical to the pre-hint argv."""
    calls = patched_run(envelope(data={"lines": []}))

    server.get_logs(tail=50)

    assert calls[0]["cmd"][4:] == ["logs", "--tail", "50"]
    assert "--port" not in calls[0]["cmd"]


@pytest.mark.parametrize(
    "bad_port",
    [
        0,  # below the IANA range
        -1,  # would render as `--port -1`, which Click reads as an option
        70000,  # above the IANA range
        True,  # `bool` is an `int` subclass — must not forward `--port 1`
        8189.5,  # a non-integer would be silently truncated by `int()`
        "8189",  # a string never becomes argv unchecked
    ],
)
def test_get_logs_rejects_invalid_port(no_spawn, bad_port):
    """An out-of-range or non-integer port is refused before comfy-cli is spawned."""
    with pytest.raises(server.ComfyCliError, match="port"):
        server.get_logs(port=bad_port)


def test_get_logs_normalizes_an_int_subclass_port(patched_run):
    """An `IntEnum` port reaches argv as its NUMBER, not as `Port.COMFY`.

    It passes the `isinstance` and range checks, so the guard's job is to hand
    back `int(port)` — otherwise `str()` renders the member name onto argv.
    """

    class _Port(enum.IntEnum):
        COMFY = 8189

    calls = patched_run(envelope(data={"lines": []}))

    server.get_logs(port=_Port.COMFY)

    assert calls[0]["cmd"][4:] == ["logs", "--tail", "200", "--port", "8189"]


def test_get_logs_bounds_the_rejected_port_it_echoes(no_spawn):
    """An oversized value is summarized, not copied verbatim into the error.

    Same rule as `_guard_prompt_id` / `_guard_download_id`: an in-process caller
    can pass a megabyte-long "port", and reflecting it whole floods the caller's
    context with the very input the guard refused.
    """
    huge = "8" * 50_000

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.get_logs(port=huge)

    message = str(excinfo.value)
    assert huge not in message
    assert f"({len(repr(huge))} characters)" in message
    # A summary of the input, not the input: a few lines of prose around the cap,
    # nowhere near the 50k it was handed.
    assert len(message) < server._MAX_PORT_REPR_CHARS + 200


def test_get_logs_rejects_an_unrenderable_int_port_as_a_range_error(no_spawn):
    """An int too large to stringify still fails as a `ComfyCliError`.

    On 3.11+ an int with more than `sys.get_int_max_str_digits()` digits raises
    `ValueError` on conversion to text, so formatting the value into the
    out-of-range message would escape this guard as an internal error.
    """
    with pytest.raises(server.ComfyCliError, match="outside the valid range"):
        server.get_logs(port=10**5000)


def test_get_logs_passes_through_source_and_staleness_metadata(patched_run):
    """`source` / `mtime` / `size` / `port_mismatch` reach the caller untouched.

    A wrong-port empty log is otherwise indistinguishable from success, so the
    staleness signal must not be dropped — while the per-line cap still applies.
    This is the auto-resolved shape (no `port` argument), the only one comfy-cli
    ever sets `port_mismatch` on.
    """
    blob = "x" * (server._MAX_LOG_LINE_CHARS + 5_000)
    payload = {
        "lines": ["boot ok\n", blob],
        "path": "/ws/user/comfyui_8188.log",
        "truncated": True,
        "source": "fallback_glob",
        "port_mismatch": True,
        "mtime": "2026-07-20T11:02:03+00:00",
        "size": 4096,
        # A field this MCP has never heard of: the payload is forwarded, not
        # projected, so a newer comfy-cli's additions must survive too.
        "rotated_from": "/ws/user/comfyui_8188.log.1",
    }
    patched_run(envelope(data=payload))

    result = server.get_logs()

    assert result["source"] == "fallback_glob"
    assert result["port_mismatch"] is True
    assert result["mtime"] == "2026-07-20T11:02:03+00:00"
    assert result["size"] == 4096
    assert result["path"] == "/ws/user/comfyui_8188.log"
    assert result["truncated"] is True
    assert result["rotated_from"] == "/ws/user/comfyui_8188.log.1"
    assert result["lines"][0] == "boot ok\n"  # capping is unchanged
    assert len(result["lines"][1]) <= server._MAX_LOG_LINE_CHARS


def test_get_logs_explicit_port_fallback_source_passes_through(patched_run):
    """With an explicit `port`, `source` — not `port_mismatch` — is the signal.

    comfy-cli suppresses `port_mismatch` when a port was asked for, so a
    `--port` that fell back to ComfyUI-Manager's unsuffixed log (which records
    no port, and may be another session's) shows up only as `source`.
    """
    payload = {
        "lines": ["boot ok\n"],
        "path": "/ws/user/comfyui.log",
        "truncated": False,
        "source": "fallback_unsuffixed",
        "port_mismatch": False,
        "mtime": "2026-07-20T11:02:03+00:00",
        "size": 128,
    }
    patched_run(envelope(data=payload))

    result = server.get_logs(port=8189)

    assert result["source"] == "fallback_unsuffixed"
    assert result["port_mismatch"] is False


def test_get_logs_no_log_file_message_is_preserved_verbatim(patched_run):
    """The multi-candidate `no_log_file` message still returns as data, intact."""
    message = (
        "No captured ComfyUI log was found. Looked for: "
        "/ws/user/comfyui_8189.log, /ws/user/comfyui.log"
    )
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "no_log_file", "message": message},
        }
    )

    result = server.get_logs(port=8189)

    assert result["error"] == "no_log_file"
    assert message in result["message"]


def test_get_logs_port_on_old_comfy_cli_raises_upgrade_error(patched_run):
    """An older comfy-cli that has no `--port` fails loudly — it never retries.

    A silent retry without the flag would return whichever log comfy-cli resolves
    on its own, i.e. the wrong-instance answer the hint exists to prevent.
    """
    calls = patched_run("", returncode=2, stderr="Error: No such option '--port'.\n")

    with pytest.raises(server.ComfyCliError, match="upgrade") as excinfo:
        server.get_logs(port=8189)

    assert "--port" in str(excinfo.value)
    # comfy-cli's own text survives: `raise ... from exc` sets `__cause__`, which
    # no MCP client sees, so a rewrite would be the only thing the caller reads —
    # and if this match were ever wrong, the real diagnostic would be gone.
    assert "No such option" in str(excinfo.value)
    assert len(calls) == 1  # exactly one spawn: no fallback attempt


def test_get_logs_unrelated_usage_error_keeps_its_own_message(patched_run):
    """The upgrade hint is scoped to `--port`; another usage error passes through."""
    patched_run("", returncode=2, stderr="Error: No such option '--tail'.\n")

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.get_logs(port=8189)

    assert "upgrade" not in str(excinfo.value)


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


# --- system_stats / free_memory (the ComfyUI resource-management passthrough) ---


def test_system_stats_maps_command_and_returns_data(patched_run):
    """system_stats wraps `comfy system-stats` and returns the envelope data as-is."""
    stats = {
        "devices": [
            {
                "name": "cuda:0",
                "type": "cuda",
                "index": 0,
                "vram_free": 11_000_000_000,
                "vram_total": 24_000_000_000,
            }
        ],
        "system": {
            "ram_free": 30_000_000_000,
            "ram_total": 64_000_000_000,
            "comfyui_version": "0.3.0",
        },
    }
    calls = patched_run(envelope(data=stats))

    # Returned UNMODIFIED — no reshaping, no derived fields (thin wrapper rule).
    assert server.system_stats() == stats
    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["system-stats"]  # no positional args, no flags


def test_free_memory_defaults_to_maximum_headroom(patched_run):
    """The MCP default is the strongest form: unload models AND reset the cache.

    `--free-memory` is opt-in on the CLI (default False) and on-by-default here
    (via the `None` = "follow unload_models" sentinel) — an agent calling this
    tool wants all the VRAM back, so the default diverges on purpose. Pinning the
    argv is what keeps that divergence honest.
    """
    calls = patched_run(envelope(data={"requested": {}, "note": "…"}))

    server.free_memory()

    assert calls[0]["cmd"][4:] == ["free", "--unload-models", "--free-memory"]


def test_free_memory_omits_free_memory_flag_when_off(patched_run):
    """`free_memory=False` OMITS the flag — comfy-cli has no `--no-free-memory`."""
    calls = patched_run(envelope(data={"requested": {}}))

    server.free_memory(free_memory=False)

    assert calls[0]["cmd"][4:] == ["free", "--unload-models"]
    assert "--free-memory" not in calls[0]["cmd"]


def test_free_memory_sends_no_unload_models_when_off(patched_run):
    """`unload_models=False` sends the paired OFF flag and drops `--free-memory`.

    The pair must not go out as `--no-unload-models --free-memory`: ComfyUI's
    `POST /free` only records `unload_models` when it is TRUE (server.py's
    `if unload_models: set_flag(...)`), and the queue worker then resolves
    `flags.get("unload_models", free_memory)` — so a `free_memory` left on would
    supply the default and unload every model, the exact opposite of what
    `unload_models=False` asks for. Defaulting `free_memory` to `None` ("follow
    `unload_models`") is what keeps the request honest.
    """
    calls = patched_run(envelope(data={"requested": {}}))

    server.free_memory(unload_models=False)

    assert calls[0]["cmd"][4:] == ["free", "--no-unload-models"]
    assert "--free-memory" not in calls[0]["cmd"]


def test_free_memory_rejects_the_contradictory_pair(patched_run):
    """`unload_models=False, free_memory=True` is refused, not silently obeyed.

    ComfyUI cannot reset the executor cache while keeping models resident, so
    sending this pair would evict them anyway. Failing up front beats an
    acknowledgement that reads as "kept your models" while they are being
    unloaded — and it must not reach comfy-cli at all.
    """
    calls = patched_run(envelope(data={"requested": {}}))

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.free_memory(unload_models=False, free_memory=True)

    assert "free_memory=True with unload_models=False" in str(excinfo.value)
    assert not calls, "the contradictory pair must never be sent to comfy-cli"


def test_free_memory_both_off_is_an_explicit_no_op(patched_run):
    """Both flags off asks ComfyUI to do nothing — sent, but nothing implied.

    The worker skips both branches, so this is a deliberate no-op kept for
    symmetry with the CLI. It stays a passthrough rather than an error because
    comfy-cli's own acknowledgement reports the flags back, letting the caller
    see that nothing was requested.
    """
    calls = patched_run(envelope(data={"requested": {}}))

    server.free_memory(unload_models=False, free_memory=False)

    assert calls[0]["cmd"][4:] == ["free", "--no-unload-models"]


def test_free_memory_returns_the_acknowledgement_payload(patched_run):
    """The return is comfy-cli's request acknowledgement, passed through whole."""
    payload = {
        "requested": {"unload_models": True, "free_memory": True},
        "note": "applies when the queue worker next iterates",
    }
    patched_run(envelope(data=payload))

    assert server.free_memory() == payload


_NO_SYSTEM_STATS = (
    2,
    "",
    "Usage: comfy [OPTIONS] COMMAND\nNo such command 'system-stats'.",
)
_NO_FREE = (2, "", "Usage: comfy [OPTIONS] COMMAND\nNo such command 'free'.")


@pytest.mark.parametrize(
    "tool, reply, verb",
    [
        ("system_stats", _NO_SYSTEM_STATS, "system-stats"),
        ("free_memory", _NO_FREE, "free"),
    ],
)
def test_resource_tools_hint_at_the_upgrade_on_a_comfy_cli_without_the_verb(
    patched_run, tool, reply, verb
):
    """A missing verb raises with the one-command fix, not Click's usage dump.

    These two verbs landed in comfy-cli after the version floor this server
    enforces, so a current-but-not-newest install hits this. Left raw it reads
    as "comfy-cli returned no JSON (exit 2)" wrapped around a usage panel —
    indistinguishable from a broken MCP.
    """
    returncode, stdout, stderr = reply
    patched_run(stdout, returncode=returncode, stderr=stderr)

    with pytest.raises(server.ComfyCliError) as excinfo:
        getattr(server, tool)()

    message = str(excinfo.value)
    assert f"`comfy {verb}`" in message
    assert "pip install -U comfy-cli" in message
    assert server._MIN_COMFY_CLI_STR in message
    # The raw wrapper/CLI text must not leak through the annotated message.
    assert "No such command" not in message
    assert "Usage: comfy" not in message
    assert "returned no JSON" not in message


@pytest.mark.parametrize("tool", ["system_stats", "free_memory"])
def test_resource_tools_keep_a_real_error_raw(patched_run, tool):
    """A verb comfy-cli DID dispatch keeps its own error, unannotated.

    Mislabelling "ComfyUI is not running" as a version problem would send the
    user off to upgrade a comfy-cli that is already fine.
    """
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "ComfyUI not running"},
        }
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        getattr(server, tool)()

    message = str(excinfo.value)
    assert "server_not_running" in message
    assert "pip install -U comfy-cli" not in message


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
    """download_model submits `model download --url … --background`, returns its data."""
    submit = {"download_id": "a1b2c3d4e5f6", "dest": "/models/x.safetensors"}
    calls = patched_run({"type": "envelope", "ok": True, "data": submit})

    assert _download_model("https://hf.co/x.safetensors", wait=False) == submit

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    # SINGULAR `model` verb group (download engine), not the plural catalog.
    assert cmd[4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/x.safetensors",
        "--background",
    ]
    # The submit is metadata-only, so it gets its own budget — never the
    # transfer's 1800s, which is what used to hold the MCP request open.
    assert calls[0]["timeout"] == server._DOWNLOAD_SUBMIT_TIMEOUT


def test_download_model_threads_relative_path(patched_run):
    """--relative-path is appended only when provided."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://hf.co/l.safetensors", relative_path="models/loras", wait=False
    )

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/l.safetensors",
        "--relative-path",
        "models/loras",
        "--background",
    ]


def test_download_model_threads_filename(patched_run):
    """--filename is appended only when provided."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://hf.co/c.safetensors", filename="renamed.safetensors", wait=False
    )

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/c.safetensors",
        "--filename",
        "renamed.safetensors",
        "--background",
    ]


def test_download_model_threads_all_optionals(patched_run):
    """Both optional args thread through together, in order, only when set."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://civitai.com/api/download/models/42",
        relative_path="models/checkpoints",
        filename="sd.safetensors",
        wait=False,
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
        "--background",
    ]


def test_download_model_omits_absent_optionals(patched_run):
    """Neither optional flag is emitted when the argument is left unset."""
    calls = patched_run(envelope(data=_submit()))

    _download_model("https://hf.co/x.safetensors", wait=False)

    cmd = calls[0]["cmd"]
    assert "--relative-path" not in cmd
    assert "--filename" not in cmd


def test_download_model_rejects_option_like_url():
    """A leading-dash url is refused so comfy-cli can't parse it as a flag."""
    with pytest.raises(server.ComfyCliError, match="invalid url"):
        _download_model("--config")


def test_download_model_rejects_option_like_relative_path():
    """A leading-dash relative_path is refused (argument injection guard)."""
    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path="-rf")


def test_download_model_rejects_option_like_filename():
    """A leading-dash filename is refused (argument injection guard)."""
    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        _download_model("https://hf.co/x.safetensors", filename="--evil")


@pytest.mark.parametrize(
    "bad_url", ["file:///etc/passwd", "ftp://host/x", "/etc/passwd"]
)
def test_download_model_rejects_non_http_scheme(bad_url):
    """Only http(s) URLs are allowed; file://, ftp:// and bare paths are refused."""
    with pytest.raises(server.ComfyCliError, match="invalid url"):
        _download_model(bad_url)


@pytest.mark.parametrize(
    "bad_path",
    ["../../etc", "models/../../etc", "/abs/models", "..\\..\\etc", "C:evil"],
)
def test_download_model_rejects_traversal_relative_path(bad_path):
    """relative_path must stay within the models dir: no `..`, absolute paths,
    or a drive prefix (``C:evil`` has no separator but is drive-relative on
    Windows)."""
    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)


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
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

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
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

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
    calls = patched_run(envelope(data=_submit()))

    _download_model("https://hf.co/x.safetensors", relative_path=good_path, wait=False)

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--relative-path") + 1] == good_path


# --- relative_path must land in the MODELS tree, not just inside the workspace -
#
# comfy-cli joins `--relative-path` to the WORKSPACE ROOT (`local_filepath =
# get_workspace() / relative_path / local_filename`, defaulting to
# `DEFAULT_COMFY_MODEL_PATH = "models"`) — which is why the documented shape is
# `models/loras` rather than a bare `loras`. Every value below is traversal-clean,
# so the checks above pass it happily; only the first-segment check keeps the
# write out of the workspace's other top-level directories.


@pytest.mark.parametrize(
    "bad_path",
    [
        "custom_nodes/pwn",  # ComfyUI's import path — see the RCE pin below
        "custom_nodes/x/y",  # deeper, same tree
        "user",  # sibling top-level workspace dirs
        "input",
        "output",
        "web",
        "loras",  # a BARE folder name is rejected, not assumed to mean models/
        "modelsx",  # a prefix of `models` is a different directory
        "models2/loras",
        "Models/loras",  # matched exactly: no case-folding a security guard
        "~",  # comfy-cli expanduser()s this, escaping the workspace entirely
        "~/evil",
    ],
)
def test_download_model_rejects_non_models_relative_path(bad_path, patched_run):
    """A traversal-clean ``relative_path`` that does not start at ``models`` is
    refused before any child spawns."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


def test_download_model_rejects_custom_nodes_init_py_rce(patched_run):
    """Named regression pin for the RCE shape this check exists for.

    ComfyUI imports `custom_nodes/*/__init__.py` on startup, so a write there is
    attacker-controlled code on the import path — code execution on the next
    ComfyUI restart, on every platform. Both arguments are individually legal
    (`custom_nodes/pwn` is traversal-clean; `__init__.py` is a bare filename);
    it is the models-tree confinement that refuses the combination.
    """
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model(
            "https://attacker.example/payload",
            relative_path="custom_nodes/pwn",
            filename="__init__.py",
        )

    assert calls == []


@pytest.mark.parametrize(
    "bad_path",
    ["../../etc", "models/../../etc", "/abs/models", "C:evil", "models/.. /evil"],
)
def test_download_model_traversal_still_reports_as_traversal(bad_path, patched_run):
    """Ordering pin: the models-tree check runs AFTER the traversal checks, so a
    traversal string keeps its own (more specific) diagnosis rather than being
    relabelled as a wrong-folder error."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match=r"invalid relative_path.*traversal"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


@pytest.mark.parametrize(
    "good_path",
    [
        "models",  # the models dir itself
        "models/loras",
        "models/checkpoints",
        "models/loras/",  # trailing slash: an empty trailing segment
        "models//loras",  # doubled slash
    ],
)
def test_download_model_forwards_models_relative_path_unchanged(good_path, patched_run):
    """Accepted values are forwarded to comfy-cli VERBATIM — the guard rejects,
    it never rewrites the argument."""
    calls = patched_run(envelope(data=_submit()))

    _download_model("https://hf.co/x.safetensors", relative_path=good_path, wait=False)

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--relative-path") + 1] == good_path


@pytest.mark.parametrize(
    "bad_path",
    [
        "models\\loras",  # the case below, as a plain parametrization
        "models\\checkpoints\\sdxl",
        "models\\",  # trailing separator, still a `\`
        "models/loras\\ckpt",  # mixed spelling
    ],
)
def test_download_model_rejects_backslash_relative_path(bad_path, patched_run):
    """A `\\` separator is refused even when every segment is otherwise legal.

    The guard splits on `\\` to decide, but comfy-cli receives the value verbatim
    — so on a POSIX host the two disagree and the write misses the models tree
    (see the named pin below). `/` reaches the same directory on Windows, so this
    costs a spelling, not a destination.
    """
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match=r"invalid relative_path.*separator"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


def test_download_model_rejects_backslash_escaping_models_tree(patched_run):
    """Named regression pin for what the `\\` rejection actually prevents.

    `models\\loras` splits to segments `models` + `loras`, so the first-segment
    check passes it — but pathlib on a POSIX host reads the backslash as an
    ordinary character, making `<workspace>/models\\loras` a top-level directory
    SIBLING of the models dir rather than a folder inside it. The write would land
    outside the tree the guard claims to confine it to.
    """
    calls = patched_run(envelope(data=_submit()))

    # Pin the mechanism, so this test fails loudly if pathlib ever changes: the
    # backslash is one literal path component, not a separator, off POSIX.
    assert (PurePosixPath("/ws") / "models\\loras").parts == (
        "/",
        "ws",
        "models\\loras",
    )

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model(
            "https://attacker.example/payload", relative_path="models\\loras"
        )

    assert calls == []


@pytest.mark.parametrize(
    ("bad_path", "expected"),
    [
        # Ordering pin: the `\` check runs LAST, so the security-meaningful
        # diagnoses keep their own (more specific) message.
        ("models\\..\\evil", "traversal"),  # `\`-split makes `..` a real segment
        ("\\evil", "traversal"),  # root-relative: an empty leading segment
        ("custom_nodes\\pwn", "must be the models dir"),  # wrong tree first
        ("loras\\x", "must be the models dir"),
    ],
)
def test_download_model_backslash_check_is_ordered_last(
    bad_path, expected, patched_run
):
    """A `\\` value that is ALSO a traversal or a wrong-tree value reports as that,
    not as a separator-spelling complaint."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match=expected):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


@pytest.mark.parametrize("bad_name", [".. ", "...", ". ", "... ", " "])
def test_download_model_rejects_dot_run_filename(bad_name, patched_run):
    """Same disguise on the bare-name side: `".. "` is a `..`, not a filename."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        _download_model("https://hf.co/x.safetensors", filename=bad_name)

    assert calls == []


@pytest.mark.parametrize("good_name", [".gitkeep", "v1.5.safetensors", "model."])
def test_download_model_accepts_dotted_but_ordinary_filename(good_name, patched_run):
    """Regression pin for the dot-run filename check: a name that merely
    *contains* dots still has something left after `strip(" .")`."""
    calls = patched_run(envelope(data=_submit()))

    _download_model("https://hf.co/x.safetensors", filename=good_name, wait=False)

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
        _download_model("https://hf.co/x.safetensors", filename=bad_name)


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
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        _download_model("https://hf.co/x.safetensors", filename=bad_name)

    assert calls == []


def test_download_model_still_accepts_ordinary_path_and_name(patched_run):
    """Regression pin: the Windows-shaped rejections above must not catch the
    ordinary values these tools are actually called with."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://hf.co/x.safetensors",
        relative_path="models/loras",
        filename="x.safetensors",
        wait=False,
    )

    cmd = calls[0]["cmd"]
    assert "--relative-path" in cmd and cmd[cmd.index("--relative-path") + 1] == (
        "models/loras"
    )
    assert "--filename" in cmd and cmd[cmd.index("--filename") + 1] == "x.safetensors"


def test_download_model_rejects_embedded_nul_url():
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        _download_model("https://hf.co/x\0.safetensors")


def test_download_model_rejects_embedded_nul_relative_path():
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        _download_model("https://hf.co/x.safetensors", relative_path="models/\0")


def test_download_model_rejects_embedded_nul_filename():
    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        _download_model("https://hf.co/x.safetensors", filename="x\0.safetensors")


def test_download_model_omits_empty_string_optionals(patched_run):
    """Explicit empty-string optionals are treated as unset, not forwarded as ``""``."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://hf.co/x.safetensors", relative_path="", filename="", wait=False
    )

    cmd = calls[0]["cmd"]
    assert "--relative-path" not in cmd
    assert "--filename" not in cmd


# --- the background download family: submit, poll, cancel ------------------
#
# `download_model` used to be one blocking `comfy model download` that held the
# MCP request open for the whole multi-GB transfer, so the client's deadline
# fired while the download quietly succeeded — a false failure with no handle to
# verify it by. It now submits to comfy-cli's background worker and polls the
# resulting `download_id`, the shape `run_workflow(wait=False)` + `wait_for_job`
# already use for long generations.


def _submit(download_id: str = "a1b2c3d4e5f6", **extra) -> dict:
    """A `model download --background` submit payload."""
    return {
        "download_id": download_id,
        "dest": "/models/x.safetensors",
        "total_bytes": 4_000_000_000,
        "status": "starting",
        **extra,
    }


def _status(status: str, **extra) -> dict:
    """A `model download-status` payload."""
    return {
        "id": "a1b2c3d4e5f6",
        "status": status,
        "completed_bytes": 1_000_000,
        "total_bytes": 4_000_000_000,
        "dest": "/models/x.safetensors",
        "error": None,
        **extra,
    }


def _sequenced(monkeypatch, replies: list) -> list[tuple]:
    """Answer successive `_run_comfy` calls from `replies`; record their argv.

    The conftest fakes hand back ONE canned reply per fixture, and a submit
    followed by N polls needs a different answer each time — the multi-call case
    AGENTS.md leaves to a local stub. An exhausted iterator fails loudly rather
    than looping forever.
    """
    calls: list[tuple] = []
    pending = iter(replies)

    def fake_run(*args, **kwargs):
        calls.append(args)
        try:
            return next(pending)
        except StopIteration:
            raise AssertionError(f"unexpected extra comfy-cli call: {args}") from None

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    return calls


def test_download_model_submits_then_polls_to_completion(monkeypatch):
    """wait=True submits in the background and polls until the transfer lands."""
    calls = _sequenced(
        monkeypatch,
        [_submit(), _status("downloading"), _status("completed")],
    )

    result = _download_model("https://hf.co/x.safetensors")

    assert result == _status("completed")
    assert calls[0][:4] == ("model", "download", "--url", "https://hf.co/x.safetensors")
    assert calls[0][-1] == "--background"
    # Polls the id the submit handed back, and keeps polling past `downloading`.
    assert calls[1] == ("model", "download-status", "a1b2c3d4e5f6")
    assert len(calls) == 3


def test_download_model_returns_a_timed_out_envelope_rather_than_raising(monkeypatch):
    """A transfer still running at the bound is PROGRESS — the id comes back.

    This inversion is the entire bug: the old blocking call let the client's
    deadline fire on a download that was succeeding, leaving no handle to verify
    it with. A bound that expires must never look like a failure.
    """
    calls = _sequenced(monkeypatch, [_submit(), _status("downloading")])

    result = _download_model("https://hf.co/x.safetensors", timeout_seconds=1e-9)

    assert result == {
        "timed_out": True,
        "download_id": "a1b2c3d4e5f6",
        "status": _status("downloading"),
    }
    # A bound that expires before the first poll still reports a real status.
    assert len(calls) == 2


def test_download_model_tiny_bound_fails_without_starting_a_transfer(monkeypatch):
    """A bound too small to resolve the download must not leave one running.

    The docstring promises this outright, and the promise has real teeth: the
    submit is capped to the caller's bound, so a tiny bound kills it — and that
    timeout must NOT be mistaken for the missing-`--background` degrade, which
    would retry the whole thing as an 1800s blocking transfer nobody asked for.
    The existing end-to-end-budget test pins the cap; this pins what happens
    when the cap actually bites.
    """
    calls: list[float] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise server.ComfyCliError(
            "comfy-cli timed out after 0.001s", timed_out=True, returncode=None
        )

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="timed out"):
        _download_model("https://hf.co/x.safetensors", timeout_seconds=0.001)

    # The submit took the caller's bound, not its own fixed budget...
    assert calls == [0.001]
    # ...and exactly one spawn happened: no legacy fallback, so nothing detached
    # and no second copy of a multi-GB file was fetched.


def test_download_model_raises_on_a_failed_download(monkeypatch):
    """A terminal `failed` carries comfy-cli's own error text into the raise."""
    _sequenced(
        monkeypatch,
        [_submit(), _status("failed", error="HTTP 404: model not found")],
    )

    with pytest.raises(server.ComfyCliError, match="HTTP 404: model not found"):
        _download_model("https://hf.co/missing.safetensors")


def test_download_model_raises_on_a_cancelled_download(monkeypatch):
    """A download cancelled out from under the wait is a failure, not a result."""
    _sequenced(monkeypatch, [_submit(), _status("cancelled")])

    with pytest.raises(server.ComfyCliError, match="cancelled"):
        _download_model("https://hf.co/x.safetensors")


def test_download_model_wait_false_returns_the_submit_payload(monkeypatch):
    """wait=False hands back the submit envelope and polls nothing."""
    calls = _sequenced(monkeypatch, [_submit()])

    assert _download_model("https://hf.co/x.safetensors", wait=False) == _submit()
    assert len(calls) == 1  # no `download-status` call at all


def test_download_model_rejects_a_submit_envelope_with_no_download_id(monkeypatch):
    """No id means nothing to poll — that is a broken contract, not a result.

    Returning the payload anyway would hand back a `status: starting` blob that
    reads like a finished download and leaves the caller no handle at all.
    """
    _sequenced(monkeypatch, [{"dest": "/models/x.safetensors", "status": "starting"}])

    with pytest.raises(server.ComfyCliError, match="no usable `download_id`"):
        _download_model("https://hf.co/x.safetensors")


def test_download_model_wait_false_also_rejects_an_envelope_with_no_id(monkeypatch):
    """The id is validated on the wait=False path too, not just the waiting one.

    Handing back a malformed envelope unchecked would leave a transfer running
    detached behind a payload carrying nothing to poll or cancel it by — the
    same broken contract, just without a poll loop to notice it.
    """
    _sequenced(monkeypatch, [{"dest": "/models/x.safetensors", "status": "starting"}])

    with pytest.raises(server.ComfyCliError, match="no usable `download_id`"):
        _download_model("https://hf.co/x.safetensors", wait=False)


def test_download_model_spends_one_end_to_end_budget(monkeypatch):
    """`timeout_seconds` covers submit AND poll, not each of them separately.

    Two independent budgets add up: a submit that used its full
    `_DOWNLOAD_SUBMIT_TIMEOUT` plus a full-length poll ran ~230s, while the 110s
    default exists precisely to come in under a typical client's ~120s request
    budget. Overshooting it means the client aborts and never receives the
    `download_id` the submit already obtained — the exact failure this tool's
    async shape was built to prevent.
    """
    seen: dict[str, float] = {}
    elapsed = 4.0

    def fake_run(*args, **kwargs):
        seen["submit_timeout"] = kwargs["timeout"]
        # Simulate a submit that spent `elapsed` seconds resolving metadata.
        base = time.monotonic()
        monkeypatch.setattr(server.time, "monotonic", lambda: base + elapsed)
        return _submit()

    def fake_poll(download_id, timeout_seconds):
        seen["poll_bound"] = timeout_seconds
        return _status("completed")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server, "_poll_download", fake_poll)

    _download_model("https://hf.co/x.safetensors", timeout_seconds=30.0)

    # The submit is capped to the caller's bound as well as its own...
    assert seen["submit_timeout"] == 30.0
    # ...and the poll gets only what the submit left of it.
    assert seen["poll_bound"] == pytest.approx(30.0 - elapsed, abs=0.5)


def test_download_model_wait_false_keeps_the_full_submit_budget(monkeypatch):
    """wait=False is exempt from the end-to-end cap — it never waits.

    A submit cut short may leave no transfer running at all, so the fire-and-
    return path keeps the fixed submit budget however impatient the caller is.
    """
    seen: dict[str, float] = {}

    def fake_run(*args, **kwargs):
        seen["submit_timeout"] = kwargs["timeout"]
        return _submit()

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    _download_model("https://hf.co/x.safetensors", wait=False, timeout_seconds=1.0)

    assert seen["submit_timeout"] == server._DOWNLOAD_SUBMIT_TIMEOUT


def test_download_model_attaches_the_id_when_the_poll_raises(monkeypatch):
    """A poll that errors must not orphan the transfer it was polling.

    The id was minted INSIDE this call, so letting the exception through
    untouched leaves a multi-GB download running detached with no handle to
    enumerate or cancel it. `wait_for_download` needs no such wrapping — its
    caller passed the id in and still holds it.
    """

    def fake_poll(download_id, timeout_seconds):
        raise server.ComfyCliError(
            "comfy-cli timed out after 1.0s", timed_out=True, returncode=None
        )

    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: _submit())
    monkeypatch.setattr(server, "_poll_download", fake_poll)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model("https://hf.co/x.safetensors")

    message = str(excinfo.value)
    assert "a1b2c3d4e5f6" in message  # the handle survives the failure
    assert "cancel_download" in message
    assert "comfy-cli timed out after 1.0s" in message  # original diagnosis kept
    # Structured attributes survive too, so a caller branching on them still can.
    assert excinfo.value.timed_out is True
    assert isinstance(excinfo.value.__cause__, server.ComfyCliError)


def test_download_model_falls_back_to_the_foreground_call_on_an_old_cli(
    patched_async_run, legacy_comfy_cli
):
    """A comfy-cli without `--background` still downloads, the old blocking way."""
    procs = patched_async_run(
        stderr="Done in 55.8s. Saved to /models/x.safetensors",
    )

    result = _download_model("https://hf.co/x.safetensors")

    assert result["ok"] is True
    assert "Done in 55.8s" in result["message"]
    # Exactly one spawn reached comfy-cli — the rejected `--background` submit
    # never ran anything — and the retry drops the flag it choked on.
    assert len(procs) == 1
    assert "--background" not in procs[0].cmd
    # Spawned through the ASYNC runner, with the same guardrails as every other
    # async spawn: own process group (so a kill reaps the tree) and no inherited
    # stdin (that fd is this server's JSON-RPC channel).
    assert procs[0].start_new_session is True
    assert procs[0].stdin_arg == asyncio.subprocess.DEVNULL
    # It ran on the legacy path, and now says so even though `wait=True` was
    # honored as well as it can be here — there is still no `download_id`.
    assert result["background_unsupported"] is True


def test_download_model_fallback_flags_a_wait_false_it_could_not_honor(
    patched_async_run, legacy_comfy_cli
):
    """The legacy path blocks even on wait=False, and says so in the payload.

    There is no `--background` to detach and no `download_id` to hand back, so
    the whole transfer runs inside the call. Marking that lets a caller who
    asked NOT to block see that it blocked — and that the download family has no
    id to poll — instead of inferring both from a missing key.

    Deliberately still a success: refusing would remove the only way to download
    a model on a comfy-cli that predates `--background`, and the file did land.
    """
    patched_async_run(stderr="Done in 55.8s. Saved to /models/x.safetensors")

    result = _download_model("https://hf.co/x.safetensors", wait=False)

    assert result["ok"] is True
    assert result["background_unsupported"] is True
    assert "download_id" not in result


def test_download_model_legacy_wait_true_spends_what_the_submit_left(monkeypatch):
    """The waited fallback is bounded by `timeout_seconds`, not a silent 30 min.

    The rejected `--background` submit already spent part of the caller's
    deadline, so the foreground transfer gets the remainder — the same two-phase
    spend the modern path does with its poll — and `_DOWNLOAD_SYNC_TIMEOUT` is
    only the cap on it.
    """
    seen: dict[str, float] = {}
    elapsed = 0.2

    def fake_run(*args, **kwargs):
        time.sleep(elapsed)  # the submit burns part of the deadline before failing
        raise _missing_background_error()

    async def fake_async(*args, timeout=None, **kwargs):
        seen["bound"] = timeout
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    _download_model("https://hf.co/x.safetensors", timeout_seconds=30.0)

    assert seen["bound"] == pytest.approx(30.0 - elapsed, abs=0.5)


def test_download_model_legacy_wait_true_is_capped_by_the_sync_ceiling(monkeypatch):
    """A caller who budgeted more than the cap is still held only to the cap."""
    seen: dict[str, float] = {}

    async def fake_async(*args, timeout=None, **kwargs):
        seen["bound"] = timeout
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", _reject_background)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    _download_model(
        "https://hf.co/x.safetensors", timeout_seconds=server._MAX_DOWNLOAD_WAIT_TIMEOUT
    )

    assert seen["bound"] == server._DOWNLOAD_SYNC_TIMEOUT


def test_download_model_legacy_refuses_an_exhausted_remainder(monkeypatch):
    """An all-but-spent deadline REFUSES the transfer instead of starting one.

    Starting one anyway would be worse than useless: `comfy model download` writes
    straight to the FINAL path, so a child killed moments after it opened the
    destination truncates whatever complete model was already there — bought with a
    bound that could never have finished the transfer. The `--background` path
    takes the same position on a bound too small to resolve the download at all.
    """
    spawned: list[float | None] = []

    def fake_run(*args, **kwargs):
        time.sleep(1.2)  # overspends the caller's whole 1s budget
        raise _missing_background_error()

    async def fake_async(*args, timeout=None, **kwargs):
        spawned.append(timeout)
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model("https://hf.co/x.safetensors", timeout_seconds=1.0)

    assert spawned == []  # nothing spawned, so nothing on disk was touched
    message = str(excinfo.value)
    # The caller's deadline expiring, just detected before the spend instead of
    # after it — so it reads as a timeout to anyone keying on that.
    assert excinfo.value.timed_out is True
    assert "NOTHING was downloaded" in message
    assert "predates `model download --background`" in message
    # And the same way out the timeout path offers, so the refusal is actionable.
    assert "timeout_seconds" in message and "1800" in message
    assert "upgrade comfy-cli" in message


def test_download_model_legacy_spends_a_remainder_above_the_minimum(monkeypatch):
    """A remainder that COULD carry a transfer is spent, not refused.

    The refusal above is scoped to a budget too small to start under; anything
    above `_MIN_LEGACY_DOWNLOAD_TIMEOUT` still gets its foreground transfer, which
    is the only way to download a model on a CLI this old.
    """
    seen: dict[str, float] = {}

    async def fake_async(*args, timeout=None, **kwargs):
        seen["bound"] = timeout
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", _reject_background)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    _download_model(
        "https://hf.co/x.safetensors",
        timeout_seconds=server._MIN_LEGACY_DOWNLOAD_TIMEOUT + 5.0,
    )

    assert seen["bound"] >= server._MIN_LEGACY_DOWNLOAD_TIMEOUT


def test_download_model_legacy_wait_false_keeps_the_full_cap(monkeypatch):
    """wait=False stays on the cap: it explicitly decoupled from waiting.

    It never reads `timeout_seconds` (the docstring says so), so tightening its
    bound would only truncate a transfer nobody asked to be quick. What it gains
    from the async runner is the reaping on cancellation, not a shorter deadline.
    """
    seen: dict[str, float] = {}

    async def fake_async(*args, timeout=None, **kwargs):
        seen["bound"] = timeout
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", _reject_background)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    _download_model("https://hf.co/x.safetensors", wait=False, timeout_seconds=5.0)

    assert seen["bound"] == server._DOWNLOAD_SYNC_TIMEOUT


def test_download_model_does_not_fall_back_on_a_real_submit_failure(monkeypatch):
    """Only an unknown `--background` degrades — anything else must propagate.

    A submit that failed for some other reason may already have started a
    transfer, and re-running it here would fetch the same multi-GB file twice.
    """
    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        raise server.ComfyCliError(
            "comfy-cli returned no JSON (exit 1). stderr: connection reset",
            no_envelope=True,
            returncode=1,
        )

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="connection reset"):
        _download_model("https://hf.co/x.safetensors")

    assert len(calls) == 1  # never retried


def test_download_model_clamps_an_oversized_timeout(monkeypatch):
    """An `inf` bound is clamped before the poll loop can run on it forever."""
    seen: dict[str, float] = {}

    def fake_poll(download_id, timeout_seconds):
        seen["bound"] = timeout_seconds
        return _status("completed")

    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: _submit())
    monkeypatch.setattr(server, "_poll_download", fake_poll)

    _download_model("https://hf.co/x.safetensors", timeout_seconds=float("inf"))

    # The poll gets the ceiling MINUS whatever the submit just spent, so this is
    # bounded-just-under rather than equal — the deduction is the point (see
    # `test_download_model_spends_one_end_to_end_budget`), and the submit here is
    # a stubbed call that returns in microseconds.
    assert 0 < seen["bound"] <= server._MAX_DOWNLOAD_WAIT_TIMEOUT
    assert seen["bound"] == pytest.approx(server._MAX_DOWNLOAD_WAIT_TIMEOUT, abs=1.0)


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_download_model_rejects_a_bad_timeout_before_submitting(monkeypatch, bad):
    """NaN/0/negative are refused BEFORE anything detaches.

    Validating after the submit would leave a background worker running that
    nobody is waiting on — and with NaN every comparison is False, so the poll
    loop would never exit on its own.
    """
    calls = _sequenced(monkeypatch, [_submit()])

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        _download_model("https://hf.co/x.safetensors", timeout_seconds=bad)

    assert calls == []


def test_download_model_wait_false_ignores_the_timeout_argument(monkeypatch):
    """wait=False never reads `timeout_seconds`, so it must not validate it.

    The submit runs on its own fixed budget; rejecting a value it ignores would
    newly refuse a call that works fine today (same rule as `run_workflow`).
    """
    _sequenced(monkeypatch, [_submit()])

    assert (
        _download_model(
            "https://hf.co/x.safetensors", wait=False, timeout_seconds=float("nan")
        )
        == _submit()
    )


def test_download_status_maps_command_and_returns_data(patched_run):
    """download_status wraps `comfy model download-status <id>`."""
    payload = _status("downloading", percent=25.0)
    calls = patched_run(envelope(data=payload))

    assert server.download_status("a1b2c3d4e5f6") == payload
    assert calls[0]["cmd"][4:] == ["model", "download-status", "a1b2c3d4e5f6"]
    assert calls[0]["timeout"] == 60.0


def test_cancel_download_maps_command_and_returns_data(patched_run):
    """cancel_download wraps `comfy model download-cancel <id>`."""
    calls = patched_run(envelope(data=_status("cancelled")))

    assert server.cancel_download("a1b2c3d4e5f6")["status"] == "cancelled"
    assert calls[0]["cmd"][4:] == ["model", "download-cancel", "a1b2c3d4e5f6"]
    assert calls[0]["timeout"] == 60.0


def test_cancel_download_unknown_id_raises_error_envelope(patched_run):
    """Cancelling an unknown download surfaces comfy-cli's error envelope."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "download_not_found", "message": "no such download"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="download_not_found"):
        server.cancel_download("nope")


# Click's usage error for a verb the installed comfy-cli does not have: exit 2,
# no envelope. The background-download verb group ships only in releases after
# 1.13.0, while this repo's floor is 1.12.0, so this is the COMMON path today.
_NO_DOWNLOAD_STATUS = (
    2,
    "",
    "Usage: comfy [OPTIONS] COMMAND\nNo such command 'download-status'.",
)
_NO_DOWNLOAD_CANCEL = (
    2,
    "",
    "Usage: comfy [OPTIONS] COMMAND\nNo such command 'download-cancel'.",
)


@pytest.mark.parametrize(
    "tool, reply, verb",
    [
        ("download_status", _NO_DOWNLOAD_STATUS, "download-status"),
        ("wait_for_download", _NO_DOWNLOAD_STATUS, "download-status"),
        ("cancel_download", _NO_DOWNLOAD_CANCEL, "download-cancel"),
    ],
)
def test_download_companions_degrade_on_a_comfy_cli_without_the_verb(
    patched_run, tool, reply, verb
):
    """A missing verb reads as a capability gap, not as a broken MCP.

    `download_model` already degrades for the option-shaped half of this same
    version gap (`--background`); these three are the verb-shaped half. The
    group is all-or-nothing, so a CLI missing them also rejects `--background`
    and can never have minted an id — nothing is actually lost, and the message
    points at the inline `download_model` that DOES work there.
    """
    returncode, stdout, stderr = reply
    patched_run(stdout, returncode=returncode, stderr=stderr)

    result = getattr(server, tool)("a1b2c3d4e5f6")

    assert result["unsupported"] is True
    assert f"model {verb} unavailable" in result["error"]
    # Points at the path that still works rather than dead-ending.
    assert "download_model" in result["error"]
    # None of the raw wrapper/CLI text leaks through.
    assert "No such command" not in result["error"]
    assert "Usage: comfy" not in result["error"]
    assert "returned no JSON" not in result["error"]


@pytest.mark.parametrize("tool", ["download_status", "wait_for_download"])
def test_download_companions_keep_a_real_error_raw(patched_run, tool):
    """A verb comfy-cli DID dispatch must never be waved through as a gap.

    An unknown id is a real answer from a working CLI. Degrading it would tell
    the agent nothing is broken while its download silently isn't there.
    """
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "download_not_found", "message": "no such download"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="download_not_found"):
        getattr(server, tool)("a1b2c3d4e5f6")


def test_wait_for_download_returns_the_terminal_payload(monkeypatch):
    """wait_for_download polls until terminal and returns that final payload."""
    calls = _sequenced(monkeypatch, [_status("downloading"), _status("completed")])

    assert server.wait_for_download("a1b2c3d4e5f6") == _status("completed")
    assert calls[0] == ("model", "download-status", "a1b2c3d4e5f6")
    assert len(calls) == 2


def test_wait_for_download_returns_a_failure_rather_than_raising(monkeypatch):
    """Like `wait_for_job`, a terminal failure comes back as a payload to read.

    Only `download_model` turns that into a raise — this tool is the polling
    primitive, and an agent chaining it wants the `status` / `error` fields.
    """
    _sequenced(monkeypatch, [_status("failed", error="checksum mismatch")])

    result = server.wait_for_download("a1b2c3d4e5f6")

    assert result["status"] == "failed"
    assert result["error"] == "checksum mismatch"


def test_wait_for_download_times_out_cleanly(monkeypatch):
    """An unfinished transfer returns the timed-out envelope, carrying the id."""
    _sequenced(monkeypatch, [_status("downloading")] * 10)

    # A monotonic clock that jumps 10s per read, so the 25s bound expires.
    clock = {"t": 0.0}

    def fake_monotonic():
        now = clock["t"]
        clock["t"] += 10.0
        return now

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    assert server.wait_for_download("a1b2c3d4e5f6") == {
        "timed_out": True,
        "download_id": "a1b2c3d4e5f6",
        "status": _status("downloading"),
    }


def test_wait_for_download_caps_each_poll_to_the_remaining_bound(monkeypatch):
    """A single poll never gets a longer subprocess budget than the wait itself."""
    seen: list[float] = []

    def fake_run(*args, timeout=None, **kwargs):
        seen.append(timeout)
        return _status("downloading")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    clock = {"t": 0.0}

    def fake_monotonic():
        now = clock["t"]
        clock["t"] += 1.0
        return now

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = server.wait_for_download("a1b2c3d4e5f6", timeout_seconds=5.0)

    assert result["timed_out"] is True
    assert seen == [4.0, 2.0]  # each poll gets only what is left of the 5s bound


@pytest.mark.parametrize("oversized", [float("inf"), 86_400.0])
def test_wait_for_download_clamps_an_oversized_timeout(monkeypatch, oversized):
    """An oversized bound is clamped to the ceiling, so the poll loop terminates."""
    _sequenced(monkeypatch, [_status("downloading")] * 4)

    reads = iter([0.0, 1.0, server._MAX_DOWNLOAD_WAIT_TIMEOUT + 1.0])

    def fake_monotonic():
        try:
            return next(reads)
        except StopIteration:
            pytest.fail("wait_for_download kept polling past the clamped ceiling")

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    assert server.wait_for_download("a1b2c3d4e5f6", timeout_seconds=oversized)[
        "timed_out"
    ]


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_wait_for_download_rejects_a_non_positive_or_nan_timeout(monkeypatch, bad):
    """NaN/0/negative are refused before the first poll."""
    calls = _sequenced(monkeypatch, [])

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        server.wait_for_download("a1b2c3d4e5f6", timeout_seconds=bad)

    assert calls == []


# The download family takes its id as a bare positional exactly as the jobs
# family takes `prompt_id`, so it carries the same guard: a leading dash would
# reach comfy-cli as an option, a NUL makes `subprocess.Popen` raise a bare
# ValueError, an empty id can only be a caller mistake, and an oversized one
# fails in the exec as an OSError nobody converts.
@pytest.mark.parametrize(
    "tool",
    ["download_status", "cancel_download", "wait_for_download"],
)
@pytest.mark.parametrize("bad_id", ["--help", "-o", "", "d\x001", "d" * 129])
def test_download_tools_reject_an_unusable_download_id(monkeypatch, tool, bad_id):
    """A dash-led / empty / NUL-bearing / oversized id is refused before any spawn."""

    def fake_run(*args, **kwargs):
        raise AssertionError(f"{tool} spawned comfy-cli with {bad_id!r}")

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="download_id"):
        getattr(server, tool)(bad_id)


def test_download_model_guards_the_id_comfy_cli_hands_back(monkeypatch):
    """The polled id is guarded too — it becomes argv on every poll."""
    _sequenced(monkeypatch, [_submit(download_id="--evil")])

    with pytest.raises(server.ComfyCliError, match="download_id"):
        _download_model("https://hf.co/x.safetensors")


@pytest.mark.parametrize(
    "rendered",
    [
        # Click >= 8's own `NoSuchOption.format_message()`, verbatim...
        "Error: No such option '--background'.",
        # ...including the suggestion it appends when a near-miss exists...
        "Error: No such option '--background'. Did you mean '--backend'?",
        # ...the older colon spelling...
        "Error: No such option: --background",
        # ...and the colourized rich panel Typer wraps it in, which can break the
        # phrase across lines. `_normalize_cli_text` folds all of that away.
        _NO_SUCH_OPTION_STDERR,
    ],
)
def test_is_missing_option_error_matches_clicks_usage_error(rendered):
    """Every shape Click's unknown-option rejection actually reaches us in."""
    exc = server.ComfyCliError(
        f"comfy-cli returned no JSON (exit 2). stderr: {rendered} | stdout: <empty>",
        no_envelope=True,
        returncode=2,
    )

    assert server._is_missing_option_error(exc, "--background") is True


def test_is_missing_option_error_requires_no_envelope():
    """An envelope means comfy-cli RAN the command — never a parse rejection.

    Otherwise a nested error comfy-cli merely relayed (a pip/git call, a registry
    message) could quote the phrase and trigger a second, blocking download.
    """
    exc = server.ComfyCliError(
        f"stderr: {_NO_SUCH_OPTION_STDERR}", no_envelope=False, returncode=2
    )

    assert server._is_missing_option_error(exc, "--background") is False


def test_is_missing_option_error_requires_the_usage_exit_status():
    """Exit 2 is Click's `UsageError` status: nothing was ever dispatched."""
    exc = server.ComfyCliError(
        f"stderr: {_NO_SUCH_OPTION_STDERR}", no_envelope=True, returncode=1
    )

    assert server._is_missing_option_error(exc, "--background") is False


def test_is_missing_option_error_does_not_match_a_longer_option_name():
    """`--background` must not match a rejection naming `--background-worker`."""
    exc = server.ComfyCliError(
        "comfy-cli returned no JSON (exit 2). stderr: "
        "Error: No such option: --background-worker",
        no_envelope=True,
        returncode=2,
    )

    assert server._is_missing_option_error(exc, "--background") is False


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


def test_wait_for_job_sleeps_the_shared_poll_interval(monkeypatch):
    """The gap between polls is the named cadence, not a second hardcoded 2.0.

    `wait_for_job` used to keep its own `poll_interval = 2.0` local while
    `_poll_download` read `_POLL_INTERVAL` — the same number spelled two ways,
    which is exactly how a cadence change lands on one poller and not the other.
    """
    statuses = iter([{"status": "running"}, {"status": "completed"}])
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: next(statuses))

    slept: list[float] = []
    monkeypatch.setattr(server.time, "sleep", slept.append)

    assert server.wait_for_job("pid", timeout_seconds=600.0) == {"status": "completed"}
    assert slept == [server._POLL_INTERVAL]  # the sleep is capped by the remaining


def test_the_two_bounded_polls_run_on_one_shared_loop(monkeypatch):
    """`wait_for_job` and `_poll_download` both route through the shared loop.

    The two ran statement-for-statement identical loops — the one-poll minimum,
    the per-poll budget cap, the three-clause re-raise — and drifted apart in the
    only place they were spelled twice. Re-inlining either one is what this
    guards against; the loop's own behavior is covered by the tests around it.
    """
    calls: list[tuple[tuple, dict]] = []

    def fake_poll(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "completed"}

    monkeypatch.setattr(server, "_poll_until_terminal", fake_poll)

    server.wait_for_job("pid", timeout_seconds=25.0)
    server._poll_download("a1b2c3d4e5f6", 25.0)

    assert calls[0][0] == ("jobs", "status", "pid")
    assert calls[0][1]["is_terminal"] is server._is_terminal
    assert "timed_out_extra" not in calls[0][1]  # jobs add no extra key
    assert calls[1][0] == ("model", "download-status", "a1b2c3d4e5f6")
    assert calls[1][1]["is_terminal"] is server._is_download_terminal
    assert calls[1][1]["timed_out_extra"] == {"download_id": "a1b2c3d4e5f6"}


def test_the_shared_poll_puts_its_extra_keys_before_the_status(monkeypatch):
    """A timed-out payload keeps the key order each caller already returned."""
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"status": "running"})
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # A clock that jumps 10s per read, so the 25s bound expires mid-poll.
    clock = {"t": 0.0}

    def fake_monotonic():
        now = clock["t"]
        clock["t"] += 10.0
        return now

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = server._poll_download("a1b2c3d4e5f6", 25.0)

    assert list(result) == ["timed_out", "download_id", "status"]
    assert result == {
        "timed_out": True,
        "download_id": "a1b2c3d4e5f6",
        "status": {"status": "running"},
    }


@pytest.mark.parametrize("key", ["timed_out", "status"])
def test_the_shared_poll_rejects_an_extra_that_shadows_its_own_keys(monkeypatch, key):
    """An extra key may not redefine the two keys every caller branches on.

    The extras are unpacked AFTER the `timed_out` literal, so a colliding key
    would win — `download_model` reads `result.get("timed_out")` to tell an
    expiry from a real result, and a shadowed `True` would send a timeout down
    the `_download_failed` path instead.
    """
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"status": "running"})

    with pytest.raises(server.ComfyCliError, match=f"reserved keys: \\['{key}'\\]"):
        server._poll_until_terminal(
            "model",
            "download-status",
            "a1b2c3d4e5f6",
            timeout_seconds=25.0,
            is_terminal=server._is_download_terminal,
            timed_out_extra={key: "shadowed"},
        )


@pytest.mark.parametrize("timeout_seconds", [float("inf"), float("nan")])
def test_the_shared_poll_rejects_an_unbounded_timeout(monkeypatch, timeout_seconds):
    """The bound the loop needs is enforced here, not just documented.

    Extracting the loop moved `_bounded_timeout` away from the code it protects:
    with `inf` every `remaining <= 0` stays False forever, and with NaN every
    comparison is False and `min(_POLL_INTERVAL, nan)` yields 2.0 — either way a
    caller that forgot to clamp re-spawns `comfy` on a worker thread until the
    client gives up. Refuse before the first spawn instead.
    """
    spawned: list[tuple] = []
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: spawned.append(a))

    with pytest.raises(server.ComfyCliError, match="invalid timeout_seconds"):
        server._poll_until_terminal(
            "jobs",
            "status",
            "pid",
            timeout_seconds=timeout_seconds,
            is_terminal=server._is_terminal,
        )
    assert spawned == []


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0])
def test_the_shared_poll_still_takes_an_expired_bound(monkeypatch, timeout_seconds):
    """A bound at or below zero is legal and hits the one-poll minimum.

    `download_model` spends what its submit left (`deadline - monotonic()`),
    which legitimately lands at or under zero on a slow submit, and still wants
    a real status payload rather than a contentless `{"status": None}`. The
    finiteness guard above must not swallow that documented path.
    """
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"status": "running"})

    assert server._poll_download("a1b2c3d4e5f6", timeout_seconds) == {
        "timed_out": True,
        "download_id": "a1b2c3d4e5f6",
        "status": {"status": "running"},
    }


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
    assert instructions  # present on the MCPServer instance

    # Call-server-info-first + the async submit -> poll -> fetch generation loop.
    for tool in ("server_info", "run_workflow", "wait_for_job", "fetch_outputs"):
        assert tool in instructions

    # The template on-ramp.
    for tool in ("search_templates", "fetch_template"):
        assert tool in instructions


def test_server_instructions_document_the_argument_naming_convention():
    """The handshake states the naming convention, so callers stop guessing.

    The convention held across the tool surface long before it was written down
    anywhere, so first-time agent callers guessed `path` / `workflow` and burned
    a round trip. Stating it up front is the fix.
    """
    instructions = server.mcp.instructions

    for argument in (
        "workflow_path",
        "out_path",
        "out_dir",
        "name",
        "prompt_id",
        "download_id",
    ):
        assert argument in instructions


def test_tool_arguments_follow_the_naming_convention():
    """The live schemas obey the convention the instructions advertise.

    Guards against drift in the direction that costs an agent a round trip: a
    new tool that spells its workflow input `path` / `workflow`, or its output
    file anything other than `out_path` (the `partner_generate` `download`
    outlier this test was written alongside). Clients introspect these schemas
    fresh each session, so the schema IS the contract.
    """
    schemas = {
        tool.name: tool.parameters.get("properties", {})
        for tool in server.mcp._tool_manager.list_tools()
    }

    # Every tool that consumes a workflow FILE names it `workflow_path`.
    for name in (
        "run_workflow",
        "validate_workflow",
        "list_workflow_slots",
        "set_workflow_slot",
        "vary_workflow",
    ):
        assert "workflow_path" in schemas[name]

    # Output file -> `out_path`; output directory -> `out_dir`.
    for name in ("fetch_template", "partner_generate"):
        assert "out_path" in schemas[name]
    for name in ("fetch_outputs", "vary_workflow"):
        assert "out_dir" in schemas[name]

    # And no tool offers a near-miss spelling of any of them — a second visible
    # name for one concept is exactly the guess space the convention removes.
    banned = {"path", "workflow", "download", "output_path", "directory"}
    for name, properties in schemas.items():
        assert not banned & set(properties), name


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

    assert _launch() == {"pid": 42}

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert cmd[4:] == ["launch", "--background"]  # no extras -> no `--` separator


def test_launch_comfyui_forwards_extra_args_after_separator(patched_run):
    """Extra args are forwarded to ComfyUI after a `--` separator."""
    calls = patched_run(envelope(data={}))

    _launch(["--port", "8189"])

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

    result = _launch()

    assert result["ok"] is True
    assert result["action"] == "launch"
    assert "Launched ComfyUI" in result["message"]


def test_launch_comfyui_nonzero_exit_still_raises(patched_plain_run):
    """A real launch failure (non-zero exit, no envelope) must still raise."""
    patched_plain_run(1, stderr="Address already in use: port 8188")

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        _launch()


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

    result = _launch()

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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _FakeProc(
            list(cmd),
            "starting run...\ncomfy-cli crashed before emitting a result\n",
            stderr_text="Traceback (most recent call last):\nRuntimeError: nope",
            env=env,
        )
        proc.returncode = 1
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _FakeProc(
            list(cmd),
            # An event that both parses AND claims success — the worst case.
            '{"type": "progress", "ok": true, "data": {"value": 3}}\n'
            "comfy-cli crashed mid-run\n",
            stderr_text="RuntimeError: nope",
            env=env,
        )
        proc.returncode = 1
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _FakeProc(
            list(cmd),
            '{"schema": "envelope/2", "type": "envelope", "ok": true, "data": {}}\n',
            env=env,
        )
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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


# --- download_model: the LEGACY synchronous fallback ------------------------
#
# A comfy-cli that predates `model download --background` rejects the flag at
# parse time, and `download_model` falls back to the old one-shot synchronous
# call. That path keeps the old contract in full, including the BE-3345
# no-envelope handling below — which is now scoped to it alone, since a
# `--background` submit returns a real envelope. `legacy_comfy_cli` is what
# makes the installed CLI look that old.


def test_download_model_synthesizes_success_on_plain_exit(
    patched_async_run, legacy_comfy_cli
):
    """`comfy model download` streams text + exits 0, no envelope -> success.

    The download landed on disk (exit 0), so instead of raising the
    "returned no JSON" false negative — which would invite a bandwidth-expensive
    retry of a multi-GB fetch — a success payload is synthesized carrying the
    CLI's printed tail (where "Done in …" and the saved path live).
    """
    patched_async_run(
        stderr="Downloading x.safetensors...\nDone in 55.8s. Saved to /models/x.safetensors",
    )

    result = _download_model("https://hf.co/x.safetensors")

    assert result["ok"] is True
    assert result["action"] == "model download"
    assert "Done in 55.8s" in result["message"]


def test_download_model_nonzero_exit_still_raises(patched_async_run, legacy_comfy_cli):
    """A real download failure (non-zero exit, no envelope) must still raise."""
    patched_async_run(returncode=1, stderr="HTTP 404: model not found")

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        _download_model("https://hf.co/missing.safetensors")


def test_download_model_synthesizes_despite_stray_non_envelope_json(
    patched_async_run, legacy_comfy_cli
):
    """A stray non-envelope JSON line on a clean download exit is still success.

    `_last_json_object` returns any JSON object (not just `type==envelope`), so a
    diagnostic line that happens to parse must NOT be mistaken for a result
    envelope and unwrapped into a spurious failure (BE-3345 edge case).
    """
    patched_async_run(
        '{"level": "info", "msg": "connection reused"}\n',
        stderr="Done in 12.3s. Saved to /models/x.safetensors",
    )

    result = _download_model("https://hf.co/x.safetensors")

    assert result["ok"] is True
    assert result["action"] == "model download"
    assert "Done in 12.3s" in result["message"]


def test_download_model_envelope_then_diagnostic_keeps_envelope_data(
    patched_async_run, legacy_comfy_cli
):
    """A real envelope FOLLOWED BY a diagnostic JSON line still wins (BE-3345).

    `_last_json_object` prefers a `type==envelope` object over a later plain JSON
    line, so a trailing diagnostic must NOT null out `real_envelope` and demote a
    genuine success into the synthesized fast-path — the envelope's `data` (the
    real saved-path metadata) must be returned, not the printed-text stopgap.
    """
    patched_async_run(
        '{"type": "envelope", "ok": true, "data": {"saved": "/models/x"}}\n'
        '{"level": "info", "msg": "cleanup done"}\n'
    )

    result = _download_model("https://hf.co/x.safetensors")

    # The legacy marker rides along on the envelope's own data, which is
    # otherwise returned verbatim.
    assert result == {"saved": "/models/x", "background_unsupported": True}


def test_download_model_error_envelope_then_diagnostic_still_raises(
    patched_async_run, legacy_comfy_cli
):
    """An error envelope followed by a diagnostic line still raises, not synthesized.

    Even on exit 0, a trailing diagnostic JSON line must not mask an earlier
    error envelope as a synthesized success — the error envelope is preferred and
    propagates its code (BE-3345).
    """
    patched_async_run(
        '{"type": "envelope", "ok": false, '
        '"error": {"code": "download_failed", "message": "checksum mismatch"}}\n'
        '{"level": "info", "msg": "cleanup done"}\n'
    )

    with pytest.raises(server.ComfyCliError, match=r"download_failed"):
        _download_model("https://hf.co/x.safetensors")


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
        _download_model("https://hf.co/x.safetensors")


def test_download_model_keeps_saved_path_tail_on_verbose_output(
    patched_async_run, legacy_comfy_cli
):
    """A verbose multi-GB fetch caps to the TAIL so the saved-path survives.

    `comfy model download` streams progress noise ahead of the `Done in …` /
    saved-path tail that the synthesized payload exists to surface. A front-slice
    cap would drop that tail as noise, so the message must keep the last chars.
    """
    tail = "Done in 903.4s. Saved to /models/checkpoints/big.safetensors"
    noise = "\n".join(f"Downloading... {i}% ({i * 40} MiB)" for i in range(100))
    patched_async_run(stderr=f"{noise}\n{tail}")

    result = _download_model("https://hf.co/big.safetensors")

    assert result["ok"] is True
    assert len(result["message"]) <= 1000
    assert "Saved to /models/checkpoints/big.safetensors" in result["message"]


def test_download_model_fallback_omits_url_when_no_output(
    patched_async_run, legacy_comfy_cli
):
    """The no-output fallback message must not echo the (possibly signed) URL.

    A `model download` URL can carry a token / userinfo in its query string; the
    synthesized fallback lands in the tool response and host logs, so it reports
    only the flag-free `model download` action, never the raw args.
    """
    patched_async_run()  # exit 0 with no stdout/stderr -> fallback message

    result = _download_model("https://hf.co/x.safetensors?sig=SECRETTOKEN")

    assert result["ok"] is True
    assert "SECRETTOKEN" not in result["message"]
    assert "hf.co" not in result["message"]
    assert "model download" in result["message"]


def test_download_model_legacy_timeout_kills_the_transfer_and_guides_the_caller(
    patched_async_run, legacy_comfy_cli, monkeypatch
):
    """A waited fallback that overruns its bound kills the transfer and says so.

    This is the BE-5167 shape: the bytes were moving inside THIS call, so unlike
    the `--background` path there is no detached worker left to poll — the
    transfer is dead and a partial file may be on disk with no `download_id` to
    check it with. The error has to carry all of that, plus the way out.
    """
    # Shrink the MINIMUM, not the cap, so a fraction-of-a-second budget is still
    # big enough to spawn under: `_DOWNLOAD_SYNC_TIMEOUT` has to stay real so the
    # retry guidance quotes the number a caller would actually pass.
    monkeypatch.setattr(server, "_MIN_LEGACY_DOWNLOAD_TIMEOUT", 0.05)
    logged: list[dict] = []
    monkeypatch.setattr(
        failure_log,
        "_log_failure",
        lambda kind, args, **kwargs: logged.append({"kind": kind, **kwargs}),
    )
    # A child that outlives its deadline, having printed some progress first: the
    # runner's drain reads that into its sinks before the bound cancels it, which
    # is the whole reason those sinks are owned by the caller rather than by the
    # reader coroutine.
    procs = patched_async_run(stderr="Downloading big.safetensors... 12%", hang=True)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model(
            "https://hf.co/big.safetensors",
            relative_path="models/checkpoints",
            filename="big.safetensors",
            # Enough budget to start the transfer, nowhere near enough to finish.
            timeout_seconds=0.3,
        )

    message = str(excinfo.value)
    assert excinfo.value.timed_out is True
    # The original diagnosis is APPENDED to, not replaced — including the stderr
    # tail the CANCELLED reader had already collected.
    assert "comfy-cli timed out after" in message
    assert "Downloading big.safetensors... 12%" in message
    assert "predates `model download --background`" in message
    # Name the exact partial file: `filename` was supplied, so it is knowable.
    assert "models/checkpoints/big.safetensors" in message
    assert "timeout_seconds" in message and "1800" in message  # how to retry
    assert "upgrade comfy-cli" in message
    # The process TREE was killed rather than left to keep writing.
    assert procs[0].killed is True
    assert [record["kind"] for record in logged] == ["timeout"]


def test_download_model_legacy_timeout_names_the_folder_without_a_filename(
    patched_async_run, legacy_comfy_cli, monkeypatch
):
    """Without `filename` the basename is comfy-cli's to derive, so name the dir.

    Guessing a name from the URL would send the caller looking for the wrong
    file — worse than reporting the folder the partial is somewhere inside.
    """
    monkeypatch.setattr(server, "_MIN_LEGACY_DOWNLOAD_TIMEOUT", 0.05)
    patched_async_run(hang=True)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model("https://hf.co/big.safetensors", timeout_seconds=0.3)

    message = str(excinfo.value)
    assert "models/ (relative to the workspace root" in message  # the default dir
    assert "name from the URL" in message


@pytest.mark.parametrize("wait", [False, True])
def test_download_model_legacy_timeout_at_the_cap_does_not_suggest_a_bigger_bound(
    patched_async_run, legacy_comfy_cli, monkeypatch, wait
):
    """A bound that WAS the cap makes "raise timeout_seconds" dead advice.

    Keyed off the effective bound, not off `wait`, because both callers here were
    already capped: `wait=False` never reads `timeout_seconds` on this path, and a
    waiting caller who passed a value at or above the cap was clamped to it. Either
    way, pointing them at the parameter would send them round a loop that changes
    nothing — the upgrade route is the real one.
    """
    monkeypatch.setattr(server, "_DOWNLOAD_SYNC_TIMEOUT", 0.05)
    patched_async_run(hang=True)

    with pytest.raises(server.ComfyCliError) as excinfo:
        # `timeout_seconds` far above the (shrunken) cap, so the waiting call is
        # clamped to it exactly as `wait=False` always is.
        _download_model(
            "https://hf.co/big.safetensors", wait=wait, timeout_seconds=30.0
        )

    message = str(excinfo.value)
    assert "retry with a larger `timeout_seconds`" not in message
    assert "cannot widen it" in message
    assert "upgrade comfy-cli" in message


def test_download_model_legacy_timeout_message_redacts_the_signed_url(
    patched_async_run, legacy_comfy_cli, monkeypatch
):
    """The argv echoed in a timeout must not carry the URL's credential.

    A CivitAI / HuggingFace model URL keeps its token in a `?token=…` query (or in
    `<user>:<pass>@` userinfo), and this message lands in the tool response the MCP
    client renders and in the host's logs — the same exposure that makes
    `_synthesize_plain_result` omit raw args altogether.
    """
    monkeypatch.setattr(server, "_DOWNLOAD_SYNC_TIMEOUT", 0.05)
    patched_async_run(hang=True)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model(
            "https://<user>:<pw>@hf.co/big.safetensors?token=SECRETTOKEN", wait=False
        )

    message = str(excinfo.value)
    assert "SECRETTOKEN" not in message
    assert "<user>:<pw>" not in message
    # Masked, not dropped: which command wedged is still legible.
    assert "model download --url https://***@hf.co/big.safetensors" in message


@pytest.mark.parametrize(
    ("relative_path", "filename", "expected"),
    [
        # No `relative_path` -> comfy-cli's own DEFAULT_COMFY_MODEL_PATH.
        (None, None, "models/ (relative"),
        (None, "x.safetensors", "models/x.safetensors (relative"),
        ("models/loras", None, "models/loras/ (relative"),
        ("models/loras", "x.safetensors", "models/loras/x.safetensors (relative"),
    ],
)
def test_legacy_download_partial_names_what_it_can(relative_path, filename, expected):
    """The exact path is only knowable when the caller supplied `filename`.

    Without it comfy-cli derives the basename from the URL, so naming a guessed
    file would send the caller looking for the wrong one — report the directory.
    """
    described = server._legacy_download_partial(relative_path, filename)

    assert described.startswith(expected)
    if not filename:
        assert "name from the URL" in described


def test_download_model_legacy_non_timeout_failure_propagates_unwrapped(
    patched_async_run, legacy_comfy_cli
):
    """Only a TIMEOUT gets the partial-file guidance; a real failure passes through.

    A 404 or a checksum mismatch is comfy-cli's own verdict — dressing it up as
    "the transfer was killed at its bound, a partial may remain" would be a lie.
    """
    patched_async_run(returncode=1, stderr="HTTP 404: model not found")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model("https://hf.co/missing.safetensors")

    message = str(excinfo.value)
    assert "HTTP 404" in message
    assert excinfo.value.timed_out is False
    assert "INCOMPLETE file may remain" not in message


def test_download_model_legacy_cancellation_reaps_the_transfer(
    patched_async_run, legacy_comfy_cli, monkeypatch
):
    """Cancelling the tool call must kill the transfer, not orphan it (BE-5167).

    This is what `asyncio.to_thread(_run_comfy, …)` could not do: its cancellation
    never reached the thread, so an MCP cancel notification (or a stdio shutdown
    tearing the task group down) left `comfy model download` and its multi-GB
    partial running, killable only by pid.
    """
    procs = patched_async_run(hang=True)

    async def drive():
        # Wrap the fixture's fake so the cancel fires at a DETERMINISTIC point —
        # once the fallback child exists. Cancelling on a fixed number of loop
        # turns would race the `to_thread` hop the rejected submit makes first.
        spawned = asyncio.Event()
        fake_exec = server.asyncio.create_subprocess_exec

        async def notifying_exec(*args, **kwargs):
            proc = await fake_exec(*args, **kwargs)
            spawned.set()
            return proc

        monkeypatch.setattr(server.asyncio, "create_subprocess_exec", notifying_exec)
        task = asyncio.ensure_future(
            server.download_model("https://hf.co/big.safetensors", wait=False)
        )
        await spawned.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())

    assert len(procs) == 1
    assert procs[0].killed is True  # the `finally` fired


def test_run_comfy_async_matches_the_plain_runners_argv_and_unwrap(patched_async_run):
    """The async twin is a twin: same global-flags-first argv, same unwrap."""
    procs = patched_async_run(envelope(data={"x": 1}))

    assert asyncio.run(server._run_comfy_async("jobs", "status", "abc")) == {"x": 1}

    cmd = procs[0].cmd
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["jobs", "status", "abc"]  # subcommand strictly after
    assert procs[0].env["COMFY_WHERE"] == "local"


def test_run_comfy_async_keeps_the_envelope_behind_oversized_output(patched_async_run):
    """The runner bounds each captured stream to its TAIL, and loses nothing by it.

    `communicate()` would retain every byte a child writes — for this runner that
    means up to `_DOWNLOAD_SYNC_TIMEOUT` of a multi-GB download's verbose progress
    text, the unbounded allocation `_STDERR_MAX_CHARS` exists to prevent. Keeping
    the tail is safe precisely because comfy-cli's envelope is the LAST JSON object
    it prints, so it survives however much noise precedes it.
    """
    noise = "Downloading... chunk\n" * 8000  # comfortably past the cap
    assert len(noise) > server._STDERR_MAX_CHARS
    patched_async_run(noise + json.dumps(envelope(data={"x": 1})), stderr=noise)

    assert asyncio.run(server._run_comfy_async("model", "download")) == {"x": 1}


def test_drain_capped_into_bounds_the_tail_and_survives_cancellation():
    """The sink keeps only the trailing bytes, and outlives a cancelled reader.

    Both halves are load-bearing for `_run_comfy_async`: the bound is what keeps a
    long-lived child's output from accumulating in this process, and the
    CALLER-owned sink is what lets a transfer killed at its deadline still report
    the tail it had printed — a local the coroutine returns dies with the
    cancellation, which is exactly when the diagnostic is wanted.
    """

    async def drive():
        reader = asyncio.StreamReader()
        reader.feed_data(b"abcdef")  # no EOF: the "child" is still running
        sink = [b""]
        task = asyncio.ensure_future(server._drain_capped_into(reader, 4, sink))
        # Let the reader consume what is buffered, then block on the open pipe.
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return sink[0]

    assert asyncio.run(drive()) == b"cdef"


def test_run_comfy_async_raises_an_error_envelopes_code(patched_async_run):
    """An error envelope still raises `ComfyCliError` carrying its code."""
    patched_async_run(
        envelope(ok=False, error={"code": "download_failed", "message": "checksum"}),
        returncode=1,
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server._run_comfy_async("model", "download"))

    assert excinfo.value.code == "download_failed"
    assert excinfo.value.returncode == 1


def test_run_comfy_async_nonzero_exit_without_an_envelope_raises(patched_async_run):
    """`plain_ok` is not in play here, so a bare non-zero exit still raises."""
    patched_async_run(returncode=3, stderr="boom")

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        asyncio.run(server._run_comfy_async("model", "download"))


def test_run_comfy_async_plain_ok_synthesizes_on_a_clean_no_envelope_exit(
    patched_async_run,
):
    """BE-3345 regression guard: exit 0 + human text + no envelope is a SUCCESS.

    Losing this in the move off `_run_comfy` would turn every legacy download
    that actually landed into a "returned no JSON" false negative — and invite a
    retry of a multi-GB fetch.
    """
    patched_async_run(stderr="Done in 55.8s. Saved to /models/x.safetensors")

    result = asyncio.run(server._run_comfy_async("model", "download", plain_ok=True))

    assert result["ok"] is True
    assert result["action"] == "model download"
    assert "Done in 55.8s" in result["message"]


def test_run_comfy_async_decodes_undecodable_bytes_instead_of_raising(
    patched_async_run,
):
    """A mis-encoded byte degrades one character; it must not abort the call.

    Same `errors="replace"` rationale as the streaming reader: comfy-cli's output
    is forced to UTF-8, so a bad byte means truncation or corruption — not a
    reason to fail a transfer that completed.
    """
    patched_async_run(b'{"type": "envelope", "ok": true, "data": {"n": "\xff"}}\n')

    assert asyncio.run(server._run_comfy_async("model", "download")) == {"n": "�"}


def test_discover_maps_command_and_returns_data(patched_run):
    """discover wraps `comfy discover` and returns the envelope data verbatim.

    The default carries `--schemas-only` (see `test_discover_defaults_to_
    schemas_only` in test_discovery.py for why); `schemas_only=False` is the
    bare subcommand.
    """
    surface = {"commands": ["run", "env"], "error_codes": ["server_not_running"]}
    calls = patched_run(envelope(data=surface))

    assert server.discover(schemas_only=False) == surface
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


def test_run_workflow_stream_self_attributes_via_user_agent_env(patched_stream):
    """The streaming spawn carries the caller label too.

    `run_workflow` is the second partner-node path — a graph with partner-API
    nodes bills them wherever it runs — and `wait=True` takes the streaming
    spawn, a different site from `_run_comfy`. Both read `_comfy_env`; this is
    the guard that keeps them from drifting apart on the attribution.
    """
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.run_workflow("wf.json", wait=True))

    assert procs[0].env["COMFY_USER_AGENT"] == "comfy-mcp"


def test_run_workflow_stream_forces_utf8_env(patched_stream):
    """The streaming spawn path also forces UTF-8 for the Windows fix."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.run_workflow("wf.json", wait=True))

    assert procs[0].env["PYTHONUTF8"] == "1"
    assert procs[0].env["PYTHONIOENCODING"] == "utf-8"


def test_run_workflow_stream_decodes_utf8_and_survives_bad_bytes(monkeypatch):
    """Stream lines decode as UTF-8, and an undecodable byte degrades one line.

    The async pipes are binary, so the parent decodes each line itself rather
    than relying on the parent locale (cp1252 on Windows would mojibake or raise
    on the non-ASCII title below). ``errors="replace"`` is the deliberate half:
    a truncated multi-byte sequence must not raise ``UnicodeDecodeError`` out of
    the middle of a live run whose envelope is still to come.
    """
    ctx = _RecordingCtx()
    good = json.dumps({"type": "executing", "node": "1", "title": "Café ☕"}).encode()
    envelope_line = json.dumps(
        {"schema": "envelope/1", "type": "envelope", "ok": True, "data": {"n": 1}}
    ).encode()
    raw = good + b"\n" + b'{"type": "executing", "node": "\xff\xfe"}\n' + envelope_line

    procs: list[object] = []

    async def fake_exec(*cmd, stdout, stderr, env, limit=None, **kwargs):
        proc = _FakeProc(list(cmd), "", env=env, limit=limit)
        proc.stdout = stream_reader(raw, limit)
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

    result = asyncio.run(server._run_comfy_streaming("run", "wf.json", ctx=ctx))

    assert result == {"n": 1}  # the undecodable line did not abort the run
    assert any("Café ☕" in (c["message"] or "") for c in ctx.calls)


def test_streaming_reads_a_line_longer_than_the_reader_limit(monkeypatch):
    """An NDJSON event bigger than the StreamReader's buffer is still read whole.

    ``asyncio.StreamReader.readline`` raises ``ValueError`` as soon as one line
    exceeds ``limit``; the blocking ``Popen`` + text ``readline`` this path used
    to run had no such ceiling. A ``queued`` event carrying a large node manifest
    is exactly the shape that would hit it, so ``_readline_unbounded`` stitches
    the overrun chunks back together instead. Spawning with a deliberately tiny
    ``limit`` makes the overrun happen at test speed.
    """
    nodes = [{"node_id": str(i)} for i in range(400)]
    stream = (
        json.dumps({"schema": "event/1", "type": "queued", "nodes": nodes})
        + "\n"
        + json.dumps(
            {"schema": "envelope/1", "type": "envelope", "ok": True, "data": {"ok": 1}}
        )
        + "\n"
    )
    assert len(stream) > 4096  # the manifest line really does overrun the limit

    ctx = _RecordingCtx()

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        return _FakeProc(list(cmd), stream, env=env, limit=1024)

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

    result = asyncio.run(server._run_comfy_streaming("run", "wf.json", ctx=ctx))

    assert result == {"ok": 1}
    # The oversized line parsed, so the node manifest set the progress total.
    assert ctx.calls[0]["total"] == float(len(nodes))


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
    """A fake child whose stdout yields ``first_lines`` then blocks (forces a timeout).

    ``returncode`` starts None (the child is running) so the timeout handler's
    kill actually fires; carries no ``pid`` so the kill takes the ``proc.kill()``
    fallback rather than signalling a made-up group. See conftest's ``_FakeProc``.
    """

    def __init__(self, cmd, first_lines):
        self.cmd = cmd
        self._lines = [line.encode("utf-8") for line in first_lines]
        self.stdout = self  # the reader protocol lives on the proc itself
        self.stderr = stream_reader("")
        self.returncode = None
        self.killed = False

    async def readuntil(self, separator=b"\n"):
        if self._lines:
            return self._lines.pop(0)
        # Outlives the test's tiny timeout, so the envelope never arrives.
        await asyncio.sleep(1.0)
        raise asyncio.IncompleteReadError(b"", None)

    async def wait(self):
        self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True


def test_watch_job_times_out_returns_payload(monkeypatch):
    """A watch bounded by timeout_seconds returns a timed-out payload, not an error."""
    queued = json.dumps(
        {"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]}
    )
    procs: list[_BlockingProc] = []

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProc(cmd, [queued + "\n"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProc(cmd, [queued + "\n", executed + "\n"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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


def test_timeout_failure_returns_the_error_instead_of_raising_it(monkeypatch):
    """The shared timeout body RETURNS its `ComfyCliError` and logs one record.

    Returning is what lets each runner keep `raise ... from exc` at its own call
    site, so the traceback starts at the runner rather than inside a formatting
    helper. The two runners' reports are pinned individually elsewhere
    (`test_sync_timeout_*`, `test_download_model_legacy_timeout_*`); this pins
    the body they share.
    """
    logged: list[dict] = []
    monkeypatch.setattr(
        failure_log,
        "_log_failure",
        lambda kind, args, **kwargs: logged.append(
            {"kind": kind, "args": args, **kwargs}
        ),
    )

    error = server._timeout_failure(
        [server.COMFY_BIN, "--json", "--where", "local", "discover"],
        ("discover",),
        60.0,
        "partial stdout",
        "partial err",
    )

    assert isinstance(error, server.ComfyCliError)  # returned, never raised
    assert error.timed_out is True
    message = str(error)
    assert "comfy-cli timed out after 60.0s" in message
    assert "stderr tail: partial err" in message
    assert "stdout tail: partial stdout" in message
    # One record, `timeout`-kinded, carrying both untruncated streams — the log
    # keeps a longer slice than the message does.
    assert logged == [
        {
            "kind": "timeout",
            "args": ("discover",),
            "message": message,
            "stdout": "partial stdout",
            "stderr": "partial err",
        }
    ]


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
        # Every bound this fake was handed, in order: [0] is the caller's own
        # timeout, [1] the post-kill drain's. Recorded so a test can pin that
        # the drain stays BOUNDED — an unbounded second `communicate()` is the
        # exact wedge `_DRAIN_TIMEOUT` exists to prevent, and a fake that merely
        # ignores the argument would never notice it going missing.
        self.timeouts: list = []

    def communicate(self, timeout=None):
        self._communicates += 1
        self.timeouts.append(timeout)
        if self._communicates == 1:
            raise self._exc
        return "drained stdout", "drained stderr"  # the post-kill drain

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
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
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: proc)
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
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: proc)
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
        def communicate(self, timeout=None):
            self._communicates += 1
            self.timeouts.append(timeout)
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
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: proc)
    monkeypatch.setattr(server.os, "killpg", lambda pgid, sig: None)

    with pytest.raises(server.ComfyCliError) as exc:
        server._run_comfy("update", "all", timeout=1800.0)

    msg = str(exc.value)
    assert "first chunk + more" in msg  # the drain's longer read, not `first`
    assert "traceback tail" in msg
    # And the drain that produced it stayed bounded: a post-kill `communicate()`
    # with no deadline is what `_DRAIN_TIMEOUT` exists to stop, since a
    # descendant that survived SIGKILL can hold the pipes open indefinitely.
    assert proc.timeouts == [1800.0, server._DRAIN_TIMEOUT]


def test_drain_non_timeout_failure_falls_back_to_the_first_capture(monkeypatch):
    """A drain that dies on a decode error has nothing better than the first read."""

    class _DecodeFailProc(_GroupKillProc):
        def communicate(self, timeout=None):
            self._communicates += 1
            self.timeouts.append(timeout)
            if self._communicates == 1:
                raise self._exc
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    proc = _DecodeFailProc(
        [server.COMFY_BIN, "update", "all"],
        _timeout(stderr="partial trace", stdout="partial out"),
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: proc)
    monkeypatch.setattr(server.os, "killpg", lambda pgid, sig: None)

    with pytest.raises(server.ComfyCliError) as exc:
        server._run_comfy("update", "all", timeout=1800.0)

    assert "partial trace" in str(exc.value)
    assert proc.timeouts == [1800.0, server._DRAIN_TIMEOUT]  # still a bounded drain


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


class _BlockingProcWithStderr(_BlockingProc):
    """A blocking child fake that also carries buffered stderr (a crashed traceback)."""

    def __init__(self, cmd, first_lines, stderr_text):
        super().__init__(cmd, first_lines)
        self.stderr = stream_reader(stderr_text)


def test_streaming_timeout_surfaces_stdout_and_stderr_tails(monkeypatch):
    """A raising streaming timeout appends the NDJSON stdout tail and the child's stderr tail."""
    queued = json.dumps({"type": "queued", "nodes": [{"node_id": "1"}]})
    procs: list[_BlockingProcWithStderr] = []

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProcWithStderr(
            cmd, [queued + "\n"], "Traceback ...\nUnicodeEncodeError: boom"
        )
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProcWithStderr(cmd, noisy, "")
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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
        self._lines = [line.encode("utf-8") for line in stdout_lines]
        self.stdout = self  # the reader protocol lives on the proc
        self.stderr = self  # read lives on the proc
        self.returncode = None
        self.killed = False

    async def readuntil(self, separator=b"\n"):
        if self._lines:
            return self._lines.pop(0)
        raise asyncio.IncompleteReadError(b"", None)  # EOF -> _pump breaks

    async def read(self, size=-1):
        # Outlives the tiny timeout; suspends `_read` at `await stderr_future`.
        await asyncio.sleep(1.0)
        return b""

    async def wait(self):
        self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        self._alive = False


def test_streaming_timeout_stderr_cancel_still_raises(monkeypatch):
    """A timeout that cancels the stderr read must still raise ComfyCliError, not CancelledError."""
    queued = json.dumps({"type": "queued", "nodes": [{"node_id": "1"}]})

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        return _StderrBlockingProc(cmd, [queued + "\n"])

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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
    assert _restart() == {"pid": 7}

    assert len(calls) == 2  # exactly stop then launch, nothing else
    assert calls[0]["cmd"][4:] == ["stop"]
    assert calls[1]["cmd"][4:] == ["launch", "--background"]


def test_restart_comfyui_forwards_extra_args_to_launch(patched_run):
    """extra_args ride the launch step after the `--` separator, not the stop."""
    calls = patched_run(envelope(data={}))

    _restart(["--port", "8189"])

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
    monkeypatch.setattr(server, "_launch_comfyui_sync", fake_launch)

    assert _restart(["--cpu"]) == {"pid": 1}
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
        server,
        "_launch_comfyui_sync",
        lambda extra_args=None: launched.append(extra_args),
    )

    with pytest.raises(server.ComfyCliError, match="permission_denied"):
        _restart()
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

    monkeypatch.setattr(server, "_launch_comfyui_sync", fake_launch)

    assert _restart(["--cpu"]) == {"pid": 3}

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
        server,
        "_launch_comfyui_sync",
        lambda extra_args=None: launched.append(extra_args),
    )

    with pytest.raises(server.ComfyCliError, match="timed out"):
        _restart()
    assert launched == []


def test_restart_comfyui_reraises_unrelated_plain_stop_failure(
    patched_plain_run, monkeypatch
):
    """A plain non-zero stop whose text ISN'T the benign phrase still aborts."""
    patched_plain_run(1, stderr="Failed to kill pid 7: operation not permitted")
    launched: list = []
    monkeypatch.setattr(
        server,
        "_launch_comfyui_sync",
        lambda extra_args=None: launched.append(extra_args),
    )

    with pytest.raises(server.ComfyCliError, match="operation not permitted"):
        _restart()
    assert launched == []


def test_restart_comfyui_explains_port_clash_after_nothing_to_stop(monkeypatch):
    """Nothing to stop + port taken = a running server comfy-cli never launched."""

    def fake_stop():
        raise server.ComfyCliError("No ComfyUI is running in the background.")

    def fake_launch(extra_args=None):
        raise server.ComfyCliError(
            "comfy-cli returned no JSON (exit 1). stderr: The 8188 port is "
            "already in use. | stdout: <empty>",
            no_envelope=True,
            returncode=1,
        )

    monkeypatch.setattr(server, "stop_comfyui", fake_stop)
    monkeypatch.setattr(server, "_launch_comfyui_sync", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart()

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

    def fake_launch(extra_args=None):
        raise server.ComfyCliError("The 8188 port is already in use.")

    monkeypatch.setattr(server, "_launch_comfyui_sync", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart()

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

    def fake_launch(extra_args=None):
        raise server.ComfyCliError("ComfyUI exited during startup: missing torch")

    monkeypatch.setattr(server, "_launch_comfyui_sync", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart()

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

    def fake_launch(extra_args=None):
        raise server.ComfyCliError("Cannot load model: the file is already in use")

    monkeypatch.setattr(server, "_launch_comfyui_sync", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart()

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

    def fake_launch(extra_args=None):
        raise server.ComfyCliError("The 8189 port is already in use.")

    monkeypatch.setattr(server, "stop_comfyui", fake_stop)
    monkeypatch.setattr(server, "_launch_comfyui_sync", fake_launch)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(["--port", str(server._ALT_PORT_SUGGESTION)])

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
        server,
        "_launch_comfyui_sync",
        lambda extra_args=None: {"pid": 42, "port": 8188},
    )

    assert _restart() == {"pid": 42, "port": 8188}


# --- the lifecycle trio is serialized against itself ------------------------
#
# `launch` / `stop` / `restart` all drive comfy-cli's ONE recorded pid and the one
# ComfyUI port. Being dispatched onto a worker thread does not order them — both
# `asyncio.to_thread` and MCPServer's sync-tool pool have many workers — so
# `_LIFECYCLE_LOCK` has to, and a `stop` slipping into the gap between a restart's
# stop and its launch is exactly the interleaving that leaves a server comfy-cli
# can no longer stop.


@pytest.fixture
def _lifecycle_lock_reset():
    """Fail loudly rather than leak a held lifecycle lock into later tests."""
    yield
    free = server._LIFECYCLE_LOCK.acquire(blocking=False)
    if free:
        server._LIFECYCLE_LOCK.release()
    assert free, "a lifecycle call left `_LIFECYCLE_LOCK` held"


def _held_lifecycle_lock():
    """Simulate another thread mid-launch, from a thread that is not this one."""
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        with server._LIFECYCLE_LOCK:
            acquired.set()
            release.wait(5)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert acquired.wait(5)
    return release, worker


@pytest.mark.parametrize(
    ("call", "verb"),
    [
        (lambda: _launch(["--port", "8189"]), "start"),
        (lambda: server.stop_comfyui(), "stop"),
        (lambda: _restart(), "restart"),
    ],
    ids=["launch", "stop", "restart"],
)
def test_lifecycle_call_is_refused_while_another_is_in_flight(
    patched_run, call, verb, _lifecycle_lock_reset
):
    """Refused immediately — never queued behind a subprocess the caller can't see."""
    calls = patched_run(envelope(data={}))
    release, worker = _held_lifecycle_lock()
    try:
        with pytest.raises(server.ComfyCliError) as excinfo:
            call()
    finally:
        release.set()
        worker.join(5)

    message = str(excinfo.value)
    assert f"cannot {verb} the local ComfyUI right now" in message
    assert "already in flight" in message
    assert calls == []  # in particular, restart did not stop the running server


def test_restart_holds_one_slot_across_both_halves(
    patched_run, monkeypatch, _lifecycle_lock_reset
):
    """A `stop_comfyui` arriving mid-restart is refused, not slipped into the gap."""
    patched_run(envelope(data={}))
    from_other_thread: list = []
    # Bound BEFORE the patch below, so the concurrent attempt exercises the real
    # `stop_comfyui` (and therefore the real lock) rather than re-entering the
    # stand-in that stands in for the restart's own stop half.
    real_stop = server.stop_comfyui

    def stop_from_another_thread():
        def attempt():
            try:
                real_stop()
            except server.ComfyCliError as exc:  # what a real concurrent call sees
                from_other_thread.append(exc)
            else:
                from_other_thread.append(None)

        thread = threading.Thread(target=attempt, daemon=True)
        thread.start()
        thread.join(5)
        return {"stopped": True}

    # The restart's own stop half; it runs while the slot is held.
    monkeypatch.setattr(server, "stop_comfyui", stop_from_another_thread)
    monkeypatch.setattr(server, "_launch_comfyui_sync", lambda extra: {"pid": 7})

    assert _restart() == {"pid": 7}

    assert len(from_other_thread) == 1
    assert "already in flight" in str(from_other_thread[0])


def test_lifecycle_lock_is_released_after_a_failed_launch(
    patched_plain_run, _lifecycle_lock_reset
):
    """A failure must not wedge the slot: the next call has to be able to run."""
    patched_plain_run(1, stderr="Address already in use: port 8188")

    with pytest.raises(server.ComfyCliError):
        _launch()

    assert server._LIFECYCLE_LOCK.acquire(blocking=False)
    server._LIFECYCLE_LOCK.release()


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

    MCPServer dispatches sync tools onto a worker thread pool, so two calls really
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
    assert content.mime_type == "image/png"
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
    assert images[0].to_image_content().mime_type == "image/jpeg"


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
    """Fake async child emitting ``lines`` (incl. the envelope) then lingering.

    Once the canned lines are exhausted ``stdout`` never EOFs and ``wait()``
    never returns — modeling comfy-cli outliving its own ``--json-stream``
    envelope under a pipe. ``stderr.read`` blocks the same way, so a test also
    proves the envelope path never awaits stderr. The block is bounded (10s) only
    so a buggy test can't hang the suite; the real exit is ``kill()``.

    Carries no ``pid``, so ``server._kill_proc_tree_async`` takes its
    ``AttributeError`` fallback to ``proc.kill()`` instead of signalling a
    made-up process group — same reasoning as conftest's ``_FakeProc``.
    """

    def __init__(self, cmd, lines):
        self.cmd = cmd
        self._lines = [line.encode("utf-8") for line in lines]
        self.stdout = self  # the reader protocol lives on the proc itself
        self.stderr = self  # read() blocks the same way -> proves we don't await it
        # None until reaped, mirroring a real child: while it lingers past its
        # envelope its returncode is unknown, and _unwrap_envelope must tolerate
        # that (it ignores returncode whenever an envelope is present).
        self.returncode = None
        self.killed = False
        self._dead = asyncio.Event()

    async def _park(self):
        """Block until killed — bounded only so a buggy test can't hang the suite."""
        try:
            await asyncio.wait_for(self._dead.wait(), 10.0)
        except (asyncio.TimeoutError, TimeoutError):
            pass

    async def readuntil(self, separator=b"\n"):
        if self._lines:
            return self._lines.pop(0)
        await self._park()  # stdout never EOFs on its own
        raise asyncio.IncompleteReadError(b"", None)

    async def read(self, size=-1):
        await self._park()  # stderr never EOFs while the child lives
        return b""

    async def wait(self):
        await self._park()  # a child that lingers past its envelope
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9  # reaped by SIGKILL
        self._dead.set()


def _lingering_exec(procs, stream_lines):
    """An ``create_subprocess_exec`` stand-in minting a ``_LingeringProc`` per call."""

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _LingeringProc(list(cmd), list(stream_lines))
        procs.append(proc)
        return proc

    return fake_exec


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
        server.asyncio,
        "create_subprocess_exec",
        _lingering_exec(procs, _ENVELOPE_THEN_LINGER),
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
        server.asyncio,
        "create_subprocess_exec",
        _lingering_exec(procs, _ERROR_ENVELOPE_THEN_LINGER),
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
        server.asyncio,
        "create_subprocess_exec",
        _lingering_exec(procs, _ENVELOPE_THEN_LINGER),
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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProc(cmd, [queued + "\n"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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
        server.asyncio,
        "create_subprocess_exec",
        _lingering_exec(procs, _ENVELOPE_THEN_LINGER),
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
        server.asyncio,
        "create_subprocess_exec",
        _lingering_exec(procs, _SPURIOUS_ENVELOPE_THEN_REAL),
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
        self._lines = [line.encode("utf-8") for line in lines]
        self.stdout = self
        self.stderr = stream_reader(stderr_text)
        self.returncode = 1  # already exited by the time we reap
        self.killed = False

    async def readuntil(self, separator=b"\n"):
        if self._lines:
            return self._lines.pop(0)
        raise asyncio.IncompleteReadError(b"", None)

    async def wait(self):
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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        return _ExitedErrorProc(cmd, lines, stderr_text)

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server, "_POST_ENVELOPE_REAP_GRACE", 0.1)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server.run_workflow("wf.json", wait=True, timeout_seconds=30.0))

    msg = str(excinfo.value)
    assert "execution_error" in msg  # the envelope's error code
    assert "kaboom" in msg  # the stderr fallback filled the empty message


# --- stderr drain is bounded in both memory and time --------------------------


class _ChunkedStream:
    """Fake async pipe handing back ``data`` in fixed ``chunk`` slices, then EOF.

    Models a real pipe delivering stderr in many small reads (ignoring the
    requested size), so a test can prove `_drain_capped_async` keeps reading to
    EOF and retains the tail across chunk boundaries — not just within one read.
    """

    def __init__(self, data, chunk):
        self._data = data
        self._chunk = chunk
        self._pos = 0

    async def read(self, size=-1):  # chunked regardless of the requested size
        piece = self._data[self._pos : self._pos + self._chunk]
        self._pos += len(piece)
        return piece


def test_drain_capped_async_retains_only_the_tail_across_chunks():
    """`_drain_capped_async` drains to EOF but keeps at most ``limit`` trailing bytes.

    A verbose child must be fully drained (so it can't wedge on a full stderr
    pipe) without letting its output drive unbounded allocation here — only the
    tail, where the actual error/traceback lands, is retained.
    """
    data = b"".join(f"line{i}\n".encode() for i in range(1000))  # ~7 KB, many chunks
    out = asyncio.run(server._drain_capped_async(_ChunkedStream(data, chunk=64), 100))
    assert out == data[-100:].decode()
    assert len(out) == 100
    # Under the cap: returned whole. Empty: drains to "".
    drain = server._drain_capped_async
    assert asyncio.run(drain(_ChunkedStream(b"short", chunk=64), 100)) == "short"
    assert asyncio.run(drain(_ChunkedStream(b"", chunk=64), 100)) == ""


def test_drain_capped_async_decodes_once_at_the_end():
    """A multi-byte character split across a chunk boundary survives the drain.

    The cap slices BYTES, so decoding per chunk would turn a UTF-8 sequence
    straddling a read into replacement characters — which is why the decode
    happens once, after the tail is assembled.
    """
    data = "ünïcödé ☕ traceback\n".encode()
    # A chunk size that lands mid-sequence on the multi-byte characters.
    out = asyncio.run(server._drain_capped_async(_ChunkedStream(data, chunk=3), 4096))
    assert out == data.decode()
    assert "�" not in out


class _StderrNeverEOFProc:
    """Envelope-then-linger child whose stderr pipe never EOFs, even after kill.

    Models comfy-cli leaving a descendant that inherited the stderr write fd:
    ``kill()`` reaps the direct child (unblocking the stdout read / ``wait``) but
    the stderr ``read()`` never returns. Cleanup must detach the parked reader
    instead of hanging the tool call forever.
    """

    def __init__(self, cmd, lines):
        self.cmd = cmd
        self._lines = [line.encode("utf-8") for line in lines]
        self.stdout = self
        self.stderr = self
        self.returncode = None
        self.killed = False
        self._child_dead = asyncio.Event()  # set by kill(): the direct child
        self._stderr_eof = asyncio.Event()  # NEVER set: descendant holds the fd

    async def _park(self, event):
        """Block on ``event`` — bounded only so a buggy test can't hang the suite."""
        try:
            await asyncio.wait_for(event.wait(), 10.0)
        except (asyncio.TimeoutError, TimeoutError):
            pass

    async def readuntil(self, separator=b"\n"):
        if self._lines:
            return self._lines.pop(0)
        await self._park(self._child_dead)
        raise asyncio.IncompleteReadError(b"", None)

    async def read(self, size=-1):
        await self._park(self._stderr_eof)
        return b""

    async def wait(self):
        await self._park(self._child_dead)
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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
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


# --- the streaming path never blocks the event loop ---------------------------


def test_streaming_run_keeps_the_event_loop_responsive(monkeypatch):
    """A live stream must not stall other coroutines on the same loop.

    This is the property `_run_comfy_streaming` exists to hold and the one a
    result-only test cannot see: an implementation that spawned with
    `subprocess.Popen` and read the pipes with blocking `readline` would return
    the exact same envelope while the loop sat frozen for the life of the child.

    So the assertion is on the HEARTBEAT, not the payload. A companion coroutine
    ticks every millisecond for the whole run; if the stream ever monopolizes the
    loop the tick count collapses and the largest gap between ticks approaches
    the run's duration. Both are checked, because either alone is foolable — a
    high tick count could come from one long stall plus a fast burst.
    """
    ticks: list[float] = []
    step = 0.02  # per-line delay: the run lasts long enough to sample the loop

    class _PacedProc:
        """Emits its canned lines one `step` apart, yielding to the loop between."""

        def __init__(self, cmd, lines):
            self.cmd = cmd
            self._lines = [line.encode("utf-8") for line in lines]
            self.stdout = self
            self.stderr = stream_reader("")
            self.returncode = None
            self.killed = False

        async def readuntil(self, separator=b"\n"):
            await asyncio.sleep(step)  # the child is "thinking" between events
            if self._lines:
                return self._lines.pop(0)
            raise asyncio.IncompleteReadError(b"", None)

        async def wait(self):
            self.returncode = 0
            return self.returncode

        def kill(self):
            self.killed = True

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        return _PacedProc(list(cmd), _OK_STREAM.splitlines(keepends=True))

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

    async def _heartbeat(stop: asyncio.Event):
        while not stop.is_set():
            ticks.append(time.monotonic())
            await asyncio.sleep(0.001)

    async def _drive():
        stop = asyncio.Event()
        beat = asyncio.ensure_future(_heartbeat(stop))
        started = time.monotonic()
        try:
            result = await server.run_workflow("wf.json", wait=True, timeout_seconds=30)
        finally:
            stop.set()
            await beat
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(_drive())

    assert result == {"outputs": ["/x.png"]}
    # The run really did span several event lines rather than resolving instantly.
    assert elapsed >= 3 * step
    # The other coroutine made progress THROUGHOUT: many ticks, and no single gap
    # anywhere near the run's length. A blocking implementation parks the loop for
    # the whole run, which shows up as a gap of roughly `elapsed`.
    assert len(ticks) > 10, f"only {len(ticks)} ticks in {elapsed:.3f}s"
    largest_gap = max(b - a for a, b in zip(ticks, ticks[1:]))
    assert largest_gap < elapsed / 2, f"loop stalled for {largest_gap:.3f}s"


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
