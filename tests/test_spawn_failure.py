"""A comfy-cli child that never STARTS still fails as a ``ComfyCliError``.

Each of the four spawn sites (``_run_comfy_raw``, ``_run_comfy_async``,
``_run_comfy_streaming``, ``_start_login``) calls its constructor OUTSIDE the
``try`` that guards the rest of the run — everything after a successful spawn
already has a handler, and a spawn that raised has no child to reap. An
exception from the constructor itself therefore used to escape unconverted: no
``ComfyCliError``, no failure-log line, just an ``OSError`` surfacing as an
internal error.

Reachable because MCP tool arguments cross the wire uncapped for the free-form
values — ``search_models``'s ``query``, a ``run_template`` param — and a big
enough one trips the OS limit on ``execve`` (measured on macOS: a ~2 MiB
argument raises ``OSError: [Errno 7] Argument list too long``). The
pre-spawn guards (``_reject_nul``, ``_guard_arg_len``) still produce the better,
argument-NAMING error first; ``_spawn_failure`` is the backstop underneath them,
so what lands here is what those cannot see.

The unencodable-argument tests drive the REAL refusal rather than injecting the
exception: the conftest fakes render argv with :func:`os.fsencode` exactly as a
POSIX spawn does (``_encode_argv_like_posix``), so a lone surrogate raises
``UnicodeEncodeError`` from the fake spawn where the real one would. A test that
injected the exception would only assert its own premise. Those calls are made
in-process on purpose: the MCP SDK's wire parser rejects a ``\\ud800`` escape as
invalid JSON before a tool ever runs, so a direct call is exactly the layer
these exercise — and the same branch is genuinely reachable over the wire on a
host whose filesystem encoding is not UTF-8, where ordinary non-ASCII in a
free-form value fails ``os.fsencode`` too.
"""

from __future__ import annotations

import asyncio
import errno
import json

import pytest
from conftest import _OK_STREAM, _RecordingCtx

from comfy_mcp import failure_log, server

# Long enough that `_cmd_for_message`'s `_MAX_ERROR_FIELD_CHARS` head slice
# cannot carry the whole thing into the message — which is the property the
# oversized-argument tests assert. The marker is distinctive so `in` is a real
# check rather than an accident of common substrings.
_OVERSIZED = "SPAWN-FAILURE-MARKER-" * 200

# A lone high surrogate: valid in a Python `str`, and NOT in the DC80..DCFF
# range `surrogateescape` round-trips, so `os.fsencode` refuses it on any
# UTF-8 POSIX host.
_UNENCODABLE = "\ud800abc"

_E2BIG = OSError(errno.E2BIG, "Argument list too long")


@pytest.fixture
def log_path(monkeypatch, tmp_path):
    """Enable the opt-in failure log at a path under ``tmp_path``.

    Patched on the OWNING module (``failure_log``), per AGENTS.md — patching the
    re-export-free ``server`` name would raise instead of silently testing
    nothing.
    """
    path = tmp_path / "state" / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(path))
    return path


@pytest.fixture
def fresh_login(monkeypatch):
    """Reset ``auth_login``'s parked-child globals, as ``test_auth_login`` does.

    The tool caches a live child in module state; a spawn-failure test must not
    inherit (or leave behind) one from another test.
    """
    monkeypatch.setattr(server, "_login_child", None, raising=False)
    monkeypatch.setattr(server, "_login_lock", None, raising=False)
    monkeypatch.setattr(server, "_login_lock_loop", None, raising=False)


# --- the plain `--json` path (`_run_comfy_raw` / `subprocess.Popen`) ---------


def test_oversized_argv_is_a_comfy_cli_error_not_an_oserror(patched_run):
    """E2BIG from the spawn comes back as the tool error, naming the OS limit."""
    patched_run(raises=_E2BIG)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.search_models(query=_OVERSIZED)

    message = str(excinfo.value)
    assert "could not start comfy-cli" in message
    assert "argument list and environment exceed" in message
    # The rendered argv is clipped, so the oversized value cannot come back whole
    # — echoing a megabyte-long "query" is the same denial-of-legibility the
    # length guards refuse to commit.
    assert _OVERSIZED not in message
    assert "models search" in message  # ...but WHICH call still reads clearly


def test_oversized_argv_records_a_spawn_failed_entry(patched_run, log_path):
    """The failure log gets one ``spawn_failed`` line with no exit code."""
    patched_run(raises=_E2BIG)

    with pytest.raises(server.ComfyCliError):
        server.search_models(query=_OVERSIZED)

    (entry,) = [
        json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
    ]
    assert entry["kind"] == "spawn_failed"
    assert entry["exit_code"] is None  # there is no child to have reported one
    assert entry["args"][:2] == ["models", "search"]
    assert "argument list and environment exceed" in entry["message"]


def test_embedded_nul_valueerror_is_converted(patched_run):
    """``subprocess``'s bare ``ValueError`` is named as the embedded NUL it is.

    The backstop under :func:`argv._reject_nul`, which normally refuses a NUL
    first and names the argument — this covers a value shape it does not reach.
    """
    patched_run(raises=ValueError("embedded null byte"))

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.search_models(query="whatever")

    assert "embedded NUL" in str(excinfo.value)


def test_vanished_binary_is_converted_with_its_strerror(patched_run):
    """A binary that disappears after ``_require_comfy_bin`` reports generically."""
    patched_run(raises=FileNotFoundError(errno.ENOENT, "No such file or directory"))

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.search_models(query="whatever")

    message = str(excinfo.value)
    assert "could not start comfy-cli: No such file or directory" in message
    # NOT diagnosed as the argument-list limit: only E2BIG earns that wording.
    assert "argument list" not in message


def test_unencodable_argument_is_converted_not_raised(patched_run):
    """A lone surrogate fails as a tool error, naming the filesystem encoding.

    ``UnicodeEncodeError`` is a ``ValueError`` SUBCLASS, so this also pins the
    isinstance ORDER inside ``_spawn_failure``: tested after it, the same input
    would come back described as an embedded NUL.
    """
    patched_run()

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.search_models(query=_UNENCODABLE)

    message = str(excinfo.value)
    assert "cannot be encoded with this system's filesystem encoding" in message
    assert "embedded NUL" not in message


def test_unencodable_argument_message_stays_bounded(patched_run):
    """A long unencodable value is clipped out of the message like any other."""
    patched_run()

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.search_models(query=_OVERSIZED + _UNENCODABLE)

    assert _OVERSIZED not in str(excinfo.value)


# --- the plain-JSON async path (`_run_comfy_async`) --------------------------


def test_async_runner_converts_a_spawn_failure(patched_async_run):
    """The cancellable twin converts identically — same helper, same wording."""
    patched_async_run(raises=_E2BIG)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server._run_comfy_async("model", "download", "--url", "u"))

    assert "argument list and environment exceed" in str(excinfo.value)


def test_async_runner_converts_an_unencodable_argument(patched_async_run):
    """Same real ``os.fsencode`` refusal as the sync path, on the async spawn."""
    patched_async_run()

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server._run_comfy_async("model", "download", "--url", _UNENCODABLE))

    assert "cannot be encoded with this system's filesystem encoding" in str(
        excinfo.value
    )


# --- the streaming path (`_run_comfy_streaming`) -----------------------------


def test_streaming_runner_converts_a_spawn_failure(patched_stream):
    """``run_workflow(wait=True)`` reports a failed spawn as the tool error."""
    patched_stream(_OK_STREAM, raises=_E2BIG)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server.run_workflow("wf.json", wait=True, ctx=_RecordingCtx()))

    assert "argument list and environment exceed" in str(excinfo.value)


def test_streaming_runner_converts_an_unencodable_argument(patched_stream):
    """The unencodable branch on the streaming spawn, driven at the runner.

    Deliberately NOT through a tool: no streaming tool forwards a raw free-form
    string to argv today. ``run_workflow`` / ``watch_job`` take only path- and
    id-shaped values, which ``_guard_arg_len`` / ``_guard_prompt_id`` already
    refuse first and name; ``run_template``'s params are ``json.dumps``-encoded
    with the default ``ensure_ascii=True``, which escapes an unencodable
    character to a pure-ASCII ``\\uXXXX`` before it ever reaches the spawn. The
    wrap still has to hold for whatever a later tool forwards, so the runner is
    the honest place to pin it.
    """
    patched_stream(_OK_STREAM)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server._run_comfy_streaming("jobs", "watch", _UNENCODABLE))

    assert "cannot be encoded with this system's filesystem encoding" in str(
        excinfo.value
    )


# --- the login path (`_start_login`) ----------------------------------------


def test_login_spawn_failure_is_converted(patched_stream, fresh_login):
    """Constants-only argv, but the environment budget and the binary can still fail.

    ``auth_login`` passes no caller value at all, so this covers the two ways a
    constant argv still refuses to start: an environment that pushes the total
    over ``ARG_MAX``, and the binary vanishing between ``_require_comfy_bin`` and
    the spawn.
    """
    patched_stream("", raises=_E2BIG)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server.auth_login())

    message = str(excinfo.value)
    assert "argument list and environment exceed" in message
    assert "cloud login" in message  # the rendered argv names the verb
    assert server._login_child is None  # nothing parked: there is no child
