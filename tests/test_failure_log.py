"""Tests for the opt-in local rotating failure log (``COMFY_LOCAL_MCP_DEBUG_LOG``).

The log exists to give a tester a durable, zippable diagnostic trail for
comfy-cli failures, so these tests hold its three defining properties:

1. **Opt-in.** With the env var unset there are ZERO filesystem effects — no
   directory, no handler, no file — even when comfy-cli fails.
2. **Failure-only.** A successful call writes nothing, and neither does a
   pre-flight validation raise (which never spawned comfy-cli at all).
3. **Structured + scrubbed.** Every line is pure JSON with the failure's
   ``kind`` / ``args`` / ``exit_code`` / ``error_code`` / stream tails, and a URL
   argument is logged with its userinfo masked and its query string dropped.

Like the rest of the suite these mock comfy-cli; nothing here needs a real
``comfy`` binary.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading

import pytest
from conftest import _FakeProc

from comfy_local_mcp import failure_log, server, tcc, textutil

# A failing envelope/1 result, the most common recorded failure.
_ERROR_ENVELOPE = json.dumps(
    {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": False,
        "error": {"code": "credential_missing", "message": "no API key configured"},
    }
)


@pytest.fixture
def log_path(monkeypatch, tmp_path):
    """Enable the failure log at a path under ``tmp_path`` and return that path.

    The parent directory deliberately does NOT exist: creating it lazily, on the
    first write, is part of the contract.
    """
    path = tmp_path / "state" / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(path))
    return path


@pytest.fixture
def fake_comfy(patched_run):
    """``setup(stdout=…, stderr=…, returncode=…, raises=…)`` for one canned result.

    The shared ``patched_run`` (conftest) with this file's own default of an
    EMPTY stdout — these tests are about what a *failing* call records, and "no
    JSON at all" is one of the failures under test.
    """

    def setup(stdout="", stderr="", returncode=0, raises=None):
        patched_run(stdout, stderr=stderr, returncode=returncode, raises=raises)

    return setup


def _entries(path) -> list[dict]:
    """Every JSONL record written to ``path``, parsed.

    Parsing each line as JSON is itself an assertion: the log must be
    ``jq``-able, so a level/timestamp prefix leaking in would fail here.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- the opt-in switch -------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   ", "0"])
def test_env_var_unset_empty_or_zero_disables(value):
    """Unset, blank, or an explicit ``0`` all mean disabled."""
    assert failure_log._resolve_failure_log_path(value) is None


def test_env_var_one_selects_the_per_os_default(monkeypatch, tmp_path):
    """``1`` means "on, at the default path" — it is never taken as a literal path."""
    default = str(tmp_path / "default.jsonl")
    monkeypatch.setattr(failure_log, "_default_failure_log_path", lambda: default)

    assert failure_log._resolve_failure_log_path("1") == default


def test_env_var_any_other_value_is_the_log_path(tmp_path):
    """Any other value IS the path, so a tester can point it at a scratch file."""
    target = str(tmp_path / "x.jsonl")

    assert failure_log._resolve_failure_log_path(target) == target


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("darwin", ("Library", "Application Support", "comfy-local-mcp")),
        ("win32", ("AppData", "Local", "comfy-local-mcp")),
        ("linux", (".config", "comfy-local-mcp")),
    ],
)
def test_default_path_mirrors_comfy_cli_app_dirs(monkeypatch, platform, expected):
    """The default path follows comfy-cli's per-OS local-state convention."""
    monkeypatch.setattr(failure_log.sys, "platform", platform)
    monkeypatch.setattr(failure_log.os.path, "expanduser", lambda _: "/home/u")

    path = failure_log._default_failure_log_path()

    assert path.endswith("failures.jsonl")
    for segment in expected:
        assert segment in path
    # Never inside the ComfyUI workspace: resolving that needs a working
    # comfy-cli, which is precisely what has failed when this log matters.
    assert path.startswith("/home/u")


def test_disabled_by_default_writes_nothing_and_creates_no_dir(
    monkeypatch, tmp_path, fake_comfy
):
    """With the log off, a failing call leaves ZERO filesystem traces.

    The default path is pointed into ``tmp_path`` so "nothing was created" is
    actually observable — otherwise a stray write would land in the real user
    app-support directory and the assertion could not see it.
    """
    default = tmp_path / "default" / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_default_failure_log_path", lambda: str(default))
    # The autouse `_isolate_failure_log` fixture already pins this to the
    # disabled state; restate it here so the test reads as its own premise.
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", None)
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("run", "wf.json")

    assert not default.exists()
    assert not default.parent.exists()
    # No handler was opened either — the memo is untouched.
    assert failure_log._failure_handler_path is None


def test_enabled_path_from_env_var_value_is_written(
    monkeypatch, tmp_path, fake_comfy, capsys
):
    """``COMFY_LOCAL_MCP_DEBUG_LOG=<path>`` writes to exactly that file."""
    target = tmp_path / "nested" / "chosen.jsonl"
    monkeypatch.setattr(
        failure_log,
        "_FAILURE_LOG_PATH",
        failure_log._resolve_failure_log_path(str(target)),
    )
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("run", "wf.json")

    assert len(_entries(target)) == 1
    # This is an MCP *stdio* server: stdout is the protocol transport, so the
    # dedicated logger must never reach a default stream handler.
    assert capsys.readouterr().out == ""


# --- what a record contains --------------------------------------------------


def test_error_envelope_is_recorded(log_path, fake_comfy):
    """An error envelope logs one line with its code, argv, and empty-stream markers."""
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("run", "wf.json")

    (entry,) = _entries(log_path)
    assert entry["kind"] == "error_envelope"
    assert entry["args"] == ["run", "wf.json"]
    assert entry["error_code"] == "credential_missing"
    assert entry["exit_code"] == 0
    assert entry["streaming"] is False
    # `message` is kept alongside the structured fields so QA can correlate a
    # line with what the MCP client actually displayed.
    assert entry["message"] == str(excinfo.value)
    assert entry["stderr_tail"] == "<empty>"
    assert "credential_missing" in entry["stdout_tail"]
    assert entry["ts"].endswith("+00:00")


@pytest.mark.parametrize("blank", ["", "   ", None, b""])
def test_blank_streams_are_marked_not_dropped(blank):
    """A blank/absent capture records ``<empty>``, never an empty string.

    Same ``_stream_tail`` marker the error messages use, so a record can never
    show a stream that looks absent when it was actually truncated away.
    """
    assert (
        textutil._stream_tail(blank, failure_log._FAILURE_LOG_TAIL_CHARS) == "<empty>"
    )


def test_concurrent_first_failures_write_one_line_each(log_path, fake_comfy):
    """Two threads failing at once must not install duplicate handlers.

    The handler is built lazily on the first write, so concurrent first failures
    race on that setup — an interleaving that left both threads' handlers attached
    would duplicate every subsequent line for the life of the process.
    """
    fake_comfy(stdout=_ERROR_ENVELOPE)
    barrier = threading.Barrier(4)

    def fail():
        barrier.wait()
        with pytest.raises(server.ComfyCliError):
            server._run_comfy("run", "wf.json")

    threads = [threading.Thread(target=fail) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(_entries(log_path)) == 4
    assert (
        len(failure_log.logging.getLogger(failure_log._FAILURE_LOGGER_NAME).handlers)
        == 1
    )


def test_no_json_exit_is_recorded_with_both_tails(log_path, fake_comfy):
    """A comfy-cli that dies before emitting JSON logs both streams and its exit code."""
    fake_comfy(stdout="usage: comfy [OPTIONS]", stderr="Traceback: boom", returncode=2)

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("jobs", "status", "abc")

    (entry,) = _entries(log_path)
    assert entry["kind"] == "no_json"
    assert entry["exit_code"] == 2
    assert entry["error_code"] is None
    assert "usage: comfy" in entry["stdout_tail"]
    assert "Traceback: boom" in entry["stderr_tail"]


def test_timeout_is_recorded(log_path, fake_comfy):
    """A killed child logs ``timeout`` with no exit code (it never reported one)."""
    fake_comfy(
        raises=subprocess.TimeoutExpired(
            cmd=["comfy"], timeout=1.0, output=b"partial stdout", stderr=b"partial err"
        )
    )

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("run", "wf.json", timeout=1.0)

    (entry,) = _entries(log_path)
    assert entry["kind"] == "timeout"
    assert entry["exit_code"] is None
    assert "partial stdout" in entry["stdout_tail"]
    assert "partial err" in entry["stderr_tail"]


def test_binary_missing_is_recorded_with_no_args(monkeypatch, log_path):
    """No binary means no invocation, so ``args`` is empty by construction."""
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    monkeypatch.setattr(tcc, "_is_macos", lambda: False)

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("run", "wf.json")

    (entry,) = _entries(log_path)
    assert entry["kind"] == "binary_missing"
    assert entry["args"] == []
    assert "not found on PATH" in entry["message"]


def test_schema_mismatch_is_recorded(log_path, fake_comfy):
    """A future envelope major is refused loudly — and recorded as its own kind."""
    fake_comfy(
        stdout=json.dumps(
            {"schema": "envelope/2", "type": "envelope", "ok": True, "data": {}}
        )
    )

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("run", "wf.json")

    (entry,) = _entries(log_path)
    assert entry["kind"] == "schema_mismatch"
    assert "envelope/2" in entry["message"]


def test_tail_cap_is_larger_than_the_message_cap():
    """The log keeps MORE output than an error message can — its reason to exist."""
    assert failure_log._FAILURE_LOG_TAIL_CHARS > server._MAX_ERROR_FIELD_CHARS


def test_long_stream_is_tail_truncated_with_a_marker(log_path, fake_comfy):
    """An oversized capture is clipped to the tail and marked, never silently."""
    noise = "x" * (failure_log._FAILURE_LOG_TAIL_CHARS * 2)
    fake_comfy(stdout="no json here", stderr=noise + "THE-REAL-ERROR", returncode=1)

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("run", "wf.json")

    (entry,) = _entries(log_path)
    tail = entry["stderr_tail"]
    # The `...` marker is additive to the bound, exactly as in the error messages
    # `_stream_tail` was written for — the payload itself stays capped.
    assert len(tail) == failure_log._FAILURE_LOG_TAIL_CHARS + len("...")
    assert tail.startswith("...")  # truncation is visible
    assert tail.endswith("THE-REAL-ERROR")  # the tail, not the head, is kept


def test_streaming_failure_is_flagged(log_path, monkeypatch):
    """The ``streaming`` flag distinguishes the ``--json-stream`` spawn path."""
    procs: list[_FakeProc] = []

    def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
        proc = _FakeProc(cmd, _ERROR_ENVELOPE + "\n", env=env, encoding=encoding)
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    with pytest.raises(server.ComfyCliError):
        server.asyncio.run(server._run_comfy_streaming("run", "wf.json"))

    (entry,) = _entries(log_path)
    assert entry["kind"] == "error_envelope"
    assert entry["streaming"] is True


# --- failure-only ------------------------------------------------------------


def test_successful_call_logs_nothing(log_path, fake_comfy):
    """Failure-only: a healthy call must not leave a line (or a file) behind."""
    fake_comfy(
        stdout=json.dumps(
            {"schema": "envelope/1", "type": "envelope", "ok": True, "data": {"x": 1}}
        )
    )

    assert server._run_comfy("jobs", "status", "abc") == {"x": 1}

    assert not log_path.exists()


def test_preflight_validation_raise_logs_nothing(log_path, fake_comfy):
    """A guard that rejects an argument never spawned comfy-cli — not our gap.

    comfy-cli is stubbed to FAIL, so a spawn would certainly write a record: the
    empty log is evidence the leading-dash guard short-circuited before it.
    """
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError):
        server.get_execution_error("-x")

    assert not log_path.exists()


# --- redaction ---------------------------------------------------------------


def test_scrub_arg_masks_userinfo_and_strips_query(log_path, fake_comfy):
    """A URL arg is logged with userinfo masked and query/fragment dropped.

    Also covers the ``message`` field, which :func:`_unwrap_envelope` builds as
    ``comfy <args> failed …`` — i.e. it echoes the RAW argv, so scrubbing only
    the ``args`` list would leave the credential in the very next key.
    """
    url = "https://user:tok@host/x?token=abc#frag"
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("model", "download", "--url", url)

    (entry,) = _entries(log_path)
    assert entry["args"] == ["model", "download", "--url", "https://***@host/x"]
    assert entry["message"].startswith(
        "comfy model download --url https://***@host/x failed"
    )
    line = log_path.read_text()
    assert "tok" not in line
    assert "token=abc" not in line


def test_scrub_message_masks_urls_but_keeps_prose():
    """Only URL-shaped substrings are rewritten; the surrounding prose survives."""
    text = "comfy model download --url https://u:p@h/m.safetensors?token=s failed"
    assert failure_log._scrub_message(text) == (
        "comfy model download --url https://***@h/m.safetensors failed"
    )
    plain = "comfy run wf.json failed [bad_workflow]: node 3 has no input"
    assert failure_log._scrub_message(plain) == plain


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        # Non-URL args pass through verbatim — mangling a path or a subcommand
        # would defeat the log.
        ("run", "run"),
        ("/Users/me/wf.json", "/Users/me/wf.json"),
        ("user:pass@notaurl", "user:pass@notaurl"),
        # Query stripped even with no credential in it (mirrors comfy-cli's own
        # scrubber: a CivitAI URL carries its token as `?token=…`).
        ("https://h/m.safetensors?token=abc", "https://h/m.safetensors"),
        ("http://h/p#f", "http://h/p"),
        # A fragment BEFORE a query still cuts at the first delimiter.
        ("https://h/p#f?q=1", "https://h/p"),
        # Userinfo masked with no query present, and the scheme is case-blind.
        ("https://u:p@h/path", "https://***@h/path"),
        ("HTTPS://u:p@h/path", "HTTPS://***@h/path"),
        # A bare host URL is untouched.
        ("https://h/path", "https://h/path"),
    ],
)
def test_scrub_arg_cases(arg, expected):
    assert failure_log._scrub_arg(arg) == expected


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        # `_render_param_args` emits combined-flag tokens, so the URL is not at
        # the token's start — a start-anchored test would log the credential.
        (
            "--image_url=https://user:tok@host/x?token=abc",
            "--image_url=https://***@host/x",
        ),
        # …and `run_template` wraps the value in JSON, offsetting it further.
        (
            '--param=src={"https://u:p@h/i.png"}',
            '--param=src={"https://***@h/i.png"}',
        ),
    ],
)
def test_scrub_arg_finds_a_url_anywhere_in_the_token(arg, expected):
    """A URL mid-token is scrubbed, not just one the token starts with."""
    assert failure_log._scrub_arg(arg) == expected


def test_message_is_scrubbed_before_it_is_capped(log_path):
    """A URL straddling the message cap is masked, not cut into an unmasked half.

    Capping first would slice ``https://user:pass@host`` before its ``@``, and
    :func:`_redact_url` only masks userinfo it can still find in the netloc — so
    the surviving ``user:pass`` would land in the file verbatim.
    """
    cap = failure_log._FAILURE_LOG_MESSAGE_CHARS
    # Place the URL so the cap falls on its `@` — the whole userinfo is inside
    # the kept prefix but the `@` that marks it as userinfo is not, which is
    # precisely what defeats `_redact_url` when the cap runs first.
    message = "x" * (cap - 16) + "https://user:tok@host/m.safetensors?token=abc"

    failure_log._log_failure("error_envelope", ("run",), message=message)

    (entry,) = _entries(log_path)
    assert "user:tok" not in entry["message"]
    assert "token=abc" not in entry["message"]
    assert len(entry["message"]) == cap
    # The cap now lands inside the *masked* URL, so what it clips is `***@…`
    # rather than the raw `user:tok@…` the old cap-then-scrub order would leave.
    assert entry["message"].endswith("https://***@host")


def test_stream_tails_are_scrubbed_like_the_message(log_path):
    """comfy-cli echoing a credential URL to a stream must not persist it raw."""
    failure_log._log_failure(
        "no_json",
        ("model", "download"),
        stdout="fetching https://user:tok@host/m.safetensors?token=abc",
        stderr="failed: https://u:p@h/x?sig=deadbeef",
    )

    (entry,) = _entries(log_path)
    assert entry["stdout_tail"] == "fetching https://***@host/m.safetensors"
    assert entry["stderr_tail"] == "failed: https://***@h/x"
    assert "user:tok" not in log_path.read_text()
    assert "deadbeef" not in log_path.read_text()


def test_stream_tail_scrubs_a_url_straddling_the_tail_cut(log_path):
    """A URL clipped by the tail bound is dropped, never half-written.

    The tail keeps the END of a capture, so scrubbing after the clip would see a
    URL already shorn of its ``https://`` and skip it entirely.
    """
    limit = failure_log._FAILURE_LOG_TAIL_CHARS
    url = "https://user:tok@host/x"
    # Sized so the clip falls immediately after the URL's `https://` — the exact
    # case where a scrub-after-clip pass sees no scheme, matches nothing, and
    # writes the whole `user:tok@host/x` remainder to disk.
    stderr = "A" * 100 + url + "B" * (limit + 108 - 100 - len(url))
    assert len(textutil._stream_tail(stderr, limit)) == limit + 3  # it IS clipped

    failure_log._log_failure("no_json", ("run",), stderr=stderr)

    (entry,) = _entries(log_path)
    assert "user:tok" not in entry["stderr_tail"]
    assert "tok@host" not in entry["stderr_tail"]
    # Still bounded, and still marked as truncated.
    assert entry["stderr_tail"].startswith("...")
    assert len(entry["stderr_tail"]) == limit + 3


# --- best-effort + rotation --------------------------------------------------


def test_unwritable_log_never_masks_the_real_error(monkeypatch, tmp_path, fake_comfy):
    """A log that cannot be opened is swallowed; the ComfyCliError is unchanged."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    # Parent is a regular file, so `os.makedirs` inside the handler setup raises.
    monkeypatch.setattr(
        failure_log, "_FAILURE_LOG_PATH", str(blocker / "failures.jsonl")
    )
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("run", "wf.json")

    assert excinfo.value.code == "credential_missing"
    assert "no API key configured" in str(excinfo.value)
    # The failed setup must not leave the memo claiming a path nothing writes to,
    # or the next failure would skip setup and silently drop its record.
    assert failure_log._failure_handler_path is None


def test_log_path_that_is_a_directory_is_swallowed(monkeypatch, tmp_path, fake_comfy):
    """Same best-effort contract when the path itself is an existing directory."""
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(tmp_path))
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("run", "wf.json")

    assert excinfo.value.code == "credential_missing"


def test_handler_is_configured_for_rotation(log_path, fake_comfy):
    """Smoke-check the stdlib handler's configuration (rotation itself is its own)."""
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("run", "wf.json")

    logger = failure_log.logging.getLogger(failure_log._FAILURE_LOGGER_NAME)
    (handler,) = logger.handlers
    assert isinstance(handler, failure_log.RotatingFileHandler)
    assert handler.maxBytes == failure_log._FAILURE_LOG_MAX_BYTES == 1_048_576
    assert handler.backupCount == failure_log._FAILURE_LOG_BACKUPS == 2
    assert handler.encoding == "utf-8"
    # Bare `%(message)s`: lines must be pure JSON for `jq`.
    assert handler.formatter._fmt == "%(message)s"
    # Never up to the root logger — stdout is the MCP transport.
    assert logger.propagate is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_log_directory_and_file_are_owner_only(monkeypatch, log_path, fake_comfy):
    """The trail holds argv and stderr tails, so it must not be world-readable."""
    # A permissive umask is exactly the case the explicit modes exist for.
    previous = os.umask(0o022)
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_MAX_BYTES", 200)
    try:
        fake_comfy(stdout=_ERROR_ENVELOPE)
        # Enough failures to force at least one rollover: the handler opens a
        # fresh file on rotation, which must be owner-only too.
        for _ in range(5):
            with pytest.raises(server.ComfyCliError):
                server._run_comfy("run", "wf.json")
    finally:
        os.umask(previous)

    backup = log_path.with_suffix(log_path.suffix + ".1")
    assert backup.exists()
    assert stat.S_IMODE(log_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_a_handler_that_fails_to_close_does_not_wedge_the_log(
    monkeypatch, log_path, fake_comfy
):
    """Teardown survives a raising ``close()`` and leaves no stale memo.

    Clearing the memo only AFTER the loop would let a raising ``close()`` leave
    it pointing at a path whose handler is already detached — and if
    ``_FAILURE_LOG_PATH`` were later repointed back to it, setup would be
    skipped and every subsequent record silently dropped.
    """
    logger = failure_log.logging.getLogger(failure_log._FAILURE_LOGGER_NAME)
    hostile = failure_log.logging.NullHandler()
    monkeypatch.setattr(
        hostile, "close", lambda: (_ for _ in ()).throw(OSError("boom")), raising=False
    )
    logger.addHandler(hostile)
    fake_comfy(stdout=_ERROR_ENVELOPE)

    with pytest.raises(server.ComfyCliError):
        server._run_comfy("run", "wf.json")

    # The hostile handler is gone, the real one is installed, and the record landed.
    assert hostile not in logger.handlers
    assert failure_log._failure_handler_path == str(log_path)
    assert len(_entries(log_path)) == 1


def test_rotation_produces_a_backup_file(monkeypatch, log_path, fake_comfy):
    """Past ``maxBytes`` the handler rolls over, so the log is self-limiting."""
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_MAX_BYTES", 200)
    fake_comfy(stdout=_ERROR_ENVELOPE)

    for _ in range(5):
        with pytest.raises(server.ComfyCliError):
            server._run_comfy("run", "wf.json")

    assert log_path.exists()
    assert log_path.with_suffix(log_path.suffix + ".1").exists()
