"""Tests for the opt-in local rotating failure log (``COMFY_MCP_DEBUG_LOG``).

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

import asyncio
import json
import os
import pathlib
import stat
import subprocess
import sys
import threading

import pytest
from conftest import _FakeProc

from comfy_mcp import errors, failure_log, server, tcc, textutil

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
        ("darwin", ("Library", "Application Support", "comfy-mcp")),
        ("win32", ("AppData", "Local", "comfy-mcp")),
        ("linux", (".config", "comfy-mcp")),
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
    """``COMFY_MCP_DEBUG_LOG=<path>`` writes to exactly that file."""
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
    assert failure_log._FAILURE_LOG_TAIL_CHARS > errors._MAX_ERROR_FIELD_CHARS


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

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _FakeProc(list(cmd), _ERROR_ENVELOPE + "\n", env=env)
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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
        asyncio.run(server.job(action="error", prompt_id="-x"))

    assert not log_path.exists()


# --- redaction ---------------------------------------------------------------


def test_scrub_arg_masks_userinfo_and_strips_query(log_path, fake_comfy):
    """A URL arg is logged with userinfo masked and query/fragment dropped.

    Also covers the ``message`` field, which :func:`_unwrap_envelope` builds as
    ``comfy <args> failed …`` — i.e. it echoes the RAW argv, so scrubbing only
    the ``args`` list would leave the credential in the very next key.
    """
    url = "https://<user>:<tok>@host/x?token=abc#frag"
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
    text = "comfy model download --url https://<u>:<p>@h/m.safetensors?token=s failed"
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
        ("<user>:<pass>@notaurl", "<user>:<pass>@notaurl"),
        # Query stripped even with no credential in it (mirrors comfy-cli's own
        # scrubber: a CivitAI URL carries its token as `?token=…`).
        ("https://h/m.safetensors?token=abc", "https://h/m.safetensors"),
        ("http://h/p#f", "http://h/p"),
        # A fragment BEFORE a query still cuts at the first delimiter.
        ("https://h/p#f?q=1", "https://h/p"),
        # Userinfo masked with no query present, and the scheme is case-blind.
        ("https://<u>:<p>@h/path", "https://***@h/path"),
        ("HTTPS://<u>:<p>@h/path", "HTTPS://***@h/path"),
        # A bare host URL is untouched.
        ("https://h/path", "https://h/path"),
    ],
)
def test_scrub_arg_cases(arg, expected):
    assert failure_log._scrub_arg(arg) == expected


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        # `_run_template_param_args` / `_generate_param_args` emit combined-flag
        # tokens, so the URL is not at the token's start — a start-anchored test
        # would log the credential.
        (
            "--image_url=https://<user>:<tok>@host/x?token=abc",
            "--image_url=https://***@host/x",
        ),
        # …and `run_template` wraps the value in JSON, offsetting it further.
        (
            '--param=src={"https://<u>:<p>@h/i.png"}',
            '--param=src={"https://***@h/i.png"}',
        ),
    ],
)
def test_scrub_arg_finds_a_url_anywhere_in_the_token(arg, expected):
    """A URL mid-token is scrubbed, not just one the token starts with."""
    assert failure_log._scrub_arg(arg) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Click quotes the offending value; `_URL_RE` swallows the closing quote
        # and `_scrub_url` cuts from the `?`, so the rest of the sentence went
        # with the query. These messages reach the MCP CLIENT, not just the log.
        (
            "Invalid value for '--url': 'https://h/x?q=1' is not reachable.",
            "Invalid value for '--url': 'https://h/x' is not reachable.",
        ),
        # `_render_error_details` joins list entries with `", "` — without the
        # peel the comma is eaten and two entries merge into one token.
        ("https://h/a?t=1, https://h/b?t=2", "https://h/a, https://h/b"),
        ("see https://h/p?q=1.", "see https://h/p."),
        ("(https://h/p?q=1)", "(https://h/p)"),
        # Lossless where there is no query to cut: the peeled characters are
        # scrubbed to themselves and put straight back.
        ("https://h/p.", "https://h/p."),
        ("'https://<u>:<p>@h/p'", "'https://***@h/p'"),
        # Punctuation INSIDE the URL is still dropped with the rest of the query.
        ("https://h/x?a=1,2,3 next", "https://h/x next"),
        # Userinfo, a query AND a closing quote at once — the shape the split
        # below has to leave alone, since a quote is not a scheme boundary.
        (
            "quoted 'https://<u>:<p>@h/x?q=1' tail",
            "quoted 'https://***@h/x' tail",
        ),
    ],
)
def test_scrub_text_keeps_the_punctuation_glued_to_a_url(text, expected):
    """`_URL_RE` cannot tell a URL from what merely FOLLOWS it.

    None of the peeled characters is credential material, so restoring them
    gives nothing back — and without them a scrubbed message loses the quote,
    comma or full stop that made it a sentence.
    """
    assert failure_log._scrub_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Two URLs glued by a comma are ONE whitespace-delimited token. Under a
        # plain `\S+` they matched as one, and only the FIRST one's userinfo was
        # masked — the second went to the client and to disk verbatim.
        (
            "opts: https://<u1>:<p1>@a.invalid/x,https://<u2>:<p2>@b.invalid/y",
            "opts: https://***@a.invalid/x,https://***@b.invalid/y",
        ),
        # Three deep, semicolon-separated: every match ends at the next scheme,
        # so the run is scrubbed pairwise rather than first-one-wins.
        (
            "https://<u1>:<p1>@a.invalid/x;https://<u2>:<p2>@b.invalid/y"
            ";https://<u3>:<p3>@c.invalid/z",
            "https://***@a.invalid/x;https://***@b.invalid/y;https://***@c.invalid/z",
        ),
        # The second URL keeps its own query cut — the delimiter it anchors on
        # is inside its own match, not the first URL's.
        (
            "https://<u1>:<p1>@a.invalid/x,https://<u2>:<p2>@b.invalid/y?token=s3cret",
            "https://***@a.invalid/x,https://***@b.invalid/y",
        ),
        # …and where the FIRST carries the query, its cut still takes the whole
        # query with it, second URL included: tempering stops at the `?`.
        ("https://h/a?token=1,https://h2/b?token=2", "https://h/a"),
        # The case that keeps the tempering query-bounded: a query VALUE that is
        # itself an absolute URL. Splitting there would strand `&token=…` in a
        # second match with no `?` left to cut on, leaking a secret the plain
        # `\S+` dropped. Over-redacting the nested host is the safe direction.
        ("https://<u>:<p>@h/x?next=https://h2/y&token=s3cret", "https://***@h/x"),
        # Regression: one URL with userinfo AND a query is unchanged, and a
        # string with no URL at all comes back byte-for-byte.
        ("https://<u>:<p>@h/x?token=s3cret", "https://***@h/x"),
        (
            "comfy run wf.json failed: node 3 has no input",
            "comfy run wf.json failed: node 3 has no input",
        ),
    ],
)
def test_scrub_text_masks_every_url_in_a_token_not_just_the_first(text, expected):
    """Adjacent URLs with no whitespace between them scrub independently."""
    assert failure_log._scrub_text(text) == expected


def test_scrub_arg_masks_both_urls_of_a_combined_flag_list():
    """The argv shape the split exists for: a URL list inside one `--param=`.

    `_run_template_param_args` renders the whole value into a single token, so a
    caller passing two source images hands the scrubber one `\\S+` run holding
    two credentials.
    """
    arg = '--param=src="https://<u1>:<p1>@a.invalid/x,https://<u2>:<p2>@b.invalid/y"'

    assert failure_log._scrub_arg(arg) == (
        '--param=src="https://***@a.invalid/x,https://***@b.invalid/y"'
    )


def test_message_is_scrubbed_before_it_is_capped(log_path):
    """A URL straddling the message cap is masked, not cut into an unmasked half.

    Capping first would slice ``https://<user>:<pass>@host`` before its ``@``,
    and :func:`_redact_url` only masks userinfo it can still find in the netloc
    — so the surviving ``<user>:<pass>`` would land in the file verbatim.
    """
    cap = failure_log._FAILURE_LOG_MESSAGE_CHARS
    # Place the URL so the cap falls on its `@` — the whole userinfo is inside
    # the kept prefix but the `@` that marks it as userinfo is not, which is
    # precisely what defeats `_redact_url` when the cap runs first. Both the
    # 16 and the userinfo's width are that placement: `https://<u>:<pw>` is
    # exactly 16 characters, and so is the `https://***@host` the masked URL
    # gets clipped to below.
    message = "x" * (cap - 16) + "https://<u>:<pw>@host/m.safetensors?token=abc"

    failure_log._log_failure("error_envelope", ("run",), message=message)

    (entry,) = _entries(log_path)
    assert "<u>:<pw>" not in entry["message"]
    assert "token=abc" not in entry["message"]
    assert len(entry["message"]) == cap
    # The cap now lands inside the *masked* URL, so what it clips is `***@…`
    # rather than the raw `<u>:<pw>@…` the old cap-then-scrub order would leave.
    assert entry["message"].endswith("https://***@host")


def test_stream_tails_are_scrubbed_like_the_message(log_path):
    """comfy-cli echoing a credential URL to a stream must not persist it raw."""
    failure_log._log_failure(
        "no_json",
        ("model", "download"),
        stdout="fetching https://<user>:<tok>@host/m.safetensors?token=abc",
        stderr="failed: https://<u>:<p>@h/x?sig=deadbeef",
    )

    (entry,) = _entries(log_path)
    assert entry["stdout_tail"] == "fetching https://***@host/m.safetensors"
    assert entry["stderr_tail"] == "failed: https://***@h/x"
    assert "tok" not in log_path.read_text()
    assert "deadbeef" not in log_path.read_text()


def test_stream_tail_scrubs_a_url_straddling_the_tail_cut(log_path):
    """A URL clipped by the tail bound is dropped, never half-written.

    The tail keeps the END of a capture, so scrubbing after the clip would see a
    URL already shorn of its ``https://`` and skip it entirely.
    """
    limit = failure_log._FAILURE_LOG_TAIL_CHARS
    url = "https://<user>:<tok>@host/x"
    # Sized so the clip falls immediately after the URL's `https://` — the exact
    # case where a scrub-after-clip pass sees no scheme, matches nothing, and
    # writes the whole `<user>:<tok>@host/x` remainder to disk.
    stderr = "A" * 100 + url + "B" * (limit + 108 - 100 - len(url))
    assert len(textutil._stream_tail(stderr, limit)) == limit + 3  # it IS clipped

    failure_log._log_failure("no_json", ("run",), stderr=stderr)

    (entry,) = _entries(log_path)
    assert "<user>:<tok>" not in entry["stderr_tail"]
    assert "<tok>@host" not in entry["stderr_tail"]
    # Still bounded, and still marked as truncated.
    assert entry["stderr_tail"].startswith("...")
    assert len(entry["stderr_tail"]) == limit + 3


def test_stream_tail_drops_the_head_fragment_of_a_url_dense_capture(log_path):
    """The straddling-URL guard holds even when scrubbing shrinks the window.

    The double-width window is bounded generously so a half-URL at its head
    falls outside the final clip — but scrubbing DELETES query strings and
    userinfo, so a URL-dense capture can shrink to under ``limit`` and take
    `_scrubbed_stream_tail`'s early return with that head still attached. The
    head fragment has no ``https://`` left for ``_URL_RE`` to anchor on, so
    nothing else in the pipeline can catch it.
    """
    limit = failure_log._FAILURE_LOG_TAIL_CHARS
    head = "user:tok@host/x "
    # Each unit scrubs 119 chars down to 12, so a window packed with them lands
    # far under `limit` — that shrinkage is what reaches the early return.
    unit = "https://h/a?token=" + "A" * 100 + " "
    filler = unit * ((limit * 2 - len(head)) // len(unit))
    # Exactly `limit * 2` after the scheme, so the window's cut lands right
    # after `https://` and `head` arrives scheme-shorn. No trailing whitespace:
    # `_tail` strips before slicing, which would otherwise shift the cut.
    tail_bytes = head + filler + "B" * (limit * 2 - len(head) - len(filler))
    stderr = "Z" * 10_000 + "https://" + tail_bytes

    failure_log._log_failure("no_json", ("run",), stderr=stderr)

    (entry,) = _entries(log_path)
    assert len(entry["stderr_tail"]) <= limit  # it DID take the early return
    assert "user:tok" not in log_path.read_text()
    # Dropped, not masked — see the next test for why masking is not enough.
    # What follows the fragment is intact, so the tail still reads.
    assert entry["stderr_tail"].startswith("... https://h/a ")


def test_stream_tail_drops_a_head_fragment_clipped_past_its_query_marker(log_path):
    """Masking the head fragment covers a clip BEFORE the `?` and no further.

    ``_scrub_url`` deletes from the first ``?``, so a window whose cut landed
    PAST one arrives as a bare ``token=…`` remnant with no delimiter left to
    anchor on — masking it is a no-op and the credential is written verbatim.
    Dropping the fragment is what closes that, and it is the case the log's
    own `Downloading <signed url>` progress lines make reachable.
    """
    limit = failure_log._FAILURE_LOG_TAIL_CHARS
    remnant = "tok=SECRETVALUE&x=1 "
    # Same shrink-to-under-`limit` trick as above: it is the early return that
    # would otherwise carry the head fragment out to disk.
    unit = "https://h/a?token=" + "A" * 100 + " "
    filler = unit * ((limit * 2 - len(remnant)) // len(unit))
    window = remnant + filler
    stderr = "Z" * 10_000 + window.ljust(limit * 2, "B")

    failure_log._log_failure("no_json", ("run",), stderr=stderr)

    (entry,) = _entries(log_path)
    assert len(entry["stderr_tail"]) <= limit  # it DID take the early return
    assert "SECRETVALUE" not in log_path.read_text()
    assert entry["stderr_tail"].startswith("... https://h/a ")


def test_stream_tail_masks_adjacent_urls_around_a_dropped_head_fragment(log_path):
    """The URL-split and the head-fragment contract hold at the same time.

    Two things share this window: a leading token that is a scheme-shorn URL
    fragment with a SECOND, full-schemed URL glued to it, and — past the first
    whitespace, where the head drop cannot reach — a comma-joined pair of full
    URLs. The pair is what a plain ``\\S+`` matched as one token, masking only
    the first credential; here both must come out masked. The head fragment
    keeps today's contract regardless: dropped outright, second URL and all.
    """
    limit = failure_log._FAILURE_LOG_TAIL_CHARS
    head = "user:tok@a.invalid/x,https://<u2>:<p2>@b.invalid/y "
    pair = "https://<u3>:<p3>@c.invalid/x,https://<u4>:<p4>@d.invalid/y "
    # Same shrink-to-under-`limit` packing as the tests above: it is the early
    # return that carries the head fragment (and the pair) out to disk.
    unit = "https://h/a?token=" + "A" * 100 + " "
    filler = unit * ((limit * 2 - len(head) - len(pair)) // len(unit))
    body = head + pair + filler
    # Exactly `limit * 2` after the scheme, so the window's cut lands right
    # after `https://` and `head` arrives scheme-shorn.
    stderr = "Z" * 10_000 + "https://" + body.ljust(limit * 2, "B")

    failure_log._log_failure("no_json", ("run",), stderr=stderr)

    (entry,) = _entries(log_path)
    assert len(entry["stderr_tail"]) <= limit  # it DID take the early return
    written = log_path.read_text()
    for secret in ("user:tok", "<p2>", "<p3>", "<p4>"):
        assert secret not in written
    # The head fragment is dropped whole; the pair behind it survives, masked on
    # BOTH sides of the comma.
    assert entry["stderr_tail"].startswith(
        "... https://***@c.invalid/x,https://***@d.invalid/y "
    )


def test_stream_tail_keeps_a_whitespace_free_window_rather_than_emptying_it(log_path):
    """The head-fragment drop stops short of throwing the whole capture away.

    A capture with no whitespace in its last ``limit * 2`` chars is ONE token,
    so dropping it would return a bare ``...`` and lose the error the tail
    exists to keep. There the fragment is masked instead — the narrower
    residual the docstring accepts.
    """
    limit = failure_log._FAILURE_LOG_TAIL_CHARS
    stderr = "x" * (limit * 3) + "THE-REAL-ERROR"

    failure_log._log_failure("no_json", ("run",), stderr=stderr)

    (entry,) = _entries(log_path)
    assert entry["stderr_tail"].endswith("THE-REAL-ERROR")
    assert len(entry["stderr_tail"]) == limit + len("...")


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


def test_readme_migration_note_names_the_live_env_var_and_log_dir():
    """The rename note is prose, so nothing else would notice it going stale.

    ``Upgrading from comfy-local-mcp`` tells an existing installer that
    ``COMFY_LOCAL_MCP_DEBUG_LOG`` is dead and that the log directory moved — the
    one place a reader learns their still-set old variable is now writing
    nothing. Rename either surface again without touching the note and it
    quietly starts giving the wrong instruction, which is worse than no note at
    all. Key the assertions on the values the code actually uses.
    """
    readme = pathlib.Path(__file__).resolve().parent.parent / "README.md"
    body = readme.read_text(encoding="utf-8")
    assert "## Upgrading from" in body, "the rename note was removed from the README"
    section = body.split("## Upgrading from", 1)[1].split("\n## ", 1)[0]

    # The variable the server reads today, and the dead one it no longer reads.
    assert failure_log._FAILURE_LOG_ENV in section
    assert "COMFY_LOCAL_MCP_DEBUG_LOG" in section

    # The directory leaf the default path lands in today, named as the "is now".
    leaf = os.path.basename(os.path.dirname(failure_log._default_failure_log_path()))
    assert f"`{leaf}/`" in section
