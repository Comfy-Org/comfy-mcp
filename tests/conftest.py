"""Shared pytest fixtures.

The comfy-cli version guard (`server._check_comfy_version`) shells out to
`comfy --version` once per process from inside `_run_comfy`. The unit tests stub
the comfy-cli spawn to emit canned envelopes and assert the exact argv — a stray
`comfy --version` call would consume that stub and pollute those assertions. So
by default we mark the guard "already checked" for every test; the dedicated
guard tests (`test_wrapper.py`) re-enable it explicitly and mock `--version`.
``partner_generate``'s spend-gate probe is a second such once-per-process
shell-out and is neutralized the same way.

This module also holds the shared test helpers for both comfy-cli spawn paths,
so a change to how ``server`` shells out lands in ONE fake rather than in a
copy per test file:

* the plain ``--json`` path (``subprocess.Popen`` + a bounded
  ``communicate``) — ``envelope`` + ``patched_run`` / ``patched_plain_run``;
* the streaming ``--json-stream`` path (``subprocess.Popen`` + incremental
  reads) — ``patched_stream``. ``run_workflow``, ``watch_job`` and
  ``generate_image`` all drive the same NDJSON stream.

Both paths spawn with ``start_new_session=True`` so a timeout can kill the whole
process group; the fakes model that too (see ``_FakeRunProc``).
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from comfy_local_mcp import failure_log, server


@pytest.fixture(autouse=True)
def _skip_version_guard(monkeypatch):
    """Neutralize the once-per-process comfy-cli version guard for unit tests."""
    monkeypatch.setattr(server, "_version_checked", True)


@pytest.fixture(autouse=True)
def _skip_spend_gate_probe(monkeypatch):
    """Neutralize ``partner_generate``'s once-per-process spend-gate probe.

    Same reason as the version guard above: the probe shells out to
    ``comfy generate consent show`` before the first spending call, which would
    consume the stubbed spawn and shift every exact-argv assertion.
    The dedicated probe tests (`test_partner_generate.py`) re-enable it.
    """
    monkeypatch.setattr(server, "_spend_gate_probed", True)


@pytest.fixture(autouse=True)
def _skip_engine_auto_confirm_probe(monkeypatch):
    """Answer ``partner_generate``'s per-call ``spend.auto_confirm`` read as OFF.

    Third once-per-call shell-out on the spending path (``comfy generate consent
    show --json``); it would consume the stubbed spawn and shift the
    exact-argv assertions the same way the two guards above would. ``False`` is
    also the default posture under test — the engine has no durable
    always-proceed, so consent has to come from the call itself. The dedicated
    auto-confirm tests (`test_partner_generate.py`) restore the real function.
    """
    monkeypatch.setattr(server, "_engine_auto_confirms", lambda: False)


@pytest.fixture(autouse=True)
def _clear_comfyui_target_env(monkeypatch):
    """Default every test to the LOCAL target (no configured remote ComfyUI).

    The run/queue tools forward ``--host`` / ``--port`` when ``COMFYUI_URL`` /
    ``COMFYUI_HOST`` is set (see ``server._comfy_target``). A stray value in the
    ambient environment would perturb the exact-argv assertions across the suite,
    so clear all three here; the remote-targeting tests set them explicitly.
    """
    for var in ("COMFYUI_URL", "COMFYUI_HOST", "COMFYUI_PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _isolate_failure_log(monkeypatch):
    """Default every test to the opt-in failure log being OFF, and never leak it.

    ``failure_log._FAILURE_LOG_PATH`` is resolved from ``COMFY_LOCAL_MCP_DEBUG_LOG`` at
    import, so a developer who has the var exported would otherwise have the whole
    suite writing real records into their app-support directory — and the
    "disabled by default" tests would fail for an environmental reason. Pin it off
    here (`test_failure_log.py` enables it explicitly per test).

    The teardown closes any handler a test opened, so no test leaves a file handle
    on a ``tmp_path`` that pytest is about to remove, and no record written by one
    test can land in the next one's log.
    """
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", None)
    monkeypatch.setattr(failure_log, "_failure_handler_path", None)
    yield
    logger = logging.getLogger(failure_log._FAILURE_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture(autouse=True)
def _clear_t2i_env(monkeypatch):
    """Default every test to ``generate_image``'s built-in template + slot keys.

    Same reason as the target env above: ``generate_image`` reads
    ``COMFY_T2I_TEMPLATE`` / ``COMFY_T2I_PROMPT_SLOT`` /
    ``COMFY_T2I_CHECKPOINT_SLOT`` per call, so an ambient value would perturb its
    exact-argv assertions. The override test sets them explicitly.
    """
    for var in (
        "COMFY_T2I_TEMPLATE",
        "COMFY_T2I_PROMPT_SLOT",
        "COMFY_T2I_CHECKPOINT_SLOT",
    ):
        monkeypatch.delenv(var, raising=False)


_UNSET = object()


def envelope(*, ok: bool = True, data=_UNSET, error=None) -> dict:
    """Build a comfy-cli ``envelope/1`` result body.

    Mirrors what the CLI emits on its ``--json`` path: an ``error`` object when
    the call failed, a ``data`` payload otherwise. One builder here rather than
    one re-derived per test file, so a schema bump has a single place to land.

    ``data`` defaults to ``{}`` only when it is OMITTED — an explicit
    ``data=None`` stays ``None``, which several tests rely on to exercise the
    non-dict-payload branches.
    """
    body: dict = {"schema": "envelope/1", "type": "envelope", "ok": ok}
    if error is not None:
        body["error"] = error
    else:
        body["data"] = {} if data is _UNSET else data
    return body


def _raises_at_spawn(exc: BaseException) -> bool:
    """Whether the real :class:`subprocess.Popen` would raise ``exc`` itself.

    The constructor fails with an ``OSError`` (no such binary, EPERM, a
    wrong-arch exec) or the bare ``ValueError`` an embedded NUL in argv
    produces; a deadline (``TimeoutExpired``) and a strict-UTF-8
    ``UnicodeDecodeError`` on the child's output both surface from the bounded
    ``communicate`` instead. The fakes place an injected exception where the
    real pair would, so it reaches the handler it would reach in production —
    ``UnicodeDecodeError`` is explicitly excluded because it is a ``ValueError``
    subclass and would otherwise be mistaken for the NUL case.
    """
    return isinstance(exc, (OSError, ValueError)) and not isinstance(
        exc, UnicodeDecodeError
    )


class _FakeRunProc:
    """A ``Popen`` stand-in for the plain path: one canned result, no real pipes.

    ``_run_comfy_raw`` spawns with :class:`subprocess.Popen` and bounds the run
    with ``communicate(timeout=…)``, so the fake models both halves — the spawn
    (argv / env / stdin / encoding) and the wait (the timeout, the canned
    output, and on the timeout path the kill → reap → drain sequence).

    It deliberately carries NO ``pid``, exactly like :class:`_FakeProc`: that
    sends ``server._kill_proc_tree`` down its ``AttributeError`` fallback to
    ``proc.kill()`` instead of calling ``os.killpg`` on a made-up pid, which on
    a real machine could signal an unrelated process group. The dedicated
    group-kill test supplies its own fake with a pid and stubs ``os.killpg``.
    """

    def __init__(self, cmd, record, *, stdout, stderr, returncode, raises):
        self.args = cmd
        self.record = record
        # Back-reference so a test can assert on what the timeout/failure
        # handler did to the process it spawned (``calls[0]["proc"].killed``).
        record["proc"] = self
        self.stdout = None  # `communicate` hands back the canned text directly
        self.stderr = None
        self.returncode = None
        self.killed = False
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = returncode
        self._raises = raises
        self._communicates = 0

    def communicate(self, timeout=None):
        self._communicates += 1
        if self._communicates > 1:
            # The post-kill drain. A real ``communicate()`` after a timeout
            # resumes the buffers it was filling when the deadline fired, so it
            # returns the partial output ``subprocess.run`` used to attach to
            # the ``TimeoutExpired`` — replay the exception's captures.
            return getattr(self._raises, "stdout", None), getattr(
                self._raises, "stderr", None
            )
        self.record["timeout"] = timeout  # the bound only reaches us here
        if self._raises is not None:
            raise self._raises
        self.returncode = self._exit_code
        return self._stdout, self._stderr

    def poll(self):
        return self.returncode  # None until it exits or is killed

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def _canonical_run(calls: list[dict], *, stdout, returncode, stderr, raises):
    """The one ``subprocess.Popen`` stand-in for the plain (non-streaming) path.

    Its parameter list mirrors ``_run_comfy_raw``'s exact ``subprocess.Popen``
    kwargs — which is precisely why it lives here: a kwarg added or renamed
    there (``start_new_session=`` was the last one, when the timeout handler
    started reaping the whole process group) is a one-line edit instead of a
    sweep across every test file.

    Each call is recorded as ``{"cmd", "env", "timeout", "encoding", "stdin",
    "start_new_session", "proc"}`` — a superset of what any caller asserts on —
    BEFORE ``raises`` fires, so a test for a spawn that blows up can still see
    the argv it blew up on. ``timeout`` is filled in by ``communicate``, which
    is where the bound now lands, and ``proc`` is the spawned
    :class:`_FakeRunProc` (absent when the spawn itself raised).
    """
    canned_stdout, canned_stderr = stdout, stderr

    def fake(cmd, stdout, stderr, stdin, text, encoding, env, start_new_session):  # noqa: ARG001
        record = {
            "cmd": cmd,
            "env": env,
            "timeout": None,
            "encoding": encoding,
            "stdin": stdin,
            "start_new_session": start_new_session,
        }
        calls.append(record)
        if raises is not None and _raises_at_spawn(raises):
            raise raises
        return _FakeRunProc(
            cmd,
            record,
            stdout=canned_stdout,
            stderr=canned_stderr,
            returncode=returncode,
            raises=raises,
        )

    return fake


@pytest.fixture
def patched_run(monkeypatch):
    """Patch ``shutil.which`` + ``subprocess.Popen`` for the plain ``--json`` path.

    Returns ``setup(stdout=…, returncode=…, stderr=…, raises=…) -> calls``:

    * ``stdout`` — a dict (JSON-encoded for you, the common case: pass an
      :func:`envelope`) or a raw string; defaults to an empty-``data`` success
      envelope.
    * ``raises`` — an exception instance the fake raises instead of returning,
      from the spawn or from the wait per :func:`_raises_at_spawn`. A
      ``TimeoutExpired`` comes out of the wait, and the fake then plays out the
      kill → reap → drain the handler runs against it.

    ``calls`` is the live list every invocation is recorded into, for the exact
    argv assertions this suite is built on.
    """

    def setup(stdout=None, *, returncode: int = 0, stderr: str = "", raises=None):
        if stdout is None:
            stdout = envelope()
        if isinstance(stdout, dict):
            stdout = json.dumps(stdout)
        calls: list[dict] = []
        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(
            server.subprocess,
            "Popen",
            _canonical_run(
                calls,
                stdout=stdout,
                returncode=returncode,
                stderr=stderr,
                raises=raises,
            ),
        )
        return calls

    return setup


@pytest.fixture
def patched_plain_run(patched_run):
    """``setup(returncode=…, stdout=…, stderr=…) -> calls`` for the NO-envelope path.

    ``comfy launch`` / ``stop`` / ``generate`` print human text through their own
    printer and never emit an ``envelope/1``, so their tests are about the exit
    code and the streams. Same fake as :func:`patched_run`, with the returncode
    first and an empty stdout default.
    """

    def setup(returncode: int = 0, stdout: str = "", stderr: str = "") -> list[dict]:
        return patched_run(stdout, returncode=returncode, stderr=stderr)

    return setup


class _FakeProc:
    """A minimal stand-in for ``subprocess.Popen`` over a canned NDJSON stream."""

    def __init__(
        self, cmd, stdout_text, stderr_text="", env=None, encoding=None, stdin=None
    ):
        self.cmd = cmd
        self.env = env
        self.encoding = encoding
        self.stdin_arg = stdin  # what `server` asked for, not a writable pipe
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO(stderr_text)
        self.returncode = 0
        self.killed = False

    def poll(self):
        return self.returncode  # already "finished" once the stream is drained

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode

    def kill(self):
        self.killed = True


class _RecordingCtx:
    """A fake FastMCP Context that records each ``report_progress`` call."""

    def __init__(self):
        self.calls: list[dict] = []

    async def report_progress(self, progress, total=None, message=None):
        self.calls.append({"progress": progress, "total": total, "message": message})


@pytest.fixture
def patched_stream(monkeypatch):
    """Patch ``shutil.which`` + ``subprocess.Popen`` for the streaming path.

    Returns ``setup(stdout_text) -> procs`` — the list capturing each spawned
    ``_FakeProc`` (so the test can assert the command line that was run).
    """

    def setup(stdout_text: str) -> list[_FakeProc]:
        procs: list[_FakeProc] = []

        def fake_popen(cmd, stdout, stderr, text, encoding, env, stdin=None, **kwargs):  # noqa: ARG001
            proc = _FakeProc(cmd, stdout_text, env=env, encoding=encoding, stdin=stdin)
            procs.append(proc)
            return proc

        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
        return procs

    return setup


# A queued event (2-node manifest), a per-node step progress event, two
# node-completion events, then the success envelope on the last line.
_OK_STREAM = (
    "\n".join(
        json.dumps(evt)
        for evt in [
            {
                "schema": "event/1",
                "type": "queued",
                "nodes": [{"node_id": "1"}, {"node_id": "2"}],
            },
            {"schema": "event/1", "type": "executing", "node": "1", "title": "Load"},
            {
                "schema": "event/1",
                "type": "progress",
                "node": "1",
                "completed": 5,
                "total": 10,
            },
            {"schema": "event/1", "type": "executed", "node": "1", "title": "Load"},
            {"schema": "event/1", "type": "executed", "node": "2", "title": "Save"},
            {
                "schema": "envelope/1",
                "type": "envelope",
                "ok": True,
                "data": {"outputs": ["/x.png"]},
            },
        ]
    )
    + "\n"
)
