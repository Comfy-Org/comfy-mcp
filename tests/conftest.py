"""Shared pytest fixtures.

The comfy-cli version guard (`server._check_comfy_version`) shells out to
`comfy --version` once per process from inside `_run_comfy`. The unit tests stub
`subprocess.run` to emit canned envelopes and assert the exact argv — a stray
`comfy --version` call would consume that stub and pollute those assertions. So
by default we mark the guard "already checked" for every test; the dedicated
guard tests (`test_wrapper.py`) re-enable it explicitly and mock `--version`.
``partner_generate``'s spend-gate probe is a second such once-per-process
shell-out and is neutralized the same way.

This module also holds the shared test helpers for the streaming
(``--json-stream``) tool paths. ``run_workflow``, ``watch_job`` and
``generate_image`` all drive the same ``subprocess.Popen`` NDJSON streaming
path, so their tests share the fakes and the ``patched_stream`` fixture defined
here rather than each redefining them.
"""

from __future__ import annotations

import io
import json

import pytest

from comfy_local_mcp import server


@pytest.fixture(autouse=True)
def _skip_version_guard(monkeypatch):
    """Neutralize the once-per-process comfy-cli version guard for unit tests."""
    monkeypatch.setattr(server, "_version_checked", True)


@pytest.fixture(autouse=True)
def _skip_spend_gate_probe(monkeypatch):
    """Neutralize ``partner_generate``'s once-per-process spend-gate probe.

    Same reason as the version guard above: the probe shells out to
    ``comfy generate consent show`` before the first spending call, which would
    consume the stubbed ``subprocess.run`` and shift every exact-argv assertion.
    The dedicated probe tests (`test_partner_generate.py`) re-enable it.
    """
    monkeypatch.setattr(server, "_spend_gate_probed", True)


@pytest.fixture(autouse=True)
def _skip_engine_auto_confirm_probe(monkeypatch):
    """Answer ``partner_generate``'s per-call ``spend.auto_confirm`` read as OFF.

    Third once-per-call shell-out on the spending path (``comfy generate consent
    show --json``); it would consume the stubbed ``subprocess.run`` and shift the
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


class _FakeProc:
    """A minimal stand-in for ``subprocess.Popen`` over a canned NDJSON stream."""

    def __init__(self, cmd, stdout_text, stderr_text="", env=None, encoding=None):
        self.cmd = cmd
        self.env = env
        self.encoding = encoding
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

        def fake_popen(cmd, stdout, stderr, text, encoding, env, **kwargs):  # noqa: ARG001
            proc = _FakeProc(cmd, stdout_text, env=env, encoding=encoding)
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
