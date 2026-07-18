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
import subprocess
import threading
import time

import pytest

from comfy_local_mcp import server


def _fake_run(envelope: dict):
    """Return a ``subprocess.run`` stand-in that captures calls and emits ``envelope``.

    Pure factory (no patching): returns ``(fake, calls)``. Tests either patch
    ``fake`` in themselves or use the ``patched_run`` fixture, which wraps this.
    """
    calls: list[dict] = []

    def fake(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
        calls.append({"cmd": cmd, "env": env})
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(envelope), stderr=""
        )

    return fake, calls


@pytest.fixture
def patched_run(monkeypatch):
    """Patch away ``shutil.which`` + ``subprocess.run`` for ``_run_comfy``.

    Returns a ``setup(envelope) -> calls`` helper: call it with the envelope
    the fake comfy-cli should emit, and get back the list that captures each
    invocation (``cmd`` + ``env``).
    """

    def setup(envelope: dict) -> list[dict]:
        fake, calls = _fake_run(envelope)
        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(server.subprocess, "run", fake)
        return calls

    return setup


def test_global_flags_precede_subcommand(patched_run):
    """Regression: `comfy run … --json` errors; it must be `comfy --json … run`."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {"x": 1}})

    assert server._run_comfy("jobs", "status", "abc") == {"x": 1}

    cmd = calls[0]["cmd"]
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["jobs", "status", "abc"]  # subcommand strictly after
    assert calls[0]["env"]["COMFY_WHERE"] == "local"  # belt-and-suspenders pin


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


def test_cancel_job_maps_command_and_returns_data(patched_run):
    """cancel_job wraps `comfy jobs cancel <id>` and returns the envelope data."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {"cancelled": "abc"}})

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


def test_get_queue_maps_command_and_returns_data(patched_run):
    """get_queue wraps `comfy jobs ls` and returns the merged job list."""
    jobs = {
        "jobs": [
            {"prompt_id": "a", "status": "running"},
            {"prompt_id": "b", "status": "completed"},
        ]
    }
    calls = patched_run({"type": "envelope", "ok": True, "data": jobs})

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


def test_upload_file_passes_paths_and_overwrite(patched_run):
    """upload_file forwards every path and appends --overwrite when asked."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {"uploaded": 2}})

    assert server.upload_file(["a.png", "b.png"], overwrite=True) == {"uploaded": 2}

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["upload", "a.png", "b.png", "--overwrite"]


def test_upload_file_omits_overwrite_by_default(patched_run):
    """Without overwrite the flag must be absent, not passed as False."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {"uploaded": 1}})

    server.upload_file(["only.png"])

    assert calls[0]["cmd"][4:] == ["upload", "only.png"]
    assert "--overwrite" not in calls[0]["cmd"]


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
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

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
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

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
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

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
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

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
    "bad_path", ["../../etc", "models/../../etc", "/abs/models", "..\\..\\etc"]
)
def test_download_model_rejects_traversal_relative_path(bad_path):
    """relative_path must stay within the models dir: no `..` or absolute paths."""
    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        server.download_model("https://hf.co/x.safetensors", relative_path=bad_path)


@pytest.mark.parametrize("bad_name", ["../evil", "sub/dir.safetensors", "..", "a\\b"])
def test_download_model_rejects_pathy_filename(bad_name):
    """filename must be a bare name: no separators or `..` to escape the dir."""
    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        server.download_model("https://hf.co/x.safetensors", filename=bad_name)


def test_download_model_omits_empty_string_optionals(patched_run):
    """Explicit empty-string optionals are treated as unset, not forwarded as ``""``."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

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
    calls = patched_run({"type": "envelope", "ok": True, "data": {"urls": []}})

    server.fetch_outputs("pid", "/tmp/out", url_only=True)

    assert calls[0]["cmd"][4:] == ["download", "pid", "-o", "/tmp/out", "--url-only"]


def test_fetch_outputs_omits_url_only_by_default(patched_run):
    """Without url_only the flag must be absent, not passed as False."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

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


def test_launch_comfyui_passes_background_flag(patched_run):
    """launch_comfyui must run `comfy … launch --background` (detached start)."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {"pid": 42}})

    assert server.launch_comfyui() == {"pid": 42}

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert cmd[4:] == ["launch", "--background"]  # no extras -> no `--` separator


def test_launch_comfyui_forwards_extra_args_after_separator(patched_run):
    """Extra args are forwarded to ComfyUI after a `--` separator."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

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


def test_discover_maps_command_and_returns_data(patched_run):
    """discover wraps `comfy discover` and returns the envelope data verbatim."""
    surface = {"commands": ["run", "env"], "error_codes": ["server_not_running"]}
    calls = patched_run({"type": "envelope", "ok": True, "data": surface})

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


class _FakeProc:
    """A minimal stand-in for ``subprocess.Popen`` over a canned NDJSON stream."""

    def __init__(self, cmd, stdout_text, stderr_text=""):
        self.cmd = cmd
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO(stderr_text)
        self.returncode = 0
        self.killed = False

    def poll(self):
        return self.returncode  # already "finished" once the stream is drained

    def wait(self):
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

        def fake_popen(cmd, stdout, stderr, text, env):  # noqa: ARG001
            proc = _FakeProc(cmd, stdout_text)
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

    def wait(self):
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

    def fake_popen(cmd, stdout, stderr, text, env):  # noqa: ARG001
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

    def fake_popen(cmd, stdout, stderr, text, env):  # noqa: ARG001
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


def test_watch_job_rejects_option_like_prompt_id():
    """A leading-dash prompt_id is refused so comfy-cli can't parse it as a flag."""
    with pytest.raises(server.ComfyCliError, match="invalid prompt_id"):
        asyncio.run(server.watch_job("--help"))


def test_watch_job_clamps_oversized_timeout(monkeypatch):
    """timeout_seconds is clamped to the module ceiling, not passed through raw."""
    seen: dict = {}

    async def fake_stream(*args, ctx=None, timeout=None, raise_on_timeout=True):
        seen["timeout"] = timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    asyncio.run(server.watch_job("pid", timeout_seconds=float("inf")))

    assert seen["timeout"] == server._MAX_WATCH_TIMEOUT


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

    def wait(self):
        self._dead.wait(timeout=10.0)  # a child that lingers past its envelope
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9  # reaped by SIGKILL
        self._dead.set()


def _lingering_popen(procs, stream_lines):
    """A ``subprocess.Popen`` stand-in minting a fresh ``_LingeringProc`` per call."""

    def fake_popen(cmd, stdout, stderr, text, env):  # noqa: ARG001
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

    def fake_popen(cmd, stdout, stderr, text, env):  # noqa: ARG001
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

    def wait(self):
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

    def fake_popen(cmd, stdout, stderr, text, env):  # noqa: ARG001
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

    def wait(self):
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

    def fake_popen(cmd, stdout, stderr, text, env):  # noqa: ARG001
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

        def wait(self):
            self._block_until_dead()
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9
            self._dead.set()

    procs: list[_InstrumentedProc] = []

    def fake_popen(cmd, stdout, stderr, text, env):  # noqa: ARG001
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
