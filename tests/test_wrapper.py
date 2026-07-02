"""Regression tests for the comfy-cli invocation and output collection.

These lock in the two behaviors that actually broke during development:
1. comfy-cli's global flags (``--json``, ``--where``) MUST precede the
   subcommand — a trailing ``--json`` errors with "No such option".
2. ``fetch_outputs`` must handle BOTH output representations a local run
   produces: bare on-disk paths (from ``comfy run --wait``) and ``/view``
   HTTP URLs (from ``comfy jobs status``) — ``comfy download`` refuses the
   path form, which is why the tool collects outputs itself.
"""

from __future__ import annotations

import io
import json
import subprocess

import pytest

from comfy_local_mcp import server


@pytest.fixture
def patched_run(monkeypatch):
    """Patch away ``shutil.which`` + ``subprocess.run`` for ``_run_comfy``.

    Returns a ``setup(envelope) -> calls`` helper: call it with the envelope
    the fake comfy-cli should emit, and get back the list that captures each
    invocation (``cmd`` + ``env``).
    """

    def setup(envelope: dict) -> list[dict]:
        calls: list[dict] = []

        def fake(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
            calls.append({"cmd": cmd, "env": env})
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(envelope), stderr=""
            )

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


def test_fetch_outputs_copies_on_disk_path(monkeypatch, tmp_path):
    """Regression: local `comfy run` emits bare paths; we copy, never `comfy download`."""
    src = tmp_path / "src" / "img.png"
    src.parent.mkdir()
    src.write_bytes(b"png-bytes")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"outputs": [str(src)]})
    result = server.fetch_outputs("pid", str(out_dir))

    assert result["saved"] == [str(out_dir / "img.png")]
    assert (out_dir / "img.png").read_bytes() == b"png-bytes"


def test_fetch_outputs_fetches_view_url(monkeypatch, tmp_path):
    """Regression: `comfy jobs status` emits /view URLs; we fetch those."""
    url = "http://127.0.0.1:8188/view?filename=gen.png&subfolder=&type=output"
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"outputs": [url]})

    fetched: list[str] = []

    def fake_urlopen(ref, timeout):  # noqa: ARG001
        fetched.append(ref)
        return io.BytesIO(b"view-bytes")

    monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)
    out_dir = tmp_path / "out"
    result = server.fetch_outputs("pid", str(out_dir))

    assert fetched == [url]
    assert result["saved"] == [str(out_dir / "gen.png")]  # name from ?filename=
    assert (out_dir / "gen.png").read_bytes() == b"view-bytes"


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
