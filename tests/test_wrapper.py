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


def _fake_run(envelope: dict):
    """Return a subprocess.run stand-in that captures the call and emits an envelope."""
    calls: list[dict] = []

    def fake(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
        calls.append({"cmd": cmd, "env": env})
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(envelope), stderr=""
        )

    return fake, calls


def test_global_flags_precede_subcommand(monkeypatch):
    """Regression: `comfy run … --json` errors; it must be `comfy --json … run`."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": {"x": 1}})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server._run_comfy("jobs", "status", "abc") == {"x": 1}

    cmd = calls[0]["cmd"]
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["jobs", "status", "abc"]  # subcommand strictly after
    assert calls[0]["env"]["COMFY_WHERE"] == "local"  # belt-and-suspenders pin


def test_error_envelope_raises_with_code(monkeypatch):
    fake, _ = _fake_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "ComfyUI not running"},
        }
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match="server_not_running"):
        server._run_comfy("env")


def test_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    with pytest.raises(server.ComfyCliError, match="not found on PATH"):
        server._run_comfy("env")


def test_cancel_job_maps_command_and_returns_data(monkeypatch):
    """cancel_job wraps `comfy jobs cancel <id>` and returns the envelope data."""
    fake, calls = _fake_run(
        {"type": "envelope", "ok": True, "data": {"cancelled": "abc"}}
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server.cancel_job("abc") == {"cancelled": "abc"}
    assert calls[0]["cmd"][4:] == ["jobs", "cancel", "abc"]  # mapped subcommand


def test_cancel_job_unknown_id_raises_error_envelope(monkeypatch):
    """Cancelling an unknown prompt_id surfaces comfy-cli's error envelope."""
    fake, _ = _fake_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "not_found", "message": "no such job: nope"},
        }
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match="not_found"):
        server.cancel_job("nope")


def test_get_queue_maps_command_and_returns_data(monkeypatch):
    """get_queue wraps `comfy jobs ls` and returns the merged job list."""
    jobs = {
        "jobs": [
            {"prompt_id": "a", "status": "running"},
            {"prompt_id": "b", "status": "completed"},
        ]
    }
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": jobs})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server.get_queue() == jobs
    assert calls[0]["cmd"][4:] == ["jobs", "ls"]  # no positional args


def test_get_queue_error_envelope_raises(monkeypatch):
    """A failing `comfy jobs ls` (e.g. server unreachable) raises ComfyCliError."""
    fake, _ = _fake_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "ComfyUI not running"},
        }
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match="server_not_running"):
        server.get_queue()


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
