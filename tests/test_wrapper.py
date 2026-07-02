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


def test_upload_file_passes_paths_and_overwrite(monkeypatch):
    """upload_file forwards every path and appends --overwrite when asked."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": {"uploaded": 2}})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server.upload_file(["a.png", "b.png"], overwrite=True) == {"uploaded": 2}

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["upload", "a.png", "b.png", "--overwrite"]


def test_upload_file_omits_overwrite_by_default(monkeypatch):
    """Without overwrite the flag must be absent, not passed as False."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": {"uploaded": 1}})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    server.upload_file(["only.png"])

    assert calls[0]["cmd"][4:] == ["upload", "only.png"]
    assert "--overwrite" not in calls[0]["cmd"]


def test_validate_workflow_returns_results_for_valid(monkeypatch):
    """A valid workflow returns comfy-cli's validation data unwrapped."""
    fake, calls = _fake_run(
        {"type": "envelope", "ok": True, "data": {"valid": True, "nodes": 7}}
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server.validate_workflow("wf.json") == {"valid": True, "nodes": 7}
    assert calls[0]["cmd"][4:] == ["validate", "--workflow", "wf.json"]


def test_validate_workflow_raises_with_error_code(monkeypatch):
    """An invalid workflow surfaces comfy-cli's structured error code, not a swallow."""
    fake, _ = _fake_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {
                "code": "workflow_unknown_nodes",
                "message": "Unknown node type: FooSampler",
            },
        }
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match="workflow_unknown_nodes"):
        server.validate_workflow("broken.json")


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
