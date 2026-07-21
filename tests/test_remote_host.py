"""Tests for driving a configurable (remote/tailnet) ComfyUI host.

The run/queue tools normally target the implicit local 127.0.0.1:8188. Setting
``COMFYUI_URL`` — or the ``COMFYUI_HOST`` / ``COMFYUI_PORT`` pair — points them
at a ComfyUI running elsewhere by forwarding ``--host`` / ``--port`` to the
comfy-cli verbs that accept them (``comfy run`` and every ``comfy jobs``
subcommand). These lock in:

1. ``_comfy_target`` env parsing (URL, host/port, defaults, malformed values).
2. ``--host`` / ``--port`` forwarded into the SUBCOMMAND (not the global prefix)
   for ``run`` / ``jobs``, and NOT for verbs that don't accept them (``env`` /
   ``download`` / ``upload`` / …).
3. Byte-identical local behavior when nothing is configured.
4. ``server_info`` surfacing the configured ``comfy_target``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess

import pytest
from conftest import _OK_STREAM

from comfy_local_mcp import server


def _fake_run(envelope: dict):
    """A ``subprocess.run`` stand-in that captures calls and emits ``envelope``."""
    calls: list[dict] = []

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
        calls.append({"cmd": cmd, "env": env})
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(envelope), stderr=""
        )

    return fake, calls


@pytest.fixture
def patched_run(monkeypatch):
    """Patch ``shutil.which`` + ``subprocess.run``; returns ``setup(envelope) -> calls``."""

    def setup(envelope: dict) -> list[dict]:
        fake, calls = _fake_run(envelope)
        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(server.subprocess, "run", fake)
        return calls

    return setup


# --- _comfy_target env parsing ---------------------------------------------


def test_target_none_when_unset():
    """Nothing configured -> None -> local default (byte-identical to today)."""
    assert server._comfy_target() is None


def test_target_from_host_defaults_port(monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    assert server._comfy_target() == (
        "gpu.example",
        server.DEFAULT_COMFYUI_PORT,
        "COMFYUI_HOST",
    )


def test_target_from_host_and_port(monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    assert server._comfy_target() == ("gpu.example", 9001, "COMFYUI_HOST")


def test_target_from_url(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    assert server._comfy_target() == ("gpu.example", 9001, "COMFYUI_URL")


def test_target_from_url_defaults_port(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example")
    assert server._comfy_target() == (
        "gpu.example",
        server.DEFAULT_COMFYUI_PORT,
        "COMFYUI_URL",
    )


def test_target_url_without_scheme(monkeypatch):
    """A scheme-less ``host:port`` still parses (prefixed with ``//`` internally)."""
    monkeypatch.setenv("COMFYUI_URL", "gpu.example:9001")
    assert server._comfy_target() == ("gpu.example", 9001, "COMFYUI_URL")


def test_target_url_ipv6(monkeypatch):
    """An IPv6 URL yields the bare host (comfy-cli re-brackets it for its URLs)."""
    monkeypatch.setenv("COMFYUI_URL", "http://[2001:db8::1]:8188")
    assert server._comfy_target() == ("2001:db8::1", 8188, "COMFYUI_URL")


def test_target_url_wins_over_host(monkeypatch):
    """COMFYUI_URL takes precedence over the COMFYUI_HOST/PORT pair."""
    monkeypatch.setenv("COMFYUI_URL", "http://from-url.example:1234")
    monkeypatch.setenv("COMFYUI_HOST", "from-host.example")
    monkeypatch.setenv("COMFYUI_PORT", "5678")
    assert server._comfy_target() == ("from-url.example", 1234, "COMFYUI_URL")


def test_target_url_no_host_raises(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://:8188")
    with pytest.raises(server.ComfyCliError, match="names no host"):
        server._comfy_target()


def test_target_bad_port_raises(monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "not-a-number")
    with pytest.raises(server.ComfyCliError, match="COMFYUI_PORT must be an integer"):
        server._comfy_target()


def test_target_port_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "70000")
    with pytest.raises(server.ComfyCliError, match="out of range"):
        server._comfy_target()


# --- forwarding into _run_comfy (plain --json path) ------------------------


def test_run_forwards_host_port_into_subcommand(patched_run, monkeypatch):
    """`comfy run` gets --host/--port appended AFTER the verb, not in the global prefix."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    calls = patched_run({"type": "envelope", "ok": True, "data": {"prompt_id": "p1"}})

    server._run_comfy("run", "--workflow", "wf.json")

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global prefix unchanged
    assert cmd[4:] == [
        "run",
        "--workflow",
        "wf.json",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_jobs_forwards_host_port_into_subcommand(patched_run, monkeypatch):
    """Every `comfy jobs` subcommand accepts --host/--port; they are forwarded."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    calls = patched_run({"type": "envelope", "ok": True, "data": {"status": "running"}})

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["cmd"][4:] == [
        "jobs",
        "status",
        "abc",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_env_is_not_forwarded_host_port(patched_run, monkeypatch):
    """`comfy env` takes no --host/--port; forwarding would error 'No such option'."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

    server._run_comfy("env")

    assert calls[0]["cmd"][4:] == ["env"]  # untouched even with a remote configured


def test_download_and_upload_not_forwarded(patched_run, monkeypatch):
    """download / upload verbs don't accept --host/--port -> stay local-only."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

    server._run_comfy("download", "abc", "-o", "/tmp/out")
    server._run_comfy("upload", "a.png")

    assert calls[0]["cmd"][4:] == ["download", "abc", "-o", "/tmp/out"]
    assert calls[1]["cmd"][4:] == ["upload", "a.png"]


def test_local_default_is_byte_identical(patched_run):
    """With nothing configured the argv is exactly today's — no --host appended."""
    calls = patched_run({"type": "envelope", "ok": True, "data": {}})

    server._run_comfy("run", "--workflow", "wf.json")

    assert calls[0]["cmd"][4:] == ["run", "--workflow", "wf.json"]  # no --host/--port


# --- forwarding into the streaming (--json-stream) path --------------------


def test_run_workflow_stream_forwards_host_port(patched_stream, monkeypatch):
    """run_workflow(wait=True) streams `comfy run --wait` with --host/--port forwarded."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    procs = patched_stream(_OK_STREAM)

    result = asyncio.run(server.run_workflow("wf.json", wait=True))

    assert result == {"outputs": ["/x.png"]}
    cmd = procs[0].cmd
    assert cmd[1:4] == ["--json-stream", "--where", "local"]  # global prefix unchanged
    assert cmd[4:] == [
        "run",
        "--workflow",
        "wf.json",
        "--wait",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_watch_job_stream_forwards_host_port(patched_stream, monkeypatch):
    """watch_job tails `comfy jobs watch <id>` with --host/--port forwarded."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.watch_job("pid"))

    assert procs[0].cmd[4:] == [
        "jobs",
        "watch",
        "pid",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


# --- server_info surfaces the configured target ----------------------------


def _patch_env_for_server_info(monkeypatch):
    """Stub `comfy env` + the version detection so `server_info()` runs offline."""
    fake, _calls = _fake_run(
        {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": True,
            "data": {"running": False},
        }
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.12.0")


def test_server_info_reports_remote_target(monkeypatch):
    _patch_env_for_server_info(monkeypatch)
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")

    result = server.server_info()

    assert result["comfy_target"]["host"] == "gpu.example"
    assert result["comfy_target"]["port"] == 9001
    assert result["comfy_target"]["source"] == "COMFYUI_URL"
    assert result["running"] is False  # local `comfy env` data still preserved
    assert result["compatibility"]["envelope_schema"] == "envelope/1"


def test_server_info_omits_target_when_local(monkeypatch):
    _patch_env_for_server_info(monkeypatch)

    result = server.server_info()

    assert "comfy_target" not in result  # no remote configured -> no block
