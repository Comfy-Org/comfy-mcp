"""Upload failure diagnostics: client envelopes and server-side logs.

Expected operational failures come back as ``kind="upload_error"``. Programming
errors stay exceptions and leave a traceback in the log.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from conftest import envelope

from comfy_mcp import cli, server, upload_session
from comfy_mcp.errors import ComfyCliError


@pytest.fixture
def spool(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_MCP_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("COMFY_MCP_UPLOAD_PUBLIC_BASE_URL", "https://uploads.example")
    monkeypatch.setenv("COMFY_MCP_UPLOAD_TTL_SECONDS", "600")
    return tmp_path


def _reader(body: bytes):
    pending = {"left": body}

    def read(n: int) -> bytes:
        chunk = pending["left"][:n]
        pending["left"] = pending["left"][n:]
        return chunk

    return read


def _ready(filename: str = "image.png", body: bytes = b"png!") -> dict:
    created = upload_session.initialize(filename, len(body), "image/png", False)
    token = created["upload_headers"]["Authorization"].split(" ", 1)[1]
    upload_session.accept_put(
        created["upload_id"],
        f"Bearer {token}",
        str(len(body)),
        "image/png",
        _reader(body),
    )
    return created


def test_ready_session_missing_safe_file_is_diagnosed(spool, caplog):
    caplog.set_level(logging.INFO, logger="comfy_mcp.upload")
    created = _ready("image.png")
    session = spool / created["upload_id"]
    (session / "image.png").unlink()
    (session / "payload.ready").write_bytes(b"stale")
    result = asyncio.run(server.complete_upload(created["upload_id"]))
    assert result == {
        "kind": "upload_error",
        "upload_id": created["upload_id"],
        "stage": "resolve_staged_file",
        "message": (
            'complete_upload failed: staged file "image.png" '
            "is missing for a ready upload session."
        ),
        "retryable": False,
    }
    assert "payload.ready" in caplog.text
    assert "manifest.json" in caplog.text
    assert "ready upload has no staged safe_filename" in caplog.text
    rendered = json.dumps(result)
    assert "payload.ready" not in rendered
    assert str(session) not in rendered
    assert str(spool) not in rendered


def test_comfy_failure_names_invoke_stage(spool, monkeypatch):
    created = _ready()

    async def fake(_paths, _overwrite):
        raise ComfyCliError("disk full")

    monkeypatch.setattr(server, "_upload_local_paths", fake)
    result = asyncio.run(server.complete_upload(created["upload_id"]))
    assert result["kind"] == "upload_error"
    assert result["stage"] == "invoke_comfy_upload"
    assert result["retryable"] is True
    assert result["message"] == (
        "complete_upload failed while running comfy upload: disk full"
    )
    manifest = json.loads(
        (spool / created["upload_id"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "ready"


def test_malformed_comfy_result_names_parse_stage(spool, patched_async_run):
    created = _ready()
    patched_async_run(envelope(data={"uploads": [{}]}))
    result = asyncio.run(server.complete_upload(created["upload_id"]))
    assert result["kind"] == "upload_error"
    assert result["stage"] == "parse_comfy_result"
    assert result["retryable"] is False
    assert "cloud_name" in result["message"]
    manifest = json.loads(
        (spool / created["upload_id"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "ready"


def test_operational_error_hides_secrets(spool, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    created = _ready()
    token = "SUPERTOKEN"
    signed = "https://files.example/x?sig=SUPERSECRET"
    spool_path = "/var/lib/comfy-mcp/uploads/leak"
    env_value = "sekret"

    async def fake(_paths, _overwrite):
        raise ComfyCliError(
            f"failed Bearer {token} {signed} {spool_path} COMFY_API_KEY={env_value}"
        )

    monkeypatch.setattr(server, "_upload_local_paths", fake)
    result = asyncio.run(server.complete_upload(created["upload_id"]))
    blob = json.dumps(result) + "\n" + caplog.text
    for secret in (token, "SUPERSECRET", signed, spool_path, env_value):
        assert secret not in blob
    assert result["stage"] == "invoke_comfy_upload"
    assert "Bearer [redacted]" in result["message"]


def test_unexpected_error_logs_a_traceback(spool, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="comfy_mcp.upload")

    def boom(_upload_id):
        raise RuntimeError("invariant broken")

    monkeypatch.setattr(upload_session, "begin_complete", boom)
    with pytest.raises(RuntimeError, match="invariant broken"):
        asyncio.run(server.complete_upload("a" * 22))
    assert "Traceback" in caplog.text
    assert "stage=load_session" in caplog.text
    assert "error_type=RuntimeError" in caplog.text


def test_put_commit_logs_ready_identity(spool, caplog):
    caplog.set_level(logging.INFO, logger="comfy_mcp.upload")
    created = _ready("image.png", b"abc")
    assert f"upload_id={created['upload_id']}" in caplog.text
    assert "stage=put_commit" in caplog.text
    assert "state=ready" in caplog.text
    assert "filename=image.png" in caplog.text
    assert "bytes=3" in caplog.text
    assert "sha256=" in caplog.text


def test_startup_log_uses_injected_sha_without_git(monkeypatch, caplog):
    monkeypatch.setenv("COMFY_MCP_GIT_SHA", "abc1234")
    monkeypatch.setattr(cli, "_git_sha_cached", None)

    def git_must_not_run(*_args, **_kwargs):
        raise AssertionError("git was invoked")

    monkeypatch.setattr(cli.subprocess, "run", git_must_not_run)
    caplog.set_level(logging.INFO, logger="comfy_mcp")
    cli._log_startup("comfy-mcp")
    cli._log_startup("comfy-mcp-upload-server")
    assert "startup process=comfy-mcp " in caplog.text
    assert "startup process=comfy-mcp-upload-server " in caplog.text
    assert "git=abc1234" in caplog.text
    assert "version=" in caplog.text
