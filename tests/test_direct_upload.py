"""Direct upload sessions: init, PUT, complete, and the ChatGPT file parameter.

No public network and no live ComfyUI. The upload HTTP service is the real
loopback handler; comfy-cli is the shared async fake.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import socket
import stat
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pytest
from conftest import envelope

from comfy_mcp import file_ingress, server, upload_server, upload_session
from comfy_mcp.errors import ComfyCliError
from comfy_mcp.file_ingress import OpenAIFile


@pytest.fixture
def spool(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_MCP_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("COMFY_MCP_UPLOAD_PUBLIC_BASE_URL", "https://uploads.example")
    monkeypatch.setenv("COMFY_MCP_UPLOAD_TTL_SECONDS", "600")
    return tmp_path


def _token(result: dict) -> str:
    header = result["upload_headers"]["Authorization"]
    scheme, token = header.split(" ", 1)
    assert scheme == "Bearer"
    return token


def _manifest(root: Path, upload_id: str) -> dict:
    return json.loads((root / upload_id / "manifest.json").read_text(encoding="utf-8"))


def _init(
    filename: str = "original.png",
    size: int = 4,
    mime: str = "image/png",
    overwrite: bool = False,
):
    return upload_session.initialize(filename, size, mime, overwrite)


def _put(base: str, upload_id: str, token: str, body: bytes, mime: str = "image/png"):
    request = urllib.request.Request(
        f"{base}/upload/{upload_id}",
        data=body,
        method="PUT",
        headers={"Authorization": f"Bearer {token}", "Content-Type": mime},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@pytest.fixture
def upload_http(spool):
    httpd = upload_server.serve("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield {"base": f"http://{host}:{port}", "port": port}
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _uploaded(name: str) -> dict:
    return envelope(
        data={
            "uploads": [
                {
                    "local_path": "/tmp/ignored",
                    "cloud_name": name,
                    "subfolder": "",
                    "type": "input",
                }
            ]
        }
    )


def _public_dns(monkeypatch):
    def getaddrinfo(host, port, *args, **kwargs):
        addresses = {
            "files.example": "8.8.8.8",
            "cdn.example": "1.1.1.1",
        }
        ip = addresses.get(host)
        if ip is None:
            raise OSError(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(file_ingress.socket, "getaddrinfo", getaddrinfo)


class _Body:
    def __init__(self, data: bytes, headers: dict | None = None, status: int = 200):
        self._data = data
        self._off = 0
        self.headers = headers or {}
        self.status = status
        self.reads = 0

    def read(self, n: int) -> bytes:
        self.reads += 1
        chunk = self._data[self._off : self._off + n]
        self._off += len(chunk)
        return chunk

    def close(self) -> None:
        return None


def test_manual_init_requires_filename_size_and_mime(spool, no_spawn):
    missing = asyncio.run(server.init_upload())
    assert missing["kind"] == "upload_error"
    assert missing["stage"] == "validate_arguments"
    partial = asyncio.run(server.init_upload(filename="a.png", file_size=1))
    assert partial["kind"] == "upload_error"
    assert "file_size" in partial["message"]


def test_init_session_token_url_curl_and_expiry(spool, caplog):
    caplog.set_level(logging.DEBUG)
    first = _init()
    second = _init(size=5)
    assert first["kind"] == "upload_initialized"
    assert first["next_tool"] == "complete_upload"
    assert first["upload_id"] != second["upload_id"]
    token = _token(first)
    other = _token(second)
    assert token != other
    assert token not in first["upload_url"]
    assert first["upload_url"] == f"https://uploads.example/upload/{first['upload_id']}"
    curl = first["curl_command"]
    assert "curl --fail-with-body" in curl
    assert "--request PUT" in curl
    assert '--upload-file "$FILE"' in curl
    assert f"Authorization: Bearer {token}" in curl
    assert "Content-Type: image/png" in curl
    assert first["upload_url"] in curl
    manifest = _manifest(spool, first["upload_id"])
    assert manifest["state"] == "initialized"
    assert manifest["token_hash"] == hashlib.sha256(token.encode()).hexdigest()
    raw = (spool / first["upload_id"] / "manifest.json").read_text(encoding="utf-8")
    assert token not in raw
    assert "Bearer" not in raw
    expires = datetime.fromisoformat(first["expires_at"]).timestamp()
    assert expires > time.time()
    assert token not in caplog.text
    assert "Bearer" not in caplog.text
    mode = stat.S_IMODE((spool / first["upload_id"]).stat().st_mode)
    assert mode == 0o700
    file_mode = stat.S_IMODE(
        (spool / first["upload_id"] / "manifest.json").stat().st_mode
    )
    assert file_mode == 0o600


def test_public_base_url_is_required_for_a_session(spool, monkeypatch):
    monkeypatch.delenv("COMFY_MCP_UPLOAD_PUBLIC_BASE_URL")
    with pytest.raises(ComfyCliError, match="COMFY_MCP_UPLOAD_PUBLIC_BASE_URL"):
        _init()


def test_filename_rejects_paths_and_controls(spool):
    outside = spool.parent / "escaped.png"
    for name in (
        "",
        ".",
        "..",
        "../x.png",
        "a/b.png",
        "a\\b.png",
        "bad\x00name.png",
        "CON.txt",
    ):
        with pytest.raises(ComfyCliError):
            _init(filename=name)
    assert not outside.exists()


def test_oversize_declared_file_is_rejected(spool, monkeypatch):
    monkeypatch.setenv("COMFY_MCP_MAX_UPLOAD_MB", "1")
    with pytest.raises(ComfyCliError, match="COMFY_MCP_MAX_UPLOAD_MB"):
        _init(size=2 * 1024 * 1024)


def test_put_preserves_bytes_nul_and_sha256(spool, upload_http, caplog):
    caplog.set_level(logging.DEBUG)
    body = b"a\x00b\xff"
    created = _init(size=len(body))
    token = _token(created)
    status, payload = _put(upload_http["base"], created["upload_id"], token, body)
    assert status == 200
    parsed = json.loads(payload)
    assert parsed["byte_size"] == len(body)
    assert parsed["sha256"] == hashlib.sha256(body).hexdigest()
    ready = spool / created["upload_id"] / "original.png"
    assert ready.read_bytes() == body
    assert not (spool / created["upload_id"] / "payload.part").exists()
    assert not (spool / created["upload_id"] / "payload.ready").exists()
    manifest = _manifest(spool, created["upload_id"])
    assert manifest["state"] == "ready"
    assert manifest["token_hash"] == ""
    assert token not in caplog.text
    assert "Authorization" not in caplog.text


def test_put_rejects_wrong_expired_and_second_token(spool, upload_http):
    body = b"png!"
    created = _init(size=len(body))
    status, _ = _put(upload_http["base"], created["upload_id"], "not-the-token", body)
    assert status == 401
    assert _manifest(spool, created["upload_id"])["state"] == "initialized"
    status, _ = _put(upload_http["base"], created["upload_id"], _token(created), body)
    assert status == 200
    status, _ = _put(upload_http["base"], created["upload_id"], _token(created), body)
    assert status == 409

    expired = _init(size=len(body))
    path = spool / expired["upload_id"] / "manifest.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["expires_at_unix"] = time.time() - 5
    path.write_text(json.dumps(stored), encoding="utf-8")
    status, _ = _put(upload_http["base"], expired["upload_id"], _token(expired), body)
    assert status in {401, 404}
    assert not (spool / expired["upload_id"]).exists()


def test_put_rejects_short_oversize_and_interrupted(spool, upload_http, monkeypatch):
    created = _init(size=4)
    token = _token(created)
    header = (
        f"PUT /upload/{created['upload_id']} HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        f"Authorization: Bearer {token}\r\n"
        "Content-Type: image/png\r\n"
        "Content-Length: 4\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    sock = socket.create_connection(("127.0.0.1", upload_http["port"]))
    sock.sendall(header + b"ab")
    sock.shutdown(socket.SHUT_WR)
    sock.settimeout(5)
    sock.recv(4096)
    sock.close()
    manifest = _manifest(spool, created["upload_id"])
    assert manifest["state"] == "initialized"
    assert not (spool / created["upload_id"] / "payload.ready").exists()
    assert not (spool / created["upload_id"] / "payload.part").exists()
    assert not (spool / created["upload_id"] / "original.png").exists()

    monkeypatch.setattr(upload_session, "max_upload_bytes", lambda: 3)
    status, _ = _put(upload_http["base"], created["upload_id"], token, b"abcd")
    assert status == 413
    assert not (spool / created["upload_id"] / "payload.ready").exists()
    assert not (spool / created["upload_id"] / "original.png").exists()


def test_stream_rejects_a_chunk_past_the_limit(tmp_path):
    def read(_n: int) -> bytes:
        return b"0123456789"

    with pytest.raises(upload_session.UploadRejected):
        upload_session._stream_exact(str(tmp_path / "payload.part"), read, 10, 4)
    assert (
        not (tmp_path / "payload.part").exists()
        or (tmp_path / "payload.part").stat().st_size <= 4
    )


def test_healthz_and_no_file_browsing(upload_http, spool):
    with urllib.request.urlopen(upload_http["base"] + "/healthz") as response:
        assert response.status == 200
        assert response.read() == b"ok\n"
    created = _init(size=1)
    try:
        urllib.request.urlopen(upload_http["base"] + "/upload/" + created["upload_id"])
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("GET served an upload")


def test_query_token_is_not_accepted(upload_http, spool, caplog):
    caplog.set_level(logging.DEBUG)
    created = _init(size=1)
    token = _token(created)
    url = f"{upload_http['base']}/upload/{created['upload_id']}?access_token={token}"
    request = urllib.request.Request(url, data=b"x", method="PUT")
    try:
        urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("query token was accepted")
    assert token not in caplog.text


def test_complete_refuses_initialized_and_uploading(spool, no_spawn):
    created = asyncio.run(
        server.init_upload(filename="original.png", file_size=4, mime_type="image/png")
    )
    waiting = asyncio.run(server.complete_upload(created["upload_id"]))
    assert waiting["kind"] == "upload_error"
    assert waiting["stage"] == "validate_state"
    assert "has not received" in waiting["message"]
    stored = _manifest(spool, created["upload_id"])
    stored["state"] = "uploading"
    (spool / created["upload_id"] / "manifest.json").write_text(
        json.dumps(stored), encoding="utf-8"
    )
    transferring = asyncio.run(server.complete_upload(created["upload_id"]))
    assert transferring["kind"] == "upload_error"
    assert "still transferring" in transferring["message"]


def test_complete_uses_cloud_name_and_cleans_up(spool, upload_http, patched_async_run):
    body = b"png!"
    created = _init(filename="original.png", size=len(body), overwrite=False)
    status, _ = _put(upload_http["base"], created["upload_id"], _token(created), body)
    assert status == 200
    procs = patched_async_run(_uploaded("actual-name-accepted-by-comfy.png"))
    result = asyncio.run(server.complete_upload(created["upload_id"]))
    assert result["kind"] == "upload_complete"
    assert result["source_filename"] == "original.png"
    assert result["comfy_filename"] == "actual-name-accepted-by-comfy.png"
    assert result["byte_size"] == len(body)
    assert result["sha256"] == hashlib.sha256(body).hexdigest()
    cmd = procs[0].cmd
    assert "upload" in cmd
    assert "--no-overwrite" in cmd
    assert Path(cmd[cmd.index("upload") + 1]).name == "original.png"
    assert not (spool / created["upload_id"]).exists()


_SKIN = "Routine_skincare_matinale_devant_le_miroir2.png"


def test_put_keeps_the_safe_filename(spool, upload_http):
    body = b"\x89PNG\r\n\x00\xff"
    created = _init(filename=_SKIN, size=len(body))
    status, _ = _put(upload_http["base"], created["upload_id"], _token(created), body)
    assert status == 200
    session = spool / created["upload_id"]
    assert not (session / "payload.part").exists()
    assert not (session / "payload.ready").exists()
    assert (session / _SKIN).read_bytes() == body
    manifest = _manifest(spool, created["upload_id"])
    assert manifest["state"] == "ready"
    assert manifest["safe_filename"] == _SKIN
    assert manifest["filename"] == _SKIN
    assert manifest["byte_size"] == len(body)
    assert manifest["file_size"] == len(body)
    assert manifest["sha256"] == hashlib.sha256(body).hexdigest()


def test_complete_passes_the_safe_filename(spool, upload_http, patched_async_run):
    body = os.urandom(32)
    created = _init(filename=_SKIN, size=len(body), overwrite=False)
    assert (
        _put(upload_http["base"], created["upload_id"], _token(created), body)[0] == 200
    )
    procs = patched_async_run(_uploaded(_SKIN))
    result = asyncio.run(server.complete_upload(created["upload_id"]))
    passed = procs[0].cmd[procs[0].cmd.index("upload") + 1]
    assert Path(passed).name == _SKIN
    assert Path(passed).name != "payload.ready"
    assert result["source_filename"] == _SKIN
    assert result["comfy_filename"] == _SKIN
    assert result["byte_size"] == len(body)
    assert result["sha256"] == hashlib.sha256(body).hexdigest()


def test_complete_keeps_a_deduplicated_cloud_name(
    spool, upload_http, patched_async_run
):
    body = b"collision"
    created = _init(filename=_SKIN, size=len(body), overwrite=False)
    assert (
        _put(upload_http["base"], created["upload_id"], _token(created), body)[0] == 200
    )
    cloud = "Routine_skincare_matinale_devant_le_miroir2_1.png"
    procs = patched_async_run(_uploaded(cloud))
    result = asyncio.run(server.complete_upload(created["upload_id"]))
    passed = procs[0].cmd[procs[0].cmd.index("upload") + 1]
    assert Path(passed).name == _SKIN
    assert result["source_filename"] == _SKIN
    assert result["comfy_filename"] == cloud


def test_complete_preserves_overwrite(spool, upload_http, patched_async_run):
    body = b"png!"
    created = _init(size=len(body), overwrite=True)
    assert (
        _put(upload_http["base"], created["upload_id"], _token(created), body)[0] == 200
    )
    procs = patched_async_run(_uploaded("kept.png"))
    asyncio.run(server.complete_upload(created["upload_id"]))
    assert "--overwrite" in procs[0].cmd
    assert "--no-overwrite" not in procs[0].cmd


def test_complete_failure_keeps_the_staged_file(spool, upload_http, patched_async_run):
    body = b"png!"
    created = _init(size=len(body))
    assert (
        _put(upload_http["base"], created["upload_id"], _token(created), body)[0] == 200
    )
    patched_async_run(
        envelope(ok=False, error={"code": "failed", "message": "disk full"})
    )
    failed = asyncio.run(server.complete_upload(created["upload_id"]))
    assert failed["kind"] == "upload_error"
    assert failed["stage"] == "invoke_comfy_upload"
    assert "disk full" in failed["message"]
    manifest = _manifest(spool, created["upload_id"])
    assert manifest["state"] == "ready"
    assert manifest["last_error"] == "comfy upload failed"
    assert (spool / created["upload_id"] / "original.png").read_bytes() == body


def test_double_complete_is_rejected(spool, monkeypatch):
    body = b"png!"
    created = _init(size=len(body))
    upload_session.accept_put(
        created["upload_id"],
        f"Bearer {_token(created)}",
        str(len(body)),
        "image/png",
        _reader(body),
    )
    hold = threading.Event()
    entered = threading.Event()

    async def fake(_paths, _overwrite):
        entered.set()
        await asyncio.to_thread(hold.wait, 2)
        return {"uploads": [{"cloud_name": "out.png"}]}

    monkeypatch.setattr(server, "_upload_local_paths", fake)
    outcome: dict = {}

    def first() -> None:
        try:
            outcome["ok"] = asyncio.run(server.complete_upload(created["upload_id"]))
        except Exception as exc:  # noqa: BLE001 - the test records whichever side lost
            outcome["first_error"] = str(exc)

    def second() -> None:
        assert entered.wait(2)
        try:
            outcome["second"] = asyncio.run(
                server.complete_upload(created["upload_id"])
            )
        finally:
            hold.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert outcome["ok"]["comfy_filename"] == "out.png"
    assert outcome["second"]["kind"] == "upload_error"
    assert "already being completed" in outcome["second"]["message"]


def test_expired_sessions_and_partials_are_removed(spool):
    fresh = _init(size=2)
    stale_ready = _init(size=2)
    stale_part = _init(size=2)
    _age(spool, stale_ready["upload_id"], "ready")
    (spool / stale_ready["upload_id"] / "payload.ready").write_bytes(b"ok")
    _age(spool, stale_part["upload_id"], "uploading")
    (spool / stale_part["upload_id"] / "payload.part").write_bytes(b"no")
    (spool / "notes.txt").write_text("keep", encoding="utf-8")
    (spool / "not-a-session").mkdir()
    upload_session.cleanup_expired()
    assert (spool / fresh["upload_id"]).is_dir()
    assert not (spool / stale_ready["upload_id"]).exists()
    assert not (spool / stale_part["upload_id"]).exists()
    assert (spool / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert (spool / "not-a-session").is_dir()


def test_openai_file_params_schema():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    assert "upload_file_base64" not in tools
    assert "upload_file_reference" not in tools
    init = tools["init_upload"].model_dump(by_alias=True)
    assert init["_meta"]["openai/fileParams"] == ["file"]
    schema = init["inputSchema"]
    assert "file" in schema["properties"]
    assert "file" not in schema.get("required", [])
    file_schema = _object_schema(schema["properties"]["file"], schema)
    assert set(file_schema["properties"]) == {
        "download_url",
        "file_id",
        "mime_type",
        "file_name",
    }
    assert file_schema["properties"]["download_url"] == {"type": "string"}
    assert file_schema["properties"]["file_name"] == {"type": "string"}
    assert file_schema["required"] == ["download_url", "file_id"]
    assert file_schema["additionalProperties"] is False
    upload = tools["upload_file"].model_dump(by_alias=True)
    assert "paths" in upload["inputSchema"]["required"]
    complete = tools["complete_upload"].model_dump(by_alias=True)
    assert list(complete["inputSchema"]["properties"]) == ["upload_id"]
    assert complete["inputSchema"]["required"] == ["upload_id"]


def test_chatgpt_file_uploads_original_bytes_without_complete(
    spool, monkeypatch, patched_async_run
):
    monkeypatch.delenv("COMFY_MCP_UPLOAD_PUBLIC_BASE_URL", raising=False)
    _public_dns(monkeypatch)
    payload = b"GIF89a\x00\xff"
    seen: dict = {}

    def open_once(url, _timeout):
        assert url == "https://files.example/obj"
        return _Body(payload)

    monkeypatch.setattr(file_ingress, "_open_once", open_once)

    def on_spawn(cmd):
        path = next(
            part
            for part in cmd
            if part.startswith(os.sep) and "comfy-mcp-upload-" in part
        )
        seen["bytes"] = Path(path).read_bytes()
        seen["cmd"] = cmd

    patched_async_run(_uploaded("from-comfy.png"), on_spawn=on_spawn)
    result = asyncio.run(
        server.init_upload(
            file=OpenAIFile(
                download_url="https://files.example/obj",
                file_id="file_123",
                mime_type="image/gif",
                file_name="original.gif",
            )
        )
    )
    assert result["kind"] == "upload_complete"
    assert result["comfy_filename"] == "from-comfy.png"
    assert result["source_filename"] == "original.gif"
    assert seen["bytes"] == payload
    assert "--no-overwrite" in seen["cmd"]
    assert list(spool.iterdir()) == []


def test_chatgpt_passes_the_sanitized_file_name(monkeypatch, patched_async_run):
    _public_dns(monkeypatch)
    monkeypatch.setattr(file_ingress, "_open_once", lambda url, timeout: _Body(b"ref"))
    seen: dict = {}

    def on_spawn(cmd):
        path = next(part for part in cmd if "comfy-mcp-upload-" in part)
        seen["name"] = Path(path).name
        seen["bytes"] = Path(path).read_bytes()

    patched_async_run(_uploaded("reference.png"), on_spawn=on_spawn)
    result = asyncio.run(
        server.init_upload(
            file=OpenAIFile(
                download_url="https://files.example/obj",
                file_id="file_123",
                mime_type="image/png",
                file_name="reference.png",
            )
        )
    )
    assert seen["name"] == "reference.png"
    assert seen["name"] not in {"payload", "payload.ready", "download.tmp"}
    assert seen["bytes"] == b"ref"
    assert result["source_filename"] == "reference.png"
    assert result["comfy_filename"] == "reference.png"


def test_chatgpt_unsafe_name_is_not_used_as_the_destination(
    tmp_path, monkeypatch, patched_async_run
):
    _public_dns(monkeypatch)
    monkeypatch.setattr(file_ingress, "_open_once", lambda url, timeout: _Body(b"xyz"))
    seen: dict = {}

    def on_spawn(cmd):
        path = next(part for part in cmd if "comfy-mcp-upload-" in part)
        seen["name"] = Path(path).name

    patched_async_run(_uploaded("safe.png"), on_spawn=on_spawn)
    result = asyncio.run(
        server.init_upload(
            file=OpenAIFile(
                download_url="https://files.example/obj",
                file_id="file_123",
                file_name="../../etc/passwd",
            )
        )
    )
    assert seen["name"] == "upload.bin"
    assert result["source_filename"] == "upload.bin"
    assert result["comfy_filename"] == "safe.png"


def test_chatgpt_signed_url_is_absent_from_logs_and_errors(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    secret = "https://files.example/obj?X-Amz-Signature=SUPERSECRET"
    _public_dns(monkeypatch)

    def open_once(url, _timeout):
        raise urllib.error.URLError(url)

    monkeypatch.setattr(file_ingress, "_open_once", open_once)
    with pytest.raises(ComfyCliError) as raised:
        file_ingress.stage_openai_file(
            OpenAIFile(download_url=secret, file_id="file_123", file_name="a.png"),
            "/tmp",
        )
    assert "SUPERSECRET" not in str(raised.value)
    assert "SUPERSECRET" not in caplog.text
    assert secret not in caplog.text


def test_chatgpt_rejects_private_targets_and_checks_redirects(tmp_path, monkeypatch):
    opened: list[str] = []

    def getaddrinfo(host, port, *args, **kwargs):
        if host == "files.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]
        if host == "mixed.example":
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", port)),
            ]
        raise OSError(host)

    monkeypatch.setattr(file_ingress.socket, "getaddrinfo", getaddrinfo)

    def open_once(url, _timeout):
        opened.append(url)
        if url.endswith("/start"):
            raise file_ingress._SeenRedirect("https://127.0.0.1/internal")
        if url.endswith("/ok"):
            raise file_ingress._SeenRedirect("https://cdn.example/obj")
        return _Body(b"ok-bytes")

    monkeypatch.setattr(file_ingress, "_open_once", open_once)
    blocked = [
        "http://files.example/a",
        "https://127.0.0.1/a",
        "https://10.1.2.3/a",
        "https://192.168.0.5/a",
        "https://172.16.0.5/a",
        "https://169.254.169.254/a",
        "https://224.0.0.1/a",
        "https://0.0.0.0/a",
        "https://[::1]/a",
        "https://[::ffff:127.0.0.1]/a",
        "https://localhost/a",
        "https://printer.local/a",
        "https://<user>:<pass>@files.example/a",
        "https://mixed.example/a",
    ]
    for url in blocked:
        opened.clear()
        with pytest.raises(ComfyCliError):
            file_ingress.stage_openai_file(
                OpenAIFile(download_url=url, file_id="file_123"),
                str(tmp_path),
            )
        assert opened == []

    opened.clear()
    with pytest.raises(ComfyCliError):
        file_ingress.stage_openai_file(
            OpenAIFile(download_url="https://files.example/start", file_id="file_123"),
            str(tmp_path),
        )
    assert opened == ["https://files.example/start"]

    def getaddrinfo_cdn(host, port, *args, **kwargs):
        if host in {"files.example", "cdn.example"}:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]
        raise OSError(host)

    monkeypatch.setattr(file_ingress.socket, "getaddrinfo", getaddrinfo_cdn)
    opened.clear()
    (tmp_path / "stage").mkdir()
    staged = file_ingress.stage_openai_file(
        OpenAIFile(
            download_url="https://files.example/ok",
            file_id="file_123",
            file_name="a.png",
        ),
        str(tmp_path / "stage"),
    )
    assert staged.byte_size == len(b"ok-bytes")
    assert Path(staged.path).read_bytes() == b"ok-bytes"
    assert "https://127.0.0.1/internal" not in opened


def _reader(body: bytes):
    pending = {"left": body}

    def read(n: int) -> bytes:
        chunk = pending["left"][:n]
        pending["left"] = pending["left"][n:]
        return chunk

    return read


def _age(root: Path, upload_id: str, state: str) -> None:
    path = root / upload_id / "manifest.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["state"] = state
    stored["expires_at_unix"] = time.time() - 5
    path.write_text(json.dumps(stored), encoding="utf-8")


def _object_schema(schema: dict, root: dict) -> dict:
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return root["$defs"][name]
    for option in schema.get("anyOf", []):
        if option.get("type") == "null":
            continue
        return _object_schema(option, root)
    return schema
