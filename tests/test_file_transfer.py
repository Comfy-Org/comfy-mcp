"""Remote MCP input staging and signed-output transfer regressions."""

from __future__ import annotations

import asyncio
import json
import pathlib
import re
import tempfile
from urllib.parse import parse_qs, unquote, urlparse

import pytest
from fastmcp import Client
from starlette.requests import Request

from comfy_mcp import file_transfer
from comfy_mcp.server import _internal as server


def _request(token: str, body: bytes) -> Request:
    return _chunked_request(token, body)


def _chunked_request(token: str, *chunks: bytes) -> Request:
    index = 0

    async def receive():
        nonlocal index
        if index < len(chunks):
            body = chunks[index]
            index += 1
            return {
                "type": "http.request",
                "body": body,
                "more_body": index < len(chunks),
            }
        return {"type": "http.disconnect"}

    return Request(
        {
            "type": "http",
            "method": "PUT",
            "path": f"/api/uploads/{token}",
            "headers": [],
            "path_params": {"token": token},
        },
        receive,
    )


def _mint(path: str = "/client/task_04.webp", os_name: str = "linux") -> str:
    request = file_transfer.prepare_upload_request(path, os_name)
    return file_transfer.mint_upload_command(
        request,
        base_url="http://127.0.0.1:9000",
    )


def _token(message: str) -> str:
    match = re.search(r"/api/uploads/([A-Za-z0-9_-]+)", message)
    assert match is not None
    return match.group(1)


def test_upload_message_matches_cloud_command_contract():
    message = _mint()

    assert "Run this command via Bash" in message
    assert "credential-free" in message
    assert "curl -sS --fail-with-body -X PUT" in message
    assert "Content-Type: image/webp" in message
    assert "--upload-file '/client/task_04.webp'" in message
    assert "-- 'http://127.0.0.1:9000/api/uploads/" in message
    assert "do NOT add any Authorization header" in message
    assert '{"name":"...","subfolder":"","type":"input"}' in message
    assert "single-use" in message and "50 MiB" in message
    assert "same MCP host" in message


def test_upload_message_uses_windows_path_and_curl_exe():
    message = _mint(r"C:\Users\me\task 04.png", "windows")

    assert "Run this command via PowerShell" in message
    assert "curl.exe" in message
    assert '"C:\\Users\\me\\task 04.png"' in message
    assert '"Content-Type: image/png"' in message


@pytest.mark.parametrize(
    ("path", "os_name", "match"),
    [
        ("relative.png", "linux", "absolute path"),
        ("/client/file.txt", "linux", "accepts image files"),
        (r"C:\client\file.png", "linux", "absolute path"),
        ("/client/file.png", "plan9", "invalid client_os"),
    ],
)
def test_upload_request_rejects_non_cloud_inputs(path, os_name, match):
    with pytest.raises(server.ComfyCliError, match=match):
        file_transfer.prepare_upload_request(path, os_name)


def test_upload_capability_puts_exact_bytes_through_cli_and_is_single_use():
    message = _mint()
    token = _token(message)
    observed: dict[str, object] = {}

    def inspect(path: str) -> None:
        source = pathlib.Path(path)
        observed["path"] = source
        observed["bytes"] = source.read_bytes()
        observed["dir_mode"] = source.parent.stat().st_mode & 0o777
        observed["file_mode"] = source.stat().st_mode & 0o777

    async def uploader(path: str):
        await asyncio.to_thread(inspect, path)
        return {"uploads": [{"cloud_name": "task_04.webp"}]}

    response = asyncio.run(
        file_transfer.receive_upload(_request(token, b"remote-image"), uploader)
    )
    repeated = asyncio.run(
        file_transfer.receive_upload(_request(token, b"again"), uploader)
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "name": "task_04.webp",
        "subfolder": "",
        "type": "input",
    }
    assert observed["bytes"] == b"remote-image"
    assert observed["dir_mode"] == 0o700
    assert observed["file_mode"] == 0o600
    assert isinstance(observed["path"], pathlib.Path)
    assert not observed["path"].exists()
    assert repeated.status_code == 404


def test_upload_capability_streams_request_chunks_to_scratch(monkeypatch):
    token = _token(_mint())
    write_sizes: list[int] = []
    open_owner_only = file_transfer._open_owner_only

    class RecordingFile:
        def __init__(self, handle):
            self._handle = handle

        def write(self, data: bytes):
            write_sizes.append(len(data))
            return self._handle.write(data)

        def close(self):
            return self._handle.close()

    def open_recording_file(path: str):
        return RecordingFile(open_owner_only(path))

    monkeypatch.setattr(file_transfer, "_open_owner_only", open_recording_file)

    async def uploader(path: str):
        uploaded = await asyncio.to_thread(pathlib.Path(path).read_bytes)
        assert uploaded == b"remote-image"
        return {"uploads": [{"cloud_name": "task_04.webp"}]}

    response = asyncio.run(
        file_transfer.receive_upload(
            _chunked_request(token, b"remote-", b"image"),
            uploader,
        )
    )

    assert response.status_code == 200
    assert write_sizes == [7, 5]


def test_upload_capability_size_failure_consumes_url(monkeypatch):
    monkeypatch.setattr(file_transfer, "_MAX_UPLOAD_BYTES", 3)
    token = _token(_mint())
    called = False

    async def uploader(path: str):
        nonlocal called
        called = True

    response = asyncio.run(
        file_transfer.receive_upload(_request(token, b"four"), uploader)
    )
    repeated = asyncio.run(
        file_transfer.receive_upload(_request(token, b"x"), uploader)
    )

    assert response.status_code == 413
    assert repeated.status_code == 404
    assert called is False


def test_upload_schema_matches_comfycloud_file_path_and_client_os():
    async def schema():
        async with Client(server.mcp, mode="legacy") as client:
            tools = await client.list_tools()
            return next(
                tool for tool in tools if tool.name == "upload_file"
            ).input_schema

    input_schema = asyncio.run(schema())
    assert input_schema["required"] == ["file_path", "client_os"]
    assert input_schema["properties"]["file_path"]["type"] == "string"
    assert input_schema["properties"]["client_os"]["enum"] == [
        "darwin",
        "linux",
        "windows",
    ]
    assert "paths" not in input_schema["properties"]


def test_output_schema_exposes_comfycloud_client_os_selector():
    async def schema():
        async with Client(server.mcp, mode="legacy") as client:
            tools = await client.list_tools()
            return next(
                tool for tool in tools if tool.name == "fetch_outputs"
            ).input_schema

    input_schema = asyncio.run(schema())
    client_os = input_schema["properties"]["client_os"]
    assert client_os["enum"] == ["darwin", "linux", "windows"]
    assert client_os["default"] == "darwin"


def test_signed_download_rejects_expired_url_and_removes_scratch(monkeypatch, tmp_path):
    now = 10_000
    monkeypatch.setattr(file_transfer.time, "time", lambda: now)
    directory = tempfile.TemporaryDirectory(dir=tmp_path)
    scratch = pathlib.Path(directory.name)
    (scratch / "result.png").write_bytes(b"png")
    store = file_transfer._SignedDownloadStore(ttl_seconds=10)
    result = store.publish(
        directory,
        {"files": [{"path": "result.png"}]},
        base_url="http://127.0.0.1:9000",
        client_out_dir="/client/results",
        client_os="linux",
    )
    parsed = urlparse(result["files"][0]["url"])
    token, filename = parsed.path.rstrip("/").rsplit("/", 2)[-2:]
    query = parse_qs(parsed.query)
    expires = int(query["expires"][0])
    signature = query["signature"][0]

    lease = store.acquire(token, unquote(filename), expires, signature)
    assert lease is not None
    store.release(lease)
    monkeypatch.setattr(file_transfer.time, "time", lambda: now + 11)
    assert store.acquire(token, unquote(filename), expires, signature) is None
    assert not scratch.exists()


def test_signed_download_releases_lease_when_client_send_fails(monkeypatch, tmp_path):
    directory = tempfile.TemporaryDirectory(dir=tmp_path)
    scratch = pathlib.Path(directory.name)
    (scratch / "result.png").write_bytes(b"png")
    store = file_transfer._SignedDownloadStore(ttl_seconds=30)
    monkeypatch.setattr(file_transfer, "_DOWNLOAD_STORE", store)

    try:
        result = store.publish(
            directory,
            {"files": [{"path": "result.png"}]},
            base_url="http://127.0.0.1:9000",
            client_out_dir="/client/results",
            client_os="linux",
        )
        parsed = urlparse(result["files"][0]["url"])
        token, filename = parsed.path.rstrip("/").rsplit("/", 2)[-2:]
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": parsed.path,
                "query_string": parsed.query.encode(),
                "headers": [],
                "path_params": {
                    "token": token,
                    "filename": unquote(filename),
                },
            }
        )
        response = asyncio.run(file_transfer.serve_signed_download(request))

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        with pytest.raises(OSError, match="client disconnected"):
            asyncio.run(response(request.scope, receive, send))

        # Expiry/close may now remove the directory because the response's
        # finally block returned the active lease even though body delivery failed.
        store.close()
        assert not scratch.exists()
    finally:
        store.close()
        directory.cleanup()


def test_signed_download_refuses_engine_path_outside_scratch(tmp_path):
    directory = tempfile.TemporaryDirectory(dir=tmp_path)
    scratch = pathlib.Path(directory.name)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    store = file_transfer._SignedDownloadStore()

    with pytest.raises(server.ComfyCliError, match="outside its.*scratch"):
        store.publish(
            directory,
            {"files": [{"path": str(outside)}]},
            base_url="http://127.0.0.1:9000",
            client_out_dir="/client/results",
            client_os="linux",
        )

    assert not scratch.exists()
    assert outside.read_bytes() == b"outside"
