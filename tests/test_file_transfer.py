"""Remote MCP input staging and signed-output transfer regressions."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import pathlib
import tempfile
from urllib.parse import parse_qs, unquote, urlparse

import pytest
from conftest import envelope
from fastmcp import Client

from comfy_mcp import failure_log, file_transfer
from comfy_mcp.server import _internal as server


def _upload_file(*args, **kwargs):
    return asyncio.run(server.upload_file(*args, **kwargs))


def _inline(name: str, data: bytes) -> dict[str, str]:
    return {
        "name": name,
        "mimeType": "application/octet-stream",
        "data": base64.b64encode(data).decode(),
    }


def test_inline_upload_materializes_exact_bytes_and_cleans_after_success(
    patched_async_run,
):
    observed: dict[str, object] = {}

    def inspect(cmd):
        source = pathlib.Path(cmd[cmd.index("upload") + 1])
        observed["path"] = source
        observed["bytes"] = source.read_bytes()
        observed["dir_mode"] = source.parent.stat().st_mode & 0o777
        observed["file_mode"] = source.stat().st_mode & 0o777

    procs = patched_async_run(
        envelope(data={"uploads": [{"cloud_name": "task_04.webp"}]}),
        on_spawn=inspect,
    )

    result = _upload_file([_inline("task_04.webp", b"\x00\xffremote-image")])

    source = observed["path"]
    assert result == {"uploads": [{"cloud_name": "task_04.webp"}]}
    assert observed["bytes"] == b"\x00\xffremote-image"
    assert observed["dir_mode"] == 0o700
    assert observed["file_mode"] == 0o600
    assert isinstance(source, pathlib.Path) and not source.exists()
    assert procs[0].cmd[-1] == "--no-overwrite"


def test_mixed_upload_preserves_source_order_and_rewrites_only_scratch_echo(
    patched_async_run,
):
    observed: dict[str, str] = {}

    def inspect(cmd):
        observed["scratch"] = cmd[cmd.index("upload") + 2]

    # The runner's canned result cannot know the generated path, so exercise
    # the exact rewrite helper after proving mixed argv ordering separately.
    procs = patched_async_run(envelope(data={"uploads": []}), on_spawn=inspect)

    _upload_file(["/srv/input/a.png", _inline("b.webp", b"b"), "/srv/input/c.png"])

    scratch = observed["scratch"]
    assert procs[0].cmd[4:] == [
        "upload",
        "/srv/input/a.png",
        scratch,
        "/srv/input/c.png",
        "--no-overwrite",
    ]
    assert not os.path.exists(scratch)
    data = {
        "uploads": [
            {"local_path": "/srv/input/a.png", "cloud_name": "a.png"},
            {"local_path": scratch, "cloud_name": "b.webp"},
        ]
    }
    assert file_transfer.rewrite_upload_result(data, {scratch: "b.webp"}) == {
        "uploads": [
            {"local_path": "/srv/input/a.png", "cloud_name": "a.png"},
            {"local_path": "b.webp", "cloud_name": "b.webp"},
        ]
    }


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../escape.png", "a/b.png", "a\\b.png"]
)
def test_inline_upload_rejects_unsafe_filename_without_spawning(name, no_spawn):
    with pytest.raises(server.ComfyCliError, match="inline upload.*name"):
        _upload_file([_inline(name, b"x")])


def test_inline_upload_rejects_bad_base64_without_echoing_payload(no_spawn):
    payload = "SECRET_INVALID_BASE64!"
    with pytest.raises(server.ComfyCliError, match="strict base64") as caught:
        _upload_file(
            [
                {
                    "name": "bad.bin",
                    "mimeType": "application/octet-stream",
                    "data": payload,
                }
            ]
        )
    assert payload not in str(caught.value)


def test_inline_upload_enforces_decoded_total_before_spawning(no_spawn):
    oversized = b"x" * (file_transfer._MAX_INLINE_UPLOAD_BYTES + 1)
    with pytest.raises(server.ComfyCliError, match="exceeds the 2 MiB total"):
        _upload_file([_inline("too-large.bin", oversized)])


def test_inline_upload_cleans_scratch_after_engine_failure(patched_async_run):
    observed: list[str] = []

    def inspect(cmd):
        observed.append(cmd[cmd.index("upload") + 1])

    patched_async_run(
        envelope(
            ok=False,
            error={"code": "upload_failed", "message": "target refused file"},
        ),
        on_spawn=inspect,
    )

    with pytest.raises(server.ComfyCliError, match="target refused file"):
        _upload_file([_inline("failure.bin", b"bytes")])

    assert observed and not os.path.exists(observed[0])


def test_inline_upload_cancellation_reaps_child_and_scratch(
    patched_async_run, monkeypatch
):
    observed: list[pathlib.Path] = []

    def inspect(cmd):
        observed.append(pathlib.Path(cmd[cmd.index("upload") + 1]))

    procs = patched_async_run(hang=True, on_spawn=inspect)

    async def drive():
        spawned = asyncio.Event()
        fake_exec = server.asyncio.create_subprocess_exec

        async def notifying_exec(*args, **kwargs):
            proc = await fake_exec(*args, **kwargs)
            spawned.set()
            return proc

        monkeypatch.setattr(server.asyncio, "create_subprocess_exec", notifying_exec)
        task = asyncio.create_task(
            server.upload_file([_inline("cancelled.bin", b"cancel-me")])
        )
        await spawned.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())

    assert procs[0].killed is True
    assert observed and not observed[0].exists()


def test_inline_upload_failure_log_never_records_file_content(
    patched_async_run, monkeypatch, tmp_path
):
    log_path = tmp_path / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(log_path))
    content = b"private-inline-file-content"
    encoded = base64.b64encode(content).decode()
    patched_async_run(
        envelope(
            ok=False,
            error={"code": "upload_failed", "message": "target refused file"},
        )
    )

    with pytest.raises(server.ComfyCliError, match="target refused file"):
        _upload_file([_inline("failure.bin", content)])

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    serialized = json.dumps(entry)
    assert entry["error_code"] == "upload_failed"
    assert encoded not in serialized
    assert content.decode() not in serialized


def test_upload_schema_keeps_paths_required_and_advertises_base64_content():
    async def schema():
        async with Client(server.mcp, mode="legacy") as client:
            tools = await client.list_tools()
            return next(
                tool for tool in tools if tool.name == "upload_file"
            ).input_schema

    input_schema = asyncio.run(schema())
    assert "paths" in input_schema["required"]
    variants = input_schema["properties"]["paths"]["items"]["anyOf"]
    inline = next(entry for entry in variants if entry.get("type") == "object")
    assert inline["properties"]["data"]["contentEncoding"] == "base64"
    assert inline["properties"]["mimeType"]["default"] == "application/octet-stream"


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
        )

    assert not scratch.exists()
    assert outside.read_bytes() == b"outside"
