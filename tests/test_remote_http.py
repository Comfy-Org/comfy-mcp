"""Streamable HTTP integration over the one shared FastMCP application."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx2
import pytest
import uvicorn
from conftest import envelope
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.client.transports import StreamableHttpTransport
from starlette.requests import Request

from comfy_mcp import failure_log, file_transfer
from comfy_mcp.server import _internal as server
from comfy_mcp.server import remote
from comfy_mcp.server.config import RemoteServerConfig


@pytest.fixture
def remote_http_server():
    """Run the production SDK app on an ephemeral loopback port."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    settings = RemoteServerConfig(port=port, log_level="CRITICAL")
    app = remote.create_http_app(settings, server.mcp)
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            log_level="critical",
            access_log=False,
        )
    )

    thread = threading.Thread(
        target=lambda: asyncio.run(uvicorn_server.serve(sockets=[sock])),
        name="comfy-mcp-http-test",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while (
        not uvicorn_server.started and thread.is_alive() and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert uvicorn_server.started, "uvicorn did not start the Remote MCP test server"

    yield settings.endpoint_url

    uvicorn_server.should_exit = True
    thread.join(timeout=5)
    sock.close()
    assert not thread.is_alive(), "Remote MCP test server did not stop"


async def _discover_and_call(
    url: str,
    name: str,
    arguments=None,
    mode="legacy",
    *,
    raise_on_error: bool = True,
):
    """A fresh client performs negotiation/discovery before one call."""

    async with Client(url, mode=mode) as client:
        discovered = await client.list_tools()
        result = await client.call_tool(
            name,
            arguments,
            raise_on_error=raise_on_error,
        )
    return discovered, result


def test_http_exposes_the_same_40_tools_and_complete_business_flow(
    remote_http_server, patched_run, patched_async_run, monkeypatch
):
    """HTTP carries input bytes in and signed output bytes back on one app."""

    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")
    upload_bytes = b"RIFF\x00\x01fake-webp"
    upload_scratch: list[str] = []
    download_scratch: list[str] = []

    async def run_flow():
        async with Client(remote_http_server, mode="legacy") as client:
            discovered = await client.list_tools()

            info_calls = patched_run(
                envelope(data={"running": True, "url": "http://127.0.0.1:8188"})
            )
            info = await client.call_tool("server_info", {})

            def inspect_upload(cmd):
                source = cmd[cmd.index("upload") + 1]
                upload_scratch.append(source)
                assert pathlib.Path(source).read_bytes() == upload_bytes

            upload_procs = patched_async_run(
                envelope(
                    data={
                        "uploads": [
                            {
                                "local_path": "task_04.webp",
                                "cloud_name": "task_04.webp",
                            }
                        ]
                    }
                ),
                on_spawn=inspect_upload,
            )
            uploaded = await client.call_tool(
                "upload_file",
                {
                    "file_path": "/client/inputs/task_04.webp",
                    "client_os": "linux",
                },
            )
            upload_message = uploaded.content[0].text
            upload_url_match = re.search(
                r"https?://[^'\s]+/api/uploads/[\w-]+", upload_message
            )
            assert upload_url_match is not None
            upload_url = upload_url_match.group(0)
            async with httpx2.AsyncClient() as http_client:
                upload_response = await http_client.put(
                    upload_url,
                    content=upload_bytes,
                    headers={"Content-Type": "image/webp"},
                )
                repeated_upload = await http_client.put(
                    upload_url,
                    content=upload_bytes,
                    headers={"Content-Type": "image/webp"},
                )

            submit_calls = patched_run(envelope(data={"prompt_id": "prompt-http"}))
            submitted = await client.call_tool(
                "run_workflow",
                {
                    "workflow_path": "/srv/workflows/smoke.json",
                    "wait": False,
                },
            )

            status_calls = patched_run(
                envelope(data={"prompt_id": "prompt-http", "status": "completed"})
            )
            status = await client.call_tool(
                "job", {"action": "status", "prompt_id": "prompt-http"}
            )

            def write_output(cmd):
                scratch = cmd[cmd.index("-o") + 1]
                path = pathlib.Path(scratch) / "result.png"
                path.write_bytes(b"\x89PNG\r\n\x1a\nremote-result")
                download_scratch.append(scratch)

            output_calls = patched_run(
                envelope(
                    data={
                        "prompt_id": "prompt-http",
                        "out_dir": "unused-engine-path",
                        "files": [{"path": "result.png", "size": 21}],
                    }
                ),
                on_spawn=write_output,
            )
            outputs = await client.call_tool(
                "fetch_outputs",
                {
                    "prompt_id": "prompt-http",
                    "out_dir": "/tmp/results",
                    "client_os": "linux",
                },
            )
            async with httpx2.AsyncClient() as http_client:
                downloaded = await http_client.get(outputs.data["files"][0]["url"])
                tampered = await http_client.get(
                    outputs.data["files"][0]["url"].replace(
                        "signature=", "signature=0", 1
                    )
                )
        return (
            discovered,
            info,
            uploaded,
            upload_message,
            upload_response,
            repeated_upload,
            submitted,
            status,
            outputs,
            downloaded,
            tampered,
            info_calls,
            upload_procs,
            submit_calls,
            status_calls,
            output_calls,
        )

    (
        discovered,
        info,
        uploaded,
        upload_message,
        upload_response,
        repeated_upload,
        submitted,
        status,
        outputs,
        downloaded,
        tampered,
        info_calls,
        upload_procs,
        submit_calls,
        status_calls,
        output_calls,
    ) = asyncio.run(run_flow())

    names = {tool.name for tool in discovered}
    assert len(names) == 40
    assert {
        "server_info",
        "upload_file",
        "run_workflow",
        "job",
        "fetch_outputs",
    } <= names
    assert info.data["running"] is True
    assert "curl -sS --fail-with-body -X PUT" in upload_message
    assert "/client/inputs/task_04.webp" in upload_message
    assert upload_response.json() == {
        "name": "task_04.webp",
        "subfolder": "",
        "type": "input",
    }
    assert repeated_upload.status_code == 404
    assert submitted.data["prompt_id"] == "prompt-http"
    assert status.data["status"] == "completed"
    output = outputs.data["files"][0]
    assert output["path"] == "/tmp/results/result.png"
    assert output["url"].startswith(remote_http_server.removesuffix("/mcp"))
    assert "expires=" in output["url"] and "signature=" in output["url"]
    assert output["url"] in output["command"]
    assert output["download_url"] == output["url"]
    assert output["download_command"] == output["command"]
    assert outputs.data["download_command"] == output["command"]
    assert "Temporary download URL" in outputs.content[0].text
    assert output["url"] in outputs.content[0].text
    assert outputs.data["download_url_ttl_seconds"] == 300
    assert 1 <= output["expires_at"] - int(time.time()) <= 300
    assert f'"{output["url"]}"' in output["windows_command"]
    assert downloaded.status_code == 200
    assert downloaded.content == b"\x89PNG\r\n\x1a\nremote-result"
    assert tampered.status_code == 404
    assert info_calls[0]["cmd"][4:] == ["env"]
    assert upload_procs[0].cmd[4] == "upload"
    assert upload_procs[0].cmd[-1] == "--no-overwrite"
    assert upload_scratch and not pathlib.Path(upload_scratch[0]).exists()
    assert submit_calls[0]["cmd"][4:] == [
        "run",
        "--workflow",
        "/srv/workflows/smoke.json",
    ]
    assert status_calls[0]["cmd"][4:] == ["jobs", "status", "prompt-http"]
    assert output_calls[0]["cmd"][4:] == [
        "download",
        "prompt-http",
        "-o",
        download_scratch[0],
    ]
    assert pathlib.Path(download_scratch[0]).exists()
    file_transfer._DOWNLOAD_STORE.close()
    assert not pathlib.Path(download_scratch[0]).exists()


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_http_upload_mints_command_without_reading_client_path(
    mode, remote_http_server, patched_async_run
):
    procs = patched_async_run(envelope(data={"uploads": []}))

    _, result = asyncio.run(
        _discover_and_call(
            remote_http_server,
            "upload_file",
            {
                "file_path": "/client/inputs/task_04.webp",
                "client_os": "linux",
            },
            mode=mode,
        )
    )

    assert result.is_error is False
    assert "/api/uploads/" in result.content[0].text
    assert "Base64/data-inline upload is unsupported" not in result.content[0].text
    assert procs == []


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (
            {"forwarded": ('for=192.0.2.1;proto=https;host="mcp.example.test:8443"')},
            "https://mcp.example.test:8443",
        ),
        (
            {
                "x-forwarded-proto": "https",
                "x-forwarded-host": "mcp.example.test",
                "x-forwarded-port": "8443",
            },
            "https://mcp.example.test:8443",
        ),
        (
            {
                "x-forwarded-proto": "https, http",
                "x-forwarded-host": "mcp.example.test:9443, proxy.internal",
                "x-forwarded-port": "8443, 6008",
            },
            "https://mcp.example.test:9443",
        ),
    ],
)
def test_remote_http_base_url_reads_complete_client_facing_proxy_origin(
    headers, expected, monkeypatch
):
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("127.0.0.1", 6008),
            "path": "/mcp",
            "headers": [
                (name.encode(), value.encode())
                for name, value in {"host": "127.0.0.1:6008", **headers}.items()
            ],
        }
    )
    monkeypatch.setattr(server, "get_http_request", lambda: request)

    assert server._remote_http_base_url() == expected


def test_remote_http_base_url_rejects_malformed_proxy_authority(monkeypatch):
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("127.0.0.1", 6008),
            "path": "/mcp",
            "headers": [
                (b"host", b"127.0.0.1:6008"),
                (b"x-forwarded-proto", b"javascript"),
                (b"x-forwarded-host", b"attacker.example/path"),
                (b"x-forwarded-port", b"70000"),
            ],
        }
    )
    monkeypatch.setattr(server, "get_http_request", lambda: request)

    assert server._remote_http_base_url() == "http://127.0.0.1:6008"


def test_remote_http_base_url_prefers_complete_operator_public_url(monkeypatch):
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("127.0.0.1", 6008),
            "path": "/mcp",
            "headers": [(b"host", b"proxy-rewritten.example")],
        }
    )
    monkeypatch.setattr(server, "get_http_request", lambda: request)
    monkeypatch.setenv(
        "COMFY_MCP_PUBLIC_URL",
        "https://mcp.example.test:8443/",
    )

    assert server._remote_http_base_url() == "https://mcp.example.test:8443"


@pytest.mark.parametrize(
    "value",
    [
        "ftp://mcp.example.test:8443",
        "https://user:pass@mcp.example.test:8443",
        "https://mcp.example.test:8443/prefix",
        "https://mcp.example.test:8443/?token=secret",
        "https://mcp.example.test:70000",
    ],
)
def test_remote_http_base_url_rejects_invalid_operator_public_url(value, monkeypatch):
    monkeypatch.setenv("COMFY_MCP_PUBLIC_URL", value)

    with pytest.raises(server.ComfyCliError, match="COMFY_MCP_PUBLIC_URL must"):
        server._remote_http_base_url()


def test_http_transfer_reports_preserve_forwarded_scheme_host_and_port(
    remote_http_server, patched_run, monkeypatch, tmp_path
):
    """Both Cloud-style transfer directions use the public proxy origin."""

    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")

    def write_output(cmd):
        scratch = pathlib.Path(cmd[cmd.index("-o") + 1])
        (scratch / "result.png").write_bytes(b"public-origin-output")

    patched_run(
        envelope(data={"files": [{"path": "result.png"}]}),
        on_spawn=write_output,
    )
    transport = StreamableHttpTransport(
        remote_http_server,
        headers={
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "mcp.example.test",
            "X-Forwarded-Port": "8443",
        },
    )

    async def read_reports():
        async with Client(transport, mode="legacy") as client:
            uploaded = await client.call_tool(
                "upload_file",
                {
                    "file_path": "/client/input.png",
                    "client_os": "linux",
                },
            )
            outputs = await client.call_tool(
                "fetch_outputs",
                {
                    "prompt_id": "public-origin",
                    "out_dir": str(tmp_path / "client"),
                    "client_os": "linux",
                },
            )
        return uploaded, outputs

    uploaded, outputs = asyncio.run(read_reports())

    assert "https://mcp.example.test:8443/api/uploads/" in uploaded.content[0].text
    assert outputs.data["files"][0]["url"].startswith(
        "https://mcp.example.test:8443/downloads/"
    )
    file_transfer._DOWNLOAD_STORE.close()


async def _accept_approval(message, response_type, params, ctx):
    return ElicitResult(action="accept", content=response_type(approve=True))


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_http_paid_workflow_round_trips_approval_on_both_protocols(
    mode, remote_http_server, patched_run, monkeypatch
):
    """The network adapter preserves the shared app's spend interlock."""

    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: True)
    calls = patched_run(envelope(data={"prompt_id": f"prompt-{mode}"}))

    async def run_paid_workflow():
        async with Client(
            remote_http_server,
            mode=mode,
            elicitation_handler=_accept_approval,
        ) as client:
            result = await client.call_tool(
                "run_workflow",
                {
                    "workflow_path": "/srv/workflows/paid.json",
                    "wait": False,
                    "confirm_spend": True,
                },
            )
            return client.protocol_version, result

    protocol, result = asyncio.run(run_paid_workflow())

    assert result.data["prompt_id"] == f"prompt-{mode}"
    assert (str(protocol) == "2026-07-28") is (mode == "auto")
    assert calls[0]["cmd"][4:] == [
        "run",
        "--workflow",
        "/srv/workflows/paid.json",
        "--allow-spend",
    ]


def test_http_comfy_cli_failure_uses_the_same_tool_error_contract(
    remote_http_server, patched_run
):
    patched_run(
        envelope(
            ok=False,
            error={"code": "server_not_running", "message": "ComfyUI is offline"},
        )
    )

    _, result = asyncio.run(
        _discover_and_call(
            remote_http_server,
            "run_workflow",
            {
                "workflow_path": "/srv/workflows/smoke.json",
                "wait": False,
            },
            raise_on_error=False,
        )
    )

    assert result.is_error is True
    assert "server_not_running" in str(result)
    assert "ComfyUI is offline" in str(result)


def test_http_comfy_cli_failure_reaches_the_shared_optin_failure_log(
    remote_http_server, patched_run, monkeypatch, tmp_path
):
    log_path = tmp_path / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(log_path))
    patched_run(
        envelope(
            ok=False,
            error={"code": "server_not_running", "message": "ComfyUI is offline"},
        )
    )

    _, result = asyncio.run(
        _discover_and_call(
            remote_http_server,
            "run_workflow",
            {
                "workflow_path": "/srv/workflows/smoke.json",
                "wait": False,
            },
            raise_on_error=False,
        )
    )

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert result.is_error is True
    assert entry["kind"] == "error_envelope"
    assert entry["error_code"] == "server_not_running"
    assert entry["args"] == [
        "run",
        "--workflow",
        "/srv/workflows/smoke.json",
    ]
    assert "ComfyUI is offline" in entry["message"]


def test_http_upload_failure_log_never_records_uploaded_bytes(
    remote_http_server, patched_async_run, monkeypatch, tmp_path
):
    """The observer sees comfy-cli diagnostics, never the capability PUT body."""

    log_path = tmp_path / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(log_path))
    procs = patched_async_run(
        envelope(
            ok=False,
            error={"code": "upload_failed", "message": "ComfyUI rejected upload"},
        )
    )
    private_bytes = b"PRIVATE-CAPABILITY-BODY-7f59a"

    async def run_failure():
        _, result = await _discover_and_call(
            remote_http_server,
            "upload_file",
            {
                "file_path": "/client/inputs/private.png",
                "client_os": "linux",
            },
        )
        match = re.search(
            r"https?://[^'\s]+/api/uploads/[\w-]+", result.content[0].text
        )
        assert match is not None
        async with httpx2.AsyncClient() as http_client:
            return await http_client.put(match.group(0), content=private_bytes)

    response = asyncio.run(run_failure())
    serialized = log_path.read_text(encoding="utf-8")
    entry = json.loads(serialized.strip())

    assert response.status_code == 502
    assert entry["kind"] == "error_envelope"
    assert entry["error_code"] == "upload_failed"
    assert private_bytes.decode() not in serialized
    scratch = pathlib.Path(procs[0].cmd[5])
    assert not scratch.exists()


def test_http_auto_mode_negotiates_the_modern_sessionless_protocol(
    remote_http_server, patched_run, monkeypatch
):
    patched_run(envelope(data={"running": True}))
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")

    discovered, result = asyncio.run(
        _discover_and_call(
            remote_http_server,
            "server_info",
            mode="auto",
        )
    )

    assert len({tool.name for tool in discovered}) == 40
    assert result.data["running"] is True


async def _raw_modern_call(url: str, *, stale_session: str | None = None):
    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": "tools/call",
        "mcp-name": "server_info",
    }
    if stale_session is not None:
        headers["mcp-session-id"] = stale_session
    async with httpx2.AsyncClient() as client:
        return await client.post(
            url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "server_info",
                    "arguments": {},
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                        "io.modelcontextprotocol/clientInfo": {
                            "name": "comfy-mcp-http-test",
                            "version": "1",
                        },
                        "io.modelcontextprotocol/clientCapabilities": {},
                    },
                },
            },
        )


def test_http_request_before_initialize_is_born_ready_and_does_not_poison_server(
    remote_http_server, patched_run, monkeypatch
):
    patched_run(envelope(data={"running": True}))
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")

    raw = asyncio.run(_raw_modern_call(remote_http_server))
    _, healthy = asyncio.run(
        _discover_and_call(remote_http_server, "server_info", mode="auto")
    )

    assert raw.status_code == 200, raw.text
    assert "mcp-session-id" not in raw.headers
    assert raw.json()["result"]["structuredContent"]["running"] is True
    assert healthy.data["running"] is True


def test_http_stale_session_header_is_inert_after_a_deployment_restart(
    remote_http_server, patched_run, monkeypatch
):
    patched_run(envelope(data={"running": True}))
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")

    response = asyncio.run(
        _raw_modern_call(
            remote_http_server,
            stale_session="session-from-a-prior-server-process",
        )
    )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["structuredContent"]["running"] is True


def test_http_concurrent_clients_use_isolated_request_contexts(
    remote_http_server, patched_run, monkeypatch
):
    patched_run(envelope(data={"running": True}))
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")

    async def run_clients():
        return await asyncio.gather(
            *(
                _discover_and_call(remote_http_server, "server_info", mode="auto")
                for _ in range(6)
            )
        )

    results = asyncio.run(run_clients())

    assert len(results) == 6
    assert all(result.data["running"] for _, result in results)


@pytest.mark.parametrize(
    ("shutdown_signal", "expected_returncode"),
    [(signal.SIGINT, 0), (signal.SIGTERM, -signal.SIGTERM)],
)
def test_installed_http_process_shuts_down_cleanly(
    shutdown_signal, expected_returncode
):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from comfy_mcp.server import main; "
                f"main(['serve', '--port', '{port}', '--log-level', 'INFO'])"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "COMFY_BIN": "/nonexistent/comfy-mcp-test-binary"},
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
        assert proc.poll() is None, "Remote process exited before accepting requests"

        async def discover():
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                return await client.list_tools()

        assert len(asyncio.run(discover())) == 40
        proc.send_signal(shutdown_signal)
        stdout, stderr = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    # Uvicorn 0.52 completes the ASGI lifespan, restores the process's prior
    # signal handler, then replays the signal. Python's SIGINT handler becomes
    # KeyboardInterrupt (normalized by ``serve``); SIGTERM keeps native Unix
    # signal-exit semantics. Both paths must finish application shutdown first.
    assert proc.returncode == expected_returncode
    assert "Application shutdown complete" in stderr
    assert "Finished server process" in stderr
    assert "Traceback" not in stderr
    assert stdout == ""
