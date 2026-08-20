"""Streamable HTTP integration over the one shared FastMCP application."""

from __future__ import annotations

import asyncio
import json
import os
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

from comfy_mcp import failure_log
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


def test_http_exposes_the_same_39_tools_and_complete_business_flow(
    remote_http_server, patched_run, monkeypatch
):
    """One HTTP client completes the same submit/poll/fetch flow as stdio."""

    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")

    async def run_flow():
        async with Client(remote_http_server, mode="legacy") as client:
            discovered = await client.list_tools()

            info_calls = patched_run(
                envelope(data={"running": True, "url": "http://127.0.0.1:8188"})
            )
            info = await client.call_tool("server_info", {})

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

            output_calls = patched_run(
                envelope(data={"files": [{"path": "/tmp/results/result.png"}]})
            )
            outputs = await client.call_tool(
                "fetch_outputs",
                {"prompt_id": "prompt-http", "out_dir": "/tmp/results"},
            )
        return (
            discovered,
            info,
            submitted,
            status,
            outputs,
            info_calls,
            submit_calls,
            status_calls,
            output_calls,
        )

    (
        discovered,
        info,
        submitted,
        status,
        outputs,
        info_calls,
        submit_calls,
        status_calls,
        output_calls,
    ) = asyncio.run(run_flow())

    names = {tool.name for tool in discovered}
    assert len(names) == 39
    assert {"server_info", "run_workflow", "job", "fetch_outputs"} <= names
    assert info.data["running"] is True
    assert submitted.data["prompt_id"] == "prompt-http"
    assert status.data["status"] == "completed"
    assert outputs.data["files"][0]["path"].endswith("result.png")
    assert info_calls[0]["cmd"][4:] == ["env"]
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
        "/tmp/results",
    ]


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

    assert len({tool.name for tool in discovered}) == 39
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

        assert len(asyncio.run(discover())) == 39
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
