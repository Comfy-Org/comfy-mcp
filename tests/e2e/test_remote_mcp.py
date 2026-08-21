"""Live Streamable HTTP MCP smoke through a client-reachable endpoint.

Unlike ``test_smoke.py``, this module never imports the application or shells
out to comfy-cli locally. It behaves as a remote MCP client: negotiate, discover
the shared tool surface, upload client-local bytes through the returned
single-use capability, submit a checkpoint-free workflow that already exists
on the server, poll it, and download the signed output URL.

It is opt-in because it mutates the configured ComfyUI by uploading one tiny
PNG and running one ``EmptyImage`` -> ``SaveImage`` job. Configure both values::

    COMFY_MCP_TEST_URL=http://127.0.0.1:9000/mcp \
    COMFY_MCP_TEST_WORKFLOW=/absolute/server/path/workflow_smoke.json \
      pytest -q tests/e2e/test_remote_mcp.py -m e2e

The endpoint may be an SSH local-forward. ``COMFY_MCP_TEST_WORKFLOW`` is a path
on the MCP server host, while the generated upload path and ``out_dir`` are
client-local paths. No model or paid node is used.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import sys
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

import httpx2
import pytest
from fastmcp import Client

_ENDPOINT_ENV = "COMFY_MCP_TEST_URL"
_WORKFLOW_ENV = "COMFY_MCP_TEST_WORKFLOW"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)
_UPLOAD_URL_RE = re.compile(r"https?://[^'\"\s]+/api/uploads/[A-Za-z0-9_-]+")
_TERMINAL = {
    "completed",
    "success",
    "succeeded",
    "failed",
    "error",
    "cancelled",
    "canceled",
}

pytestmark = [pytest.mark.e2e]


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"set {name} to run the live remote MCP smoke")
    return value


def _client_os() -> str:
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def _prompt_id(data: Any) -> str:
    if isinstance(data, dict):
        value = data.get("prompt_id")
        if isinstance(value, str) and value:
            return value
    pytest.fail(f"workflow submission returned no prompt_id: {data!r}")


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_remote_mcp_negotiates_and_discovers_shared_application(mode):
    """Both HTTP protocol modes expose the same ComfyCloud-compatible schema."""

    endpoint = _required_env(_ENDPOINT_ENV)

    async def exercise() -> None:
        async with Client(endpoint, mode=mode) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert client.protocol_version is not None
            assert len(tools) == 39
            assert tools["upload_file"].input_schema["required"] == [
                "file_path",
                "client_os",
            ]
            assert tools["fetch_outputs"].input_schema["properties"]["client_os"][
                "enum"
            ] == ["darwin", "linux", "windows"]

            info = await client.call_tool("server_info", {})
            assert info.data["server"]["running"] is True
            assert info.data["compatibility"]["envelope_schema"] == "envelope/1"

    asyncio.run(asyncio.wait_for(exercise(), timeout=90))


def test_remote_mcp_upload_submit_poll_and_signed_fetch(tmp_path):
    """Client bytes make the complete remote MCP/ComfyUI round trip."""

    endpoint = _required_env(_ENDPOINT_ENV)
    workflow_path = _required_env(_WORKFLOW_ENV)
    client_os = _client_os()
    image_path = tmp_path / f"comfy_mcp_remote_smoke_{uuid.uuid4().hex}.png"
    image_path.write_bytes(_ONE_PIXEL_PNG)

    async def exercise() -> None:
        async with Client(endpoint, mode="auto") as client:
            upload = await client.call_tool(
                "upload_file",
                {"file_path": str(image_path.resolve()), "client_os": client_os},
            )
            upload_text = upload.content[0].text
            match = _UPLOAD_URL_RE.search(upload_text)
            assert match is not None, upload_text
            upload_url = match.group(0)

            async with httpx2.AsyncClient(timeout=60) as http:
                uploaded = await http.put(
                    upload_url,
                    content=image_path.read_bytes(),
                    headers={"Content-Type": "image/png"},
                )
                reused = await http.put(upload_url, content=_ONE_PIXEL_PNG)
            assert uploaded.status_code == 200, uploaded.text
            assert uploaded.json() == {
                "name": image_path.name,
                "subfolder": "",
                "type": "input",
            }
            assert reused.status_code == 404

            submitted = await client.call_tool(
                "run_workflow",
                {"workflow_path": workflow_path, "wait": False},
            )
            prompt_id = _prompt_id(submitted.data)

            deadline = time.monotonic() + 120
            final: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                status = await client.call_tool(
                    "job", {"action": "status", "prompt_id": prompt_id}
                )
                assert isinstance(status.data, dict), status
                final = status.data
                if final.get("status") in _TERMINAL:
                    break
                await asyncio.sleep(1)
            assert final is not None
            assert final.get("status") == "completed", final

            outputs = await client.call_tool(
                "fetch_outputs",
                {
                    "prompt_id": prompt_id,
                    "out_dir": str(tmp_path / "outputs"),
                    "client_os": client_os,
                },
            )
            assert outputs.data["download_url_ttl_seconds"] == 300
            files = outputs.data["files"]
            assert files
            download_url = files[0]["download_url"]
            endpoint_origin = urlsplit(endpoint)
            output_origin = urlsplit(download_url)
            assert (output_origin.hostname, output_origin.port) == (
                endpoint_origin.hostname,
                endpoint_origin.port,
            )

            async with httpx2.AsyncClient(timeout=60) as http:
                downloaded = await http.get(download_url)
                tampered = await http.get(
                    download_url.replace("signature=", "signature=0", 1)
                )
            assert downloaded.status_code == 200
            assert downloaded.content.startswith(_PNG_MAGIC)
            assert tampered.status_code == 404

    asyncio.run(asyncio.wait_for(exercise(), timeout=240))
