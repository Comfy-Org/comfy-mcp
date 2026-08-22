"""Installed stdio process -> FastMCP -> client layer -> fake comfy executable."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

_FAKE_COMFY = r"""#!{python}
import json
import sys


def envelope(data):
    print(json.dumps({{
        "type": "envelope",
        "schema": "envelope/1",
        "ok": True,
        "data": data,
    }}))


args = sys.argv[1:]
if args == ["--version"]:
    print("comfy-cli 1.14.0")
    raise SystemExit(0)

try:
    local = args.index("local")
except ValueError:
    raise SystemExit(2)
command = args[local + 1:]

if command == ["env"]:
    envelope({{
        "server": {{"running": True, "url": "http://127.0.0.1:8188"}},
        "hardware": {{"ram_total": 32000000000, "gpus": []}},
    }})
elif command == ["outdated"]:
    envelope({{"core": {{"outdated": False}}, "packs": []}})
elif command[0] == "upload" and command[-1] == "--no-overwrite":
    with open(command[1], "rb") as handle:
        if handle.read() != b"stdio-input":
            raise SystemExit(4)
    envelope({{
        "uploads": [{{
            "local_path": command[1],
            "cloud_name": "stdio-input.bin",
        }}],
    }})
elif command == ["run", "--workflow", "/srv/workflows/smoke.json"]:
    envelope({{"prompt_id": "prompt-stdio"}})
elif command == ["jobs", "status", "prompt-stdio"]:
    envelope({{"prompt_id": "prompt-stdio", "status": "completed"}})
elif command == ["download", "prompt-stdio", "-o", "/tmp/results"]:
    envelope({{"files": [{{"path": "/tmp/results/result.png"}}]}})
else:
    print("unexpected command: " + repr(command), file=sys.stderr)
    raise SystemExit(3)
"""


async def _run_stdio_flow(fake_comfy: str, input_path: str):
    env = {
        **os.environ,
        "COMFY_BIN": fake_comfy,
        "COMFY_CLI_MIN_VERSION": "1.14.0",
    }
    transport = StdioTransport(
        command=sys.executable,
        args=["-c", "from comfy_mcp.server import main; main()"],
        env=env,
    )
    async with Client(transport, mode="legacy") as client:
        tools = await client.list_tools()
        info = await client.call_tool("server_info", {})
        uploaded = await client.call_tool(
            "upload_file",
            {
                "file_path": input_path,
                "client_os": "darwin",
            },
        )
        submitted = await client.call_tool(
            "run_workflow",
            {
                "workflow_path": "/srv/workflows/smoke.json",
                "wait": False,
            },
        )
        status = await client.call_tool(
            "job", {"action": "status", "prompt_id": "prompt-stdio"}
        )
        outputs = await client.call_tool(
            "fetch_outputs",
            {"prompt_id": "prompt-stdio", "out_dir": "/tmp/results"},
        )
    return tools, info, uploaded, submitted, status, outputs


@pytest.mark.skipif(os.name != "posix", reason="the fake comfy binary uses a shebang")
def test_real_stdio_process_completes_submit_poll_fetch(tmp_path):
    fake = tmp_path / "fake-comfy"
    fake.write_text(_FAKE_COMFY.format(python=sys.executable), encoding="utf-8")
    fake.chmod(0o755)
    input_path = tmp_path / "stdio-input.png"
    input_path.write_bytes(b"stdio-input")

    tools, info, uploaded, submitted, status, outputs = asyncio.run(
        asyncio.wait_for(
            _run_stdio_flow(str(fake), str(input_path)),
            timeout=120,
        )
    )

    assert len(tools) == 40
    assert info.data["server"]["running"] is True
    assert uploaded.data["uploads"] == [
        {"local_path": str(input_path), "cloud_name": "stdio-input.bin"}
    ]
    assert submitted.data["prompt_id"] == "prompt-stdio"
    assert status.data["status"] == "completed"
    assert outputs.data["files"][0]["path"].endswith("result.png")
