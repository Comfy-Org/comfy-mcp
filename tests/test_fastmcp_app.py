"""FastMCP 4 application flows over the shared comfy-cli client boundary."""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client, Context, FastMCP
from fastmcp.client.elicitation import ElicitResult

from comfy_mcp.client import context as client_context
from comfy_mcp.server import _internal as server


class _FlowClient:
    """A typed engine fake that stands in for comfy-cli, not for MCP."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run_raw(self, *args: str, timeout=None):
        self.calls.append(args)
        data = {
            "server": {"running": True, "url": "http://127.0.0.1:8188"},
            "hardware": {"ram_total": 32_000_000_000, "gpus": []},
        }
        body = {
            "type": "envelope",
            "schema": "envelope/1",
            "ok": True,
            "data": data,
        }
        return body, json.dumps(body), args, 0, ""

    def run(self, *args: str, timeout=None, plain_ok=False):
        self.calls.append(args)
        if args == ("outdated",):
            return {"core": {"outdated": False}, "packs": []}
        if args[:1] == ("run",):
            return {"prompt_id": "prompt-1"}
        if args[:2] == ("jobs", "status"):
            return {"prompt_id": args[2], "status": "completed"}
        if args[:1] == ("download",):
            return {"files": [{"path": "/tmp/result.png"}]}
        raise AssertionError(f"unexpected comfy-cli call: {args!r}")

    async def run_async(
        self, *args: str, timeout=None, plain_ok=False, stdout_cap=None
    ):
        self.calls.append(args)
        return {"ok": True}

    async def run_streaming(
        self,
        *args: str,
        ctx=None,
        timeout=None,
        raise_on_timeout=True,
        timeout_returns_handle=False,
    ):
        self.calls.append(args)
        return {"prompt_id": "prompt-1", "status": "completed"}


async def _main_application_flow(engine: _FlowClient):
    with client_context.bind_client(engine):
        async with Client(server.mcp, mode="legacy") as client:
            tools = await client.list_tools()
            info = await client.call_tool("server_info", {})
            submitted = await client.call_tool(
                "run_workflow",
                {
                    "workflow_path": "/srv/workflows/smoke.json",
                    "wait": False,
                },
            )
            status = await client.call_tool(
                "job", {"action": "status", "prompt_id": "prompt-1"}
            )
            outputs = await client.call_tool(
                "fetch_outputs",
                {"prompt_id": "prompt-1", "out_dir": "/tmp/results"},
            )
    return tools, info, submitted, status, outputs


def test_stdio_application_surface_runs_the_complete_submit_poll_fetch_flow(
    monkeypatch,
):
    engine = _FlowClient()
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")

    tools, info, submitted, status, outputs = asyncio.run(
        _main_application_flow(engine)
    )

    assert len(tools) == 40
    assert info.data["server"]["running"] is True
    assert submitted.data["prompt_id"] == "prompt-1"
    assert status.data["status"] == "completed"
    assert outputs.data["files"][0]["path"] == "/tmp/result.png"
    assert ("run", "--workflow", "/srv/workflows/smoke.json") in engine.calls
    assert ("jobs", "status", "prompt-1") in engine.calls


async def _accept_approval(message, response_type, params, ctx):
    return ElicitResult(action="accept", content=response_type(approve=True))


async def _shared_paid_workflow(mode: str, engine: _FlowClient):
    with client_context.bind_client(engine):
        async with Client(
            server.mcp,
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
            protocol = client.protocol_version
        return protocol, result


def test_shared_application_paid_workflow_uses_modern_guard_round_trip(monkeypatch):
    engine = _FlowClient()
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: True)

    protocol, result = asyncio.run(_shared_paid_workflow("auto", engine))

    assert str(protocol) == "2026-07-28"
    assert result.data["prompt_id"] == "prompt-1"
    assert (
        "run",
        "--workflow",
        "/srv/workflows/paid.json",
        "--allow-spend",
    ) in engine.calls


def test_shared_application_paid_workflow_keeps_legacy_elicitation_compatible(
    monkeypatch,
):
    engine = _FlowClient()
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: True)

    protocol, result = asyncio.run(_shared_paid_workflow("legacy", engine))

    assert str(protocol) != "2026-07-28"
    assert result.data["prompt_id"] == "prompt-1"


def test_modern_guard_carries_only_prior_gate_approval_between_rounds():
    app = FastMCP("approval-rounds")
    body_runs = []
    prompts = []

    @app.tool
    @server._interactive_tool
    async def two_gates(ctx: Context):
        body_runs.append("run")
        network = await server._elicit_approval(
            ctx,
            "Approve network exposure?",
            server.NetworkExposureApproval,
            server._NETWORK_APPROVAL_WORDING,
        )
        if not network:
            raise server.ComfyCliError("network exposure declined")
        kill = await server._elicit_approval(
            ctx,
            "Approve stopping the untracked server?",
            server.KillUntrackedApproval,
            server._KILL_UNTRACKED_APPROVAL_WORDING,
        )
        if not kill:
            raise server.ComfyCliError("stopping declined")
        return "approved"

    async def approve(message, response_type, params, ctx):
        prompts.append(message)
        return ElicitResult(action="accept", content=response_type(approve=True))

    async def run():
        async with Client(app, mode="auto", elicitation_handler=approve) as client:
            return await client.call_tool("two_gates", {})

    result = asyncio.run(run())

    # FastMCP 4 represents an untyped string tool result as text content; it
    # does not invent a structured ``data`` payload for it.
    assert result.data is None
    assert result.content[0].text == "approved"
    assert prompts == [
        "Approve network exposure?",
        "Approve stopping the untracked server?",
    ]
    assert len(body_runs) == 3
