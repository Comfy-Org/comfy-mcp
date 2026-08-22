"""FastMCP 4 application flows over the shared comfy-cli client boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from conftest import EXPECTED_TOOL_NAMES
from fastmcp import Client, Context, FastMCP
from fastmcp.client.elicitation import ElicitResult
from mcp_types.version import (
    HANDSHAKE_PROTOCOL_VERSIONS,
    LATEST_HANDSHAKE_VERSION,
    LATEST_MODERN_VERSION,
    MODERN_PROTOCOL_VERSIONS,
)

from comfy_mcp.client import context as client_context
from comfy_mcp.server import _internal as server


async def _main_application_flow(engine):
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
    fake_comfy_client,
):
    engine = fake_comfy_client
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.14.0")

    tools, info, submitted, status, outputs = asyncio.run(
        _main_application_flow(engine)
    )

    assert {tool.name for tool in tools} == EXPECTED_TOOL_NAMES
    assert info.data["server"]["running"] is True
    assert submitted.data["prompt_id"] == "prompt-1"
    assert status.data["status"] == "completed"
    assert outputs.data["files"][0]["path"] == "/tmp/result.png"
    assert ("run", "--workflow", "/srv/workflows/smoke.json") in engine.calls
    assert ("jobs", "status", "prompt-1") in engine.calls


async def _accept_approval(message, response_type, params, ctx):
    return ElicitResult(action="accept", content=response_type(approve=True))


async def _shared_paid_workflow(mode: str, engine):
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


def test_shared_application_paid_workflow_uses_modern_guard_round_trip(
    monkeypatch,
    fake_comfy_client,
):
    engine = fake_comfy_client
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: True)

    protocol, result = asyncio.run(_shared_paid_workflow("auto", engine))

    assert protocol == LATEST_MODERN_VERSION
    assert result.data["prompt_id"] == "prompt-1"
    assert (
        "run",
        "--workflow",
        "/srv/workflows/paid.json",
        "--allow-spend",
    ) in engine.calls


def test_shared_application_paid_workflow_keeps_legacy_elicitation_compatible(
    monkeypatch,
    fake_comfy_client,
):
    engine = fake_comfy_client
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: True)

    protocol, result = asyncio.run(_shared_paid_workflow("legacy", engine))

    assert protocol == LATEST_HANDSHAKE_VERSION
    assert result.data["prompt_id"] == "prompt-1"


def test_protocol_classifier_accepts_only_sdk_registered_generations():
    def context(version):
        return SimpleNamespace(
            request_context=SimpleNamespace(protocol_version=version)
        )

    for version in MODERN_PROTOCOL_VERSIONS:
        assert server._protocol_info(context(version)) == (version, "modern")
        assert server._is_modern_protocol(context(version)) is True
    for version in HANDSHAKE_PROTOCOL_VERSIONS:
        assert server._protocol_info(context(version)) == (version, "handshake")
        assert server._is_modern_protocol(context(version)) is False

    assert server._protocol_info(context("draft")) == ("draft", "unknown")
    assert server._is_modern_protocol(context("draft")) is False
    assert server._protocol_info(context(20260728)) == (None, "unknown")
    assert server._protocol_info(None) == (None, "absent")


def test_unknown_protocol_cannot_fall_back_to_legacy_confirmation():
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(protocol_version="draft"),
    )

    assert server._client_elicitation_support(ctx) is None
    with pytest.raises(
        server.ComfyCliError,
        match="unsupported MCP protocol version 'draft'",
    ):
        asyncio.run(
            server._elicit_approval(
                ctx,
                "Approve spending?",
                server.SpendApproval,
                server._SPEND_APPROVAL_WORDING,
            )
        )


def test_spend_gates_have_distinct_round_keys_but_no_env_consent_tokens():
    generate_key = server._approval_key(server._SPEND_APPROVAL_WORDING)
    optin_key = server._approval_key(server._OPTIN_SPEND_APPROVAL_WORDING)

    assert generate_key == "comfy_mcp_spend_generate"
    assert optin_key == "comfy_mcp_spend_optin"
    assert generate_key != optin_key
    assert server._SPEND_APPROVAL_WORDING.consent_token == ""
    assert server._OPTIN_SPEND_APPROVAL_WORDING.consent_token == ""


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
