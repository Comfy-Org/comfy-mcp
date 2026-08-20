"""The outbound comfy-cli client is the shared engine boundary."""

from __future__ import annotations

import asyncio

from comfy_mcp.client import context as client_context
from comfy_mcp.client.subprocess_client import SubprocessComfyCliClient
from comfy_mcp.server import _internal as server


def test_subprocess_client_preserves_each_runner_contract():
    calls = []

    def raw(*args, **kwargs):
        calls.append(("raw", args, kwargs))
        return None, "", tuple(args), 0, ""

    def run(*args, **kwargs):
        calls.append(("run", args, kwargs))
        return {"sync": True}

    async def async_run(*args, **kwargs):
        calls.append(("async", args, kwargs))
        return {"async": True}

    async def stream(*args, **kwargs):
        calls.append(("stream", args, kwargs))
        return {"stream": True}

    client = SubprocessComfyCliClient(raw, run, async_run, stream)

    assert client.run_raw("env", timeout=1)[3] == 0
    assert client.run("stop", timeout=2, plain_ok=True) == {"sync": True}
    assert asyncio.run(client.run_async("upload", timeout=3, stdout_cap=4096)) == {
        "async": True
    }
    assert asyncio.run(
        client.run_streaming("run", timeout=4, raise_on_timeout=False)
    ) == {"stream": True}
    assert calls == [
        ("raw", ("env",), {"timeout": 1}),
        ("run", ("stop",), {"timeout": 2, "plain_ok": True}),
        ("async", ("upload",), {"timeout": 3, "stdout_cap": 4096}),
        ("stream", ("run",), {"timeout": 4, "raise_on_timeout": False}),
    ]


def test_request_binding_routes_legacy_server_entry_points_through_client():
    class FakeClient:
        def run_raw(self, *args, timeout=None):
            return None, "", args, 0, ""

        def run(self, *args, timeout=None, plain_ok=False):
            return {"args": args, "timeout": timeout, "plain_ok": plain_ok}

        async def run_async(self, *args, timeout=None, plain_ok=False, stdout_cap=None):
            return {"args": args, "stdout_cap": stdout_cap}

        async def run_streaming(
            self, *args, ctx=None, timeout=None, raise_on_timeout=True
        ):
            return {"args": args, "raise_on_timeout": raise_on_timeout}

    original = client_context.get_client()
    fake = FakeClient()
    with client_context.bind_client(fake):
        assert server._run_comfy("jobs", "status", "p1", timeout=5) == {
            "args": ("jobs", "status", "p1"),
            "timeout": 5,
            "plain_ok": False,
        }
        assert (
            asyncio.run(
                server._run_comfy_streaming(
                    "jobs", "watch", "p1", raise_on_timeout=False
                )
            )["raise_on_timeout"]
            is False
        )

    assert client_context.get_client() is original
