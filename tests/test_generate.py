# Tests for the ``generate_image`` tool — the thin passthrough to
# ``comfy generate`` (text prompt -> image). The streaming fakes and the
# ``patched_stream`` fixture live in ``conftest.py`` and are shared with the
# other streaming tools' tests.
from __future__ import annotations

import asyncio

from conftest import _OK_STREAM, _RecordingCtx

from comfy_local_mcp import server


def test_generate_image_streams_and_maps_command(patched_stream):
    """wait=True drives `comfy --json-stream … generate --prompt … --wait`."""
    procs = patched_stream(_OK_STREAM)
    ctx = _RecordingCtx()

    result = asyncio.run(server.generate_image("a red fox in snow", ctx=ctx))

    assert result == {"outputs": ["/x.png"]}  # same envelope shape as run_workflow
    assert len(ctx.calls) >= 1  # progress notifications forwarded when wait=True

    cmd = procs[0].cmd
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json-stream", "--where", "local"]  # global flags first
    # No checkpoint given -> no --checkpoint pair in the command.
    assert cmd[4:] == ["generate", "--prompt", "a red fox in snow", "--wait"]


def test_generate_image_forwards_checkpoint_when_streaming(patched_stream):
    """A checkpoint is forwarded as `--checkpoint <name>` before `--wait`."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.generate_image("a cat", checkpoint="sd_xl.safetensors"))

    assert procs[0].cmd[4:] == [
        "generate",
        "--prompt",
        "a cat",
        "--checkpoint",
        "sd_xl.safetensors",
        "--wait",
    ]


def test_generate_image_wait_false_uses_plain_json_no_stream(monkeypatch):
    """wait=False keeps the plain --json _run_comfy path (no streaming, no --wait)."""
    seen: dict = {}

    def fake_run_comfy(*args, timeout=None):
        seen["args"] = args
        seen["timeout"] = timeout
        return {"prompt_id": "p1"}

    def boom(*a, **k):  # streaming must not be taken for wait=False
        raise AssertionError("wait=False must not stream")

    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)
    monkeypatch.setattr(server, "_run_comfy_streaming", boom)

    result = asyncio.run(server.generate_image("a red fox in snow", wait=False))

    assert result == {"prompt_id": "p1"}
    assert seen["args"] == ("generate", "--prompt", "a red fox in snow")  # no --wait
    assert seen["timeout"] == 60.0


def test_generate_image_wait_false_forwards_checkpoint(monkeypatch):
    """wait=False still forwards a checkpoint to `comfy generate --checkpoint`."""
    seen: dict = {}

    def fake_run_comfy(*args, timeout=None):
        seen["args"] = args
        return {"prompt_id": "p2"}

    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)

    server_result = asyncio.run(
        server.generate_image("a dog", checkpoint="dreamshaper.safetensors", wait=False)
    )

    assert server_result == {"prompt_id": "p2"}
    assert seen["args"] == (
        "generate",
        "--prompt",
        "a dog",
        "--checkpoint",
        "dreamshaper.safetensors",
    )
