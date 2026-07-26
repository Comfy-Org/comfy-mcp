# Tests for the ``generate_image`` tool — the text-prompt on-ramp, which
# delegates to ``comfy run-template <template> --param=KEY=VALUE`` (the same verb
# ``run_template`` wraps) rather than the partner-only, credit-spending
# ``comfy generate``. The streaming fakes and the ``patched_stream`` fixture live
# in ``conftest.py`` and are shared with the other streaming tools' tests.
#
# What these lock in:
#
# 1. The argv shape: global flags first, then ``run-template <template>``, the
#    prompt as a single ``--param=<slot>=<json>`` token, and the engine's
#    per-event ``--timeout`` — with ``--async`` only on ``wait=False``.
# 2. That the prompt rides INSIDE that token, so a prompt starting with ``-``
#    can never be re-read as an option (the wf-006 class of bug).
# 3. That nothing spend-related is forwarded — the default template is a free,
#    fully local OSS graph.
# 4. That ``COMFY_T2I_TEMPLATE`` (and its slot-key companions) actually retarget
#    the call.
from __future__ import annotations

import asyncio

import pytest
from conftest import _OK_STREAM, _RecordingCtx

from comfy_local_mcp import server


def test_generate_image_streams_and_maps_command(patched_stream):
    """wait=True drives `comfy --json-stream … run-template default --param=…`."""
    procs = patched_stream(_OK_STREAM)
    ctx = _RecordingCtx()

    result = asyncio.run(server.generate_image("a red fox in snow", ctx=ctx))

    assert result == {"outputs": ["/x.png"]}  # same envelope shape as run_workflow
    assert len(ctx.calls) >= 1  # progress notifications forwarded when wait=True

    cmd = procs[0].cmd
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json-stream", "--where", "local"]  # global flags first
    # No checkpoint given -> no ckpt_name param. The prompt rides inside a single
    # `--param=KEY=VALUE` token (JSON-encoded value), and the engine gets its
    # per-event deadline — capped at comfy-cli's own 120s default, never raised.
    assert cmd[4:] == [
        "run-template",
        "default",
        '--param=6.text="a red fox in snow"',
        "--timeout=120",
    ]
    # Nothing spend-related: the default template is a free local OSS graph.
    assert "--allow-spend" not in cmd
    # And no trace of the partner-only verb this tool used to (mis)invoke.
    assert "generate" not in cmd


def test_generate_image_forwards_checkpoint_when_streaming(patched_stream):
    """A checkpoint fills the template's `ckpt_name` slot, after the prompt."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.generate_image("a cat", checkpoint="sd_xl.safetensors"))

    assert procs[0].cmd[4:] == [
        "run-template",
        "default",
        '--param=6.text="a cat"',
        '--param=ckpt_name="sd_xl.safetensors"',
        "--timeout=120",
    ]


def test_generate_image_leading_dash_prompt_is_not_parsed_as_flag(patched_stream):
    """A prompt starting with `-` is carried inside `--param=<slot>=<value>`."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.generate_image("--not-a-flag, just text"))

    assert procs[0].cmd[4:] == [
        "run-template",
        "default",
        '--param=6.text="--not-a-flag, just text"',
        "--timeout=120",
    ]
    # The whole prompt is one argv token, so comfy-cli's parser never sees a
    # bare `--not-a-flag`.
    assert not any(tok.startswith("--not-a-flag") for tok in procs[0].cmd)


def test_generate_image_short_timeout_lowers_the_engine_deadline(patched_stream):
    """A budget under comfy-cli's 120s default tightens `--timeout` to match."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.generate_image("a cat", timeout_seconds=45.0))

    assert procs[0].cmd[-1] == "--timeout=45"


def test_generate_image_wait_false_uses_plain_json_no_stream(monkeypatch):
    """wait=False submits `--async` on the plain --json path (no streaming)."""
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
    assert seen["args"] == (
        "run-template",
        "default",
        '--param=6.text="a red fox in snow"',
        "--timeout=60",
        "--async",
    )
    # Submit budget is the fixed short one, plus the grace that lets comfy-cli
    # report its own error instead of dying to the parent's signal.
    assert seen["timeout"] == pytest.approx(
        server._RUN_TEMPLATE_ASYNC_TIMEOUT + server._RUN_TEMPLATE_TIMEOUT_GRACE
    )


def test_generate_image_wait_false_forwards_checkpoint(monkeypatch):
    """wait=False still fills the template's checkpoint slot."""
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
        "run-template",
        "default",
        '--param=6.text="a dog"',
        '--param=ckpt_name="dreamshaper.safetensors"',
        "--timeout=60",
        "--async",
    )


def test_generate_image_env_overrides_template_and_slots(patched_stream, monkeypatch):
    """COMFY_T2I_TEMPLATE (+ slot keys) retarget the run at another graph."""
    procs = patched_stream(_OK_STREAM)
    monkeypatch.setenv("COMFY_T2I_TEMPLATE", "image_flux2")
    monkeypatch.setenv("COMFY_T2I_PROMPT_SLOT", "44.text")
    monkeypatch.setenv("COMFY_T2I_CHECKPOINT_SLOT", "unet_name")

    asyncio.run(server.generate_image("a cat", checkpoint="flux2.safetensors"))

    assert procs[0].cmd[4:] == [
        "run-template",
        "image_flux2",
        '--param=44.text="a cat"',
        '--param=unet_name="flux2.safetensors"',
        "--timeout=120",
    ]


def test_generate_image_env_template_override_alone_keeps_default_slots(
    patched_stream, monkeypatch
):
    """Overriding only the template leaves the built-in slot keys in place."""
    procs = patched_stream(_OK_STREAM)
    monkeypatch.setenv("COMFY_T2I_TEMPLATE", "some_other_template")

    asyncio.run(server.generate_image("a cat"))

    assert procs[0].cmd[4:6] == ["run-template", "some_other_template"]
    assert procs[0].cmd[6] == '--param=6.text="a cat"'


def test_generate_image_rejects_option_like_template(monkeypatch):
    """A malformed COMFY_T2I_TEMPLATE is refused by name, before any child spawns."""
    monkeypatch.setenv("COMFY_T2I_TEMPLATE", "--json")

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)
    monkeypatch.setattr(server, "_run_comfy_streaming", boom)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(server.generate_image("a cat"))

    assert "COMFY_T2I_TEMPLATE" in str(exc.value)


def test_generate_image_empty_env_falls_back_to_the_builtin_template(
    patched_stream, monkeypatch
):
    """An empty COMFY_T2I_TEMPLATE is treated as unset, not as an invalid name."""
    procs = patched_stream(_OK_STREAM)
    monkeypatch.setenv("COMFY_T2I_TEMPLATE", "")

    asyncio.run(server.generate_image("a cat"))

    assert procs[0].cmd[4:6] == ["run-template", server._T2I_TEMPLATE]


def test_generate_image_slot_error_names_the_env_knobs(monkeypatch):
    """A `workflow_slot_invalid` from the engine gains a which-knob-to-set hint."""

    def fake_run_comfy(*args, timeout=None):
        raise server.ComfyCliError(
            "--param key '6.text' matches no slot in this template",
            code="workflow_slot_invalid",
        )

    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(server.generate_image("a cat", wait=False))

    message = str(exc.value)
    assert "matches no slot" in message  # the engine's own diagnosis survives
    assert "COMFY_T2I_PROMPT_SLOT" in message
    assert exc.value.code == "workflow_slot_invalid"
    # No checkpoint was passed, so that slot was never sent — naming it here
    # would send the reader after a knob that cannot be the cause.
    assert "checkpoint slot" not in message
    assert "COMFY_T2I_CHECKPOINT_SLOT" not in message


def test_generate_image_slot_error_names_checkpoint_only_when_filled(monkeypatch):
    """With a checkpoint actually filled, the hint names that knob too."""

    def fake_run_comfy(*args, timeout=None):
        raise server.ComfyCliError(
            "--param key 'ckpt_name' matches no slot in this template",
            code="workflow_slot_invalid",
        )

    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(
            server.generate_image("a cat", checkpoint="sd_xl.safetensors", wait=False)
        )

    message = str(exc.value)
    assert "checkpoint slot 'ckpt_name'" in message
    assert "COMFY_T2I_CHECKPOINT_SLOT" in message


def test_generate_image_rejects_colliding_slot_overrides(monkeypatch):
    """One key for both slots would silently drop the prompt — refuse it instead."""
    monkeypatch.setenv("COMFY_T2I_PROMPT_SLOT", "6.text")
    monkeypatch.setenv("COMFY_T2I_CHECKPOINT_SLOT", "6.text")

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)
    monkeypatch.setattr(server, "_run_comfy_streaming", boom)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(server.generate_image("a cat", checkpoint="sd_xl.safetensors"))

    message = str(exc.value)
    assert "COMFY_T2I_PROMPT_SLOT" in message
    assert "COMFY_T2I_CHECKPOINT_SLOT" in message


def test_generate_image_colliding_slots_are_fine_without_a_checkpoint(
    patched_stream, monkeypatch
):
    """Nothing overwrites the prompt when no checkpoint is passed, so the run proceeds."""
    procs = patched_stream(_OK_STREAM)
    monkeypatch.setenv("COMFY_T2I_PROMPT_SLOT", "6.text")
    monkeypatch.setenv("COMFY_T2I_CHECKPOINT_SLOT", "6.text")

    asyncio.run(server.generate_image("a cat"))

    assert procs[0].cmd[4:] == [
        "run-template",
        "default",
        '--param=6.text="a cat"',
        "--timeout=120",
    ]


def test_generate_image_streaming_parent_deadline_has_the_engine_grace(monkeypatch):
    """The parent outlives the deadline it hands the engine, so comfy-cli reports first."""
    seen: dict = {}

    async def fake_stream(*args, ctx=None, timeout=None, **kwargs):
        seen["args"] = args
        seen["timeout"] = timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    # A budget at/under comfy-cli's 120s cap is the dangerous case: the child's
    # `--timeout` equals the budget, so without grace both deadlines fire at the
    # same instant and the parent's SIGKILL wins over the engine's own report.
    asyncio.run(server.generate_image("a cat", timeout_seconds=45.0))

    assert seen["args"][-1] == "--timeout=45"  # engine's deadline: the budget
    assert seen["timeout"] == pytest.approx(45.0 + server._RUN_TEMPLATE_TIMEOUT_GRACE)


def test_generate_image_other_errors_pass_through_unchanged(monkeypatch):
    """A non-slot failure is re-raised as-is, not dressed up with the slot hint."""

    def fake_run_comfy(*args, timeout=None):
        raise server.ComfyCliError("ComfyUI is not running", code="server_not_running")

    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(server.generate_image("a cat", wait=False))

    assert str(exc.value) == "ComfyUI is not running"
    assert "COMFY_T2I_PROMPT_SLOT" not in str(exc.value)
    # Re-raised bare, so it carries no self-referential __cause__ chain.
    assert exc.value.__cause__ is not exc.value
