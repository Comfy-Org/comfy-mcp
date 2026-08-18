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
# 5. That a ``wait=True`` call which outlives its bound returns the ``prompt_id``
#    of the still-running job instead of erroring with no handle, and that the
#    default bound is short enough to expire here rather than at the client's
#    own (invisible) transport cap.
from __future__ import annotations

import asyncio
import json

import pytest
from conftest import _OK_STREAM, _RecordingCtx, stream_reader

from comfy_mcp import server


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
    # per-event deadline — capped at comfy-cli's own 120s default, never raised,
    # and LOWERED here to the 90s default budget (`_T2I_DEFAULT_TIMEOUT`).
    assert cmd[4:] == [
        "run-template",
        "image_z_image_turbo",
        '--param=57.text="a red fox in snow"',
        "--timeout=90",
    ]
    # Nothing spend-related: the default template is a free local OSS graph.
    assert "--allow-spend" not in cmd
    # And no trace of the partner-only verb this tool used to (mis)invoke.
    assert "generate" not in cmd


def test_generate_image_forwards_checkpoint_when_streaming(patched_stream, monkeypatch):
    """A checkpoint fills the template's `ckpt_name` slot, after the prompt.

    Retargeted at a CheckpointLoaderSimple graph via env: the built-in on-ramp
    template no longer has a checkpoint slot (the gallery moved to split
    UNET/CLIP/VAE loaders), so forwarding is only meaningful for a graph that
    still has one. The refusal on the default is asserted separately by
    ``test_generate_image_refuses_checkpoint_without_a_slot``.
    """
    monkeypatch.setenv("COMFY_T2I_CHECKPOINT_SLOT", "ckpt_name")
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.generate_image("a cat", checkpoint="sd_xl.safetensors"))

    assert procs[0].cmd[4:] == [
        "run-template",
        "image_z_image_turbo",
        '--param=57.text="a cat"',
        '--param=ckpt_name="sd_xl.safetensors"',
        "--timeout=90",
    ]


def test_generate_image_leading_dash_prompt_is_not_parsed_as_flag(patched_stream):
    """A prompt starting with `-` is carried inside `--param=<slot>=<value>`."""
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.generate_image("--not-a-flag, just text"))

    assert procs[0].cmd[4:] == [
        "run-template",
        "image_z_image_turbo",
        '--param=57.text="--not-a-flag, just text"',
        "--timeout=90",
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
        "image_z_image_turbo",
        '--param=57.text="a red fox in snow"',
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

    monkeypatch.setenv("COMFY_T2I_CHECKPOINT_SLOT", "ckpt_name")
    server_result = asyncio.run(
        server.generate_image("a dog", checkpoint="dreamshaper.safetensors", wait=False)
    )

    assert server_result == {"prompt_id": "p2"}
    assert seen["args"] == (
        "run-template",
        "image_z_image_turbo",
        '--param=57.text="a dog"',
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
        "--timeout=90",
    ]


def test_generate_image_env_template_override_alone_keeps_default_slots(
    patched_stream, monkeypatch
):
    """Overriding only the template leaves the built-in slot keys in place."""
    procs = patched_stream(_OK_STREAM)
    monkeypatch.setenv("COMFY_T2I_TEMPLATE", "some_other_template")

    asyncio.run(server.generate_image("a cat"))

    assert procs[0].cmd[4:6] == ["run-template", "some_other_template"]
    assert procs[0].cmd[6] == '--param=57.text="a cat"'


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
            "--param key '57.text' matches no slot in this template",
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
    # A checkpoint slot has to EXIST for a checkpoint to be forwarded at all —
    # the built-in on-ramp graph no longer has one.
    monkeypatch.setenv("COMFY_T2I_CHECKPOINT_SLOT", "ckpt_name")

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
        "image_z_image_turbo",
        '--param=6.text="a cat"',
        "--timeout=90",
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


def test_generate_image_refuses_checkpoint_without_a_slot(no_spawn):
    """`checkpoint=` on a split-loader graph is refused BY NAME, before submitting.

    The built-in on-ramp template loads weights through UNETLoader/CLIPLoader/
    VAELoader, so there is no `ckpt_name` to address. Forwarding anyway would
    reach comfy-cli and come back as `workflow_slot_invalid` — an error about
    slot syntax for what is really an unsupported argument on this graph.
    """

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(server.generate_image("a cat", checkpoint="sd_xl.safetensors"))

    message = str(exc.value)
    assert "not supported by template" in message
    assert "COMFY_T2I_CHECKPOINT_SLOT" in message


def test_generate_image_default_template_is_not_the_retired_one():
    """Regression guard for the `default`-template rot.

    `generate_image` shelled out to `comfy run-template default` long after
    `default` left the gallery, so the documented on-ramp failed on every call —
    and the suite stayed green because its tests asserted the same dead constant.
    Naming the retired value here is what makes a silent re-introduction fail.
    """
    template, prompt_slot, _checkpoint_slot = server._t2i_config()
    assert template != "default"
    assert template and prompt_slot


def test_generate_image_missing_template_error_is_actionable(monkeypatch):
    """A template that is not in the gallery yields names, not a dead-end hint.

    comfy-cli's own text says only "try `comfy templates ls --name <substring>`",
    which does not say WHICH name — and the caller never chose the template that
    failed. This is the durable half of the fix: the next gallery rotation
    surfaces as one actionable error.
    """

    def fake_run_comfy(*args, timeout=None):
        if args[:2] == ("templates", "ls"):
            return {"rows": [{"name": "image_flux2_text_to_image", "tags": []}]}
        raise server.ComfyCliError("no template named 'gone' in the gallery")

    monkeypatch.setenv("COMFY_T2I_TEMPLATE", "gone")
    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)
    monkeypatch.setattr(server, "_run_comfy_streaming", fake_run_comfy)

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(server.generate_image("a cat", wait=False))

    message = str(exc.value)
    assert "not in the gallery" in message
    assert "image_flux2_text_to_image" in message
    assert "COMFY_T2I_TEMPLATE" in message


# --- The wait that expires: a handle, never an orphan -----------------------
#
# `generate_image` is the flagship one-call on-ramp, so its `wait=True` is the
# wait most likely to be outlived by a real generation on real local hardware.
# Expiring must therefore hand back the `prompt_id`: killing our comfy-cli child
# stops the WATCHER, never the job ComfyUI already accepted, so a timeout that
# returns no handle leaves a generation running that nobody can poll, collect or
# cancel.


class _BlockingProc:
    """A child fake that emits ``first_lines`` and then never yields an envelope.

    Local rather than in ``conftest`` for the reason AGENTS.md allows: this is the
    one case where the call genuinely differs — the shared ``patched_stream`` fake
    drains its canned stream to EOF instantly and reports itself already exited,
    so it can never hold the read past a deadline, which is the whole state these
    tests are about. Mirrors the same fake in ``test_run_template.py``.

    ``returncode`` starts None so the timeout handler's kill fires; no ``pid``, so
    that kill takes ``server._kill_proc_tree_async``'s ``proc.kill()`` fallback
    instead of signalling a made-up process group.
    """

    def __init__(self, cmd, first_lines):
        self.cmd = cmd
        self._lines = [line.encode("utf-8") for line in first_lines]
        self.stdout = self  # the reader protocol lives on the proc itself
        self.stderr = stream_reader("")
        self.returncode = None
        self.killed = False

    async def readuntil(self, separator=b"\n"):
        if self._lines:
            return self._lines.pop(0)
        # Outlives the test's tiny deadline; no envelope ever comes.
        await asyncio.sleep(1.0)
        raise asyncio.IncompleteReadError(b"", None)

    async def wait(self):
        self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True


def _blocking_stream(monkeypatch, first_lines):
    """Spawn fake whose child emits ``first_lines`` and then blocks forever."""
    procs: list[_BlockingProc] = []

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProc(cmd, first_lines)
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    # Zeroed purely so the deadline is reachable inside a test; the real grace is
    # 30s of deliberate slack past the caller's budget.
    monkeypatch.setattr(server, "_RUN_TEMPLATE_TIMEOUT_GRACE", 0.0)
    return procs


def test_generate_image_expired_wait_returns_the_prompt_id(monkeypatch):
    """An expired wait returns the handle payload instead of raising."""
    queued = json.dumps(
        {
            "schema": "event/1",
            "type": "queued",
            "prompt_id": "11111111-2222-3333-4444-555555555555",
            "nodes": [{"node_id": "1"}, {"node_id": "2"}],
        }
    )
    procs = _blocking_stream(monkeypatch, [queued + "\n"])

    result = asyncio.run(server.generate_image("a cat", timeout_seconds=0.25))

    # Not an error: the job is alive and this is its handle.
    assert result["timed_out"] is True
    assert result["prompt_id"] == "11111111-2222-3333-4444-555555555555"
    assert result["status"]["total"] == 2.0  # the queued manifest was seen
    # The payload names the tools that poll, collect and cancel it, so the caller
    # never has to go hunting through `job(action="queue")` for the id.
    assert 'job(action="status")' in result["note"]
    assert "fetch_outputs" in result["note"]
    assert 'job(action="cancel")' in result["note"]
    assert "NOT" in result["note"]  # ... and that it was not cancelled
    assert procs[0].killed  # our child is still cleaned up


def test_generate_image_expired_wait_before_submit_still_raises(monkeypatch):
    """No `prompt_id` reported means nothing was submitted — that stays an error.

    The handle payload is not a blanket "timeouts are fine" switch: a deadline
    reached before the engine ever queued anything (ComfyUI down, a stalled
    template fetch) has no job to hand back, and reporting it as a successful
    `timed_out` would invent a job the caller could never poll.
    """
    # A well-formed run event that carries no `prompt_id` — the pre-submit state.
    preflight = json.dumps({"schema": "event/1", "type": "converted", "node_count": 2})
    procs = _blocking_stream(monkeypatch, [preflight + "\n"])

    with pytest.raises(server.ComfyCliError) as exc:
        asyncio.run(server.generate_image("a cat", timeout_seconds=0.25))

    message = str(exc.value)
    assert "comfy-cli timed out after" in message
    assert "already submitted as prompt_id" not in message
    assert procs[0].killed


def test_generate_image_default_wait_fits_a_conservative_client_cap(monkeypatch):
    """The default call cannot outlive a conservative MCP transport cap.

    The client's cap is invisible to this server and fires first when it is
    shorter, returning nothing at all — so the DEFAULT has to expire on this side,
    where the handle can still be returned.
    """
    seen: dict = {}

    async def fake_stream(*args, ctx=None, timeout=None, **kwargs):
        seen["timeout"] = timeout
        seen["kwargs"] = kwargs
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    asyncio.run(server.generate_image("a cat"))

    # Whole-call wall clock = the caller's budget + the engine grace.
    assert seen["timeout"] == pytest.approx(
        server._T2I_DEFAULT_TIMEOUT + server._RUN_TEMPLATE_TIMEOUT_GRACE
    )
    # 300s is the cap observed in the field; 120s is what `run_workflow`'s own
    # default is chosen against. Stay under the tighter one.
    assert seen["timeout"] <= 120.0
    # And the expiry hands back the handle rather than raising.
    assert seen["kwargs"]["timeout_returns_handle"] is True
