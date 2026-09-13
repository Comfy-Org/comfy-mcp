"""Tests for ``update_comfyui``'s ``target="all"`` gate — ``comfy update all``.

comfy-cli maps that one target onto the node manager's ``update all``, which
``git pull``s and ``pip install``s EVERY installed third-party custom node pack
into ComfyUI's Python environment — so it runs code this server never saw, from
authors it cannot vouch for. comfy-cli does not gate that, which makes the MCP
prompt raised here the only thing between a tool call and third-party code
running on the user's machine. These tests lock in that posture:

1. Only ``"all"`` is gated. ``"comfy"`` (ComfyUI core) and ``"cli"`` (comfy-cli
   itself) move first-party code from known repositories and must reach argv with
   no prompt raised at all — the regression this file exists to guard hardest,
   since a gate that crept onto them would put a confirmation in front of the
   ordinary "you are out of date" flow.
2. The consent posture the destructive gates share: on a client that can be
   prompted the USER is asked every call, ``confirm_update_all=True`` is not a way
   around that prompt, and a refusal is enforced here with no child spawned. Only
   a client that cannot be prompted falls back to the explicit argument, whose
   ``False`` default means a bare ``target="all"`` runs nothing.
3. The ORDER: a bad target is rejected before anyone is asked, and an update
   already in flight refuses before anyone is asked — nobody approves a call that
   was never going to run.
4. The async plumbing the gate required: the update keeps its own worker thread
   rather than asyncio's shared pool, and the lock it holds belongs to the
   subprocess rather than to the request.

comfy-cli is mocked throughout — no real ComfyUI, checkout, or pack is touched.
"""

from __future__ import annotations

import asyncio
import threading
from unittest import mock

import pytest
from conftest import envelope
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_mcp.server import _internal as server


def _update(*args, **kwargs):
    """Drive the async ``update_comfyui`` tool from a sync test."""
    return asyncio.run(server.update_comfyui(*args, **kwargs))


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake FastMCP ``Context`` that answers the elicitation with ``action``.

    A local copy of the switch/spend tests' fake rather than a shared one,
    following the convention those files set: this gate's prompt must be
    assertable on its own, so a change to another tool's prompt cannot silently
    retune these tests. ``tests/test_sdk_conformance.py`` registers the copy and
    asserts it against the installed SDK's real signatures.
    """

    def __init__(self, action="accept", approve=True, supports_elicitation=True):
        self.session = _FakeSession(supports_elicitation)
        self._action = action
        self._approve = approve
        self.elicitations: list[str] = []

    async def elicit(self, message, response_type):
        self.elicitations.append(message)
        if self._action == "accept":
            return AcceptedElicitation(data=response_type(approve=self._approve))
        if self._action == "decline":
            return DeclinedElicitation()
        return CancelledElicitation()


# --- only `target="all"` is gated -------------------------------------------


@pytest.mark.parametrize("target", ["comfy", "cli"])
def test_first_party_targets_are_never_prompted(patched_plain_run, target):
    """Core and comfy-cli updates run unprompted, exactly as they did before.

    The ctx handed in here would DECLINE if it were asked, so a prompt creeping
    onto these targets fails twice over: the elicitation list is non-empty and the
    call raises instead of updating.
    """
    calls = patched_plain_run(0, stderr="Already up to date.")
    ctx = _FakeCtx(action="decline")

    result = _update(target, ctx=ctx)

    assert ctx.elicitations == []  # nobody was asked about first-party code
    assert calls[0]["cmd"][4:] == ["update", target]
    assert result["ok"] is True


@pytest.mark.parametrize("target", ["comfy", "cli"])
def test_first_party_targets_ignore_the_confirm_flag(patched_plain_run, target):
    """`confirm_update_all` is inert off the `"all"` path, in both positions."""
    calls = patched_plain_run(0, stderr="done")

    _update(target, confirm_update_all=False, ctx=_FakeCtx(action="decline"))
    _update(target, confirm_update_all=True, ctx=_FakeCtx(action="decline"))

    assert [c["cmd"][4:] for c in calls] == [["update", target]] * 2


# --- consent: a client that CAN be prompted ---------------------------------


def test_approved_pack_update_runs_the_command(patched_plain_run):
    """Accept -> `comfy update all` runs, and the prompt said what it would do."""
    calls = patched_plain_run(0, stderr="Updating custom nodes...")
    ctx = _FakeCtx(action="accept", approve=True)

    result = _update("all", ctx=ctx)

    assert len(ctx.elicitations) == 1
    prompt = ctx.elicitations[0]
    # What the user is actually approving: third-party code, pulled and installed
    # into ComfyUI's environment, slowly, with version fallout for other packs.
    assert "git pull" in prompt
    assert "pip install" in prompt
    assert "EVERY third-party custom node pack" in prompt
    assert "executes code those packs' own authors have published" in prompt
    assert "long time" in prompt
    assert "do not work with" in prompt
    assert "restarted" in prompt
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert cmd[4:] == ["update", "all"]
    assert result["ok"] is True


@pytest.mark.parametrize(
    ("action", "approve"),
    [
        ("decline", False),  # said no
        ("cancel", False),  # dismissed the prompt
        ("accept", False),  # accepted without actually answering yes
    ],
)
def test_a_refusal_spawns_no_child(patched_plain_run, action, approve):
    """A refusal is enforced HERE — comfy-cli never starts, no pack code runs."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="node pack update not confirmed"):
        _update("all", ctx=ctx)

    assert calls == []


def test_the_refusal_says_no_pack_code_ran(patched_plain_run):
    """The reassurance has to name the thing the gate exists for, not just "nothing"."""
    patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _update("all", ctx=_FakeCtx(action="decline"))

    assert "no pack code was run" in str(excinfo.value)


def test_confirm_update_all_is_not_a_way_around_the_prompt(patched_plain_run):
    """An agent setting `confirm_update_all=True` itself does not authorize this.

    Same hole the spend and switch gates close: a host's blanket "always allow
    this tool" toggle lets an agent set the argument for itself, which would
    otherwise be standing authority to run every pack author's latest code.
    """
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="node pack update not confirmed"):
        _update("all", confirm_update_all=True, ctx=ctx)

    assert len(ctx.elicitations) == 1  # asked anyway
    assert calls == []


def test_the_prompt_is_raised_even_when_confirm_update_all_is_true(patched_plain_run):
    """The approving case of the rule above: asked, then run."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="accept", approve=True)

    _update("all", confirm_update_all=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert len(calls) == 1


def test_an_unknown_capability_still_asks(patched_plain_run):
    """A probe that ERRORS is "could not tell", never "cannot elicit".

    Demoting a capable client onto the caller's own say-so is the outcome this
    gate exists to prevent, so the unknown case takes the prompt path.
    """

    class _BrokenSession:
        def check_client_capability(self, capability):
            raise RuntimeError("probe exploded")

    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="decline")
    ctx.session = _BrokenSession()

    with pytest.raises(server.ComfyCliError, match="node pack update not confirmed"):
        _update("all", confirm_update_all=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert calls == []


def test_unanswered_prompt_lapses_into_a_refusal(patched_plain_run, monkeypatch):
    """A client that advertises elicitation but never answers must not hang."""
    monkeypatch.setattr(server, "_ELICIT_TIMEOUT", 0.05)
    calls = patched_plain_run(0, stderr="done")

    class _SilentCtx(_FakeCtx):
        async def elicit(self, message, response_type):
            self.elicitations.append(message)
            await asyncio.sleep(3600)

    with pytest.raises(server.ComfyCliError, match="node pack update not confirmed"):
        _update("all", confirm_update_all=True, ctx=_SilentCtx())

    assert calls == []


def test_a_client_that_errors_on_the_prompt_names_the_manual_route(patched_plain_run):
    """No engine-side durable consent exists here, so the way out is the terminal."""
    calls = patched_plain_run(0, stderr="done")

    class _BoomCtx(_FakeCtx):
        async def elicit(self, message, response_type):
            raise RuntimeError("no prompt surface")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _update("all", confirm_update_all=True, ctx=_BoomCtx())

    message = str(excinfo.value)
    assert "could not confirm updating the installed custom node packs" in message
    assert "comfy update all" in message
    assert "Nothing was updated." in message
    assert calls == []


# --- consent: a client that CANNOT be prompted ------------------------------


@pytest.mark.parametrize(
    "make_ctx",
    [
        lambda: None,  # no context at all (a direct call, or a host injecting none)
        lambda: _FakeCtx(supports_elicitation=False),
    ],
    ids=["no-context", "no-elicitation"],
)
def test_bare_call_on_a_non_eliciting_client_updates_nothing(
    patched_plain_run, make_ctx
):
    """`confirm_update_all=False` (the default) runs nothing, and says why."""
    calls = patched_plain_run(0, stderr="done")

    with pytest.raises(
        server.ComfyCliError, match="confirm_update_all=True"
    ) as excinfo:
        _update("all", ctx=make_ctx())

    message = str(excinfo.value)
    # The stakes travel with the refusal, so an agent relaying only this error
    # cannot present it as a formality.
    assert "EVERY third-party custom node pack" in message
    # ...and it points at the two targets that need no confirmation, so a caller
    # that wanted "get me current" is not stuck.
    assert 'target="comfy"' in message
    assert calls == []


def test_confirm_update_all_is_the_fallback_when_the_client_cannot_elicit(
    patched_plain_run,
):
    """On a client with no elicitation, the explicit argument is the documented route."""
    calls = patched_plain_run(0, stderr="Updating custom nodes...")
    ctx = _FakeCtx(supports_elicitation=False)

    result = _update("all", confirm_update_all=True, ctx=ctx)

    assert ctx.elicitations == []
    assert calls[0]["cmd"][4:] == ["update", "all"]
    assert result["ok"] is True


# --- order: nobody is asked about a call that cannot run --------------------


@pytest.mark.parametrize("target", ["nodes", "", "  ", "all; rm -rf /", "--help"])
def test_an_invalid_target_is_rejected_before_the_prompt(patched_run, target):
    """A typo must not cost the user a confirmation prompt, or a subprocess."""
    calls = patched_run(envelope(data={}))
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError, match="invalid update target"):
        _update(target, ctx=ctx)

    assert calls == []
    assert ctx.elicitations == []


def test_an_in_flight_update_refuses_before_the_prompt(patched_plain_run):
    """The advisory lock peek: nobody approves an update that is then refused."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx()
    assert server._UPDATE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(server.ComfyCliError, match="already running"):
            _update("all", ctx=ctx)
    finally:
        server._UPDATE_LOCK.release()

    assert ctx.elicitations == []
    assert calls == []


def test_the_busy_refusal_names_every_lock_sharer(patched_plain_run):
    """`_UPDATE_LOCK` is shared three ways, so "an update" is not the whole story.

    A 25-minute `install_node` (or a `switch_comfyui_version`) is enough to refuse
    this call, and a caller told only about "an update" goes looking for an
    in-flight call that does not exist.
    """
    patched_plain_run(0, stderr="done")
    assert server._UPDATE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(server.ComfyCliError) as excinfo:
            _update("comfy", ctx=_FakeCtx())
    finally:
        server._UPDATE_LOCK.release()

    message = str(excinfo.value)
    assert "version switch" in message
    assert "node install" in message
    assert "Nothing was updated." in message


def test_a_declined_update_leaves_the_lock_free(patched_plain_run):
    """A refusal must not park the lock a legitimate update (or switch) needs.

    The prompt is raised BEFORE the lock is taken for the same reason: a human
    thinking for `_ELICIT_TIMEOUT` must not block `switch_comfyui_version`, which
    shares it.
    """
    patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError, match="node pack update not confirmed"):
        _update("all", ctx=_FakeCtx(action="decline"))

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_the_lock_is_not_held_across_the_prompt(patched_plain_run):
    """While the user is deciding, the lock must be free for whoever else wants it."""
    patched_plain_run(0, stderr="done")
    held_during_prompt: list[bool] = []

    class _PeekingCtx(_FakeCtx):
        async def elicit(self, message, response_type):
            held_during_prompt.append(server._UPDATE_LOCK.locked())
            return await super().elicit(message, response_type)

    _update("all", ctx=_PeekingCtx())

    assert held_during_prompt == [False]


# --- the async plumbing the gate required ----------------------------------


def test_the_update_stays_off_the_default_executor(patched_plain_run):
    """A 30-minute blocking call on asyncio's shared pool starves everything else.

    `_GENERATE_EXECUTOR` and `_SWITCH_EXECUTOR` exist for exactly this reason;
    this is the longest-running child in the server and gets the same treatment.
    """
    patched_plain_run(0, stderr="done")
    threads: list[str] = []

    def _capture(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return {"ok": True}

    with mock.patch.object(server, "_run_comfy", _capture):
        _update(
            "all", confirm_update_all=True, ctx=_FakeCtx(supports_elicitation=False)
        )

    assert threads and all(name.startswith("comfy-update") for name in threads)


def test_the_lock_is_held_until_the_subprocess_finishes(patched_plain_run):
    """Cancelling the REQUEST must not hand the lock to a second concurrent install.

    Cancellation raises `CancelledError` at the await but neither interrupts the
    worker thread nor kills the `comfy update` it spawned, so git and pip keep
    installing. If the lock were released in a `finally` here, a retry or a
    `switch_comfyui_version` would acquire it and run a second install against the
    same workspace and venv — the half-installed state the lock exists to prevent.
    """
    started = threading.Event()
    finish = threading.Event()

    def _slow(*_args, **_kwargs):
        started.set()
        finish.wait(5)
        return {"ok": True}

    patched_plain_run(0, stderr="done")

    async def _drive():
        task = asyncio.ensure_future(
            server.update_comfyui("all", confirm_update_all=True, ctx=None)
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The abandoned update is still running, so the lock must still be taken.
        assert not server._UPDATE_LOCK.acquire(blocking=False)
        finish.set()
        # ...and released once it actually ends, rather than leaked forever.
        for _ in range(500):
            if server._UPDATE_LOCK.acquire(blocking=False):
                server._UPDATE_LOCK.release()
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the lock was never released")

    with mock.patch.object(server, "_run_comfy", _slow):
        try:
            asyncio.run(_drive())
        finally:
            finish.set()
