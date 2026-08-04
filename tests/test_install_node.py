"""Tests for ``install_node`` — ``comfy node install <name...> --exit-on-fail``.

The engine owns the install itself (registry lookup, clone, dependency resolve).
These lock in what the WRAPPER owns, which is almost entirely about not running
third-party code by accident:

1. The input guard: a registry pack id is the whole accepted set. A git URL, a
   filesystem path, a leading dash, ``all``, or an oversized list is refused
   before any subprocess is spawned. The URL case is the load-bearing one — the
   consent prompt promises the user a NAMED PACK FROM THE REGISTRY, so a value
   that could be anything else would make the prompt lie about what it collected
   approval for.
2. The CONSENT posture, mirroring ``switch_comfyui_version``: on a client that can
   be prompted the USER is asked every call, ``confirm_install=True`` is not a way
   around that prompt, and a refusal is enforced here with no child spawned. Only
   a client that cannot be prompted falls back to the explicit argument, whose
   ``False`` default means a bare call installs nothing.
3. ``--exit-on-fail`` is always forwarded. Without it comfy-cli swallows a failed
   install and exits 0, so the flag is what makes a failure reach the caller at
   all rather than being reported as success.
4. The shared ``_UPDATE_LOCK``: an install pip-installs into the same environment
   an update or a version switch does, so it is refused rather than queued.
5. The result contract, including ``restart_required: True`` — this tool never
   restarts anything, which is what lets a user say "install it, I'll restart the
   server myself".
6. The async plumbing that posture required, mirroring ``test_update_consent``'s:
   the install keeps its OWN worker thread rather than asyncio's shared pool, it
   forwards the 30-minute timeout, and the lock it holds belongs to the SUBPROCESS
   rather than to the request. Without those three, replacing the done-callback
   release with a ``try/finally`` around the await passes every other test in this
   file while handing the lock to a retry that then runs a second concurrent pip.

comfy-cli is mocked throughout: no real ComfyUI and nothing is ever installed.
"""

from __future__ import annotations

import asyncio
import threading
from unittest import mock

import pytest
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_mcp import server


def _install(*args, **kwargs):
    """Drive the async ``install_node`` tool from a sync test."""
    return asyncio.run(server.install_node(*args, **kwargs))


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake MCPServer ``Context`` that answers the elicitation with ``action``.

    A local copy of the switch/spend tests' fake rather than a shared one,
    following the convention those files set: this gate's prompt must be
    assertable on its own, so a change to another gate's prompt cannot silently
    retune these tests.
    """

    def __init__(self, action="accept", approve=True, supports_elicitation=True):
        self.session = _FakeSession(supports_elicitation)
        self._action = action
        self._approve = approve
        self.elicitations: list[str] = []

    async def elicit(self, message, schema):
        self.elicitations.append(message)
        if self._action == "accept":
            return AcceptedElicitation(data=schema(approve=self._approve))
        if self._action == "decline":
            return DeclinedElicitation()
        return CancelledElicitation()


# --- the input guard --------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        [],  # nothing to install
        "comfyui-impact-pack",  # a bare string is not the list this takes
        None,
        [""],
        ["   "],
        [None],
        ["comfyui-impact-pack", ""],  # one bad entry poisons the call
    ],
)
def test_rejects_a_malformed_name_list_before_spawning(patched_plain_run, names):
    """A list this tool cannot vouch for never reaches argv — or the prompt."""
    calls = patched_plain_run(0, stderr="installed")
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError, match="invalid names"):
        _install(names, ctx=ctx)

    assert calls == []
    assert ctx.elicitations == []


@pytest.mark.parametrize(
    "name",
    [
        "-rf",  # argument injection: reads as an option, not a pack id
        "--exit-on-fail",
        "-",
        # A git URL or a path. Refused DELIBERATELY, and this is the guard's whole
        # reason for being stricter than the engine: the prompt says "from the
        # registry", so nothing that isn't a registry id may ride through it.
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "git@github.com:ltdrdata/ComfyUI-Impact-Pack.git",
        "/tmp/evil-pack",
        "../../etc/passwd",
        "./local-pack",
        "pack;rm -rf /",
        "pack && curl evil.example",
        "pack|sh",
        "pack$(whoami)",
        "pack`whoami`",
        "pack with spaces",
        "pack\nsecond",
        "pack\0",
        ".leading-dot",  # must start alphanumeric
        "_leading-underscore",
    ],
)
def test_rejects_a_name_that_is_not_a_registry_id(patched_plain_run, name):
    """Only a registry slug is installable; everything else stops here."""
    calls = patched_plain_run(0, stderr="installed")
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError):
        _install([name], ctx=ctx)

    assert calls == []
    assert ctx.elicitations == []


def test_an_oversized_name_is_reported_by_length_not_echoed(patched_plain_run):
    """A megabyte-long "pack name" must not come back through the response."""
    patched_plain_run(0, stderr="installed")
    huge = "a" * 5000

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install([huge], ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "5000 characters" in message
    assert huge not in message


def test_an_oversized_invalid_name_is_still_reported_by_length(patched_plain_run):
    """The ordering the guard documents, pinned: length is checked before shape.

    The test above cannot see that ordering — ``"a" * 5000`` MATCHES the registry
    pattern, so it reaches the length error whichever check runs first. A value
    that is both oversized and malformed is the case that separates them: with the
    checks reversed it takes the id-shape branch, which echoes the value, and a
    5000-character echo is exactly what the length branch exists to avoid.
    """
    patched_plain_run(0, stderr="installed")
    junk = "/" * 5000

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install([junk], ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "5000 characters" in message
    assert "not a registry pack id" not in message  # took the length branch
    assert junk not in message
    assert "/" * 100 not in message  # nor any long slice of it


def test_a_batch_that_joins_too_long_is_refused(patched_plain_run):
    """The count cap and the per-id cap leave their PRODUCT unbounded.

    32 ids of 128 characters each is a legal list by both other bounds and a
    4,698-character confirmation prompt — the "scrolls past rather than reads"
    failure the count cap exists to prevent, reached by padding instead of by
    counting. Refused rather than shown truncated: the prompt naming every pack IS
    the consent, so a batch padded with look-alike slugs must not be able to hide
    the real pack past a truncation marker.
    """
    calls = patched_plain_run(0, stderr="installed")
    names = [f"pack-{n}-" + "x" * 120 for n in range(server._MAX_NODE_PACK_NAMES)]
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install(names, ctx=ctx)

    message = str(excinfo.value)
    assert "smaller batches" in message
    assert str(server._MAX_NODE_PACK_NAMES_CHARS) in message
    assert calls == []
    assert ctx.elicitations == []  # nobody was asked to approve an unreadable list


def test_a_full_size_batch_of_real_looking_ids_still_installs(patched_plain_run):
    """The joined cap must not refuse a batch anyone would actually send.

    Registry slugs run ~15-30 characters, so a full 32-pack call of realistic ids
    lands under the bound; a cap that refused this would be the wrapper inventing
    a limitation rather than keeping its own prompt readable.
    """
    calls = patched_plain_run(0, stderr="installed")
    names = [f"comfyui-node-pack-{n:02d}" for n in range(server._MAX_NODE_PACK_NAMES)]

    result = _install(names, ctx=(ctx := _FakeCtx()))

    assert len(calls) == 1
    assert result["installed"] == names
    assert names[-1] in ctx.elicitations[0]  # every pack was named in the prompt


def test_too_many_packs_is_refused_so_the_prompt_stays_readable(patched_plain_run):
    """An approval the user cannot actually read is not an approval."""
    calls = patched_plain_run(0, stderr="installed")
    names = [f"pack-{n}" for n in range(server._MAX_NODE_PACK_NAMES + 1)]

    with pytest.raises(server.ComfyCliError, match="exceeds the"):
        _install(names, ctx=_FakeCtx())

    assert calls == []


def test_all_is_refused_and_points_at_update_comfyui(patched_plain_run):
    """`all` is a different intent, and comfy-cli's own refusal is opaque."""
    calls = patched_plain_run(0, stderr="installed")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install(["all"], ctx=_FakeCtx())

    assert 'update_comfyui(target="all")' in str(excinfo.value)
    assert calls == []


# --- what reaches argv ------------------------------------------------------


def test_approved_install_forwards_exit_on_fail(patched_plain_run):
    """`--exit-on-fail` is not optional: without it a failure exits 0."""
    calls = patched_plain_run(0, stderr="Installed comfyui-impact-pack")

    result = _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert cmd[4:] == ["node", "install", "comfyui-impact-pack", "--exit-on-fail"]
    assert result["installed"] == ["comfyui-impact-pack"]
    assert result["restart_required"] is True


def test_multiple_packs_are_forwarded_in_order_then_the_flag(patched_plain_run):
    """Several packs in one call, with the flag after the positionals."""
    calls = patched_plain_run(0, stderr="done")

    _install(["comfyui-impact-pack", "comfyui_controlnet_aux"], ctx=_FakeCtx())

    assert calls[0]["cmd"][4:] == [
        "node",
        "install",
        "comfyui-impact-pack",
        "comfyui_controlnet_aux",
        "--exit-on-fail",
    ]


def test_names_are_stripped_before_reaching_argv(patched_plain_run):
    """Surrounding whitespace is normalized away, like `update_comfyui`'s target."""
    calls = patched_plain_run(0, stderr="done")

    result = _install(["  comfyui-impact-pack  "], ctx=_FakeCtx())

    assert calls[0]["cmd"][6] == "comfyui-impact-pack"
    assert result["installed"] == ["comfyui-impact-pack"]


def test_a_failed_install_still_raises(patched_plain_run):
    """The flag's whole point: a non-zero exit must reach the caller."""
    patched_plain_run(1, stderr="ERROR: pack not found in registry")

    with pytest.raises(server.ComfyCliError):
        _install(["no-such-pack"], ctx=_FakeCtx())


# --- consent: a client that CAN be prompted ---------------------------------


def test_approved_install_runs_and_the_prompt_named_the_stakes(patched_plain_run):
    """Accept -> the install runs, and the user was told what it does."""
    calls = patched_plain_run(0, stderr="done")

    result = _install(
        ["comfyui-impact-pack"], ctx=(ctx := _FakeCtx(action="accept", approve=True))
    )

    assert len(ctx.elicitations) == 1
    prompt = ctx.elicitations[0]
    assert "comfyui-impact-pack" in prompt
    assert "DOWNLOADS" in prompt  # third-party code arrives
    assert "RUNS" in prompt  # and is executed
    assert "restarted" in prompt  # and won't be visible until then
    assert len(calls) == 1
    assert result["restart_required"] is True


@pytest.mark.parametrize(
    ("action", "approve"),
    [
        ("decline", False),  # said no
        ("cancel", False),  # dismissed the prompt
        ("accept", False),  # accepted without actually answering yes
    ],
)
def test_a_refusal_spawns_no_child(patched_plain_run, action, approve):
    """A refusal is enforced HERE — comfy-cli is never started, nothing installed."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], ctx=ctx)

    assert calls == []


def test_confirm_install_is_not_a_way_around_the_prompt(patched_plain_run):
    """An agent setting `confirm_install=True` itself does not authorize the run.

    The hole this closes is the spend gates': a host's blanket "always allow this
    tool" toggle lets an agent set the argument for itself, which would otherwise
    be standing authority to execute third-party code on the user's machine — and
    the pack names are frequently the model's own guess.
    """
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], confirm_install=True, ctx=ctx)

    assert len(ctx.elicitations) == 1  # asked anyway
    assert calls == []


def test_the_prompt_is_raised_even_when_confirm_install_is_true(patched_plain_run):
    """The approving case of the rule above: asked, then run."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="accept", approve=True)

    _install(["comfyui-impact-pack"], confirm_install=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert len(calls) == 1


def test_an_unknown_capability_still_asks(patched_plain_run):
    """A probe that ERRORS is "could not tell", never "cannot elicit"."""

    class _BrokenSession:
        def check_client_capability(self, capability):
            raise RuntimeError("probe exploded")

    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="decline")
    ctx.session = _BrokenSession()

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], confirm_install=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert calls == []


def test_an_unanswered_prompt_is_a_refusal(patched_plain_run, monkeypatch):
    """A prompt left hanging past the timeout installs nothing."""
    calls = patched_plain_run(0, stderr="done")
    monkeypatch.setattr(server, "_ELICIT_TIMEOUT", 0.05)

    class _HangingCtx(_FakeCtx):
        async def elicit(self, message, schema):
            self.elicitations.append(message)
            await asyncio.sleep(10)

    ctx = _HangingCtx()

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], ctx=ctx)

    assert calls == []


def test_a_client_that_errors_on_the_prompt_is_a_refusal(patched_plain_run):
    """A broken elicit() fails closed, and names the terminal escape hatch."""
    calls = patched_plain_run(0, stderr="done")

    class _ExplodingCtx(_FakeCtx):
        async def elicit(self, message, schema):
            raise RuntimeError("client went away")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install(["comfyui-impact-pack"], ctx=_ExplodingCtx())

    message = str(excinfo.value)
    assert "Nothing was installed." in message
    assert "comfy node install" in message  # the way out
    assert calls == []


# --- consent: a client that CANNOT be prompted ------------------------------


@pytest.mark.parametrize(
    "make_ctx",
    [
        lambda: None,  # no context at all (a direct call, or a host injecting none)
        lambda: _FakeCtx(supports_elicitation=False),
    ],
)
def test_an_unpromptable_client_installs_nothing_by_default(
    patched_plain_run, make_ctx
):
    """The `False` default is what makes a bare call from such a client safe."""
    calls = patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], ctx=make_ctx())

    assert calls == []


def test_an_unpromptable_client_may_pass_the_explicit_flag(patched_plain_run):
    """With no prompt available, the argument is the documented consent route."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(supports_elicitation=False)

    result = _install(["comfyui-impact-pack"], confirm_install=True, ctx=ctx)

    assert ctx.elicitations == []  # there was nothing to ask
    assert calls[0]["cmd"][4:] == [
        "node",
        "install",
        "comfyui-impact-pack",
        "--exit-on-fail",
    ]
    assert result["restart_required"] is True


def test_the_unpromptable_refusal_states_the_stakes(patched_plain_run):
    """An agent reading the error has to learn what it would be authorizing."""
    patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install(["comfyui-impact-pack"], ctx=None)

    message = str(excinfo.value)
    assert "confirm_install=True" in message
    assert "third-party code" in message
    assert "never just to clear this error" in message


# --- the shared update lock -------------------------------------------------


def test_an_in_flight_update_refuses_before_the_prompt(patched_plain_run):
    """Refused, not queued — and without asking the user to approve a dead end.

    The peek runs before consent so a caller does not answer a prompt for an
    install that was never going to start.
    """
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx()

    assert server._UPDATE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(server.ComfyCliError, match="already running"):
            _install(["comfyui-impact-pack"], ctx=ctx)
    finally:
        server._UPDATE_LOCK.release()

    assert ctx.elicitations == []
    assert calls == []


def test_the_lock_is_released_after_a_successful_install(patched_plain_run):
    """A leaked lock would wedge every later update, switch and install."""
    patched_plain_run(0, stderr="done")

    _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_the_lock_is_released_after_a_failed_install(patched_plain_run):
    """Same, on the failure path."""
    patched_plain_run(1, stderr="boom")

    with pytest.raises(server.ComfyCliError):
        _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_a_refused_install_never_took_the_lock(patched_plain_run):
    """A declined call must not block an update that is legitimately in flight."""
    patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], ctx=_FakeCtx(action="decline"))

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_the_busy_refusal_names_every_lock_sharer(patched_plain_run):
    """The holder may be an update OR a version switch; say both.

    A caller told only about "an update" goes hunting for an in-flight call that
    may not exist. `update_comfyui` and `switch_comfyui_version` name a node
    install in their own refusals for the same reason.
    """
    patched_plain_run(0, stderr="done")

    assert server._UPDATE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(server.ComfyCliError) as excinfo:
            _install(["comfyui-impact-pack"], ctx=_FakeCtx())
    finally:
        server._UPDATE_LOCK.release()

    message = str(excinfo.value)
    assert "version switch" in message
    assert "Nothing was installed." in message


# --- discoverability --------------------------------------------------------


def test_the_handshake_instructions_teach_the_install_flow():
    """An agent reads `INSTRUCTIONS` once, at handshake — it must find the door.

    Before this, the missing-node guidance ended at "tell the USER what is
    missing", which is precisely the dead end this tool removes: an agent that
    learned the wall at handshake never discovers the tool that resolves it. The
    restart is part of the same clause because installing without restarting looks
    like a no-op from `search_nodes`.
    """
    flat = " ".join(server.INSTRUCTIONS.split())

    assert "install_node" in flat
    assert "`install_node` -> `restart_comfyui`" in flat


def test_the_module_docstring_lists_the_tool():
    """The inventory at the top of `server.py` is the other place a reader looks."""
    assert "install_node" in server.__doc__


# --- the async plumbing the gate required -----------------------------------


def test_the_install_stays_off_the_default_executor(patched_plain_run):
    """A 30-minute blocking call on asyncio's shared pool starves everything else.

    `_UPDATE_EXECUTOR` and `_SWITCH_EXECUTOR` exist for exactly this reason, and an
    install runs as long as either: on the default pool it would park the only
    worker every other `to_thread` caller in the process shares.
    """
    patched_plain_run(0, stderr="done")
    threads: list[str] = []

    def _capture(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return {"ok": True}

    with mock.patch.object(server, "_run_comfy", _capture):
        _install(
            ["comfyui-impact-pack"],
            confirm_install=True,
            ctx=_FakeCtx(supports_elicitation=False),
        )

    assert threads and all(name.startswith("comfy-node-install") for name in threads)


def test_the_timeout_is_generous(patched_plain_run):
    """Resolving and installing a pack's dependencies is minutes, not seconds."""
    calls = patched_plain_run(0, stderr="done")

    _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert calls[0]["timeout"] == server._INSTALL_TIMEOUT
    assert calls[0]["timeout"] >= 1800.0


def test_the_lock_is_held_until_the_subprocess_finishes(patched_plain_run):
    """Cancelling the REQUEST must not hand the lock to a second concurrent pip.

    Cancellation raises `CancelledError` at the await but neither interrupts the
    worker thread nor kills the `comfy node install` it spawned, so pip keeps
    resolving. If the lock were released in a `finally` here, a retry, an
    `update_comfyui` or a `switch_comfyui_version` would acquire it and run a
    second install against the same venv — the corrupted environment the lock
    exists to prevent. The done-callback ties the release to the JOB's lifetime
    instead, so this is what fails if anyone "simplifies" it back to a `finally`.
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
            server.install_node(["comfyui-impact-pack"], confirm_install=True, ctx=None)
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The abandoned install is still running, so the lock must still be taken.
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
