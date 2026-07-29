"""Tests for ``switch_comfyui_version`` — ``comfy update comfy --version <X>``.

The engine owns the switch itself (stash, checkout, dependency reinstall). These
lock in what the WRAPPER owns, which is entirely about not doing it by accident:

1. The input guard: ``nightly`` / ``latest`` / a semver tag is the whole accepted
   set, and a leading dash or a malformed value is refused before any subprocess
   is spawned — a bad version must not cost the user a stashed working tree.
2. The CONSENT posture, mirroring the spend gates: on a client that can be
   prompted the USER is asked every call, ``confirm_switch=True`` is not a way
   around that prompt, and a refusal is enforced here with no child spawned. Only
   a client that cannot be prompted falls back to the explicit argument, whose
   ``False`` default means a bare call changes nothing.
3. The refusal while a local ComfyUI is running, naming ``stop_comfyui``.
4. The old-comfy-cli degrade: a ``--version`` this comfy-cli does not accept
   surfaces as "upgrade comfy-cli", not as a raw Click usage dump.
5. The result contract, including ``restart_required: True`` — this tool never
   restarts anything.

comfy-cli is mocked throughout: no real ComfyUI and no real checkout is touched.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import envelope
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_local_mcp import server


def _switch(*args, **kwargs):
    """Drive the async ``switch_comfyui_version`` tool from a sync test."""
    return asyncio.run(server.switch_comfyui_version(*args, **kwargs))


#: The unpatched running probe, kept so the one test that exercises it can reach
#: past the autouse fixture that stubs it for everyone else.
_REAL_RUNNING_PROBE = server._local_comfyui_running


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake MCPServer ``Context`` that answers the elicitation with ``action``.

    A local copy of the spend tests' fake rather than a shared one, following the
    convention those files set: this gate's prompt must be assertable on its own,
    so a change to a spend prompt cannot silently retune these tests.
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


@pytest.fixture(autouse=True)
def _server_is_stopped(monkeypatch):
    """Default every test to "no local ComfyUI is running".

    The running check is the tool's FIRST gate and shells out through
    ``server_info``, which would consume the stubbed spawn and shift every
    exact-argv assertion. The refusal tests override it explicitly, and
    :func:`test_running_probe_reads_comfy_env` exercises the real helper.
    """
    monkeypatch.setattr(server, "_local_comfyui_running", lambda: False)


# --- the input guard --------------------------------------------------------


@pytest.mark.parametrize(
    "version",
    [
        "-rf",  # argument injection: reads as an option, not a version
        "-",  # not a valid value at any call site
        "--version",
        "",
        "   ",
        "0.24",  # not semver — a plausible typo, and the costly one
        "main",
        "HEAD",
        "v",
        "0.24.0; rm -rf /",
        "0.24.0 --force",
        "latest\nnightly",
        "0.24.0\0",
        "1" * 65,  # over `_MAX_VERSION_LEN`
    ],
)
def test_rejects_a_malformed_version_before_spawning(patched_run, version, monkeypatch):
    """A version this tool cannot vouch for never reaches argv — or the prompt."""
    calls = patched_run(envelope(data={}))
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError, match="invalid version"):
        _switch(version, ctx=ctx)

    assert calls == []  # nothing was forwarded to comfy-cli
    assert ctx.elicitations == []  # and the user was not asked about a typo


def test_leading_dash_error_names_the_injection_guard(patched_run):
    """The dash case keeps `_reject_option_like`'s own wording, not the format one."""
    patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError) as excinfo:
        _switch("--force", ctx=_FakeCtx())

    assert "leading '-'" in str(excinfo.value)


def test_oversized_version_reports_a_length_not_the_value(patched_run):
    """Echoing a huge "version" back is the illegibility the cap exists to stop."""
    patched_run(envelope(data={}))
    huge = "9" * 5000

    with pytest.raises(server.ComfyCliError) as excinfo:
        _switch(huge, ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "5000 characters" in message
    assert huge not in message


@pytest.mark.parametrize(
    ("given", "forwarded"),
    [
        ("0.24.0", "0.24.0"),
        ("v0.24.0", "v0.24.0"),  # both spellings name the same tag
        ("1.2.3-rc.1", "1.2.3-rc.1"),
        # Prerelease AND build metadata together — `semver.VersionInfo.parse`,
        # which is comfy-cli's own validator, accepts this, so refusing it here
        # would be the wrapper inventing a limitation the engine does not have.
        ("1.2.3-rc.1+build.5", "1.2.3-rc.1+build.5"),
        ("1.2.3+build.5", "1.2.3+build.5"),
        ("nightly", "nightly"),
        ("latest", "latest"),
        ("  NIGHTLY  ", "nightly"),  # normalized like `update_comfyui`'s target
        (" 0.24.0 ", "0.24.0"),
    ],
)
def test_accepted_versions_reach_argv_normalized(patched_plain_run, given, forwarded):
    """Every accepted spelling, and exactly what lands on the command line."""
    calls = patched_plain_run(0, stderr="Switched to 0.24.0")

    result = _switch(given, ctx=_FakeCtx())

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert cmd[4:] == ["update", "comfy", "--version", forwarded]
    assert result["switched_to"] == forwarded


# --- consent: a client that CAN be prompted ---------------------------------


def test_approved_switch_runs_the_command(patched_plain_run):
    """Accept -> the switch runs, and the prompt said what it would do."""
    calls = patched_plain_run(0, stderr="Updating ComfyUI in /ws...")

    result = _switch("0.24.0", ctx=(ctx := _FakeCtx(action="accept", approve=True)))

    assert len(ctx.elicitations) == 1
    prompt = ctx.elicitations[0]
    assert "0.24.0" in prompt
    assert "STASHES" in prompt
    assert "REINSTALLS" in prompt
    assert "restarted" in prompt
    assert calls[0]["cmd"][4:] == ["update", "comfy", "--version", "0.24.0"]
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
    """A refusal is enforced HERE — comfy-cli is never started, nothing changes."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="version switch not confirmed"):
        _switch("0.24.0", ctx=ctx)

    assert calls == []


def test_confirm_switch_is_not_a_way_around_the_prompt(patched_plain_run):
    """An agent setting `confirm_switch=True` itself does not authorize the switch.

    The hole this closes is the spend gates': a host's blanket "always allow this
    tool" toggle lets an agent set the argument for itself, which would otherwise
    be standing authority to rewrite the user's ComfyUI install.
    """
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="version switch not confirmed"):
        _switch("0.24.0", confirm_switch=True, ctx=ctx)

    assert len(ctx.elicitations) == 1  # asked anyway
    assert calls == []


def test_the_prompt_is_raised_even_when_confirm_switch_is_true(patched_plain_run):
    """The approving case of the rule above: asked, then run."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="accept", approve=True)

    _switch("0.24.0", confirm_switch=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert len(calls) == 1


def test_an_unknown_capability_still_asks(patched_plain_run):
    """A probe that ERRORS is "could not tell", never "cannot elicit".

    Demoting a capable client onto the caller's own say-so is the one outcome
    this gate exists to prevent, so the unknown case takes the prompt path.
    """

    class _BrokenSession:
        def check_client_capability(self, capability):
            raise RuntimeError("probe exploded")

    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="decline")
    ctx.session = _BrokenSession()

    with pytest.raises(server.ComfyCliError, match="version switch not confirmed"):
        _switch("0.24.0", confirm_switch=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
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
def test_bare_call_on_a_non_eliciting_client_changes_nothing(
    patched_plain_run, make_ctx
):
    """`confirm_switch=False` (the default) destroys nothing, and says why."""
    calls = patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError, match="confirm_switch=True"):
        _switch("0.24.0", ctx=make_ctx())

    assert calls == []


def test_confirm_switch_is_the_fallback_when_the_client_cannot_elicit(
    patched_plain_run,
):
    """On a client with no elicitation, `confirm_switch` is the documented route."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(supports_elicitation=False)

    result = _switch("0.24.0", confirm_switch=True, ctx=ctx)

    assert ctx.elicitations == []
    assert calls[0]["cmd"][4:] == ["update", "comfy", "--version", "0.24.0"]
    assert result["restart_required"] is True


# --- refusing while a server is running -------------------------------------


def test_refuses_while_a_local_comfyui_is_running(patched_plain_run, monkeypatch):
    """Reinstalling under a live process is unsafe — the caller is sent to stop it."""
    monkeypatch.setattr(server, "_local_comfyui_running", lambda: True)
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError, match="stop_comfyui") as excinfo:
        _switch("0.24.0", confirm_switch=True, ctx=ctx)

    assert calls == []  # no switch was run
    assert ctx.elicitations == []  # and nobody was asked to approve the impossible
    assert "Nothing was changed" in str(excinfo.value)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"server": {"running": True, "url": "http://localhost:8188"}}, True),
        ({"server": {"running": False, "url": None}}, False),
        ({"running": True}, True),  # a payload that hoists the flag
        ({"server": {}}, False),  # unreadable -> not running (see the docstring)
        ({}, False),
    ],
)
def test_running_probe_reads_comfy_env(monkeypatch, data, expected):
    """The real helper, against the shapes `comfy env` reports it in.

    Calls the captured original rather than the module attribute: the autouse
    fixture above has replaced that one, and this is the single test that is
    about the probe itself rather than about what the tool does with it.
    """
    monkeypatch.setattr(server, "server_info", lambda: dict(data))

    assert _REAL_RUNNING_PROBE() is expected


# --- an old comfy-cli -------------------------------------------------------


def test_old_comfy_cli_gets_an_upgrade_instruction(patched_plain_run):
    """`--version` rejected at parse time reads as a version gap, not a mystery."""
    calls = patched_plain_run(
        2,
        stderr=(
            "Usage: comfy update [OPTIONS] [TARGET]\n"
            "Try 'comfy update --help' for help.\n\n"
            "Error: No such option: --version"
        ),
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        _switch("0.24.0", ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "comfy update cli" in message
    assert "Nothing was changed" in message
    assert len(calls) == 1  # it really did try


def test_a_real_failure_keeps_its_own_message(patched_plain_run):
    """The degrade is narrow: a genuine failure is not relabelled a version gap."""
    patched_plain_run(1, stderr="error: Your local changes would be overwritten")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _switch("0.24.0", ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "returned no JSON" in message
    assert "comfy update cli" not in message


def test_an_error_envelope_still_propagates(patched_run):
    """comfy-cli's own coded refusal reaches the caller intact."""
    patched_run(
        envelope(
            ok=False,
            error={"code": "workspace_not_found", "message": "no ComfyUI path"},
        )
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        _switch("0.24.0", ctx=_FakeCtx())

    assert excinfo.value.code == "workspace_not_found"


# --- the rest of the contract ------------------------------------------------


def test_result_shape(patched_plain_run):
    """`{switched_to, result, restart_required}` — and it never restarts anything."""
    patched_plain_run(0, stderr="Updating ComfyUI in /ws...\nDone.")

    result = _switch("v0.24.0", ctx=_FakeCtx())

    assert result["switched_to"] == "v0.24.0"
    assert result["restart_required"] is True
    assert result["result"]["ok"] is True
    assert "Done." in result["result"]["message"]


def test_timeout_is_generous(patched_plain_run):
    """A dependency reinstall is minutes; the bound must not cut it off."""
    calls = patched_plain_run(0, stderr="done")

    _switch("0.24.0", ctx=_FakeCtx())

    assert calls[0]["timeout"] == server._SWITCH_TIMEOUT
    assert calls[0]["timeout"] >= 900.0


def test_refuses_while_an_update_holds_the_shared_lock(patched_plain_run):
    """`update_comfyui` and this tool mutate the same checkout — one at a time."""
    calls = patched_plain_run(0, stderr="done")
    assert server._UPDATE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(server.ComfyCliError, match="already running"):
            _switch("0.24.0", ctx=_FakeCtx())
    finally:
        server._UPDATE_LOCK.release()

    assert calls == []


def test_a_declined_switch_leaves_the_lock_free(patched_plain_run):
    """A refusal must not park the lock an in-flight update legitimately needs."""
    patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError, match="version switch not confirmed"):
        _switch("0.24.0", ctx=_FakeCtx(action="decline"))

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_the_switch_is_non_interactive(patched_plain_run):
    """git/pip must never stop to ask a question nobody can answer."""
    calls = patched_plain_run(0, stderr="done")

    _switch("0.24.0", ctx=_FakeCtx())

    assert calls[0]["stdin"] == server.subprocess.DEVNULL
    assert calls[0]["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert calls[0]["env"]["PIP_NO_INPUT"] == "1"


def test_unanswered_prompt_lapses_into_a_refusal(patched_plain_run, monkeypatch):
    """A client that advertises elicitation but never answers must not hang."""
    monkeypatch.setattr(server, "_ELICIT_TIMEOUT", 0.05)
    calls = patched_plain_run(0, stderr="done")

    class _SilentCtx(_FakeCtx):
        async def elicit(self, message, schema):
            self.elicitations.append(message)
            await asyncio.sleep(3600)

    with pytest.raises(server.ComfyCliError, match="version switch not confirmed"):
        _switch("0.24.0", confirm_switch=True, ctx=_SilentCtx())

    assert calls == []


def test_a_client_that_errors_on_the_prompt_names_the_manual_route(patched_plain_run):
    """No engine-side durable consent exists here, so the way out is the terminal."""
    calls = patched_plain_run(0, stderr="done")

    class _BoomCtx(_FakeCtx):
        async def elicit(self, message, schema):
            raise RuntimeError("no prompt surface")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _switch("0.24.0", confirm_switch=True, ctx=_BoomCtx())

    message = str(excinfo.value)
    assert "could not confirm the ComfyUI version switch" in message
    assert "comfy update comfy --version" in message
    assert calls == []


def test_instructions_teach_the_rollback_flow():
    """The server INSTRUCTIONS carry the flow, next to the lifecycle guidance."""
    assert "switch_comfyui_version" in server.INSTRUCTIONS
    assert "confirm_switch=True" in server.INSTRUCTIONS
