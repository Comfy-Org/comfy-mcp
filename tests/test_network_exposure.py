"""Tests for the network-exposure consent gate on the lifecycle launch tools.

``launch_comfyui`` / ``restart_comfyui`` forward ``extra_args`` to ComfyUI
verbatim after a ``--`` separator, and the local ComfyUI has **no
authentication** — so ``--listen`` on a non-loopback address or
``--enable-cors-header`` hands its full HTTP API to anything that can reach the
machine. The caller here may be a prompt-injected agent, so these lock in that
the USER, not the caller, authorizes that:

1. The DETECTOR: what counts as exposing (a bare ``--listen``, whose ComfyUI
   default is every interface; any non-loopback address; an argparse
   abbreviation of either flag) and what does not (the loopback carve-out,
   ``--port``, every other flag).
2. The CONSENT posture, mirroring ``switch_comfyui_version``: on a client that
   can be prompted the USER is asked, ``confirm_network_exposure=True`` is not a
   way around that prompt, and a refusal is enforced with no child spawned. Only
   a client that cannot be prompted falls back to the explicit argument, whose
   ``False`` default means a bare call exposes nothing.
3. That the gate costs the ordinary caller nothing: a launch with no extras, or
   with ``--port``, behaves exactly as before — no prompt, same argv.
4. Input hygiene on ``extra_args``: a NUL, an oversized entry, or too many of
   them is a named ``ComfyCliError`` rather than a bare ``ValueError`` /
   ``OSError`` out of ``subprocess``.
5. WHAT THE USER IS SHOWN, which is the thing their answer actually rests on:
   the whole argument list is echoed (so consent is not to a narrower action
   than the one that runs), that echo cannot restructure the prompt around
   itself, and the wording claims only what the detector established.

comfy-cli is mocked throughout: no real ComfyUI is ever launched.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from conftest import envelope
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_mcp import argv
from comfy_mcp.server import _internal as server


def _launch(*args, **kwargs):
    """Drive the async ``launch_comfyui`` tool from a sync test."""
    return asyncio.run(server.launch_comfyui(*args, **kwargs))


def _restart(*args, **kwargs):
    """Drive the async ``restart_comfyui`` tool from a sync test."""
    return asyncio.run(server.restart_comfyui(*args, **kwargs))


#: The two tools under test, parametrized together everywhere the contract is
#: shared — `restart_comfyui` forwards `extra_args` to the same launch, so a gate
#: that covered only one of them would leave the bypass wide open.
_TOOLS = pytest.mark.parametrize(
    "drive", [_launch, _restart], ids=["launch", "restart"]
)

#: How many `comfy` invocations each tool makes on the happy path: `restart` runs
#: `stop` first. Keyed by the driver so the argv assertions can index the LAUNCH
#: call without caring which tool produced it.
_LAUNCH_CALL_INDEX = {_launch: 0, _restart: 1}


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake FastMCP ``Context`` that answers the elicitation with ``action``.

    A local copy of the other gates' fakes rather than a shared one, following
    the convention those files set: this gate's prompt must be assertable on its
    own, so a change to the spend or version-switch prompt cannot silently
    retune these tests.
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


#: A client that advertised no elicitation capability — the `confirm_*` fallback
#: path. `ctx=None` reaches the same branch; both are exercised below.
def _blind_ctx() -> _FakeCtx:
    return _FakeCtx(supports_elicitation=False)


# --- the detector -----------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args",
    [
        # Bare `--listen`: ComfyUI's parser declares it `nargs="?"` with
        # `const="0.0.0.0,::"`, so "no address" IS every interface.
        ["--listen"],
        ["--cpu", "--listen"],  # last token, nothing to consume
        # `--listen` followed by another FLAG: argparse does not take an
        # option-like token as the optional value, so this is the bare form too.
        ["--listen", "--cpu"],
        ["--listen", "0.0.0.0"],
        ["--listen=0.0.0.0"],
        ["--listen", "::"],
        ["--listen", "192.168.1.10"],
        ["--listen", "0.0.0.0,::"],
        # One public address among loopback ones still exposes the server.
        ["--listen", "127.0.0.1,0.0.0.0"],
        # Unclassifiable values fail CLOSED rather than being waved through.
        ["--listen", ""],
        ["--listen=  "],
        ["--listen", "0177.0.0.1"],  # octal obfuscation `ipaddress` rejects
        ["--listen", "my-host.local"],
        # argparse abbreviations reach `args.listen` just as surely.
        ["--liste", "0.0.0.0"],
        ["--lis=0.0.0.0"],
        ["--l"],
        # CORS at any value: bare means `*`, and a named origin is still a grant.
        ["--enable-cors-header"],
        ["--enable-cors-header", "https://example.test"],
        ["--enable-cors-header=*"],
        ["--enable-cors-heade"],
        # Buried among innocuous flags.
        ["--port", "8189", "--cpu", "--listen", "0.0.0.0", "--lowvram"],
    ],
)
def test_detector_flags_network_exposing_args(extra_args):
    """Every spelling that would publish the API has to reach the gate."""
    assert server._network_exposing_args(extra_args)


@pytest.mark.parametrize(
    "extra_args",
    [
        [],
        ["--port", "8189"],
        ["--port=8189", "--cpu", "--lowvram"],
        ["--cpu"],
        # The loopback carve-out: this is the DEFAULT bind spelled out, i.e. LESS
        # reach than a bare launch, so prompting here would only teach the user to
        # click through the prompt that matters.
        ["--listen", "127.0.0.1"],
        ["--listen=127.0.0.1"],
        ["--listen", "127.0.0.53"],  # all of 127.0.0.0/8
        ["--listen", "::1"],
        ["--listen", "[::1]"],  # the bracketed form users copy out of URLs
        ["--listen", "localhost"],
        ["--listen", "LOCALHOST"],
        ["--listen", "127.0.0.1,::1"],  # both loopback stacks
        # IPv4-mapped v6: the kernel binds the v4 loopback it carries.
        ["--listen", "::ffff:127.0.0.1"],
        ["--listen", " 127.0.0.1 "],
        ["--listen", "127.0.0.1", "--cpu"],  # the value is consumed, not rescanned
        # A flag that merely shares a prefix with `--listen` is not `--listen`.
        ["--listen-really-hard"],
        ["--listener", "0.0.0.0"],
    ],
)
def test_detector_passes_everything_else_through(extra_args):
    """No prompt for the args every existing caller actually passes."""
    assert server._network_exposing_args(extra_args) == ()


def test_detector_reports_both_flags_when_both_are_present():
    """The prompt names everything that exposes, not just the first one found."""
    flags = server._network_exposing_args(
        ["--enable-cors-header", "--listen", "0.0.0.0"]
    )

    assert flags == (server._CORS_FLAG, server._LISTEN_FLAG)


def test_detector_flags_a_repeated_listen_whose_earlier_value_was_public():
    """A deliberate over-rejection: last-wins is not what a security gate bets on."""
    assert server._network_exposing_args(
        ["--listen", "0.0.0.0", "--listen", "::1"]
    ) == (server._LISTEN_FLAG,)


def test_detector_does_not_repeat_a_flag_it_found_twice():
    """`--listen` twice is one thing to confirm, named once in the prompt."""
    assert server._network_exposing_args(["--listen", "--listen", "0.0.0.0"]) == (
        server._LISTEN_FLAG,
    )


# --- the ordinary caller pays nothing ---------------------------------------


@_TOOLS
def test_ordinary_launch_needs_no_consent_and_keeps_its_argv(patched_run, drive):
    """No extras, no ctx, no prompt — byte-identical to the pre-gate behavior."""
    calls = patched_run(envelope(data={"pid": 42}))

    assert drive() == {"pid": 42}

    launch = calls[_LAUNCH_CALL_INDEX[drive]]["cmd"]
    assert launch[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert launch[4:] == ["launch", "--background"]  # no extras -> no `--`


@_TOOLS
@pytest.mark.parametrize(
    "extra_args",
    [["--port", "8189"], ["--listen", "127.0.0.1"], ["--listen=::1"]],
)
def test_unexposing_extra_args_are_forwarded_without_a_prompt(
    patched_run, drive, extra_args
):
    """A loopback bind or an unrelated flag reaches ComfyUI verbatim, unprompted."""
    calls = patched_run(envelope(data={}))
    ctx = _FakeCtx()

    drive(extra_args, ctx=ctx)

    assert ctx.elicitations == []  # the user was never bothered
    assert calls[_LAUNCH_CALL_INDEX[drive]]["cmd"][4:] == [
        "launch",
        "--background",
        "--",
        *extra_args,
    ]


# --- the consent gate -------------------------------------------------------

#: The exposing spellings the consent tests run end to end. Kept small on
#: purpose — the full spelling matrix is the detector's job above; these prove
#: the WIRING from each shape through to a refusal.
_EXPOSING = pytest.mark.parametrize(
    "extra_args",
    [
        ["--listen"],
        ["--listen", "0.0.0.0"],
        ["--listen=0.0.0.0"],
        ["--enable-cors-header"],
    ],
    ids=["bare-listen", "listen-any", "listen-any-inline", "cors"],
)


@_TOOLS
@_EXPOSING
@pytest.mark.parametrize("ctx", [None, "blind"], ids=["no-ctx", "no-elicitation"])
def test_unpromptable_client_refuses_without_the_confirm_flag(
    patched_run, drive, extra_args, ctx
):
    """The `False` default is why a bare call from such a client exposes nothing."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError) as excinfo:
        drive(extra_args, ctx=_blind_ctx() if ctx == "blind" else None)

    message = str(excinfo.value)
    assert "confirm_network_exposure=True" in message
    assert "NO authentication" in message  # says what exposure actually means
    assert "only once they have actually agreed" in message
    assert "left as it was" in message
    assert calls == []  # nothing was stopped and nothing was launched


@_TOOLS
@_EXPOSING
def test_unpromptable_client_proceeds_with_the_confirm_flag(
    patched_run, drive, extra_args
):
    """The documented fallback: the user agreed out-of-band, so the launch runs."""
    calls = patched_run(envelope(data={"pid": 9}))

    assert drive(extra_args, confirm_network_exposure=True, ctx=_blind_ctx()) == {
        "pid": 9
    }

    assert calls[_LAUNCH_CALL_INDEX[drive]]["cmd"][4:] == [
        "launch",
        "--background",
        "--",
        *extra_args,
    ]


@_TOOLS
@_EXPOSING
def test_promptable_client_asks_the_user_and_launches_on_approval(
    patched_run, drive, extra_args
):
    """The human is shown what will happen, and an approval lets it through."""
    calls = patched_run(envelope(data={"pid": 5}))
    ctx = _FakeCtx()

    assert drive(extra_args, ctx=ctx) == {"pid": 5}

    assert len(ctx.elicitations) == 1
    prompt = ctx.elicitations[0]
    assert "NO authentication" in prompt
    assert "local network" in prompt
    assert calls[_LAUNCH_CALL_INDEX[drive]]["cmd"][4:] == [
        "launch",
        "--background",
        "--",
        *extra_args,
    ]


@_TOOLS
@_EXPOSING
@pytest.mark.parametrize(
    ("action", "approve"),
    [("decline", True), ("cancel", True), ("accept", False)],
    ids=["declined", "cancelled", "accepted-but-said-no"],
)
def test_promptable_client_refusal_spawns_nothing(
    patched_run, drive, extra_args, action, approve
):
    """Every non-approval fails closed — including an accept that never said yes."""
    calls = patched_run(envelope(data={}))
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError) as excinfo:
        drive(extra_args, ctx=ctx)

    assert "network exposure not confirmed" in str(excinfo.value)
    assert calls == []  # in particular, `restart` did not stop the running server


@_TOOLS
@_EXPOSING
def test_confirm_flag_does_not_suppress_the_prompt(patched_run, drive, extra_args):
    """A host's "always allow this tool" toggle is not the user's permission.

    The whole point of this gate is that the CALLER may be a prompt-injected
    agent, so on a client that can be prompted the human is asked every time and
    `confirm_network_exposure` grants nothing.
    """
    calls = patched_run(envelope(data={}))
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="declined"):
        drive(extra_args, confirm_network_exposure=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert calls == []


@_TOOLS
def test_unreadable_client_capability_asks_rather_than_trusting_the_caller(
    patched_run, drive
):
    """An errored capability probe is UNKNOWN, and unknown must not demote to the flag."""

    class _ExplodingSession:
        def check_client_capability(self, capability):
            raise RuntimeError("probe blew up")

    ctx = _FakeCtx(action="decline")
    ctx.session = _ExplodingSession()
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="declined"):
        drive(["--listen", "0.0.0.0"], confirm_network_exposure=True, ctx=ctx)

    assert len(ctx.elicitations) == 1  # asked anyway
    assert calls == []


@_TOOLS
def test_refusal_names_both_flags_when_both_are_present(patched_run, drive):
    """The user must be told everything they are being asked to approve."""
    patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError) as excinfo:
        drive(["--listen", "0.0.0.0", "--enable-cors-header"], ctx=_blind_ctx())

    message = str(excinfo.value)
    assert "`--listen`" in message
    assert "`--enable-cors-header`" in message


def test_a_declined_restart_does_not_stop_the_running_server(monkeypatch):
    """The gate runs BEFORE the stop, so a refusal leaves the server alone.

    Gating after the stop would be worse than not gating at all: the user's
    running ComfyUI would be killed and then not brought back.
    """
    stopped: list = []
    launched: list = []
    monkeypatch.setattr(server, "stop_comfyui", lambda: stopped.append(True))
    monkeypatch.setattr(
        server, "_launch_comfyui_sync", lambda extra_args: launched.append(extra_args)
    )

    with pytest.raises(server.ComfyCliError, match="network exposure not confirmed"):
        _restart(["--listen", "0.0.0.0"], ctx=_blind_ctx())

    assert stopped == []
    assert launched == []


# --- `extra_args` input hygiene ---------------------------------------------


@_TOOLS
def test_nul_in_extra_args_is_a_named_error(patched_run, drive):
    """A NUL must be this module's error, not `subprocess`'s bare ValueError."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        drive(["--cpu", "--port\0"])

    assert calls == []


@_TOOLS
def test_nul_is_rejected_before_the_consent_prompt(patched_run, drive):
    """Malformed input is named without first making the user answer a prompt."""
    patched_run(envelope(data={}))
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        drive(["--listen", "0.0.0.0\0"], ctx=ctx)

    assert ctx.elicitations == []


@_TOOLS
@pytest.mark.parametrize(
    "extra_args",
    [
        # A bare string would otherwise be splatted one CHARACTER per argv slot.
        "--cpu",
        {"--cpu": True},
        [8189],
        ["--port", 8189],
        [None],
    ],
)
def test_malformed_extra_args_are_named_not_crashed(patched_run, drive, extra_args):
    """A non-list, or a non-string entry, is a `ComfyCliError` naming the input."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="extra_args"):
        drive(extra_args)

    assert calls == []


@_TOOLS
def test_too_many_extra_args_is_a_named_error(patched_run, drive):
    """An argv the kernel would refuse must be this module's error, not an OSError."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="entry maximum"):
        drive(["--cpu"] * (argv._MAX_EXTRA_ARGS + 1))

    assert calls == []


@_TOOLS
def test_oversized_extra_arg_entry_is_a_named_error(patched_run, drive):
    """Same for one huge entry: `subprocess` would raise `Argument list too long`."""
    calls = patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError, match="character maximum"):
        drive(["--listen", "1" * (argv._MAX_EXTRA_ARG_LEN + 1)])

    assert calls == []


def test_the_size_bound_leaves_realistic_argument_lists_alone(patched_run):
    """The ceilings are guards, not a usability tax: a real flag list still runs."""
    calls = patched_run(envelope(data={}))

    _launch(["--port", "8189", "--cpu", "--base-directory", "/tmp/" + "x" * 200])

    assert calls[0]["cmd"][4:5] == ["launch"]


# --- what the user is actually shown ----------------------------------------


@_TOOLS
def test_prompt_echoes_the_whole_argument_list(patched_run, drive):
    """Consent must be to the ACTUAL command line, not just the flag categories.

    The same `extra_args` that trip the gate can carry `--base-directory` (which
    moves the file roots the exposure reaches) or an address the user did not
    expect, so naming only "`--listen`" would ask them to approve a materially
    narrower action than the one that runs.
    """
    patched_run(envelope(data={"pid": 1}))
    ctx = _FakeCtx()

    drive(["--listen", "0.0.0.0", "--base-directory", "/srv/models"], ctx=ctx)

    prompt = ctx.elicitations[0]
    assert "--listen 0.0.0.0 --base-directory /srv/models" in prompt


@_TOOLS
def test_prompt_cannot_be_rewritten_by_the_arguments_it_echoes(patched_run, drive):
    """The echoed args are CALLER text, so they must not break out of the code span.

    A backtick would close the markdown span on a client that renders it, letting
    a prompt-injected agent append its own reassurance ("loopback only") to the
    very warning the user is answering.
    """
    patched_run(envelope(data={}))
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="declined"):
        drive(["--listen", "0.0.0.0`\n**(loopback only)**"], ctx=ctx)

    prompt = ctx.elicitations[0]
    # Still shown — the user should see what was asked for — but with the span
    # delimiter and the line break neutralized, so it cannot restructure the
    # prompt around itself.
    assert "--listen 0.0.0.0' **(loopback only)**" in prompt
    assert "\n" not in prompt


@_TOOLS
def test_prompt_and_refusal_do_not_overstate_what_the_detector_knows(
    patched_run, drive
):
    """A last-wins repeat really does bind loopback, so the wording stays hedged.

    The gate flags it anyway (over-rejecting is the safe direction), but the
    sentence the user decides on must claim only what was established: the flag
    is there and this server could not confirm it keeps the server private.
    """
    patched_run(envelope(data={}))

    with pytest.raises(server.ComfyCliError) as excinfo:
        drive(["--listen", "0.0.0.0", "--listen", "127.0.0.1"], ctx=_blind_ctx())

    message = str(excinfo.value)
    assert "could not confirm" in message
    assert "may bind" in message


# --- QA 0.8.0: a refusal nobody made ----------------------------------------
# Four gates reported "the user declined ..." in sessions where no prompt was
# ever displayed. Failing closed was correct; naming a human refusal was not —
# it sends the reader to argue with a user who was never asked, and hides that
# the CLIENT is what needs fixing.


def test_cancelled_elicitation_does_not_claim_the_user_declined():
    """A cancelled/auto-answered prompt is not a decision by anyone."""
    ctx = _FakeCtx(action="cancel")

    with pytest.raises(server.ComfyCliError) as exc:
        _launch(extra_args=["--listen"], ctx=ctx)

    message = str(exc.value)
    assert "did not present the confirmation prompt" in message
    assert "nobody was asked" in message
    # The specific lie this guards against.
    assert "declined" not in message
    # Still fails closed.
    assert "not confirmed" in message


def test_declined_elicitation_still_reports_a_decline():
    """An actual decline keeps reporting a refusal — the fix is not a blanket rename."""
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError) as exc:
        _launch(extra_args=["--listen"], ctx=ctx)

    assert "declined" in str(exc.value)


# --- QA 0.10.0: a decline is not proof of a person ---------------------------
# The 0.8.0 fix above rests on "action == 'decline'" meaning a human said no.
# It does not. At least one shipping agent client answers "decline" for a
# session with no approval surface registered, a dispatch exception, a CLI
# prompt that raised, and a failed session lookup — none of which involve a
# person. The server cannot tell those from a real refusal, so it must not
# claim one.


def test_decline_does_not_assert_that_a_person_refused():
    """A decline reports the refusal without naming who made it."""
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError) as exc:
        _launch(extra_args=["--listen"], ctx=ctx)

    message = str(exc.value)
    # Reports the refusal, and still fails closed.
    assert "declined" in message
    assert "not confirmed" in message
    # But never asserts the human. This is the claim the server cannot support.
    assert "the user declined" not in message
    # And says so explicitly, so a reader who saw no prompt knows why.
    assert "cannot display a prompt" in message
    assert "nobody was asked" in message


def test_decline_hedge_names_no_way_around_the_gate():
    """The hedge must not hand an agent a bypass for a refusal it was just given."""
    message = server._DECLINE_MAY_BE_AUTOMATIC
    for bypass in ("confirm_", "COMFY_MCP_ASSUME_CONSENT", "consent always"):
        assert bypass not in message


def _declined_messages(monkeypatch) -> dict[str, str]:
    """Every gate's DELIVERED refusal, driven through a client that declines.

    All eight gates funnel through `_elicit_approval`, so forcing its False —
    the `action == "decline"` outcome, and the only one that does not raise on
    its own — reaches each gate's own declined text without standing up eight
    different tool stacks. What is asserted is therefore the ASSEMBLED string a
    caller receives, not a constant read in isolation.
    """

    async def _declines(*_args, **_kwargs):
        return False

    monkeypatch.setattr(server, "_elicit_approval", _declines)
    monkeypatch.setattr(server, "_client_elicitation_support", lambda _ctx: True)
    monkeypatch.setattr(server, "_engine_auto_confirms", lambda: False)
    ctx = object()

    def _refusal(coro):
        with pytest.raises(server.ComfyCliError) as exc:
            asyncio.run(coro)
        return str(exc.value)

    messages = {
        "partner_generate": _refusal(
            server._resolve_spend_consent("flux-pro", False, ctx)
        ),
        "run_template": _refusal(
            server._resolve_template_spend_consent("txt2img", True, ctx)
        ),
        "run_workflow": _refusal(
            server._resolve_workflow_spend_consent("graph.json", True, ctx)
        ),
        "launch_network_exposure": _refusal(
            server._resolve_network_exposure_consent(["--listen"], False, ctx, "launch")
        ),
        "update_comfyui_all": _refusal(server._resolve_update_all_consent(False, ctx)),
        "switch_comfyui_version": _refusal(
            server._resolve_switch_consent("v0.3.0", False, ctx)
        ),
        "install_nodes": _refusal(
            server._resolve_install_consent("some-pack", False, ctx)
        ),
    }
    # The kill gate reports its refusal as a `reason` folded into the restart's
    # guidance error rather than as a raise (see `_KillDecision`), so it is
    # collected differently — but it is the same decline, and the same contract.
    listener = server._UntrackedListener(pid=4242, port=8188, cmdline="python main.py")
    decision = asyncio.run(server._resolve_kill_untracked_consent(listener, False, ctx))
    assert not decision.approved
    messages["restart_kill_untracked"] = decision.reason
    return messages


def test_no_gate_asserts_a_person_on_a_decline(monkeypatch):
    """Every gate, not just the launch one, reports the refusal anonymously.

    Matching on "the user" rather than the exact old sentence: the claim that
    must not survive is naming WHO refused, in any phrasing.
    """
    messages = _declined_messages(monkeypatch)
    # All eight, so a gate added later cannot quietly skip the contract.
    assert len(messages) == 8

    for gate, message in messages.items():
        assert "declined" in message, gate  # still reports the refusal
        assert "the user" not in message, gate  # but never names who made it
        assert "You declined" not in message, gate


def test_every_gate_carries_the_hedge_on_a_decline(monkeypatch):
    """The hedge is what tells a reader who saw no prompt why they saw none."""
    for gate, message in _declined_messages(monkeypatch).items():
        assert server._DECLINE_MAY_BE_AUTOMATIC.strip() in message, gate


def test_no_gate_lets_a_caller_relayed_name_break_the_refusal(monkeypatch):
    """A refusal quotes caller text into a code span, exactly as the prompt does.

    `_validate_generate_model` permits backticks and unbounded length, and
    `repr()` escapes newlines but not backticks — so before this both spend
    refusals let a relayed name close the span and write its own text into the
    message. The prompts were already sanitized; the refusals had to match.
    """

    async def _declines(*_args, **_kwargs):
        return False

    monkeypatch.setattr(server, "_elicit_approval", _declines)
    monkeypatch.setattr(server, "_client_elicitation_support", lambda _ctx: True)
    monkeypatch.setattr(server, "_engine_auto_confirms", lambda: False)
    evil = "x`  **FREE — no credits will be spent** `y"
    ctx = object()

    for coro in (
        server._resolve_spend_consent(evil, False, ctx),
        server._resolve_template_spend_consent(evil, True, ctx),
    ):
        with pytest.raises(server.ComfyCliError) as exc:
            asyncio.run(coro)
        message = str(exc.value)
        assert evil not in message  # never echoed raw
        assert "**FREE" in message  # the text is still SHOWN, just declawed
        # Only the code spans this server opened survive, so the relayed text
        # cannot restructure the message around itself.
        assert message.count("`") % 2 == 0


def test_the_generate_refusal_as_delivered_names_no_bypass(monkeypatch):
    """The constant's invariant, checked on the message a caller actually gets.

    Only `partner_generate` is asserted whole: it is the gate where every path
    spends, so any route named in its refusal is a route to unconsented spend.
    The two opt-in verbs deliberately DO name `confirm_spend=False` — for
    `run_template` that is the free-run path and cannot spend, and
    `run_workflow` names it only to forbid it — so a blanket assertion there
    would pin the wrong contract.
    """
    message = _declined_messages(monkeypatch)["partner_generate"]

    for bypass in ("confirm_", "COMFY_MCP_ASSUME_CONSENT", "consent always"):
        assert bypass not in message


# --- Operator pre-authorization (COMFY_MCP_ASSUME_CONSENT) --------------------
# The clients agents actually run under answer the elicitation WITHOUT rendering
# a prompt, so five gated tools were unusable. This lets the human who CONFIGURED
# the server pre-authorize specific gates out-of-band. The agent cannot set an
# environment variable, which is what keeps this from becoming self-consent.


def test_preauthorized_gate_skips_the_prompt(monkeypatch):
    """A named gate proceeds without contacting the client at all."""
    monkeypatch.setenv("COMFY_MCP_ASSUME_CONSENT", "network_exposure")
    ctx = _FakeCtx(action="cancel")  # would fail closed if it were asked

    # Whether the launch itself then succeeds is not what this asserts (no CLI is
    # mocked here); the property is that the GATE let it through without ever
    # contacting the client.
    with contextlib.suppress(server.ComfyCliError):
        _launch(extra_args=["--listen"], ctx=ctx)

    assert ctx.elicitations == []


def test_unnamed_gate_still_asks(monkeypatch):
    """Authorizing one gate does not authorize another."""
    monkeypatch.setenv("COMFY_MCP_ASSUME_CONSENT", "install_node")
    ctx = _FakeCtx(action="cancel")

    with pytest.raises(server.ComfyCliError):
        _launch(extra_args=["--listen"], ctx=ctx)

    assert ctx.elicitations != []


def test_spend_can_never_be_preauthorized_even_by_all(monkeypatch):
    """`all` must not reach the money gates.

    The spend wordings carry no `consent_token`, so no value of the variable can
    match them. Money keeps ONE owner — comfy-cli's own durable
    `comfy generate consent always` — rather than two mechanisms that can drift.
    """
    monkeypatch.setenv("COMFY_MCP_ASSUME_CONSENT", "all")

    assert server._is_preauthorized(server._SPEND_APPROVAL_WORDING) is False
    assert server._is_preauthorized(server._OPTIN_SPEND_APPROVAL_WORDING) is False
    # While a non-spend gate IS covered by "all".
    assert server._is_preauthorized(server._NETWORK_APPROVAL_WORDING) is True


def test_unset_variable_changes_nothing(monkeypatch):
    """Default posture is unchanged: every gate still prompts."""
    monkeypatch.delenv("COMFY_MCP_ASSUME_CONSENT", raising=False)

    for wording in (
        server._NETWORK_APPROVAL_WORDING,
        server._UPDATE_ALL_APPROVAL_WORDING,
        server._SWITCH_APPROVAL_WORDING,
        server._INSTALL_APPROVAL_WORDING,
        server._SPEND_APPROVAL_WORDING,
    ):
        assert server._is_preauthorized(wording) is False


def test_token_parsing_tolerates_spacing_and_case(monkeypatch):
    """An operator hand-editing mcp.json should not be caught by whitespace."""
    monkeypatch.setenv("COMFY_MCP_ASSUME_CONSENT", " Install_Node , version_switch ")

    assert server._is_preauthorized(server._INSTALL_APPROVAL_WORDING) is True
    assert server._is_preauthorized(server._SWITCH_APPROVAL_WORDING) is True
    assert server._is_preauthorized(server._NETWORK_APPROVAL_WORDING) is False
