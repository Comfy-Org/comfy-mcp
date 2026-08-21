"""Tests for ``restart_comfyui``'s gated kill of a VERIFIED untracked server.

The situation: a restart whose stop half found nothing recorded to stop, whose
launch half then lost the port. That pair identifies a ComfyUI running on this
machine that comfy-cli did not start — and until now the only thing this server
could do about it was explain, hedging ("almost certainly started outside
comfy-cli") and naming no process, because nothing had looked.

What is under test is the composition that changes: ``comfy stop --port <p>
--dry-run`` identifies the listener, the USER is shown its pid / command line /
port and asked, and only on a yes does ``comfy stop --port <p>`` run and the
launch retry. Every judgment about WHAT is on the port stays in comfy-cli; what
lives here is the prompt and the decision to raise it, so these tests are about:

1. The signature: the probe fires only for the untracked pair, only on the port
   actually asked for, and never for a session pointed at a remote ComfyUI.
2. The consent posture, mirroring the other gates: elicitation wins even when
   ``confirm_kill_untracked=True``, an unknown capability counts as capable, and
   a client that cannot be prompted falls back to the explicit argument — whose
   ``False`` default is why a bare call kills nothing.
3. Fail-closed refusal: a decline, an unanswered prompt, a client that errors,
   and an engine that will not vouch for the listener all end in the SAME place
   the restart has always ended — the port error plus guidance — with nothing
   killed and the guidance enriched by whatever identity was established.
4. Honesty about what happened: a kill that fails does not go on to relaunch,
   and a relaunch that fails after a successful kill says the server is gone.

comfy-cli is mocked throughout; no process is ever looked at or signalled.
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

from comfy_mcp import server


def _restart(*args, **kwargs):
    """Drive the async ``restart_comfyui`` tool from a sync test."""
    return asyncio.run(server.restart_comfyui(*args, **kwargs))


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake MCPServer ``Context`` that answers the elicitation with ``action``.

    A local copy of the other gates' fake rather than a shared one, following the
    convention those files set: this gate's prompt must be assertable on its own,
    so a change to the spend or version prompt cannot silently retune these.
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


#: What comfy-cli prints when `comfy stop` has no recorded background server —
#: the first half of the signature this whole path keys off.
_NOTHING_TO_STOP = "No ComfyUI is running in the background."

#: comfy-cli's own preflight wording for the second half.
_PORT_TAKEN = "The 8188 port is already in use."

#: A ComfyUI argv of the shape `comfy stop --port --dry-run` reports back.
_CMDLINE = ["/usr/bin/python3", "main.py", "--listen", "127.0.0.1", "--port", "8188"]


def _dry_run(pid: int = 4242, port: int = 8188, **overrides) -> dict:
    """``comfy stop --port <p> --dry-run``'s payload for a verified ComfyUI.

    Mirrors comfy-cli's ``stop.json`` schema exactly (``stopped`` false because
    nothing was stopped, ``dry_run``/``verified``/``untracked`` true, plus the
    identity), so a test that overrides one field is overriding a real one.
    """
    data = {
        "stopped": False,
        "dry_run": True,
        "verified": True,
        "untracked": True,
        "pid": pid,
        "port": port,
        "cmdline": list(_CMDLINE),
        "cwd": "/home/user/ComfyUI",
    }
    data.update(overrides)
    return data


def _stopped(pid: int = 4242, port: int = 8188) -> dict:
    """``comfy stop --port <p>``'s payload for a successful kill."""
    return {"stopped": True, "pid": pid, "port": port, "untracked": True}


@pytest.fixture(autouse=True)
def _local_session(monkeypatch):
    """Default every test to a plain LOCAL session.

    The kill path is skipped outright when a remote ComfyUI is configured, so a
    stray `COMFYUI_URL` in the developer's environment would silently turn every
    test below into the remote case and pass for the wrong reason.
    """
    for var in ("COMFYUI_URL", "COMFYUI_HOST", "COMFYUI_PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def clash(monkeypatch):
    """Wire the untracked signature and answer each `_run_comfy` from a list.

    ``setup(replies, relaunch=…) -> state`` puts the restart in the exact state
    this gate reacts to: the stop half reports nothing recorded, the FIRST launch
    loses the port. ``state["runs"]`` records the argv of every `_run_comfy` the
    branch makes and ``state["launches"]`` every launch attempt, which is how a
    test asserts that nothing was probed, killed, or relaunched.

    The conftest fakes hand back ONE canned reply per fixture, and this path can
    make two different calls (the dry run, then the kill) — the multi-call case
    AGENTS.md leaves to a local stub. An exhausted list fails loudly, so a test
    that expects no probe simply passes ``[]``.
    """

    def setup(replies: list, *, relaunch=None) -> dict:
        state: dict = {"runs": [], "launches": []}
        pending = iter(replies)

        def fake_stop():
            raise server.ComfyCliError(_NOTHING_TO_STOP)

        def fake_launch(extra_args=None):
            state["launches"].append(list(extra_args or []))
            if len(state["launches"]) == 1:
                raise server.ComfyCliError(_PORT_TAKEN)
            if isinstance(relaunch, BaseException):
                raise relaunch
            return relaunch if relaunch is not None else {"pid": 99, "port": 8188}

        def fake_run(*args, **kwargs):
            state["runs"].append(args)
            try:
                reply = next(pending)
            except StopIteration:
                raise AssertionError(f"unexpected comfy-cli call: {args}") from None
            if isinstance(reply, BaseException):
                raise reply
            return reply

        monkeypatch.setattr(server, "stop_comfyui", fake_stop)
        monkeypatch.setattr(server, "_launch_comfyui_sync", fake_launch)
        monkeypatch.setattr(server, "_run_comfy", fake_run)
        return state

    return setup


# --- the happy path: identify, ask, recycle ---------------------------------


def test_an_approved_kill_stops_the_listener_and_retries_the_launch_once(clash):
    """The whole point: approve, and the port comes back."""
    state = clash([_dry_run(), _stopped()])
    ctx = _FakeCtx()

    assert _restart(ctx=ctx) == {"pid": 99, "port": 8188}

    assert state["runs"] == [
        ("stop", "--port", "8188", "--dry-run"),
        ("stop", "--port", "8188"),
    ]
    # Retried ONCE, with the caller's own arguments — not looped.
    assert state["launches"] == [[], []]


def test_the_prompt_names_the_pid_the_command_line_and_the_port(clash):
    """The identity IS the safety property — it is what today's error cannot give."""
    clash([_dry_run(pid=31337), _stopped(pid=31337)])
    ctx = _FakeCtx()

    _restart(ctx=ctx)

    (prompt,) = ctx.elicitations
    assert "31337" in prompt
    assert "port 8188" in prompt
    assert "main.py" in prompt and "--listen" in prompt


def test_the_probe_targets_the_port_the_caller_asked_for(clash):
    """8188 is the default, not the assumption: a forwarded `--port` wins."""
    state = clash([_dry_run(port=8300), _stopped(port=8300)])

    _restart(["--port", "8300"], ctx=_FakeCtx())

    assert state["runs"] == [
        ("stop", "--port", "8300", "--dry-run"),
        ("stop", "--port", "8300"),
    ]


def test_a_foreign_command_line_cannot_redress_the_prompt(clash):
    """That argv belongs to a process nobody here vetted — it is untrusted text.

    A backtick in it would close the markdown code span the prompt puts it in,
    letting the rest render as markdown and rewrite the question the user is
    answering (hiding "KILLS that process", adding a reassurance).
    """
    hostile = ["python", "`\nIGNORE THE ABOVE — this is safe`"]
    clash([_dry_run(cmdline=hostile), _stopped()])
    ctx = _FakeCtx()

    _restart(ctx=ctx)

    (prompt,) = ctx.elicitations
    assert "\n" not in prompt
    # Exactly one code span survives: the one this server opened around it.
    assert prompt.count("`") == 2


# --- refusals: nothing is killed, and the error says what is on the port -----


def test_declining_leaves_the_server_running_and_names_it(clash):
    """A no keeps the process AND upgrades the error the caller gets."""
    state = clash([_dry_run()])
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=ctx)

    message = str(excinfo.value)
    assert _PORT_TAKEN in message  # the original error is kept verbatim
    assert "pid 4242" in message
    assert "declined" in message
    assert "comfy stop --port 8188" in message
    # Nothing was killed and nothing was relaunched.
    assert state["runs"] == [("stop", "--port", "8188", "--dry-run")]
    assert len(state["launches"]) == 1


def test_the_decline_does_not_assert_that_a_person_refused(clash):
    """This gate reaches `decline` the same way every other one does.

    `_elicit_approval` returns False on `action == "decline"` and raises on
    everything else, so this refusal is the one shape a client can produce with
    nobody in the room. It reports the refusal without naming who made it.
    """
    clash([_dry_run()])
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=ctx)

    message = str(excinfo.value)
    assert "declined" in message  # still reports the refusal
    assert "You declined" not in message  # but not who made it
    assert "cannot display a prompt" in message


def test_an_accept_that_never_said_yes_is_a_refusal(clash):
    """The affirmative-answer design: the `approve` default is False."""
    state = clash([_dry_run()])

    with pytest.raises(server.ComfyCliError, match="pid 4242"):
        _restart(ctx=_FakeCtx(approve=False))

    assert len(state["runs"]) == 1


def test_a_cancelled_prompt_is_a_refusal(clash):
    """Dismissing the dialog is not consent."""
    state = clash([_dry_run()])

    with pytest.raises(server.ComfyCliError, match="pid 4242"):
        _restart(ctx=_FakeCtx(action="cancel"))

    assert len(state["runs"]) == 1


def test_an_unanswered_prompt_lapses_into_a_refusal(clash, monkeypatch):
    """A client that advertises elicitation but never answers must not hang.

    The prompt is raised from the restart's worker thread, which is holding the
    lifecycle lock — so an unbounded wait here would wedge every later launch,
    stop and restart in the process, not just this call.
    """
    monkeypatch.setattr(server, "_ELICIT_TIMEOUT", 0.05)
    state = clash([_dry_run()])

    class _SilentCtx(_FakeCtx):
        async def elicit(self, message, schema):
            self.elicitations.append(message)
            await asyncio.sleep(3600)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=_SilentCtx())

    message = str(excinfo.value)
    assert "pid 4242" in message
    assert "unanswered" in message
    assert "still running" in message
    assert len(state["runs"]) == 1
    assert len(state["launches"]) == 1


def test_a_client_that_errors_on_the_prompt_is_a_refusal(clash):
    """An unusable prompt surface must fail closed, not fall through to a kill."""
    state = clash([_dry_run()])

    class _BoomCtx(_FakeCtx):
        async def elicit(self, message, schema):
            raise RuntimeError("no prompt surface")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(confirm_kill_untracked=True, ctx=_BoomCtx())

    message = str(excinfo.value)
    assert "pid 4242" in message
    assert "could not confirm stopping the server holding the port" in message
    # The escape hatch names the real port, not a placeholder.
    assert "comfy stop --port 8188" in message
    assert len(state["runs"]) == 1
    assert len(state["launches"]) == 1


# --- the consent posture, shared with the other gates -----------------------


def test_confirm_kill_untracked_does_not_bypass_a_prompt_the_client_can_show(clash):
    """The caller's say-so is not the user's consent to kill their process.

    The host's "always allow restart_comfyui" toggle answers whether this tool
    may be CALLED — a different question from whether an unrelated process may
    be killed — and the caller may itself be a prompt-injected agent.
    """
    state = clash([_dry_run()])
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="declined"):
        _restart(confirm_kill_untracked=True, ctx=ctx)

    assert len(ctx.elicitations) == 1  # asked anyway
    assert len(state["runs"]) == 1


def test_an_unknown_capability_is_asked_rather_than_assumed_incapable(clash):
    """A probe that raises is UNKNOWN — reading it as "no" would skip the human.

    That is the failure that matters: with `confirm_kill_untracked=True` a
    silently-demoted client would kill the process on the caller's say-so alone.
    """
    state = clash([_dry_run()])

    class _UnreadableSession(_FakeSession):
        def check_client_capability(self, capability):
            raise RuntimeError("capability probe exploded")

    ctx = _FakeCtx(action="decline")
    ctx.session = _UnreadableSession(True)

    with pytest.raises(server.ComfyCliError, match="declined"):
        _restart(confirm_kill_untracked=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert len(state["runs"]) == 1


@pytest.mark.parametrize("ctx", [None, _FakeCtx(supports_elicitation=False)])
def test_a_client_that_cannot_be_prompted_kills_nothing_by_default(clash, ctx):
    """The `False` default is why a bare call from such a client is harmless."""
    state = clash([_dry_run()])

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=ctx)

    message = str(excinfo.value)
    assert "confirm_kill_untracked=True" in message
    assert "pid 4242" in message
    assert "Ask the USER first" in message
    assert len(state["runs"]) == 1
    assert len(state["launches"]) == 1


def test_a_client_that_cannot_be_prompted_falls_back_to_the_explicit_flag(clash):
    """The documented escape hatch still works where no prompt can be shown."""
    state = clash([_dry_run(), _stopped()])

    assert _restart(
        confirm_kill_untracked=True, ctx=_FakeCtx(supports_elicitation=False)
    ) == {"pid": 99, "port": 8188}

    assert len(state["runs"]) == 2
    assert len(state["launches"]) == 2


# --- when the engine will not vouch for the listener: no prompt at all ------


@pytest.mark.parametrize(
    "reply",
    [
        # comfy-cli found a listener it could not identify as ComfyUI, and says
        # so with an error envelope rather than a payload.
        server.ComfyCliError(
            "Refusing to stop pid 900 on port 8188: it cannot be identified as "
            "ComfyUI (no HTTP answer).",
            code="unverified_process",
        ),
        # Nothing is listening at all by the time we look.
        server.ComfyCliError(
            "Nothing is listening on port 8188.", code="port_not_listening"
        ),
        # A comfy-cli that predates `comfy stop --port` rejects the option while
        # PARSING — no envelope, exit 2. This is the version probe: there is no
        # version to compare, the refusal is the answer.
        server.ComfyCliError(
            "comfy-cli returned no JSON (exit 2). stderr: No such option '--port'.",
            no_envelope=True,
            returncode=2,
        ),
        # Defensive payload checks — a shape this server will not read as a
        # verified dry run even though the call succeeded.
        _dry_run(verified=False),
        _dry_run(dry_run=False),
        _dry_run(pid="4242"),
        _dry_run(pid=True),
        "not a mapping",
    ],
)
def test_an_unvouched_listener_never_raises_a_prompt(clash, reply):
    """No identity, no prompt — and the old, honestly-hedged error comes back."""
    state = clash([reply])
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=ctx)

    message = str(excinfo.value)
    assert ctx.elicitations == []
    assert _PORT_TAKEN in message
    # The pre-existing wording, hedge intact: nothing here KNOWS what is on the
    # port, so the message must not pretend otherwise.
    assert "almost certainly started outside comfy-cli" in message
    assert len(state["runs"]) == 1
    assert len(state["launches"]) == 1


def test_a_probe_that_cannot_even_run_is_not_an_error_on_top_of_an_error(clash):
    """An OSError reading the child is still just "we could not tell"."""
    state = clash([OSError("no fd left")])

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=_FakeCtx())

    assert "almost certainly started outside comfy-cli" in str(excinfo.value)
    assert len(state["runs"]) == 1


# --- the paths that must never reach the kill at all ------------------------


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("COMFYUI_URL", "http://comfy.example:8188"),
        ("COMFYUI_HOST", "comfy.example"),
        # Set but MALFORMED: `_comfy_target` raises, and that reads as
        # "configured", never as "local" — an unreadable config must not unlock
        # a kill.
        ("COMFYUI_URL", "http://[::1"),
    ],
)
def test_a_remote_session_neither_probes_nor_prompts(clash, monkeypatch, var, value):
    """The lifecycle verbs are local-only, so whose port this is stops being obvious.

    `[]` as the reply list is the assertion: any `_run_comfy` call at all fails
    the test.
    """
    monkeypatch.setenv(var, value)
    state = clash([])
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=ctx)

    assert ctx.elicitations == []
    assert state["runs"] == []
    assert "almost certainly started outside comfy-cli" in str(excinfo.value)


def test_an_unreadable_port_is_never_guessed_at(clash):
    """`--port` present but unparseable is UNKNOWN, not "probably 8188".

    Both cases read as `None` out of `_requested_port`, and conflating them is
    how the user gets shown a process on a port they never asked about — and
    kills it by approving. Only "no `--port` at all" may fall back to the
    default. `[]` as the reply list is the assertion: no probe may run.
    """
    state = clash([])
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(["--port", "not-a-number"], ctx=ctx)

    assert state["runs"] == []
    assert ctx.elicitations == []
    assert "almost certainly started outside comfy-cli" in str(excinfo.value)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (None, 8188),  # comfy-cli's own default — the clash really was there
        ([], 8188),
        (["--cpu"], 8188),
        (["--port", "8300"], 8300),
        (["--port=8300"], 8300),
        (["--portable"], 8188),  # a flag that merely shares the prefix
        (["--port", "bad"], None),  # asked for a port we cannot read
        (["--port"], None),  # dangling flag
        (["--port=70000"], None),  # out of range
        (["--port", "8300", "--port", "bad"], None),  # the last one is the one
    ],
)
def test_the_kill_target_port_separates_unset_from_unreadable(extra_args, expected):
    """The unit behind the guard above."""
    assert server._kill_target_port(extra_args) == expected


def test_a_port_clash_after_a_real_stop_is_left_alone(monkeypatch):
    """Half the signature is not the signature.

    comfy-cli DID have a server to stop, so a port that is still busy is a
    different problem (a lingering process, a second ComfyUI) — and killing
    whatever is there on that reading would be exactly the overreach the gate is
    built to avoid.
    """
    runs: list = []
    monkeypatch.setattr(server, "stop_comfyui", lambda: {"stopped": True})
    monkeypatch.setattr(
        server,
        "_launch_comfyui_sync",
        lambda extra_args=None: (_ for _ in ()).throw(
            server.ComfyCliError(_PORT_TAKEN)
        ),
    )
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: runs.append(a))
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=ctx)

    assert str(excinfo.value) == _PORT_TAKEN  # untouched, no guidance appended
    assert runs == []
    assert ctx.elicitations == []


def test_a_launch_failure_that_is_not_a_port_clash_is_left_alone(monkeypatch):
    """The other half of the signature, from the other side."""
    runs: list = []

    def fake_stop():
        raise server.ComfyCliError(_NOTHING_TO_STOP)

    monkeypatch.setattr(server, "stop_comfyui", fake_stop)
    monkeypatch.setattr(
        server,
        "_launch_comfyui_sync",
        lambda extra_args=None: (_ for _ in ()).throw(
            server.ComfyCliError("CUDA device 0 is already in use")
        ),
    )
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: runs.append(a))
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError, match="CUDA device"):
        _restart(ctx=ctx)

    assert runs == []
    assert ctx.elicitations == []


# --- honesty after an approved kill -----------------------------------------


def test_a_failed_kill_does_not_go_on_to_relaunch(clash):
    """comfy-cli could not free the port, so there is nothing to launch into."""
    state = clash(
        [
            _dry_run(),
            server.ComfyCliError(
                "Stopped pid 4242, but port 8188 is still held by pid 77.",
                code="stop_failed",
            ),
        ]
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "could not stop it" in message
    assert "still held by pid 77" in message
    assert excinfo.value.code == "stop_failed"
    assert len(state["launches"]) == 1  # the retry never happened


def test_a_failed_relaunch_says_the_server_was_already_stopped(clash):
    """The kill is not undoable, so its error must not hide that it happened.

    Re-raising the launch error alone would leave the user believing the server
    they approved stopping is still up.
    """
    state = clash(
        [_dry_run(), _stopped()],
        relaunch=server.ComfyCliError("ComfyUI exited during startup."),
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        _restart(ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "ComfyUI exited during startup." in message
    assert "WAS stopped first" in message
    assert "pid 4242" in message
    assert len(state["launches"]) == 2


# --- the probe itself, at the argv level ------------------------------------


def test_the_dry_run_probe_spawns_the_documented_command(patched_run):
    """One `_run_comfy` passthrough, global flags first, and nothing else."""
    calls = patched_run(envelope(data=_dry_run()))

    listener = server._verified_untracked_listener(8188)

    assert calls[0]["cmd"] == [
        server.COMFY_BIN,
        "--json",
        "--where",
        "local",
        "stop",
        "--port",
        "8188",
        "--dry-run",
    ]
    assert calls[0]["timeout"] == server._STOP_PORT_TIMEOUT
    assert listener == server._UntrackedListener(
        pid=4242, port=8188, cmdline=" ".join(_CMDLINE)
    )


def test_the_probe_survives_a_payload_with_no_readable_command_line(patched_run):
    """A pid with no argv is still an identity worth confirming."""
    patched_run(envelope(data=_dry_run(cmdline=[])))

    listener = server._verified_untracked_listener(8188)

    assert listener is not None
    assert listener.pid == 4242
    assert listener.display_cmdline() == "<unreadable command line>"


def test_the_probe_drops_non_string_command_line_parts(patched_run):
    """comfy-cli's argv is strings; anything else is not rendered as one."""
    patched_run(envelope(data=_dry_run(cmdline=["python", 7, None, "main.py"])))

    listener = server._verified_untracked_listener(8188)

    assert listener.cmdline == "python main.py"


# --- the guidance text, on its own ------------------------------------------


def test_the_guidance_is_unchanged_when_nothing_was_identified():
    """The pre-existing message is the fallback, hedge and all."""
    guidance = server._untracked_server_guidance(["--cpu"])

    assert "almost certainly started outside comfy-cli" in guidance
    assert 'extra_args=["--cpu", "--port", "8189"]' in guidance


def test_the_guidance_names_the_process_once_one_is_identified():
    """With an identity in hand the message stops guessing — and stops hedging."""
    listener = server._UntrackedListener(pid=4242, port=8188, cmdline="python main.py")

    guidance = server._untracked_server_guidance(
        ["--cpu"], listener, "You declined to stop pid 4242, so it is still running."
    )

    assert "pid 4242" in guidance
    assert "python main.py" in guidance
    assert "almost certainly" not in guidance
    assert "comfy stop --port 8188" in guidance
    # The alternate-port suggestion survives: recycling is an offer, not the
    # only way out.
    assert 'extra_args=["--cpu", "--port", "8189"]' in guidance
