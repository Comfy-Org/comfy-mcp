"""Tests for ``run_workflow``'s SPEND CONSENT posture.

``run_workflow`` is the last of this server's run verbs to get a spend gate. Most
workflows are free graphs that execute entirely on the user's own machine, but
one that embeds partner-API nodes — a graph written by ``emit_partner_workflow``,
or an ``API``-tagged gallery template fetched with ``fetch_template`` — bills the
user's Comfy credits when it runs. comfy-cli owns the interlock
(``comfy run --allow-spend`` / a fail-closed ``spend_consent_required``); these
lock in the thin passthrough this server adds on top:

1. ``--allow-spend`` is forwarded only when the USER granted consent for that
   call — the per-call elicitation prompt, or the explicit ``confirm_spend``
   fallback on a client that cannot elicit — on BOTH the streaming ``wait=True``
   path and the ``wait=False`` submit.
2. **The prompt is driven off the ENGINE'S verdict, not off ``confirm_spend``.**
   The run always goes out unflagged first, so a free graph is never prompted no
   matter what the caller passed, and the user is asked only once comfy-cli has
   refused the graph with ``spend_consent_required`` — a refusal that submits
   nothing. ``confirm_spend=False`` is still never prompted and never flagged.
3. An agent's own ``confirm_spend=True`` is not a way around the prompt: a
   decline means no run carrying ``--allow-spend`` is ever spawned.
4. Consent is resolved ONCE per call, outside the credential retry loop, so a
   transient ``partner_node_requires_credential`` retry re-runs the child and
   never re-asks the human — and a second ``spend_consent_required`` on the
   consented run is the engine's answer rather than a cue to ask again.
5. The engine's fail-closed refusal raises rather than reading as a success, is
   NOT retried (``spend_consent_required`` is deterministic), and names the
   offending ``partner_nodes`` from the error envelope.
6. The flag is forwarded only to a comfy-cli that HAS it. It has shipped since
   comfy-cli 1.14.0, which is this server's own enforced floor, so a genuinely
   gateless engine arrives only through the fail-open version check — and that
   is the one case where the prompt has to be raised up front, because no
   refusal will ever arrive to raise it. The CAPABILITY PROBE can also report
   False when it simply could not answer (a `comfy run --help` that errors or
   will not spawn), so a capable engine reaches that branch too. When it does,
   the engine's own refusal overrides the probe and the consent already
   collected is forwarded — never discarded, never asked for twice.

comfy-cli is mocked throughout: no real ComfyUI, and no real credit spend.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from conftest import _OK_STREAM, envelope
from mcp import types
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_mcp import server

# The negotiated protocol revision on which `ctx.elicit()` has no back-channel, so
# the spend prompt must round-trip through the tool result (see
# `server._INPUT_REQUIRED_PROTOCOL`).
_INPUT_REQUIRED = "2026-07-28"


def _run_workflow(*args, **kwargs):
    """Drive the async ``run_workflow`` tool from a sync test."""
    return asyncio.run(server.run_workflow(*args, **kwargs))


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake MCPServer ``Context`` that answers the elicitation with ``action``.

    Deliberately a local copy of ``test_run_template``'s fake rather than a
    shared one, for the same reason that one is a copy of
    ``test_partner_generate``'s: each tool's consent path is asserted
    independently, so a change to one tool's prompt must not be able to silently
    retune another's tests.

    It records progress notifications too — ``run_workflow`` hands the SAME
    context to the spend elicitation and to the streaming run, exactly as the
    real ``Context`` serves both.
    """

    def __init__(
        self,
        action="accept",
        approve=True,
        supports_elicitation=True,
        protocol_version=None,
        input_responses=None,
        request_state=None,
        while_deciding=None,
    ):
        self.session = _FakeSession(supports_elicitation)
        self._action = action
        self._approve = approve
        self.elicitations: list[str] = []
        self.progress: list[dict] = []
        # The 2026-07-28 input-required transport: a negotiated version and the
        # client's retry answers, read by `_elicit_approval` instead of `elicit`.
        # Left `None` (the default) this fake is the handshake transport unchanged.
        self.protocol_version = protocol_version
        self.input_responses = input_responses
        self.request_state = request_state
        # Called INSIDE `elicit`, i.e. while the human is deciding. The prompt is
        # the only place in the call where real wall-clock time passes, so it is
        # the only honest place to model something else writing the graph file.
        self._while_deciding = while_deciding

    async def report_progress(self, progress, total=None, message=None):
        self.progress.append({"progress": progress, "total": total, "message": message})

    async def elicit(self, message, schema):
        self.elicitations.append(message)
        if self._while_deciding is not None:
            self._while_deciding()
        if self._action == "accept":
            return AcceptedElicitation(data=schema(approve=self._approve))
        if self._action == "decline":
            return DeclinedElicitation()
        return CancelledElicitation()


#: The real probe, captured before `allow_spend_capable` shadows the module
#: attribute — the probe's OWN tests must exercise this, not the stand-in.
_REAL_ALLOW_SPEND_PROBE = server._comfy_run_takes_allow_spend


@pytest.fixture(autouse=True)
def allow_spend_capable(monkeypatch):
    """Default every test here to a comfy-cli whose ``comfy run`` HAS the flag.

    The assertions below are about this server's consent POLICY, not about which
    engine happens to be installed, so they run against the engine
    ``--allow-spend`` was designed for. That default is a convenience, not a
    blind spot: :func:`server._comfy_run_takes_allow_spend` has its own tests
    further down, and the incapable engine — no published release, only a build
    that slips past the fail-open version check — is covered by the tests that
    override this fixture.
    """
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: True)


@pytest.fixture
def streamed_run(patched_stream, monkeypatch):
    """``setup(stdout=...) -> calls`` — record the argv of the STREAMING path.

    ``run_workflow(wait=True)`` reads NDJSON incrementally off the spawned pipes,
    so ``patched_run``'s canned-result stub cannot serve it. Mirrors
    ``test_run_template``'s ``patched_streamed_run``: spy on
    :func:`server._run_comfy_streaming` (delegating to the real one, so the whole
    streaming path still runs) and record ``cmd`` from the spawned fake process.

    ``calls`` stays empty when no child is spawned, which is exactly what a
    declined-consent test asserts.
    """

    def setup(stdout=None) -> list[dict]:
        if stdout is None:
            stdout = _OK_STREAM
        elif isinstance(stdout, dict):
            stdout = json.dumps(stdout) + "\n"
        procs = patched_stream(stdout)
        calls: list[dict] = []
        real_streaming = server._run_comfy_streaming

        async def spy(*args, **kwargs):
            record: dict = {"timeout": kwargs.get("timeout")}
            calls.append(record)
            try:
                return await real_streaming(*args, **kwargs)
            finally:
                # The spawn happens inside the call, so argv is only readable
                # once it has returned — including on the raising path.
                if procs:
                    record["cmd"] = procs[-1].cmd

        monkeypatch.setattr(server, "_run_comfy_streaming", spy)
        return calls

    return setup


@pytest.fixture
def refusing_engine(monkeypatch):
    """``setup(result=…) -> calls`` — a comfy-cli that fails closed, then runs.

    The spend interlock lives in comfy-cli, and this server now reaches its own
    prompt only THROUGH that interlock: the first run carries no
    ``--allow-spend`` and is refused with ``spend_consent_required``, and the
    prompt is raised on that refusal. So every test about the prompt has to
    model both halves of the exchange, which is what this does — any invocation
    without the flag refuses, any invocation with it succeeds.

    :func:`patched_stream` cans ONE stdout for every spawn and so cannot express
    a sequence. Patching the two run entry points directly is the same technique
    :func:`test_run_workflow_elicits_once_across_credential_retries` already
    used for the credential retry, and it covers the ``wait=False`` submit
    (``_run_comfy`` in a thread) and the ``wait=True`` stream
    (``_run_comfy_streaming``) with one fake, since the policy under test is the
    same on both.

    ``calls`` is the argv of every invocation in order, so an assertion can say
    WHICH run carried the flag rather than only that one did.
    """

    def setup(*, result=None):
        calls: list[list[str]] = []
        payload = {"outputs": ["/x.png"]} if result is None else result

        def _answer(args: tuple):
            calls.append(list(args))
            if "--allow-spend" not in args:
                raise server.ComfyCliError(
                    "comfy run --workflow failed [spend_consent_required]: "
                    "workflow uses partner-API (paid) nodes; re-run with "
                    "--allow-spend to consent to spending Comfy credits "
                    "(partner_nodes: VeoNode)",
                    code="spend_consent_required",
                )
            return payload

        async def fake_stream(*args, **kwargs):
            return _answer(args)

        def fake_run(*args, **kwargs):
            return _answer(args)

        monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
        monkeypatch.setattr(server, "_run_comfy", fake_run)
        return calls

    return setup


# --- the default: free, silent, no flag -------------------------------------


def test_run_workflow_withholds_consent_by_default(streamed_run):
    """The default sends NO `--allow-spend`: a paid graph is left to fail closed."""
    calls = streamed_run()

    assert _run_workflow("wf.json") == {"outputs": ["/x.png"]}

    assert "--allow-spend" not in calls[0]["cmd"]


def test_run_workflow_free_run_is_never_prompted(streamed_run):
    """`confirm_spend=False` can't spend, so the user is not asked anything.

    Nearly every `run_workflow` is a free local graph. Prompting on each one
    would train the user to click through the one prompt that actually matters,
    and there is nothing to consent to: with no `--allow-spend` the engine's gate
    fails closed on a paid graph.
    """
    calls = streamed_run()
    ctx = _FakeCtx()

    _run_workflow("wf.json", ctx=ctx)

    assert ctx.elicitations == []
    assert "--allow-spend" not in calls[0]["cmd"]


def test_run_workflow_free_submit_is_never_prompted(patched_run):
    """Same on the `wait=False` submit — no prompt, no flag."""
    calls = patched_run(envelope(data={"prompt_id": "p1"}))
    ctx = _FakeCtx()

    assert _run_workflow("wf.json", wait=False, ctx=ctx) == {"prompt_id": "p1"}

    assert ctx.elicitations == []
    assert "--allow-spend" not in calls[0]["cmd"]


def test_run_workflow_does_not_consult_the_generate_auto_confirm(
    refusing_engine, monkeypatch
):
    """comfy-cli scopes `spend.auto_confirm` to `comfy generate`.

    `comfy run` never reads it, so treating it as consent here would forward
    nothing (the engine cannot consent to itself for this verb) and fail closed
    having asked nobody — the same reason `run_template` must not read it.
    """
    monkeypatch.setattr(
        server,
        "_engine_auto_confirms",
        lambda: pytest.fail("run_workflow must not read the generate auto-confirm"),
    )
    refusing_engine()
    ctx = _FakeCtx()

    _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert len(ctx.elicitations) == 1


# --- a free graph is never prompted, whatever the caller passed --------------


def test_run_workflow_does_not_prompt_a_free_graph_with_confirm_spend_true(
    streamed_run,
):
    """THE regression this policy exists to prevent (comfy-mcp#300).

    `confirm_spend` is the CALLER's permission, not a statement about the graph,
    and an agent reading this tool's description has every reason to pass it
    prophylactically — the reporter's harness does exactly that, which put a
    credit prompt in front of every run of a workflow that cannot spend
    anything. Only comfy-cli can tell a paid graph from a free one, so the run
    goes out unflagged first: a partner-free graph is byte-identical without
    `--allow-spend`, so it simply runs, and the user is never asked.
    """
    calls = streamed_run()
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow("wf.json", confirm_spend=True, ctx=ctx) == {
        "outputs": ["/x.png"]
    }

    assert ctx.elicitations == []  # nothing could be spent, so nothing was asked
    assert len(calls) == 1
    assert "--allow-spend" not in calls[0]["cmd"]


def test_run_workflow_probes_the_engine_once_when_confirm_spend_is_set(
    streamed_run, monkeypatch
):
    """`confirm_spend=True` costs one `--help` probe, and only that.

    The probe decides which SHAPE of gate this engine has — an interlock to be
    refused by, or nothing, in which case the prompt has to be raised up front
    because no refusal will ever arrive. It is skipped entirely when
    `confirm_spend` is False (see the test below), and a positive answer latches
    for the process, so the cost is one extra spawn per server, not per run.
    """
    probes = {"n": 0}

    def counting_probe():
        probes["n"] += 1
        return True

    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", counting_probe)
    streamed_run()

    _run_workflow("wf.json", confirm_spend=True)

    assert probes["n"] == 1


# --- granting consent, on the engine's refusal ------------------------------


def test_run_workflow_asks_the_user_once_the_engine_refuses(refusing_engine):
    """The engine's fail-closed refusal is what raises the prompt.

    `spend_consent_required` is comfy-cli saying this graph embeds partner-API
    nodes AND that it submitted nothing — so at the moment the user is asked,
    the question is real and the answer still costs nothing if it is no.
    """
    calls = refusing_engine()
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow("wf.json", confirm_spend=True, ctx=ctx) == {
        "outputs": ["/x.png"]
    }

    assert len(ctx.elicitations) == 1
    assert "wf.json" in ctx.elicitations[0]
    assert "SPENDS Comfy credits" in ctx.elicitations[0]
    # The prompt can now state the paid-ness as fact rather than hedging.
    assert "embeds partner-API nodes" in ctx.elicitations[0]
    # Two runs: the unflagged one the engine refused, then the one it consented
    # to. In that order, and the flag only on the second.
    assert [list(c) for c in calls] == [
        ["run", "--workflow", "wf.json", "--wait"],
        ["run", "--workflow", "wf.json", "--wait", "--allow-spend"],
    ]


def test_run_workflow_forwards_consent_on_the_submit_path(refusing_engine):
    """The `wait=False` submit carries the flag too — both branches, one gate."""
    calls = refusing_engine(result={"prompt_id": "p1"})
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow("wf.json", confirm_spend=True, wait=False, ctx=ctx) == {
        "prompt_id": "p1"
    }

    assert len(ctx.elicitations) == 1
    assert calls == [
        ["run", "--workflow", "wf.json"],
        ["run", "--workflow", "wf.json", "--allow-spend"],
    ]


def test_run_workflow_falls_back_to_confirm_spend_when_client_cannot_elicit(
    refusing_engine,
):
    """On a client with no elicitation, `confirm_spend` is the documented fallback."""
    calls = refusing_engine()
    ctx = _FakeCtx(supports_elicitation=False)

    _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert ctx.elicitations == []
    assert "--allow-spend" in calls[-1]


def test_run_workflow_forwards_consent_without_any_ctx(refusing_engine):
    """No context at all is the same fallback — a direct call cannot be prompted."""
    calls = refusing_engine()

    _run_workflow("wf.json", confirm_spend=True)

    assert "--allow-spend" in calls[-1]


# --- refusing consent -------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "approve"),
    [
        ("decline", False),  # said no
        ("cancel", False),  # dismissed the prompt
        ("accept", False),  # accepted without actually answering yes
    ],
)
def test_run_workflow_declined_spend_never_reaches_a_paid_run(
    refusing_engine, action, approve
):
    """A refusal is enforced HERE: the only run that happened could not spend.

    The assertion moved with the policy. It used to be "comfy-cli is never
    started", which is no longer the shape of the exchange — the engine has to
    be asked before anyone knows there is something to consent to. What must
    still hold is that no run carrying `--allow-spend` is ever spawned, i.e.
    exactly one unflagged run, already refused by the engine with nothing
    submitted.
    """
    calls = refusing_engine()
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert calls == [["run", "--workflow", "wf.json", "--wait"]]


def test_run_workflow_declined_spend_never_reaches_a_paid_submit(refusing_engine):
    """Same on `wait=False`, which is a different spawn with the same policy."""
    calls = refusing_engine(result={"prompt_id": "p1"})
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _run_workflow("wf.json", confirm_spend=True, wait=False, ctx=ctx)

    assert calls == [["run", "--workflow", "wf.json"]]


def test_run_workflow_confirm_spend_is_not_a_way_around_the_prompt(refusing_engine):
    """An agent setting `confirm_spend=True` itself does not authorize the spend.

    The hole this closes: a host's blanket "always allow this tool" toggle lets
    an agent set the argument for itself, which would otherwise be standing
    authority over the user's credits — the same reason `partner_generate` and
    `run_template` prompt even when it is passed.
    """
    calls = refusing_engine()
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert "--allow-spend" not in calls[-1]
    msg = str(excinfo.value)
    assert "Nothing was spent" in msg
    # And it no longer offers `confirm_spend=False` as a way to run this graph:
    # the engine has already said it fails closed, so that advice would send the
    # caller round the same refusal.
    assert "confirm_spend=False" in msg
    assert "will fail closed the same way" in msg


def test_run_workflow_does_not_re_ask_when_the_engine_refuses_the_consented_run(
    monkeypatch,
):
    """A second `spend_consent_required` is the engine's answer, not a retry cue.

    If the engine refuses a run it was explicitly told to allow, re-prompting
    would ask the user to re-approve something their approval did not move. One
    prompt per call, then the error reaches the caller.
    """
    calls: list[tuple] = []

    async def always_refuses(*args, **kwargs):
        calls.append(args)
        raise server.ComfyCliError("refused anyway", code="spend_consent_required")

    monkeypatch.setattr(server, "_run_comfy_streaming", always_refuses)
    ctx = _FakeCtx(action="accept", approve=True)

    with pytest.raises(server.ComfyCliError, match="refused anyway"):
        _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert len(ctx.elicitations) == 1  # asked once, not once per refusal
    assert len(calls) == 2  # the unflagged run, then the consented one
    assert "--allow-spend" in calls[1]


# --- the engine's fail-closed refusal ---------------------------------------


def _spend_refusal() -> dict:
    """The `envelope/1` comfy-cli emits when the gate withholds a paid run."""
    return envelope(
        ok=False,
        error={
            "code": "spend_consent_required",
            "message": (
                "workflow uses partner-API (paid) nodes; re-run with "
                "--allow-spend to consent to spending Comfy credits"
            ),
            "details": {"partner_nodes": ["VeoNode", "KlingNode"]},
        },
    )


def test_run_workflow_spend_refusal_raises_and_names_the_paid_nodes(streamed_run):
    """The engine's fail-closed refusal raises — and says WHICH nodes cost money.

    `_SURFACED_DETAIL_KEYS` already lifts `partner_nodes` out of any error
    envelope, so the fail-closed error is actionable for free: the caller learns
    the graph is paid and which node made it so, without a second call.
    """
    streamed_run(_spend_refusal())

    with pytest.raises(server.ComfyCliError) as excinfo:
        _run_workflow("wf.json")

    msg = str(excinfo.value)
    assert "spend_consent_required" in msg
    assert "partner_nodes" in msg
    assert "VeoNode" in msg
    assert "KlingNode" in msg
    assert excinfo.value.code == "spend_consent_required"


def test_run_workflow_spend_refusal_is_not_retried(monkeypatch):
    """`spend_consent_required` is deterministic — retrying only burns backoff.

    The credential retry exists for TRANSIENT auth failures. A withheld consent
    is a decision, not a hiccup: the same argv would be refused all three times,
    so the code stays out of `_RETRYABLE_CREDENTIAL_CODES` and the caller gets
    its answer on the first attempt.
    """
    assert "spend_consent_required" not in server._RETRYABLE_CREDENTIAL_CODES

    calls = {"n": 0}

    async def fake_stream(*args, **kwargs):
        calls["n"] += 1
        raise server.ComfyCliError(
            "comfy run --workflow wf.json failed [spend_consent_required]: "
            "partner_nodes: VeoNode",
            code="spend_consent_required",
        )

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    with pytest.raises(server.ComfyCliError, match="spend_consent_required"):
        _run_workflow("wf.json")

    assert calls["n"] == 1  # one invocation, no retries


def test_run_workflow_spend_refusal_on_the_submit_path(patched_run):
    """The same fail-closed refusal on `wait=False`, which is a different spawn.

    The two paths forward `spend_args` independently — `_run_comfy` in a thread
    for the submit, `_run_comfy_streaming` for the stream — so the streaming
    tests above do not cover this one. A `wait=False` caller gets the refusal
    with the same shape: raised rather than returned as a result, carrying the
    code, naming the paid nodes, and attempted exactly once.
    """
    calls = patched_run(_spend_refusal())

    with pytest.raises(server.ComfyCliError) as excinfo:
        _run_workflow("wf.json", wait=False)

    assert excinfo.value.code == "spend_consent_required"
    assert "VeoNode" in str(excinfo.value)
    assert len(calls) == 1  # deterministic, so the credential retry stays out


# --- the engine capability probe ---------------------------------------------


def test_run_workflow_still_runs_when_the_engine_has_no_gate(streamed_run, monkeypatch):
    """An APPROVED run must still run against a comfy-cli that predates the flag.

    This is also the ONLY path that still prompts up front, and it has to: an
    engine with no interlock never refuses, so a prompt waiting for a refusal
    would wait forever and the graph would spend having asked nobody.

    `comfy run` is a plain Click command with no `ignore_unknown_options`, so an
    unrecognized `--allow-spend` exits 2 with a usage error and no `envelope/1`
    — which would turn the approval the user just gave into an opaque "returned
    no JSON" failure. Dropping the flag runs the graph exactly as it ran before
    this argument existed: the human's answer to the prompt is what authorizes
    the spend, and an engine with no interlock has nothing to engage.
    """
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: False)
    calls = streamed_run()
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow("wf.json", confirm_spend=True, ctx=ctx) == {
        "outputs": ["/x.png"]
    }

    assert len(ctx.elicitations) == 1  # the human was still asked
    assert "--allow-spend" not in calls[0]["cmd"]  # nothing to forward it to


def test_a_mis_probed_engine_honours_the_consent_it_already_collected(
    refusing_engine, monkeypatch
):
    """A capable engine mis-read as gateless must not discard the approval it took.

    `_comfy_run_takes_allow_spend` answers False whenever it CANNOT answer — a
    `comfy run --help` that exits nonzero, or that cannot be spawned at all —
    and neither of those says anything about whether the flag exists. So a fully
    capable comfy-cli can be sent up the no-interlock branch by a `--help`
    hiccup, which is why that branch is not reachable only through the fail-open
    version check the way the module docstring above once claimed.

    Then the engine refuses, and the refusal settles it: an engine with no
    `--allow-spend` has no gate to engage, so it can never emit
    `spend_consent_required` — it would have run the graph and spent. A refusal
    in hand therefore PROVES the probe wrong, and the engine is the more
    authoritative of the two answers.

    The approval is forwarded, not re-raised and not re-asked. Re-raising spent
    the user's attention, took a yes, and then failed the very run that yes
    authorized with `spend_consent_required` — told them consent was required
    moments after they gave it. Re-asking would be the second prompt this whole
    design exists to avoid. Exactly one prompt, honoured.
    """
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: False)
    calls = refusing_engine()
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow("wf.json", confirm_spend=True, ctx=ctx) == {
        "outputs": ["/x.png"]
    }

    assert len(ctx.elicitations) == 1  # asked once, up front, and never again
    assert "--allow-spend" not in calls[0]  # the unflagged run that drew the refusal
    assert "--allow-spend" in calls[1]  # the consent the user had already given


def test_a_mis_probed_engine_still_fails_closed_on_a_declined_prompt(
    refusing_engine, monkeypatch
):
    """Honouring an up-front yes must not also honour an up-front no.

    The forwarding above is reached only past
    `_resolve_workflow_spend_consent`, which RAISES on a decline before any
    child is spawned. So a declined up-front prompt on the same mis-probed
    engine never reaches the engine at all, and no run can carry the flag.
    """
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: False)
    calls = refusing_engine()
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert "spend not confirmed" in str(excinfo.value)
    assert calls == []  # nothing spawned, so nothing could spend
    assert len(ctx.elicitations) == 1


def test_the_probe_is_skipped_when_no_consent_was_granted(streamed_run, monkeypatch):
    """A free run never pays for the extra `--help` spawn — the common case."""
    probes = {"n": 0}

    def counting_probe():
        probes["n"] += 1
        return True

    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", counting_probe)
    streamed_run()

    _run_workflow("wf.json")  # confirm_spend defaults to False

    assert probes["n"] == 0


def test_allow_spend_probe_reads_the_help_text(patched_run, monkeypatch):
    """The probe asks `comfy run --help`, which spends nothing to learn the answer."""
    monkeypatch.setattr(server, "_run_allow_spend_engine", None)
    calls = patched_run(
        "Usage: comfy run [OPTIONS]\n\nOptions:\n  --allow-spend  Consent to spend.\n"
    )

    assert _REAL_ALLOW_SPEND_PROBE() is True
    assert calls[0]["cmd"][4:] == ["run", "--help"]


def test_allow_spend_probe_is_false_when_the_help_lacks_the_flag(
    patched_run, monkeypatch
):
    """A comfy-cli predating 1.14.0: the verb exists, the flag does not."""
    monkeypatch.setattr(server, "_run_allow_spend_engine", None)
    patched_run("Usage: comfy run [OPTIONS]\n\nOptions:\n  --workflow TEXT\n")

    assert _REAL_ALLOW_SPEND_PROBE() is False


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"returncode": 2}, "the probe itself failed"),
        ({"raises": OSError("no binary")}, "the binary could not be spawned"),
    ],
)
def test_allow_spend_probe_is_false_when_it_cannot_answer(
    patched_run, monkeypatch, kwargs, why
):
    """An unusable engine reads as "no flag" — never as an exception out of consent.

    Reporting False costs an up-front prompt and a run without the engine
    interlock; raising instead would break a call the user approved because a
    `--help` spawn hiccuped.
    """
    monkeypatch.setattr(server, "_run_allow_spend_engine", None)
    patched_run("--allow-spend", **kwargs)

    assert _REAL_ALLOW_SPEND_PROBE() is False, why


def test_allow_spend_probe_latches_only_a_positive_answer(
    patched_run, monkeypatch, real_engine
):
    """A True is cached for THAT engine; a False is re-asked, so an upgrade lands."""
    monkeypatch.setattr(server, "_run_allow_spend_engine", None)
    calls = patched_run("Usage: comfy run\n  --workflow TEXT\n")

    assert _REAL_ALLOW_SPEND_PROBE() is False
    assert _REAL_ALLOW_SPEND_PROBE() is False
    assert len(calls) == 2  # re-probed, not wedged on the negative

    calls2 = patched_run("Usage: comfy run\n  --allow-spend\n")
    real_engine.bind()
    assert _REAL_ALLOW_SPEND_PROBE() is True
    assert _REAL_ALLOW_SPEND_PROBE() is True
    assert len(calls2) == 1  # latched, so the second call spawns nothing


# --- the capability latch is bound to the ENGINE, not to the process ---------
#
# [P1] on comfy-mcp#322 (@skishore23): a process-wide boolean let ONE positive
# probe silence the up-front prompt forever. Swap `COMFY_BIN` for an ungated
# build inside that lifetime and the paid graph goes out unflagged to an engine
# with no interlock — it runs, it spends, and nothing ever asks.


class _RealEngine:
    """A `COMFY_BIN` that is a real file, so it has a stat-able identity.

    `patched_run` points `shutil.which` at `/fake/comfy`, which cannot be
    stat'ed — and an unreadable identity deliberately re-probes rather than
    latching, so these tests need a file that exists. `patched_run` re-applies
    its own `which` stub on every call, hence `bind()` rather than a one-shot
    fixture: call it after each `patched_run(...)` to point the lookup back.
    """

    def __init__(self, path, monkeypatch):
        self._path = path
        self._monkeypatch = monkeypatch
        self._path.write_text("#!/bin/sh\nexit 0\n")
        self.bind()

    def bind(self):
        self._monkeypatch.setattr(server.shutil, "which", lambda _: str(self._path))

    def swap(self):
        # Unlink + rewrite, which is what an uninstall/install does to a console
        # script: a NEW inode, so the identity moves even at the same path.
        self._path.unlink()
        self._path.write_text("#!/bin/sh\nexit 1\n")


@pytest.fixture
def real_engine(tmp_path, monkeypatch):
    return _RealEngine(tmp_path / "comfy", monkeypatch)


def test_a_rolled_back_engine_is_reprobed_rather_than_trusted(
    patched_run, monkeypatch, real_engine
):
    """[P1] The latch is keyed on the executable, so a rollback re-asks the engine."""
    monkeypatch.setattr(server, "_run_allow_spend_engine", None)
    patched_run("Usage: comfy run\n  --allow-spend\n")
    real_engine.bind()
    assert _REAL_ALLOW_SPEND_PROBE() is True

    real_engine.swap()  # the gated build is replaced by an ungated one
    calls = patched_run("Usage: comfy run\n  --workflow TEXT\n")
    real_engine.bind()

    assert _REAL_ALLOW_SPEND_PROBE() is False, "a stale latch answered for a new engine"
    assert len(calls) == 1, "the new engine was never asked"


def test_an_unreadable_engine_identity_reprobes_instead_of_refusing(
    patched_run, monkeypatch
):
    """An unstat-able `COMFY_BIN` costs the spawn, not a false up-front prompt.

    A macOS TCC denial resolves on PATH and refuses to be stat'ed. Refusing on
    that would re-raise comfy-mcp#300 for every `confirm_spend=True` caller: a
    free graph would be prompted because the CACHE KEY could not be read, even
    though the interlock itself answers fine.
    """
    monkeypatch.setattr(server, "_run_allow_spend_engine", None)
    calls = patched_run("Usage: comfy run\n  --allow-spend\n")  # which -> /fake/comfy

    assert _REAL_ALLOW_SPEND_PROBE() is True
    assert _REAL_ALLOW_SPEND_PROBE() is True
    assert len(calls) == 2, "an unknown identity must not latch"


@pytest.mark.parametrize("wait", [True, False])
def test_a_rolled_back_engine_asks_before_the_paid_submit(
    monkeypatch, patched_run, patched_stream, real_engine, wait
):
    """[P1] Both wait modes: no child carries the graph until a human has said yes.

    The latch is primed against a gated engine, THEN the engine becomes ungated
    — the exact transition that used to submit unflagged, with no refusal to
    catch and no prompt ever raised.
    """
    monkeypatch.setattr(server, "_run_allow_spend_engine", None)
    # Drop the autouse fixture's blanket "capable": this test is ABOUT the probe.
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", _REAL_ALLOW_SPEND_PROBE)

    patched_run("Usage: comfy run\n  --allow-spend\n")
    real_engine.bind()
    assert _REAL_ALLOW_SPEND_PROBE() is True  # latched against the gated build

    real_engine.swap()  # ...and now it is not that build any more

    spawned: list[tuple] = []
    probe_text = "Usage: comfy run\n  --workflow TEXT\n"

    def record(*args, **kwargs):
        spawned.append(args)
        return {"outputs": ["/x.png"]}

    async def record_async(*args, **kwargs):
        spawned.append(args)
        return {"outputs": ["/x.png"]}

    def fake_raw(*args, **kwargs):
        # The capability probe, and ONLY the capability probe, reaches here.
        assert args[:2] == ("run", "--help"), args
        return ("", probe_text, "", 0, None)

    monkeypatch.setattr(server, "_run_comfy_raw", fake_raw)
    monkeypatch.setattr(server, "_run_comfy", record)
    monkeypatch.setattr(server, "_run_comfy_streaming", record_async)

    ctx = _FakeCtx(action="decline")
    with pytest.raises(server.ComfyCliError) as excinfo:
        _run_workflow("wf.json", confirm_spend=True, wait=wait, ctx=ctx)

    assert "spend not confirmed" in str(excinfo.value)
    assert spawned == [], "a paid graph was submitted before anyone was asked"
    assert len(ctx.elicitations) == 1


# --- consent resolves once, outside the retry loop ---------------------------


def test_run_workflow_elicits_once_across_credential_retries(monkeypatch):
    """The human is asked ONCE per call, however many times the child is retried.

    Consent is resolved outside the credential retry loop, never inside
    `_attempt`. Resolving it per attempt would re-prompt a user who already
    answered — up to three prompts for one tool call — and each retry would be a
    fresh chance to answer differently, which is not a thing consent should
    have. The loop now runs twice per call (once unflagged, once consented), so
    this pins the count across BOTH passes.
    """

    async def _fast(_seconds):
        return None

    monkeypatch.setattr(server.asyncio, "sleep", _fast)

    calls: list[tuple] = []

    async def fake_stream(*args, **kwargs):
        calls.append(args)
        if "--allow-spend" not in args:
            raise server.ComfyCliError("paid graph", code="spend_consent_required")
        # Consented: now be transiently broken twice before succeeding.
        if len(calls) < 4:
            raise server.ComfyCliError(
                "needs a credential", code="partner_node_requires_credential"
            )
        return {"outputs": ["/x.png"]}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    ctx = _FakeCtx(action="accept", approve=True)

    result = _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert result == {"outputs": ["/x.png"]}
    # 1 unflagged probe run + 1 consented + 2 credential retries.
    assert len(calls) == 4
    assert len(ctx.elicitations) == 1  # asked once, not once per attempt
    # Every attempt after the prompt carried the consent the user granted.
    assert all("--allow-spend" in argv for argv in calls[1:])


# --- the prompt's own hygiene ------------------------------------------------


def test_run_workflow_prompt_neutralizes_markdown_in_the_path(refusing_engine):
    """A caller-supplied path cannot inject markup into the prompt it appears in.

    The path rides inside a markdown code span; a backtick in it would close the
    span on a rendering client and let the rest write its own text — including a
    reassuring "this is free" over the SPENDS warning the user is answering.
    """
    refusing_engine()
    hostile = "wf`.json`\n**this is free**"
    ctx = _FakeCtx(action="accept", approve=True)

    _run_workflow(hostile, confirm_spend=True, ctx=ctx)

    prompt = ctx.elicitations[0]
    # Exactly the two delimiters this message opens and closes the span with —
    # the path's own backticks are gone, so it cannot escape the span.
    assert prompt.count("`") == 2
    assert "\n" not in prompt
    # The `**…**` survives but is inert inside the span, and the warning the
    # user is answering is still the one this message states.
    assert "SPENDS Comfy credits" in prompt


def test_run_workflow_prompt_keeps_the_filename_of_a_long_path(refusing_engine):
    """A path too long for the prompt falls back to its BASENAME, not a prefix.

    Truncating the tail of a deep path drops the filename — the only part that
    identifies which graph is about to spend the user's money. The drop is
    MARKED: an unmarked basename reads as the whole path, so `/tmp/x.json` and
    `~/audited/x.json` would look identical to the user answering the prompt.
    """
    refusing_engine()
    long_path = "/Users/someone/" + "nested/" * 20 + "the-expensive-one.json"
    ctx = _FakeCtx(action="accept", approve=True)

    _run_workflow(long_path, confirm_spend=True, ctx=ctx)

    prompt = ctx.elicitations[0]
    assert "…/the-expensive-one.json" in prompt
    assert long_path not in prompt  # the directory chain was dropped, not the name


def test_run_workflow_prompt_marks_a_padded_path_as_elided(refusing_engine):
    """Padding a path past the cap cannot silently hide where the graph came from.

    Two graphs with the same filename in different directories render the same
    once the directory is dropped, so a caller could pad `/tmp/wf.json` with
    redundant segments to keep `/tmp` off the prompt. The marker does not
    recover the directory — it tells the user one existed.
    """
    refusing_engine()
    padded = "/tmp/" + "./" * 45 + "wf.json"
    ctx = _FakeCtx(action="accept", approve=True)

    _run_workflow(padded, confirm_spend=True, ctx=ctx)

    assert "…/wf.json" in ctx.elicitations[0]


@pytest.mark.parametrize("empty", ["", "   ", "\t\n"])
def test_run_workflow_rejects_an_empty_path_without_prompting(streamed_run, empty):
    """An empty path cannot spend, so it must not cost the user a prompt.

    Without this guard the call reaches the gate and raises a credit-spend
    prompt for `<unnamed workflow>` before comfy-cli rejects the path anyway —
    so a caller could mint approval prompts for runs that were never going to
    happen, wearing down the attention the one real prompt depends on.
    """
    calls = streamed_run()
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError, match="workflow_path is empty"):
        _run_workflow(empty, confirm_spend=True, ctx=ctx)

    assert ctx.elicitations == []
    assert calls == []


def test_run_workflow_rejects_a_bad_path_without_prompting(streamed_run):
    """Input guards run BEFORE the gate: a malformed call never reaches the user."""
    calls = streamed_run()
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError):
        _run_workflow("--oops", confirm_spend=True, ctx=ctx)

    assert ctx.elicitations == []
    assert calls == []


# --- the approval is about a GRAPH, not about a path ------------------------
#
# `comfy run` re-reads `workflow_path` when the consented run spawns, and the
# human's decision sits between that read and the one the refusal came from.
# That window is as long as a person takes, so an approval that travels on the
# path alone can be spent on bytes nobody was shown — a parallel
# `emit_partner_workflow` writing the same `out_path` does it without anybody
# acting in bad faith. The approval is pinned to the file's digest instead, on
# both transports: re-read in-frame for `ctx.elicit()`'s back-channel, and
# carried in the prompt text (which keys the `request_state`) for the retry
# transport, where no local variable survives the round trip.


_PAID_GRAPH = '{"1": {"class_type": "VeoNode", "inputs": {"prompt": "a cat"}}}'
_OTHER_GRAPH = '{"1": {"class_type": "VeoNode", "inputs": {"prompt": "a dog"}}}'


def _graph(tmp_path, body=_PAID_GRAPH):
    path = tmp_path / "wf.json"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_a_graph_swapped_while_the_prompt_was_open_is_never_run(
    refusing_engine, tmp_path
):
    """A yes for one graph is not a yes for whatever replaced it.

    The engine refuses, the user is asked, and while they are deciding the file
    is rewritten. The consented re-run would read the NEW bytes — so the run is
    refused instead, and nothing is spent on a graph nobody saw.
    """
    path = _graph(tmp_path)
    calls = refusing_engine()
    ctx = _FakeCtx(
        action="accept",
        approve=True,
        while_deciding=lambda: open(path, "w", encoding="utf-8").write(_OTHER_GRAPH),
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        _run_workflow(path, confirm_spend=True, ctx=ctx)

    assert "CHANGED while the spend prompt was open" in str(excinfo.value)
    assert "Nothing was spent" in str(excinfo.value)
    # Exactly the unflagged run the engine refused. The flagged one — the only
    # one that could have cost anything — never happened.
    assert len(calls) == 1
    assert "--allow-spend" not in calls[0]
    assert len(ctx.elicitations) == 1


def test_a_graph_that_held_still_runs_and_its_prompt_names_the_digest(
    refusing_engine, tmp_path
):
    """The pin is not a brake on the ordinary case, and it is visible in the prompt.

    The digest is quoted as bare hex rather than in a code span on purpose: the
    prompt opens exactly two backticks, around the caller-supplied path, and
    `test_run_workflow_prompt_neutralizes_markdown_in_the_path` asserts that.
    """
    path = _graph(tmp_path)
    calls = refusing_engine()
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow(path, confirm_spend=True, ctx=ctx) == {"outputs": ["/x.png"]}

    digest = hashlib.sha256(_PAID_GRAPH.encode()).hexdigest()
    assert digest[:12] in ctx.elicitations[0]
    assert ctx.elicitations[0].count("`") == 2
    assert "--allow-spend" in calls[1]


def test_a_graph_this_process_cannot_read_still_runs(refusing_engine):
    """An unreadable path digests to `None` and pins on that, quoting nothing.

    Whether a path resolves is comfy-cli's question — it answers it, and this
    server must not start refusing runs that work today because it could not
    open the file itself (a relative path resolved by the child, say).
    """
    calls = refusing_engine()
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow("wf.json", confirm_spend=True, ctx=ctx) == {
        "outputs": ["/x.png"]
    }

    assert "sha256" not in ctx.elicitations[0]
    assert "--allow-spend" in calls[1]


def test_a_swapped_graph_is_asked_about_afresh_on_the_retry_transport(
    refusing_engine, tmp_path
):
    """2026-07-28: the answer comes back on a RETRY, so the pin rides the prompt.

    Nothing from the asking round survives into the answering one — the tool is
    re-entered from the top — so there is no variable left to compare. The
    digest is in the prompt text, and `_consent_state` keys the sealed state on
    that text, so a graph that changed under the prompt renders a different
    question: the nonce claim fails and the human is asked again rather than the
    first yes being spent on the new bytes.
    """
    path = _graph(tmp_path)
    calls = refusing_engine()

    first = _FakeCtx(protocol_version=_INPUT_REQUIRED)
    pending = _run_workflow(path, confirm_spend=True, ctx=first)
    assert isinstance(pending, types.InputRequiredResult)
    [key] = pending.input_requests

    # The writer lands between the rounds, which is where the human's wait is.
    open(path, "w", encoding="utf-8").write(_OTHER_GRAPH)
    retry = _FakeCtx(
        protocol_version=_INPUT_REQUIRED,
        input_responses={
            key: types.ElicitResult(action="accept", content={"approve": True})
        },
        request_state=pending.request_state,
    )
    replay = _run_workflow(path, confirm_spend=True, ctx=retry)

    # Asked afresh, and the new question names the new graph.
    assert isinstance(replay, types.InputRequiredResult)
    [(_, request)] = replay.input_requests.items()
    assert (
        hashlib.sha256(_OTHER_GRAPH.encode()).hexdigest()[:12] in request.params.message
    )
    # Two unflagged runs, one per round, both refused. Neither carried consent.
    assert all("--allow-spend" not in call for call in calls)


# --- the prompt mode must survive the input-required retry -------------------
#
# [P2] on comfy-mcp#322 (@skishore23): the asking round and the answering round
# are two separate executions of the tool, and the branch that picks the prompt
# WORDING is a `comfy run --help` probe that can fail transiently. A probe that
# failed on the asking call and succeeded on the answering one renders the gate's
# other wording; `_consent_state` is keyed on the prompt text, so the nonce claim
# fails and a human who already answered is asked a second time.


def _spend_answer(key, pending, *, action="accept", approve=True):
    """The retry context that carries one answered consent prompt back in."""
    return _FakeCtx(
        protocol_version=_INPUT_REQUIRED,
        input_responses={
            key: types.ElicitResult(action=action, content={"approve": approve})
        },
        request_state=pending.request_state,
    )


@pytest.fixture
def flaky_probe(monkeypatch):
    """`set(capable: bool)` — drive the capability probe between the two rounds.

    The up-front wording is reached only when the probe says False, the
    after-refusal wording only when it says True, so flipping this between the
    rounds is exactly the transition [P2] describes.
    """
    state = {"capable": False}
    monkeypatch.setattr(
        server, "_comfy_run_takes_allow_spend", lambda: state["capable"]
    )
    return lambda capable: state.__setitem__("capable", capable)


@pytest.mark.parametrize(
    ("first_capable", "then_capable", "why"),
    [
        (False, True, "a transient probe failure asked up front, then recovered"),
        (True, False, "a capable engine asked after the refusal, then probed badly"),
    ],
)
def test_an_approval_survives_a_capability_flip_between_the_rounds(
    refusing_engine, tmp_path, flaky_probe, first_capable, then_capable, why
):
    """[P2] The answer is consumed, the graph runs, and nobody is asked twice."""
    path = _graph(tmp_path)
    calls = refusing_engine()

    flaky_probe(first_capable)
    pending = _run_workflow(
        path, confirm_spend=True, ctx=_FakeCtx(protocol_version=_INPUT_REQUIRED)
    )
    assert isinstance(pending, types.InputRequiredResult), why
    [key] = pending.input_requests

    flaky_probe(then_capable)
    retry = _spend_answer(key, pending)
    assert _run_workflow(path, confirm_spend=True, ctx=retry) == {
        "outputs": ["/x.png"]
    }, why
    assert any("--allow-spend" in call for call in calls), why


@pytest.mark.parametrize(
    ("first_capable", "then_capable"), [(False, True), (True, False)]
)
def test_a_decline_survives_a_capability_flip_between_the_rounds(
    refusing_engine, tmp_path, flaky_probe, first_capable, then_capable
):
    """[P2] A "no" must be consumed too — re-asking a refusal is the same bug.

    Discarding a decline is not a safety hole the way discarding an approval is
    a usability one, but it is the SAME mechanism, and asking a person who just
    said no to say it again is how a click-through habit gets trained.
    """
    path = _graph(tmp_path)
    calls = refusing_engine()

    flaky_probe(first_capable)
    pending = _run_workflow(
        path, confirm_spend=True, ctx=_FakeCtx(protocol_version=_INPUT_REQUIRED)
    )
    [key] = pending.input_requests

    flaky_probe(then_capable)
    retry = _spend_answer(key, pending, action="decline", approve=False)
    with pytest.raises(server.ComfyCliError) as excinfo:
        _run_workflow(path, confirm_spend=True, ctx=retry)

    assert "spend not confirmed" in str(excinfo.value)
    assert all("--allow-spend" not in call for call in calls)


def test_a_changed_graph_still_defeats_the_claim_under_a_capability_flip(
    refusing_engine, tmp_path, flaky_probe
):
    """Accepting either wording must not loosen the pin that matters.

    Both candidate texts carry the same graph digest, so widening the claim to
    two messages widens it across the WORDING only. A graph swapped under the
    prompt still fails both, and the human is asked afresh.
    """
    path = _graph(tmp_path)
    refusing_engine()

    flaky_probe(False)
    pending = _run_workflow(
        path, confirm_spend=True, ctx=_FakeCtx(protocol_version=_INPUT_REQUIRED)
    )
    [key] = pending.input_requests

    open(path, "w", encoding="utf-8").write(_OTHER_GRAPH)
    flaky_probe(True)
    replay = _run_workflow(path, confirm_spend=True, ctx=_spend_answer(key, pending))

    assert isinstance(replay, types.InputRequiredResult)
    [(_, request)] = replay.input_requests.items()
    assert (
        hashlib.sha256(_OTHER_GRAPH.encode()).hexdigest()[:12] in request.params.message
    )


def test_one_answer_is_still_spent_only_once_across_a_capability_flip(
    refusing_engine, tmp_path, flaky_probe
):
    """The nonce is consumed by whichever wording claimed it, and only once.

    `any()` short-circuits over the candidate texts, so a replay of the same
    sealed state cannot be redeemed a second time by the OTHER wording.
    """
    path = _graph(tmp_path)
    refusing_engine()

    flaky_probe(False)
    pending = _run_workflow(
        path, confirm_spend=True, ctx=_FakeCtx(protocol_version=_INPUT_REQUIRED)
    )
    [key] = pending.input_requests

    flaky_probe(True)
    assert _run_workflow(path, confirm_spend=True, ctx=_spend_answer(key, pending)) == {
        "outputs": ["/x.png"]
    }

    # Same sealed state, replayed. Flip back so the OTHER wording is in force.
    flaky_probe(False)
    replay = _run_workflow(path, confirm_spend=True, ctx=_spend_answer(key, pending))
    assert isinstance(replay, types.InputRequiredResult), "a nonce was spent twice"
