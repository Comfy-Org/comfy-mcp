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
2. A free-by-default run (``confirm_spend=False``) is never prompted at all and
   never carries the flag.
3. An agent's own ``confirm_spend=True`` is not a way around the prompt: a
   decline raises before any child is spawned.
4. Consent is resolved ONCE per call, outside the credential retry loop, so a
   transient ``partner_node_requires_credential`` retry re-runs the child and
   never re-asks the human.
5. The engine's fail-closed refusal raises rather than reading as a success, is
   NOT retried (``spend_consent_required`` is deterministic), and names the
   offending ``partner_nodes`` from the error envelope.
6. The flag is forwarded only to a comfy-cli that HAS it. ``comfy run`` predates
   its gate, so the verb proves nothing and the flag is probed; against an
   engine without it an approved run still runs, rather than dying on the usage
   error Click raises for an unknown option.

comfy-cli is mocked throughout: no real ComfyUI, and no real credit spend.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import _OK_STREAM, envelope
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_mcp.server import _internal as server


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
    """A fake FastMCP ``Context`` that answers the elicitation with ``action``.

    Deliberately a local copy of ``test_run_template``'s fake rather than a
    shared one, for the same reason that one is a copy of
    ``test_partner_generate``'s: each tool's consent path is asserted
    independently, so a change to one tool's prompt must not be able to silently
    retune another's tests.

    It records progress notifications too — ``run_workflow`` hands the SAME
    context to the spend elicitation and to the streaming run, exactly as the
    real ``Context`` serves both.
    """

    def __init__(self, action="accept", approve=True, supports_elicitation=True):
        self.session = _FakeSession(supports_elicitation)
        self._action = action
        self._approve = approve
        self.elicitations: list[str] = []
        self.progress: list[dict] = []

    async def report_progress(self, progress, total=None, message=None):
        self.progress.append({"progress": progress, "total": total, "message": message})

    async def elicit(self, message, response_type):
        self.elicitations.append(message)
        if self._action == "accept":
            return AcceptedElicitation(data=response_type(approve=self._approve))
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
    further down, and the incapable engine — every comfy-cli release to date —
    is covered by the tests that override this fixture.
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
    streamed_run, monkeypatch
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
    streamed_run()
    ctx = _FakeCtx()

    _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert len(ctx.elicitations) == 1


# --- granting consent -------------------------------------------------------


def test_run_workflow_asks_the_user_before_unlocking_spend(streamed_run):
    """`confirm_spend=True` is a REQUEST to spend; the human grants it per call."""
    calls = streamed_run()
    ctx = _FakeCtx(action="accept", approve=True)

    _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert "wf.json" in ctx.elicitations[0]
    assert "SPENDS Comfy credits" in ctx.elicitations[0]
    assert calls[0]["cmd"][4:] == [
        "run",
        "--workflow",
        "wf.json",
        "--wait",
        "--allow-spend",
    ]
    # One context, both jobs: it asked, then reported the run it unlocked.
    assert ctx.progress


def test_run_workflow_forwards_consent_on_the_submit_path(patched_run):
    """The `wait=False` submit carries the flag too — both branches, one gate."""
    calls = patched_run(envelope(data={"prompt_id": "p1"}))
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow("wf.json", confirm_spend=True, wait=False, ctx=ctx) == {
        "prompt_id": "p1"
    }

    assert len(ctx.elicitations) == 1
    assert calls[0]["cmd"][1:4] == ["--json", "--where", "local"]
    assert calls[0]["cmd"][4:] == ["run", "--workflow", "wf.json", "--allow-spend"]


def test_run_workflow_falls_back_to_confirm_spend_when_client_cannot_elicit(
    streamed_run,
):
    """On a client with no elicitation, `confirm_spend` is the documented fallback."""
    calls = streamed_run()
    ctx = _FakeCtx(supports_elicitation=False)

    _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert ctx.elicitations == []
    assert "--allow-spend" in calls[0]["cmd"]


def test_run_workflow_forwards_consent_without_any_ctx(streamed_run):
    """No context at all is the same fallback — a direct call cannot be prompted."""
    calls = streamed_run()

    _run_workflow("wf.json", confirm_spend=True)

    assert "--allow-spend" in calls[0]["cmd"]


# --- refusing consent -------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "approve"),
    [
        ("decline", False),  # said no
        ("cancel", False),  # dismissed the prompt
        ("accept", False),  # accepted without actually answering yes
    ],
)
def test_run_workflow_declined_spend_spawns_no_child(streamed_run, action, approve):
    """A refusal is enforced HERE — comfy-cli is never started, nothing is spent."""
    calls = streamed_run()
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert calls == []


def test_run_workflow_declined_spend_spawns_no_child_on_the_submit_path(patched_run):
    """Same on `wait=False`: the refusal precedes the submit, not just the stream."""
    calls = patched_run(envelope(data={"prompt_id": "p1"}))
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _run_workflow("wf.json", confirm_spend=True, wait=False, ctx=ctx)

    assert calls == []


def test_run_workflow_confirm_spend_is_not_a_way_around_the_prompt(streamed_run):
    """An agent setting `confirm_spend=True` itself does not authorize the spend.

    The hole this closes: a host's blanket "always allow this tool" toggle lets
    an agent set the argument for itself, which would otherwise be standing
    authority over the user's credits — the same reason `partner_generate` and
    `run_template` prompt even when it is passed.
    """
    calls = streamed_run()
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert calls == []
    msg = str(excinfo.value)
    assert "Nothing was spent" in msg
    assert "confirm_spend=False" in msg  # names the free way to run it


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

    `comfy run` is a plain Click command with no `ignore_unknown_options`, so an
    unrecognized `--allow-spend` exits 2 with a usage error and no `envelope/1`
    — which would turn the approval the user just gave into an opaque "returned
    no JSON" failure on every comfy-cli released so far. Dropping the flag runs
    the graph exactly as it ran before this argument existed: the human's answer
    to the prompt is what authorizes the spend, and an engine with no interlock
    has nothing to engage.
    """
    monkeypatch.setattr(server, "_comfy_run_takes_allow_spend", lambda: False)
    calls = streamed_run()
    ctx = _FakeCtx(action="accept", approve=True)

    assert _run_workflow("wf.json", confirm_spend=True, ctx=ctx) == {
        "outputs": ["/x.png"]
    }

    assert len(ctx.elicitations) == 1  # the human was still asked
    assert "--allow-spend" not in calls[0]["cmd"]  # nothing to forward it to


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
    monkeypatch.setattr(server, "_run_allow_spend_probed", False)
    calls = patched_run(
        "Usage: comfy run [OPTIONS]\n\nOptions:\n  --allow-spend  Consent to spend.\n"
    )

    assert _REAL_ALLOW_SPEND_PROBE() is True
    assert calls[0]["cmd"][4:] == ["run", "--help"]


def test_allow_spend_probe_is_false_when_the_help_lacks_the_flag(
    patched_run, monkeypatch
):
    """Today's comfy-cli: the verb exists, the flag does not."""
    monkeypatch.setattr(server, "_run_allow_spend_probed", False)
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

    Reporting False costs a run without the engine interlock, which is the
    status quo on every current release; raising instead would break a call the
    user approved because a `--help` spawn hiccuped.
    """
    monkeypatch.setattr(server, "_run_allow_spend_probed", False)
    patched_run("--allow-spend", **kwargs)

    assert _REAL_ALLOW_SPEND_PROBE() is False, why


def test_allow_spend_probe_latches_only_a_positive_answer(patched_run, monkeypatch):
    """A True is cached for the process; a False is re-asked, so an upgrade lands."""
    monkeypatch.setattr(server, "_run_allow_spend_probed", False)
    calls = patched_run("Usage: comfy run\n  --workflow TEXT\n")

    assert _REAL_ALLOW_SPEND_PROBE() is False
    assert _REAL_ALLOW_SPEND_PROBE() is False
    assert len(calls) == 2  # re-probed, not wedged on the negative

    calls2 = patched_run("Usage: comfy run\n  --allow-spend\n")
    assert _REAL_ALLOW_SPEND_PROBE() is True
    assert _REAL_ALLOW_SPEND_PROBE() is True
    assert len(calls2) == 1  # latched, so the second call spawns nothing


# --- consent resolves once, outside the retry loop ---------------------------


def test_run_workflow_elicits_once_across_credential_retries(monkeypatch):
    """The human is asked ONCE per call, however many times the child is retried.

    Consent is resolved before the retry loop, not inside `_attempt`. Resolving
    it per attempt would re-prompt a user who already answered — up to three
    prompts for one tool call — and each retry would be a fresh chance to answer
    differently, which is not a thing consent should have.
    """

    async def _fast(_seconds):
        return None

    monkeypatch.setattr(server.asyncio, "sleep", _fast)

    calls: list[tuple] = []

    async def fake_stream(*args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise server.ComfyCliError(
                "needs a credential", code="partner_node_requires_credential"
            )
        return {"outputs": ["/x.png"]}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    ctx = _FakeCtx(action="accept", approve=True)

    result = _run_workflow("wf.json", confirm_spend=True, ctx=ctx)

    assert result == {"outputs": ["/x.png"]}
    assert len(calls) == 3  # 1 initial + 2 credential retries
    assert len(ctx.elicitations) == 1  # asked once, not once per attempt
    # And every attempt carried the consent the user granted on the first.
    assert all("--allow-spend" in argv for argv in calls)


# --- the prompt's own hygiene ------------------------------------------------


def test_run_workflow_prompt_neutralizes_markdown_in_the_path(streamed_run):
    """A caller-supplied path cannot inject markup into the prompt it appears in.

    The path rides inside a markdown code span; a backtick in it would close the
    span on a rendering client and let the rest write its own text — including a
    reassuring "this is free" over the SPENDS warning the user is answering.
    """
    streamed_run()
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


def test_run_workflow_prompt_keeps_the_filename_of_a_long_path(streamed_run):
    """A path too long for the prompt falls back to its BASENAME, not a prefix.

    Truncating the tail of a deep path drops the filename — the only part that
    identifies which graph is about to spend the user's money. The drop is
    MARKED: an unmarked basename reads as the whole path, so `/tmp/x.json` and
    `~/audited/x.json` would look identical to the user answering the prompt.
    """
    streamed_run()
    long_path = "/Users/someone/" + "nested/" * 20 + "the-expensive-one.json"
    ctx = _FakeCtx(action="accept", approve=True)

    _run_workflow(long_path, confirm_spend=True, ctx=ctx)

    prompt = ctx.elicitations[0]
    assert "…/the-expensive-one.json" in prompt
    assert long_path not in prompt  # the directory chain was dropped, not the name


def test_run_workflow_prompt_marks_a_padded_path_as_elided(streamed_run):
    """Padding a path past the cap cannot silently hide where the graph came from.

    Two graphs with the same filename in different directories render the same
    once the directory is dropped, so a caller could pad `/tmp/wf.json` with
    redundant segments to keep `/tmp` off the prompt. The marker does not
    recover the directory — it tells the user one existed.
    """
    streamed_run()
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
