"""Tests for ``run_template`` — the thin passthrough to ``comfy run-template <name>``.

All the real work (gallery fetch, slot filling, the spend gate, the run itself)
lives in comfy-cli. These lock in what the wrapper owns:

1. The passthrough argv: global flags before the subcommand, the template name as
   the first positional, slot fills as ``--param=KEY=VALUE`` (JSON-encoded, so
   Python types round-trip), and ``--async`` only on ``wait=False``.
2. The SPEND CONSENT posture — ``--allow-spend`` is sent only when the USER
   granted consent for that call (the per-call elicitation prompt, or the
   explicit ``confirm_spend`` fallback on a client that cannot elicit), a
   free-by-default run is never prompted at all, and an agent's own
   ``confirm_spend=True`` is not a way around the prompt. The engine still owns
   the fail-closed interlock; these only prove what we forward, and that a
   decline spawns no child.
3. The result contract: ``comfy run-template`` emits an ``envelope/1`` (unlike the
   ``comfy generate`` path), so its ``data`` is unwrapped and an error envelope
   (e.g. the gate's ``spend_consent_required``) raises.
4. Which comfy-cli dialect each branch takes: ``wait=True`` STREAMS
   (``--json-stream``, live MCP progress notifications — a template run can take
   an hour), while the ``wait=False`` submit stays on the plain ``--json`` path
   since there is no stream to follow.

comfy-cli is mocked throughout: no real ComfyUI, and no real credit spend.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import _OK_STREAM, _RecordingCtx, envelope, stream_reader
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_mcp import server


def _run_template(*args, **kwargs):
    """Drive the async ``run_template`` tool from a sync test.

    Matches the ``asyncio.run`` convention the other async tools' tests use; the
    tool went async in order to raise the per-call spend-confirmation
    elicitation before unlocking ``--allow-spend``.
    """
    return asyncio.run(server.run_template(*args, **kwargs))


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake MCPServer ``Context`` that answers the elicitation with ``action``.

    Deliberately a local copy of ``test_partner_generate``'s fake rather than a
    shared one: the two tools' consent paths are asserted independently, so a
    change to one tool's prompt must not be able to silently retune the other's
    tests.

    It also records progress notifications: ``run_template`` hands the SAME
    context to the spend elicitation and to the streaming run, exactly as the real
    ``Context`` serves both, so a consent-path fake that could not report progress
    would be pretending the two are separate objects.
    """

    def __init__(self, action="accept", approve=True, supports_elicitation=True):
        self.session = _FakeSession(supports_elicitation)
        self._action = action
        self._approve = approve
        self.elicitations: list[str] = []
        self.progress: list[dict] = []

    async def report_progress(self, progress, total=None, message=None):
        self.progress.append({"progress": progress, "total": total, "message": message})

    async def elicit(self, message, schema):
        self.elicitations.append(message)
        if self._action == "accept":
            return AcceptedElicitation(data=schema(approve=self._approve))
        if self._action == "decline":
            return DeclinedElicitation()
        return CancelledElicitation()


@pytest.fixture
def patched_streamed_run(patched_stream, monkeypatch):
    """``setup(stdout=...) -> calls`` — the same recorder over the STREAMING path.

    ``wait=True`` reads NDJSON incrementally off the ``Popen`` pipes rather than
    draining them with one bounded ``communicate``, so ``patched_run``'s
    canned-result stub cannot serve it. This wraps the shared ``patched_stream``
    fixture and spies on :func:`server._run_comfy_streaming` (delegating to the
    real one, so the whole streaming path still runs) to record the same
    ``{"cmd", "timeout"}`` shape the plain-path tests assert — ``cmd`` from the
    spawned fake process, ``timeout`` being the parent's backstop budget, which
    the streaming path bounds the read loop with rather than handing to
    ``communicate``.

    ``stdout`` mirrors ``patched_run``'s contract — a dict (an :func:`envelope`,
    NDJSON-encoded as the stream's final line for you) or a raw NDJSON string;
    it defaults to conftest's shared ``_OK_STREAM``.

    ``calls`` stays empty when no child is spawned, so the input-guard tests read
    identically on both paths.
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
                # Popen runs inside the call, so the argv is only readable once
                # it has returned — including on the raising (error-envelope)
                # path, where the cmd is exactly what the test wants to see.
                if procs:
                    record["cmd"] = procs[-1].cmd

        monkeypatch.setattr(server, "_run_comfy_streaming", spy)
        return calls

    return setup


# --- passthrough argv -------------------------------------------------------


def test_run_template_argv_is_a_local_json_stream_passthrough(patched_streamed_run):
    """`comfy --json-stream --where local run-template <name> --param=k=v`."""
    calls = patched_streamed_run()

    result = _run_template("image_flux2", params={"prompt": "a red fox"})

    assert result == {"outputs": ["/x.png"]}  # envelope data unwrapped
    cmd = calls[0]["cmd"]
    assert cmd[0] == server.COMFY_BIN
    # wait=True rides the streaming dialect — the same one `generate_image`
    # already uses for this verb, and the reason a long run reports progress.
    assert cmd[1:4] == ["--json-stream", "--where", "local"]  # global flags first
    assert cmd[4:] == [
        "run-template",
        "image_flux2",
        '--param=prompt="a red fox"',
        "--timeout=120",  # engine per-event bound, left at comfy-cli's default
    ]
    assert "--async" not in cmd  # wait=True runs to completion
    # wait=True: the caller's budget plus the parent's backstop slack.
    assert calls[0]["timeout"] == 600.0 + server._RUN_TEMPLATE_TIMEOUT_GRACE


def test_run_template_streams_progress_notifications(patched_streamed_run):
    """A long template run reports per-node progress instead of blocking silently."""
    patched_streamed_run()
    ctx = _RecordingCtx()

    result = _run_template("image_flux2", params={"prompt": "a red fox"}, ctx=ctx)

    assert result == {"outputs": ["/x.png"]}
    # The canned stream carries queued + executing + progress + two executed
    # events; each is forwarded to the client as a progress notification.
    assert len(ctx.calls) >= 4
    assert ctx.calls[0]["message"] == "queued"
    assert ctx.calls[-1]["progress"] == 2.0  # both nodes done
    assert ctx.calls[-1]["total"] == 2.0


def test_run_template_streams_without_a_ctx(patched_streamed_run):
    """No client context (no elicitation-capable host) still runs and returns."""
    calls = patched_streamed_run()

    assert _run_template("image_flux2") == {"outputs": ["/x.png"]}
    assert calls[0]["cmd"][1] == "--json-stream"


def test_run_template_survives_a_failing_progress_notification(patched_streamed_run):
    """A notification that fails mid-run must not abort the run it describes.

    Streaming is what newly puts a CREDIT-SPENDING path under
    ``ctx.report_progress``: an exception escaping the send would reach
    ``_run_comfy_streaming``'s cleanup, which kills the comfy-cli tree — losing a
    run whose credits are already spent, with no ``prompt_id`` to recover the
    outputs. A disconnected client is telemetry loss, not a reason to cancel.
    """

    class _BrokenCtx:
        def __init__(self):
            self.attempts = 0

        async def report_progress(self, progress, total=None, message=None):
            self.attempts += 1
            raise RuntimeError("client disconnected")

    calls = patched_streamed_run()
    ctx = _BrokenCtx()

    result = _run_template("api_seedance", params={"prompt": "a cat"}, ctx=ctx)

    assert result == {"outputs": ["/x.png"]}  # the run's own result still lands
    assert calls[0]["cmd"][1] == "--json-stream"
    # Every event was attempted and every failure swallowed — one broken send does
    # not stop the pump either.
    assert ctx.attempts > 1


def test_run_template_marshals_param_types_as_json(patched_streamed_run):
    """Each value is JSON-encoded so its Python type round-trips through the slot."""
    calls = patched_streamed_run()

    _run_template(
        "image_flux2",
        params={
            "prompt": "a cat",  # str -> quoted JSON string (stays a string)
            "6.text": "hi",  # slot address key
            "steps": 30,  # int
            "guidance": 3.5,  # float
            "raw": True,  # bool -> true/false, never Python's "True"
            "tiled": False,
            "sizes": [512, 768],  # list -> JSON array
            "opts": {"a": 1},  # dict -> JSON object
            "seed": None,  # dropped entirely, not sent as "None"
        },
    )

    assert calls[0]["cmd"][4:] == [
        "run-template",
        "image_flux2",
        '--param=prompt="a cat"',
        '--param=6.text="hi"',
        "--param=steps=30",
        "--param=guidance=3.5",
        "--param=raw=true",
        "--param=tiled=false",
        "--param=sizes=[512, 768]",
        '--param=opts={"a": 1}',
        "--timeout=120",
    ]


def test_run_template_param_value_with_equals_and_dash_stays_intact(
    patched_streamed_run,
):
    """The single `--param=KEY=VALUE` token keeps `=`/leading-dash values whole.

    comfy-cli splits only on the FIRST `=`, so a value carrying its own `=` (or a
    dash) survives as one argv element rather than being split or read as a flag.
    """
    calls = patched_streamed_run()

    _run_template("t", params={"expr": "a=b-c"})

    assert calls[0]["cmd"][4:] == [
        "run-template",
        "t",
        '--param=expr="a=b-c"',
        "--timeout=120",
    ]


def test_run_template_omits_params_when_none(patched_streamed_run):
    """No params -> a bare `run-template <name>`."""
    calls = patched_streamed_run()

    _run_template("image_flux2")

    assert calls[0]["cmd"][4:] == ["run-template", "image_flux2", "--timeout=120"]


# --- spend consent ----------------------------------------------------------


def test_run_template_withholds_consent_by_default(patched_streamed_run):
    """The default sends NO `--allow-spend`: a paid template is left to fail closed."""
    calls = patched_streamed_run()

    _run_template("api_seedance", params={"prompt": "a cat"})

    assert "--allow-spend" not in calls[0]["cmd"]


def test_run_template_forwards_consent_when_confirmed(patched_streamed_run):
    """`confirm_spend=True` forwards `--allow-spend`, comfy-cli's paid-node consent.

    Consent resolves BEFORE the spawn, so the flag has to survive onto the
    streaming command line — the interplay between the gate and the dialect is
    pure sequencing, and this pins it.
    """
    calls = patched_streamed_run()

    _run_template("api_seedance", params={"prompt": "a cat"}, confirm_spend=True)

    assert calls[0]["cmd"][1] == "--json-stream"  # granted consent, still streamed
    assert calls[0]["cmd"][4:] == [
        "run-template",
        "api_seedance",
        '--param=prompt="a cat"',
        "--timeout=120",
        "--allow-spend",
    ]


def test_run_template_free_run_is_never_prompted(patched_streamed_run):
    """`confirm_spend=False` can't spend, so the user is not asked anything.

    Most gallery templates are free OSS graphs. Prompting on every call would
    train the user to click through the one prompt that actually matters, and
    there is nothing to consent to: with no `--allow-spend` the engine's gate
    fails closed on a paid template.
    """
    patched_streamed_run()
    ctx = _FakeCtx()

    _run_template("image_flux2", params={"prompt": "a cat"}, ctx=ctx)

    assert ctx.elicitations == []


def test_run_template_asks_the_user_before_unlocking_spend(patched_streamed_run):
    """`confirm_spend=True` is a REQUEST to spend; the human grants it per call."""
    calls = patched_streamed_run()
    ctx = _FakeCtx(action="accept", approve=True)

    _run_template(
        "api_seedance", params={"prompt": "a cat"}, confirm_spend=True, ctx=ctx
    )

    assert len(ctx.elicitations) == 1
    assert "api_seedance" in ctx.elicitations[0]
    assert "SPENDS Comfy credits" in ctx.elicitations[0]
    assert "--allow-spend" in calls[0]["cmd"]
    # One context, both jobs: it asked, then reported the run it unlocked.
    assert ctx.progress


@pytest.mark.parametrize(
    ("action", "approve"),
    [
        ("decline", False),  # said no
        ("cancel", False),  # dismissed the prompt
        ("accept", False),  # accepted without actually answering yes
    ],
)
def test_run_template_declined_spend_spawns_no_child(
    patched_streamed_run, action, approve
):
    """A refusal is enforced HERE — comfy-cli is never started, nothing is spent."""
    calls = patched_streamed_run()
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _run_template("api_seedance", confirm_spend=True, ctx=ctx)

    assert calls == []


def test_run_template_confirm_spend_is_not_a_way_around_the_prompt(
    patched_streamed_run,
):
    """An agent setting `confirm_spend=True` itself does not authorize the spend.

    The hole this closes: a host's blanket "always allow this tool" toggle lets
    an agent set the argument for itself, which would otherwise be standing
    authority over the user's credits — the same reason `partner_generate`
    prompts even when it is passed.
    """
    calls = patched_streamed_run()
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _run_template("api_seedance", confirm_spend=True, ctx=ctx)

    assert calls == []


def test_run_template_falls_back_to_confirm_spend_when_client_cannot_elicit(
    patched_streamed_run,
):
    """On a client with no elicitation, `confirm_spend` is the documented fallback."""
    calls = patched_streamed_run()
    ctx = _FakeCtx(supports_elicitation=False)

    _run_template("api_seedance", confirm_spend=True, ctx=ctx)

    assert ctx.elicitations == []
    assert "--allow-spend" in calls[0]["cmd"]


def test_run_template_does_not_consult_the_generate_auto_confirm(
    patched_streamed_run, monkeypatch
):
    """comfy-cli scopes `spend.auto_confirm` to `comfy generate`.

    `run-template` never reads it, so treating it as consent here would forward
    nothing (the engine cannot consent to itself for this verb) and fail closed
    having asked nobody. The template path must not call it at all.
    """
    monkeypatch.setattr(
        server,
        "_engine_auto_confirms",
        lambda: pytest.fail("run_template must not read the generate auto-confirm"),
    )
    ctx = _FakeCtx()

    # A free run: no consent machinery should be reached at all.
    patched_streamed_run()

    _run_template("image_flux2", confirm_spend=True, ctx=ctx)

    assert len(ctx.elicitations) == 1


def test_run_template_spend_refusal_raises_with_code(patched_streamed_run):
    """The engine's fail-closed refusal (error envelope) raises — no false success."""
    patched_streamed_run(
        envelope(
            ok=False,
            error={
                "code": "spend_consent_required",
                "message": "template uses paid nodes; re-run with --allow-spend",
            },
        )
    )

    with pytest.raises(server.ComfyCliError, match="spend_consent_required"):
        _run_template("api_seedance", params={"prompt": "a cat"})


# --- wait / async -----------------------------------------------------------


def test_run_template_wait_false_submits_async(patched_run, monkeypatch):
    """`wait=False` appends `--async` and uses the short fire-and-return timeout.

    It also stays on the plain `--json` dialect: there is nothing to stream from a
    submit that returns as soon as the job is queued.
    """
    calls = patched_run(envelope(data={"prompt_id": "p9"}))
    monkeypatch.setattr(
        server,
        "_run_comfy_streaming",
        lambda *a, **k: pytest.fail("wait=False must not stream"),
    )

    result = _run_template("image_flux2", params={"prompt": "x"}, wait=False)

    assert result == {"prompt_id": "p9"}
    assert calls[0]["cmd"][1:4] == ["--json", "--where", "local"]
    assert calls[0]["cmd"][4:] == [
        "run-template",
        "image_flux2",
        '--param=prompt="x"',
        # The 60s submit budget is handed to the engine too, so its 120s server
        # probe can no longer outlive the parent and eat the whole call.
        "--timeout=60",
        "--async",
    ]
    assert (
        calls[0]["timeout"]
        == server._RUN_TEMPLATE_ASYNC_TIMEOUT + server._RUN_TEMPLATE_TIMEOUT_GRACE
    )


class _BlockingProc:
    """A child fake that emits ``first_lines`` and then never yields an envelope.

    Local rather than in ``conftest`` because this is the one case where the call
    genuinely differs (see AGENTS.md): the shared ``patched_stream`` fake drains a
    canned stream to EOF instantly and reports itself already exited, so it can
    never hold the read past a deadline — which is precisely the state this test
    is about.

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


def test_run_template_timeout_raises_the_streaming_shape(monkeypatch):
    """An expired `wait=True` run raises the STREAMING timeout error, with progress.

    The dialect change moves this failure off `_run_comfy`'s
    `subprocess.TimeoutExpired` message onto the streaming one, which carries how
    far the run got and points at the async path — pin that shape so it cannot
    silently regress to a bare parent kill.

    The parent's grace is zeroed here purely so the deadline is reachable inside a
    test: the real one is 30s of deliberate slack past the caller's budget.
    """
    queued = json.dumps({"schema": "event/1", "type": "queued", "nodes": [{"id": "1"}]})
    procs: list[_BlockingProc] = []

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProc(cmd, [queued + "\n"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server, "_RUN_TEMPLATE_TIMEOUT_GRACE", 0.0)

    with pytest.raises(server.ComfyCliError) as exc:
        _run_template("image_flux2", timeout_seconds=0.25)

    message = str(exc.value)
    assert "comfy-cli timed out after 0.25s" in message
    assert "Progress so far" in message  # the tracker snapshot, not a bare kill
    assert "wait=False" in message  # points at the path a long run should take
    assert procs[0].killed  # the child is not left running behind the deadline


# --- input guards -----------------------------------------------------------


@pytest.mark.parametrize("name", ["", "--help", "-x"])
def test_run_template_rejects_option_like_name(patched_streamed_run, name):
    """An empty/dash-leading name would be parsed by comfy-cli as an option."""
    calls = patched_streamed_run()

    with pytest.raises(server.ComfyCliError, match="invalid template name"):
        _run_template(name)

    assert calls == []  # never reached comfy-cli


@pytest.mark.parametrize("key", ["--prompt", "-p", ""])
def test_run_template_rejects_option_like_param_key(patched_streamed_run, key):
    """A dash-leading/empty slot key is a caller mistake, refused before shell-out."""
    calls = patched_streamed_run()

    with pytest.raises(server.ComfyCliError, match="invalid param key"):
        _run_template("t", params={key: "v"})

    assert calls == []


@pytest.mark.parametrize("key", ["a=b", "a b", "a\tb"])
def test_run_template_rejects_param_keys_with_equals_or_whitespace(
    patched_streamed_run, key
):
    """A `=`/whitespace in the KEY would mis-split against comfy-cli's first-`=` rule."""
    calls = patched_streamed_run()

    with pytest.raises(server.ComfyCliError, match="cannot contain"):
        _run_template("t", params={key: "v"})

    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "t\0mpl"}, "invalid template name"),
        ({"name": "t", "params": {"pro\0mpt": "a"}}, "param key"),
        ({"name": "t", "params": {"prompt": "a\0b"}}, "value for param"),
        # Nested in a container: json.dumps would escape it to a literal
        # `\u0000` and quietly fill it into the graph rather than crash.
        ({"name": "t", "params": {"sizes": ["a\0b"]}}, "value for param"),
        ({"name": "t", "params": {"opts": {"k": ["deep\0"]}}}, "value for param"),
        ({"name": "t", "params": {"opts": {"k\0": "v"}}}, "value for param"),
    ],
)
def test_run_template_rejects_embedded_nul(patched_streamed_run, kwargs, match):
    """A NUL is legal in a JSON string but `subprocess` raises a bare ValueError.

    Catch the shape here — before any child is spawned — so it surfaces as a
    ComfyCliError rather than an internal error.
    """
    calls = patched_streamed_run()

    with pytest.raises(server.ComfyCliError, match=match):
        _run_template(**kwargs)

    assert calls == []


def test_run_template_clamps_timeout(patched_streamed_run):
    """An absurd timeout is clamped, so no `comfy run-template` child runs forever."""
    calls = patched_streamed_run()

    _run_template("t", timeout_seconds=float("inf"))

    assert (
        calls[0]["timeout"]
        == server._MAX_RUN_TEMPLATE_TIMEOUT + server._RUN_TEMPLATE_TIMEOUT_GRACE
    )


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (30.0, "--timeout=30"),  # below comfy-cli's default -> tightened
        (0.5, "--timeout=1"),  # int() floors to 0; the engine needs >= 1
        (120.0, "--timeout=120"),  # exactly the default
        (600.0, "--timeout=120"),  # above it -> never RAISED past the default
    ],
)
def test_run_template_hands_engine_a_deadline_within_the_caller_budget(
    patched_streamed_run, budget, expected
):
    """The engine's per-event bound is lowered to a short budget, never raised.

    comfy-cli's own default is 120s and also bounds its "is ComfyUI up?" probe,
    so without this a `timeout_seconds` under 120 was spent entirely inside that
    probe and the child was SIGKILLed by the parent with no diagnostic. Raising
    it past the default would instead blunt stall detection on long runs.
    """
    calls = patched_streamed_run()

    _run_template("t", timeout_seconds=budget)

    assert expected in calls[0]["cmd"]
    # The parent stays a backstop only: strictly later than the engine's bound.
    assert calls[0]["timeout"] == budget + server._RUN_TEMPLATE_TIMEOUT_GRACE


def test_run_template_engine_timeout_is_an_int(patched_streamed_run):
    """comfy-cli types `--timeout` as an int, so a float would be a parse error."""
    calls = patched_streamed_run()

    _run_template("t", timeout_seconds=45.7)

    flag = next(a for a in calls[0]["cmd"] if a.startswith("--timeout="))
    assert flag == "--timeout=45"
    assert "." not in flag


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -5.0])
def test_run_template_rejects_non_positive_or_nan_timeout(patched_streamed_run, bad):
    """NaN/non-positive timeouts are refused (NaN survives min/max comparisons)."""
    calls = patched_streamed_run()

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        _run_template("t", timeout_seconds=bad)

    assert calls == []


# --- discoverability --------------------------------------------------------


def test_instructions_mention_run_template(patched_run):
    """The handshake should teach the one-command template run, spend posture too."""
    instructions = server.mcp.instructions

    assert "run_template" in instructions
