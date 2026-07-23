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

comfy-cli is mocked throughout: no real ComfyUI, and no real credit spend.
"""

from __future__ import annotations

import asyncio
import json
import subprocess

import pytest
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_local_mcp import server


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
    """A fake FastMCP ``Context`` that answers the elicitation with ``action``.

    Deliberately a local copy of ``test_partner_generate``'s fake rather than a
    shared one: the two tools' consent paths are asserted independently, so a
    change to one tool's prompt must not be able to silently retune the other's
    tests.
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


def _envelope(*, ok: bool = True, data=None, error=None) -> str:
    body: dict = {"schema": "envelope/1", "type": "envelope", "ok": ok}
    if error is not None:
        body["error"] = error
    else:
        body["data"] = data if data is not None else {}
    return json.dumps(body)


@pytest.fixture
def patched_run(monkeypatch):
    """``setup(stdout=..., returncode=...) -> calls`` mocking ``comfy`` via subprocess.

    Defaults to a successful ``envelope/1`` so the happy path returns real data;
    pass ``stdout`` to shape the envelope (e.g. an error) per test.
    """

    def setup(stdout: str | None = None, returncode: int = 0) -> list[dict]:
        calls: list[dict] = []
        payload = (
            _envelope(data={"prompt_id": "p1", "outputs": ["/x.png"]})
            if stdout is None
            else stdout
        )

        def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
            calls.append({"cmd": cmd, "env": env, "timeout": timeout})
            return subprocess.CompletedProcess(
                cmd, returncode, stdout=payload, stderr=""
            )

        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(server.subprocess, "run", fake)
        return calls

    return setup


# --- passthrough argv -------------------------------------------------------


def test_run_template_argv_is_a_local_json_passthrough(patched_run):
    """`comfy --json --where local run-template <name> --param=k=v` (flags first)."""
    calls = patched_run()

    result = _run_template("image_flux2", params={"prompt": "a red fox"})

    assert result == {
        "prompt_id": "p1",
        "outputs": ["/x.png"],
    }  # envelope data unwrapped
    cmd = calls[0]["cmd"]
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == [
        "run-template",
        "image_flux2",
        '--param=prompt="a red fox"',
        "--timeout=120",  # engine per-event bound, left at comfy-cli's default
    ]
    # wait=True: the caller's budget plus the parent's backstop slack.
    assert calls[0]["timeout"] == 600.0 + server._RUN_TEMPLATE_TIMEOUT_GRACE


def test_run_template_marshals_param_types_as_json(patched_run):
    """Each value is JSON-encoded so its Python type round-trips through the slot."""
    calls = patched_run()

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


def test_run_template_param_value_with_equals_and_dash_stays_intact(patched_run):
    """The single `--param=KEY=VALUE` token keeps `=`/leading-dash values whole.

    comfy-cli splits only on the FIRST `=`, so a value carrying its own `=` (or a
    dash) survives as one argv element rather than being split or read as a flag.
    """
    calls = patched_run()

    _run_template("t", params={"expr": "a=b-c"})

    assert calls[0]["cmd"][4:] == [
        "run-template",
        "t",
        '--param=expr="a=b-c"',
        "--timeout=120",
    ]


def test_run_template_omits_params_when_none(patched_run):
    """No params -> a bare `run-template <name>`."""
    calls = patched_run()

    _run_template("image_flux2")

    assert calls[0]["cmd"][4:] == ["run-template", "image_flux2", "--timeout=120"]


# --- spend consent ----------------------------------------------------------


def test_run_template_withholds_consent_by_default(patched_run):
    """The default sends NO `--allow-spend`: a paid template is left to fail closed."""
    calls = patched_run()

    _run_template("api_seedance", params={"prompt": "a cat"})

    assert "--allow-spend" not in calls[0]["cmd"]


def test_run_template_forwards_consent_when_confirmed(patched_run):
    """`confirm_spend=True` forwards `--allow-spend`, comfy-cli's paid-node consent."""
    calls = patched_run()

    _run_template("api_seedance", params={"prompt": "a cat"}, confirm_spend=True)

    assert calls[0]["cmd"][4:] == [
        "run-template",
        "api_seedance",
        '--param=prompt="a cat"',
        "--timeout=120",
        "--allow-spend",
    ]


def test_run_template_free_run_is_never_prompted(patched_run):
    """`confirm_spend=False` can't spend, so the user is not asked anything.

    Most gallery templates are free OSS graphs. Prompting on every call would
    train the user to click through the one prompt that actually matters, and
    there is nothing to consent to: with no `--allow-spend` the engine's gate
    fails closed on a paid template.
    """
    patched_run()
    ctx = _FakeCtx()

    _run_template("image_flux2", params={"prompt": "a cat"}, ctx=ctx)

    assert ctx.elicitations == []


def test_run_template_asks_the_user_before_unlocking_spend(patched_run):
    """`confirm_spend=True` is a REQUEST to spend; the human grants it per call."""
    calls = patched_run()
    ctx = _FakeCtx(action="accept", approve=True)

    _run_template(
        "api_seedance", params={"prompt": "a cat"}, confirm_spend=True, ctx=ctx
    )

    assert len(ctx.elicitations) == 1
    assert "api_seedance" in ctx.elicitations[0]
    assert "SPENDS Comfy credits" in ctx.elicitations[0]
    assert "--allow-spend" in calls[0]["cmd"]


@pytest.mark.parametrize(
    ("action", "approve"),
    [
        ("decline", False),  # said no
        ("cancel", False),  # dismissed the prompt
        ("accept", False),  # accepted without actually answering yes
    ],
)
def test_run_template_declined_spend_spawns_no_child(patched_run, action, approve):
    """A refusal is enforced HERE — comfy-cli is never started, nothing is spent."""
    calls = patched_run()
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _run_template("api_seedance", confirm_spend=True, ctx=ctx)

    assert calls == []


def test_run_template_confirm_spend_is_not_a_way_around_the_prompt(patched_run):
    """An agent setting `confirm_spend=True` itself does not authorize the spend.

    The hole this closes: a host's blanket "always allow this tool" toggle lets
    an agent set the argument for itself, which would otherwise be standing
    authority over the user's credits — the same reason `partner_generate`
    prompts even when it is passed.
    """
    calls = patched_run()
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _run_template("api_seedance", confirm_spend=True, ctx=ctx)

    assert calls == []


def test_run_template_falls_back_to_confirm_spend_when_client_cannot_elicit(
    patched_run,
):
    """On a client with no elicitation, `confirm_spend` is the documented fallback."""
    calls = patched_run()
    ctx = _FakeCtx(supports_elicitation=False)

    _run_template("api_seedance", confirm_spend=True, ctx=ctx)

    assert ctx.elicitations == []
    assert "--allow-spend" in calls[0]["cmd"]


def test_run_template_does_not_consult_the_generate_auto_confirm(monkeypatch):
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
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 0, stdout=_envelope(data={"prompt_id": "p1"}), stderr=""
        ),
    )

    _run_template("image_flux2", confirm_spend=True, ctx=ctx)

    assert len(ctx.elicitations) == 1


def test_run_template_spend_refusal_raises_with_code(patched_run):
    """The engine's fail-closed refusal (error envelope) raises — no false success."""
    patched_run(
        stdout=_envelope(
            ok=False,
            error={
                "code": "spend_consent_required",
                "message": "template uses paid nodes; re-run with --allow-spend",
            },
        ),
        returncode=1,
    )

    with pytest.raises(server.ComfyCliError, match="spend_consent_required"):
        _run_template("api_seedance", params={"prompt": "a cat"})


# --- wait / async -----------------------------------------------------------


def test_run_template_wait_false_submits_async(patched_run):
    """`wait=False` appends `--async` and uses the short fire-and-return timeout."""
    calls = patched_run(stdout=_envelope(data={"prompt_id": "p9"}))

    result = _run_template("image_flux2", params={"prompt": "x"}, wait=False)

    assert result == {"prompt_id": "p9"}
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


# --- input guards -----------------------------------------------------------


@pytest.mark.parametrize("name", ["", "--help", "-x"])
def test_run_template_rejects_option_like_name(patched_run, name):
    """An empty/dash-leading name would be parsed by comfy-cli as an option."""
    calls = patched_run()

    with pytest.raises(server.ComfyCliError, match="invalid template name"):
        _run_template(name)

    assert calls == []  # never reached comfy-cli


@pytest.mark.parametrize("key", ["--prompt", "-p", ""])
def test_run_template_rejects_option_like_param_key(patched_run, key):
    """A dash-leading/empty slot key is a caller mistake, refused before shell-out."""
    calls = patched_run()

    with pytest.raises(server.ComfyCliError, match="invalid param key"):
        _run_template("t", params={key: "v"})

    assert calls == []


@pytest.mark.parametrize("key", ["a=b", "a b", "a\tb"])
def test_run_template_rejects_param_keys_with_equals_or_whitespace(patched_run, key):
    """A `=`/whitespace in the KEY would mis-split against comfy-cli's first-`=` rule."""
    calls = patched_run()

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
def test_run_template_rejects_embedded_nul(patched_run, kwargs, match):
    """A NUL is legal in a JSON string but `subprocess` raises a bare ValueError.

    Catch the shape here — before any child is spawned — so it surfaces as a
    ComfyCliError rather than an internal error.
    """
    calls = patched_run()

    with pytest.raises(server.ComfyCliError, match=match):
        _run_template(**kwargs)

    assert calls == []


def test_run_template_clamps_timeout(patched_run):
    """An absurd timeout is clamped, so no `comfy run-template` child runs forever."""
    calls = patched_run()

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
    patched_run, budget, expected
):
    """The engine's per-event bound is lowered to a short budget, never raised.

    comfy-cli's own default is 120s and also bounds its "is ComfyUI up?" probe,
    so without this a `timeout_seconds` under 120 was spent entirely inside that
    probe and the child was SIGKILLed by the parent with no diagnostic. Raising
    it past the default would instead blunt stall detection on long runs.
    """
    calls = patched_run()

    _run_template("t", timeout_seconds=budget)

    assert expected in calls[0]["cmd"]
    # The parent stays a backstop only: strictly later than the engine's bound.
    assert calls[0]["timeout"] == budget + server._RUN_TEMPLATE_TIMEOUT_GRACE


def test_run_template_engine_timeout_is_an_int(patched_run):
    """comfy-cli types `--timeout` as an int, so a float would be a parse error."""
    calls = patched_run()

    _run_template("t", timeout_seconds=45.7)

    flag = next(a for a in calls[0]["cmd"] if a.startswith("--timeout="))
    assert flag == "--timeout=45"
    assert "." not in flag


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -5.0])
def test_run_template_rejects_non_positive_or_nan_timeout(patched_run, bad):
    """NaN/non-positive timeouts are refused (NaN survives min/max comparisons)."""
    calls = patched_run()

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        _run_template("t", timeout_seconds=bad)

    assert calls == []


# --- discoverability --------------------------------------------------------


def test_instructions_mention_run_template(patched_run):
    """The handshake should teach the one-command template run, spend posture too."""
    instructions = server.mcp.instructions

    assert "run_template" in instructions
