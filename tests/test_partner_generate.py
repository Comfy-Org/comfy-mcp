"""Tests for ``partner_generate`` — the thin passthrough to ``comfy generate <model>``.

These lock in the three things this wrapper is responsible for, given that all
the partner logic (model catalog, schema validation, the spend gate itself)
lives in comfy-cli:

1. The passthrough argv: global flags before the subcommand, the model as the
   first positional, per-model params as ``--name=value``.
2. The SPEND CONSENT posture — ``--yes`` is sent only when the USER granted
   consent for that call (the elicitation prompt, or the explicit
   ``confirm_spend`` fallback on a client that cannot elicit), the engine's
   durable always-proceed is honored without one, and there is no second, covert
   way to send it. The engine still owns the interlock; this only proves what we
   forward, and that a decline spawns no child at all.
3. The no-envelope result contract: ``comfy generate`` prints human text and
   exits 0 without an ``envelope/1``, so success is synthesized (``plain_ok``)
   while a non-zero exit — the engine's fail-closed consent refusal — raises.

comfy-cli is mocked throughout: no real ComfyUI, and no real credit spend.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata

import pytest
from conftest import envelope
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_mcp import server


def _generate(*args, **kwargs):
    """Drive the async ``partner_generate`` tool from a sync test.

    Matches the ``asyncio.run`` convention the other async tools' tests use
    (`test_wrapper.py`); the tool went async in order to raise the per-call
    spend-confirmation elicitation.
    """
    return asyncio.run(server.partner_generate(*args, **kwargs))


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake MCPServer ``Context`` that answers the elicitation with ``action``.

    Records every elicitation raised so a test can assert the prompt happened
    (and what it said), and can be told to have no elicitation capability at all
    — the two clients the consent path has to tell apart.
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


# --- passthrough argv -------------------------------------------------------


def test_partner_generate_argv_is_a_local_json_passthrough(patched_plain_run):
    """`comfy --json --where local generate <model> --param=value` (flags first)."""
    calls = patched_plain_run(0, stdout="Saved image to /tmp/out.png")

    _generate("flux-pro", params={"prompt": "a red fox"})

    cmd = calls[0]["cmd"]
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["generate", "flux-pro", "--prompt=a red fox", "--timeout=600.0"]
    assert calls[0]["env"]["COMFY_WHERE"] == "local"  # belt-and-suspenders pin


def test_partner_generate_is_attributed_to_this_server(patched_plain_run):
    """The paid partner call carries this server's caller label.

    `comfy generate` is the spend path, so it is the one whose origin the
    engine's usage attribution most needs to record: without the label an
    MCP-driven partner generation is indistinguishable from a human running the
    same command. `_comfy_env` sets it for every spawn; this pins it on the
    partner one specifically, since that is the reason it is set at all.
    """
    calls = patched_plain_run(0, stdout="Saved image to /tmp/out.png")

    _generate("flux-pro", params={"prompt": "a red fox"})

    assert calls[0]["env"]["COMFY_USER_AGENT"] == "comfy-mcp"


def test_partner_generate_marshals_param_types(patched_plain_run):
    """Params render in the spellings comfy-cli's schema parser accepts."""
    calls = patched_plain_run(0, stdout="done")

    _generate(
        "flux-pro",
        params={
            "prompt": "a cat",
            "steps": 30,  # int -> plain str
            "guidance": 3.5,  # float -> plain str
            "raw": True,  # bool -> lowercase true/false, never Python's "True"
            "tiled": False,
            "sizes": [512, 768],  # list -> JSON, the form its array parser takes
            "seed": None,  # dropped entirely, NOT sent as the string "None"
        },
    )

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--prompt=a cat",
        "--steps=30",
        "--guidance=3.5",
        "--raw=true",
        "--tiled=false",
        "--sizes=[512, 768]",
        "--timeout=600.0",
    ]


def test_partner_generate_param_value_with_leading_dash_stays_a_value(
    patched_plain_run,
):
    """`--name=value` keeps a dash-leading value from being read as the next flag."""
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro", params={"prompt": "--not-a-flag, just text"})

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--prompt=--not-a-flag, just text",
        "--timeout=600.0",
    ]


def test_partner_generate_forwards_out_path(patched_plain_run):
    """`out_path` forwards comfy-cli's `--download`, in the `=value` form.

    The ARGUMENT was renamed `download` -> `out_path` to match the server-wide
    naming convention (an output file is `out_path` everywhere); the comfy-cli
    FLAG it forwards is unchanged, so the argv assertion pins `--download=`.
    """
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro", out_path="/tmp/out.png")

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--download=/tmp/out.png",
        "--timeout=600.0",
    ]


def test_partner_generate_forwards_out_path_template_verbatim(patched_plain_run):
    """A save-path TEMPLATE (`{index}` / trailing slash) is forwarded unchanged.

    comfy-cli resolves the template itself (`generate/output.py` `save_urls`),
    so this wrapper must not normalize, expand, or split it.
    """
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro", out_path="/tmp/gen/{index}.{ext}")

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--download=/tmp/gen/{index}.{ext}",
        "--timeout=600.0",
    ]


def test_partner_generate_schema_names_the_save_path_out_path():
    """The rename is a real rename: no `download` alias survives in the schema.

    Clients introspect this schema fresh each session, so `out_path` is what an
    agent sees. Two visible names for one save path would enlarge exactly the
    guess space the naming convention exists to shrink — hence no alias.
    """
    properties = server.mcp._tool_manager.get_tool("partner_generate").parameters[
        "properties"
    ]

    assert "out_path" in properties
    assert "download" not in properties


def test_partner_generate_omits_params_when_none(patched_plain_run):
    """No params -> a bare `generate <model>`, not an empty flag token."""
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro")

    assert calls[0]["cmd"][4:] == ["generate", "flux-pro", "--timeout=600.0"]


# --- spend consent ----------------------------------------------------------


def test_partner_generate_withholds_consent_by_default(patched_plain_run):
    """The default sends NO `--yes`: comfy-cli's gate is left to fail closed."""
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro", params={"prompt": "a cat"})

    assert "--yes" not in calls[0]["cmd"]


def test_partner_generate_forwards_consent_when_confirmed(patched_plain_run):
    """`confirm_spend=True` forwards `--yes` on a client that cannot elicit.

    The fallback path: with no elicitation-capable client there is no prompt to
    raise, so the explicit argument is the only consent available.
    """
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro", params={"prompt": "a cat"}, confirm_spend=True)

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--prompt=a cat",
        "--timeout=600.0",
        "--yes",
    ]


# --- per-call spend elicitation ---------------------------------------------


def test_partner_generate_elicits_and_forwards_consent_on_approval(patched_plain_run):
    """An elicitation-capable client is PROMPTED, and approval forwards `--yes`."""
    calls = patched_plain_run(0, stdout="done")
    ctx = _FakeCtx(action="accept", approve=True)

    _generate("flux-pro", params={"prompt": "a cat"}, ctx=ctx)

    assert len(ctx.elicitations) == 1  # the user was actually asked
    assert "flux-pro" in ctx.elicitations[0]
    assert "credits" in ctx.elicitations[0]  # and told what it costs
    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--prompt=a cat",
        "--timeout=600.0",
        "--yes",
    ]


@pytest.mark.parametrize(
    ("action", "approve"),
    [
        ("decline", False),  # the user said no
        ("cancel", False),  # the user dismissed the prompt
        ("accept", False),  # accepted the form without answering yes
    ],
)
def test_partner_generate_spends_nothing_when_the_user_does_not_approve(
    patched_plain_run, action, approve
):
    """Decline -> no spend AND no `--yes`: comfy-cli is never even invoked.

    Refusing here rather than letting the engine fail closed matters because the
    engine's gate is not the only thing that could let the call through — a
    `spend.auto_confirm` flipped on between the read and the run would spend the
    user's credits right after they said no. No child, no spend.
    """
    calls = patched_plain_run(0, stdout="done")
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _generate("flux-pro", params={"prompt": "a cat"}, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert calls == []  # nothing ran, so nothing was spent


def test_partner_generate_elicits_even_when_confirm_spend_is_set(patched_plain_run):
    """`confirm_spend=True` is NOT a way around the per-call prompt.

    The design rule this locks in: spend consent != tool permission. An agent
    host that grants blanket "always allow this tool" (and a caller that sets
    the argument off the back of it) must not be able to turn that into standing
    authority over the user's credits — on a client that can ask, the user is
    asked, and a decline still spends nothing.
    """
    calls = patched_plain_run(0, stdout="done")
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _generate("flux-pro", confirm_spend=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert calls == []


def test_partner_generate_does_not_elicit_without_client_support(patched_plain_run):
    """A client that never advertised elicitation is not sent a prompt it can't answer."""
    calls = patched_plain_run(0, stdout="done")
    ctx = _FakeCtx(supports_elicitation=False)

    _generate("flux-pro", confirm_spend=True, ctx=ctx)

    assert ctx.elicitations == []  # no prompt raised at a client that can't show one
    assert "--yes" in calls[0]["cmd"]  # the explicit argument still carries consent


def test_partner_generate_surfaces_a_broken_elicitation_without_spending(
    patched_plain_run,
):
    """A client that ERRORS on the prompt is not silently treated as approval."""
    calls = patched_plain_run(0, stdout="done")

    class _BrokenCtx(_FakeCtx):
        async def elicit(self, message, schema):
            raise RuntimeError("client closed the connection")

    with pytest.raises(
        server.ComfyCliError, match="could not confirm the credit spend"
    ):
        _generate("flux-pro", ctx=_BrokenCtx())

    assert calls == []


def test_partner_generate_honors_the_engines_durable_always_proceed(
    monkeypatch, patched_plain_run
):
    """`spend.auto_confirm` -> no prompt, and no `--yes`: the engine consents itself.

    The durable "always proceed" is the user's own choice, persisted in
    comfy-cli's config — this server keeps no such state. Honoring it is what
    keeps `comfy generate consent always` meaningful through the MCP, instead of
    the wrapper re-asking a question the user already answered for good.
    """
    monkeypatch.setattr(server, "_engine_auto_confirms", lambda: True)
    calls = patched_plain_run(0, stdout="done")
    ctx = _FakeCtx(action="decline")

    _generate("flux-pro", ctx=ctx)

    assert ctx.elicitations == []  # nothing left to ask
    # consent is the engine's, not ours to assert
    assert "--yes" not in calls[0]["cmd"]


# --- reading the engine's durable always-proceed ----------------------------


# Captured at import, BEFORE conftest's autouse stub replaces it with a
# constant "off" — the tests below exercise the real probe.
_real_engine_auto_confirms = server._engine_auto_confirms


def _patched_consent_show(
    patched_plain_run, returncode: int, stdout: str
) -> list[dict]:
    """Stub `comfy generate consent show --json` with a canned exit + stdout."""
    return patched_plain_run(returncode, stdout=stdout)


def test_engine_auto_confirms_reads_the_json_consent_payload(patched_plain_run):
    """comfy-cli pretty-prints the setting, so it is read from the WHOLE stdout.

    `output.print_json` uses `indent=2`, so no single line parses — the
    line-oriented envelope scanner would see nothing and report "off" for a user
    who had turned it on.
    """
    calls = _patched_consent_show(
        patched_plain_run,
        0,
        '{\n  "spend_auto_confirm": true,\n  "action": "show"\n}\n',
    )

    assert _real_engine_auto_confirms() is True
    assert calls[0]["cmd"][4:] == ["generate", "consent", "show", "--json"]


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (0, '{"spend_auto_confirm": false, "action": "show"}'),  # explicitly off
        (0, '{"spend_auto_confirm": "true"}'),  # a string is not a JSON true
        (0, '{"action": "show"}'),  # key absent
        (0, "spend.auto_confirm: true"),  # human text, not JSON
        (0, ""),  # nothing at all
        (1, '{"spend_auto_confirm": true}'),  # a failed read authorizes nothing
    ],
)
def test_engine_auto_confirms_is_false_for_anything_unreadable(
    patched_plain_run, returncode, stdout
):
    """Only a real JSON `true` from a clean exit counts as pre-authorization.

    Every other answer falls through to ASKING the user — the failure direction
    that costs a prompt, not the one that costs money.
    """
    _patched_consent_show(patched_plain_run, returncode, stdout)

    assert _real_engine_auto_confirms() is False


def test_engine_auto_confirms_is_false_when_the_probe_cannot_run(monkeypatch):
    """No `comfy` on PATH -> a ComfyCliError, absorbed as "not pre-authorized"."""
    monkeypatch.setattr(server.shutil, "which", lambda _: None)

    assert _real_engine_auto_confirms() is False


@pytest.mark.parametrize(
    "exc",
    [
        PermissionError(13, "Permission denied"),  # `comfy` present, not executable
        OSError(8, "Exec format error"),  # wrong-arch binary
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
)
def test_engine_auto_confirms_is_false_when_the_probe_blows_up(patched_run, exc):
    """The docstring promises EVERY failure answers False — including non-CLI ones.

    `_run_comfy_raw` converts a timeout and a missing binary into `ComfyCliError`,
    but a present-but-unusable one (`PermissionError`/`OSError`) and invalid-UTF-8
    child output (`UnicodeDecodeError`) escape it raw. Crashing `partner_generate`
    on those is strictly worse than falling back to asking the user.

    The shared fake raises each of these from the place the real spawn would —
    the two `OSError`s from `Popen`, the decode error from `communicate` — so
    this also covers the handler that kills the process group on a non-timeout
    failure.
    """
    patched_run(raises=exc)

    assert _real_engine_auto_confirms() is False


def test_engine_auto_confirms_asks_comfy_cli_for_json_explicitly(patched_plain_run):
    """The trailing `--json` is load-bearing and must not be "cleaned up" away.

    `comfy generate` takes `allow_extra_args`, so the tail after the target
    reaches `consent`'s OWN meta-flag parser — and it prints JSON only when IT
    sees `--json`. The global `--json` before the subcommand does not reach it,
    so dropping this flag would make the command print rich human text, fail the
    parse, and silently turn `comfy generate consent always` into a dead setting.
    """
    calls = _patched_consent_show(patched_plain_run, 0, '{"spend_auto_confirm": true}')

    assert _real_engine_auto_confirms() is True
    # The global flag precedes the subcommand; the subcommand's own flag trails it.
    assert calls[0]["cmd"][:2] == [server.COMFY_BIN, "--json"]
    assert calls[0]["cmd"][-1] == "--json"


def test_engine_auto_confirms_unwraps_a_future_envelope(patched_plain_run):
    """If comfy-cli ever wraps this verb in an `envelope/1`, read `data`."""
    _patched_consent_show(
        patched_plain_run,
        0,
        '{"schema": "envelope/1", "type": "envelope", "ok": true, '
        '"data": {"spend_auto_confirm": true}}',
    )

    assert _real_engine_auto_confirms() is True


def test_partner_generate_consent_refusal_raises_and_surfaces_the_reason(
    patched_plain_run,
):
    """The engine's fail-closed refusal (non-zero exit) raises — no false success.

    This is the interlock working: comfy-cli exits 1 without spending when no
    consent was given, so `plain_ok` must NOT synthesize a success for it, and
    the CLI's own remediation text has to reach the caller.
    """
    patched_plain_run(
        1,
        stderr=(
            "`comfy generate flux-pro` spends Comfy credits and no consent was "
            "given. Re-run with --yes, or persist consent with "
            "`comfy generate consent always`."
        ),
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        _generate("flux-pro", params={"prompt": "a cat"})

    assert "spends Comfy credits" in str(excinfo.value)  # the reason survives
    assert "--yes" in str(excinfo.value)  # and so does the way forward


@pytest.mark.parametrize("name", ["yes", "async", "json", "download", "api_key"])
def test_partner_generate_refuses_run_level_flags_as_params(patched_plain_run, name):
    """A run-level comfy-cli flag can't be smuggled in through `params`.

    Most important is `yes`: without this guard, `params={"yes": True}` would
    render `--yes` and grant spend consent behind `confirm_spend`'s back — a
    second, undocumented consent path. `json`/`async` would instead break the
    result contract this tool documents. Underscore spellings are caught too.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="run-level"):
        _generate("flux-pro", params={name: True})

    assert calls == []  # refused before comfy-cli was ever invoked


# --- input guards -----------------------------------------------------------


@pytest.mark.parametrize("model", ["", "--help", "-x"])
def test_partner_generate_rejects_option_like_model(patched_plain_run, model):
    """An empty/dash-leading model would be parsed by comfy-cli as an option."""
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="invalid model"):
        _generate(model)

    assert calls == []  # never reached comfy-cli


@pytest.mark.parametrize(
    "target", ["list", "schema", "refresh", "upload", "resume", "consent"]
)
def test_partner_generate_rejects_reserved_subactions(patched_plain_run, target):
    """comfy-cli's own `generate` sub-actions are not partner models.

    `consent` matters most — it is the spend gate's configuration surface, and
    this tool must not become a back door to it.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="sub-action"):
        _generate(target)

    assert calls == []


def test_partner_generate_rejects_option_like_param_name(patched_plain_run):
    """A dash-leading/empty param name would emit a malformed `---flag` token."""
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="invalid parameter name"):
        _generate("flux-pro", params={"--prompt": "a cat"})

    assert calls == []


def test_partner_generate_rejects_param_names_that_smuggle_a_value(
    patched_plain_run,
):
    """A `=` inside the KEY would land as a run-level flag comfy-cli honors.

    comfy-cli splits `--<body>` at the FIRST `=`, so `{"output-prefix=/tmp/x": v}`
    renders `--output-prefix=/tmp/x=v` and sets the blocked run-level
    `output-prefix` — the meta-flag check above never matches because it only
    sees the whole key.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="cannot contain"):
        _generate("flux-pro", params={"output-prefix=/tmp/x": "v"})

    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "flux\0pro"}, "invalid model"),
        ({"model": "flux-pro", "params": {"pro\0mpt": "a cat"}}, "parameter name"),
        ({"model": "flux-pro", "params": {"prompt": "a\0cat"}}, "value for parameter"),
        ({"model": "flux-pro", "out_path": "/tmp/\0.png"}, "invalid out_path"),
    ],
)
def test_partner_generate_rejects_embedded_nul(patched_plain_run, kwargs, match):
    """A NUL is legal in a JSON string but `subprocess` raises a bare ValueError.

    Uncaught, that surfaces as an internal error instead of a ComfyCliError, so
    catch the shape here — before any child is spawned.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match=match):
        _generate(**kwargs)

    assert calls == []


def test_partner_generate_rejects_an_empty_out_path(patched_plain_run):
    """`out_path=""` is a caller mistake, not "use the default location"."""
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="invalid out_path"):
        _generate("flux-pro", out_path="")

    assert calls == []


def test_partner_generate_rejects_an_oversized_out_path(no_spawn):
    """An oversized `out_path` is refused before it can reach argv.

    The sixth of the path-shaped siblings, and the one whose empty-string branch
    runs FIRST: `out_path=""` carries distinct None-vs-empty semantics, so the
    size cap sits between that branch and `_reject_nul` rather than at the top.
    """
    oversized = "/tmp/" + "p" * server._MAX_PATH_ARG_LEN + ".png"

    with pytest.raises(server.ComfyCliError, match="exceeds") as excinfo:
        _generate("flux-pro", out_path=oversized)

    # Length-not-value: `_reject_nul`'s message would name the value instead.
    assert oversized not in str(excinfo.value)


def test_partner_generate_allows_an_out_path_at_the_ceiling(patched_plain_run):
    """The boundary value itself rides through into `--download=`."""
    calls = patched_plain_run(0, stdout="done")
    at_ceiling = "/tmp/" + "p" * (server._MAX_PATH_ARG_LEN - len("/tmp/"))
    assert len(at_ceiling) == server._MAX_PATH_ARG_LEN

    _generate("flux-pro", out_path=at_ceiling)

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        f"--download={at_ceiling}",
        "--timeout=600.0",
    ]


def test_partner_generate_clamps_timeout(patched_plain_run):
    """An absurd timeout is clamped, so no `comfy generate` child runs forever."""
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro", timeout_seconds=float("inf"))

    assert f"--timeout={server._MAX_GENERATE_TIMEOUT}" in calls[0]["cmd"]
    assert (
        calls[0]["timeout"]
        == server._MAX_GENERATE_TIMEOUT + server._GENERATE_TIMEOUT_GRACE
    )


def test_partner_generate_lets_the_engine_own_the_deadline(patched_plain_run):
    """`timeout_seconds` becomes comfy-cli's own `--timeout`; we only backstop it.

    Without the forwarded flag comfy-cli falls back to its 300s default, so a
    caller asking for longer silently got five minutes — and a parent kill at
    the same moment would destroy the resumable handle for a job the partner had
    already accepted and CHARGED for, inviting a double-spending retry.
    """
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro", timeout_seconds=900.0)

    assert "--timeout=900.0" in calls[0]["cmd"]
    assert calls[0]["timeout"] == 900.0 + server._GENERATE_TIMEOUT_GRACE
    assert calls[0]["timeout"] > 900.0  # the engine gives up first


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -5.0])
def test_partner_generate_rejects_a_non_positive_or_nan_timeout(patched_plain_run, bad):
    """NaN survives `min(max(...))` — and reaches `subprocess.run` as a ValueError.

    Every NaN comparison is False, so the old clamp returned it unchanged,
    defeating the ceiling with the one value it most needed to stop. A
    non-positive timeout is refused rather than floored to an instant timeout.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        _generate("flux-pro", timeout_seconds=bad)

    assert calls == []


# --- the engine's spend gate must actually be installed ---------------------


def test_partner_generate_refuses_when_comfy_cli_has_no_spend_gate(
    monkeypatch, patched_plain_run
):
    """No `comfy generate consent` -> no interlock -> refuse BEFORE spending.

    The fail-closed guarantee is comfy-cli's. It ships in 1.13.0, below the floor
    this server enforces, but the floor check fails OPEN (an unparseable `--version`,
    a source build, a fork), so the version check still cannot PROVE the gate is
    there. Against a CLI without it a default `confirm_spend=False` call would
    charge the user with nothing to stop it.
    """
    monkeypatch.setattr(server, "_spend_gate_probed", False)
    # An older comfy-cli reads `consent` as a model name and exits non-zero.
    calls = patched_plain_run(1, stderr="no such model")

    with pytest.raises(server.ComfyCliError, match="no `comfy generate` spend gate"):
        _generate("flux-pro", params={"prompt": "a cat"})

    assert len(calls) == 1  # only the probe ran
    # never the generation
    assert calls[0]["cmd"][4:] == ["generate", "consent", "show"]
    assert server._spend_gate_probed is False  # a failed probe is not latched


def test_partner_generate_checks_the_gate_before_prompting_the_user(
    monkeypatch, patched_plain_run
):
    """The gate probe runs BEFORE the elicitation, not after.

    Prompting first would ask the user to authorize a spend this call was always
    going to refuse — a confirmation dialog whose only outcome is an error.
    """
    monkeypatch.setattr(server, "_spend_gate_probed", False)
    patched_plain_run(1, stderr="no such model")
    ctx = _FakeCtx(action="accept", approve=True)

    with pytest.raises(server.ComfyCliError, match="no `comfy generate` spend gate"):
        _generate("flux-pro", ctx=ctx)

    assert ctx.elicitations == []  # the user was never asked a pointless question


@pytest.mark.parametrize(
    "ctx",
    [
        None,  # no context at all (a direct call)
        _FakeCtx(supports_elicitation=False),  # client never advertised it
    ],
)
def test_client_elicitation_support_is_false_without_a_capable_client(ctx):
    """Only a client that advertised elicitation is sent a prompt."""
    assert server._client_elicitation_support(ctx) is False


def test_client_elicitation_support_is_false_outside_a_live_request():
    """`Context.session` raises outside a request — that is "cannot ask", not a crash."""

    class _NoSessionCtx:
        async def elicit(self, message, schema):
            raise AssertionError("must never be reached")

        @property
        def session(self):
            raise ValueError("Context is not available outside of a request")

    assert server._client_elicitation_support(_NoSessionCtx()) is False


def test_client_elicitation_support_is_true_for_a_capable_client():
    """The positive case, so the two guards above can't pass by always saying no."""
    assert server._client_elicitation_support(_FakeCtx()) is True


def test_client_elicitation_support_is_unknown_when_the_probe_raises():
    """A probe that ERRORS is `None` — "could not tell", not "cannot ask"."""

    class _BrokenProbeCtx(_FakeCtx):
        def __init__(self):
            super().__init__()
            self.session = _BrokenSession()

    class _BrokenSession:
        def check_client_capability(self, capability):
            raise RuntimeError("capability table is corrupt")

    assert server._client_elicitation_support(_BrokenProbeCtx()) is None


def test_an_errored_capability_probe_still_asks_before_spending(patched_plain_run):
    """A probe error must not silently promote `confirm_spend` into a free pass.

    The failure this locks out: an elicitation-CAPABLE client whose probe merely
    errored gets treated as incapable, so `confirm_spend=True` forwards `--yes`
    and spends credits with no human prompt — exactly the "tool permission
    became spend consent" outcome this tool exists to prevent. Unknown means
    ASK.
    """
    calls = patched_plain_run(0, stdout="done")

    class _BrokenSession:
        def check_client_capability(self, capability):
            raise RuntimeError("capability table is corrupt")

    ctx = _FakeCtx(action="decline")
    ctx.session = _BrokenSession()

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _generate("flux-pro", confirm_spend=True, ctx=ctx)

    assert len(ctx.elicitations) == 1  # asked, despite the broken probe
    assert calls == []  # and the decline still spent nothing


def test_an_unanswered_prompt_lapses_into_a_refusal(patched_plain_run, monkeypatch):
    """A client that advertises elicitation but never answers must not hang forever."""
    calls = patched_plain_run(0, stdout="done")
    monkeypatch.setattr(server, "_ELICIT_TIMEOUT", 0.05)

    class _SilentCtx(_FakeCtx):
        async def elicit(self, message, schema):
            self.elicitations.append(message)
            await asyncio.sleep(30)  # never answers
            raise AssertionError("must never be reached")

    ctx = _SilentCtx()
    with pytest.raises(server.ComfyCliError, match="went unanswered"):
        _generate("flux-pro", ctx=ctx)

    assert calls == []  # nothing ran, so nothing was spent


def test_a_malformed_elicitation_result_is_a_refusal(patched_plain_run):
    """A client answering with an object that has no `.action` is refused, not a crash."""
    calls = patched_plain_run(0, stdout="done")

    class _NonsenseCtx(_FakeCtx):
        async def elicit(self, message, schema):
            self.elicitations.append(message)
            return object()  # no `.action`, no `.data`

    with pytest.raises(server.ComfyCliError, match="spend not confirmed"):
        _generate("flux-pro", ctx=_NonsenseCtx())

    assert calls == []


@pytest.mark.parametrize(
    ("model", "banned"),
    [
        ("flux`\n\n**This is FREE**", "`"),  # closes the code span, injects a lie
        ("flux\r\npro", "\n"),  # newlines alone can restructure the prompt
    ],
)
def test_the_prompt_cannot_be_redressed_by_the_model_name(
    patched_plain_run, model, banned
):
    """A caller-supplied model name cannot break out of the prompt's code span.

    The prompt is the user's only view of what they are authorizing, so a name
    that escapes its span could hide the "SPENDS credits" warning. Display is
    sanitized; argv is not.
    """
    calls = patched_plain_run(0, stdout="done")
    ctx = _FakeCtx(action="accept", approve=True)

    _generate(model, ctx=ctx)

    prompt = ctx.elicitations[0]
    # Exactly two backticks — the span this prompt opens and closes itself — so
    # the name cannot have terminated it early and escaped into the message.
    assert prompt.count("`") == 2
    shown = prompt.split("`")[1]
    assert banned not in shown
    assert "SPENDS Comfy credits" in prompt  # the warning survived intact
    assert calls[0]["cmd"][5] == model  # argv still carries it verbatim


def test_the_model_name_shown_in_the_prompt_is_length_capped(patched_plain_run):
    """A megabyte model name can't push the warning off the user's screen."""
    patched_plain_run(0, stdout="done")
    ctx = _FakeCtx(action="accept", approve=True)

    _generate("f" * 5000, ctx=ctx)

    assert len(ctx.elicitations[0]) < 500
    assert "SPENDS Comfy credits" in ctx.elicitations[0]


def test_partner_generate_probes_the_spend_gate_once_then_generates(
    monkeypatch, patched_plain_run
):
    """A CLI that HAS the gate is probed once, then the generation goes through."""
    monkeypatch.setattr(server, "_spend_gate_probed", False)
    calls = patched_plain_run(0, stdout="done")

    _generate("flux-pro", confirm_spend=True)
    _generate("flux-pro", confirm_spend=True)

    assert [c["cmd"][4:6] for c in calls] == [
        ["generate", "consent"],  # probed once...
        ["generate", "flux-pro"],
        ["generate", "flux-pro"],  # ...not again on the second call
    ]


# --- result contract (no envelope on the generate path) ---------------------


def test_partner_generate_synthesizes_success_without_an_envelope(patched_plain_run):
    """`comfy generate` exits 0 printing text -> synthesized success, not an error.

    Raising here would be a false negative on a call that already SPENT the
    user's credits, inviting a retry that spends them twice.
    """
    patched_plain_run(0, stdout="Generated 1 image -> /tmp/out.png")

    result = _generate("flux-pro", params={"prompt": "a cat"}, confirm_spend=True)

    assert result["ok"] is True
    assert result["action"] == "generate flux-pro"  # stops before the flags
    assert "/tmp/out.png" in result["message"]


# --- saved_paths (the resolved destinations, as data) -----------------------
#
# `comfy generate` prints where it put each asset as an indented `Saved:` block,
# and comfy-cli renders it through rich, whose Console is a fixed 80 columns off
# a TTY. A path longer than that is FOLDED mid-filename with no indent on the
# continuation, so `message` alone is not parseable by a caller — which in
# practice forced a shell-out to `ls` to confirm anything landed at all.


def _cells(text: str) -> int:
    """Cell width of `text`, computed independently of the server's own helper.

    rich measures its line breaks in terminal CELLS, so the renderer below must
    too — and it must not borrow `server._cell_len`, or a bug in that helper
    would cancel itself out and every fold test would pass on input the real
    parser cannot handle.
    """
    return sum(
        0
        if unicodedata.combining(char)
        else 2
        if unicodedata.east_asian_width(char) in ("W", "F")
        else 1
        for char in text
    )


def _chop(text: str, width: int) -> tuple[str, str]:
    """Split `text` at `width` CELLS — rich's `chop_cells`, minus the caching."""
    used = 0
    for position, char in enumerate(text):
        size = _cells(char)
        if used + size > width:
            return text[:position], text[position:]
        used += size
    return text, ""


def _fold(
    path: str, *, width: int = server._RICH_DEFAULT_WIDTH, indent: str = "  "
) -> str:
    """Render one saved path the way rich actually would, at `width` cells.

    Built here rather than hard-coding a wrapped blob so a test states the PATH
    it means and the break falls out, and so a three-line break is as easy to
    write as a two-line one.

    This models rich's `divide_line`, not a fixed-width slice, because the two
    disagree on exactly the inputs that used to come back truncated: rich
    WORD-WRAPS at whitespace before it folds (so a path with a space breaks
    early, leaving a line SHORTER than `width`), and it measures in cells (so a
    path with wide characters breaks after fewer code points than `width`).
    Slicing at fixed character counts reproduced the parser's own model of rich
    instead of rich, which is what let both silent-truncation modes ship green.
    """
    text = indent + path
    lines: list[str] = []
    current = ""
    # rich's `words()`: the whitespace before a word rides with it, so a break
    # taken before the word is what swallows that whitespace.
    for chunk in re.findall(r"\s*\S+", text):
        word = chunk.lstrip()
        lead = chunk[: len(chunk) - len(word)]
        if current and _cells(current + lead + word) > width:
            lines.append(current)
            current = ""
            lead = ""
        piece = current + lead + word
        # A single word too long for a whole line is FOLDED at the cell width.
        while _cells(piece) > width:
            head, piece = _chop(piece, width)
            lines.append(head)
        current = piece
    if current:
        lines.append(current)
    return "\n".join(lines)


def test_the_fold_renderer_matches_rich_where_the_two_could_drift():
    """Guard the renderer above: these are the shapes the parser is judged on."""
    # A word too long for the line folds at exactly `width` cells, indent included.
    assert _fold("a" * 100, width=20).splitlines() == [
        "  " + "a" * 18,
        *["a" * 20] * 4,
        "a" * 2,
    ]
    # A space wraps EARLY, leaving a line SHORTER than `width` — the shape a
    # fixed-width slice can never produce, and the one that used to truncate.
    assert _fold("/tmp/my dir/" + "z" * 30, width=40).splitlines() == [
        "  /tmp/my",
        "dir/" + "z" * 30,
    ]
    # Wide characters reach the edge after fewer code points than `width`.
    assert _fold("漢" * 20, width=20).splitlines()[0] == "  " + "漢" * 9


def test_partner_generate_reports_saved_paths(patched_plain_run):
    """The destination comes back as data, alongside the raw text."""
    patched_plain_run(0, stdout="Saved:\n  /tmp/a.png\n  /tmp/b.png\n", stderr="  \n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == ["/tmp/a.png", "/tmp/b.png"]
    assert "Saved:" in result["message"]  # the raw text is RETAINED, not replaced


def test_partner_generate_reassembles_a_path_rich_folded(patched_plain_run):
    """A path longer than the console width is stitched back together.

    This is the case that made `message` unusable: the fold lands mid-filename
    (`…/partner-jellyfish.p` / `ng`), so a caller reading the first line alone
    gets a path that does not exist.
    """
    path = (
        "/Users/someone/Pictures/comfy-generations/2026-07/"
        "partner-jellyfish-final-v2.png"
    )
    rendered = _fold(path)
    assert len(rendered.splitlines()) == 2  # it genuinely folded, mid-filename
    patched_plain_run(0, stdout=f"Saved:\n{rendered}\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == [path]


def test_partner_generate_reassembles_a_path_folded_more_than_once(
    patched_plain_run,
):
    """Three physical lines still reassemble — the fold width is per-line."""
    path = "/Users/someone/" + "deeply-nested-directory/" * 8 + "out.png"
    assert len(path) > 160  # genuinely spans three 80-column lines
    patched_plain_run(0, stdout=f"Saved:\n{_fold(path)}\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == [path]


def test_a_path_containing_spaces_survives_richs_word_wrap(patched_plain_run):
    """rich wraps at the SPACE before it folds, and the path still comes back whole.

    The macOS shape (`~/My Pictures/…`). rich breaks at the space, so the first
    physical line is SHORTER than the console width — and a parser that treats
    only exact-width lines as foldable reads the block as finished, hands back
    `/Users/someone/My` as a resolved destination, and silently drops every
    remaining path. A truncated path is worse than no path: the caller acts on a
    file that was never written.
    """
    path = (
        "/Users/someone/My Pictures/comfy-generations/2026-07-28/"
        "partner-jellyfish-final.png"
    )
    rendered = _fold(path)
    assert len(rendered.splitlines()) == 2  # it really did wrap
    assert rendered.splitlines()[0] == "  /Users/someone/My"  # ...at the space
    patched_plain_run(0, stdout=f"Saved:\n{rendered}\n  /tmp/b.png\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == [path, "/tmp/b.png"]


def test_a_path_of_wide_characters_is_measured_in_cells_not_code_points(
    patched_plain_run,
):
    """A CJK path folds after fewer CHARACTERS than the console is wide.

    Measuring the rendered line with `len` makes the fold look like a finished
    path, so the continuation is dropped and a truncated path is reported.
    """
    path = "/Users/someone/" + "画像フォルダ/" * 6 + "out.png"
    assert len(path) < server._RICH_DEFAULT_WIDTH  # `len` says it never folded...
    rendered = _fold(path)
    assert len(rendered.splitlines()) > 1  # ...but in CELLS it did
    patched_plain_run(0, stdout=f"Saved:\n{rendered}\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == [path]


def test_a_path_left_unfinished_is_dropped_rather_than_truncated(patched_plain_run):
    """Text that ends mid-fold yields a PREFIX, so report nothing instead.

    A full-width final line means rich had more to print. Returning what we hold
    would name a file that does not exist; `message` still carries the text.
    """
    path = "/Users/someone/" + "deeply-nested-directory/" * 4 + "jellyfish.png"
    first_line = _fold(path).splitlines()[0]
    assert len(first_line) == server._RICH_DEFAULT_WIDTH  # the text stops here
    patched_plain_run(0, stdout=f"Saved:\n  /tmp/a.png\n{first_line}")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == ["/tmp/a.png"]  # the prefix is NOT reported


def test_an_over_long_path_is_dropped_rather_than_sliced(patched_plain_run):
    """A path past PATH_MAX cannot name a real file, and a prefix of it names a
    DIFFERENT plausible one — so drop it and keep the paths around it."""
    huge = "/tmp/" + "d" * (server._MAX_SAVED_PATH_CHARS + 10) + ".png"
    patched_plain_run(0, stdout=f"Saved:\n  /tmp/a.png\n  {huge}\n  /tmp/b.png\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == ["/tmp/a.png", "/tmp/b.png"]


def test_saved_paths_stop_at_unrelated_output(patched_plain_run):
    """A short path is complete; the next unindented line is not part of it.

    Gluing trailing output onto a path would be worse than reporting none: the
    caller would act on a destination that never existed.
    """
    patched_plain_run(
        0,
        stdout="Saved:\n  /tmp/a.png\nAll done in 12.4s\n",
    )

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == ["/tmp/a.png"]


def test_saved_paths_stop_at_unrelated_output_after_a_long_path(patched_plain_run):
    """The same, for a path long enough to be a plausible fold candidate.

    The short-path case above is the easy one. This path is long enough that a
    naive "the widest line in the block is the fold width" rule would call the
    line below it a continuation and hand back a destination that never existed.
    """
    path = "/Users/someone/Pictures/comfy-generations/partner-jellyfish.png"
    assert 2 + len(path) < server._RICH_DEFAULT_WIDTH  # rendered, it did not fold
    patched_plain_run(0, stdout=f"Saved:\n  {path}\nAll done in 12.4s\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == [path]


def test_saved_paths_survive_the_message_cap(patched_plain_run):
    """Parsed from the UNCAPPED text — `message` keeps only the last 1000 chars.

    Without this the block could be sliced mid-path by the very cap that exists
    to bound the response.
    """
    noise = "\n".join(f"progress line {i}" for i in range(200))
    patched_plain_run(0, stdout=f"Saved:\n  /tmp/a.png\n\n{noise}\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == ["/tmp/a.png"]
    assert "/tmp/a.png" not in result["message"]  # the cap really did drop it


def test_saved_paths_follow_the_console_width_the_child_rendered_at(
    patched_plain_run, monkeypatch
):
    """`COLUMNS` is forwarded to the child, so it is where the fold really lands.

    Reading the width from the same place rich does — rather than assuming 80 —
    is what keeps this correct for a host that exports `COLUMNS`.
    """
    monkeypatch.setenv("COLUMNS", "50")
    path = "/Users/someone/Pictures/comfy-generations/partner-jellyfish.png"
    patched_plain_run(0, stdout=f"Saved:\n{_fold(path, width=50)}\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == [path]


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "wide",
        "0",
        "-5",
        # Values `int()` accepts but rich's `isdigit()` gate does not, so rich
        # rendered at 80 and measuring against them would mis-assemble the block.
        " 120 ",
        "+120",
        "1_20",
        "120\n",
        # `isdigit()` accepts this one; `int()` is what rejects it.
        "²",
    ],
)
def test_an_unusable_columns_falls_back_to_richs_own_default(monkeypatch, bad):
    """Junk in `COLUMNS` must not make the width nonsense — rich ignores it too."""
    monkeypatch.setenv("COLUMNS", bad)

    assert server._child_console_width() == server._RICH_DEFAULT_WIDTH


def test_saved_paths_are_absent_when_nothing_was_saved(patched_plain_run):
    """No `Saved:` block -> no key at all, rather than a misleading empty list."""
    patched_plain_run(0, stdout="Generated 1 image -> https://example.invalid/x.png")

    assert "saved_paths" not in _generate("flux-pro", confirm_spend=True)


def test_saved_paths_are_bounded(patched_plain_run):
    """A runaway block cannot turn a success payload into an unbounded response."""
    block = "\n".join(f"  /tmp/out-{i}.png" for i in range(200))
    patched_plain_run(0, stdout=f"Saved:\n{block}\n")

    result = _generate("flux-pro", confirm_spend=True)

    assert len(result["saved_paths"]) == server._MAX_SAVED_PATHS


def test_saved_paths_are_not_confused_by_a_long_stderr_line(patched_plain_run):
    """stderr is not rich-rendered, so its lines must not set the fold width.

    `_synthesize_plain_result` prepends stderr (the PostHog/telemetry noise a
    real run carries), and measuring THAT would make every genuine fold look
    unfoldable — leaving a truncated path in `saved_paths`.
    """
    path = "/Users/someone/Pictures/comfy-generations/2026-07/partner-jellyfish.png"
    patched_plain_run(
        0,
        stdout=f"Saved:\n{_fold(path)}\n",
        stderr="x" * 400,  # longer than any console width
    )

    result = _generate("flux-pro", confirm_spend=True)

    assert result["saved_paths"] == [path]


def test_partner_generate_prefers_a_real_envelope_when_comfy_cli_emits_one(
    patched_run,
):
    """If comfy-cli ever DOES emit an envelope for generate, its data wins.

    `plain_ok` is a stopgap, not an override: a real `envelope/1` is unwrapped
    normally, so this tool needs no change when the CLI starts emitting one.
    """

    patched_run(envelope(data={"images": ["/tmp/out.png"]}))

    result = _generate("flux-pro", confirm_spend=True)

    assert result == {"images": ["/tmp/out.png"]}


def test_partner_generate_error_envelope_raises_with_code(patched_run):
    """A real error envelope still surfaces comfy-cli's structured code."""
    patched_run(
        envelope(ok=False, error={"code": "unknown_model", "message": "no such model"}),
        returncode=1,
    )

    with pytest.raises(server.ComfyCliError, match="unknown_model"):
        _generate("nope-not-a-model", confirm_spend=True)


def test_instructions_warn_that_partner_generate_spends_credits():
    """The handshake must teach the spend posture, not just the tool's name."""
    instructions = server.mcp.instructions

    assert "partner_generate" in instructions
    assert "confirm_spend" in instructions
    assert "credits" in instructions
    assert "elicitation" in instructions  # and that the user is asked per call


def test_partner_generate_takes_an_injected_context():
    """MCPServer must recognise `ctx` — or the spend prompt silently never fires.

    The elicitation path is reachable only if the server injects the request
    context; a rename or a retype would quietly degrade every call to the
    `confirm_spend` fallback with nothing failing loudly. `ctx` also has to stay
    OUT of the tool's public input schema, so a caller can't pass one.
    """
    tool = server.mcp._tool_manager.get_tool("partner_generate")

    assert tool.context_kwarg == "ctx"
    assert "ctx" not in tool.parameters["properties"]
