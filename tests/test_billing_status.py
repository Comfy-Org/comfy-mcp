"""Tests for ``billing_status`` — the read-only ``comfy cloud status`` wrapper.

The tool answers "how many credits do I have / what plan am I on / how many
jobs can I run at once" by passing comfy-cli's billing snapshot straight
through. The behaviors these lock in, beyond the passthrough argv:

1. The payload is relayed VERBATIM — including ``message``, comfy-cli's own
   neutral copy for a balance it could not confirm, and the ``null`` balance
   fields that copy explains. Re-deriving either here is what the CLI-side
   trust guard exists to prevent.
2. A comfy-cli that predates the verb degrades to the ``unsupported`` shape
   rather than relaying Click's usage dump. That is the COMMON path today —
   ``cloud status`` is newer than comfy-cli's most recent release — which is
   why it gets an actionable upgrade message rather than a raw failure.
3. Every failure comfy-cli DID dispatch and report — no credential, a rejected
   one, an unreachable billing endpoint — keeps its own code and hint. Those
   are the ones a user can act on, and burying them under "your CLI is old"
   would send them to the wrong fix.
4. It spends nothing, so it takes no ``confirm_spend`` argument and raises no
   elicitation: exactly one subprocess per call.
5. It runs on ``_run_comfy_async``, not the thread pool: the fan-out's budget is
   long enough that a client which disconnects mid-call must not leave the
   ``comfy`` child alive for the rest of it.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from conftest import envelope

from comfy_mcp.server import _internal as server


def _billing_status():
    """Drive the async tool from a sync test.

    Matches the ``asyncio.run`` convention the other async tools' tests use
    (``workflow_deps``, ``upload_file``); the tool takes no arguments, so there
    is nothing to forward.
    """
    return asyncio.run(server.billing_status())


# A full, healthy `comfy cloud status` payload, shaped per comfy-cli's
# `cloud_status.json` schema. Values are illustrative, not a real account.
_STATUS = {
    "cloud_workspace": {
        "id": "ws_123",
        "name": "Personal",
        "type": "personal",
        "role": "owner",
    },
    "credit_balance_usd": 12.5,
    "credit_balance_credits": 2637,
    "effective_balance_micros": 12500000,
    "currency": "USD",
    "balance_confirmed": True,
    "subscription": {
        "tier": "PRO",
        "status": "active",
        "is_active": True,
        "plan_slug": "pro-monthly",
        "renewal_date": "2026-09-21",
        "cancel_at": None,
    },
    "billing_rail": "stripe",
    "max_concurrent_jobs": 8,
    "tier_default_concurrent_jobs": 8,
    "free_tier_balance": None,
    "upgrade_suggestion": {
        "plan_slug": "team-monthly",
        "price_usd": 100.0,
        "credits": 21100,
    },
    "blocked_upgrades": [],
    "seats": {"max": 1, "occupied": 1},
    "manage_url": "https://api.comfy.example/billing",
    "message": None,
}


def test_billing_status_argv(patched_async_run):
    """Passthrough: `comfy --json --where local cloud status`."""
    procs = patched_async_run(envelope(data=_STATUS))

    assert _billing_status() == _STATUS

    cmd = procs[0].cmd
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    # `cloud status` resolves its own cloud target internally, so the global
    # `--where local` is inert here — same as `cloud whoami` under `auth_status`.
    assert cmd[4:] == ["cloud", "status"]
    # Own process group, so one kill reaps the tree. The deadline itself is not
    # a spawn kwarg on this path — `_run_comfy_async` applies it with
    # `asyncio.wait_for`, which the timeout test below exercises end to end.
    assert procs[0].start_new_session is True


def test_billing_status_reads_an_envelope_larger_than_the_default_stdout_bound(
    patched_async_run,
):
    """A big payload survives: the call widens `_run_comfy_async`'s stdout cap.

    That runner keeps only the TRAILING `_STDERR_MAX_CHARS` of stdout by
    default — a bound meant for a multi-GB download's progress spew. An
    envelope clipped from the front loses its opening brace and comes back as
    "returned no JSON" instead of the payload, so a `cloud status` snapshot that
    outgrew 64 KiB would vanish. The schema is `additionalProperties: true` and
    already grows rows, so its size is the engine's to decide.
    """
    # Comfortably past the default bound, well inside the widened one.
    big = {**_STATUS, "blocked_upgrades": ["plan-%04d" % n for n in range(6000)]}
    assert len(str(big)) > server._STDERR_MAX_CHARS
    patched_async_run(envelope(data=big))

    assert _billing_status() == big


def test_billing_status_forwards_the_whole_payload_unfiltered(patched_async_run):
    """Every field comfy-cli emits reaches the caller, not a hand-picked subset.

    The schema declares `additionalProperties: true` and the command already
    grows rows (`seats`, `blocked_upgrades`, `free_tier_balance`), so projecting
    to a fixed field list here would silently drop whatever ships next — and
    would drop `balance_confirmed`, which is what stops a consumer rendering an
    unknown balance as $0.
    """
    patched_async_run(envelope(data=_STATUS))

    result = _billing_status()

    assert set(result) == set(_STATUS)
    for key, value in _STATUS.items():
        assert result[key] == value


def test_billing_status_relays_the_unconfirmed_copy_verbatim(patched_async_run):
    """The trust-guard case: null balance + comfy-cli's own neutral `message`.

    `balance_confirmed: false` means no balance figure could be ESTABLISHED, so
    the three balance fields are null rather than 0. The wording that explains
    that is comfy-cli's, derived from state this server cannot see (whether the
    balance call answered at all, whether the account is mid-migration) — so it
    has to survive the trip byte for byte.
    """
    copy = (
        "We could not confirm a balance for this account. If you have just "
        "signed up, check https://api.comfy.example/billing."
    )
    unconfirmed = {
        **_STATUS,
        "credit_balance_usd": None,
        "credit_balance_credits": None,
        "effective_balance_micros": None,
        "balance_confirmed": False,
        "message": copy,
    }
    patched_async_run(envelope(data=unconfirmed))

    result = _billing_status()

    assert result["message"] == copy
    assert result["balance_confirmed"] is False
    # Null, never coerced to a renderable zero.
    assert result["credit_balance_usd"] is None
    assert result["credit_balance_credits"] is None
    assert result["effective_balance_micros"] is None


def test_billing_status_degrades_without_the_verb(patched_async_run):
    """A comfy-cli predating `cloud status` reads as a version gap, not a break.

    Today this is the common path, not the rare one: the verb landed in
    comfy-cli after its most recent release. Relaying Click's raw usage dump
    would read as a broken MCP server, so it degrades to the `unsupported`
    shape `_freshness_report` established and names the upgrade.
    """
    patched_async_run(
        "",
        returncode=2,
        stderr="Usage: comfy cloud [OPTIONS] COMMAND\nNo such command 'status'.",
    )

    result = _billing_status()

    assert result["unsupported"] is True
    assert "billing status unavailable" in result["error"]
    # Actionable: names both the MCP tool and the terminal command.
    assert 'update_comfyui(target="cli")' in result["error"]
    assert "comfy update cli" in result["error"]
    # Says what still works instead of dead-ending on the denial.
    assert "auth_status" in result["error"]
    # None of the raw wrapper/CLI text leaks through.
    assert "No such command" not in result["error"]
    assert "Usage: comfy" not in result["error"]
    assert "returned no JSON" not in result["error"]


def test_billing_status_degrade_carries_the_balance_verdict(patched_async_run):
    """`balance_confirmed` travels with the degrade rather than going missing.

    The docstring tells consumers to read `balance_confirmed` before rendering
    any figure, so a degrade that omitted it would hand a caller that believed
    it a `KeyError` — or, on the `.get(..., 0)` spelling, exactly the $0 the
    field exists to prevent. `_download_verb_unsupported` set the precedent of
    shipping the promised verdict keys with the capability gap.
    """
    patched_async_run(
        "",
        returncode=2,
        stderr="Usage: comfy cloud [OPTIONS] COMMAND\nNo such command 'status'.",
    )

    result = _billing_status()

    assert result["balance_confirmed"] is False
    # The fields the verdict is ABOUT come with it, so a consumer implementing
    # the docstring's "false means the balance fields are null" invariant by
    # subscript does not hit a `KeyError` on the one shape where the verdict is
    # always false. `None` invents nothing — it is precisely "no figure".
    assert result["credit_balance_usd"] is None
    assert result["credit_balance_credits"] is None
    assert result["effective_balance_micros"] is None


def test_billing_status_degrade_does_not_push_an_unprompted_pip_upgrade(
    patched_async_run,
):
    """The upgrade pointer leads with the terminal command, not the MCP tool.

    `update_comfyui(target="cli")` pip-upgrades the user's Python environment
    and — unlike `target="all"` — raises no elicitation, so an agent that acted
    on it would turn a read-only balance query into an unconfirmed mutation.
    The tool is still named (dead-ending helps nobody), but as the option that
    needs the user's say-so.
    """
    patched_async_run(
        "",
        returncode=2,
        stderr="Usage: comfy cloud [OPTIONS] COMMAND\nNo such command 'status'.",
    )

    error = _billing_status()["error"]

    assert "`comfy update cli` in a terminal" in error
    assert 'update_comfyui(target="cli")' in error
    # The tool is qualified by who owns the decision, not offered bare.
    assert "ask" in error and "no confirmation prompt" in error
    assert error.index("comfy update cli") < error.index("update_comfyui")


def test_billing_status_degrades_when_the_whole_cloud_group_is_missing(
    patched_async_run,
):
    """A comfy-cli predating `comfy cloud` names `cloud`, not `status`.

    Both shapes of this version gap are the same gap with the same fix, and the
    `_MIN_COMFY_CLI` floor guard fails OPEN — a source build whose `--version`
    cannot be parsed reaches this tool from below the floor. Reading only the
    leaf verb would hand that caller the raw usage dump the degrade exists to
    hide, plus a message about the wrong missing command.
    """
    patched_async_run(
        "",
        returncode=2,
        stderr="Usage: comfy [OPTIONS] COMMAND\nNo such command 'cloud'.",
    )

    result = _billing_status()

    assert result["unsupported"] is True
    assert result["balance_confirmed"] is False
    assert "billing status unavailable" in result["error"]
    assert "No such command" not in result["error"]
    # The fallback pointer follows the gap: a CLI with no `comfy cloud` group
    # cannot answer `cloud whoami` either, so steering to `auth_status` here
    # would just buy the user a second usage dump.
    assert "no `comfy cloud` group at all" in result["error"]
    assert "`auth_status` still reports" not in result["error"]


def test_billing_status_unrelated_missing_verb_still_raises(patched_async_run):
    """Widening to `cloud` must not widen to ANY missing command.

    The degrade asserts nothing is broken, so it stays keyed to the two names
    this call actually spells — a usage error naming some third command is a
    failure the caller needs to see raw.
    """
    patched_async_run(
        "",
        returncode=2,
        stderr="Usage: comfy [OPTIONS] COMMAND\nNo such command 'wheelbarrow'.",
    )

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        _billing_status()


def test_billing_status_timeout_kills_the_child(patched_async_run, monkeypatch):
    """The long budget runs on `_run_comfy_async`, which reaps the process tree.

    A sync `def` tool would be off-loaded to `anyio.to_thread`, where neither a
    deadline nor a client disconnect can reach the `comfy` child: it would keep
    the pool worker and the child alive for the whole budget. Spawning through
    the async runner is what makes the kill possible, so assert the kill rather
    than the wiring.
    """
    monkeypatch.setattr(server, "_BILLING_STATUS_TIMEOUT", 0.05)
    procs = patched_async_run(envelope(data=_STATUS), hang=True)

    with pytest.raises(server.ComfyCliError, match="timed out"):
        _billing_status()

    assert procs[0].killed is True


def test_billing_status_cancellation_kills_the_child(patched_async_run):
    """A client that gives up mid-fan-out does not orphan the `comfy` child.

    This is the case the thread pool cannot serve at all — `asyncio.to_thread`
    swallows the cancellation and the child runs on unattended. Here the task is
    cancelled while the fake child is still "talking to the cloud", and the
    runner's `finally` reaps it.
    """

    async def drive():
        # Set up inside the loop so the spawn signal is an `asyncio.Event`
        # bound to it, rather than a poll loop.
        spawned = asyncio.Event()
        procs = patched_async_run(
            envelope(data=_STATUS), hang=True, on_spawn=lambda _cmd: spawned.set()
        )
        task = asyncio.ensure_future(server.billing_status())
        # Pull the rug only once the child is actually running.
        await asyncio.wait_for(spawned.wait(), timeout=5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return procs

    procs = asyncio.run(drive())

    assert procs[0].killed is True


def test_billing_status_degrade_names_no_version_floor(patched_async_run):
    """The upgrade message must not pin a release number that would go stale.

    `cloud status` is not yet in a tagged comfy-cli, so there is no shipped
    version to name — and inventing one would tell users to upgrade to a
    release that does not carry the verb. The install either has it or does not.
    """
    patched_async_run(
        "",
        returncode=2,
        stderr="Usage: comfy cloud [OPTIONS] COMMAND\nNo such command 'status'.",
    )

    error = _billing_status()["error"]

    assert "1.1" not in error  # no `1.14.0`-shaped floor smuggled in
    assert server._MIN_COMFY_CLI_STR not in error


def test_billing_status_missing_verb_match_is_case_insensitive(patched_async_run):
    """`Error: no such command 'status'` (lowercase) also degrades cleanly."""
    patched_async_run("", returncode=2, stderr="Error: no such command 'status'.")

    assert _billing_status()["unsupported"] is True


def test_billing_status_signed_out_keeps_comfy_clis_own_error(patched_async_run):
    """No credential is a real, actionable verdict — never the version degrade.

    comfy-cli fails before it makes a request in this case, so the code and the
    hint it raises are the whole diagnosis. Waving it through as `unsupported`
    would tell a signed-out user to upgrade their CLI.
    """
    patched_async_run(
        envelope(
            ok=False,
            error={
                "code": "cloud_not_configured",
                "message": "No Comfy Cloud credential found.",
                "hint": "run `comfy cloud login`",
            },
        ),
        returncode=1,
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        _billing_status()

    assert "cloud_not_configured" in str(excinfo.value)
    assert "comfy cloud login" in str(excinfo.value)  # comfy-cli's hint survives


def test_billing_status_billing_endpoint_failure_stays_raw(patched_async_run):
    """An unreachable billing endpoint is a retry, not an upgrade.

    `cloud_billing_unavailable` is the CLI's own fatal row — every other
    endpoint it calls degrades one field instead. The agent needs that code and
    hint to know to retry rather than to reinstall anything.
    """
    patched_async_run(
        envelope(
            ok=False,
            error={
                "code": "cloud_billing_unavailable",
                "message": "HTTP 503 from https://api.comfy.example/api/billing/status",
                "hint": "retry shortly; if it persists, contact Comfy support",
            },
        ),
        returncode=1,
    )

    with pytest.raises(server.ComfyCliError, match="cloud_billing_unavailable"):
        _billing_status()


def test_billing_status_relayed_phrase_is_not_unsupported(patched_async_run):
    """A failure that merely QUOTES the phrase, inside an envelope, stays raw.

    `clitext._is_missing_verb_error` requires the no-envelope + usage-exit pair
    exactly so a nested error relaying "No such command 'status'" from
    somewhere else cannot be mistaken for the verb itself being absent.
    """
    patched_async_run(
        envelope(
            ok=False,
            error={
                "code": "cloud_billing_unavailable",
                "message": "upstream said: No such command 'status'.",
            },
        ),
        returncode=2,
    )

    with pytest.raises(server.ComfyCliError, match="cloud_billing_unavailable"):
        _billing_status()


def test_billing_status_malformed_envelope_raises_rather_than_degrading(
    patched_async_run,
):
    """Garbled stdout is a broken engine, not a missing verb.

    A crash mid-run can exit non-zero with no envelope too, so the degrade is
    gated on Click's usage exit as well. Anything else keeps the wrapper's
    "returned no JSON" diagnosis, which carries both captured streams — the only
    thing that explains what actually happened.
    """
    patched_async_run(
        "{not json at all", returncode=1, stderr="Traceback (most recent call last)"
    )

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        _billing_status()


def test_billing_status_non_envelope_json_is_not_a_result(patched_async_run):
    """A stray JSON line on stdout must not be unwrapped as the payload."""
    patched_async_run(
        '{"ok": true, "data": {"credit_balance_usd": 999.0}}', returncode=0
    )

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        _billing_status()


def test_billing_status_is_read_only(patched_async_run):
    """No spend gate: no `confirm_spend` argument, no elicitation, one spawn.

    The read-only framing is part of the tool's contract — an agent should be
    able to answer "can I afford this run" without a consent prompt standing in
    the way — so it is asserted rather than left to the docstring.
    """
    procs = patched_async_run(envelope(data=_STATUS))

    _billing_status()

    assert list(inspect.signature(server.billing_status).parameters) == []
    assert len(procs) == 1  # no capability probe, no consent read
    doc = server.billing_status.__doc__ or ""
    assert "READ-ONLY" in doc and "spends nothing" in doc
