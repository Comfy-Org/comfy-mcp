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
"""

from __future__ import annotations

import inspect

import pytest
from conftest import envelope

from comfy_mcp import server

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


def test_billing_status_argv(patched_run):
    """Passthrough: `comfy --json --where local cloud status`."""
    calls = patched_run(envelope(data=_STATUS))

    assert server.billing_status() == _STATUS

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    # `cloud status` resolves its own cloud target internally, so the global
    # `--where local` is inert here — same as `cloud whoami` under `auth_status`.
    assert cmd[4:] == ["cloud", "status"]
    assert calls[0]["timeout"] == server._BILLING_STATUS_TIMEOUT


def test_billing_status_forwards_the_whole_payload_unfiltered(patched_run):
    """Every field comfy-cli emits reaches the caller, not a hand-picked subset.

    The schema declares `additionalProperties: true` and the command already
    grows rows (`seats`, `blocked_upgrades`, `free_tier_balance`), so projecting
    to a fixed field list here would silently drop whatever ships next — and
    would drop `balance_confirmed`, which is what stops a consumer rendering an
    unknown balance as $0.
    """
    patched_run(envelope(data=_STATUS))

    result = server.billing_status()

    assert set(result) == set(_STATUS)
    for key, value in _STATUS.items():
        assert result[key] == value


def test_billing_status_relays_the_unconfirmed_copy_verbatim(patched_run):
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
    patched_run(envelope(data=unconfirmed))

    result = server.billing_status()

    assert result["message"] == copy
    assert result["balance_confirmed"] is False
    # Null, never coerced to a renderable zero.
    assert result["credit_balance_usd"] is None
    assert result["credit_balance_credits"] is None
    assert result["effective_balance_micros"] is None


def test_billing_status_degrades_without_the_verb(patched_run):
    """A comfy-cli predating `cloud status` reads as a version gap, not a break.

    Today this is the common path, not the rare one: the verb landed in
    comfy-cli after its most recent release. Relaying Click's raw usage dump
    would read as a broken MCP server, so it degrades to the `unsupported`
    shape `_freshness_report` established and names the upgrade.
    """
    patched_run(
        "",
        returncode=2,
        stderr="Usage: comfy cloud [OPTIONS] COMMAND\nNo such command 'status'.",
    )

    result = server.billing_status()

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


def test_billing_status_degrade_names_no_version_floor(patched_run):
    """The upgrade message must not pin a release number that would go stale.

    `cloud status` is not yet in a tagged comfy-cli, so there is no shipped
    version to name — and inventing one would tell users to upgrade to a
    release that does not carry the verb. The install either has it or does not.
    """
    patched_run(
        "",
        returncode=2,
        stderr="Usage: comfy cloud [OPTIONS] COMMAND\nNo such command 'status'.",
    )

    error = server.billing_status()["error"]

    assert "1.1" not in error  # no `1.14.0`-shaped floor smuggled in
    assert server._MIN_COMFY_CLI_STR not in error


def test_billing_status_missing_verb_match_is_case_insensitive(patched_run):
    """`Error: no such command 'status'` (lowercase) also degrades cleanly."""
    patched_run("", returncode=2, stderr="Error: no such command 'status'.")

    assert server.billing_status()["unsupported"] is True


def test_billing_status_signed_out_keeps_comfy_clis_own_error(patched_run):
    """No credential is a real, actionable verdict — never the version degrade.

    comfy-cli fails before it makes a request in this case, so the code and the
    hint it raises are the whole diagnosis. Waving it through as `unsupported`
    would tell a signed-out user to upgrade their CLI.
    """
    patched_run(
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
        server.billing_status()

    assert "cloud_not_configured" in str(excinfo.value)
    assert "comfy cloud login" in str(excinfo.value)  # comfy-cli's hint survives


def test_billing_status_billing_endpoint_failure_stays_raw(patched_run):
    """An unreachable billing endpoint is a retry, not an upgrade.

    `cloud_billing_unavailable` is the CLI's own fatal row — every other
    endpoint it calls degrades one field instead. The agent needs that code and
    hint to know to retry rather than to reinstall anything.
    """
    patched_run(
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
        server.billing_status()


def test_billing_status_relayed_phrase_is_not_unsupported(patched_run):
    """A failure that merely QUOTES the phrase, inside an envelope, stays raw.

    `clitext._is_missing_verb_error` requires the no-envelope + usage-exit pair
    exactly so a nested error relaying "No such command 'status'" from
    somewhere else cannot be mistaken for the verb itself being absent.
    """
    patched_run(
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
        server.billing_status()


def test_billing_status_malformed_envelope_raises_rather_than_degrading(patched_run):
    """Garbled stdout is a broken engine, not a missing verb.

    A crash mid-run can exit non-zero with no envelope too, so the degrade is
    gated on Click's usage exit as well. Anything else keeps the wrapper's
    "returned no JSON" diagnosis, which carries both captured streams — the only
    thing that explains what actually happened.
    """
    patched_run(
        "{not json at all", returncode=1, stderr="Traceback (most recent call last)"
    )

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        server.billing_status()


def test_billing_status_non_envelope_json_is_not_a_result(patched_run):
    """A stray JSON line on stdout must not be unwrapped as the payload."""
    patched_run('{"ok": true, "data": {"credit_balance_usd": 999.0}}', returncode=0)

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        server.billing_status()


def test_billing_status_is_read_only(patched_run):
    """No spend gate: no `confirm_spend` argument, no elicitation, one spawn.

    The read-only framing is part of the tool's contract — an agent should be
    able to answer "can I afford this run" without a consent prompt standing in
    the way — so it is asserted rather than left to the docstring.
    """
    calls = patched_run(envelope(data=_STATUS))

    server.billing_status()

    assert list(inspect.signature(server.billing_status).parameters) == []
    assert len(calls) == 1  # no capability probe, no consent read
    doc = server.billing_status.__doc__ or ""
    assert "READ-ONLY" in doc and "spends nothing" in doc
