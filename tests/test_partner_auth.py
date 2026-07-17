"""Partner-API credential handling (BE-3344).

Two behaviors:
1. `_unwrap_envelope` must preserve `error.code` on `ComfyCliError.code` and
   append `error.hint` verbatim + the useful `error.details` keys (at minimum
   `partner_nodes`) to the raised message — the exact workaround testers needed
   used to be silently dropped.
2. `run_workflow` retries a bounded number of times on a *transient credential*
   code, and ONLY on codes that comfy-cli raises pre-submission on the local
   path (so a retry can't double-submit). `transient_auth` is deliberately not
   retried: on the local path it is raised from the execution watcher AFTER
   submission.
"""

from __future__ import annotations

import asyncio

import pytest

from comfy_local_mcp import server


def _partner_error_envelope() -> dict:
    """A fabricated `partner_node_requires_credential` error envelope."""
    return {
        "type": "envelope",
        "ok": False,
        "error": {
            "code": "partner_node_requires_credential",
            "message": "Workflow uses partner-API node(s) that need a credential.",
            "hint": (
                "re-submit with `--where cloud` (the CLI auto-injects the key "
                "there), or store the key locally with "
                "`comfy auth set comfy-cloud-api-key --key …`"
            ),
            "details": {
                "partner_nodes": ["SeedreamNode", "KlingNode"],
                "host": "127.0.0.1",
                "port": 8188,
            },
        },
    }


# --- envelope fidelity ------------------------------------------------------


def test_error_envelope_preserves_code_hint_and_partner_nodes():
    """`.code` is set, and both the hint and `partner_nodes` survive to the message."""
    with pytest.raises(server.ComfyCliError) as excinfo:
        server._unwrap_envelope(
            _partner_error_envelope(), ("run", "--workflow", "wf.json"), 1, ""
        )

    err = excinfo.value
    assert err.code == "partner_node_requires_credential"  # structured attribute

    msg = str(err)
    assert "[partner_node_requires_credential]" in msg
    assert "comfy auth set comfy-cloud-api-key" in msg  # hint verbatim
    assert "partner_nodes: SeedreamNode, KlingNode" in msg  # details rendered
    # Noisy detail keys are not surfaced (only the allow-listed ones).
    assert "127.0.0.1" not in msg


def test_local_wrapper_error_has_no_code():
    """A failure the wrapper raises itself (no envelope) carries `code is None`."""
    with pytest.raises(server.ComfyCliError) as excinfo:
        server._unwrap_envelope(None, ("env",), 1, "boom")
    assert excinfo.value.code is None


def test_error_without_hint_or_details_still_raises_cleanly():
    """A bare error envelope (no hint/details) formats without spurious blank lines."""
    envelope = {
        "type": "envelope",
        "ok": False,
        "error": {"code": "server_not_running", "message": "ComfyUI not running"},
    }
    with pytest.raises(server.ComfyCliError) as excinfo:
        server._unwrap_envelope(envelope, ("env",), 1, "")
    msg = str(excinfo.value)
    assert msg == "comfy env failed [server_not_running]: ComfyUI not running"
    assert excinfo.value.code == "server_not_running"


# --- bounded retry ----------------------------------------------------------


@pytest.fixture
def no_sleep(monkeypatch):
    """Make the retry backoff instantaneous."""

    async def _fast(_seconds):
        return None

    monkeypatch.setattr(server.asyncio, "sleep", _fast)


def _cred_error(code: str = "partner_node_requires_credential") -> server.ComfyCliError:
    return server.ComfyCliError(
        f"comfy run --workflow wf.json failed [{code}]: needs a credential\n"
        "hint: store the key locally with `comfy auth set comfy-cloud-api-key --key …`",
        code=code,
    )


def test_retry_set_membership():
    """The retry set is exactly the pre-submission-safe codes (regression guard)."""
    assert "partner_node_requires_credential" in server._RETRYABLE_CREDENTIAL_CODES
    assert "cloud_unauthorized" in server._RETRYABLE_CREDENTIAL_CODES
    # transient_auth fires POST-submission on the local path -> never retried.
    assert "transient_auth" not in server._RETRYABLE_CREDENTIAL_CODES


def test_transient_then_success_succeeds_on_attempt_2(monkeypatch, no_sleep):
    """A single transient credential failure is retried and the retry succeeds."""
    calls = {"n": 0}

    async def fake_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _cred_error()
        return {"outputs": ["/x.png"]}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    result = asyncio.run(server.run_workflow("wf.json", wait=True))

    assert result == {"outputs": ["/x.png"]}
    assert calls["n"] == 2  # succeeded on the second attempt


def test_persistent_failure_surfaces_final_hint_error(monkeypatch, no_sleep):
    """Exhausted retries surface the hint-bearing error, noting the retries."""
    calls = {"n": 0}

    async def fake_stream(*args, **kwargs):
        calls["n"] += 1
        raise _cred_error()

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server.run_workflow("wf.json", wait=True))

    assert calls["n"] == 3  # 1 initial + 2 retries
    msg = str(excinfo.value)
    assert "comfy auth set comfy-cloud-api-key" in msg  # hint still present
    assert "gave up after 2 retries" in msg  # retries were noted
    assert excinfo.value.code == "partner_node_requires_credential"


def test_non_retryable_code_fails_immediately(monkeypatch, no_sleep):
    """A non-credential error is not retried."""
    calls = {"n": 0}

    async def fake_stream(*args, **kwargs):
        calls["n"] += 1
        raise server.ComfyCliError("boom", code="execution_error")

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    with pytest.raises(server.ComfyCliError, match="boom"):
        asyncio.run(server.run_workflow("wf.json", wait=True))

    assert calls["n"] == 1  # no retries


def test_transient_auth_is_not_retried(monkeypatch, no_sleep):
    """`transient_auth` fires post-submission on the local path -> retrying would double-submit."""
    calls = {"n": 0}

    async def fake_stream(*args, **kwargs):
        calls["n"] += 1
        raise server.ComfyCliError(
            "Unauthorized: Please login first to use this node", code="transient_auth"
        )

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    with pytest.raises(server.ComfyCliError, match="Unauthorized"):
        asyncio.run(server.run_workflow("wf.json", wait=True))

    assert calls["n"] == 1  # NOT retried


def test_retry_applies_to_wait_false_path(monkeypatch, no_sleep):
    """The bounded retry covers the async-submit (`wait=False`) path too."""
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _cred_error()
        return {"prompt_id": "p1"}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    result = asyncio.run(server.run_workflow("wf.json", wait=False))

    assert result == {"prompt_id": "p1"}
    assert calls["n"] == 2  # retried on the plain --json path as well


def test_instructions_mention_partner_credential():
    """The handshake instructions name the partner-credential env var."""
    assert "COMFY_API_KEY" in server.mcp.instructions
