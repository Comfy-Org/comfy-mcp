"""The startup `Machine snapshot` the handshake instructions carry.

The routing policy in ``instructions.INSTRUCTIONS`` keys on ``server_info``'s
``hardware`` block, but it only helps an agent that remembers to call the tool
before its first generation — and QA kept catching agents that do not, running
diffusion on machines the policy would have routed to partner/cloud. ``main()``
therefore probes ``comfy env`` once before serving and appends the block
verbatim to the instructions every client receives. These tests pin that
mechanism end to end: the probe's argv, the fail-open on every probe failure,
the UNKNOWN wording when a healthy payload has no ``hardware``, the
remote-target note, the scrubbing/redaction, and ``main()`` actually applying
it before ``mcp.run``.

FastMCP exposes ``mcp.instructions`` as the mutable application-level setting;
every assertion reads that public property so a framework change fails loudly.
"""

from __future__ import annotations

import threading
import time

import pytest
from conftest import envelope

from comfy_mcp.server import _internal as server

HARDWARE = {
    "os": "Darwin",
    "arch": "arm64",
    "ram_bytes": 68719476736,
    "gpu": {
        "vendor": "Apple",
        "model": "Apple M4 Max",
        "vram_bytes": None,
        "unified_memory": True,
    },
}


# Captured at import, BEFORE conftest's autouse ``_skip_machine_snapshot_probe``
# replaces the module attribute — the same re-enable pattern the version-guard
# tests use. Every test here runs the REAL probe against a stubbed spawn.
_REAL_SNAPSHOT_BLOCK = server._machine_snapshot_block


@pytest.fixture(autouse=True)
def _real_probe(monkeypatch):
    """Undo conftest's fail-open stub — these tests exist to exercise the probe."""
    monkeypatch.setattr(server, "_machine_snapshot_block", _REAL_SNAPSHOT_BLOCK)


@pytest.fixture(autouse=True)
def _restore_instructions():
    """Never leak a mutated handshake between tests — ``mcp`` is module-global."""
    original = server.mcp.instructions
    yield
    server.mcp.instructions = original


def test_snapshot_appends_hardware_to_the_handshake(patched_run):
    """The probe is one `comfy env` and its `hardware` rides `mcp.instructions`."""
    calls = patched_run(
        envelope(data={"server": {"running": False}, "hardware": HARDWARE})
    )

    server._apply_startup_instructions()

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]
    assert cmd[4:] == ["env"]
    assert len(calls) == 1  # no freshness probe — the snapshot is routing data only

    text = server.mcp.instructions
    # The static policy is retained in full, the snapshot appended after it.
    assert text.startswith(server.instructions.INSTRUCTIONS)
    tail = text[len(server.instructions.INSTRUCTIONS) :]
    assert server._SNAPSHOT_HEADER in tail
    # The block is comfy-cli's payload quoted verbatim, not a paraphrase.
    assert '"Apple M4 Max"' in tail
    assert '"unified_memory": true' in tail
    # The snapshot must not claim to be live server state.
    assert "still call `server_info`" in tail


def test_routing_block_points_at_the_snapshot():
    """The policy text names the snapshot, so an agent knows not to wait on a call.

    Same tripwire style as ``test_routing_instructions.py``: the mechanism is
    only reachable if the prose an agent actually reads mentions it.
    """
    flat = " ".join(server.instructions.INSTRUCTIONS.split())
    assert "A `Machine snapshot` section at the END of these instructions" in flat
    assert "the startup probe failed" in flat


@pytest.mark.parametrize(
    "setup_kwargs",
    [
        {"stdout": envelope(ok=False, error={"code": "boom", "message": "no"})},
        {"raises": OSError("no such binary")},
        {"stdout": "not json at all", "returncode": 1},
        {"stdout": envelope(data=None)},  # healthy envelope, non-dict payload
        {"stdout": envelope(data=[1, 2])},  # drifted payload shape
    ],
)
def test_probe_failure_falls_open_to_the_static_instructions(patched_run, setup_kwargs):
    """Every probe failure leaves the handshake exactly as it was.

    The static ``instructions.INSTRUCTIONS`` already tell the agent to call
    ``server_info`` first, so a failed probe must cost nothing — especially
    not startup.
    """
    patched_run(**setup_kwargs)

    server._apply_startup_instructions()

    assert server.mcp.instructions == server.instructions.INSTRUCTIONS


def test_wedged_probe_is_bounded_by_wall_clock_not_inner_timeouts(monkeypatch):
    """A probe that hangs longer than the budget must not stall the handshake.

    The inner subprocess timeouts do not compose into a startup bound: in a
    fresh process the once-per-process ``comfy --version`` guard (30s) runs
    before the env probe (15s), so a binary that hangs rather than errors
    holds the sum. The bounded thread join is the actual cap; this pins it by
    handing ``_apply_startup_instructions`` a probe that outlives the (shrunk)
    budget and asserting the handshake ships unchanged, promptly.
    """
    release = threading.Event()
    monkeypatch.setattr(server, "_SNAPSHOT_PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(
        server, "_machine_snapshot_block", lambda: release.wait(10) or "TOO LATE"
    )

    started = time.monotonic()
    server._apply_startup_instructions()
    elapsed = time.monotonic() - started

    try:
        assert elapsed < 5, f"join did not bound the stall: {elapsed:.1f}s"
        assert server.mcp.instructions == server.instructions.INSTRUCTIONS
    finally:
        release.set()  # let the daemon probe thread finish before teardown


def test_oversized_hardware_payload_falls_open(patched_run):
    """A `hardware` block past the size cap drops the snapshot entirely.

    The payload is another program's output quoted into every conversation's
    context; a drifted or hostile `comfy env` must not be able to bloat the
    handshake. Truncated JSON would be worse than none, so the whole section
    falls open to the static instructions.
    """
    bloated = dict(HARDWARE, notes="x" * (server._SNAPSHOT_MAX_HARDWARE_CHARS + 1))
    patched_run(envelope(data={"hardware": bloated}))

    server._apply_startup_instructions()

    assert server.mcp.instructions == server.instructions.INSTRUCTIONS


def test_missing_hardware_is_stated_as_unknown_not_omitted(patched_run):
    """A healthy `comfy env` with no `hardware` key still produces a snapshot.

    Silence would read as "probe failed, call server_info" and cost the agent a
    round trip that reports the same absence; the snapshot instead states the
    STEP 3 verdict outright: UNKNOWN, ask the user.
    """
    patched_run(envelope(data={"server": {"running": True}}))

    server._apply_startup_instructions()

    tail = server.mcp.instructions[len(server.instructions.INSTRUCTIONS) :]
    assert server._SNAPSHOT_HEADER in tail
    assert "UNKNOWN" in tail
    assert "ask the user" in tail.lower()
    assert '"hardware"' not in tail


def test_remote_target_is_named_with_userinfo_redacted(patched_run, monkeypatch):
    """A configured remote rides the snapshot — host redacted, STEP 1 named.

    ``COMFYUI_HOST`` is taken verbatim by ``target._comfy_target``, so a
    URL-style value carries userinfo; the snapshot is quoted into every
    conversation and must mask it exactly as ``server_info`` does.
    """
    monkeypatch.setenv("COMFYUI_HOST", "<user>:<s3cret>@gpu-box")
    patched_run(envelope(data={"hardware": HARDWARE}))

    server._apply_startup_instructions()

    tail = server.mcp.instructions[len(server.instructions.INSTRUCTIONS) :]
    assert "***@gpu-box:8188" in tail
    assert "s3cret" not in tail
    assert "STEP 1" in tail
    assert "THIS machine" in tail


def test_no_target_configured_means_no_target_sentence(patched_run):
    """The local default adds no remote-target prose to confuse STEP 1."""
    patched_run(envelope(data={"hardware": HARDWARE}))

    server._apply_startup_instructions()

    tail = server.mcp.instructions[len(server.instructions.INSTRUCTIONS) :]
    assert "remote ComfyUI target is configured" not in tail


def test_malformed_target_surfaces_error_shaped_note(patched_run, monkeypatch):
    """A malformed COMFYUI_URL must not read as "nothing configured".

    Mirrors ``server_info``: the snapshot states the malformation (already
    userinfo-masked by ``target._comfy_target``'s own message) so the user
    hears about the typo before the first submit tool raises it.
    """
    monkeypatch.setenv("COMFYUI_URL", "http://host:not-a-port")
    patched_run(envelope(data={"hardware": HARDWARE}))

    server._apply_startup_instructions()

    tail = server.mcp.instructions[len(server.instructions.INSTRUCTIONS) :]
    assert "MALFORMED" in tail
    assert "no remote is resolved" in tail


def test_snapshot_dump_is_credential_scrubbed(patched_run):
    """The rendered payload passes through `failure_log._scrub_text`.

    Hardware fields should never carry a credential URL, but the payload is
    another program's output and the snapshot is quoted into every
    conversation — same standing as the validate relay's masking.
    """
    tainted = dict(HARDWARE, os="see https://<user>:<tok3n>@example.com/x")
    patched_run(envelope(data={"hardware": tainted}))

    server._apply_startup_instructions()

    tail = server.mcp.instructions[len(server.instructions.INSTRUCTIONS) :]
    assert "tok3n" not in tail
    assert "example.com" in tail  # the URL survives, only the secret is masked


def test_main_applies_the_snapshot_before_serving(monkeypatch):
    """``main()`` enriches the handshake and only then hands off to ``mcp.run``."""
    order: list[str] = []
    monkeypatch.setattr(
        server, "_machine_snapshot_block", lambda: order.append("probe") or "SNAP"
    )

    def fake_run(*, transport, show_banner):
        order.append(f"run:{transport}")
        assert show_banner is False

    monkeypatch.setattr(server.mcp, "run", fake_run)

    server.main()

    assert order == ["probe", "run:stdio"]
    assert server.mcp.instructions == f"{server.instructions.INSTRUCTIONS}\nSNAP\n"
