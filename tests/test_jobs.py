"""Tests for the grouped ``job(action=...)`` tool.

Consolidates six former tools — ``job_status`` / ``wait_for_job`` /
``watch_job`` / ``get_execution_error`` / ``cancel_job`` / ``get_queue`` —
into one ``action`` enum. Every per-action body below is the exact body the
tool it replaced ran (see ``_job_status_sync`` / ``_job_error_sync`` /
``_job_wait_sync`` / ``_job_cancel_sync`` / ``_job_queue_sync``, and the
``"watch"`` branch inline in ``job`` itself); these tests were moved from
``test_wrapper.py`` / ``test_diagnostics.py`` and adapted to the new call
shape (``asyncio.run(server.job(action=..., prompt_id=..., ...))`` in place
of a direct, synchronous call — ``job`` is ``async def`` because the
``"watch"`` action must stream on the event loop).

R2 is the load-bearing risk this file exists to pin: ``job`` is one
``async def`` tool, but five of its six actions are blocking (a
``subprocess`` call, or — for ``"wait"`` — a ``time.sleep`` poll loop that can
run up to an hour). Only ``"watch"`` is genuinely async (asyncio subprocess +
stream reads). Those five MUST be off-loaded to a worker thread via
``anyio.to_thread.run_sync`` (the same mechanism the MCP SDK itself uses to
run a plain ``def`` tool) or they would block the event loop for the whole
call — wedging every OTHER concurrent MCP request this server is handling.
``test_a_blocking_branch_does_not_wedge_the_event_loop`` proves that
concurrency guarantee directly, ahead of the thread-identity checks below it.
"""

from __future__ import annotations

import ast
import asyncio
import json
import threading
from pathlib import Path

import pytest
from conftest import _OK_STREAM, _RecordingCtx, envelope

from comfy_mcp.server import _internal as server

_SERVER_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "comfy_mcp"
    / "server"
    / "_internal.py"
)


# --- R2: the off-load proof ---------------------------------------------------


def test_a_blocking_branch_does_not_wedge_the_event_loop(monkeypatch):
    """While `job(action="status")`'s blocking call is parked in a worker
    thread, the event loop keeps servicing OTHER concurrent async work —
    the exact failure mode `anyio.to_thread.run_sync` exists to prevent.

    Without the off-load, `job` (a single `async def` coroutine) would run
    `_run_comfy`'s blocking `subprocess.Popen(...).communicate()` directly on
    the event-loop thread, and nothing else — no other tool call, no other
    client — could make progress until it returned.
    """
    entered = threading.Event()
    release = threading.Event()

    def fake_run(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5.0), "test deadlocked waiting for release"
        return {"status": "completed"}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    async def main():
        ticks: list[int] = []

        async def ticker() -> None:
            for _ in range(5):
                await asyncio.sleep(0.01)
                ticks.append(1)

        job_task = asyncio.create_task(server.job(action="status", prompt_id="pid"))
        # Wait (off-thread, so as not to itself block this loop) for the job
        # task to actually reach the blocking call.
        await asyncio.to_thread(entered.wait, 5.0)
        # If the event loop were wedged by job_task, this ticker would never
        # get a turn and `ticks` would still be empty when we release below.
        await ticker()
        assert len(ticks) == 5  # the loop kept running WHILE job_task blocked
        release.set()
        return await job_task

    result = asyncio.run(main())
    assert result == {"status": "completed"}


@pytest.mark.parametrize(
    ("action", "kwargs", "patch_target"),
    [
        ("status", {"prompt_id": "pid"}, "_run_comfy"),
        ("error", {"prompt_id": "pid"}, "_run_comfy"),
        ("cancel", {"prompt_id": "pid"}, "_run_comfy"),
        ("queue", {}, "_run_comfy"),
        ("wait", {"prompt_id": "pid"}, "_poll_until_terminal"),
    ],
)
def test_every_non_watch_action_runs_off_the_event_loop_thread(
    monkeypatch, action, kwargs, patch_target
):
    """Every sync branch executes on a DIFFERENT thread than the event loop.

    A weaker, faster-running companion to the concurrency proof above,
    parametrized across all five off-loaded actions (not just "status") so a
    regression that off-loads only one of them is still caught.
    """
    main_thread_ident = threading.get_ident()
    seen: dict[str, int] = {}

    def fake(*args, **kwargs2):
        seen["ident"] = threading.get_ident()
        if action == "wait":
            return {"status": "completed"}
        return {"status": "completed"} if action != "queue" else {"jobs": []}

    monkeypatch.setattr(server, patch_target, fake)

    asyncio.run(server.job(action=action, **kwargs))

    assert seen["ident"] != main_thread_ident


def test_watch_runs_on_the_event_loop_not_offloaded(monkeypatch):
    """The one action that must NOT be off-loaded: `_run_comfy_streaming` is
    already async (asyncio subprocess + stream reads), so wrapping it in
    `anyio.to_thread.run_sync` would be both unnecessary and wrong (you
    cannot `await` from inside a worker thread)."""
    seen: dict[str, int] = {}
    main_thread_ident = threading.get_ident()

    async def fake_stream(*args, ctx=None, timeout=None, raise_on_timeout=True):
        seen["ident"] = threading.get_ident()
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    asyncio.run(server.job(action="watch", prompt_id="pid"))

    assert seen["ident"] == main_thread_ident


# --- action enum: unknown action, default action ------------------------------


def test_job_rejects_an_unknown_action(monkeypatch):
    def fake_run(*args, **kwargs):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="invalid job action"):
        asyncio.run(server.job(action="bogus", prompt_id="pid"))


def test_job_bad_action_error_names_the_valid_ones(monkeypatch):
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {})

    with pytest.raises(
        server.ComfyCliError,
        match=r"'status'.*'error'.*'wait'.*'watch'.*'cancel'.*'queue'",
    ):
        asyncio.run(server.job(action="delete", prompt_id="pid"))


def test_job_default_action_is_status(patched_run):
    calls = patched_run(envelope(data={"status": "running"}))

    asyncio.run(server.job(prompt_id="pid"))

    assert calls[0]["cmd"][4:] == ["jobs", "status", "pid"]


# --- extraneous-parameter policy: REJECT LOUDLY --------------------------------


@pytest.mark.parametrize("action", ["status", "error", "wait", "watch", "cancel"])
def test_job_missing_required_prompt_id_is_rejected(monkeypatch, action):
    """Every action but "queue" requires `prompt_id`; a missing one is named
    by ACTION and PARAM rather than falling through to
    `argv._guard_prompt_id("")`'s generic "empty or leading '-'" message."""

    def fake_run(*args, **kwargs):
        raise AssertionError(f"{action} spawned comfy-cli with no prompt_id")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server, "_run_comfy_streaming", fake_run)

    with pytest.raises(server.ComfyCliError, match=f"action={action!r}.*prompt_id"):
        asyncio.run(server.job(action=action))


def test_job_queue_rejects_a_supplied_prompt_id(no_spawn):
    """`prompt_id` is supplied-but-ignored for "queue" — refused, not dropped."""
    with pytest.raises(
        server.ComfyCliError, match=r"job\(action='queue'\) does not take prompt_id"
    ) as excinfo:
        asyncio.run(server.job(action="queue", prompt_id="pid"))
    # Names every action that DOES consume it.
    for name in ("status", "error", "wait", "watch", "cancel"):
        assert f"'{name}'" in str(excinfo.value)


@pytest.mark.parametrize("action", ["status", "error", "cancel", "queue"])
def test_job_rejects_a_supplied_timeout_seconds_where_unused(no_spawn, action):
    """`timeout_seconds` is only for "wait"/"watch" — supplying it elsewhere
    is REJECT LOUDLY, not a silent no-op."""
    kwargs = {"prompt_id": "pid"} if action != "queue" else {}
    with pytest.raises(
        server.ComfyCliError,
        match=rf"job\(action={action!r}\) does not take timeout_seconds",
    ) as excinfo:
        asyncio.run(server.job(action=action, timeout_seconds=5.0, **kwargs))
    assert "'wait'" in str(excinfo.value)
    assert "'watch'" in str(excinfo.value)


# --- action="status" -----------------------------------------------------------


def test_job_status_maps_command_and_returns_data(patched_run):
    calls = patched_run(envelope(data={"status": "completed", "outputs": ["/x.png"]}))

    result = asyncio.run(server.job(action="status", prompt_id="pid"))

    assert result == {"status": "completed", "outputs": ["/x.png"]}
    assert calls[0]["cmd"][4:] == ["jobs", "status", "pid"]


# --- action="error" (the extracted `_execution_error_verdict` body) -----------


def test_job_error_extracts_fields_and_caps_traceback(monkeypatch):
    """A failed status → flat fields extracted and the traceback tailed + capped."""
    frames = [f"frame {i}: " + "x" * 500 for i in range(50)]
    status = {
        "status": "error",
        "error": {
            "exception_message": "Tensor size mismatch",
            "exception_type": "RuntimeError",
            "node_id": 7,  # server may send an int; contract coerces to str
            "node_type": "KSampler",
            "traceback": frames,
        },
    }
    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return status

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    result = asyncio.run(server.job(action="error", prompt_id="pid-123"))

    assert calls[0] == ("jobs", "status", "pid-123")
    assert result["prompt_id"] == "pid-123"
    assert result["status"] == "error"
    assert result["exception_message"] == "Tensor size mismatch"
    assert result["exception_type"] == "RuntimeError"
    assert result["node_id"] == "7"
    assert result["node_type"] == "KSampler"

    tail = result["traceback_tail"]
    assert len(tail) <= server._TRACEBACK_TAIL_FRAMES + 1
    assert tail[0] == "...(truncated)"
    joined = "\n".join(tail[1:])
    assert len(joined) <= server._TRACEBACK_TAIL_MAX_CHARS
    assert tail[-1] == frames[-1]


def test_job_error_small_traceback_is_not_marked(monkeypatch):
    status = {
        "status": "error",
        "error": {
            "exception_message": "boom",
            "exception_type": "ValueError",
            "node_id": None,
            "node_type": "LoadImage",
            "traceback": "single frame string",
        },
    }
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: status)

    result = asyncio.run(server.job(action="error", prompt_id="pid"))

    assert result["node_id"] is None
    assert result["traceback_tail"] == ["single frame string"]


def test_job_error_tolerates_non_dict_error(monkeypatch):
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {"status": "error", "error": "raw failure text"},
    )

    result = asyncio.run(server.job(action="error", prompt_id="pid"))

    assert result["status"] == "error"
    assert result["exception_message"] == "raw failure text"
    assert result["node_id"] is None
    assert result["traceback_tail"] == []


def test_job_error_preserves_wrapper_verdict(patched_run):
    patched_run(
        envelope(
            data={
                "status": "error",
                "error": {
                    "code": "server_died",
                    "message": "Lost connection to ComfyUI while job pid was running",
                    "details": {"last_node": "KSampler"},
                },
            }
        )
    )

    result = asyncio.run(server.job(action="error", prompt_id="pid"))

    assert result["error_code"] == "server_died"
    assert (
        result["exception_message"]
        == "Lost connection to ComfyUI while job pid was running"
    )
    assert result["exception_type"] is None
    assert result["node_id"] is None
    assert result["node_type"] is None
    assert result["traceback_tail"] == []


def test_job_error_node_failure_reports_error_code_none(patched_run):
    patched_run(
        envelope(
            data={
                "status": "error",
                "error": {
                    "exception_message": "Tensor size mismatch",
                    "exception_type": "RuntimeError",
                    "node_id": 7,
                    "node_type": "KSampler",
                    "traceback": ["frame one", "frame two"],
                },
            }
        )
    )

    assert asyncio.run(server.job(action="error", prompt_id="pid")) == {
        "prompt_id": "pid",
        "status": "error",
        "error_code": None,
        "exception_message": "Tensor size mismatch",
        "exception_type": "RuntimeError",
        "node_id": "7",
        "node_type": "KSampler",
        "traceback_tail": ["frame one", "frame two"],
    }


def test_job_error_node_fields_win_over_wrapper_message(patched_run):
    patched_run(
        envelope(
            data={
                "status": "error",
                "error": {
                    "code": "execution_error",
                    "message": "the job failed",
                    "exception_message": "Tensor size mismatch",
                    "exception_type": "RuntimeError",
                    "node_id": 7,
                    "node_type": "KSampler",
                    "traceback": ["frame one"],
                },
            }
        )
    )

    result = asyncio.run(server.job(action="error", prompt_id="pid"))

    assert result["error_code"] == "execution_error"
    assert result["exception_message"] == "Tensor size mismatch"
    assert result["exception_type"] == "RuntimeError"
    assert result["node_id"] == "7"
    assert result["node_type"] == "KSampler"
    assert result["traceback_tail"] == ["frame one"]


def test_job_error_no_error_returns_explicit_none(monkeypatch):
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {"status": "completed", "outputs": ["/tmp/gen.png"]},
    )

    result = asyncio.run(server.job(action="error", prompt_id="pid"))

    assert result == {"prompt_id": "pid", "status": "completed", "error": None}


def test_job_error_status_with_empty_payload_is_not_healthy(monkeypatch):
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {"status": "error", "error": {}},
    )

    result = asyncio.run(server.job(action="error", prompt_id="pid"))

    assert result["status"] == "error"
    assert "error" not in result
    assert result["exception_message"] is None
    assert result["traceback_tail"] == []


def test_job_error_tolerates_non_sequence_traceback(monkeypatch):
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {
            "status": "error",
            "error": {
                "exception_message": "boom",
                "traceback": {"unexpected": "shape"},
            },
        },
    )

    result = asyncio.run(server.job(action="error", prompt_id="pid"))

    assert result["exception_message"] == "boom"
    assert result["traceback_tail"] == []


def test_job_error_caps_oversized_exception_message(monkeypatch):
    huge = "z" * (server._EXCEPTION_TEXT_MAX_CHARS + 10_000)
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {"status": "error", "error": {"exception_message": huge}},
    )

    result = asyncio.run(server.job(action="error", prompt_id="pid"))

    msg = result["exception_message"]
    assert len(msg) <= server._EXCEPTION_TEXT_MAX_CHARS + len("...(truncated)")
    assert msg.endswith("...(truncated)")


def test_job_error_rejects_leading_dash(monkeypatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        asyncio.run(server.job(action="error", prompt_id="-rf"))
    assert not called


def test_job_error_rejects_embedded_nul(monkeypatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        asyncio.run(server.job(action="error", prompt_id="abc\0"))
    assert not called


def test_cap_traceback_tail_hard_truncates_single_oversized_frame():
    frame = "y" * (server._TRACEBACK_TAIL_MAX_CHARS + 500)
    capped = server._cap_traceback_tail([frame])

    assert capped[0] == "...(truncated)"
    assert len(capped) == 2
    joined = "\n".join(capped)
    assert len(joined) <= server._TRACEBACK_TAIL_MAX_CHARS
    assert capped[1] == frame[-len(capped[1]) :]


# --- action="wait" --------------------------------------------------------------


def test_job_wait_returns_terminal_status(monkeypatch):
    statuses = iter(
        [
            {"status": "running"},
            {"status": "running"},
            {"status": "completed", "outputs": ["/tmp/gen.png"]},
        ]
    )
    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return next(statuses)

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    result = asyncio.run(
        server.job(action="wait", prompt_id="pid", timeout_seconds=25.0)
    )

    assert result == {"status": "completed", "outputs": ["/tmp/gen.png"]}
    assert calls[0] == ("jobs", "status", "pid")
    assert len(calls) == 3


def test_job_wait_times_out_cleanly(monkeypatch):
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"status": "running"})
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    clock = {"t": 0.0}

    def fake_monotonic():
        now = clock["t"]
        clock["t"] += 10.0
        return now

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = asyncio.run(
        server.job(action="wait", prompt_id="pid", timeout_seconds=25.0)
    )

    assert result == {"timed_out": True, "status": {"status": "running"}}


# --- R1: default timeouts (wait=25.0, watch=600.0) ------------------------------


def test_job_wait_default_timeout_is_25_seconds(monkeypatch):
    seen: dict = {}

    def fake_poll(*args, **kwargs):
        seen["timeout_seconds"] = kwargs["timeout_seconds"]
        return {"status": "completed"}

    monkeypatch.setattr(server, "_poll_until_terminal", fake_poll)

    asyncio.run(server.job(action="wait", prompt_id="pid"))

    assert seen["timeout_seconds"] == 25.0


def test_job_watch_default_timeout_is_600_seconds(monkeypatch):
    seen: dict = {}

    async def fake_stream(*args, ctx=None, timeout=None, raise_on_timeout=True):
        seen["timeout"] = timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    asyncio.run(server.job(action="watch", prompt_id="pid"))

    assert seen["timeout"] == 600.0


@pytest.mark.parametrize("oversized", [float("inf"), 86_400.0])
def test_job_wait_clamps_an_oversized_timeout(monkeypatch, oversized):
    """An oversized/infinite bound is clamped to `_MAX_WATCH_TIMEOUT` before it
    ever reaches the poll loop.

    Proven at the `_poll_until_terminal` call boundary (mocked out) rather
    than by feeding the loop a scripted `time.monotonic()` sequence, the way
    `wait_for_job`'s equivalent test did: `job`'s "wait" branch now runs
    `_job_wait_sync` inside a REAL `anyio.to_thread.run_sync` worker thread
    (R2), and that thread hand-off itself reads `time.monotonic()` some
    unpredictable number of times. `time.monotonic` is a process-global
    function — `monkeypatch.setattr(server.time, "monotonic", ...)` patches
    the actual `time` module, not a copy — so a fake that stops advancing
    (either a fixed final value via `next(reads, X)`, or `StopIteration`
    aborting the sequence) livelocks the anyio thread hand-off itself: it
    never observes time moving forward and never signals completion back to
    this coroutine. Confirmed by reproducing the hang while writing this
    test, not merely asserted; the fix is to never freeze the CLOCK under
    this branch — assert the CLAMP at the boundary instead.
    """
    seen: dict = {}

    def fake_poll(*args, **kwargs):
        seen["timeout_seconds"] = kwargs["timeout_seconds"]
        return {"status": "running"}

    monkeypatch.setattr(server, "_poll_until_terminal", fake_poll)

    asyncio.run(server.job(action="wait", prompt_id="pid", timeout_seconds=oversized))

    assert seen["timeout_seconds"] == server._MAX_WATCH_TIMEOUT


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_job_wait_rejects_a_non_positive_or_nan_timeout(monkeypatch, bad):
    def fake_run(*args, **kwargs):
        raise AssertionError("job(action='wait') polled with an invalid timeout")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        asyncio.run(server.job(action="wait", prompt_id="pid", timeout_seconds=bad))


def test_poll_until_terminal_always_polls_at_least_once(monkeypatch):
    """A bound that expires before the first poll still reports a real status.

    This is `_poll_until_terminal`'s own invariant (shared by `job(action=
    "wait")` and `_poll_download`), so it is exercised by calling the helper
    DIRECTLY — plain, synchronous, no `job()` / anyio thread hand-off
    involved — rather than through `job(action="wait")`. Two reasons: (1) the
    invariant belongs to the helper, not to `job`'s dispatch, so this is the
    more precise unit test regardless of threading; (2) a scripted, exhausting
    `time.monotonic()` fake like the one this test used to run through
    `wait_for_job` freezes once exhausted, and freezing the global clock
    livelocks the REAL worker thread `job(action="wait")` now hands this call
    to (see `test_job_wait_clamps_an_oversized_timeout`'s docstring for the
    confirmed repro) — calling the sync helper directly sidesteps that
    entirely, since no second thread is ever involved here.
    """
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"status": "running"}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    # A clock that is already past the deadline by the loop's first read: the
    # bound is set at t=0 and every later read is t=1 — safe here because
    # nothing but this test's own thread ever calls `time.monotonic()`.
    reads = iter([0.0])

    def fake_monotonic():
        return next(reads, 1.0)

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = server._poll_until_terminal(
        "jobs",
        "status",
        "pid",
        timeout_seconds=1e-9,
        is_terminal=server._is_terminal,
    )

    assert calls == 1
    assert result == {"timed_out": True, "status": {"status": "running"}}


# The remaining `_poll_until_terminal` timing-edge-case tests below are ported
# the same way and for the same reason (direct calls, ever-advancing or
# single-shot clocks only — never a fake that stops advancing; see this
# module's docstring and `test_job_wait_clamps_an_oversized_timeout` above).


def test_poll_until_terminal_caps_each_poll_to_the_remaining_bound(monkeypatch):
    """A single poll never gets a longer subprocess budget than the wait itself.

    Each `comfy jobs status` used a fixed 60s timeout, so a short wait was only
    bounded *between* polls: one wedged status call could hold a
    `timeout_seconds=1` wait open for a full minute.
    """
    seen: list[float] = []

    def fake_run(*args, timeout=None, **kwargs):
        seen.append(timeout)
        return {"status": "running"}

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # A clock that advances 1s per read, so the 5s bound expires after two polls.
    clock = {"t": 0.0}

    def fake_monotonic():
        now = clock["t"]
        clock["t"] += 1.0
        return now

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    result = server._poll_until_terminal(
        "jobs", "status", "pid", timeout_seconds=5.0, is_terminal=server._is_terminal
    )

    assert result == {"timed_out": True, "status": {"status": "running"}}
    assert seen == [4.0, 2.0]  # each poll gets only what is left of the 5s bound


def test_poll_until_terminal_gives_a_long_wait_the_full_poll_budget(monkeypatch):
    """The per-poll cap only bites when it is *below* the normal poll budget."""
    seen: list[float] = []

    def fake_run(*args, timeout=None, **kwargs):
        seen.append(timeout)
        return {"status": "completed"}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    result = server._poll_until_terminal(
        "jobs", "status", "pid", timeout_seconds=600.0, is_terminal=server._is_terminal
    )

    assert result == {"status": "completed"}
    assert seen == [server._JOB_STATUS_POLL_TIMEOUT]


def test_poll_until_terminal_reports_a_deadline_poll_timeout_as_timed_out(monkeypatch):
    """A poll killed by the caller's own bound returns `timed_out`, not an error.

    Capping each poll to the time left makes the poll's deadline double as the
    caller's, so a slow-but-healthy `comfy jobs status` near the bound now raises
    where the old fixed 60s budget let it finish. That is this call expiring, and
    it must not discard the last real status behind a `ComfyCliError`.
    """
    statuses = iter([{"status": "running"}])

    def fake_run(*args, **kwargs):
        try:
            return next(statuses)
        except StopIteration:
            raise server.ComfyCliError("comfy-cli timed out after 1.0s", timed_out=True)

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # 10s bound: the first poll succeeds, the second is granted the remainder
    # and dies at it — by which time the clock is past the deadline. A single
    # extra trailing value (rather than the exhausting `next(reads)` the
    # original synchronous test used) keeps this safe to call directly too.
    reads = iter([0.0, 1.0, 2.0, 3.0, 11.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(reads, 11.0))

    result = server._poll_until_terminal(
        "jobs", "status", "pid", timeout_seconds=10.0, is_terminal=server._is_terminal
    )

    assert result == {"timed_out": True, "status": {"status": "running"}}


def test_poll_until_terminal_reraises_a_wedged_poll_with_budget_left(monkeypatch):
    """A poll that burns the FULL budget with time to spare is a real failure.

    Only the caller's bound expiring earns the `timed_out` envelope. A poll that
    exhausted `_JOB_STATUS_POLL_TIMEOUT` while the wait still has time left means
    comfy-cli is wedged — which raised before the per-poll cap existed too.
    """

    def fake_run(*args, **kwargs):
        raise server.ComfyCliError("comfy-cli timed out after 60.0s", timed_out=True)

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    # A 600s bound with the clock barely moving: the deadline is nowhere near.
    reads = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(reads, 2.0))

    with pytest.raises(server.ComfyCliError, match="timed out"):
        server._poll_until_terminal(
            "jobs",
            "status",
            "pid",
            timeout_seconds=600.0,
            is_terminal=server._is_terminal,
        )


def test_poll_until_terminal_reraises_when_no_status_was_ever_read(monkeypatch):
    """With no status in hand, the error beats a contentless `{"status": None}`.

    `{"timed_out": True, "status": None}` would bury the real diagnosis (and the
    budget that produced it) under an envelope the caller can do nothing with.
    """

    def fake_run(*args, **kwargs):
        raise server.ComfyCliError("comfy-cli timed out after 1.0s", timed_out=True)

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    reads = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(reads, 2.0))

    with pytest.raises(server.ComfyCliError, match="timed out"):
        server._poll_until_terminal(
            "jobs",
            "status",
            "pid",
            timeout_seconds=1.0,
            is_terminal=server._is_terminal,
        )


def test_poll_until_terminal_reraises_a_non_timeout_poll_failure(monkeypatch):
    """An ordinary comfy-cli error still propagates, deadline or not."""

    def fake_run(*args, **kwargs):
        raise server.ComfyCliError("job not found", code="not_found")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    reads = iter([0.0, 1.0, 99.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(reads, 99.0))

    with pytest.raises(server.ComfyCliError, match="job not found"):
        server._poll_until_terminal(
            "jobs",
            "status",
            "pid",
            timeout_seconds=1.0,
            is_terminal=server._is_terminal,
        )


def test_job_wait_sleeps_the_shared_poll_interval(monkeypatch):
    statuses = iter([{"status": "running"}, {"status": "completed"}])
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: next(statuses))

    slept: list[float] = []
    monkeypatch.setattr(server.time, "sleep", slept.append)

    result = asyncio.run(
        server.job(action="wait", prompt_id="pid", timeout_seconds=600.0)
    )
    assert result == {"status": "completed"}
    assert slept == [server._POLL_INTERVAL]


# --- action="watch" (streamed) --------------------------------------------------


def test_job_watch_streams_progress_and_returns_data(patched_stream):
    procs = patched_stream(_OK_STREAM)
    ctx = _RecordingCtx()

    result = asyncio.run(server.job(action="watch", prompt_id="pid", ctx=ctx))

    assert result == {"outputs": ["/x.png"]}
    assert len(ctx.calls) >= 1

    cmd = procs[0].cmd
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json-stream", "--where", "local"]
    assert cmd[4:] == ["jobs", "watch", "pid"]

    assert all(c["total"] == 2.0 for c in ctx.calls if c["total"] is not None)
    values = [c["progress"] for c in ctx.calls]
    assert values == sorted(values)
    assert values[-1] == 2.0


def test_job_watch_stream_error_envelope_raises_with_code(patched_stream):
    stream = (
        "\n".join(
            json.dumps(evt)
            for evt in [
                {"type": "queued", "nodes": [{"node_id": "1"}]},
                {
                    "schema": "envelope/1",
                    "type": "envelope",
                    "ok": False,
                    "error": {"code": "execution_error", "message": "boom"},
                },
            ]
        )
        + "\n"
    )
    patched_stream(stream)

    with pytest.raises(server.ComfyCliError, match="execution_error"):
        asyncio.run(server.job(action="watch", prompt_id="pid"))


def test_job_watch_times_out_returns_payload(blocking_stream):
    """R8: a genuine watch expiry is a `timed_out` payload, never a raise —
    proof that `job`'s "watch" branch passes `raise_on_timeout=False` through
    to `_run_comfy_streaming`, exactly as `watch_job` did."""
    queued = json.dumps(
        {"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]}
    )
    procs = blocking_stream([queued + "\n"])

    result = asyncio.run(
        server.job(
            action="watch", prompt_id="pid", timeout_seconds=0.25, ctx=_RecordingCtx()
        )
    )

    assert result["timed_out"] is True
    assert result["status"]["total"] == 2.0
    assert result["status"]["nodes_done"] == 0
    assert procs[0].killed


def test_job_watch_raise_on_timeout_is_false(monkeypatch):
    """R8, pinned directly at the call site (not just observed behaviorally
    above): `job`'s "watch" branch must pass `raise_on_timeout=False`."""
    seen: dict = {}

    async def fake_stream(*args, ctx=None, timeout=None, raise_on_timeout=True):
        seen["raise_on_timeout"] = raise_on_timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    asyncio.run(server.job(action="watch", prompt_id="pid"))

    assert seen["raise_on_timeout"] is False


def test_job_watch_times_out_reports_progress_without_ctx(blocking_stream):
    queued = json.dumps(
        {"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]}
    )
    executed = json.dumps({"type": "executed", "node": "1"})
    procs = blocking_stream([queued + "\n", executed + "\n"])

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["timed_out"] is True
    assert result["status"]["total"] == 2.0
    assert result["status"]["nodes_done"] == 1
    assert result["status"]["progress"] == 1.0
    assert procs[0].killed


@pytest.mark.parametrize("bad_id", ["--help", "p\x001"])
def test_job_watch_rejects_an_unusable_prompt_id(bad_id):
    with pytest.raises(server.ComfyCliError, match="prompt_id"):
        asyncio.run(server.job(action="watch", prompt_id=bad_id))


def test_job_watch_clamps_oversized_timeout(monkeypatch):
    seen: dict = {}

    async def fake_stream(*args, ctx=None, timeout=None, raise_on_timeout=True):
        seen["timeout"] = timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=float("inf"))
    )

    assert seen["timeout"] == server._MAX_WATCH_TIMEOUT


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_job_watch_rejects_a_non_positive_or_nan_timeout(monkeypatch, bad):
    started = False

    async def fake_stream(*args, ctx=None, timeout=None, raise_on_timeout=True):
        nonlocal started
        started = True
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        asyncio.run(server.job(action="watch", prompt_id="pid", timeout_seconds=bad))

    assert started is False


# --- action="cancel" -------------------------------------------------------------


def test_job_cancel_maps_command_and_returns_data(patched_run):
    calls = patched_run(envelope(data={"cancelled": "abc"}))

    assert asyncio.run(server.job(action="cancel", prompt_id="abc")) == {
        "cancelled": "abc"
    }
    assert calls[0]["cmd"][4:] == ["jobs", "cancel", "abc"]


def test_job_cancel_unknown_id_raises_error_envelope(patched_run):
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "not_found", "message": "no such job: nope"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="not_found"):
        asyncio.run(server.job(action="cancel", prompt_id="nope"))


# --- action="queue" ---------------------------------------------------------------


def test_job_queue_maps_command_and_returns_data(patched_run):
    jobs = {
        "jobs": [
            {"prompt_id": "a", "status": "running"},
            {"prompt_id": "b", "status": "completed"},
        ]
    }
    calls = patched_run(envelope(data=jobs))

    assert asyncio.run(server.job(action="queue")) == jobs
    assert calls[0]["cmd"][4:] == ["jobs", "ls"]


def test_job_queue_error_envelope_raises(patched_run):
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "server_not_running", "message": "ComfyUI not running"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="server_not_running"):
        asyncio.run(server.job(action="queue"))


def test_job_queue_drops_cloud_rows(patched_run):
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {
                "host": "127.0.0.1",
                "port": 8188,
                "where": "local",
                "count": 3,
                "jobs": [
                    {"prompt_id": "a", "status": "running", "where": "local"},
                    {"prompt_id": "b", "status": "completed", "where": "cloud"},
                    {"prompt_id": "c", "status": "completed", "where": "local"},
                ],
            },
        }
    )

    result = asyncio.run(server.job(action="queue"))

    assert [job["prompt_id"] for job in result["jobs"]] == ["a", "c"]
    assert result["count"] == 2
    assert result["where"] == "local"


def test_job_queue_keeps_rows_without_a_where(patched_run):
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {
                "count": 2,
                "jobs": [
                    {"prompt_id": "a", "status": "running"},
                    {"prompt_id": "b", "status": "completed", "where": None},
                ],
            },
        }
    )

    result = asyncio.run(server.job(action="queue"))
    assert [job["prompt_id"] for job in result["jobs"]] == ["a", "b"]


def test_job_queue_keeps_rows_that_are_not_dicts(patched_run):
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": {
                "count": 3,
                "jobs": ["a-bare-id", None, {"prompt_id": "b", "where": "cloud"}],
            },
        }
    )

    result = asyncio.run(server.job(action="queue"))

    assert result["jobs"] == ["a-bare-id", None]
    assert result["count"] == 2


def test_job_queue_passes_through_foreign_payload_shapes(patched_run):
    patched_run(
        {
            "type": "envelope",
            "ok": True,
            "data": [{"prompt_id": "a", "where": "cloud"}],
        }
    )

    assert asyncio.run(server.job(action="queue")) == [
        {"prompt_id": "a", "where": "cloud"}
    ]


def test_job_queue_passes_through_payload_without_jobs(patched_run):
    patched_run(
        {"type": "envelope", "ok": True, "data": {"host": "127.0.0.1", "port": 8188}}
    )

    assert asyncio.run(server.job(action="queue")) == {
        "host": "127.0.0.1",
        "port": 8188,
    }


# --- shared prompt_id guard family (status / error / wait / watch / cancel) ---


@pytest.mark.parametrize("action", ["status", "error", "wait", "cancel"])
@pytest.mark.parametrize("bad_id", ["--help", "-o", "p\x001"])
def test_job_rejects_an_unusable_prompt_id(monkeypatch, action, bad_id):
    """A dash-led / NUL-bearing prompt_id is refused before any spawn — the
    non-streaming half of the family guard (see the "watch" variant above)."""

    def fake_run(*args, **kwargs):
        raise AssertionError(
            f"job(action={action!r}) spawned comfy-cli with {bad_id!r}"
        )

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="prompt_id"):
        asyncio.run(server.job(action=action, prompt_id=bad_id))


def test_job_prompt_id_guard_rejects_an_oversized_id(monkeypatch):
    from comfy_mcp import argv

    oversized = "p" * (argv._MAX_PROMPT_ID_LEN + 1)

    def fake_run(*args, **kwargs):
        raise AssertionError("spawned comfy-cli with an oversized prompt_id")

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="exceeds"):
        asyncio.run(server.job(action="status", prompt_id=oversized))

    assert argv._guard_prompt_id("p" * argv._MAX_PROMPT_ID_LEN)


# --- tool docstring budget (R5: <=300 est. tokens) -----------------------------


def test_job_tool_docstring_within_its_own_token_budget():
    tree = ast.parse(_SERVER_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "job":
            doc = ast.get_docstring(node)
            assert doc is not None
            est_tokens = len(doc) // 4
            assert est_tokens <= 300, f"job() docstring ~{est_tokens} est. tokens"
            # First line must not open with "LOCAL" — `job` is target-aware
            # (follows a configured COMFYUI_URL/COMFYUI_HOST), so claiming
            # LOCAL there would be simply wrong (test_address_env_docs.py).
            assert not doc.strip().splitlines()[0].startswith("LOCAL")
            return
    pytest.fail("job() tool not found in server/_internal.py")
