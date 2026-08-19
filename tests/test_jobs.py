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
import subprocess
import threading
import time
from pathlib import Path

import pytest
from conftest import _OK_STREAM, _RecordingCtx, envelope, stream_reader

from comfy_mcp import server

_SERVER_SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_mcp" / "server.py"


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

    async def fake_stream(
        *args, ctx=None, timeout=None, raise_on_timeout=True, **kwargs
    ):
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
    """600s is the CALL's budget, and the ONE serial `jobs status` poll (the
    fallback at the deadline) is carved out of it rather than added on top — so
    the stream gets 600 less one `_WATCH_POLL_MAX`, and the wall time stays
    bounded by what the caller asked for. The seed poll races the stream, so it
    costs the budget nothing."""
    seen: dict = {}
    calls = _patch_jobs_status(monkeypatch, {"status": "running"})

    async def fake_stream(
        *args, ctx=None, timeout=None, raise_on_timeout=True, total_seed=None, **kwargs
    ):
        seen["timeout"] = timeout
        # Stand in for the runner: it CALLS the seed thunk (concurrently with
        # the stream), so a test asserting on the seed's budget has to too.
        await total_seed()
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)

    asyncio.run(server.job(action="watch", prompt_id="pid"))

    assert seen["timeout"] == 600.0 - server._WATCH_POLL_MAX
    assert calls[0]["timeout"] == server._WATCH_POLL_MAX


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


def test_job_watch_streams_progress_and_returns_data(patched_stream, monkeypatch):
    # Stub the two `jobs status` polls: they spawn through the same
    # `create_subprocess_exec` the stream fake patches, so without this
    # `procs[0]` would be the seed poll rather than the `jobs watch` child.
    _patch_jobs_status(monkeypatch, {"status": "running"})
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


class _BlockingProc:
    """A fake child whose stdout yields ``first_lines`` then blocks (forces a
    timeout). Copied from the former ``test_watch_job`` coverage — see
    ``conftest._FakeProc`` for the sibling used by the non-timeout paths."""

    def __init__(self, cmd, first_lines):
        self.cmd = cmd
        self._lines = [line.encode("utf-8") for line in first_lines]
        self.stdout = self
        self.stderr = stream_reader("")
        self.returncode = None
        self.killed = False

    async def readuntil(self, separator=b"\n"):
        if self._lines:
            return self._lines.pop(0)
        await asyncio.sleep(1.0)
        raise asyncio.IncompleteReadError(b"", None)

    async def wait(self):
        self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True


def test_job_watch_times_out_returns_payload(monkeypatch):
    """R8: a genuine watch expiry is a `timed_out` payload, never a raise —
    proof that `job`'s "watch" branch passes `raise_on_timeout=False` through
    to `_run_comfy_streaming`, exactly as `watch_job` did."""
    queued = json.dumps(
        {"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]}
    )
    procs: list[_BlockingProc] = []

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProc(cmd, [queued + "\n"])
        procs.append(proc)
        return proc

    # Stub the diagnostic polls out of the spawn list: `procs[0]` must be the
    # `jobs watch` child whose kill this test is asserting.
    _patch_jobs_status(monkeypatch, {"status": "running"})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

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

    async def fake_stream(
        *args, ctx=None, timeout=None, raise_on_timeout=True, **kwargs
    ):
        seen["raise_on_timeout"] = raise_on_timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    asyncio.run(server.job(action="watch", prompt_id="pid"))

    assert seen["raise_on_timeout"] is False


def test_job_watch_times_out_reports_progress_without_ctx(monkeypatch):
    queued = json.dumps(
        {"type": "queued", "nodes": [{"node_id": "1"}, {"node_id": "2"}]}
    )
    executed = json.dumps({"type": "executed", "node": "1"})
    procs: list[_BlockingProc] = []

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProc(cmd, [queued + "\n", executed + "\n"])
        procs.append(proc)
        return proc

    _patch_jobs_status(monkeypatch, {"status": "running"})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["timed_out"] is True
    assert result["status"]["total"] == 2.0
    assert result["status"]["nodes_done"] == 1
    assert result["status"]["progress"] == 1.0
    assert procs[0].killed


# --- action="watch": the comfy-cli 1.16.0 floor + the timeout payload ----------
#
# `jobs watch` below comfy-cli 1.16.0 attached to ComfyUI's websocket with a
# fresh client_id, and ComfyUI addresses execution events to the SUBMITTING
# session only — so the watch received nothing and blocked for its whole
# timeout (comfy-cli #693 fixed it in 1.16.0). These pin the loud gate that
# replaces that silence, and the timeout payload that has to stay informative
# on the installs the gate deliberately fails OPEN for.


def _patch_comfy_version(monkeypatch, version_text: str) -> None:
    """Un-memoize the version guard and answer `comfy --version` with *version_text*.

    A local stub rather than a shared fixture, for the one reason AGENTS.md
    allows one: the `--version` probe is `subprocess.run` with its own kwargs,
    not the `Popen` the conftest fakes model. It resets BOTH globals the guard
    owns — the latch and the parsed version the watch gate reads.
    """

    def fake(cmd, capture_output, text, timeout, check, errors=None, cwd=None):
        return subprocess.CompletedProcess(cmd, 0, stdout=version_text, stderr="")

    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    monkeypatch.setattr(server.subprocess, "run", fake)


def _patch_jobs_status(monkeypatch, payload, *, raises=None) -> list[dict]:
    """Answer BOTH `jobs status` shell-outs the watch branch makes.

    The seed poll before the stream and the fallback poll at the deadline both
    run `_run_comfy_async("jobs", "status", ...)` — the CANCELLABLE spawn path,
    not `_run_comfy` on the thread pool — so one stub covers both, and the
    returned call list is how a test tells them apart (by count and order).
    Each entry records the `timeout` that poll was handed, which is how the
    budget test asserts the polls are carved OUT of the caller's bound rather
    than added on top of it.

    Stubbing at `_run_comfy_async` rather than at `create_subprocess_exec` also
    keeps the polls out of the fake-spawn lists the stream tests index into:
    `procs[0]` stays the `jobs watch` child.

    `_WATCH_POLL_MIN` is dropped to zero for the same reason the timeouts here
    are sub-second: the real floor makes the watch branch SKIP diagnosis
    entirely below a ~5s budget (see the call site), which is right in
    production — a poll that short cannot outrun comfy-cli's startup — and
    useless in a suite whose polls answer instantly. Tests that are ABOUT the
    floor set it back themselves.
    """
    monkeypatch.setattr(server, "_WATCH_POLL_MIN", 0.0)
    calls: list[dict] = []

    async def fake_run(*args, **kwargs):
        calls.append({"args": args, "timeout": kwargs.get("timeout")})
        if raises is not None:
            raise raises
        return payload

    monkeypatch.setattr(server, "_run_comfy_async", fake_run)
    return calls


def _patch_blocking_stream(monkeypatch, first_lines) -> list[_BlockingProc]:
    """Spawn `_BlockingProc`s that emit *first_lines* and then hang (force a timeout)."""
    procs: list[_BlockingProc] = []

    async def fake_exec(*cmd, stdout, stderr, env, **kwargs):
        proc = _BlockingProc(cmd, list(first_lines))
        procs.append(proc)
        return proc

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    return procs


def test_job_watch_refuses_a_comfy_cli_below_1_16_0(monkeypatch, patched_stream):
    """The gate is LOUD and fires BEFORE the spawn: on 1.15.0 the caller gets one
    sentence naming the fix, not a silent block for the full timeout."""
    procs = patched_stream(_OK_STREAM)
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.15.0")

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(server.job(action="watch", prompt_id="pid"))

    message = str(excinfo.value)
    assert "1.16.0" in message
    assert "pip install -U comfy-cli" in message
    # The gate DENIES a capability, so it must name the path that still works.
    assert 'job(action="wait")' in message
    assert procs == []  # nothing was spawned


def test_job_watch_version_gate_fails_open_on_an_unreadable_version(
    monkeypatch, patched_stream
):
    """Same policy as the server-wide floor: a `--version` that cannot be parsed
    is UNKNOWN, not too-old, so the watch proceeds exactly as it did before."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    procs = patched_stream(_OK_STREAM)
    _patch_comfy_version(monkeypatch, "comfy-cli, version unreleased-dev")

    result = asyncio.run(server.job(action="watch", prompt_id="pid"))

    assert server._comfy_cli_version is None
    assert result == {"outputs": ["/x.png"]}
    assert procs[0].cmd[4:] == ["jobs", "watch", "pid"]


def test_job_watch_allows_1_16_0_itself(monkeypatch, patched_stream):
    """Pin the boundary: the floor release is accepted, not merely "newer than"."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    procs = patched_stream(_OK_STREAM)
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    assert asyncio.run(server.job(action="watch", prompt_id="pid")) == {
        "outputs": ["/x.png"]
    }
    assert server._comfy_cli_version == (1, 16, 0)
    assert procs[0].cmd[4:] == ["jobs", "watch", "pid"]


def test_job_watch_zero_event_timeout_embeds_a_status_poll_and_a_hint(monkeypatch):
    """The regression this ticket exists for: a watch that streamed NOTHING used
    to time out into `{"timed_out": true, "status": {progress: null, total:
    null, nodes_done: 0}}` — strictly less than `job(action="wait")` would have
    said about the same job. Now it embeds the poll and says what zero means."""
    calls = _patch_jobs_status(monkeypatch, {"status": "running", "workflow_size": 4})
    procs = _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["timed_out"] is True
    assert result["events_seen"] == 0
    assert result["status"]["events_seen"] == 0
    assert result["job"] == {"status": "running", "workflow_size": 4}
    # The gate VERIFIED 1.16.0 here, so the hint must not blame the version it
    # just cleared — it names the causes that are still on the table.
    assert "comfy-cli < 1.16.0" not in result["hint"]
    assert "queued behind another" in result["hint"]
    # Seeded from the same poll, so `total` is no longer structurally null.
    assert result["status"]["total"] == 4.0
    assert procs[0].killed
    # Two polls: one seeding `total` before the stream, one at the deadline.
    assert [c["args"] for c in calls] == [("jobs", "status", "pid")] * 2


def test_job_watch_zero_event_timeout_omits_job_when_the_poll_fails(monkeypatch):
    """A failed fallback poll must never mask the timeout it was only describing."""
    _patch_jobs_status(monkeypatch, None, raises=server.ComfyCliError("no ComfyUI"))
    _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["timed_out"] is True
    assert result["events_seen"] == 0
    assert "job" not in result
    assert "hint" not in result
    assert result["status"]["total"] is None  # the seed failed too: unseeded


def test_job_watch_zero_event_timeout_omits_the_hint_on_a_terminal_job(monkeypatch):
    """ "Still running" would be WRONG for a job that already finished — the watch
    simply missed the end — so the embedded poll ships without the hint."""
    _patch_jobs_status(monkeypatch, {"status": "completed", "workflow_size": 2})
    _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["job"] == {"status": "completed", "workflow_size": 2}
    assert "hint" not in result


def test_job_watch_timeout_that_saw_events_skips_the_fallback_poll(monkeypatch):
    """A watch with live progress has its own answer in `status`; only the seed
    poll runs, and no `job`/`hint` is attached."""
    calls = _patch_jobs_status(monkeypatch, {"status": "running", "workflow_size": 3})
    _patch_blocking_stream(
        monkeypatch, [json.dumps({"type": "executed", "node": "1"}) + "\n"]
    )
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["events_seen"] == 1
    assert result["status"]["nodes_done"] == 1
    assert "job" not in result
    assert "hint" not in result
    assert len(calls) == 1  # the seed only


def test_job_watch_seeds_total_from_the_jobs_status_workflow_size(monkeypatch):
    """`jobs watch` attaches post-submit and never sees the run dialect's
    `queued` manifest, so `total` comes from `jobs status`'s `workflow_size`."""
    _patch_jobs_status(monkeypatch, {"status": "running", "workflow_size": 10})
    _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["status"]["total"] == 10.0


@pytest.mark.parametrize(
    "status_payload",
    [
        {"status": "running"},  # no workflow_size at all
        {"status": "running", "workflow_size": None},
        {"status": "running", "workflow_size": 0},  # a bar that can never fill
        {"status": "running", "workflow_size": True},  # bool is an int subclass
        {"status": "running", "workflow_size": "10"},
        ["not", "a", "dict"],
    ],
)
def test_job_watch_leaves_total_unseeded_on_an_unusable_workflow_size(
    monkeypatch, status_payload
):
    """Every non-usable shape degrades to today's behavior — an unseeded
    `total: null` — rather than fabricating a node count."""
    _patch_jobs_status(monkeypatch, status_payload)
    _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["status"]["total"] is None


def test_job_watch_counts_a_cached_node_LIST_as_that_many_nodes(monkeypatch):
    """`jobs watch` relays ComfyUI's single `execution_cached` message, whose
    `nodes` is the whole cached list — counting it as one node reported
    `nodes_done: 1` for a fully-cached ten-node workflow."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    _patch_blocking_stream(
        monkeypatch,
        [json.dumps({"type": "execution_cached", "nodes": ["1", "2", "3"]}) + "\n"],
    )
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")
    ctx = _RecordingCtx()

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25, ctx=ctx)
    )

    assert result["status"]["nodes_done"] == 3
    assert result["status"]["progress"] == 3.0
    assert result["events_seen"] == 1
    assert ctx.calls[-1]["message"] == "cached 3 node(s)"


@pytest.mark.parametrize(
    ("event", "expected_message"),
    [
        ({"node": "7"}, "cached 7"),
        ({"title": "Load Checkpoint", "node": "7"}, "cached Load Checkpoint"),
        ({"nodes": [], "node": "7"}, "cached 7"),
        ({}, "cached 1 node(s)"),
    ],
)
def test_job_watch_counts_a_per_node_cached_event_as_one(
    monkeypatch, event, expected_message
):
    """The OTHER dialect is unchanged: `comfy run` emits one `execution_cached`
    per node keyed `node`, and an absent/empty `nodes` still counts as one."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    _patch_blocking_stream(
        monkeypatch, [json.dumps({"type": "execution_cached", **event}) + "\n"]
    )
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")
    ctx = _RecordingCtx()

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25, ctx=ctx)
    )

    assert result["status"]["nodes_done"] == 1
    # The run dialect names its one node; splitting the branch must not drop it.
    assert ctx.calls[-1]["message"] == expected_message


@pytest.mark.parametrize("bad_id", ["--help", "p\x001"])
def test_job_watch_rejects_an_unusable_prompt_id(bad_id):
    with pytest.raises(server.ComfyCliError, match="prompt_id"):
        asyncio.run(server.job(action="watch", prompt_id=bad_id))


def test_job_watch_clamps_oversized_timeout(monkeypatch):
    """The ceiling still binds, and the polls still come out of the clamped bound
    — `_MAX_WATCH_TIMEOUT` caps the CALL, not just the stream."""
    seen: dict = {}
    _patch_jobs_status(monkeypatch, {"status": "running"})

    async def fake_stream(
        *args, ctx=None, timeout=None, raise_on_timeout=True, **kwargs
    ):
        seen["timeout"] = timeout
        return {"outputs": []}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=float("inf"))
    )

    assert seen["timeout"] == server._MAX_WATCH_TIMEOUT - server._WATCH_POLL_MAX


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_job_watch_rejects_a_non_positive_or_nan_timeout(monkeypatch, bad):
    started = False

    async def fake_stream(
        *args, ctx=None, timeout=None, raise_on_timeout=True, **kwargs
    ):
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
    pytest.fail("job() tool not found in server.py")


# --- review follow-ups on the watch gate (PR #241) ------------------------------
#
# Everything below pins a fix made in review, and each one is a case where the
# first cut of the gate said something the code itself could disprove.


def test_a_rejected_per_verb_floor_un_memoizes_the_version(monkeypatch):
    """The upgrade the error message asks for has to be able to TAKE EFFECT.

    1.15.0 clears the server-wide `_MIN_COMFY_CLI`, so `_check_comfy_version`
    latches `_version_checked` on its way past. Without dropping that latch the
    watch gate's refusal outlives the fix it prescribes: the user runs `pip
    install -U comfy-cli`, retries in the same long-lived server process, and is
    told to upgrade again — forever.
    """
    versions = iter(["comfy-cli, version 1.15.0", "comfy-cli, version 1.16.0"])
    probes: list[int] = []

    def fake(cmd, capture_output, text, timeout, check, errors=None, cwd=None):
        probes.append(1)
        return subprocess.CompletedProcess(cmd, 0, stdout=next(versions), stderr="")

    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    # The rate limit exists to stop a CLIENT retry loop re-probing per refused
    # call; the human upgrade-and-retry this test models takes far longer than
    # the window, so zero it rather than sleep through it.
    monkeypatch.setattr(server, "_VERSION_REPROBE_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(server.subprocess, "run", fake)
    _patch_jobs_status(monkeypatch, {"status": "running"})
    procs = _patch_blocking_stream(monkeypatch, [])

    with pytest.raises(server.ComfyCliError, match="1.16.0"):
        asyncio.run(server.job(action="watch", prompt_id="pid"))

    # The refusal dropped BOTH halves of the memo, so the next call re-probes.
    assert server._version_checked is False
    assert server._comfy_cli_version is None

    # Same process, upgraded install: the watch now runs.
    asyncio.run(server.job(action="watch", prompt_id="pid", timeout_seconds=0.25))

    assert len(probes) == 2
    assert server._comfy_cli_version == (1, 16, 0)
    assert list(procs[0].cmd[4:]) == ["jobs", "watch", "pid"]


def test_a_reprobe_that_cannot_read_the_version_clears_the_stale_one(monkeypatch):
    """`_comfy_cli_version` is reset at the TOP of the probe, so "unparseable =>
    fail OPEN" keeps holding on a retry. A source build installed over the
    release an earlier probe read must leave the version UNKNOWN, not leave the
    old tuple in place for a per-verb floor to keep refusing on."""
    versions = iter(["comfy-cli, version 1.15.0", "comfy-cli, version unreleased-dev"])

    def fake(cmd, capture_output, text, timeout, check, errors=None, cwd=None):
        return subprocess.CompletedProcess(cmd, 0, stdout=next(versions), stderr="")

    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    monkeypatch.setattr(server, "_VERSION_REPROBE_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()
    assert server._comfy_cli_version == (1, 15, 0)

    server._invalidate_version_cache()
    server._check_comfy_version()

    assert server._comfy_cli_version is None


def test_concurrent_first_callers_probe_the_version_once(monkeypatch):
    """`_version_lock` closes the check-then-set: without it every thread that
    arrives before the latch is set spawns its own 30s `comfy --version`, and a
    reader can catch `_comfy_cli_version` mid-probe."""
    probes: list[int] = []

    def fake(cmd, capture_output, text, timeout, check, errors=None, cwd=None):
        probes.append(1)
        time.sleep(0.05)  # widen the window the lock has to close
        return subprocess.CompletedProcess(
            cmd, 0, stdout="comfy-cli, version 1.16.0", stderr=""
        )

    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    monkeypatch.setattr(server.subprocess, "run", fake)

    threads = [threading.Thread(target=server._check_comfy_version) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert probes == [1]
    assert server._comfy_cli_version == (1, 16, 0)


def test_the_watch_polls_are_carved_out_of_the_callers_timeout(monkeypatch):
    """A 5s watch must not hold the MCP request for ~125s. Only the FALLBACK
    poll is serial, so only it comes out of the bound (a tenth, capped at
    `_WATCH_POLL_MAX`) — the stream gets the rest and the serial sum never
    exceeds `timeout_seconds`. `_patch_jobs_status` zeroes `_WATCH_POLL_MIN` so
    the tenth is what is actually asserted here; the floor's own behavior has
    its own test below."""
    calls = _patch_jobs_status(monkeypatch, {"status": "running"})
    seen: dict = {}

    async def fake_stream(
        *args,
        ctx=None,
        timeout=None,
        raise_on_timeout=True,
        total_seed=None,
        on_timeout_fallback=None,
        **kwargs,
    ):
        # Stand in for the runner on a zero-event expiry: it runs the seed
        # thunk concurrently and then the fallback thunk, so both budgets show
        # up in `calls` in that order.
        seen["timeout"] = timeout
        await total_seed()
        await on_timeout_fallback()
        return {"timed_out": True, "events_seen": 0, "status": {}}

    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    asyncio.run(server.job(action="watch", prompt_id="pid", timeout_seconds=5.0))

    # The seed ran first and got the whole (concurrent) stream window; the
    # fallback got the carved tenth. Their SUM is irrelevant — only the serial
    # part has to fit.
    assert [c["timeout"] for c in calls] == [4.5, 0.5]
    assert seen["timeout"] == 4.5
    assert calls[1]["timeout"] + seen["timeout"] <= 5.0


def test_the_watch_polls_run_on_the_cancellable_spawn_path(monkeypatch):
    """Both polls must go through `_run_comfy_async`, not
    `asyncio.to_thread(_run_comfy, ...)`: a client that gives up mid-watch can
    cancel a coroutine but never a worker thread's `comfy jobs status` child."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    def forbidden(*args, **kwargs):
        raise AssertionError("a watch poll used the blocking `_run_comfy` path")

    monkeypatch.setattr(server, "_run_comfy", forbidden)

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["job"] == {"status": "running"}


def test_the_no_events_hint_names_the_version_when_it_is_unknown(monkeypatch):
    """The fail-OPEN case is the one the version half of the hint exists for: an
    install the gate could not identify may well be the too-old one, and this is
    the only place left to say so."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version unreleased-dev")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert server._comfy_cli_version is None
    assert "comfy-cli < 1.16.0" in result["hint"]


@pytest.mark.parametrize(
    "job_payload",
    [
        {"state": "running"},  # `status` renamed
        {"status": 42},  # `status` not a string
        {"error": "something went sideways"},  # an error blob, not a report
        {"status": "some-status-nobody-here-knows"},
        ["not", "a", "dict"],
    ],
)
def test_the_no_events_hint_needs_a_POSITIVELY_active_status(monkeypatch, job_payload):
    """`not _is_terminal(...)` is not the test the docstring promised: it is also
    False for a payload carrying no readable status at all, so an unreadable
    blob used to be answered with "the job is still running" — exactly the guess
    the hint must never be. Only a recognized ACTIVE status earns the claim."""
    _patch_jobs_status(monkeypatch, job_payload)
    _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["job"] == job_payload  # still embedded: it is what we know
    assert "hint" not in result  # but no claim about a job we cannot read


@pytest.mark.parametrize(
    "workflow_size",
    [float("inf"), float("nan"), 10**400],
)
def test_a_non_finite_workflow_size_leaves_total_unseeded(monkeypatch, workflow_size):
    """`json.loads` accepts bare `Infinity`/`NaN`, and a Python int has no upper
    bound — so `float(size)` could either seed a non-finite `total` that leaves
    the payload as non-standard JSON, or raise `OverflowError` straight out of a
    best-effort seed and abort the whole watch."""
    _patch_jobs_status(
        monkeypatch, {"status": "running", "workflow_size": workflow_size}
    )
    _patch_blocking_stream(monkeypatch, [])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["timed_out"] is True
    assert result["status"]["total"] is None


@pytest.mark.parametrize("etype", ["execution_start", "execution_error", "output"])
def test_a_non_progress_execution_event_still_counts_as_seen(monkeypatch, etype):
    """`events_seen` is a claim about the WATCHER, not about progress. An
    `execution_error` on the first node proves `jobs watch` attached — reporting
    zero there sent the caller off to check a comfy-cli version that was fine,
    and burned a second `jobs status` spawn to do it."""
    calls = _patch_jobs_status(monkeypatch, {"status": "running"})
    _patch_blocking_stream(monkeypatch, [json.dumps({"type": etype}) + "\n"])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["events_seen"] == 1
    assert result["status"]["progress"] is None  # it was not a progress tick
    assert "hint" not in result
    assert len(calls) == 1  # the seed only: no fallback poll


def test_unrecognized_stream_chatter_does_not_count_as_an_event(monkeypatch):
    """The other half of the same claim: a custom node's own line proves nothing
    about the websocket attach, so it must not suppress the diagnosis."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    _patch_blocking_stream(
        monkeypatch, [json.dumps({"type": "some-custom-node-line"}) + "\n"]
    )
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["events_seen"] == 0
    assert result["job"] == {"status": "running"}


def test_progress_never_exceeds_its_own_total(monkeypatch):
    """`total` is a lower bound an under-counting `workflow_size` can disprove.
    When the run gets further than the seed promised, the SEED was wrong — raise
    it rather than hand a client a progress above its own total to render."""
    _patch_jobs_status(monkeypatch, {"status": "running", "workflow_size": 2})
    _patch_blocking_stream(
        monkeypatch,
        [
            json.dumps({"type": "execution_cached", "nodes": ["1", "2", "3", "4"]})
            + "\n"
        ],
    )
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")
    ctx = _RecordingCtx()

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25, ctx=ctx)
    )

    assert result["status"]["nodes_done"] == 4  # what was actually observed
    assert result["status"]["progress"] == 4.0
    assert result["status"]["total"] == 4.0  # grown, not left at the stale 2
    assert all(c["progress"] <= c["total"] for c in ctx.calls if c["total"] is not None)


def test_the_version_gate_reads_the_version_the_check_RETURNED(monkeypatch):
    """The gate must not run `_check_comfy_version` and then re-read the mutable
    `_comfy_cli_version`: those two steps are not atomic, and a concurrent
    refusal's `_invalidate_version_cache` nulls the global in between — so the
    caller would fail OPEN and start a silently broken watch on a CLI the
    process had just positively read as too old. Here the global says "unknown"
    (the fail-open shape) while the atomic reader returns 1.15.0; the refusal
    proves which one the gate believes."""
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    monkeypatch.setattr(server, "_checked_comfy_version", lambda: (1, 15, 0))
    procs = _patch_blocking_stream(monkeypatch, [])

    with pytest.raises(server.ComfyCliError, match="1.16.0"):
        asyncio.run(server.job(action="watch", prompt_id="pid"))

    assert procs == []


def test_checked_comfy_version_returns_the_tuple_it_established(monkeypatch):
    """The atomic reader's contract: the probe's own answer, not None."""
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    assert server._checked_comfy_version() == (1, 16, 0)


def test_the_version_probe_prefers_stdout_over_stderr(monkeypatch):
    """`_parse_version` takes the FIRST dotted number it finds, and stderr is
    where unrelated ones land — a deprecation warning, a ComfyUI core `0.3.x`
    banner. Reading the combined text let one of those become the authoritative
    version, which now drives a hard `job(action="watch")` refusal."""

    def fake(cmd, capture_output, text, timeout, check, errors=None, cwd=None):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="comfy-cli, version 1.16.0",
            stderr="WARNING: ComfyUI 0.3.10 is deprecated",
        )

    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()

    assert server._comfy_cli_version == (1, 16, 0)


def test_the_version_probe_falls_back_to_stderr_when_stdout_has_none(monkeypatch):
    """Preferring stdout must not mean IGNORING stderr: a build that prints its
    version there is still readable, and reading it is what keeps the guard from
    failing open on an install it could identify."""

    def fake(cmd, capture_output, text, timeout, check, errors=None, cwd=None):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr="comfy-cli, version 1.16.0"
        )

    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()

    assert server._comfy_cli_version == (1, 16, 0)


def test_a_failed_version_probe_records_no_version(monkeypatch):
    """`--version` that EXITS NONZERO has not reported a version, whatever text
    it printed. Scraping a number out of a failure and memoizing it for the life
    of the process is how a traceback's `1.2.3` ends up gating `jobs watch`."""

    def fake(cmd, capture_output, text, timeout, check, errors=None, cwd=None):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="Traceback: something at line 1.15.0"
        )

    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    monkeypatch.setattr(server.subprocess, "run", fake)

    server._check_comfy_version()

    assert server._comfy_cli_version is None  # unknown => fail OPEN


def test_a_refused_watch_does_not_re_probe_the_version_on_every_retry(monkeypatch):
    """`_invalidate_version_cache` is rate-limited. Every refused watch calls it,
    so without the limit a client retrying `job(action="watch")` in a loop forces
    a fresh `comfy --version` per call — each taken with `_version_lock` HELD, so
    every other tool call in the process queues behind it."""
    probes: list[int] = []

    def fake(cmd, capture_output, text, timeout, check, errors=None, cwd=None):
        probes.append(1)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="comfy-cli, version 1.15.0", stderr=""
        )

    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server, "_comfy_cli_version", None)
    monkeypatch.setattr(server, "_version_probed_at", None)
    monkeypatch.setattr(server.subprocess, "run", fake)

    for _ in range(5):
        with pytest.raises(server.ComfyCliError, match="1.16.0"):
            asyncio.run(server.job(action="watch", prompt_id="pid"))

    assert probes == [1]


def test_the_total_seed_poll_does_not_delay_the_websocket_attach(monkeypatch):
    """The seed costs a whole extra `comfy jobs status` child. Awaited BEFORE the
    spawn it sits between submit and the websocket attach and every execution
    event in that window is lost — manufacturing the `events_seen: 0` payload
    this diagnosis exists to explain. It has to race the stream instead."""
    procs = _patch_blocking_stream(monkeypatch, [])
    monkeypatch.setattr(server, "_WATCH_POLL_MIN", 0.0)
    order: list[str] = []

    async def fake_run(*args, **kwargs):
        # How many children existed at the moment the poll actually ran.
        order.append(len(procs))
        return {"status": "running", "workflow_size": 4}

    monkeypatch.setattr(server, "_run_comfy_async", fake_run)
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    # The `jobs watch` child was already spawned when the seed poll ran — and
    # the seed still landed, so racing it did not cost the `total`.
    assert order[0] == 1
    assert list(procs[0].cmd[4:]) == ["jobs", "watch", "pid"]
    assert result["status"]["total"] == 4.0


def test_a_late_seed_never_lowers_total_below_what_was_observed(monkeypatch):
    """Racing the stream means the seed can land AFTER events were counted. It
    must not then pull `total` down under the progress already reported — that is
    the >100% bar the growth rule in `report` exists to prevent."""
    tracker = server._StreamProgress()
    asyncio.run(tracker.report(None, {"type": "execution_cached", "nodes": ["1", "2"]}))
    assert tracker.total is None

    tracker.seed(1.0)  # `workflow_size` under-counted the run

    assert tracker.total == 2.0


def test_a_seed_never_overwrites_a_queued_manifest():
    """The manifest is the authoritative node count; the seed only fills the hole
    left by the dialect (`jobs watch`) that never sees one."""
    tracker = server._StreamProgress()
    asyncio.run(tracker.report(None, {"type": "queued", "nodes": ["1", "2", "3"]}))

    tracker.seed(99.0)

    assert tracker.total == 3.0


def test_a_tiny_watch_budget_skips_diagnosis_rather_than_shrinking_it(monkeypatch):
    """`bound / 10` alone gives a short watch less than comfy-cli's own startup,
    so the fallback poll is spawned, killed at its deadline and charged to the
    watch for nothing. Below the floor the whole budget goes to the stream."""
    seen: dict = {}
    calls: list[dict] = []

    async def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "running"}

    async def fake_stream(
        *args,
        ctx=None,
        timeout=None,
        raise_on_timeout=True,
        on_timeout_fallback=None,
        **kwargs,
    ):
        seen["timeout"] = timeout
        seen["fallback"] = on_timeout_fallback
        return {"timed_out": True, "events_seen": 0, "status": {}}

    monkeypatch.setattr(server, "_run_comfy_async", fake_run)
    monkeypatch.setattr(server, "_run_comfy_streaming", fake_stream)
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    asyncio.run(server.job(action="watch", prompt_id="pid", timeout_seconds=6.0))

    assert seen["timeout"] == 6.0  # nothing carved out
    assert seen["fallback"] is None  # and nothing to spend it on


def test_the_no_events_hint_uses_the_version_the_gate_captured(monkeypatch):
    """`_watch_no_events_hint` used to re-read `_comfy_cli_version` at TIMEOUT
    time. That global is nulled by `_invalidate_version_cache` — which every
    refused watch calls — so a watch that positively cleared 1.16.0 could still
    be told its comfy-cli was too old. Here the fallback poll nulls it mid-flight
    to stand in for that concurrent refusal."""
    _patch_blocking_stream(monkeypatch, [])
    monkeypatch.setattr(server, "_WATCH_POLL_MIN", 0.0)

    async def fake_run(*args, **kwargs):
        server._comfy_cli_version = None
        return {"status": "running"}

    monkeypatch.setattr(server, "_run_comfy_async", fake_run)
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert "comfy-cli < 1.16.0" not in result["hint"]
    assert "queued behind another" in result["hint"]


def test_a_progress_state_event_counts_as_proof_of_attach(monkeypatch):
    """`progress_state` is ComfyUI's per-step tick and the handler
    `_MIN_WATCH_COMFY_CLI` exists to require. Uncounted, a window carrying only
    those reports `events_seen: 0` — so a demonstrably attached, mid-node watch
    earns the "nothing reached the watcher" diagnosis."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    _patch_blocking_stream(
        monkeypatch, [json.dumps({"type": "progress_state", "nodes": {}}) + "\n"]
    )
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["events_seen"] == 1
    assert "hint" not in result


@pytest.mark.parametrize("etype", [["execution_start"], {"a": 1}])
def test_an_unhashable_event_type_is_ignored_not_fatal(monkeypatch, etype):
    """`etype in _NON_PROGRESS_EVENTS` raises `TypeError: unhashable type` on a
    `type` that JSON made a list or a dict — aborting the whole stream over one
    malformed line instead of ignoring it as chatter."""
    _patch_jobs_status(monkeypatch, {"status": "running"})
    _patch_blocking_stream(monkeypatch, [json.dumps({"type": etype}) + "\n"])
    _patch_comfy_version(monkeypatch, "comfy-cli, version 1.16.0")

    result = asyncio.run(
        server.job(action="watch", prompt_id="pid", timeout_seconds=0.25)
    )

    assert result["timed_out"] is True
    assert result["events_seen"] == 0


def test_the_watch_status_polls_widen_the_stdout_cap(monkeypatch):
    """`_run_comfy_async` keeps only the TRAILING bytes of stdout, and a `jobs
    status` envelope scales with the job — every output path, plus a failed run's
    whole traceback. At the default 64 KiB tail a big one loses its opening brace
    and parses as nothing, so both diagnostic callers would silently drop a
    status they did receive."""
    seen: dict = {}

    async def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "running"}

    monkeypatch.setattr(server, "_run_comfy_async", fake_run)

    asyncio.run(server._job_status_async("pid", 5.0))

    assert seen["stdout_cap"] == server._JOB_STATUS_STDOUT_MAX_CHARS
    assert seen["stdout_cap"] > server._STDERR_MAX_CHARS
