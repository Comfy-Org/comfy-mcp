"""comfy-local-mcp — a thin MCP wrapper over comfy-cli.

Every tool shells out to the ``comfy`` command (comfy-cli), pinned to the LOCAL
target, asks for JSON, parses comfy-cli's versioned ``envelope/1`` result, and
returns its ``data``. There is deliberately no HTTP client and no code shared
with the Comfy Cloud MCP — comfy-cli is the engine.

Tools so far: the run -> get-output core loop plus job management
(``job_status`` / ``wait_for_job`` / ``watch_job`` / ``get_execution_error`` /
``cancel_job`` / ``get_queue``), the ``launch_comfyui`` / ``stop_comfyui`` /
``restart_comfyui`` lifecycle trio (``comfy launch --background`` /
``comfy stop`` / stop-then-launch), and the
``discover`` / ``which`` introspection pair (``comfy discover`` /
``comfy which``) that lets an agent learn the CLI's own contract and selection.

NOTE: the exact ``comfy`` invocation + envelope shape still need a smoke test
against a real comfy-cli install and a running local ComfyUI.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

from mcp.server.fastmcp import Context, FastMCP, Image

# Rides every client handshake — teach an agent the canonical flows up front so
# it does not have to rediscover them tool-by-tool. Keep this short.
INSTRUCTIONS = """\
This server drives a LOCAL ComfyUI through comfy-cli. Canonical flows:

- Call `server_info` FIRST to confirm a local ComfyUI is running before anything
  else.
- Long generations: submit non-blocking with `run_workflow(wait=False)` to get a
  `prompt_id`, poll `wait_for_job` (a short bounded wait — chain several) or
  `job_status` until it finishes, then collect files with `fetch_outputs`.
  Prefer this over `run_workflow(wait=True)` for slow runs so nothing blocks.
  For LIVE progress on an already-submitted job, `watch_job(prompt_id)` tails
  its execution events (bounded, like `wait_for_job`).
- Start from a template: `search_templates` to find one, `fetch_template` to save
  its workflow JSON, then `run_workflow` on that file. To change the prompt / seed
  / steps / model of a fetched template before running, inspect its tweakable slots
  with `list_workflow_slots` and edit them with `set_workflow_slot` (non-destructive
  by default) — the loop is `fetch_template` -> `set_workflow_slot` -> `run_workflow`.
- When custom nodes or models may be missing, pre-flight with `validate_workflow`
  before running.
- Manage in-flight work with `get_queue` (list jobs) and `cancel_job`.

Everything targets the LOCAL server only — there is no cloud access here.
"""

mcp = FastMCP("comfy-local-mcp", instructions=INSTRUCTIONS)

# Allow overriding the binary (e.g. a venv path) without touching code.
COMFY_BIN = os.environ.get("COMFY_BIN", "comfy")

# Hard ceiling for a single bounded watch so `float('inf')` / an absurd value
# can't hold a `comfy jobs watch` child open effectively forever (1 hour).
_MAX_WATCH_TIMEOUT = 3600.0


class ComfyCliError(RuntimeError):
    """comfy-cli was missing, timed out, or returned an error envelope."""


def _run_comfy(*args: str, timeout: float | None = None) -> Any:
    """Run ``comfy <args> --where local --json`` and return the envelope's ``data``.

    comfy-cli emits a versioned ``envelope/1`` object on stdout (a single line
    for ``--json``, or an NDJSON stream whose final line is the envelope). We
    keep the last JSON object and unwrap ``ok`` / ``data`` / ``error``.
    """
    if shutil.which(COMFY_BIN) is None:
        raise ComfyCliError(
            f"`{COMFY_BIN}` not found on PATH. Install comfy-cli "
            "(`pip install comfy-cli`) or set the COMFY_BIN env var."
        )
    # Global flags (--json, --where) MUST precede the subcommand in comfy-cli;
    # a trailing --json errors with "No such option". (Verified against comfy-cli.)
    cmd = [COMFY_BIN, "--json", "--where", "local", *args]
    # Belt-and-suspenders: pin the target via env too, so we never touch cloud.
    env = {**os.environ, "COMFY_WHERE": "local"}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ComfyCliError(
            f"comfy-cli timed out after {timeout}s: {' '.join(cmd)}"
        ) from exc

    return _unwrap_envelope(
        _last_json_object(proc.stdout), args, proc.returncode, proc.stderr
    )


def _unwrap_envelope(
    envelope: dict | None, args: tuple[str, ...], returncode: int, stderr: str
) -> Any:
    """Unwrap comfy-cli's ``envelope/1`` result, raising on error/absence.

    Shared by the plain (`--json`) and streaming (`--json-stream`) paths so both
    have identical terminal behavior: return ``data`` on success, and raise a
    :class:`ComfyCliError` carrying the envelope's ``error.code`` on failure.
    """
    if envelope is None:
        raise ComfyCliError(
            f"comfy-cli returned no JSON (exit {returncode}). "
            f"stderr: {stderr.strip()[:500]}"
        )
    if not envelope.get("ok", False):
        err = envelope.get("error") or {}
        raise ComfyCliError(
            f"comfy {' '.join(args)} failed "
            f"[{err.get('code', 'unknown')}]: "
            f"{err.get('message') or stderr.strip()[:500]}"
        )
    return envelope.get("data")


def _last_json_object(stdout: str) -> dict | None:
    """Return the last JSON object on stdout, preferring a ``type==envelope`` one."""
    best: dict | None = None
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "envelope":
            best = obj  # an explicit envelope always wins; keep the latest
        elif best is None or best.get("type") != "envelope":
            best = obj  # fallback to any JSON object until an envelope appears
    return best


def _parse_event(line: str) -> dict | None:
    """Parse one NDJSON stream line into a dict, or None if it isn't JSON."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class _StreamProgress:
    """Maps comfy-cli's ``--json-stream`` run events to MCP progress values.

    comfy-cli's run dialect (see comfy-cli ``execution.py``) emits, per line:
    a ``queued`` event carrying the workflow's node manifest, then per node an
    ``executing`` event, throttled ``progress`` events (per-node step counts,
    ~10Hz), and an ``executed`` / ``execution_cached`` event. We turn those into
    a single overall bar: ``total`` = node count from the manifest, and
    ``progress`` = fully-finished nodes plus the current node's step fraction, so
    the value climbs monotonically 0..total across the run.
    """

    def __init__(self) -> None:
        self.total: float | None = None  # node count (from the queued manifest)
        self.done = 0  # nodes fully executed or served from cache
        self._last = -1.0  # last value reported (kept non-decreasing)

    def snapshot(self) -> dict:
        """Last-known progress, for a bounded watch that timed out mid-run.

        ``progress`` is None until the first tick is reported (``_last`` starts
        below zero), so a timed-out payload never claims phantom progress.
        """
        return {
            "progress": self._last if self._last >= 0 else None,
            "total": self.total,
            "nodes_done": self.done,
        }

    async def report(self, ctx: Context | None, event: dict) -> None:
        """Advance tracker state from one stream event; notify via ``ctx`` if set.

        State (``total`` / ``done`` / ``_last``) is updated unconditionally so a
        bounded, ctx-less watch still reports real progress in its timed-out
        :meth:`snapshot`; the MCP notification is the only ctx-gated part.
        """
        etype = event.get("type")
        if etype == "queued":
            nodes = event.get("nodes")
            if isinstance(nodes, list) and nodes:
                self.total = float(len(nodes))
            progress, message = 0.0, "queued"
        elif etype == "executing":
            progress = float(self.done)
            message = f"executing {event.get('title') or event.get('node')}"
        elif etype in ("executed", "execution_cached"):
            self.done += 1
            progress = float(self.done)
            message = f"finished {event.get('title') or event.get('node')}"
        elif etype == "progress":
            completed = event.get("completed") or 0
            node_total = event.get("total") or 0
            frac = (completed / node_total) if node_total else 0.0
            progress = self.done + frac
            message = f"node {event.get('node')}: {completed}/{node_total}"
        else:
            return  # output / execution_error / unknown -> not a progress tick
        # MCP guidance: progress should not go backwards, even as nodes reset.
        progress = max(progress, self._last)
        self._last = progress
        if ctx is not None:
            await ctx.report_progress(
                progress=progress, total=self.total, message=message
            )


async def _run_comfy_streaming(
    *args: str,
    ctx: Context | None = None,
    timeout: float | None = None,
    raise_on_timeout: bool = True,
) -> Any:
    """Run ``comfy --json-stream --where local <args>`` and stream progress.

    Spawns comfy-cli with :class:`subprocess.Popen`, reads its NDJSON stdout
    line-by-line (each ``readline`` off-loaded to a thread so the event loop
    stays free), and forwards run events as MCP progress notifications via
    ``ctx.report_progress``. The final ``envelope/1`` line is unwrapped exactly
    as :func:`_run_comfy` does, so an error envelope raises
    :class:`ComfyCliError` with the same code — terminal behavior is unchanged.

    ``timeout`` bounds the whole stream. By default an expiry raises
    :class:`ComfyCliError` (the run-workflow contract); pass
    ``raise_on_timeout=False`` for a bounded *tail* that should instead return a
    ``{"timed_out": True, "status": <progress snapshot>}`` payload (mirroring
    :func:`wait_for_job`) rather than surface the deadline as an error.
    """
    if shutil.which(COMFY_BIN) is None:
        raise ComfyCliError(
            f"`{COMFY_BIN}` not found on PATH. Install comfy-cli "
            "(`pip install comfy-cli`) or set the COMFY_BIN env var."
        )
    # --json-stream is a global flag and, like --json/--where, MUST precede the
    # subcommand; a trailing form errors with "No such option".
    cmd = [COMFY_BIN, "--json-stream", "--where", "local", *args]
    env = {**os.environ, "COMFY_WHERE": "local"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    lines: list[str] = []
    tracker = _StreamProgress()

    async def _pump() -> None:
        assert proc.stdout is not None
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)
            if not line:  # EOF: comfy-cli closed stdout
                break
            lines.append(line)
            # Advance the tracker even without a ctx so a timed-out ctx-less
            # watch still returns real progress; report() no-ops the notify.
            event = _parse_event(line)
            if event is not None:
                await tracker.report(ctx, event)

    # Drain stderr concurrently so a chatty child can't deadlock on a full pipe.
    stderr_future = (
        asyncio.ensure_future(asyncio.to_thread(proc.stderr.read))
        if proc.stderr is not None
        else None
    )

    async def _drain() -> Any:
        # Read the whole stream, then reap the child and its stderr. Bounding
        # this entire coroutine (not just _pump) means a child that closes
        # stdout without exiting can't wedge the unbounded proc.wait/stderr read.
        await _pump()
        returncode = await asyncio.to_thread(proc.wait)
        stderr = (await stderr_future) if stderr_future is not None else ""
        return _unwrap_envelope(
            _last_json_object("".join(lines)), args, returncode, stderr
        )

    try:
        try:
            if timeout is not None:
                return await asyncio.wait_for(_drain(), timeout=timeout)
            return await _drain()
        except (asyncio.TimeoutError, TimeoutError) as exc:
            if not raise_on_timeout:
                # Bounded tail: report how far the run got instead of erroring
                # (the finally below still kills the child).
                return {"timed_out": True, "status": tracker.snapshot()}
            raise ComfyCliError(
                f"comfy-cli timed out after {timeout}s: {' '.join(cmd)}"
            ) from exc
    finally:
        # Never leave a stray child or a dangling stderr reader on any exit path
        # (timeout, a report_progress error, or normal completion).
        if proc.poll() is None:
            proc.kill()
            await asyncio.to_thread(proc.wait)
        if stderr_future is not None and not stderr_future.done():
            stderr_future.cancel()


@mcp.tool()
def server_info() -> Any:
    """Report the local ComfyUI / comfy-cli environment.

    Wraps ``comfy env``. Returns whether a local ComfyUI server is running and
    its URL, plus the selected workspace and Python info. Call this first to
    confirm a local ComfyUI is up before running a workflow.
    """
    return _run_comfy("env", timeout=60.0)


@mcp.tool()
async def run_workflow(
    workflow_path: str,
    wait: bool = True,
    timeout_seconds: float = 600.0,
    ctx: Context | None = None,
) -> Any:
    """Run a ComfyUI workflow JSON on the LOCAL ComfyUI.

    Accepts an API-format or UI-export workflow file. Wraps
    ``comfy run --workflow <path>``. With ``wait=True`` (default) this waits
    until the run finishes and returns the full result, streaming live progress
    as MCP progress notifications (per-node execution + sampler step counts) so
    a long generation is not a silent block; with ``wait=False`` it submits and
    returns immediately with a ``prompt_id`` to poll via ``job_status``.
    """
    if not wait:
        # Fire-and-return: no stream to follow, so keep the plain --json path.
        return _run_comfy("run", "--workflow", workflow_path, timeout=60.0)
    return await _run_comfy_streaming(
        "run",
        "--workflow",
        workflow_path,
        "--wait",
        ctx=ctx,
        timeout=timeout_seconds,
    )


@mcp.tool()
def job_status(prompt_id: str) -> Any:
    """Check a submitted job's status (queued / running / completed / error).

    Wraps ``comfy jobs status <prompt_id>``. Returns the job status and, when
    finished, its output references. Poll this after ``run_workflow(wait=False)``.
    """
    return _run_comfy("jobs", "status", prompt_id, timeout=60.0)


# How many trailing traceback frames survive into a get_execution_error verdict.
# A full ComfyUI traceback can run hundreds of frames; the tail carries the
# actual failure site. Mirrors comfy-cli's execution_errors._TRACEBACK_TAIL_FRAMES
# (a smaller tail there — this tool is the deliberate deep-dive companion).
_TRACEBACK_TAIL_FRAMES = 20

# Character cap on the joined traceback tail, so a pathological (megabyte)
# traceback can't dump into an agent's context. ``len()`` counts Unicode code
# points, not bytes; that's close enough for a context-size guard. Content is a
# Python traceback — no secret redaction is required, only a size bound.
_TRACEBACK_TAIL_MAX_CHARS = 8000

# Marker prepended to a truncated tail so the caller knows frames were dropped.
_TRACEBACK_TRUNCATION_MARKER = "...(truncated)"

# Character cap on free-text failure fields (``exception_message`` etc.). A
# hostile or buggy custom node can raise with a multi-megabyte message; bound it
# for the same context-bloat reason the traceback tail is capped.
_EXCEPTION_TEXT_MAX_CHARS = 8000

# Reported statuses that mean the run failed. Used to tell a genuinely healthy
# run apart from a failure that carried a falsy/empty `error` field, so the
# latter is not reported as `error: None`. Compared case-insensitively.
_ERROR_STATUSES = frozenset({"error", "failed", "failure"})


def _cap_traceback_tail(frames: list[str]) -> list[str]:
    """Bound the joined traceback tail to ``_TRACEBACK_TAIL_MAX_CHARS`` chars.

    Drops leading (oldest) frames until the remainder fits, prepending a
    ``"...(truncated)"`` marker so the caller knows frames were dropped. If a
    single frame alone exceeds the cap, its characters are hard-truncated (keep
    the tail — that's the failure site). The marker and its separator are
    charged to the budget, so the joined result stays within the cap. Returns
    the frames unchanged, with no marker, when already under the cap.
    """

    def joined_len(items: list[str]) -> int:
        # Newline-joined length: chars plus one separator between frames.
        return sum(len(f) for f in items) + max(0, len(items) - 1)

    frames = list(frames)
    if joined_len(frames) <= _TRACEBACK_TAIL_MAX_CHARS:
        return frames
    # Reserve room for the marker plus its trailing separator so the final
    # joined tail (marker + frames) never exceeds the documented cap.
    budget = max(0, _TRACEBACK_TAIL_MAX_CHARS - (len(_TRACEBACK_TRUNCATION_MARKER) + 1))
    while len(frames) > 1 and joined_len(frames) > budget:
        frames.pop(0)
    if frames and joined_len(frames) > budget:
        # One oversized frame remains; hard-cap its characters.
        frames = [frames[0][-budget:]] if budget else [""]
    return [_TRACEBACK_TRUNCATION_MARKER, *frames]


def _cap_text(value: Any, limit: int = _EXCEPTION_TEXT_MAX_CHARS) -> Any:
    """Bound a free-text failure field to ``limit`` chars.

    Non-strings (including ``None``) pass through untouched so the field's shape
    is preserved for callers that key off it; only oversized strings are cut and
    marked truncated.
    """
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + _TRACEBACK_TRUNCATION_MARKER
    return value


@mcp.tool()
def get_execution_error(prompt_id: str) -> Any:
    """Diagnostics companion to ``job_status``: the failure verdict for a run.

    Call this after a run reports failure (``job_status`` returning
    ``status: error``) to get the compact cause an agent needs to self-repair
    the workflow — the failing ``node_type``/``node_id``, the
    ``exception_type``/``exception_message``, and a bounded tail of the Python
    traceback — without digging it out of the large raw status blob. Wraps
    ``comfy jobs status <prompt_id>`` (the same source comfy-cli points at for
    the full traceback) and normalizes ComfyUI's raw ``execution_error`` payload
    from that snapshot's ``error`` field.

    On a healthy prompt (completed / queued / running — no ``error``) it returns
    ``{"prompt_id", "status", "error": None}`` rather than raising, so it is safe
    to call speculatively.
    """
    if prompt_id.startswith("-"):
        # comfy-cli parses a leading-dash positional as an option/flag; reject
        # it rather than let `jobs status` misread the id (argument injection).
        raise ComfyCliError(f"invalid prompt_id: {prompt_id!r} (leading '-')")

    status = _run_comfy("jobs", "status", prompt_id, timeout=60.0)

    error = status.get("error") if isinstance(status, dict) else None
    reported = status.get("status") if isinstance(status, dict) else None
    if not error:
        # No error payload. Distinguish a genuinely healthy run from a failed
        # one that reported an error status but a falsy/empty `error` field
        # ({}, "", 0): the latter must not masquerade as `error: None` and let a
        # caller treat the failure as healthy.
        reported_l = reported.strip().lower() if isinstance(reported, str) else None
        if reported_l in _ERROR_STATUSES:
            return {
                "prompt_id": prompt_id,
                "status": "error",
                "exception_message": None,
                "exception_type": None,
                "node_id": None,
                "node_type": None,
                "traceback_tail": [],
            }
        # Job completed, still queued/running, or an unexpected payload shape.
        return {"prompt_id": prompt_id, "status": reported, "error": None}

    # `error` is normally ComfyUI's execution_error dict, but tolerate a bare
    # string (some failure paths surface just a message) so the tool never
    # crashes on an unexpected shape — mirrors comfy-cli's parse_error_message.
    if not isinstance(error, dict):
        error = {"exception_message": str(error)}

    # Mirror comfy-cli's parse_error_message shape (execution_errors.py): flat
    # fields plus a tail of the traceback frames, with node_id coerced to str.
    node_id = error.get("node_id")
    traceback = error.get("traceback") or []
    if isinstance(traceback, str):
        traceback = [traceback]
    elif not isinstance(traceback, (list, tuple)):
        # Malformed payload (dict / int / etc.): a non-sequence would raise on
        # the slice below. Drop it rather than crash — the "never crashes on an
        # unexpected shape" contract wins over salvaging a garbage traceback.
        traceback = []
    traceback_tail = [str(frame) for frame in traceback[-_TRACEBACK_TAIL_FRAMES:]]
    traceback_tail = _cap_traceback_tail(traceback_tail)

    return {
        "prompt_id": prompt_id,
        "status": "error",
        "exception_message": _cap_text(error.get("exception_message")),
        "exception_type": _cap_text(error.get("exception_type")),
        "node_id": str(node_id) if node_id is not None else None,
        "node_type": _cap_text(error.get("node_type")),
        "traceback_tail": traceback_tail,
    }


# Statuses that mean a job is finished (no point polling further). comfy-cli
# surfaces ComfyUI's own states plus its wrapper's, so match generously and
# case-insensitively; anything else (queued / pending / running) keeps polling.
_TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "complete",
        "success",
        "succeeded",
        "done",
        "error",
        "failed",
        "cancelled",
        "canceled",
    }
)


def _is_terminal(status: Any) -> bool:
    """True if a ``jobs status`` payload reports a finished state."""
    if isinstance(status, dict):
        value = status.get("status")
        if isinstance(value, str):
            return value.lower() in _TERMINAL_STATUSES
    return False


@mcp.tool()
def wait_for_job(prompt_id: str, timeout_seconds: float = 25.0) -> Any:
    """Wait (bounded) for a submitted LOCAL job to reach a terminal status.

    Polls ``comfy jobs status <prompt_id>`` with a short sleep between polls
    until the job finishes (completed / error / cancelled) or
    ``timeout_seconds`` elapses. Returns the final status payload on completion,
    or ``{"timed_out": True, "status": <last payload>}`` on expiry. The wait is
    bounded by design — chain several short ``wait_for_job`` calls (checking
    ``job_status`` in between) rather than issuing one long block. Use after
    ``run_workflow(wait=False)``.
    """
    deadline = time.monotonic() + timeout_seconds
    poll_interval = 2.0
    last: Any = None
    while True:
        last = _run_comfy("jobs", "status", prompt_id, timeout=60.0)
        if _is_terminal(last):
            return last
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"timed_out": True, "status": last}
        time.sleep(min(poll_interval, remaining))


@mcp.tool()
async def watch_job(
    prompt_id: str,
    timeout_seconds: float = 600.0,
    ctx: Context | None = None,
) -> Any:
    """Tail a submitted LOCAL job's live execution, streaming progress.

    Wraps ``comfy jobs watch <prompt_id>``, which follows a job's execution
    events (per-node execution + sampler step counts) and ends on the terminal
    envelope. Runs through the same streaming machinery as
    ``run_workflow(wait=True)``, forwarding those events as MCP progress
    notifications, and returns the final result ``data`` on completion.

    Use this to get LIVE progress on a job already submitted with
    ``run_workflow(wait=False)`` — the streaming counterpart to the polled
    ``wait_for_job``. The wait is bounded by ``timeout_seconds`` (clamped to a
    sane maximum) so it can never block forever; on expiry it returns the same
    ``{"timed_out": True, "status": ...}`` envelope shape as ``wait_for_job``,
    except ``status`` here carries a live progress snapshot
    (``{progress, total, nodes_done}``) rather than a raw ``jobs status`` dict.
    """
    if prompt_id.startswith("-"):
        # comfy-cli parses a leading-dash positional as an option/flag; reject
        # it rather than let `jobs watch` misread the id (argument injection).
        raise ComfyCliError(f"invalid prompt_id: {prompt_id!r} (leading '-')")
    timeout_seconds = min(max(timeout_seconds, 0.0), _MAX_WATCH_TIMEOUT)
    return await _run_comfy_streaming(
        "jobs",
        "watch",
        prompt_id,
        ctx=ctx,
        timeout=timeout_seconds,
        raise_on_timeout=False,
    )


@mcp.tool()
def cancel_job(prompt_id: str) -> Any:
    """Cancel a queued or running LOCAL job.

    Wraps ``comfy jobs cancel <prompt_id>``. Use this to stop a job you
    submitted via ``run_workflow(wait=False)`` before it finishes; cancelling an
    unknown or already-finished ``prompt_id`` surfaces comfy-cli's error envelope.
    """
    return _run_comfy("jobs", "cancel", prompt_id, timeout=60.0)


@mcp.tool()
def get_queue() -> Any:
    """List known LOCAL jobs with their status (pending / running / completed).

    Wraps ``comfy jobs ls``. comfy-cli merges its on-disk job state with the
    running ComfyUI server's queue, so this returns both jobs still in the queue
    and recently completed ones — call it to find a ``prompt_id`` to inspect with
    ``job_status`` or cancel with ``cancel_job``.
    """
    return _run_comfy("jobs", "ls", timeout=60.0)


# Image suffixes we return inline from ``fetch_outputs`` — kept to the formats
# ``mcp.server.fastmcp.Image`` maps to a real ``image/*`` MIME type (an unknown
# suffix would fall back to ``application/octet-stream`` and not render).
_INLINE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def _iter_strings(obj: Any) -> Any:
    """Yield every string value nested anywhere inside ``obj`` (dicts/lists/scalars)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_strings(value)


def _collect_output_images(data: Any, out_dir: str) -> list[str]:
    """Resolve image files referenced by ``comfy download``'s data to on-disk paths.

    Walks every string in the envelope ``data``, keeps those with an image
    suffix, resolves each against ``out_dir`` when it is not already an existing
    absolute/relative path, and returns the files that actually exist on disk
    (deduped, order-preserving). Anything that does not resolve to a real file is
    skipped — inline return is best-effort and never masks the on-disk copy.
    """
    resolved: dict[str, None] = {}
    for value in _iter_strings(data):
        if not value.lower().endswith(_INLINE_IMAGE_SUFFIXES):
            continue
        # Try the path as given, then relative to out_dir, then just its
        # basename inside out_dir — comfy-cli may report absolute paths, paths
        # relative to out_dir, or bare filenames depending on the invocation.
        for candidate in (
            value,
            os.path.join(out_dir, value),
            os.path.join(out_dir, os.path.basename(value)),
        ):
            if os.path.isfile(candidate):
                resolved.setdefault(os.path.abspath(candidate), None)
                break
    return list(resolved)


@mcp.tool()
def fetch_outputs(
    prompt_id: str,
    out_dir: str,
    url_only: bool = False,
    inline_images: bool = False,
) -> Any:
    """Download a completed LOCAL job's output files into ``out_dir``.

    Thin passthrough to ``comfy download <prompt_id> --where local -o <out_dir>``:
    comfy-cli resolves the job's outputs and writes them into ``out_dir``, so
    there is no hand-rolled HTTP client here. (The ``--where local`` flag is
    supplied by :func:`_run_comfy` as a global flag.) Pass ``url_only=True`` to
    add ``--url-only`` — comfy-cli then emits the output URLs without downloading,
    handy for handing URLs to other tools instead of copying bytes.

    Pass ``inline_images=True`` to ALSO return the copied images as inline MCP
    image content (base64) so the calling agent can see the result without a
    second read — the on-disk copy into ``out_dir`` is unchanged either way. In
    that mode the return is a list whose first element is comfy-cli's usual
    metadata and whose remaining elements are the image files just written; a
    non-image output (or ``url_only=True``, which downloads no bytes) simply
    yields no inline images.
    """
    args = ["download", prompt_id, "-o", out_dir]
    if url_only:
        args.append("--url-only")
    data = _run_comfy(*args, timeout=300.0)
    if not inline_images:
        return data
    images = [Image(path=path) for path in _collect_output_images(data, out_dir)]
    return [data, *images]


@mcp.tool()
def launch_comfyui(extra_args: list[str] | None = None) -> Any:
    """Start the LOCAL ComfyUI server, detached, and return once it is up.

    Wraps ``comfy launch --background``, which boots ComfyUI as a background
    process and records its pid so ``stop_comfyui`` can later shut it down. Any
    ``extra_args`` are forwarded to ComfyUI itself after a ``--`` separator
    (e.g. ``["--port", "8189"]`` -> ``comfy launch --background -- --port 8189``).
    The timeout is generous because the first boot loads torch and can take a
    while.

    Call ``server_info`` first if you only want to check whether a server is
    already running — launching a second one will fail on the port.

    NOTE (temporary upstream caveat): ``comfy launch --background`` currently
    crashes on Python 3.14 (comfy-cli asyncio ``get_event_loop`` issue; a fix is
    in review upstream). On affected comfy-cli versions the crash surfaces here
    as a clean :class:`ComfyCliError` from the error envelope. Remove this note
    once the upstream fix ships.
    """
    args = ["launch", "--background"]
    if extra_args:
        args += ["--", *extra_args]
    return _run_comfy(*args, timeout=180.0)


@mcp.tool()
def stop_comfyui() -> Any:
    """Stop the LOCAL ComfyUI server that comfy-cli launched.

    Wraps ``comfy stop``. Ownership semantics: comfy-cli only kills the pid it
    recorded when IT launched the server via ``launch_comfyui`` /
    ``comfy launch --background``. It therefore cannot stop a ComfyUI started by
    the desktop app or by hand — in that case comfy-cli reports it has no
    recorded server and this tool raises a :class:`ComfyCliError` carrying that
    message, rather than killing an unrelated process.
    """
    return _run_comfy("stop", timeout=60.0)


@mcp.tool()
def restart_comfyui(extra_args: list[str] | None = None) -> Any:
    """Restart the LOCAL ComfyUI server: stop the running one, then launch a fresh one.

    Composes the existing :func:`stop_comfyui` and :func:`launch_comfyui` — there
    is no ``comfy restart`` subcommand, so this is a thin stop-then-launch over
    comfy-cli, not a new engine feature. ``extra_args`` are forwarded to the new
    ComfyUI exactly as :func:`launch_comfyui` forwards them (after a ``--``
    separator), so a restart is also how you relaunch with different flags.
    Returns the new server's status (``launch_comfyui``'s envelope data).

    The stop step is best-effort: if comfy-cli has no recorded server to stop
    (e.g. nothing is running, or ComfyUI was started outside comfy-cli), that
    ``ComfyCliError`` is swallowed and the launch proceeds — a restart should
    still bring the server up. Any genuine problem the failed stop would cause
    (such as the old process still holding the port) surfaces from the launch.
    """
    try:
        stop_comfyui()
    except ComfyCliError:
        pass
    return launch_comfyui(extra_args)


@mcp.tool()
def discover() -> Any:
    """Return comfy-cli's self-describing command surface (its own contract).

    Wraps ``comfy discover``. comfy-cli emits a machine-readable description of
    itself — the available commands, their argument schemas, and the error codes
    they can return — so an agent can learn the CLI's contract at runtime instead
    of hard-coding it. Returns that description verbatim.
    """
    return _run_comfy("discover", timeout=60.0)


@mcp.tool()
def which() -> Any:
    """Report which ComfyUI install/workspace comfy-cli currently targets.

    Wraps ``comfy which``. A lightweight "which one is selected?" answer; note
    that ``server_info`` (``comfy env``) already reports the same selected
    workspace alongside the running-server and Python details, so reach for this
    only when the bare selection is all you want.
    """
    return _run_comfy("which", timeout=60.0)


def _template_matches(item: Any, query_lower: str) -> bool:
    """True if ``query_lower`` (already lowercased) is a substring of a template entry.

    Handles both shapes comfy-cli might emit for a template: a bare name string,
    or a dict of fields (name / title / description / …) — we match against every
    string value so the query hits any of them.
    """
    if isinstance(item, str):
        return query_lower in item.lower()
    if isinstance(item, dict):
        return any(
            isinstance(v, str) and query_lower in v.lower() for v in item.values()
        )
    return False


@mcp.tool()
def search_templates(query: str = "") -> Any:
    """Search the built-in ComfyUI workflow templates by name/description.

    Wraps ``comfy templates ls``. When ``query`` is non-empty the listing is
    filtered client-side (case-insensitive substring match against each
    template's name and any text fields) — comfy-cli's ``ls`` has no server-side
    filter argument, so the narrowing happens here; an empty ``query`` returns
    the full list.

    Step 1 of the template on-ramp: pick a ``name`` from the results, inspect it
    with ``get_template(name)``, then ``fetch_template(name, out_path)`` to write
    a runnable workflow JSON and pass that path straight to ``run_workflow`` — a
    working generation without hand-authoring workflow JSON.
    """
    data = _run_comfy("templates", "ls", timeout=60.0)
    if not query:
        return data
    q = query.lower()
    if isinstance(data, list):
        return [item for item in data if _template_matches(item, q)]
    if isinstance(data, dict) and isinstance(data.get("templates"), list):
        return {
            **data,
            "templates": [t for t in data["templates"] if _template_matches(t, q)],
        }
    # Unknown shape — return it unfiltered rather than silently dropping data.
    return data


@mcp.tool()
def get_template(name: str) -> Any:
    """Show one template's details/schema (inputs, description, node graph).

    Wraps ``comfy templates show <name>``, using a ``name`` from
    ``search_templates``. Step 2 of the on-ramp: inspect a template before
    fetching it, then ``fetch_template(name, out_path)`` writes the runnable JSON
    for ``run_workflow``.
    """
    return _run_comfy("templates", "show", name, timeout=60.0)


@mcp.tool()
def fetch_template(name: str, out_path: str) -> str:
    """Write a template's runnable workflow JSON to ``out_path``; return its absolute path.

    Wraps ``comfy templates fetch <name> --out <path>``, which materializes the
    template as a workflow JSON file on disk. Returns the ABSOLUTE path so it can
    be passed straight to ``run_workflow(workflow_path=...)``, completing the
    template on-ramp::

        search_templates("flux")               # find a template
        get_template("flux_dev")               # inspect it
        path = fetch_template("flux_dev", "/tmp/flux.json")
        run_workflow(path)                      # generate — no hand-authored JSON

    so an agent reaches a working generation without hand-authoring workflow JSON.
    """
    _run_comfy("templates", "fetch", name, "--out", out_path, timeout=60.0)
    return os.path.abspath(out_path)


@mcp.tool()
def search_nodes(query: str) -> Any:
    """Search node classes in the LOCAL ComfyUI's live ``object_info``.

    Wraps ``comfy nodes search <query>``. Because the catalog is read from the
    user's running install, results include their INSTALLED custom nodes — not a
    static/bundled catalog. Use this to find the class name of a node (e.g.
    "KSampler", "load image") before authoring or repairing a workflow graph;
    pass the returned name to ``get_node`` for its full schema.
    """
    return _run_comfy("nodes", "search", query, timeout=60.0)


@mcp.tool()
def get_node(name: str) -> Any:
    """Return one node class's full input/output schema from the live local catalog.

    Wraps ``comfy nodes show <ClassName>``. ``name`` is the node's class name
    (as returned by ``search_nodes``). The schema — required/optional inputs,
    their types and defaults, and outputs — is what an agent needs to author or
    repair a workflow graph. Reflects the user's live install, so it resolves
    custom-node classes too (not just built-ins).
    """
    return _run_comfy("nodes", "show", name, timeout=60.0)


@mcp.tool()
def list_nodes(
    produces: str = "",
    accepts: str = "",
    category: str = "",
    pack: str = "",
    label: str = "",
) -> Any:
    """List node classes from the live local ``object_info``, with optional filters.

    Wraps ``comfy nodes ls``. Each argument, when non-empty, adds the matching
    filter flag (empty ones are omitted, so a bare call lists everything):

    - ``produces`` → ``--produces <TYPE>``: nodes whose outputs include ``<TYPE>``
      (e.g. ``IMAGE``, ``MODEL``).
    - ``accepts`` → ``--accepts <TYPE>``: nodes with an input of ``<TYPE>``.
    - ``category`` → ``--category <path>``: nodes under a menu category
      (e.g. ``loaders``).
    - ``pack`` → ``--pack <name>``: nodes from a given custom-node pack.
    - ``label`` → ``--label <text>``: nodes matching a display-label substring.

    Reads the user's live install, so results include installed custom nodes —
    the broad "what nodes can do X?" companion to ``search_nodes``' name search.
    """
    args = ["nodes", "ls"]
    for flag, value in (
        ("--produces", produces),
        ("--accepts", accepts),
        ("--category", category),
        ("--pack", pack),
        ("--label", label),
    ):
        if value:
            args += [flag, value]
    return _run_comfy(*args, timeout=60.0)


@mcp.tool()
def nodes_upstream(name: str, limit: int | None = None) -> Any:
    """List node classes whose outputs can feed ``name``'s inputs.

    Wraps ``comfy nodes upstream <name> [--limit N]``. Answers "what can I wire
    INTO this node?" — the candidates that produce the types ``name`` accepts,
    computed against the live local ``object_info`` (custom nodes included). Pass
    ``limit`` to cap the number of results; omit it for the full set.
    """
    args = ["nodes", "upstream", name]
    if limit is not None:
        args += ["--limit", str(limit)]
    return _run_comfy(*args, timeout=60.0)


@mcp.tool()
def nodes_downstream(name: str, limit: int | None = None) -> Any:
    """List node classes that accept ``name``'s output types.

    Wraps ``comfy nodes downstream <name> [--limit N]``. Answers "what can I wire
    this node INTO?" — the candidates whose inputs accept the types ``name``
    produces, computed against the live local ``object_info`` (custom nodes
    included). Pass ``limit`` to cap the number of results; omit it for the full
    set.
    """
    args = ["nodes", "downstream", name]
    if limit is not None:
        args += ["--limit", str(limit)]
    return _run_comfy(*args, timeout=60.0)


@mcp.tool()
def nodes_path(
    from_type: str, to_type: str, max_depth: int = 6, max_paths: int = 10
) -> Any:
    """Find node chains that route a value from ``from_type`` to ``to_type``.

    Wraps ``comfy nodes path <FROM> <TO> --max-depth N --max-paths N``. Given two
    connection types (e.g. ``MODEL`` → ``IMAGE``), returns sequences of nodes
    whose wiring carries a value from ``from_type`` to ``to_type`` over the live
    local ``object_info`` graph. ``max_depth`` bounds the chain length and
    ``max_paths`` caps how many routes are returned.
    """
    return _run_comfy(
        "nodes",
        "path",
        from_type,
        to_type,
        "--max-depth",
        str(max_depth),
        "--max-paths",
        str(max_paths),
        timeout=60.0,
    )


@mcp.tool()
def nodes_types() -> Any:
    """List every connection type in the live local graph, ranked by connectivity.

    Wraps ``comfy nodes types``. Returns the set of edge types (``MODEL``,
    ``IMAGE``, ``LATENT``, ``CONDITIONING``, …) present across the user's
    installed nodes, ordered by how connective each is — the vocabulary you wire
    with. Reflects custom nodes, so install-specific types show up too.
    """
    return _run_comfy("nodes", "types", timeout=60.0)


@mcp.tool()
def nodes_categories() -> Any:
    """Return the node category tree from the live local ``object_info``.

    Wraps ``comfy nodes categories``. Gives the menu-category hierarchy the
    user's installed nodes fall under — a map for browsing what is available by
    area (loaders, sampling, image, …) rather than by name. Reflects the live
    install, so custom-node categories appear too.
    """
    return _run_comfy("nodes", "categories", timeout=60.0)


@mcp.tool()
def search_models(query: str = "", folder: str = "") -> Any:
    """Search / list model files available to the LOCAL ComfyUI install.

    Thin passthrough with three modes, in precedence order:

    - ``query`` given → ``comfy models search <query>`` (match model filenames).
    - else ``folder`` given → ``comfy models list-folder <folder>`` (list one
      model folder, e.g. ``checkpoints``, ``loras``).
    - else (both empty) → ``comfy models list-folders`` (list the folder names).

    LOCAL DEGRADATION: unlike the cloud catalog, this returns only what is on
    disk — filenames, with no enrichment (no base-model / hash / description /
    download metadata). Agents should set expectations accordingly: it answers
    "which model files does this install have?", not "tell me about this model".
    """
    if query:
        return _run_comfy("models", "search", query, timeout=60.0)
    if folder:
        return _run_comfy("models", "list-folder", folder, timeout=60.0)
    return _run_comfy("models", "list-folders", timeout=60.0)


@mcp.tool()
def download_model(
    url: str, relative_path: str | None = None, filename: str | None = None
) -> Any:
    """Download a model file into the LOCAL ComfyUI models dir, by URL.

    Wraps ``comfy model download --url <url> [--relative-path <path>]
    [--filename <name>]`` (note the SINGULAR ``model`` verb group — the download
    engine — distinct from the plural ``models`` catalog that ``search_models``
    reads). comfy-cli understands HuggingFace and CivitAI URLs; any access
    tokens are configured out-of-band via comfy-cli / environment variables and
    are NOT passed through this tool. The file lands in the workspace models
    directory, optionally under ``relative_path`` (e.g. ``models/loras`` to place
    a LoRA in the right folder) and optionally renamed via ``filename``.

    DOWNLOAD-BY-URL ONLY: this is a fetch of a known URL, not a hub search —
    there is no HuggingFace/CivitAI browse or discovery here (comfy-cli has no
    such search), so the caller must already have the direct model URL. Returns
    comfy-cli's envelope ``data`` (the saved path / download metadata).
    """
    # comfy-cli parses a leading-dash value as an option/flag; reject any so a
    # crafted argument can't be smuggled in as a CLI flag (argument injection).
    if url.startswith("-"):
        raise ComfyCliError(f"invalid url: {url!r} (leading '-')")
    # Restrict to http(s): this is a remote fetch of a known model URL, so a
    # `file://` path or other scheme — an SSRF / local-file-read primitive whose
    # body would be written straight into the models dir — is never legitimate.
    if urlparse(url).scheme.lower() not in ("http", "https"):
        raise ComfyCliError(f"invalid url: {url!r} (scheme must be http/https)")
    # Optional args are treated as unset when falsy (None or ""), so an explicit
    # empty string is omitted rather than forwarded as `--relative-path ""`.
    if relative_path:
        if relative_path.startswith("-"):
            raise ComfyCliError(
                f"invalid relative_path: {relative_path!r} (leading '-')"
            )
        # relative_path is a models-dir SUBFOLDER (e.g. `models/loras`); keep the
        # write inside the models dir by rejecting absolute paths and `..`.
        parts = relative_path.replace("\\", "/").split("/")
        if os.path.isabs(relative_path) or ".." in parts:
            raise ComfyCliError(
                f"invalid relative_path: {relative_path!r} (path traversal)"
            )
    if filename:
        if filename.startswith("-"):
            raise ComfyCliError(f"invalid filename: {filename!r} (leading '-')")
        # filename is a single output name, not a path; reject separators and `..`
        # so it can't redirect the write out of the target directory.
        if filename in (".", "..") or "/" in filename or "\\" in filename:
            raise ComfyCliError(
                f"invalid filename: {filename!r} (must be a bare filename)"
            )
    args = ["model", "download", "--url", url]
    if relative_path:
        args += ["--relative-path", relative_path]
    if filename:
        args += ["--filename", filename]
    # Generous timeout: multi-GB checkpoints can take a long time to fetch.
    return _run_comfy(*args, timeout=1800.0)


@mcp.tool()
def upload_file(paths: list[str], overwrite: bool = False) -> Any:
    """Upload local files into the LOCAL ComfyUI ``input`` directory.

    Wraps ``comfy upload <files...> [--overwrite]``. Use this to stage source
    images/masks a workflow references by filename before running it — it is
    what unlocks img2img / inpaint workflows on a local ComfyUI. Pass
    ``overwrite=True`` to replace files that already exist in the input dir
    (otherwise comfy-cli skips or errors on collisions).
    """
    args = ["upload", *paths]
    if overwrite:
        args.append("--overwrite")
    return _run_comfy(*args, timeout=300.0)


@mcp.tool()
def validate_workflow(workflow_path: str) -> Any:
    """Pre-flight a workflow against the live local ComfyUI before running it.

    Wraps ``comfy validate --workflow <path>``. Checks the workflow's
    class_types, input shapes, enum values and wiring against the running
    ComfyUI's ``object_info`` and returns the validation result — cheap
    insurance before a slow ``run_workflow``. On an invalid workflow this
    raises :class:`ComfyCliError` carrying comfy-cli's structured error code
    (e.g. ``workflow_unknown_nodes``) and message, so a missing-node or
    missing-model problem stays actionable instead of failing deep inside a run.
    """
    return _run_comfy("validate", "--workflow", workflow_path, timeout=60.0)


@mcp.tool()
def list_workflow_slots(workflow_path: str) -> Any:
    """List the agent-tweakable slots a frontend-format workflow exposes.

    Wraps ``comfy workflow slots <path>``. A "slot" is a parameter comfy-cli
    surfaces as a stable ``ADDR`` (e.g. the positive prompt text, a seed, step
    count, or model name) together with its current value, so an agent can see
    what a template exposes without hand-reading the raw workflow JSON. Operates
    on the frontend-format (UI export) workflow that ``fetch_template`` writes and
    ``run_workflow`` accepts. Pass a slot's ``ADDR`` back to ``set_workflow_slot``
    (or ``vary_workflow``) to change it.
    """
    return _run_comfy("workflow", "slots", workflow_path, timeout=60.0)


@mcp.tool()
def set_workflow_slot(
    workflow_path: str, overrides: list[str], stdout: bool = True
) -> Any:
    """Set one or more slot values on a frontend-format workflow.

    Wraps ``comfy workflow set-slot <path> ADDR=VALUE [ADDR=VALUE ...]``, where
    ``overrides`` is a list of ``"ADDR=VALUE"`` strings (the ``ADDR``s come from
    ``list_workflow_slots``). This is the parameterize step of the template
    on-ramp — change the prompt / seed / steps / model of a fetched template
    without hand-editing its JSON.

    ``stdout`` defaults to ``True`` (``--stdout``), so the tool is
    NON-DESTRUCTIVE: comfy-cli returns the modified workflow instead of mutating
    ``workflow_path`` in place. Set ``stdout=False`` to write the change back to
    the file. Canonical loop::

        path = fetch_template("flux_dev", "/tmp/flux.json")
        modified = set_workflow_slot(path, ["6.text=a red bicycle", "3.seed=42"])
        # write `modified` to disk (or call with stdout=False), then run_workflow
    """
    args = ["workflow", "set-slot", workflow_path, *overrides]
    if stdout:
        args.append("--stdout")
    return _run_comfy(*args, timeout=60.0)


@mcp.tool()
def vary_workflow(
    workflow_path: str, slots: list[str], out_dir: str | None = None
) -> Any:
    """Fan a frontend-format workflow out into variants over slot value lists.

    Wraps ``comfy workflow vary <path> --slot "ADDR=[v1,v2,...]" [--slot ...]``.
    ``slots`` is a list of ``"ADDR=[v1,v2,...]"`` strings, one per address (the
    ``ADDR``s come from ``list_workflow_slots``); comfy-cli ZIPS the value lists,
    so every list MUST be the same length — e.g. ``["3.seed=[1,2,3]",
    "6.text=[cat,dog,fish]"]`` yields three variants pairing seed 1/cat, 2/dog,
    3/fish.

    With ``out_dir`` unset (default) comfy-cli emits the variants as NDJSON to
    stdout; set ``out_dir`` to instead write ``<stem>_<N>.json`` files there (and
    forward ``--out-dir``). Run each variant with ``run_workflow`` to sweep a
    parameter grid.
    """
    args = ["workflow", "vary", workflow_path]
    for slot in slots:
        args += ["--slot", slot]
    if out_dir:
        args += ["--out-dir", out_dir]
    return _run_comfy(*args, timeout=120.0)


def main() -> None:
    """Entry point: serve the MCP over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
