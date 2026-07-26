"""comfy-local-mcp — a thin MCP wrapper over comfy-cli.

Every tool shells out to the ``comfy`` command (comfy-cli), pinned to the LOCAL
target (``--where local``, defaulting to ComfyUI on ``127.0.0.1:8188``), asks
for JSON, parses comfy-cli's versioned ``envelope/1`` result, and returns its
``data``. The run/queue tools can be pointed at a ComfyUI running ELSEWHERE by
setting ``COMFYUI_URL`` / ``COMFYUI_HOST`` (see ``_comfy_target``), which
forwards ``--host`` / ``--port`` to comfy-cli. A LOCAL ComfyUI on a non-default
address (e.g. ``:8189``) instead needs no code here at all: ``COMFY_LOCAL_URL``
rides the environment passthrough (see ``_comfy_env``) and is resolved by
comfy-cli, which ranks a ``--host``/``--port`` flag above ``COMFY_LOCAL_URL``,
that above a background record, and ``127.0.0.1:8188`` last. There is
deliberately no HTTP client and no code shared with the Comfy Cloud MCP —
comfy-cli is the engine.

Tools so far: the run -> get-output core loop plus job management
(``job_status`` / ``wait_for_job`` / ``watch_job`` / ``get_execution_error`` /
``cancel_job`` / ``get_queue``), the ``launch_comfyui`` / ``stop_comfyui`` /
``restart_comfyui`` lifecycle trio (``comfy launch --background`` /
``comfy stop`` / stop-then-launch) with ``get_logs`` (``comfy logs``) to read a
detached launch's captured output, and the
``discover`` / ``which`` introspection pair (``comfy discover`` /
``comfy which``) that lets an agent learn the CLI's own contract and selection.
``partner_generate`` (``comfy generate <model>``) reaches the hosted PARTNER
models; it spends credits, so comfy-cli's own consent interlock gates it and
this wrapper only passes that consent through (``--yes``) when the USER granted
it for that call — asked per call over MCP elicitation, or pre-authorized in
comfy-cli's own config. The durable "always proceed" stays engine-side, so this
server holds no spend state of its own.

Requires comfy-cli >= 1.12.0 (the ``comfy logs`` verb + the ``envelope/1``
contract): :func:`_run_comfy` guards this once, up front, with an actionable
upgrade error so a stale install fails clearly rather than cryptically.

NOTE: the exact ``comfy`` invocation + envelope shape still need a smoke test
against a real comfy-cli install and a running local ComfyUI.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any
from urllib.parse import urlparse

from mcp import types
from mcp.server.fastmcp import Context, FastMCP, Image
from pydantic import BaseModel, Field

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
- Start from a template: `search_templates(query=...)` to find one (free-text
  search, paged 25 at a time via `limit`/`offset`; narrow with `tag`/`type`/
  `model`/`provider`, or `exclude_api=True` for templates that run without a
  hosted-API key), `fetch_template` to save its workflow JSON, then `run_workflow`
  on that file. To change the prompt / seed
  / steps / model of a fetched template before running, inspect its tweakable slots
  with `list_workflow_slots` and edit them with `set_workflow_slot` (non-destructive
  by default) — the loop is `fetch_template` -> `set_workflow_slot` -> `run_workflow`.
  For a one-shot run, `run_template(name, params=...)` does fetch + fill + run in a
  single call; a template that embeds partner (paid) nodes spends credits and is
  gated by the same `confirm_spend` flag as `partner_generate` (free templates ignore it).
  For the quickest path from text to an image, `generate_image(prompt)` runs the
  default local text-to-image template through that same verb — free, no API key.
- When custom nodes or models may be missing, pre-flight with `validate_workflow`
  before running.
- Manage in-flight work with `get_queue` (list jobs) and `cancel_job`.
- Before running a workflow whose nodes call partner APIs (Seedream / Veo /
  Kling / Gemini / …), call `auth_status` to check Comfy Cloud credentials.
  Treat credentials as GOOD if `signed_in` is true OR
  `registration_env_key_present` is true — a registration-env key authenticates
  partner-API runs even though whoami can't see it, so do NOT nag the user to
  re-auth in that case. Only when BOTH are false, tell the USER to
  authenticate, in this order: (1) run `comfy cloud login` in a terminal
  (canonical), or (2) set `COMFY_API_KEY` in the MCP client's registration env,
  or (3) persist a key with `comfy auth set comfy-cloud-api-key --key <KEY>`.
  Never put a key in a workflow file. If a run still hits a credential error
  despite good `auth_status`, it is retried briefly and surfaces a hint with
  alternatives.
- After a detached `launch_comfyui`, read the background server's own output with
  `get_logs` — it tails the captured ComfyUI log (invisible otherwise).
- Hosted PARTNER models (Flux / Ideogram / DALL·E / …) run via `partner_generate`,
  which SPENDS the user's Comfy credits — local `run_workflow` / `generate_image`
  runs are free. Every call confirms the spend with the USER first: on a client
  that supports MCP elicitation you will be shown a confirmation prompt, and a
  decline cancels the call without spending. On a client that cannot elicit,
  comfy-cli's gate fails closed and the call errors unless you pass
  `confirm_spend=True` — set that ONLY when the user has actually agreed to
  spend credits for that call, never just to clear the error, and never because
  the host granted blanket permission to call the tool. A user who prefers not
  to be asked persists it engine-side with `comfy generate consent always`.

Everything targets the LOCAL server only — there is no cloud access here.
"""

mcp = FastMCP("comfy-local-mcp", instructions=INSTRUCTIONS)

# Allow overriding the binary (e.g. a venv path) without touching code. The
# companion address override needs no constant here: a LOCAL ComfyUI on a
# non-default address is selected with ``COMFY_LOCAL_URL``, which comfy-cli
# reads straight off the environment ``_comfy_env`` forwards (precedence:
# comfy-cli flags > env > background record > ``127.0.0.1:8188``).
COMFY_BIN = os.environ.get("COMFY_BIN", "comfy")

# --- opt-in local failure log -------------------------------------------------
# `COMFY_LOCAL_MCP_DEBUG_LOG` turns on a rotating, local-only JSONL record of
# every comfy-cli failure this server surfaces, so a tester can zip up a durable
# diagnostic trail instead of scraping an MCP client's transcript after the fact.
# Failure-only (a successful call writes nothing), local-only (nothing is
# transmitted anywhere), and OFF by default — while disabled there are ZERO
# filesystem effects: no directory is created and no handler is ever opened.
#
#   unset / "" / "0"  -> disabled (the default)
#   "1"               -> enabled at the per-OS default path
#   anything else     -> enabled, and the value IS the log file path
_FAILURE_LOG_ENV = "COMFY_LOCAL_MCP_DEBUG_LOG"

# Chars kept per stream tail, deliberately far larger than the
# `_MAX_ERROR_FIELD_CHARS` cap an error *message* carries: preserving more output
# than the message can is this log's whole reason to exist.
_FAILURE_LOG_TAIL_CHARS = 4000

# Cap on the recorded `message` (the sentence the MCP client displayed) so a
# pathological envelope cannot write an unbounded line.
_FAILURE_LOG_MESSAGE_CHARS = 2000

# Rotation, via the stdlib handler: 1 MiB per file plus two older generations,
# so the log is self-limiting at ~3 MiB no matter how long a tester leaves it on.
_FAILURE_LOG_MAX_BYTES = 1_048_576
_FAILURE_LOG_BACKUPS = 2

# A dedicated logger with `propagate = False`. Non-propagation is not cosmetic:
# this is an MCP **stdio** server, so stdout is the protocol transport, and a
# record reaching a default root handler could corrupt the session.
_FAILURE_LOGGER_NAME = "comfy_local_mcp.failures"


def _default_failure_log_path() -> str:
    """Per-OS default path for the failure log. Creates nothing.

    Mirrors comfy-cli's own local-state convention (its ``constants.py``
    ``DEFAULT_CONFIG``) with a ``comfy-local-mcp`` leaf, hand-rolled rather than
    imported: comfy-cli is the *engine* this server shells out to, not a Python
    dependency of it (``mcp`` is the only one), so there is nothing to import.

    Deliberately NOT under the ComfyUI workspace: resolving that requires
    *running* comfy-cli, and this log's prime scenarios — a missing binary, a
    crash before any JSON, a timeout — are exactly when comfy-cli cannot answer.
    """
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    elif sys.platform == "win32":
        base = os.path.join(home, "AppData", "Local")
    else:
        base = os.path.join(home, ".config")
    return os.path.join(base, "comfy-local-mcp", "failures.jsonl")


def _resolve_failure_log_path(value: str | None) -> str | None:
    """The log path ``COMFY_LOCAL_MCP_DEBUG_LOG=<value>`` selects, else ``None``."""
    value = (value or "").strip()
    if not value or value == "0":
        return None
    if value == "1":
        return _default_failure_log_path()
    return value


# Single source of truth for "is the failure log on, and where" — ``None`` means
# disabled. One global rather than a separate enabled flag plus a path, so the
# two can never disagree (enabled-with-no-path would be a live tripwire).
_FAILURE_LOG_PATH = _resolve_failure_log_path(os.environ.get(_FAILURE_LOG_ENV))

# Optional: point the run/queue tools at a ComfyUI running ELSEWHERE (e.g. a GPU
# box reachable over a private network / tailnet) instead of the implicit local
# 127.0.0.1:8188. Configure with a single ``COMFYUI_URL`` (e.g.
# ``http://10.0.0.5:8188``) OR the ``COMFYUI_HOST`` (+ optional ``COMFYUI_PORT``)
# pair. When UNSET the tools behave byte-identically to the local-only default
# (no ``--host`` forwarded); when set, ``_with_target`` forwards ``--host`` /
# ``--port`` to the comfy-cli verbs that accept them (see ``_comfy_target`` /
# ``_with_target`` and ``_TARGET_AWARE_SUBCOMMANDS`` below).
DEFAULT_COMFYUI_PORT = 8188

# The comfy-cli verbs this server forwards ``--host`` / ``--port`` to: ``comfy
# run`` and every ``comfy jobs`` subcommand — the "run/queue" tools this ticket
# scopes, and the pair comfy-cli's ``comfy_cli/host_port.py`` contractually
# guarantees accept the options. Deliberately NOT forwarded (v1 scope):
#   * ``env`` / ``download`` / ``upload`` / ``templates`` / ``models`` /
#     ``generate`` / the lifecycle verbs take NO ``--host`` / ``--port`` at all,
#     so forwarding would error "No such option" — they stay local-only (a real
#     comfy-cli limitation; e.g. ``download`` can't fetch a remote job's files).
#   * ``nodes`` / ``validate`` DO accept ``--host`` / ``--port`` in current
#     comfy-cli, but remoting live discovery/validation is out of this pass's
#     "run/queue" scope; forwarding them is a clean follow-up.
# Forwarding is a no-op for the local default regardless, so unconfigured
# behavior is unchanged for every tool.
_TARGET_AWARE_SUBCOMMANDS = frozenset({"run", "jobs"})

# The envelope schema major version this server speaks. comfy-cli tags every
# result with a ``schema`` like ``envelope/1``; the whole contract (result
# shape, error codes) is versioned by that major. A mismatch means comfy-cli
# made a breaking change to the shape this wrapper parses, so we refuse it
# loudly (``_unwrap_envelope``) rather than silently misread its ``data``.
ENVELOPE_SCHEMA_MAJOR = 1

# Optional minimum comfy-cli version, opt-in via the ``COMFY_CLI_MIN_VERSION``
# env var (e.g. ``"1.5.0"``). Unset by default ON PURPOSE: the envelope-schema
# assertion above is the load-bearing compatibility gate, and the contract this
# server wraps is carried by a specific comfy-cli build reached through
# PATH/COMFY_BIN — not a PyPI release we could meaningfully hard-pin. Deployments
# that DO know their required floor enforce it by setting this; ``server_info``
# then rejects an older CLI (see ``_check_comfy_cli_version``).
MIN_COMFY_CLI_VERSION = os.environ.get("COMFY_CLI_MIN_VERSION") or None

# Hard ceiling for a single bounded watch so `float('inf')` / an absurd value
# can't hold a `comfy jobs watch` child open effectively forever (1 hour).
_MAX_WATCH_TIMEOUT = 3600.0


def _bounded_timeout(timeout_seconds: float, ceiling: float) -> float:
    """Bound a caller-supplied timeout to ``(0, ceiling]``, rejecting NaN.

    ``min(max(t, 0.0), ceiling)`` looks like it does this and does not: every
    NaN comparison is False, so ``max(nan, 0.0)`` returns ``nan``, ``min`` keeps
    it, and it reaches ``subprocess.run(timeout=nan)`` — where the selector
    raises a bare :class:`ValueError` that no caller catches (only
    ``TimeoutExpired`` is handled). The one value the ceiling exists to stop was
    the one that slipped through, and NaN is reachable because JSON and pydantic
    accept it. ``inf`` clamps down to ``ceiling`` as before.

    A non-positive timeout is rejected rather than floored to ``0.0``, which
    would fire an immediate, baffling "timed out after 0.0s" on a call that
    never really ran.
    """
    if math.isnan(timeout_seconds):
        raise ComfyCliError(
            "invalid timeout_seconds: NaN — expected a positive number of seconds."
        )
    if timeout_seconds <= 0:
        raise ComfyCliError(
            f"invalid timeout_seconds: {timeout_seconds!r} — expected a positive "
            "number of seconds."
        )
    return min(timeout_seconds, ceiling)


def _reject_nul(label: str, value: str) -> str:
    """Reject an embedded NUL, which ``subprocess`` cannot carry in argv.

    A NUL is a valid character in a JSON (and so MCP) string, but
    ``subprocess.run`` raises a bare ``ValueError: embedded null byte`` on one —
    uncaught here, so it would surface as an internal error rather than the
    :class:`ComfyCliError` every other bad input produces. Only NUL is refused:
    values are free-form model input (a prompt legitimately spans lines).
    """
    if "\0" in value:
        raise ComfyCliError(
            f"invalid {label}: embedded NUL character — a command argument "
            "cannot contain one."
        )
    return value


def _reject_option_like(label: str, value: str, expected: str = "") -> str:
    """Reject a leading-dash value that comfy-cli would parse as an option/flag
    instead of the intended positional or flag value (argument injection)."""
    if value.startswith("-"):
        hint = f" — expected {expected}" if expected else ""
        raise ComfyCliError(f"invalid {label}: {value!r} (leading '-'){hint}")
    return value


# Once the terminal envelope is read the authoritative result is in hand, but
# comfy-cli can outlive its own envelope under a pipe (observed with
# comfy-cli v1.12.0 `--json-stream`). Give such a child a short grace to exit on
# its own, then fall through to the `finally` that kills it — never block on a
# lingering child once the answer is already parsed.
_POST_ENVELOPE_REAP_GRACE = 5.0

# Dedicated, bounded thread pool for the blocking pipe reads / process waits in
# `_run_comfy_streaming` (`stdout.readline`, `stderr.read`, `proc.wait`).
#
# Cancelling an `asyncio.to_thread` NEVER interrupts the underlying OS thread —
# it stays parked on the pipe until the child is killed and its stdio closes.
# On the success-envelope path the stderr reader is never awaited, only
# cancelled, so it lingers until the `finally` kills the child (up to
# `_POST_ENVELOPE_REAP_GRACE`). Running these on asyncio's shared *default*
# executor means many concurrent `run_workflow(wait=True)` / `watch_job` calls
# could pile parked readers/waiters onto that pool and starve unrelated
# `to_thread` work for the grace window. Confining them to their own bounded
# pool caps the blast radius to this pool; the `finally` block additionally
# joins the stderr reader once the child is dead so it does not outlive the run.
_PIPE_POOL_MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
_PIPE_EXECUTOR = ThreadPoolExecutor(
    max_workers=_PIPE_POOL_MAX_WORKERS,
    thread_name_prefix="comfy-pipe",
)


def _in_pipe_pool(func, *args):
    """Off-load a blocking pipe read / process wait to the dedicated pool.

    Mirrors :func:`asyncio.to_thread` but targets :data:`_PIPE_EXECUTOR`
    instead of the loop's shared default executor, so subprocess pipe threads
    can never saturate the pool other `to_thread` callers rely on.
    """
    return asyncio.get_running_loop().run_in_executor(_PIPE_EXECUTOR, func, *args)


# Dedicated, bounded thread pool for `partner_generate`'s blocking `comfy
# generate` run.
#
# That run is the longest blocking call in this server — up to
# `_MAX_GENERATE_TIMEOUT` (an hour) parked in `subprocess.run`. Cancelling the
# awaiting coroutine (an MCP cancellation, a client disconnect) does NOT
# interrupt the OS thread, so on asyncio's shared *default* executor a handful
# of abandoned partner runs could occupy that pool for an hour and starve every
# other `to_thread` caller in the process. Confining them here caps the blast
# radius to partner generation itself, exactly as `_PIPE_EXECUTOR` does for the
# streaming pipe reads.
#
# Shared with `run_template`, which is the same class of call: a blocking
# comfy-cli run bounded only by the hour-long ceiling, on a tool that is async
# for its own spend-consent round-trip.
#
# Sized like the pipe pool. Saturating it queues further partner runs rather
# than growing threads without bound — deliberate backpressure on a paid,
# hour-long call, and far beyond any realistic concurrent use of one local
# server.
_GENERATE_POOL_MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
_GENERATE_EXECUTOR = ThreadPoolExecutor(
    max_workers=_GENERATE_POOL_MAX_WORKERS,
    thread_name_prefix="comfy-generate",
)


def _in_generate_pool(func, *args, **kwargs):
    """Off-load the blocking `comfy generate` run to the dedicated pool.

    Mirrors :func:`asyncio.to_thread` but targets :data:`_GENERATE_EXECUTOR`.
    ``run_in_executor`` takes no keyword arguments, so they are bound here.
    """
    call = functools.partial(func, *args, **kwargs)
    return asyncio.get_running_loop().run_in_executor(_GENERATE_EXECUTOR, call)


# Bound how long cleanup will block joining the parked stderr reader. `proc.kill`
# reaps only the DIRECT comfy-cli child; a descendant that inherited the stderr
# write fd keeps the pipe open, so the reader's `read()` never EOFs. Cap the join
# so cleanup can never hang the tool call, and detach the reader on timeout.
_STDERR_JOIN_GRACE = 5.0

# Retain at most this many trailing chars of a child's stderr. The reader must
# keep draining for the whole run (a chatty child would otherwise wedge on a full
# stderr pipe), but retaining every byte lets a misbehaving child drive unbounded
# allocation in this process — a memory-exhaustion DoS. Keep only the tail, where
# the actual error / traceback that `_unwrap_envelope` falls back to usually is.
_STDERR_MAX_CHARS = 64 * 1024
_STDERR_READ_CHUNK = 64 * 1024


def _drain_capped(stream: Any, limit: int) -> str:
    """Read ``stream`` to EOF but keep only the trailing ``limit`` chars.

    Draining to EOF keeps the child from blocking on a full stderr pipe; slicing
    to the tail on every chunk bounds memory to ``limit`` + one chunk regardless
    of how much the child spams.
    """
    tail = ""
    while True:
        chunk = stream.read(_STDERR_READ_CHUNK)
        if not chunk:
            break
        tail = (tail + chunk)[-limit:]
    return tail


# comfy-cli floor. `comfy logs` (get_logs) and the structured `envelope/1`
# contract this server relies on require comfy-cli >= 1.12.0; against an older
# install `comfy logs` doesn't exist and would surface as a cryptic "No such
# command", so `_run_comfy` guards this once, up front, with an upgrade message.
_MIN_COMFY_CLI = (1, 12, 0)
_MIN_COMFY_CLI_STR = "1.12.0"

# The version guard shells out to `comfy --version`; memoize so it runs at most
# once per process (it sits on the hot path of every _run_comfy call).
_version_checked = False


def _comfy_env() -> dict[str, str]:
    """Child-process environment for every comfy-cli spawn.

    Single source of truth so the two spawn sites (``_run_comfy`` /
    ``_run_comfy_streaming``) cannot drift. The inherited ``os.environ`` is
    forwarded WHOLESALE on purpose — that passthrough is what lets a variable
    set in the MCP client's ``env`` block configure comfy-cli without any code
    here, e.g. ``COMFY_LOCAL_URL`` to target a local ComfyUI on a non-default
    address (``:8189``) or ``COMFY_API_KEY`` for partner-API nodes. Injected
    keys are placed AFTER ``os.environ`` so they win over any inherited values:

    - ``COMFY_WHERE=local`` — belt-and-suspenders pin so we never touch cloud.
    - ``COMFY_NO_WATCH=1`` — suppress comfy-cli's file watcher for agentic
      callers like this MCP; a harmless no-op on versions that lack the flag.
    - ``PYTHONUTF8=1`` / ``PYTHONIOENCODING=utf-8`` — force UTF-8 on the child's
      console. Without them a default Windows (cp1252) console raises
      ``UnicodeEncodeError`` printing the UTF-8 catalog output and wedges, so the
      discovery tools present as a 60s timeout. UTF-8 is already the practical
      default on macOS/Linux, so this is a no-op there.
    """
    return {
        **os.environ,
        "COMFY_WHERE": "local",
        "COMFY_NO_WATCH": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }


# --- macOS protected-folder (TCC) diagnostics --------------------------------
#
# macOS gates ~/Documents, ~/Desktop and ~/Downloads behind TCC (Transparency,
# Consent & Control). An app without Full Disk Access cannot read them, and
# neither can the processes it spawns — so when a ComfyUI install (and its
# venv) lives under one of those folders, the `comfy` binary an MCP client
# spawns dies during interpreter startup:
#
#     Fatal Python error: init_import_site: Failed to import the site module
#     PermissionError: [Errno 1] Operation not permitted: '.../venv/pyvenv.cfg'
#
# That is a macOS privacy setting, not a ComfyUI/comfy-cli fault, so the helpers
# below detect the signature and answer with the fix instead of relaying a raw
# Python traceback the user can do nothing with.
_MACOS_PROTECTED_DIRS = ("Documents", "Desktop", "Downloads")

# The denied path as CPython prints it in the OSError above. CPython's format is
# ``[Errno 1] <strerror>: <repr(path)>`` and macOS can localize ``<strerror>``,
# so match on the errno marker first and fall back to the English phrase for a
# denial reported without one. `repr` quotes with `'` unless the path itself
# contains one (then `"`), so accept either; the capture stops at a newline and
# is bounded well past macOS's PATH_MAX so a garbled stderr line cannot drag an
# unbounded blob into the message.
_TCC_PATH_RE = re.compile(
    r"(?:\[Errno 1\][^:\n]*|Operation not permitted):[ \t]*"
    r"b?(?:'([^'\n]{0,1024})'|\"([^\"\n]{0,1024})\")"
)

# EPERM as it reaches us in text. `[errno 1]` is here because the strerror text
# next to it comes from libc and macOS translates it under a non-English
# `LC_MESSAGES` — the bracketed errno is what stays constant. It cannot collide
# with another errno: the closing bracket rules out `[Errno 13]` and friends.
_EPERM_MARKERS = ("operation not permitted", "[errno 1]")

# CPython's own marker for "the interpreter died before `site` was imported",
# which is the shape a venv under a protected folder takes.
_STARTUP_CRASH_MARKER = "init_import_site"


def _is_macos() -> bool:
    """True on macOS. Read at call time so tests can patch ``sys.platform``."""
    return sys.platform == "darwin"


def _macos_protected_dir(path: str | bytes | None) -> str | None:
    """Name of the protected home folder ``path`` sits under, else ``None``.

    Compared case-insensitively: macOS volumes are case-insensitive by default,
    so ``~/downloads/ComfyUI`` is the very same TCC-protected folder as
    ``~/Downloads/ComfyUI`` and must be named as such rather than silently
    falling through to the generic wording.
    """
    if not path:
        return None
    # An OSError raised on a bytes path carries a bytes `filename`; decode it
    # rather than let a str/bytes comparison raise TypeError below.
    resolved = os.path.abspath(os.path.expanduser(os.fsdecode(path))).lower()
    home = os.path.expanduser("~")
    for name in _MACOS_PROTECTED_DIRS:
        root = os.path.join(home, name).lower()
        if resolved == root or resolved.startswith(root + os.sep):
            return name
    return None


def _looks_like_tcc_denial(text: str | None) -> bool:
    """True if ``text`` carries the macOS protected-folder denial signature.

    macOS-only by design: EPERM ("operation not permitted") means something
    else entirely on Linux, and the guidance below is System-Settings-specific,
    so a non-macOS failure must keep its original message.

    EPERM alone is NOT enough even on macOS — SIP, the app sandbox and signalling
    a protected process all raise it, and rewriting one of those with Full Disk
    Access guidance would send the user off fixing the wrong thing. Require
    corroboration: either CPython's startup-crash marker (the venv-under-a-
    protected-folder shape this exists for) or a denied path that really does
    resolve under one of the three folders. Anything else keeps its own message.
    """
    if not text or not _is_macos():
        return False
    lowered = text.lower()
    if not any(marker in lowered for marker in _EPERM_MARKERS):
        return False
    return (
        _STARTUP_CRASH_MARKER in lowered
        or _macos_protected_dir(_tcc_path_from(text)) is not None
    )


def _tcc_path_from(text: str | None) -> str | None:
    """The denied path CPython named in ``text``, when it named one."""
    match = _TCC_PATH_RE.search(text or "")
    if match is None:
        return None
    # Exactly one of the two quote-style alternatives captured.
    return match.group(1) if match.group(1) is not None else match.group(2)


def _tcc_guidance(path: str | bytes | None = None) -> str:
    """Actionable fix for a macOS protected-folder denial, as one message.

    ``path`` is the denied file when we know it (parsed out of the child's
    stderr, or an unreadable ``COMFY_BIN``); naming its protected folder makes
    the message concrete. Without one — or with one outside the protected set —
    the wording stays general rather than asserting a location we haven't
    verified. A ``bytes`` path (what an ``OSError`` from a bytes-path syscall
    carries) is decoded, so it reads as a path rather than as ``b'...'``.
    """
    if path is not None:
        path = os.fsdecode(path)
    folder = _macos_protected_dir(path)
    if folder:
        where = (
            f"{path} is under ~/{folder}, which macOS protects (TCC): an app — "
            "and every process it spawns — cannot read it unless that app has "
            "Full Disk Access."
        )
    else:
        where = (
            "macOS protects ~/Documents, ~/Desktop and ~/Downloads (TCC): an "
            "app — and every process it spawns — cannot read them unless that "
            "app has Full Disk Access, so a ComfyUI install or venv under one "
            "of them is unreadable from here."
        )
    return (
        f"macOS denied access to a protected folder. {where}\n"
        "Fix it either way:\n"
        "  1. Grant your MCP client Full Disk Access — System Settings > "
        "Privacy & Security > Full Disk Access > add the app (Claude Desktop, "
        "Cursor, or the terminal you launch the client from) — then quit and "
        "reopen it.\n"
        "  2. Or move the ComfyUI folder somewhere unprotected (e.g. ~/ComfyUI) "
        "and re-point comfy-cli at it with `comfy set-default <path>`.\n"
        "This is a macOS privacy setting, not a ComfyUI or comfy-cli fault."
    )


class ComfyCliError(RuntimeError):
    """comfy-cli was missing, timed out, or returned an error envelope.

    ``code`` carries the envelope's structured ``error.code`` when the failure
    came from an error envelope (used to drive the bounded credential retry in
    ``run_workflow``); it is ``None`` for local failures the wrapper raises
    itself (missing binary, timeout, no-JSON output), so callers can branch on a
    specific code without string-matching the message — e.g. ``get_logs``
    swallows ``no_log_file`` but re-raises the rest.

    ``no_envelope`` is the stronger, unambiguous provenance signal: ``True`` only
    when comfy-cli ran to completion and emitted NO envelope at all. A null
    ``code`` does NOT imply that — a well-formed error envelope may simply omit
    ``error.code`` — so a caller asking "did comfy-cli fail *before* it could
    report structurally?" must check this flag rather than ``code is None``.
    :func:`_is_missing_verb_error` is exactly that caller.

    ``returncode`` is the child's exit status wherever :func:`_unwrap_envelope`
    knows it — on the no-envelope path AND on an error envelope — so it is
    genuinely independent of ``no_envelope`` rather than a proxy for it. It
    distinguishes *how* comfy-cli failed: a usage error the argument parser
    rejected before dispatch versus a failure partway through a command it did
    accept, which the message text alone cannot tell you. It stays ``None`` for
    the failures raised without ever reading a child's status (missing binary,
    timeout).
    """

    def __init__(
        self,
        *args: object,
        code: str | None = None,
        no_envelope: bool = False,
        returncode: int | None = None,
    ) -> None:
        super().__init__(*args)
        self.code = code
        self.no_envelope = no_envelope
        self.returncode = returncode


def _comfy_bin_candidates() -> list[str]:
    """Every filesystem location ``COMFY_BIN`` could name, as ``which`` would look.

    A ``COMFY_BIN`` carrying a directory separator names exactly one file. A bare
    name (the default, ``comfy``) is resolved against ``PATH`` — NOT against the
    current working directory, which is what a plain ``os.stat(COMFY_BIN)`` would
    do and which would miss a `comfy` that lives on ``PATH`` inside a protected
    folder: precisely the install this diagnostic exists for.
    """
    if os.path.dirname(COMFY_BIN):
        return [COMFY_BIN]
    return [os.path.join(entry, COMFY_BIN) for entry in os.get_exec_path() if entry]


def _tcc_blocked_comfy_bin() -> str | None:
    """The candidate ``comfy`` path macOS is denying us, if that's what's wrong.

    Only a candidate that BOTH sits under a protected folder and refuses to be
    stat'ed counts. The protected-folder test is what keeps an ordinary EACCES —
    a restrictive mode or ACL on a ``COMFY_BIN`` somewhere else entirely — from
    being mislabelled as a Full Disk Access problem it isn't.
    """
    for candidate in _comfy_bin_candidates():
        if _macos_protected_dir(candidate) is None:
            continue
        try:
            os.stat(candidate)
        except PermissionError:
            return candidate
        except OSError:
            continue  # absent, a broken link, an unresolvable name — not ours
    return None


def _require_comfy_bin() -> None:
    """Resolve the ``comfy`` binary, raising an actionable error when it can't be.

    Shared by both spawn sites (:func:`_run_comfy_raw` / :func:`_run_comfy_streaming`)
    so their missing-binary behavior cannot drift. On macOS a ``comfy`` that exists
    but cannot even be stat'ed is a protected-folder denial, not a missing install
    — ``shutil.which`` reports both as ``None``, so say which one it is instead of
    the misleading "not found on PATH".
    """
    if shutil.which(COMFY_BIN) is not None:
        return
    if _is_macos():
        blocked = _tcc_blocked_comfy_bin()
        if blocked is not None:
            message = f"`{COMFY_BIN}` could not be read.\n\n{_tcc_guidance(blocked)}"
            # `args=()` on both raises: no comfy-cli invocation ever happened, so
            # there is no argv to record — the failure IS that there is no binary.
            _log_failure("binary_missing", (), message=message)
            raise ComfyCliError(message)
    message = (
        f"`{COMFY_BIN}` not found on PATH. Install comfy-cli "
        "(`pip install comfy-cli`) or set the COMFY_BIN env var."
    )
    _log_failure("binary_missing", (), message=message)
    raise ComfyCliError(message)


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """Extract a dotted numeric version (e.g. ``1.12.0``) from ``text``.

    Prefers a token that follows the word "version" (see below), else the first
    dotted-numeric token. Returns a ``(major, minor, patch)`` tuple (a missing
    patch defaults to 0), or ``None`` when no version-looking token is present.
    """
    # Prefer a version token that follows the word "version" (comfy-cli prints
    # "comfy-cli, version X.Y.Z"), so we don't latch onto an earlier dotted
    # token — a Python version, a path segment like ``.../3.10/...`` — and end
    # up comparing the wrong value. Fall back to the first dotted-numeric token
    # anywhere in the text (still fails OPEN on no match, per the guard's docs).
    match = re.search(r"version[^\d]*(\d+)\.(\d+)(?:\.(\d+))?", text, re.IGNORECASE)
    if match is None:
        match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    major, minor, patch = match.groups(default="0")
    return int(major), int(minor), int(patch)


def _check_comfy_version() -> None:
    """Guard: refuse to run against a comfy-cli older than :data:`_MIN_COMFY_CLI`.

    Runs ``comfy --version`` once per process (memoized via ``_version_checked``).
    If the reported version is below the floor, raises a clear, actionable
    :class:`ComfyCliError` telling the user to upgrade — so a stale install fails
    with "upgrade comfy-cli to >= 1.12.0" instead of a cryptic "No such command:
    logs" deep inside a tool call. Fails OPEN on anything it can't positively read
    as too-old (an unparseable ``--version``, a ``--version`` that errors) so a
    future comfy-cli output-format change can never wedge a working install.
    """
    global _version_checked
    if _version_checked:
        return
    try:
        proc = subprocess.run(
            [COMFY_BIN, "--version"],
            capture_output=True,
            text=True,
            errors="replace",  # never crash on undecodable `--version` bytes
            timeout=30.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # A hung `--version` is latched so we don't re-block every later call on
        # the same 30s wait; fail OPEN for the rest of the process.
        _version_checked = True
        return
    except PermissionError as exc:
        # The spawn ITSELF was denied (not the child exiting non-zero) — e.g. a
        # `comfy` launcher whose interpreter sits in a protected folder. Without
        # this branch the generic handler below fails open and the raw EPERM
        # escapes from the real spawn a moment later, unexplained. Must precede
        # the OSError handler: PermissionError is a subclass of it.
        denied = getattr(exc, "filename", None) or _tcc_path_from(str(exc))
        if _is_macos() and (
            _looks_like_tcc_denial(str(exc)) or _macos_protected_dir(denied) is not None
        ):
            raise ComfyCliError(
                f"`{COMFY_BIN}` could not be started.\n\n{_tcc_guidance(denied)}\n\n"
                f"Original error: {exc}"
            ) from exc
        return  # any other permission problem: fail OPEN, exactly as before
    except (OSError, subprocess.SubprocessError):
        # A transient spawn failure fails OPEN for THIS call but is NOT latched —
        # a later call re-checks rather than permanently disabling the guard.
        return
    if proc.returncode != 0 and _looks_like_tcc_denial(proc.stderr):
        # comfy-cli's own interpreter could not start because macOS denied it
        # its venv — the reported failure for a ComfyUI install under
        # ~/Documents. This guard runs before the first tool call of the
        # process, so catching it here is what turns the raw `Fatal Python
        # error` traceback into the fix. Deliberately NOT memoized: granting
        # Full Disk Access and retrying in the same process must re-check.
        raise ComfyCliError(
            f"`{COMFY_BIN}` could not start.\n\n"
            f"{_tcc_guidance(_tcc_path_from(proc.stderr))}\n\n"
            f"Original error: {_tail(proc.stderr)}"
        )
    version = _parse_version(f"{proc.stdout}\n{proc.stderr}")
    if version is not None and version < _MIN_COMFY_CLI:
        # Deliberately do NOT memoize a too-old verdict: if the user upgrades and
        # retries within the same process, re-check rather than latch the failure.
        raise ComfyCliError(
            f"comfy-cli {'.'.join(map(str, version))} is too old — this server "
            f"requires comfy-cli >= {_MIN_COMFY_CLI_STR}. Upgrade it with "
            f"`pip install --upgrade 'comfy-cli>={_MIN_COMFY_CLI_STR}'`."
        )
    _version_checked = True


# comfy-cli's error code for "I have no server pid recorded to stop" — the one
# stop failure ``restart_comfyui`` treats as benign (see its docstring).
_NO_RECORDED_SERVER_CODE = "no_recorded_server"


def _is_no_recorded_server(exc: ComfyCliError) -> bool:
    """True when ``exc`` is comfy-cli's benign 'nothing recorded to stop' error.

    Prefers the structured ``code`` and falls back to the message so it also
    recognizes the error when only the human-readable string carries the marker.
    """
    return exc.code == _NO_RECORDED_SERVER_CODE or _NO_RECORDED_SERVER_CODE in str(exc)


def _redact_url(url: str) -> str:
    """Return ``url`` with any ``user:pass@`` userinfo masked to ``***@``.

    :class:`ComfyCliError` messages echo the offending config value and may reach
    the MCP client or logs, so a credential embedded in ``COMFYUI_URL`` (e.g.
    ``http://user:token@host``) must not be surfaced raw. Only the netloc
    (scheme-separator up to the first path/query/fragment delimiter) is inspected
    so a stray ``@`` in a path can't confuse the masking.
    """
    scheme_sep = url.find("://")
    start = scheme_sep + 3 if scheme_sep != -1 else 0
    end = len(url)
    for delim in ("/", "?", "#"):
        i = url.find(delim, start)
        if i != -1:
            end = min(end, i)
    netloc = url[start:end]
    if "@" in netloc:
        return url[:start] + "***@" + netloc.rsplit("@", 1)[1] + url[end:]
    return url


def _strip_brackets(host: str) -> str:
    """Strip surrounding ``[...]`` from a bracketed IPv6 host for consistency.

    ``urlparse`` already returns an IPv6 ``.hostname`` bracket-free, so normalize
    a bracketed ``COMFYUI_HOST`` (``[::1]``) the same way — both config paths then
    forward a bare host to comfy-cli, which re-brackets it when building its URL.
    """
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _comfy_target() -> tuple[str, int, str] | None:
    """Resolve the configured ComfyUI ``(host, port, source)``, or None for local.

    Precedence: ``COMFYUI_URL`` (a full URL, parsed into host + port) wins;
    otherwise ``COMFYUI_HOST`` (+ optional ``COMFYUI_PORT``, default
    :data:`DEFAULT_COMFYUI_PORT`). Returns ``None`` when nothing is set, so the
    tools stay byte-identical to the local-only default (no ``--host`` forwarded,
    comfy-cli's own 127.0.0.1:8188). Raises :class:`ComfyCliError` on a set but
    malformed value rather than silently retargeting to the wrong place.

    comfy-cli's ``--host`` / ``--port`` carry only a host and port, so a
    ``COMFYUI_URL`` that also names a non-``http`` scheme (``https://``) or a base
    path (``/comfyui``) is REJECTED rather than silently dropped — otherwise a
    user asking for TLS or a reverse-proxy path would be quietly downgraded.
    """
    url = os.environ.get("COMFYUI_URL", "").strip()
    if url:
        # urlparse needs a scheme to populate .hostname/.port; a bare
        # "host:port" is otherwise read as scheme:path. Prefix "//" so a
        # scheme-less value parses as a netloc. urlparse itself raises
        # ValueError on a malformed value (e.g. an unbalanced IPv6 bracket
        # "http://[::1"), so it lives inside the try alongside the .port access.
        try:
            parsed = urlparse(url if "://" in url else f"//{url}")
            host, port = parsed.hostname, parsed.port
        except ValueError as exc:  # bad port, or malformed URL (IPv6 brackets)
            raise ComfyCliError(
                f"COMFYUI_URL is malformed: {_redact_url(url)!r} ({exc})."
            ) from exc
        if parsed.scheme and parsed.scheme != "http":
            raise ComfyCliError(
                f"COMFYUI_URL scheme {parsed.scheme!r} is not supported "
                f"({_redact_url(url)!r}): comfy-cli's --host/--port speak plain "
                "http only, so an https:// target would be silently downgraded. "
                "Use http://<host>:<port>."
            )
        if parsed.path not in ("", "/"):
            raise ComfyCliError(
                f"COMFYUI_URL must not include a path ({_redact_url(url)!r}): "
                "comfy-cli forwards only host/port, so a reverse-proxy base path "
                "would be dropped. Point COMFYUI_URL at the bare host:port."
            )
        if not host:
            raise ComfyCliError(
                f"COMFYUI_URL is set but names no host: {_redact_url(url)!r}. "
                "Use e.g. http://<host>:8188 (or set COMFYUI_HOST/COMFYUI_PORT)."
            )
        # `port or DEFAULT` alone would treat an explicit :0 as absent and
        # silently target 8188; reject it to match the COMFYUI_PORT path.
        if port == 0:
            raise ComfyCliError(
                f"COMFYUI_URL port is out of range (1-65535): {_redact_url(url)!r}."
            )
        return _strip_brackets(host), port or DEFAULT_COMFYUI_PORT, "COMFYUI_URL"

    host = os.environ.get("COMFYUI_HOST", "").strip()
    raw_port = os.environ.get("COMFYUI_PORT", "").strip()
    if not host:
        # A port alone does not select a remote; raise rather than silently
        # ignoring it and defaulting back to the local 127.0.0.1:8188.
        if raw_port:
            raise ComfyCliError(
                "COMFYUI_PORT is set but COMFYUI_HOST is not; a port alone does "
                "not select a remote. Set COMFYUI_HOST (or COMFYUI_URL) to "
                "target a remote ComfyUI."
            )
        return None
    if not raw_port:
        return _strip_brackets(host), DEFAULT_COMFYUI_PORT, "COMFYUI_HOST"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ComfyCliError(
            f"COMFYUI_PORT must be an integer, got {raw_port!r}."
        ) from exc
    if not (1 <= port <= 65535):
        raise ComfyCliError(f"COMFYUI_PORT is out of range (1-65535): {port}.")
    return _strip_brackets(host), port, "COMFYUI_HOST"


def _with_target(args: tuple[str, ...]) -> tuple[str, ...]:
    """Append ``--host`` / ``--port`` to a target-aware subcommand, if configured.

    The flags are injected into the SUBCOMMAND args (after the ``run`` / ``jobs``
    verb), never into the global ``--json`` / ``--where`` prefix, since
    ``--host`` / ``--port`` are ``comfy run`` / ``comfy jobs`` subcommand options.
    A no-op for the local default (``_comfy_target`` is None) and for any
    subcommand that doesn't accept the flags (see :data:`_TARGET_AWARE_SUBCOMMANDS`),
    so unconfigured behavior is byte-identical to today.
    """
    # Check the verb FIRST, then resolve the target. A malformed
    # COMFYUI_URL/PORT must not brick local-only verbs (server_info's `env`,
    # download, the stop/logs lifecycle) that never touch the remote — they'd
    # otherwise raise ComfyCliError here despite ignoring the target entirely,
    # breaking the "local behavior unchanged" contract (BE-3869 review).
    if not args or args[0] not in _TARGET_AWARE_SUBCOMMANDS:
        return args
    target = _comfy_target()
    if target is None:
        return args
    host, port, _source = target
    return (*args, "--host", host, "--port", str(port))


def _run_comfy_raw(
    *args: str, timeout: float | None = None
) -> tuple[dict | None, str, tuple[str, ...], int, str]:
    """Run ``comfy --json --where local <args>`` and return the RAW envelope + context.

    The shared subprocess half of :func:`_run_comfy`: it resolves the binary,
    runs comfy-cli, and returns ``(envelope, stdout, args, returncode, stderr)``
    WITHOUT unwrapping — so a caller that needs the envelope itself (e.g.
    ``server_info`` reading the versioned ``schema``) can inspect it, or the raw
    ``stdout`` (e.g. the lifecycle-success synthesizer), while :func:`_run_comfy`
    just unwraps it down to ``data``.
    """
    _require_comfy_bin()
    _check_comfy_version()
    # Forward --host/--port into the subcommand when a remote ComfyUI is
    # configured (no-op for the local default; see _with_target). Reassigning
    # args here means the forwarded flags also appear in the error/timeout
    # context returned below, so a remote failure reports the real invocation.
    args = _with_target(args)
    # Global flags (--json, --where) MUST precede the subcommand in comfy-cli;
    # a trailing --json errors with "No such option". (Verified against comfy-cli.)
    cmd = [COMFY_BIN, "--json", "--where", "local", *args]
    env = _comfy_env()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # Pin the parent-side decode to UTF-8 so it matches what the child
            # is forced to emit (_comfy_env). Without this, text=True decodes
            # the pipe with the system locale (cp1252 on a default Windows
            # console) and the non-ASCII catalog output raises UnicodeDecodeError
            # or yields mojibake before _unwrap_envelope — the exact crash this
            # fix targets, just moved to the reader.
            encoding="utf-8",
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run attaches whatever the child wrote before being killed
        # (capture_output=True) to the exception — surface it so a crashed,
        # wedged comfy-cli (e.g. a traceback on stderr) is not indistinguishable
        # from a genuinely slow one. See BE-3343.
        message = (
            f"comfy-cli timed out after {timeout}s: {' '.join(cmd)}. "
            f"stderr tail: {_tail(exc.stderr) or '<empty>'}; "
            f"stdout tail: {_tail(exc.stdout) or '<empty>'}"
        )
        # `exit_code=None`: the child was killed at the deadline, so it never
        # reported one. The log keeps a longer slice of both streams than the
        # message above does — see `_FAILURE_LOG_TAIL_CHARS`.
        _log_failure(
            "timeout",
            args,
            message=message,
            stdout=exc.stdout,
            stderr=exc.stderr,
        )
        raise ComfyCliError(message) from exc

    return (
        _last_json_object(proc.stdout),
        proc.stdout,
        args,
        proc.returncode,
        proc.stderr,
    )


def _run_comfy(*args: str, timeout: float | None = None, plain_ok: bool = False) -> Any:
    """Run ``comfy <args> --where local --json`` and return the envelope's ``data``.

    comfy-cli emits a versioned ``envelope/1`` object on stdout (a single line
    for ``--json``, or an NDJSON stream whose final line is the envelope). We
    keep the last JSON object and unwrap ``ok`` / ``data`` / ``error``.

    ``plain_ok`` relaxes the envelope requirement for the commands that print
    human text and exit 0 WITHOUT emitting an envelope — the lifecycle verbs
    ``launch`` / ``stop`` (BE-2953) and ``model download`` (BE-3345): a clean
    exit with no JSON is treated as success and a result dict is synthesized
    from the printed text, rather than raising the "returned no JSON" error on
    an action that actually succeeded. A non-zero exit, or a real error
    envelope, still raises as usual.
    """
    envelope, stdout, args, returncode, stderr = _run_comfy_raw(*args, timeout=timeout)
    # A plain_ok command that exits 0 without a *real* envelope is a success
    # (BE-2953 launch/stop, BE-3345 model download). `_last_json_object` may
    # return a stray non-envelope JSON line (e.g. a diagnostic log that happens
    # to parse), so key the fast-path off the absence of a `type==envelope`
    # object rather than the absence of any JSON — otherwise one incidental JSON
    # line on a successful run would be mis-unwrapped into a spurious "failed"
    # raise. A real error envelope still has `type==envelope`, so it flows to
    # `_unwrap_envelope` and raises as usual.
    real_envelope = _real_envelope(envelope)
    if plain_ok and real_envelope is None and returncode == 0:
        return _synthesize_plain_result(args, stdout, stderr)
    # Enforce the envelope contract on the normal path too: pass `real_envelope`
    # (not `envelope`) so a stray non-envelope JSON line — e.g. an incidental
    # `{"ok": true, "data": ...}` diagnostic — can't be mis-unwrapped as a valid
    # response for a non-`plain_ok` tool; it raises the "returned no JSON" error
    # like any other missing envelope. A real error envelope still has
    # `type==envelope`, so it flows through and raises with its code as usual.
    return _unwrap_envelope(real_envelope, args, returncode, stderr, stdout=stdout)


def _envelope_schema(envelope: dict) -> str | None:
    """The envelope's declared ``schema`` string (e.g. ``"envelope/1"``), or None."""
    value = envelope.get("schema")
    return value if isinstance(value, str) else None


def _envelope_major(envelope: dict) -> int | None:
    """Major version an envelope declares via ``schema`` (``envelope/<N>``), or None.

    ``None`` means the ``schema`` string is absent OR present-but-unparseable.
    :func:`_unwrap_envelope` disambiguates the two: it only calls this once it
    knows a ``schema`` was declared, so a ``None`` here then means "declared but
    not ``envelope/<N>``" and is refused. The pattern is fully anchored, so a
    decorated schema like ``envelope/1-foo`` or a future ``envelope-v2`` does
    NOT masquerade as a bare major.
    """
    schema = _envelope_schema(envelope)
    if not schema:
        return None
    match = re.fullmatch(r"envelope/(\d+)", schema.strip())
    return int(match.group(1)) if match else None


def _tail(text: str | bytes | None, limit: int = 500) -> str:
    """Bounded tail of captured output; decodes bytes defensively.

    On POSIX, ``TimeoutExpired.stdout``/``.stderr`` arrive as *bytes* even under
    ``text=True`` (CPython quirk: ``communicate()`` raises with raw bytes; on
    Windows ``run()`` re-communicates after the kill and returns ``str``), so
    callers can hand us either. The ``limit`` hard-bounds the result so a chatty
    child cannot inflate an error payload.
    """
    if not text or limit <= 0:
        # ``[-0:]`` is ``[0:]`` (the whole string), so a non-positive limit would
        # silently defeat the hard-bound; treat it as "no tail".
        return ""
    if isinstance(text, bytes):
        # Slice the raw bytes before decoding so a huge capture doesn't incur a
        # full decode+copy just to keep the last ``limit`` chars. UTF-8 is at
        # most 4 bytes/char, so the last ``4 * limit`` bytes always contain
        # enough to yield ``limit`` decoded chars (a leading byte may be dropped,
        # which ``errors="replace"`` handles cleanly).
        text = text[-4 * limit :].decode("utf-8", errors="replace")
    return text.strip()[-limit:]


def _stream_tail(text: str | bytes | None, limit: int = 500) -> str:
    """Bounded tail of a captured stream, with explicit empty/truncation markers.

    The error-message dressing over :func:`_tail`, for the messages a user
    actually reads when comfy-cli fails:

    - a blank/absent capture renders as ``<empty>`` rather than nothing, so a
      message can never end in a dangling ``stderr:`` that is indistinguishable
      from a capture we truncated away;
    - a capture that WAS clipped is prefixed with ``...`` so the truncation is
      visible instead of looking like the whole stream.

    The *tail* is what's kept (not the head): a CLI traceback puts the
    exception — the part that says what actually went wrong — last.
    """
    if limit <= 0:
        # Mirror `_tail`'s own non-positive guard. It matters more here: we ask
        # `_tail` for `limit + 1`, so a `limit` of 0 would slip past its check
        # and the `[-limit:]` below would then be `[0:]` — the WHOLE capture,
        # exactly the unbounded payload the limit exists to prevent.
        return "<empty>"
    # `_tail` clips silently, so ask it ONCE for one char more than the bound:
    # an over-long answer is itself the evidence that something was dropped, and
    # re-slicing it locally costs nothing (asking `_tail` twice would re-run the
    # strip and any bytes-decode over the whole capture just to learn that).
    tail = _tail(text, limit=limit + 1)
    if not tail:
        return "<empty>"
    return "..." + tail[-limit:] if len(tail) > limit else tail


def _scrub_arg(arg: str) -> str:
    """A comfy-cli argv token, safe to record in the failure log.

    A URL argument gets two treatments and everything else passes through
    verbatim (an arg is usually a path or a subcommand, and mangling those would
    defeat the log):

    - :func:`_redact_url` masks ``user:pass@`` userinfo, exactly as it does for
      the config values the error messages echo;
    - the query string and fragment are dropped entirely. That mirrors
      comfy-cli's own ``tracking.py`` scrubber, which exists because a CivitAI
      model URL carries its credential as ``?token=…`` — a secret no amount of
      userinfo masking would catch.
    """
    if not arg.lower().startswith(("http://", "https://")):
        return arg
    scrubbed = _redact_url(arg)
    # Cut at whichever of `?` / `#` comes first: a fragment can precede a
    # (meaningless, but present) query, and slicing on each in turn would leave
    # the earlier delimiter's contents behind.
    cut = min(
        (i for i in (scrubbed.find("?"), scrubbed.find("#")) if i != -1),
        default=-1,
    )
    return scrubbed if cut == -1 else scrubbed[:cut]


# A URL anywhere in a recorded `message`. Anchored on the literal scheme so the
# engine can skip ahead to a candidate rather than re-scan from every offset,
# and `\S+` is a possessive-free single-pass match — no backtracking risk on a
# multi-KB message.
_MESSAGE_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _scrub_message(message: str) -> str:
    """Apply :func:`_scrub_arg`'s URL scrubbing to every URL inside ``message``.

    Scrubbing ``args`` alone would be theatre: the error-envelope message this log
    records is built by :func:`_unwrap_envelope` as ``comfy <args> failed …``, so
    the RAW argv — including a signed ``?token=…`` model URL — is echoed right
    back into it. That string is already what the MCP client sees (unchanged
    here), but this log PERSISTS it to disk for a tester to zip up and share, so
    the same masking has to reach it.

    Only URL-shaped substrings are touched; the surrounding prose (the point of
    keeping ``message`` at all) is preserved byte-for-byte.
    """
    return _MESSAGE_URL_RE.sub(lambda match: _scrub_arg(match.group(0)), message)


# The path the currently-open rotating handler writes to; ``None`` until the
# first failure is logged. Keyed on the path rather than a bare "set up yet?"
# flag so the handler is rebuilt if `_FAILURE_LOG_PATH` is ever repointed, and
# so nothing here touches the filesystem while the log is disabled.
_failure_handler_path: str | None = None

# Serializes the setup below. Tool calls reach `_run_comfy` from several worker
# pools (`_in_pipe_pool` / `_in_generate_pool`), so two threads can fail at once
# and race here. `logging` makes *emitting* thread-safe but not this
# check-then-swap: interleaved remove-all/add passes can leave the logger holding
# BOTH handlers, which would duplicate every subsequent line for the life of the
# process. Only the (once-per-path) setup takes the lock; the common case is the
# unguarded `!=` comparison plus a normal `.info()`.
_failure_handler_lock = threading.Lock()


def _failure_logger(path: str) -> logging.Logger:
    """The dedicated failure logger, opening its rotating handler on first use.

    Created lazily — the handler opens the file, so building it eagerly at import
    would give a disabled log a filesystem footprint. ``path`` is passed in
    rather than read from the global so this can never be reached without one.
    """
    global _failure_handler_path
    logger = logging.getLogger(_FAILURE_LOGGER_NAME)
    if _failure_handler_path != path:
        with _failure_handler_lock:
            # Re-check under the lock: the thread that lost the race must not
            # tear down and rebuild the handler the winner just installed.
            if _failure_handler_path != path:
                for existing in list(logger.handlers):
                    logger.removeHandler(existing)
                    existing.close()
                # Cleared BEFORE the (fallible) setup below, so a handler that
                # fails to open cannot leave the global claiming a path nothing
                # is actually writing to.
                _failure_handler_path = None
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                handler = RotatingFileHandler(
                    path,
                    maxBytes=_FAILURE_LOG_MAX_BYTES,
                    backupCount=_FAILURE_LOG_BACKUPS,
                    encoding="utf-8",
                )
                # Bare `%(message)s`: every line must be pure JSON, so `jq` can
                # read the file directly with no level/timestamp prefix to strip
                # first (the record carries its own `ts`).
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                logger.propagate = False
                _failure_handler_path = path
    return logger


def _log_failure(
    kind: str,
    args: tuple[str, ...] | list[str],
    exit_code: int | None = None,
    error_code: str | None = None,
    message: str = "",
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    streaming: bool = False,
) -> None:
    """Append one JSONL record for a comfy-cli failure, if the log is enabled.

    Called immediately before each raise, so every line recorded corresponds to a
    failure a caller actually saw. ``kind`` is one of ``error_envelope`` /
    ``no_json`` / ``timeout`` / ``binary_missing`` / ``schema_mismatch``.

    The record is STRUCTURED, not just the formatted sentence, so the log is
    ``jq``/grep-able without parsing prose — ``message`` is kept alongside it so
    QA can correlate a line with what the MCP client displayed. Stream tails go
    through :func:`_stream_tail` (tail-not-head, ``<empty>`` marker, ``...``
    truncation prefix) for consistency with those messages.

    Best-effort by construction: a disabled log returns before touching
    anything, and ANY error while writing is swallowed — a diagnostic aid must
    never mask, replace, or delay the real error.
    """
    path = _FAILURE_LOG_PATH
    if path is None:
        return
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "args": [_scrub_arg(arg) for arg in args],
            "exit_code": exit_code,
            "error_code": error_code,
            # Cap first, then scrub: bounding the work keeps the regex pass off a
            # pathological multi-MB message, and a URL that survives the cap is
            # still scrubbed in full.
            "message": _scrub_message(message[:_FAILURE_LOG_MESSAGE_CHARS]),
            "stdout_tail": _stream_tail(stdout, _FAILURE_LOG_TAIL_CHARS),
            "stderr_tail": _stream_tail(stderr, _FAILURE_LOG_TAIL_CHARS),
            "streaming": streaming,
        }
        _failure_logger(path).info(json.dumps(entry, ensure_ascii=False))
    except Exception:
        pass  # diagnostics are best-effort; never mask the real error


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Kill the child *and* any grandchildren it spawned.

    comfy-cli can fork a ComfyUI/helper grandchild that inherits the stderr
    pipe's write-end; killing only the direct child leaves that fd open, so the
    blocking ``proc.stderr.read()`` we run in a ``to_thread`` worker never sees
    EOF and the thread leaks — repeated timeouts then exhaust the default
    ``to_thread`` pool and wedge the server. Killing the whole process group
    (the child is spawned with ``start_new_session=True``, so it leads its own
    group) closes every copy of the pipe. Falls back to a plain ``kill`` on
    Windows / test fakes, where ``killpg`` is unavailable. (BE-3343)
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, AttributeError, ValueError):
        try:
            proc.kill()
        except (OSError, AttributeError):
            pass


def _reap(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Reap a (killed) child without blocking forever.

    A child stuck in uninterruptible sleep (D state) can ignore ``SIGKILL``
    indefinitely; ``Popen.wait(timeout=...)`` polls rather than blocking on it,
    so the timeout handler returns promptly instead of leaking the reaper
    thread. Best-effort: a still-unreaped child is left to the OS. (BE-3343)
    """
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def _unwrap_envelope(
    envelope: dict | None,
    args: tuple[str, ...],
    returncode: int | None,
    stderr: str,
    stdout: str = "",
    streaming: bool = False,
) -> Any:
    """Unwrap comfy-cli's ``envelope/1`` result, raising on error/absence.

    Shared by the plain (`--json`) and streaming (`--json-stream`) paths so both
    have identical terminal behavior: return ``data`` on success, and raise a
    :class:`ComfyCliError` carrying the envelope's ``error.code`` on failure.

    ``stdout`` is the RAW captured stdout the caller parsed ``envelope`` out of.
    It is only read on the no-envelope path, where it is the whole point: a
    comfy-cli that dies before emitting JSON usually prints its diagnosis as
    plain text, and every caller parses stdout through :func:`_last_json_object`
    — which drops that text on the floor. Passing it here is what keeps the
    failure legible; it defaults to ``""`` (rendered ``<empty>``) so a caller
    that genuinely has no stdout still produces a well-formed message.

    ``streaming`` only tags the failure-log record (:func:`_log_failure`) with
    which spawn path produced it — ``_run_comfy`` (``--json``) or
    ``_run_comfy_streaming`` (``--json-stream``) — since this function is shared
    by both and the raised error is otherwise identical either way.

    Also the envelope-version assertion: if comfy-cli declares an envelope
    ``schema`` whose major differs from :data:`ENVELOPE_SCHEMA_MAJOR`, the whole
    result shape is presumed incompatible and we refuse it with a clear error
    (rather than silently misreading a differently-shaped ``data``). An envelope
    with no declared schema is assumed compatible.
    """
    if envelope is None:
        if _looks_like_tcc_denial(stderr):
            # comfy-cli emitted no envelope because macOS denied it a protected
            # folder (see the TCC block above) — a permission problem the user
            # can fix, not the opaque "returned no JSON" this would otherwise be.
            message = (
                f"comfy-cli could not run (exit {returncode}).\n\n"
                f"{_tcc_guidance(_tcc_path_from(stderr))}\n\n"
                # `_stream_tail` for the truncation marker: `_looks_like_tcc_denial`
                # only fires on a non-empty stderr, so the `<empty>` half is
                # unreachable here — but a long denial traceback still gets
                # clipped, and silently is how you misread it as the whole thing.
                # No stdout here on purpose: this branch has already identified
                # the cause, so its curated guidance beats a second raw stream.
                f"Original error: {_stream_tail(stderr)}"
            )
            # Still `no_json` — a TCC denial is the *reason* comfy-cli emitted no
            # envelope, not a different kind of failure. Unlike the message, the
            # record keeps the raw stdout too: a diagnostic trail is read after
            # the fact, when a curated guidance string is no longer enough.
            _log_failure(
                "no_json",
                args,
                exit_code=returncode,
                message=message,
                stdout=stdout,
                stderr=stderr,
                streaming=streaming,
            )
            raise ComfyCliError(message, no_envelope=True, returncode=returncode)
        # Both streams, both explicitly marked when blank: comfy-cli splits its
        # diagnostics unpredictably (a Python traceback lands on stderr, a
        # Typer/click usage error or a plain-text status line on stdout), and
        # `_last_json_object` has already discarded the stdout text by the time
        # we get here. Rendering only stderr — and rendering an empty one as
        # nothing at all — is what made this error opaque.
        message = (
            f"comfy-cli returned no JSON (exit {returncode}). "
            f"stderr: {_stream_tail(stderr)} | stdout: {_stream_tail(stdout)}"
        )
        _log_failure(
            "no_json",
            args,
            exit_code=returncode,
            message=message,
            stdout=stdout,
            stderr=stderr,
            streaming=streaming,
        )
        raise ComfyCliError(message, no_envelope=True, returncode=returncode)
    # A declared schema must be a recognized ``envelope/<N>`` whose major matches.
    # Absent schema -> assume compatible (older comfy-cli); declared-but-unparseable
    # or a different major -> refuse loudly rather than fail open on a shape we
    # can't vouch for.
    schema = _envelope_schema(envelope)
    if schema is not None and _envelope_major(envelope) != ENVELOPE_SCHEMA_MAJOR:
        message = (
            f"incompatible comfy-cli envelope schema {schema!r}: "
            f"this server speaks envelope/{ENVELOPE_SCHEMA_MAJOR}. "
            "Upgrade or pin comfy-cli to a version whose envelope contract matches."
        )
        _log_failure(
            "schema_mismatch",
            args,
            exit_code=returncode,
            message=message,
            stdout=stdout,
            stderr=stderr,
            streaming=streaming,
        )
        raise ComfyCliError(message)
    if not envelope.get("ok", False):
        # A malformed envelope may set `error` to a non-dict (e.g. a bare
        # string); fall back to `{}` so `.get()` below can't raise AttributeError.
        err = envelope.get("error")
        if not isinstance(err, dict):
            err = {}
        code = err.get("code")
        # `error.code` can be any JSON type in a malformed envelope, including a
        # non-hashable list/dict that would make the `in _RETRYABLE_...`
        # membership test in run_workflow raise TypeError. Coerce to a string so
        # the retry check and the rendered message both stay well-defined.
        if code is not None and not isinstance(code, str):
            code = str(code)
        # Keep comfy-cli's actionable extras: `error.hint` (e.g. the working
        # `comfy auth set comfy-cloud-api-key --key …` credential fallback) and
        # useful `error.details` (e.g. the `partner_nodes` that lack a
        # credential) — dropping them was the exact workaround testers needed.
        # Each field is length-capped so a huge/malformed envelope can't bloat
        # the message propagated to the MCP client.
        # The stderr fallback goes through `_stream_tail` so an envelope with an
        # empty `error.message` AND an empty stderr can't render a bare trailing
        # colon with nothing after it. Note the cap is applied to the envelope's
        # own message only — `_stream_tail` already bounds its result, and
        # re-slicing its HEAD here would chop off the truncation marker plus the
        # very end of the tail, i.e. the part worth keeping.
        # Strip BEFORE the truthiness test: a whitespace-only `error.message`
        # ("   ") is truthy, so it would keep the fallback from firing and render
        # exactly the dangling-colon message this branch exists to prevent —
        # `_stream_tail` already treats a whitespace-only capture as `<empty>`,
        # so treat the envelope's own field the same way.
        raw_message = err.get("message")
        message = str(raw_message).strip() if raw_message else ""
        message = (
            message[:_MAX_ERROR_FIELD_CHARS]
            if message
            else _stream_tail(stderr, _MAX_ERROR_FIELD_CHARS)
        )
        parts = [f"comfy {' '.join(args)} failed [{code or 'unknown'}]: {message}"]
        hint = err.get("hint")
        if hint:
            parts.append(f"hint: {str(hint)[:_MAX_ERROR_FIELD_CHARS]}")
        detail_str = _render_error_details(err.get("details"))
        if detail_str:
            parts.append(detail_str)
        text = "\n".join(parts)
        # `error_code` carries the envelope's own `error.code` as a first-class
        # field, so a tester can `jq 'select(.error_code == "…")'` a run's
        # failures instead of string-matching the rendered sentence.
        _log_failure(
            "error_envelope",
            args,
            exit_code=returncode,
            error_code=code,
            message=text,
            stdout=stdout,
            stderr=stderr,
            streaming=streaming,
        )
        # `returncode` rides along on the envelope path too, so
        # `_is_missing_verb_error`'s Click-usage-exit condition stays genuinely
        # independent of its `no_envelope` provenance condition rather than
        # being a proxy for it (an envelope-borne failure can also exit 2).
        raise ComfyCliError(text, code=code, returncode=returncode)
    return envelope.get("data")


def _synthesize_plain_result(args: tuple[str, ...], stdout: str, stderr: str) -> dict:
    """Success payload for a ``plain_ok`` command that exited 0 without an envelope.

    Some comfy-cli commands print human-readable text and exit 0 instead of
    emitting an ``envelope/1`` object: the lifecycle verbs ``launch`` / ``stop``
    (BE-2953) and ``model download`` (BE-3345), whose stderr carries the progress
    tail (e.g. ``Done in 55.8s``) and the saved-path text. For those a clean exit
    IS the success signal, so we return a result dict carrying whatever text
    comfy-cli printed (preferring stderr, per the CLI's logging) rather than
    raising on the absent envelope — a false negative that would invite a retry
    of an action that already succeeded (a non-idempotent lifecycle change, or a
    bandwidth-expensive multi-GB refetch).

    The synthesized ``message`` is the only result data available without an
    envelope, so it carries the printed text verbatim (capped). This path is a
    stopgap: once comfy-cli emits an envelope for a verb, a real envelope always
    wins in the ``_run_comfy`` fast-path and this synthesis is bypassed.
    """
    # `action` is the subcommand path: the leading non-flag tokens, so `launch
    # --background` -> "launch", `stop` -> "stop", `model download --url ...` ->
    # "model download". Stops at the first flag so option values never leak in.
    action_parts: list[str] = []
    for arg in args:
        if arg.startswith("-"):
            break
        action_parts.append(arg)
    text = " ".join(part.strip() for part in (stderr, stdout) if part.strip())
    # Fallback echoes only the flag-free `action_parts`, never the raw args: a
    # `model download` URL can carry a signed token / userinfo in its query
    # string, and this message lands in the tool response and host-side logs.
    message = text or f"comfy {' '.join(action_parts)} completed (exit 0)."
    return {
        "ok": True,
        "action": " ".join(action_parts),
        # Keep the TAIL, not the front: `model download` streams verbose progress
        # to stderr and the saved-path / `Done in …` metadata this payload exists
        # to surface lands at the END, so a front slice would drop it as noise.
        "message": message[-1000:],  # cap both real output and the fallback
        "note": (
            "comfy-cli emitted no JSON envelope for this command; "
            "a clean exit is treated as success."
        ),
    }


# Error-envelope ``error.details`` keys worth surfacing verbatim in the raised
# message. ``partner_nodes`` names the offending nodes on a partner-credential
# failure; keep the set small so a large envelope can't bloat the message.
_SURFACED_DETAIL_KEYS = ("partner_nodes",)

# Per-field cap for the rendered error message (mirrors the stderr cap) so a
# multi-KB `message`/`hint` or a huge `partner_nodes` array can't produce an
# unbounded error string in the MCP client / logs.
_MAX_ERROR_FIELD_CHARS = 500


def _render_error_details(details: Any) -> str | None:
    """Render the useful keys of an envelope's ``error.details`` for the message."""
    if not isinstance(details, dict):
        return None
    parts: list[str] = []
    for key in _SURFACED_DETAIL_KEYS:
        value = details.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{key}: {str(value)[:_MAX_ERROR_FIELD_CHARS]}")
    return "; ".join(parts) if parts else None


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


def _real_envelope(obj: dict | None) -> dict | None:
    """Keep ``obj`` only if it is a genuine ``type==envelope``; else ``None``.

    The companion filter to :func:`_last_json_object`, which deliberately falls
    back to ANY JSON object on stdout so a caller can still see what comfy-cli
    printed. That fallback must never reach :func:`_unwrap_envelope` unfiltered:
    a stray progress/custom-node line would then be unwrapped as if it were the
    result — one carrying ``ok: true`` read as a successful run, and one without
    it raising a bogus ``failed [unknown]`` that SUPPRESSES the no-envelope
    branch and its stdout/stderr diagnostics, which is the only thing that
    explains a mid-run crash. Every path into ``_unwrap_envelope`` filters here
    first. An envelope declaring an incompatible ``schema`` still passes through
    on purpose — that is a real envelope, and ``_unwrap_envelope`` owns refusing
    it with the version error rather than a generic "returned no JSON".
    """
    return obj if obj and obj.get("type") == "envelope" else None


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
    _require_comfy_bin()
    # `_check_comfy_version` runs a synchronous `comfy --version` (up to 30s on
    # the first call per process); offload it so the async event loop is never
    # blocked while it runs.
    await asyncio.to_thread(_check_comfy_version)
    # Forward --host/--port into the subcommand for a configured remote ComfyUI
    # (no-op for the local default; see _with_target). run_workflow(wait=True)
    # -> `run` and watch_job -> `jobs watch` are both target-aware verbs.
    args = _with_target(args)
    # --json-stream is a global flag and, like --json/--where, MUST precede the
    # subcommand; a trailing form errors with "No such option".
    cmd = [COMFY_BIN, "--json-stream", "--where", "local", *args]
    env = _comfy_env()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Match the child's forced UTF-8 output (see _comfy_env); otherwise
        # readline()/stderr.read() decode with the parent locale (cp1252 on
        # Windows) and non-ASCII stream lines raise UnicodeDecodeError or
        # corrupt to mojibake before _parse_event/_last_json_object.
        encoding="utf-8",
        env=env,
        # Own process group so a timeout can kill the whole tree (child +
        # grandchildren) and close every copy of the stderr pipe — otherwise a
        # grandchild that inherited the fd keeps the blocking stderr read (and
        # its to_thread worker) alive forever. See _kill_proc_tree. (BE-3343)
        start_new_session=True,
    )
    lines: list[str] = []
    tracker = _StreamProgress()

    async def _pump() -> bool:
        """Read stdout until the terminal ``envelope/1`` line or stdout EOF.

        Returns True if the loop stopped on the terminal envelope (the
        authoritative result is already appended to ``lines``), False if it
        stopped on stdout EOF. Only the ``schema == "envelope/1"`` line is
        treated as terminal — an earlier or relayed ``type == "envelope"`` line
        that is not the run's result envelope (e.g. custom-node output) must not
        abort the read and kill the still-running child. Breaking on the real
        envelope keeps a fast run from sitting in ``readline`` when comfy-cli
        lingers after emitting it (see ``_POST_ENVELOPE_REAP_GRACE``).
        """
        assert proc.stdout is not None
        while True:
            line = await _in_pipe_pool(proc.stdout.readline)
            if not line:  # EOF: comfy-cli closed stdout
                return False
            lines.append(line)
            # Advance the tracker even without a ctx so a timed-out ctx-less
            # watch still returns real progress; report() no-ops the notify.
            event = _parse_event(line)
            if event is not None:
                if (
                    event.get("type") == "envelope"
                    and event.get("schema") == "envelope/1"
                ):
                    # Full result is in `lines`; it is unwrapped unchanged after
                    # the reap grace. Don't block in readline for a child that
                    # may outlive its own envelope under a pipe.
                    return True
                await tracker.report(ctx, event)

    # Drain stderr concurrently so a chatty child can't deadlock on a full pipe;
    # retain only the tail so it can't drive unbounded allocation here.
    stderr_future = (
        asyncio.ensure_future(
            _in_pipe_pool(_drain_capped, proc.stderr, _STDERR_MAX_CHARS)
        )
        if proc.stderr is not None
        else None
    )

    async def _read() -> tuple[bool, Any]:
        # Read up to the terminal envelope, then — on the EOF path only — reap
        # the child and its stderr. Both are bounded by the caller's `timeout`
        # (a child that closes stdout without exiting can't wedge the unbounded
        # proc.wait/stderr read). On the envelope path the reap is deliberately
        # left to the caller so it runs OFF the client budget.
        got_envelope = await _pump()
        if got_envelope:
            return True, None
        # EOF: the child has closed stdout, so a plain wait is safe and its
        # stderr is collectible for the error message.
        returncode = await _in_pipe_pool(proc.wait)
        stderr = (await stderr_future) if stderr_future is not None else ""
        # Keep the joined stdout around rather than only its parsed JSON: this is
        # the EOF path, so comfy-cli died without an envelope and whatever plain
        # text it printed is the only diagnosis there is.
        stdout_text = "".join(lines)
        # `_real_envelope` for the same reason `_run_comfy` applies it: reaching
        # EOF means `_pump` never saw a terminal envelope, so `_last_json_object`
        # here is usually its fallback — the last progress/custom-node event of a
        # crashed run. Unwrapping that would discard the diagnostics just
        # collected; filtering to None routes it to the no-envelope branch, which
        # is what actually reports why comfy-cli died.
        return False, _unwrap_envelope(
            _real_envelope(_last_json_object(stdout_text)),
            args,
            returncode,
            stderr,
            stdout=stdout_text,
            streaming=True,
        )

    try:
        # `_read` (reaching the envelope, plus the EOF-path reap) is bounded by
        # the client's `timeout`. Once the envelope has been read it IS the
        # answer; reaping a lingering child must NOT run on the client budget, or
        # an envelope that lands within the reap grace of the deadline would be
        # discarded and returned to sender as a spurious timeout (driving retries
        # / duplicate jobs).
        try:
            if timeout is not None:
                got_envelope, eof_result = await asyncio.wait_for(
                    _read(), timeout=timeout
                )
            else:
                got_envelope, eof_result = await _read()
            if not got_envelope:
                return eof_result
        except (asyncio.TimeoutError, TimeoutError) as exc:
            if not raise_on_timeout:
                # Bounded tail: report how far the run got instead of erroring
                # (the finally below still kills the child).
                return {"timed_out": True, "status": tracker.snapshot()}
            # Surface what the child wrote before the deadline (BE-3343). Kill
            # the whole tree FIRST so every copy of the stderr pipe closes and
            # the drain returns the buffered output (a wedged child — or a
            # grandchild holding the fd — would otherwise block the read).
            if proc.poll() is None:
                _kill_proc_tree(proc)
                await _in_pipe_pool(_reap, proc)
            # Keep the drained text itself, not only its 500-char message tail:
            # the failure log records a much longer slice (`_stream_tail` at
            # `_FAILURE_LOG_TAIL_CHARS`), and pre-truncating here would silently
            # cap it back down to the message's bound.
            stderr_text = ""
            if stderr_future is not None:
                try:
                    stderr_text = await asyncio.wait_for(stderr_future, 2.0) or ""
                except (Exception, asyncio.CancelledError):
                    # Diagnostics are best-effort: never let gathering the tail
                    # mask the timeout itself. CancelledError is a BaseException
                    # (not caught by `except Exception`) and DOES fire here: the
                    # outer wait_for cancels _drain while it awaits stderr_future,
                    # which cancels the future too — so awaiting it below re-raises
                    # CancelledError. Swallow it so we still raise ComfyCliError.
                    stderr_text = ""
            # Slice to the last lines before joining so a chatty child's full
            # stdout history isn't copied just to keep the 500-char tail.
            timeout_stdout = "".join(lines[-500:])
            message = (
                f"comfy-cli timed out after {timeout}s: {' '.join(cmd)}. "
                f"Progress so far: {tracker.snapshot()}. The run may still be "
                "going — check `job_status`, or for long generations submit "
                "with `wait=False` and poll `wait_for_job` / `watch_job`. "
                f"stderr tail: {_tail(stderr_text) or '<empty>'}; "
                f"stdout tail: {_tail(timeout_stdout) or '<empty>'}"
            )
            _log_failure(
                "timeout",
                args,
                message=message,
                stdout=timeout_stdout,
                stderr=stderr_text,
                streaming=True,
            )
            raise ComfyCliError(message) from exc

        # Envelope path: the authoritative result is already in `lines`, read
        # within the deadline. Give the child a brief grace to exit on a SEPARATE
        # budget; the `finally` kills a still-live one.
        stdout_text = "".join(lines)
        envelope = _last_json_object(stdout_text)
        child_reaped = True
        try:
            await asyncio.wait_for(
                _in_pipe_pool(proc.wait), timeout=_POST_ENVELOPE_REAP_GRACE
            )
        except (asyncio.TimeoutError, TimeoutError):
            child_reaped = False  # lingering child; `finally` reaps it
        # stderr only matters for an error envelope whose `error.message` is
        # empty (then `_unwrap_envelope` falls back to it). Collect it only when
        # the child already exited during the grace — its stderr pipe has EOF'd,
        # so the read can't block; a lingering child would, so skip it.
        stderr = ""
        if (
            child_reaped
            and stderr_future is not None
            and not (envelope or {}).get("ok", False)
        ):
            # The direct child exited during the grace, so its stderr pipe has
            # normally EOF'd — but a descendant holding the write fd could still
            # block this read. Bound it; `shield` keeps a timeout from cancelling
            # the reader here so the `finally` join can still detach it.
            try:
                stderr = await asyncio.wait_for(
                    asyncio.shield(stderr_future), _STDERR_JOIN_GRACE
                )
            except (asyncio.TimeoutError, TimeoutError):
                stderr = ""
        return _unwrap_envelope(
            envelope,
            args,
            proc.returncode,
            stderr,
            stdout=stdout_text,
            streaming=True,
        )
    finally:
        # Never leave a stray child or a dangling stderr reader on any exit path
        # (timeout, a report_progress error, or normal completion). Kill the
        # whole process tree (not just the direct child) so a descendant
        # holding the stderr write fd can't keep the pipe from EOFing — see
        # _kill_proc_tree. (BE-3343) Reap on the dedicated pipe pool, not the
        # default `to_thread` pool, so this wait can never contend with (or be
        # confused for) unrelated `to_thread` callers.
        if proc.poll() is None:
            _kill_proc_tree(proc)
            await _in_pipe_pool(_reap, proc)
        if stderr_future is not None and not stderr_future.done():
            stderr_future.cancel()


def _detect_comfy_cli_version() -> str | None:
    """Best-effort comfy-cli version via ``comfy --version`` (None if undetermined).

    A version string is a NICE-TO-HAVE, not load-bearing: comfy-cli builds that
    don't expose ``--version`` (or whose output we can't parse) return ``None``
    here and are reported as "unknown" rather than rejected — the envelope-schema
    assertion is the real gate. Kept separate from :func:`_run_comfy` because
    ``--version`` is a plain flag, not a ``--json`` envelope command.
    """
    if shutil.which(COMFY_BIN) is None:
        return None
    try:
        proc = subprocess.run(
            [COMFY_BIN, "--version"],
            capture_output=True,
            text=True,
            errors="replace",  # never crash on non-UTF-8 bytes; report None instead
            timeout=30.0,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    # Only trust stdout on a clean exit: a non-zero exit or a stderr warning can
    # carry an unrelated dotted number (an embedded Python / ComfyUI core version)
    # that _parse_version's first-match would wrongly report as the CLI version,
    # then falsely trip or bypass the COMFY_CLI_MIN_VERSION floor.
    if proc.returncode != 0:
        return None
    parsed = _parse_version(proc.stdout)
    return ".".join(str(part) for part in parsed) if parsed else None


def _check_comfy_cli_version() -> dict:
    """Compatibility report for the comfy-cli backing this server.

    Reports the detected comfy-cli version, the configured floor, and the
    envelope schema major this server speaks. Raises :class:`ComfyCliError` ONLY
    on a POSITIVE incompatibility — a detected version below a configured
    :data:`MIN_COMFY_CLI_VERSION`. An undetectable version is reported as
    ``None`` with a warning, never a hard failure, so a comfy-cli that simply
    doesn't expose ``--version`` still works.
    """
    detected = _detect_comfy_cli_version()
    report: dict[str, Any] = {
        "comfy_cli_version": detected,
        "min_comfy_cli_version": MIN_COMFY_CLI_VERSION,
        "envelope_schema_major": ENVELOPE_SCHEMA_MAJOR,
        "warnings": [],
    }
    if MIN_COMFY_CLI_VERSION:
        floor = _parse_version(MIN_COMFY_CLI_VERSION)
        got = _parse_version(detected) if detected else None
        if floor is None:
            # A misconfigured floor (e.g. "2" or "latest") would otherwise make
            # the whole check a silent no-op: the deployment believes it enforces
            # a minimum that never runs. Warn loudly instead of failing open.
            report["warnings"].append(
                f"COMFY_CLI_MIN_VERSION={MIN_COMFY_CLI_VERSION!r} is not a parseable "
                'version (expected e.g. "1.5.0"), so the configured minimum was '
                "NOT enforced."
            )
        elif got is None:
            report["warnings"].append(
                "could not determine the comfy-cli version, so the configured "
                f"minimum {MIN_COMFY_CLI_VERSION} was not verified."
            )
        elif got < floor:
            raise ComfyCliError(
                f"comfy-cli {detected} is older than the required minimum "
                f"{MIN_COMFY_CLI_VERSION} (set via COMFY_CLI_MIN_VERSION). "
                "Upgrade comfy-cli to a compatible version."
            )
    elif detected is None:
        report["warnings"].append("could not determine the comfy-cli version.")
    return report


# `click.UsageError.exit_code` — the status Click exits with when its parser
# rejects the command line (an unknown subcommand, a bad option) before
# dispatching to any command body. Typer inherits it.
_CLICK_USAGE_ERROR_EXIT = 2

# Click/Typer's "No such command '<verb>'." usage error, made robust to the ways
# that text arrives mangled. `\s+` between the words (not a literal space)
# because rich renders Typer errors inside a bordered panel and wraps them at the
# terminal width, so a newline can land mid-phrase; the box-drawing characters
# and ANSI colour codes that wrapping and styling introduce are stripped by
# `_normalize_cli_text` first. The verb follows within a few non-word characters
# (the quotes/colon/period around it) and must END there: `\b` would treat the
# hyphen in a DIFFERENT command like `outdated-notifier` as a word boundary and
# match it, so the lookahead rejects every character a command name could
# continue with — `\w` plus the `.`, `:`, `/`, `-` that appear in namespaced or
# hyphenated verbs — and `outdated.foo` no longer reads as `outdated`. At least
# one separator, since Click always writes a space and a quote there. See
# `_is_missing_verb_error`.
_MISSING_VERB_RE_TEMPLATE = r"no\s+such\s+command\W{{1,8}}{verb}(?![\w.:/-])"

# A CSI escape sequence (`\x1b[...m` and friends). Rich colourizes its error
# panels, and those codes contain word characters (digits, `m`), so leaving them
# in would let styling land between the matched words and defeat the pattern.
# The parameter-byte class is ECMA-48's full 0x30-0x3F range, not just `[0-9;]`:
# colon-separated SGR (`\x1b[38:5:130m`, true-colour on many terminals) would
# otherwise survive stripping and reintroduce the very problem.
_ANSI_RE = re.compile(r"\x1b\[[0-9:;<=>?]*[ -/]*[@-~]")

# Whitespace, the Unicode Box Drawing block (U+2500-U+257F), and ASCII `|` —
# i.e. every character rich can use to frame and wrap an error panel.
_PANEL_NOISE_RE = re.compile(r"[\s─-╿|]+")


def _normalize_cli_text(text: str) -> str:
    """Lowercased text with ANSI, panel borders, and wrapping folded away.

    Typer renders errors inside a rich panel when rich is installed, so the raw
    stderr of a usage error can read ``"│ No such command\\n│ 'outdated'. │"``,
    optionally colourized. Dropping the escape sequences and folding the border
    glyphs and any run of whitespace into one space puts that back on a single
    plain line, so a phrase match cannot be defeated by the terminal width the
    child happened to render at, or by whether it decided to emit colour.
    """
    return _PANEL_NOISE_RE.sub(" ", _ANSI_RE.sub("", text)).strip().lower()


def _is_missing_verb_error(exc: ComfyCliError, verb: str) -> bool:
    """Is *exc* comfy-cli rejecting ``verb`` as unknown, rather than *running* it?

    Deliberately narrow, because the caller's degrade tells the agent that
    NOTHING is broken: a false positive here silently buries a real failure.
    Two independent conditions must both hold.

    ``exc.no_envelope`` — comfy-cli emitted no envelope at all. An envelope, even
    a codeless one, means comfy-cli *recognized* the verb, ran it, and reported
    why it failed. A missing verb never gets that far: Click aborts with a usage
    error before any envelope is emitted, so the failure can only reach us via
    the wrapper-raised "returned no JSON" path. This is what stops a relayed
    nested error — a git/pip call, a custom-node pack name, a registry response
    that happens to contain "no such command" — from being mistaken for the verb
    itself being absent. Note this is checked instead of ``exc.code is None``,
    which looks equivalent but is not: an error envelope that merely omits
    ``error.code`` also yields a null code, and gating on that would let exactly
    the relayed-message case above through.

    ``exc.returncode == 2`` — Click's ``UsageError.exit_code``, i.e. the argument
    parser rejected the command line before dispatching anything. On its own
    ``no_envelope`` only says comfy-cli died before emitting JSON, which a verb
    it DID accept can also do by crashing mid-run; if such a crash happened to
    print our phrase, the degrade would swallow a genuine failure. Requiring the
    usage-error status narrows it to "never dispatched". The trade is
    deliberate and one-directional: a comfy-cli that someday reports an unknown
    verb with a different exit status just falls through to the raw passthrough
    below — the pre-existing behaviour, noisy but honest — whereas a wrong
    ``unsupported`` actively tells the user nothing is broken.

    Two residuals are known and accepted, both bounded by the conditions above:

    - Exit 2 is Click's status for ANY ``UsageError``, including one a command
      body raises after dispatch, so it does not *strictly* prove the parser
      rejected the verb. To reach a false ``unsupported`` through that door a
      recognized ``comfy outdated`` would have to raise a usage error mid-run,
      emit no envelope, AND print "no such command" naming ``outdated`` itself
      with a closing delimiter — i.e. reproduce the parser's own message about
      its own name. No further heuristic buys much here; the alternative is
      pattern-matching Click's usage preamble, which a mid-run ``UsageError``
      also prints.
    - The message this reads is built from bounded stream tails, so a wide rich
      panel could in principle push the phrase out of the slice. Click prints
      the error line LAST and the tail is what's kept, so it lands inside; if it
      ever did not, the miss fails toward the raw passthrough.

    The phrase must also name ``verb`` itself, within a few punctuation
    characters (Click writes ``No such command 'outdated'.``) and ending at a
    real delimiter, so a different command that merely starts with the same
    letters — ``outdated-notifier`` — does not match. Matching the bare phrase
    anywhere in the message would fold in the same relayed stderr the first
    condition exists to exclude.
    """
    if not exc.no_envelope or exc.returncode != _CLICK_USAGE_ERROR_EXIT:
        return False
    pattern = _MISSING_VERB_RE_TEMPLATE.format(verb=re.escape(verb))
    normalized = _normalize_cli_text(str(exc))
    return re.search(pattern, normalized, re.IGNORECASE) is not None


def _freshness_report() -> Any:
    """Best-effort installed-vs-latest report via ``comfy outdated``.

    Returns the ``comfy outdated`` payload (``core`` install status, one row per
    custom node ``packs`` entry, ``checked_at``) on success. It never raises, so
    the probe can never take ``server_info`` down with it; it degrades to one of
    two shapes instead.

    The MISSING-VERB degrade is its own shape: ``comfy outdated`` does not exist
    on any released comfy-cli (through 1.12.0), so on those installs this probe
    fails every time — and Click/Typer's raw ``No such command 'outdated'.``
    usage dump, relayed verbatim, reads like a broken MCP rather than the benign
    capability gap it is. That case returns
    ``{"error": "freshness unavailable: ...", "unsupported": True}``, with
    ``unsupported`` machine-readable so a client can branch on it without
    matching strings. :func:`_is_missing_verb_error` decides that case, and is
    deliberately strict: this degrade asserts nothing is broken, so a failure
    that merely *relays* a "no such command" from somewhere else must keep the
    raw passthrough below rather than be waved through as a capability gap.

    EVERY OTHER failure keeps the raw ``{"error": "<reason>"}`` passthrough — for
    a network failure, a timeout, or a decode error the underlying reason IS the
    diagnostic, so relaying it is the useful thing to do. ``OSError`` is caught
    because a spawn failure on this second subprocess (the env probe already
    succeeded) is still just the freshness probe failing, never grounds to fail
    ``server_info``. ``UnicodeDecodeError`` is caught too: ``_run_comfy_raw``
    decodes the child's stdout with strict ``encoding="utf-8"`` (no
    ``errors="replace"``), so non-UTF-8 bytes in a pack name/path from the
    user's live custom-node install can raise it here, same as the other probe
    failures above.
    """
    try:
        return _run_comfy("outdated", timeout=15.0)
    except (ComfyCliError, OSError, UnicodeDecodeError) as exc:
        # Click/Typer emits `No such command 'outdated'.` on stderr, which
        # `_unwrap_envelope` embeds in the raised message. `_is_missing_verb_error`
        # keeps that detection narrow — a relayed nested error that merely quotes
        # the same phrase must NOT reach this degrade, which claims nothing is
        # wrong.
        if isinstance(exc, ComfyCliError) and _is_missing_verb_error(exc, "outdated"):
            return {
                "error": (
                    "freshness unavailable: the installed comfy-cli does not support "
                    "'comfy outdated' (the verb ships in releases after 1.12.0). "
                    "Workflows are unaffected; update checks were skipped."
                ),
                "unsupported": True,
            }
        return {"error": str(exc)}


@mcp.tool()
def server_info() -> Any:
    """Report the local ComfyUI / comfy-cli environment and verify compatibility.

    Wraps ``comfy env``. Returns whether a local ComfyUI server is running and
    its URL, plus the selected workspace and Python info. Call this first to
    confirm a local ComfyUI is up before running a workflow.

    The reported server URL is the address comfy-cli RESOLVED, not a fixed
    default: ``COMFY_LOCAL_URL`` wins, else a background record, else
    ``127.0.0.1:8188`` (``comfy env`` itself takes no ``--host``). So this is
    also the right first call to verify a ``COMFY_LOCAL_URL`` override took
    effect. A URL still reading ``:8188`` after setting it has three causes,
    all silent and indistinguishable from here: the value did not reach
    comfy-cli, the comfy-cli on ``PATH`` predates the variable and ignored it,
    or the value was MALFORMED — comfy-cli then falls back to the default and
    emits only a one-line stderr warning, which the success path of this
    wrapper discards. Do not send the user straight to reinstalling comfy-cli:
    have them re-check the value's syntax (see the README's *Accepted values*)
    and, to see the dropped warning, run ``COMFY_LOCAL_URL=<value> comfy env``
    in a terminal.

    Also the compatibility gate for the unpinned comfy-cli this server shells
    out to: it asserts comfy-cli's envelope schema major matches the
    ``envelope/N`` this wrapper parses, and — when a ``COMFY_CLI_MIN_VERSION``
    floor is configured — that comfy-cli meets it. On a mismatch it raises
    :class:`ComfyCliError` saying so, catching an incompatible comfy-cli here
    rather than deep inside a later tool. On success it attaches a
    ``compatibility`` block (detected version, floor, envelope schema, warnings)
    alongside the ``comfy env`` data.

    Also attaches a ``freshness`` block (``comfy outdated``): ``core``
    (installed vs latest ComfyUI, with an ``outdated`` bool) and ``packs`` (one
    row per installed custom node pack). If ``freshness.core.outdated`` is true
    or any pack row has ``outdated: true``, the install is STALE — when a model,
    node, or template seems missing, tell the user to update FIRST
    (``comfy update comfy`` for core, ``comfy node update <pack>`` for a pack)
    before concluding the catalog lacks it; silent staleness is the usual
    culprit. The probe is best-effort and degrades two ways — ``server_info``
    itself still succeeds either way. On a comfy-cli that lacks the ``outdated``
    verb (no release through 1.12.0 has it), ``freshness`` is
    ``{"error": "freshness unavailable: ...", "unsupported": true}``:
    ``unsupported: true`` means SKIP staleness advice entirely and do NOT tell
    the user anything is broken — nothing failed, this comfy-cli just cannot
    answer the question, and workflows are unaffected. On any other probe
    failure (a network failure, a timeout, a decode error) ``freshness`` is
    ``{"error": "<reason>"}`` with no ``unsupported`` key, and that reason is
    the real diagnostic.

    Remote target: when a remote ComfyUI is configured (``COMFYUI_URL`` or
    ``COMFYUI_HOST`` — see :func:`_comfy_target`), a ``comfy_target`` block is
    attached reporting the ``host`` / ``port`` the run/queue tools drive, so an
    agent knows they are NOT targeting localhost. NOTE: the ``comfy env`` fields
    (running / url / workspace / python) always describe the LOCAL comfy-cli
    install — ``comfy env`` takes no ``--host`` — and this server never opens an
    HTTP socket (AGENTS.md), so it does not live-probe the remote here;
    reachability is confirmed by the first run/queue call, which targets the
    same host.
    """
    envelope, stdout, args, returncode, stderr = _run_comfy_raw("env", timeout=60.0)
    # `_run_comfy_raw` hands back `_last_json_object`'s answer unfiltered, so
    # enforce the envelope contract here exactly as `_run_comfy` does: an
    # incidental non-envelope JSON line from `comfy env` must raise "returned no
    # JSON" (with both stream tails) rather than be reported as server info.
    envelope = _real_envelope(envelope)
    # _unwrap_envelope raises if envelope is None, so it is non-None below.
    data = _unwrap_envelope(envelope, args, returncode, stderr, stdout=stdout)
    compat = _check_comfy_cli_version()
    compat["envelope_schema"] = _envelope_schema(envelope)
    freshness = _freshness_report()
    report = dict(data) if isinstance(data, dict) else {"env": data}
    report["compatibility"] = compat
    report["freshness"] = freshness
    # server_info is the "call first" diagnostic, so surface a malformed remote
    # config as a data field rather than raising — an agent debugging its env
    # then sees WHAT is wrong instead of an opaque failure of the whole tool.
    try:
        target = _comfy_target()
    except ComfyCliError as exc:
        report["comfy_target"] = {
            "error": str(exc),
            "note": (
                "COMFYUI_URL/COMFYUI_HOST is set but malformed; the run/queue "
                "tools will raise this same error until it is fixed."
            ),
        }
    else:
        if target is not None:
            host, port, source = target
            report["comfy_target"] = {
                "host": host,
                "port": port,
                "source": source,
                "note": (
                    "run/queue tools target this remote ComfyUI via --host/--port; "
                    "the env fields above describe the LOCAL comfy-cli install."
                ),
            }
    return report


# comfy-cli error codes worth a short bounded retry from ``run_workflow`` —
# transient credential failures the run's PREFLIGHT raises BEFORE the job is
# submitted, so re-invoking `comfy run` cannot double-submit. Verified against
# comfy-cli source (BE-3344):
#   * `partner_node_requires_credential` — raised in run preflight
#     (`command/run/__init__.py`) BEFORE `execution.queue()`; safe to retry.
#   * `cloud_unauthorized` — only raised on the CLOUD execute path; it never
#     fires on `--where local` (all this server ever runs), so it is dormant
#     here. Included defensively per the field request; it can't double-submit.
# `transient_auth` is deliberately EXCLUDED: on the local path it is raised
# from the execution watcher (`command/run/execution.py` `on_error`) AFTER
# submission, so retrying it would re-run a job that already executed.
_RETRYABLE_CREDENTIAL_CODES = frozenset(
    {"partner_node_requires_credential", "cloud_unauthorized"}
)

# Backoff (seconds) before each RETRY attempt — so up to 2 extra attempts after
# the initial one, at 1s then 2s.
_CREDENTIAL_RETRY_BACKOFFS = (1.0, 2.0)


@mcp.tool()
def auth_status() -> Any:
    """Comfy Cloud credential status for partner-API nodes (read-only; never returns secrets).

    Wraps ``comfy cloud whoami`` and returns comfy-cli's whoami payload as-is:
    ``signed_in``, ``auth_method`` (``oauth`` / ``api_key`` / ``null``),
    ``api_key_source`` (``env`` / ``store``), ``base_url``, plus
    ``expired`` / ``session`` (already REDACTED by comfy-cli) / ``stale_base_url``
    when a session exists. Secrets are pre-redacted upstream — this passes the
    payload through unchanged and never re-derives or returns key material.

    Call this before running a workflow whose nodes hit partner APIs
    (Seedream / Veo / Kling / Gemini / …) to self-diagnose credentials; the
    server instructions cover what to tell the user when not signed in.

    BLIND SPOT: a ``COMFY_API_KEY`` set in the MCP client's registration env
    (injected per-run for ``comfy run --api-key``) is NOT reflected in
    ``api_key_source`` — whoami inspects only the cloud-purpose
    ``COMFY_CLOUD_API_KEY`` / stored key slot. So this tool ALSO reports
    ``registration_env_key_present`` (a local presence check — a bool, never
    the value) so that path is at least visible. The flag is ALWAYS present in
    the returned mapping; on the rare non-dict whoami payload the raw value is
    nested under ``whoami`` alongside it.
    """
    data = _run_comfy("cloud", "whoami", timeout=30.0)
    present = bool(os.environ.get("COMFY_API_KEY"))
    # Add the local presence flag WITHOUT altering any whoami field comfy-cli
    # returned (which stays redacted as-is); only augment the dict shape. Always
    # report the flag as the docstring promises: if whoami ever hands back a
    # non-dict payload, wrap it under `whoami` rather than dropping the flag.
    if isinstance(data, dict):
        return {**data, "registration_env_key_present": present}
    return {"whoami": data, "registration_env_key_present": present}


@mcp.tool()
async def run_workflow(
    workflow_path: str,
    wait: bool = True,
    timeout_seconds: float = 110.0,
    ctx: Context | None = None,
) -> Any:
    """Run a ComfyUI workflow JSON on the LOCAL ComfyUI.

    Accepts an API-format or UI-export workflow file. Wraps
    ``comfy run --workflow <path>``. With ``wait=True`` (default) this waits
    until the run finishes and returns the full result, streaming live progress
    as MCP progress notifications (per-node execution + sampler step counts) so
    a long generation is not a silent block; with ``wait=False`` it submits and
    returns immediately with a ``prompt_id`` to poll via ``job_status``.

    ``timeout_seconds`` defaults to 110s — deliberately BELOW a typical MCP
    client's ~120s tool budget, so a genuinely slow run surfaces this wrapper's
    own actionable timeout (with a progress snapshot + next-step hint) instead
    of an opaque client-side deadline. Keep it under your client's tool timeout;
    for generations that may exceed it, submit with ``wait=False`` and poll
    ``wait_for_job`` / ``watch_job`` (the server INSTRUCTIONS teach this flow)
    rather than raising this bound.

    Partner-API nodes (Seedream/Veo/Kling/Gemini/…) need a Comfy credential in
    the server's environment (``COMFY_API_KEY`` in the client registration). A
    transient credential failure is retried up to twice with a short backoff;
    the surfaced error carries comfy-cli's hint (including the working
    ``comfy auth set comfy-cloud-api-key`` fallback).
    """

    async def _attempt() -> Any:
        if not wait:
            # Fire-and-return: no stream to follow, so keep the plain --json
            # path — but run the blocking subprocess in a worker thread so the
            # submit doesn't stall the event loop (and other concurrent MCP
            # requests) for up to the 60s timeout.
            return await asyncio.to_thread(
                _run_comfy, "run", "--workflow", workflow_path, timeout=60.0
            )
        return await _run_comfy_streaming(
            "run",
            "--workflow",
            workflow_path,
            "--wait",
            ctx=ctx,
            timeout=timeout_seconds,
        )

    # Try once, then up to len(_CREDENTIAL_RETRY_BACKOFFS) more times on a
    # transient credential code. ``backoff is None`` marks the final attempt.
    for attempt, backoff in enumerate((*_CREDENTIAL_RETRY_BACKOFFS, None)):
        try:
            return await _attempt()
        except ComfyCliError as exc:
            retryable = exc.code in _RETRYABLE_CREDENTIAL_CODES
            if backoff is None or not retryable:
                if attempt and retryable:
                    # Retries exhausted on a credential error: surface the
                    # hint-bearing error, noting the retries already made.
                    plural = "y" if attempt == 1 else "ies"
                    raise ComfyCliError(
                        f"{exc}\n(gave up after {attempt} retr{plural} on "
                        f"transient `{exc.code}`)",
                        code=exc.code,
                    ) from exc
                raise
            await asyncio.sleep(backoff)


# The gallery template `generate_image` runs: ComfyUI's own default graph — the
# basic SD1.5 text-to-image workflow whose CheckpointLoaderSimple default is
# `v1-5-pruned-emaonly-fp16.safetensors`. Free (core nodes only, no partner-API
# node, no `API` gallery tag), so the run never trips comfy-cli's spend gate.
_T2I_TEMPLATE = "default"

# Slot keys for that template's prompt + checkpoint inputs. These VERSION WITH
# `_T2I_TEMPLATE` — they are properties of that one graph, verified with
# `comfy templates fetch default -o wf.json && comfy workflow slots wf.json`:
#
#   4.ckpt_name | ckpt_name       | CheckpointLoaderSimple
#   6.text      | text (positive) | CLIPTextEncode
#   7.text      | text (negative) | CLIPTextEncode
#
# The prompt MUST use the node-address form: `text` is carried by BOTH
# CLIPTextEncode nodes, so the bare name is ambiguous and comfy-cli refuses it
# (`workflow_slot_invalid`) rather than guessing which one is the positive
# prompt. `ckpt_name` is unique in this graph, so the name form is used there —
# it survives a template revision that renumbers nodes.
_T2I_PROMPT_SLOT = "6.text"
_T2I_CHECKPOINT_SLOT = "ckpt_name"


def _t2i_config() -> tuple[str, str, str]:
    """Resolve ``generate_image``'s (template, prompt slot, checkpoint slot).

    Each is env-overridable so a user can point the on-ramp at a different local
    text-to-image graph without a code change. All three move TOGETHER: the slot
    keys describe one specific template, so overriding ``COMFY_T2I_TEMPLATE``
    alone will almost certainly leave the prompt address matching no slot in the
    new graph. List a replacement's slots with ``comfy templates fetch <name> -o
    wf.json && comfy workflow slots wf.json``.

    Read per call rather than latched at import so a test (or a client that
    re-execs with different env) sees the current value.
    """
    return (
        os.environ.get("COMFY_T2I_TEMPLATE") or _T2I_TEMPLATE,
        os.environ.get("COMFY_T2I_PROMPT_SLOT") or _T2I_PROMPT_SLOT,
        os.environ.get("COMFY_T2I_CHECKPOINT_SLOT") or _T2I_CHECKPOINT_SLOT,
    )


@mcp.tool()
async def generate_image(
    prompt: str,
    checkpoint: str | None = None,
    wait: bool = True,
    timeout_seconds: float = 600.0,
    ctx: Context | None = None,
) -> Any:
    """Generate an image from a text prompt on the LOCAL ComfyUI — the fast on-ramp.

    A single call that turns a text prompt into an image, so an agent does not
    have to hand-assemble a workflow graph. It runs ComfyUI's default SD1.5
    text-to-image gallery template through ``comfy run-template <name>
    --param=KEY=VALUE`` — the same verb (and the same local run path) as
    ``run_template``, with the prompt filled into the template's positive
    CLIPTextEncode slot. Returns the same envelope shape as ``run_workflow``
    (``prompt_id`` + outputs).

    The template is ``default`` unless ``COMFY_T2I_TEMPLATE`` overrides it; its
    prompt / checkpoint slot keys are overridable alongside it via
    ``COMFY_T2I_PROMPT_SLOT`` / ``COMFY_T2I_CHECKPOINT_SLOT``, and must be
    overridden together with the template since slot keys describe one specific
    graph; the two must name DIFFERENT slots (one key for both is refused rather
    than silently dropping the prompt). Pass ``checkpoint`` to swap the
    template's checkpoint model (it must
    already be installed locally — see ``search_models`` / ``download_model``);
    omit it to use the template's own default. The default template is a free,
    fully local OSS graph: nothing here spends Comfy credits, so no spend
    consent is passed and none is needed. (For hosted PARTNER models, which do
    spend, use ``partner_generate``.)

    With ``wait=True`` (default) this waits until the generation finishes and
    streams live progress as MCP progress notifications (per-node execution +
    sampler step counts) so a long generation is not a silent block; with
    ``wait=False`` it submits and returns immediately with a ``prompt_id`` to
    poll via ``job_status`` / ``wait_for_job`` / ``watch_job``.
    ``timeout_seconds`` only bounds the ``wait=True`` streaming path; the
    ``wait=False`` submit-and-return branch uses a fixed short timeout, so
    callers should not expect it to govern that case.

    This is the quickest path to an image. For full control — choosing a
    template, editing its graph, or running a hand-authored workflow — use the
    ``search_templates`` -> ``fetch_template`` -> ``run_workflow`` chain instead.

    Everything targets the LOCAL server (``--where local`` is injected by
    ``_run_comfy``), so there is no cloud reachability here.
    """
    template, prompt_slot, checkpoint_slot = _t2i_config()
    if not template:
        # Defensive: `_t2i_config` already falls back to the built-in template on
        # an empty env value, so an empty name should be unreachable from here.
        raise ComfyCliError(
            f"invalid COMFY_T2I_TEMPLATE: {template!r} — expected a gallery "
            "template name (e.g. 'default'), not an empty or option-like value."
        )
    # A leading-dash name is read by comfy-cli as an option, not the template
    # positional. Only reachable via a malformed COMFY_T2I_TEMPLATE, but a
    # named error beats comfy-cli's "No such option".
    _reject_option_like(
        "COMFY_T2I_TEMPLATE",
        template,
        expected="a gallery template name (e.g. 'default')",
    )
    _reject_nul("template name", template)
    # The free-form prompt rides inside a single `--param=KEY=VALUE` token, so a
    # prompt that begins with `-` (or contains `=`) is carried as the value
    # rather than mis-parsed by comfy-cli as an option. `_run_template_param_args`
    # owns that escaping, the JSON value rendering, and the key validation.
    params: dict[str, Any] = {prompt_slot: prompt}
    if checkpoint:
        if checkpoint_slot == prompt_slot:
            # Same key for both slots would have the checkpoint overwrite the
            # prompt already stored under it, running the template's DEFAULT
            # prompt with no error at all — the worst failure mode available
            # (a plausible wrong image). Only reachable via a misconfigured
            # override pair; refuse it by name instead.
            raise ComfyCliError(
                f"generate_image's prompt slot and checkpoint slot both resolve "
                f"to {prompt_slot!r} — the checkpoint would overwrite the "
                "prompt. Set COMFY_T2I_PROMPT_SLOT / COMFY_T2I_CHECKPOINT_SLOT "
                f"to the two different slots of template {template!r} (list "
                f"them with `comfy templates fetch {template} -o wf.json && "
                "comfy workflow slots wf.json`)."
            )
        params[checkpoint_slot] = checkpoint
    timeout_seconds = _bounded_timeout(timeout_seconds, _MAX_RUN_TEMPLATE_TIMEOUT)
    args, budget = _run_template_argv(
        template,
        _run_template_param_args(params),
        wait=wait,
        timeout_seconds=timeout_seconds,
    )
    # No `--allow-spend`, and deliberately no `_require_spend_gate` probe: that
    # gate is `comfy generate`-scoped, and this template is free. A
    # `spend_consent_required` here would mean the constant above names a paid
    # template — fix the constant, not the consent plumbing.
    try:
        if not wait:
            # Fire-and-return: no stream to follow, so keep the plain --json
            # path — off the event loop, in the same pool `run_template` uses.
            args.append("--async")
            return await _in_generate_pool(
                _run_comfy, *args, timeout=budget + _RUN_TEMPLATE_TIMEOUT_GRACE
            )
        # Same grace as the submit path above (and as `run_template`): the child
        # was handed `--timeout=min(budget, 120)`, so for a budget at or under
        # comfy-cli's 120s cap the engine's deadline and the parent's kill land
        # on the SAME instant. Without slack the parent can SIGKILL comfy-cli
        # mid-write of its own structured timeout / `server_not_running` result,
        # replacing an actionable error with a generic parent kill (and orphaning
        # an already-enqueued run). The engine must be the side that gives up.
        return await _run_comfy_streaming(
            *args, ctx=ctx, timeout=budget + _RUN_TEMPLATE_TIMEOUT_GRACE
        )
    except ComfyCliError as exc:
        hinted = _t2i_slot_hint(
            exc, template, prompt_slot, checkpoint_slot if checkpoint else None
        )
        if hinted is exc:
            # Not a slot failure — let the engine's own error through untouched,
            # with its original traceback rather than a self-referential cause.
            raise
        raise hinted from exc


def _t2i_slot_hint(
    exc: ComfyCliError, template: str, prompt_slot: str, checkpoint_slot: str | None
) -> ComfyCliError:
    """Re-raise a slot-resolution failure with the knob that fixes it, else pass through.

    The slot keys above are pinned to one revision of one template, so the day
    the gallery renumbers that graph (or a ``COMFY_T2I_TEMPLATE`` override names
    a graph with a different shape) comfy-cli answers ``workflow_slot_invalid``
    with the template's real addresses — accurate, but it says nothing about
    WHICH knob in this server produced the bad key. Name them.

    ``checkpoint_slot`` is None when the call passed no ``checkpoint``: that slot
    was never sent, so naming it would implicate a knob that cannot be the cause
    and send the reader after the wrong env var.
    """
    if exc.code != "workflow_slot_invalid":
        return exc
    filled = f"prompt slot {prompt_slot!r}"
    knobs = "COMFY_T2I_TEMPLATE / COMFY_T2I_PROMPT_SLOT"
    if checkpoint_slot is not None:
        filled += f" and checkpoint slot {checkpoint_slot!r}"
        knobs += " / COMFY_T2I_CHECKPOINT_SLOT"
    return ComfyCliError(
        f"{exc}\n(generate_image filled template {template!r} using {filled}; "
        f"set {knobs} to match the addresses above, or use run_template "
        "directly)",
        code=exc.code,
    )


# comfy-cli reserves these words as `comfy generate` SUB-ACTIONS (its own
# list / schema / refresh / upload / resume / consent verbs) rather than model
# aliases. This tool's contract is "run this partner MODEL", so a reserved word
# is refused instead of silently dispatching a different verb — `consent` in
# particular is the spend gate's own configuration surface.
_GENERATE_RESERVED_TARGETS = frozenset(
    {"list", "schema", "refresh", "upload", "resume", "consent"}
)

# comfy-cli treats these `comfy generate` flags as RUN-level rather than model
# inputs (its `_separate_meta_flags`): they change how the call runs, not what
# is generated. They are refused inside `params` so a "model parameter" can
# never silently retarget the run — above all `yes`, which would otherwise be a
# second, undocumented way to grant spend consent behind `confirm_spend`'s back,
# and `json` / `async`, which would break this tool's result contract.
_GENERATE_META_FLAGS = frozenset(
    {
        "download",
        "async",
        "json",
        "timeout",
        "api-key",
        "emit-workflow",
        "output-prefix",
        "yes",
    }
)

# Hard ceiling for one partner generation, so `float('inf')` / an absurd value
# can't hold the `comfy generate` child open effectively forever (1 hour).
# Partner VIDEO models are the slow end of the range, hence an hour not minutes.
_MAX_GENERATE_TIMEOUT = 3600.0

# Head-room between the deadline comfy-cli is given (`--timeout`) and the one
# this process enforces by killing the child. The ENGINE must be the side that
# gives up: it ends the run cleanly, reports why, and — for a job the partner
# already accepted (and charged for) — leaves a resumable handle behind. A
# parent SIGKILL at the same instant would instead destroy that handle and
# surface as a generic failure, inviting a retry that spends the credits twice.
# The parent timeout is only the backstop for a child that ignores its own
# deadline. (comfy-cli applies its deadline per phase — request, then poll — so
# a pathologically slow request followed by a full poll can still reach this
# backstop; it is a floor on engine-owned failure, not a proof of it.)
_GENERATE_TIMEOUT_GRACE = 60.0

# Whether the installed comfy-cli carries the credit-spend interlock. Latched
# only on success: a probe that fails for a transient reason must not wedge the
# tool for the life of the process.
_spend_gate_probed = False


def _require_spend_gate() -> None:
    """Refuse to run a spending call unless comfy-cli's spend gate is installed.

    This tool's core safety claim is that ``confirm_spend=False`` spends nothing
    because comfy-cli fails CLOSED. That interlock landed in comfy-cli AFTER the
    ``>= 1.12.0`` floor :data:`_MIN_COMFY_CLI` enforces (which was chosen for
    ``comfy logs`` / ``envelope/1``), so the version check cannot prove it is
    present — and against a comfy-cli without it the default call would silently
    charge the user's card.

    ``comfy generate consent`` is the gate's OWN configuration surface and ships
    with it, so a clean exit is the capability signal; on an older CLI
    ``consent`` falls through to the model lookup and exits non-zero. It is a
    local, read-only config query (no network, no spend).

    Unlike :func:`_check_comfy_version`, which fails OPEN so an unreadable
    ``--version`` can never wedge a working install, this fails CLOSED: the cost
    of guessing wrong here is the user's money, not an error message.
    """
    global _spend_gate_probed
    if _spend_gate_probed:
        return
    try:
        _run_comfy("generate", "consent", "show", timeout=30.0, plain_ok=True)
    # Broad on purpose: the probe must fail CLOSED with THIS explanation, not
    # leak a raw OSError/UnicodeDecodeError from a present-but-unusable binary.
    except Exception as exc:
        raise ComfyCliError(
            "this comfy-cli has no `comfy generate` spend gate, so a generation "
            "would spend Comfy credits with no consent interlock — refusing. "
            "Upgrade comfy-cli (`pip install -U comfy-cli`) to a release that "
            "includes `comfy generate consent`, or run `comfy generate` yourself "
            f"if you intend to spend. (probe: {exc})"
        ) from exc
    _spend_gate_probed = True


def _engine_auto_confirms() -> bool:
    """True when comfy-cli's persisted ``spend.auto_confirm`` is on.

    The DURABLE "always proceed" for credit spending lives in comfy-cli's own
    config (``comfy generate consent always``), not here — this server stays
    stateless and remembers nothing between calls. When the user has set it, the
    engine consents to its own spending call and there is nothing left to ask,
    so :func:`partner_generate` skips the per-call prompt and forwards no
    ``--yes``: the consent is the engine's, and it stays the engine's.

    ``comfy generate consent show --json`` prints the setting as a JSON object
    (a pretty-printed one, so it is read from the whole of stdout rather than
    the line-oriented envelope parser). Read fresh on every call — a latched
    answer would keep prompting after the user turned the setting on, or worse,
    keep NOT prompting after they turned it off.

    The trailing ``--json`` is REQUIRED and is not the global one: ``comfy
    generate`` is registered with ``allow_extra_args``/``ignore_unknown_options``
    so the argv tail after the target reaches the subcommand's own meta-flag
    parser, and ``consent`` only prints JSON when IT sees ``--json``. Without it
    the command prints rich human text and this parse fails — which is why the
    global ``--json`` (which must still precede the subcommand) is not enough.

    Best-effort, and every failure answers ``False``: an unreadable setting must
    fall through to ASKING the user, never to assuming they already said yes.
    :func:`_require_spend_gate` — not this — is what refuses a comfy-cli with no
    interlock at all, so a ``False`` here is never mistaken for "no gate".
    """
    try:
        _, stdout, _, returncode, _ = _run_comfy_raw(
            "generate", "consent", "show", "--json", timeout=30.0
        )
    # Broad on purpose, to keep the "every failure answers False" contract
    # above literally true: a present-but-non-executable binary
    # (`PermissionError`/`OSError`) or invalid-UTF-8 child output
    # (`UnicodeDecodeError`) escapes `_run_comfy_raw` uncaught, and crashing
    # `partner_generate` is strictly worse than falling back to asking.
    # `False` is the safe direction — it can only ever cause a prompt.
    except Exception:
        return False
    if returncode != 0:
        return False
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return False
    # Tolerate comfy-cli one day wrapping this verb in an `envelope/1` the way
    # the other verbs are: read the setting out of `data` when it does.
    if isinstance(payload, dict) and payload.get("type") == "envelope":
        payload = payload.get("data")
    # `is True` on purpose: only a real JSON `true` authorizes spending. A
    # string, a 1, or a missing key is not consent.
    return isinstance(payload, dict) and payload.get("spend_auto_confirm") is True


class SpendApproval(BaseModel):
    """What the client returns from the per-call spend-confirmation prompt.

    Deliberately one boolean rather than a bare accept/decline: consent has to
    be an AFFIRMATIVE answer to the question "spend credits?", so a client (or
    an agent host) that accepts the elicitation without actually answering it
    lands on the ``False`` default and is treated as a refusal.
    """

    approve: bool = Field(
        default=False,
        title="Spend Comfy credits on this generation?",
        description=(
            "Yes runs the hosted partner model and spends credits from the "
            "Comfy account this machine is signed into. No cancels it and "
            "spends nothing."
        ),
    )


def _client_elicitation_support(ctx: Context | None) -> bool | None:
    """Whether the connected MCP client advertised the elicitation capability.

    Tri-state, because "the client said no" and "we could not find out" must not
    be answered the same way on the money path:

    - ``False`` — DEFINITELY not capable: no context (a direct call, or a host
      that injects none), no ``elicit``, or a session predating elicitation.
      The caller falls back to the explicit ``confirm_spend`` argument rather
      than hanging on a request the client will never answer.
    - ``True`` — the client declared the capability at handshake.
    - ``None`` — UNKNOWN: the capability probe itself raised. Answering ``False``
      here would silently downgrade a genuinely capable client to the fallback
      path, so a caller-supplied ``confirm_spend=True`` would spend credits with
      no human prompt — the one outcome this tool exists to prevent. The caller
      treats ``None`` as "ask anyway" (see :func:`_resolve_spend_consent`).
    """
    if ctx is None or not callable(getattr(ctx, "elicit", None)):
        return False
    try:
        session = ctx.session
    except (AttributeError, ValueError):
        # `Context.session` raises ValueError outside a live request.
        return False
    check = getattr(session, "check_client_capability", None)
    if check is None:
        return False
    try:
        return bool(
            check(types.ClientCapabilities(elicitation=types.ElicitationCapability()))
        )
    except Exception:
        return None


# How long the user gets to answer the spend prompt before it lapses into a
# refusal. `timeout_seconds` bounds only the generation that follows, so without
# this a client that advertises elicitation but never answers leaves the request
# pending forever and stuck calls accumulate with nothing to reclaim them.
# Generous, because a human has to notice the prompt and decide.
_SPEND_ELICIT_TIMEOUT = 300.0

# Cap on how much of a caller-supplied model name is echoed into the prompt.
_ELICIT_MODEL_DISPLAY_MAX = 80


def _display_model(model: str) -> str:
    """Render a caller-supplied model name safely inside the elicitation prompt.

    The prompt quotes the model in a markdown code span, and the name arrives
    from the CALLER — an agent that may be relaying untrusted text. Backticks or
    newlines in it would close that span on a client that renders markdown,
    letting the name inject its own content: hiding the "SPENDS credits" warning
    or appending a reassuring "this is free". That redresses the very prompt the
    user is answering, so it is neutralized before display.

    Display only — argv still carries the model verbatim, so a name comfy-cli
    would accept is never mangled into one it would not.
    """
    cleaned = "".join(
        " " if ch.isspace() or not ch.isprintable() else ch for ch in model
    )
    # The span delimiter itself: without a backtick the rest of markdown is
    # inert inside the code span, so this is the only character that must go.
    cleaned = " ".join(cleaned.replace("`", "'").split())
    if len(cleaned) > _ELICIT_MODEL_DISPLAY_MAX:
        cleaned = cleaned[:_ELICIT_MODEL_DISPLAY_MAX] + "…"
    # `partner_generate` rejects an empty model before reaching here; the
    # fallback only covers a name that was ENTIRELY unprintable.
    return cleaned or "<unnamed model>"


async def _elicit_spend_approval(ctx: Context, message: str, schema: type) -> bool:
    """Raise one spend-confirmation prompt and report whether it was approved.

    The shared body behind every spend prompt (``partner_generate``'s and
    ``run_template``'s): only the wording and the answer schema differ, while the
    fail-closed handling below must not. True = the user affirmatively approved.
    """
    try:
        result = await asyncio.wait_for(
            ctx.elicit(message=message, schema=schema),
            timeout=_SPEND_ELICIT_TIMEOUT,
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        # Ordered before the catch-all: on 3.11+ these are the same class, but
        # an unanswered prompt deserves its own message.
        raise ComfyCliError(
            "spend not confirmed: the confirmation prompt went unanswered for "
            f"{_SPEND_ELICIT_TIMEOUT:.0f}s, so it was treated as a refusal. "
            "Nothing was spent."
        ) from exc
    except Exception as exc:
        # Name the way out. Because an errored capability probe now routes here
        # rather than to `confirm_spend`, a client this server cannot prompt
        # would otherwise dead-end with no route to a generation it is entitled
        # to run — and the user's own durable consent is exactly that route.
        raise ComfyCliError(
            "could not confirm the credit spend with the user: the client "
            f"failed to answer the confirmation prompt ({exc}). Nothing was "
            "spent. If this client cannot show prompts, record your consent "
            "with comfy-cli directly — `comfy generate consent always` — and "
            "this tool will honor it without asking."
        ) from exc
    # Every read is a `getattr`: a non-conforming client can return an object
    # with no `.action`/`.data`, and an AttributeError here would escape as an
    # uncaught crash instead of the refusal this contract promises.
    if getattr(result, "action", None) != "accept":
        return False
    return getattr(getattr(result, "data", None), "approve", False) is True


async def _elicit_spend_consent(ctx: Context, model: str) -> bool:
    """Ask the USER to approve this one credit-spending call. True = approved.

    The MCP-native spend confirmation: one prompt per call, answered by the
    human, never remembered. A decline, a cancel, an accept that did not
    actually say yes, a client that errors on the request, a client that answers
    with something malformed, and a prompt left unanswered past
    :data:`_SPEND_ELICIT_TIMEOUT` all fail closed — the caller spends nothing.
    """
    return await _elicit_spend_approval(
        ctx,
        (
            f"Run the hosted partner model `{_display_model(model)}`? "
            "This SPENDS Comfy credits from the account this machine is "
            "signed into. Running a workflow on the local ComfyUI is free."
        ),
        SpendApproval,
    )


async def _resolve_spend_consent(
    model: str, confirm_spend: bool, ctx: Context | None
) -> bool:
    """Decide whether this call may spend, and whether to forward ``--yes``.

    Returns True to forward ``--yes`` (comfy-cli's explicit non-interactive
    consent) and False to forward nothing. Raises :class:`ComfyCliError` — with
    no child process ever spawned — when consent was actively refused.

    The precedence, and the reason for it:

    1. **The engine's durable always-proceed** (``spend.auto_confirm``) wins. The
       user pre-authorized spending in comfy-cli's own config, so there is
       nothing to ask and no ``--yes`` to add: the engine consents to itself.
    2. **Elicitation**, unless the client is KNOWN not to support it — the
       per-call human confirmation this tool is built around. Approve forwards
       ``--yes``; decline raises here, so the refusal is enforced BEFORE
       comfy-cli runs rather than relying on the engine to fail closed
       afterwards. A capability probe that could not answer counts as "ask":
       being wrong that way costs a prompt, the other way costs money.
    3. **The explicit ``confirm_spend`` argument**, only as the fallback for a
       client that cannot elicit. Left ``False`` (the default) nothing is
       forwarded and comfy-cli's own gate fails closed.

    Note what is NOT in that list: the agent host's permission to CALL this
    tool. Spend consent and tool permission are different questions, and an
    "always allow this tool" toggle answers only the second — so on an
    elicitation-capable client the prompt is raised even when the caller passed
    ``confirm_spend=True``. Otherwise a host-level convenience setting would
    quietly become standing authority over the user's credits.
    """
    if await asyncio.to_thread(_engine_auto_confirms):
        return False
    # `None` is the probe's "could not tell" and is treated as CAPABLE, so an
    # errored probe cannot quietly demote a real client onto the `confirm_spend`
    # fallback and spend without asking. Trying to elicit is the safe way to be
    # wrong: on a client that truly cannot answer, `_elicit_spend_consent`
    # raises (or lapses at `_SPEND_ELICIT_TIMEOUT`) having spent nothing.
    if _client_elicitation_support(ctx) is not False:
        if await _elicit_spend_consent(ctx, model):
            return True
        raise ComfyCliError(
            f"spend not confirmed: the user declined to spend Comfy credits on "
            f"`{model}`. Nothing was spent and no generation was started."
        )
    return confirm_spend


def _generate_param_args(params: dict[str, Any]) -> list[str]:
    """Marshal per-model ``params`` into ``comfy generate`` ``--name=value`` tokens.

    ``comfy generate`` takes a model's inputs as schema-driven flags whose names
    and types come from that model's OWN schema, so this wrapper neither knows
    nor validates them: each pair is forwarded verbatim for comfy-cli to accept
    or reject. The ``--name=value`` form (rather than two argv tokens) means a
    value that begins with ``-`` is read as the value instead of being
    mis-parsed as the next option.

    Conversions are spelling-only, so comfy-cli's parser sees the form it
    expects: ``None`` drops the flag entirely (rather than sending the string
    "None"), bools become ``true`` / ``false``, and list / dict values are
    JSON-encoded — what its array parser accepts. Everything else is
    ``str()``-rendered.
    """
    argv: list[str] = []
    for name, value in params.items():
        if not name:
            raise ComfyCliError(
                f"invalid parameter name: {name!r} — expected a model parameter "
                "name (e.g. 'prompt'), not an empty or option-like value."
            )
        _reject_option_like(
            "parameter name",
            name,
            expected="a model parameter name (e.g. 'prompt')",
        )
        # A name carrying its own `=` (or whitespace) is not a parameter name:
        # comfy-cli splits `--<body>` at the FIRST `=`, so `{"output-prefix=/tmp/x":
        # v}` renders `--output-prefix=/tmp/x=v` and lands as the run-level
        # `output-prefix` flag — smuggling past the meta-flag check below, which
        # only ever sees the whole key. Refuse the shape instead of trying to
        # out-parse comfy-cli's splitter.
        if "=" in name or any(ch.isspace() for ch in name):
            raise ComfyCliError(
                f"invalid parameter name: {name!r} — a parameter name cannot "
                "contain '=' or whitespace. Pass the value as the dict value, "
                "not inside the key."
            )
        _reject_nul(f"parameter name {name!r}", name)
        # Compare hyphen-normalized so `api_key` / `emit_workflow` are caught
        # too; agents naturally spell CLI flags with underscores. Case is NOT
        # normalized: comfy-cli matches its run-level flags case-sensitively in
        # lower case, so `Json` can never reach one, while a model's schema
        # flags come verbatim from its OpenAPI property names and may legitimately
        # be capitalized — folding case here would refuse a real parameter to
        # block an unreachable one.
        if name.replace("_", "-") in _GENERATE_META_FLAGS:
            raise ComfyCliError(
                f"`{name}` is a run-level `comfy generate` flag, not a model "
                "parameter. Use this tool's own arguments where they cover it "
                "(confirm_spend for --yes, download, timeout_seconds); the "
                "remaining run-level flags are not forwarded by this tool, so "
                "use comfy-cli directly for those."
            )
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        _reject_nul(f"value for parameter {name!r}", rendered)
        argv.append(f"--{name}={rendered}")
    return argv


@mcp.tool()
async def partner_generate(
    model: str,
    params: dict[str, Any] | None = None,
    confirm_spend: bool = False,
    download: str | None = None,
    timeout_seconds: float = 600.0,
    ctx: Context | None = None,
) -> Any:
    """Run a hosted PARTNER model (Flux / Ideogram / DALL·E / Recraft / …) — SPENDS CREDITS.

    Thin passthrough to ``comfy generate <model> [--param=value]…``. Unlike
    ``generate_image`` / ``run_workflow``, which execute on the user's own
    machine for free, this calls a hosted partner API through comfy-cli and so
    **spends the user's Comfy credits**.

    SPEND CONSENT — read before calling. comfy-cli puts the credit-spending call
    behind a consent interlock, and this wrapper does not implement, weaken, or
    reimplement it: the engine decides whether a call may spend, and this only
    reports the consent it was actually given. Where that consent comes from,
    in precedence order (see :func:`_resolve_spend_consent`):

    - The user's DURABLE always-proceed in comfy-cli's own config
      (``comfy generate consent always``). It stays engine-side — this server
      remembers nothing between calls — so when it is set there is nothing to
      ask and no ``--yes`` to send; the engine consents to itself.
    - A PER-CALL confirmation raised on the client through MCP **elicitation**,
      the same primitive an interactive terminal's y/N prompt serves. Approve
      and ``--yes`` is forwarded; decline and this raises :class:`ComfyCliError`
      without ever starting comfy-cli, so nothing is spent.
    - ``confirm_spend=True``, the fallback for a client that cannot elicit: it
      forwards ``--yes`` directly. Set it ONLY when the user has actually agreed
      to spend credits on this call — never merely to clear the error you just
      hit. On a client that CAN elicit the user is asked anyway, so it is not a
      way around the prompt.

    Spend consent is not tool permission: a host's "always allow this tool"
    setting authorizes CALLING this tool, never spending the user's money, and
    is never read as consent here.

    With none of the three, comfy-cli fails CLOSED (an MCP server has no TTY for
    its own prompt) and this raises having spent nothing. Because that
    fail-closed guarantee is the engine's, this refuses to run at all against a
    comfy-cli that predates the gate (see :func:`_require_spend_gate`).

    ``params`` carries the model's OWN inputs (``prompt``, ``aspect_ratio``,
    ``seed``, …). These are schema-driven per model and are forwarded verbatim;
    discover a model's real parameters, and the available aliases, with
    comfy-cli directly: ``comfy generate schema <model>`` / ``comfy generate
    list``. ``download`` forwards ``--download <path>`` so comfy-cli saves the
    generated asset there. ``timeout_seconds`` is forwarded as comfy-cli's own
    ``--timeout`` (clamped to an hour; partner video models are the slow end),
    so the ENGINE owns the deadline and can report a resumable job rather than
    being killed mid-flight; this process only enforces a slightly later
    backstop.

    NOTE: ``comfy generate`` prints its result as human-readable text and exits
    0 WITHOUT emitting an ``envelope/1``, so this runs through the same
    ``plain_ok`` stopgap as ``launch`` / ``stop`` / ``model download``: a clean
    exit is the success signal and the payload carries the printed text. A
    non-zero exit — including the consent refusal — still raises.
    """
    if not model:
        raise ComfyCliError(
            f"invalid model: {model!r} — expected a partner model alias "
            "(e.g. 'flux-pro'), not an empty or option-like value."
        )
    # A leading-dash target is read by comfy-cli as an option rather than a
    # model (the same guard watch_job applies to prompt_id).
    _reject_option_like(
        "model", model, expected="a partner model alias (e.g. 'flux-pro')"
    )
    if model in _GENERATE_RESERVED_TARGETS:
        raise ComfyCliError(
            f"invalid model: {model!r} is a `comfy generate` sub-action, not a "
            "partner model. Use comfy-cli directly for those verbs."
        )
    _reject_nul("model", model)
    timeout_seconds = _bounded_timeout(timeout_seconds, _MAX_GENERATE_TIMEOUT)
    args = ["generate", model, *_generate_param_args(params or {})]
    if download is not None:
        if not download:
            # Distinguish "no path given" (None -> comfy-cli's default location)
            # from an empty string, which is a caller mistake: silently dropping
            # it saves the asset somewhere the caller did not ask for.
            raise ComfyCliError(
                "invalid download: empty path — omit `download` to let comfy-cli "
                "choose the default location, or pass a real path."
            )
        _reject_nul("download", download)
        # `--flag=value` so a path beginning with `-` stays the value.
        args.append(f"--download={download}")
    # Hand the deadline to the engine so IT owns giving up (see
    # `_GENERATE_TIMEOUT_GRACE`); the parent timeout below is only the backstop.
    # This is also what makes `timeout_seconds` real: comfy-cli's own default is
    # 300s, so before this a caller asking for longer silently got five minutes.
    args.append(f"--timeout={timeout_seconds}")
    # Prove the engine's interlock exists BEFORE asking the user to approve a
    # spend — there is no point prompting for a call this would refuse anyway.
    await asyncio.to_thread(_require_spend_gate)
    if await _resolve_spend_consent(model, confirm_spend, ctx):
        # comfy-cli's non-interactive spend consent; a bare boolean meta flag.
        args.append("--yes")
    # `_run_comfy` blocks for as long as the generation takes (up to an hour),
    # so it runs off the event loop — this tool is async for the elicitation
    # round-trip above and must not wedge the server while a partner model runs.
    # Its OWN pool, not the shared `to_thread` one: cancelling this await does
    # not interrupt the thread, so an abandoned run stays parked until comfy-cli
    # returns and would otherwise sit on the default executor for up to an hour.
    # See `_GENERATE_EXECUTOR`.
    return await _in_generate_pool(
        _run_comfy,
        *args,
        timeout=timeout_seconds + _GENERATE_TIMEOUT_GRACE,
        plain_ok=True,
    )


# Hard ceiling for one template run (video templates are the slow end), so a
# `float('inf')` / absurd value can't hold the `comfy run-template` child open
# effectively forever. Matches partner_generate's ceiling.
_MAX_RUN_TEMPLATE_TIMEOUT = 3600.0

# comfy-cli's own default for `run-template --timeout`. That flag is a PER-EVENT
# bound (the same semantics as `comfy run --timeout`) rather than a whole-run
# deadline, and it also bounds the engine's initial "is ComfyUI up?" probe.
_RUN_TEMPLATE_EVENT_TIMEOUT = 120

# Wall-clock budget for a `wait=False` submit: fetch the template, fill slots,
# enqueue, return the prompt_id. Not a run deadline — the run outlives the call.
_RUN_TEMPLATE_ASYNC_TIMEOUT = 60.0

# Slack the parent allows beyond the budget handed to the engine, so comfy-cli
# gets to report its OWN error (`server_not_running`, a per-event stall) instead
# of being SIGKILLed mid-write. Mirrors `_GENERATE_TIMEOUT_GRACE`.
_RUN_TEMPLATE_TIMEOUT_GRACE = 30.0


class TemplateSpendApproval(BaseModel):
    """What the client returns from the template spend-confirmation prompt.

    Separate from :class:`SpendApproval` only for its wording: a template MAY
    spend (most are free OSS graphs) where a partner model always does, and the
    prompt should not overstate. The affirmative-answer design is the same — an
    accept that never answered lands on ``False`` and reads as a refusal.
    """

    approve: bool = Field(
        default=False,
        title="Allow this template to spend Comfy credits?",
        description=(
            "Yes lets the run proceed even if the template contains "
            "partner-API (paid) nodes, spending credits from the Comfy account "
            "this machine is signed into. No cancels it and spends nothing; a "
            "template with no paid nodes runs free either way."
        ),
    )


async def _resolve_template_spend_consent(
    name: str, confirm_spend: bool, ctx: Context | None
) -> bool:
    """Decide whether to forward ``--allow-spend`` for this template run.

    The same principle as :func:`_resolve_spend_consent` — an agent's own
    ``confirm_spend=True`` is not the user's consent to spend money, so on a
    client that can elicit, the human is asked — but the shape differs on two
    points that are specific to this verb:

    1. **No prompt when nothing can be spent.** ``confirm_spend=False`` forwards
       nothing, so comfy-cli's gate fails closed and a paid template cannot
       spend; there is nothing to consent to. Most gallery templates are free
       OSS graphs, so prompting on every call would train the user to click
       through the one prompt that matters. The prompt is raised only when the
       caller is actually asking to unlock spending.
    2. **comfy-cli's durable always-proceed does NOT apply here.** ``run-template``
       never reads ``spend.auto_confirm`` — the setting is scoped to
       ``comfy generate`` (it is that gate's own configuration surface, and its
       own status line says so). Unlike :func:`_resolve_spend_consent`, there is
       therefore no branch that lets the engine consent to itself: it would send
       no flag and the run would fail closed anyway, having asked nobody.

    Returns True to append ``--allow-spend``. Raises :class:`ComfyCliError` —
    before any child is spawned — when the user actively declined.
    """
    if not confirm_spend:
        return False
    # `None` (the probe itself errored) counts as "ask", for the same reason as
    # on the generate path: guessing "cannot elicit" would silently demote a
    # capable client onto the caller's own say-so and spend without a human.
    if _client_elicitation_support(ctx) is not False:
        if await _elicit_template_spend_consent(ctx, name):
            return True
        raise ComfyCliError(
            f"spend not confirmed: the user declined to let the template "
            f"{name!r} spend Comfy credits. Nothing was spent and no run was "
            "started. (A template with no partner-API nodes runs for free — "
            "call again with confirm_spend=False to run it without spending.)"
        )
    # Client cannot elicit: `confirm_spend` is the documented fallback.
    return True


async def _elicit_template_spend_consent(ctx: Context, name: str) -> bool:
    """Ask the USER to approve credit spend for this one template run."""
    return await _elicit_spend_approval(
        ctx,
        (
            f"Run the gallery template `{_display_model(name)}` with credit "
            "spending ALLOWED? Most templates are free graphs that run on this "
            "machine, but one containing partner-API nodes SPENDS Comfy credits "
            "from the account this machine is signed into."
        ),
        TemplateSpendApproval,
    )


def _reject_nul_deep(label: str, value: Any) -> None:
    """Reject an embedded NUL anywhere inside a JSON-shaped param value.

    Slot values are JSON-encoded, so ``json.dumps`` escapes a NUL to ``\\u0000``
    and no raw NUL ever reaches argv — this is not an injection guard. It exists
    because a NUL in a template slot is never intentional, and rejecting it only
    at the top level (``{"a": "\\0"}``) while silently forwarding it one level
    down (``{"a": ["\\0"]}``) is the worse of the two behaviors: the nested case
    lands a literal ``\\u0000`` in the filled graph. Recurses into lists/dicts —
    including dict KEYS, which are slot-internal JSON, not the ``--param`` key.

    Depth is bounded by the same recursion limit the MCP layer's own JSON parse
    already survived, so this adds no new failure mode.
    """
    if isinstance(value, str):
        _reject_nul(label, value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                _reject_nul(label, key)
            _reject_nul_deep(label, item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nul_deep(label, item)


def _run_template_param_args(params: dict[str, Any]) -> list[str]:
    """Marshal template ``params`` into ``comfy run-template`` ``--param=KEY=VALUE`` tokens.

    ``comfy run-template`` fills a template's parameterized slots: KEY is a slot
    address (``6.text``) or a unique slot name (``prompt``), and VALUE parses as
    JSON with a string fallback. Each value is JSON-encoded so its Python type
    round-trips exactly — ``42`` stays an int, the string ``"42"`` stays a
    string, ``True`` becomes ``true``, and lists/dicts become JSON arrays/objects
    — rather than leaning on the bare-string fallback, which would coerce a
    numeric-looking string to a number. ``None`` drops the pair entirely. The
    single ``--param=KEY=VALUE`` token (comfy-cli splits on the FIRST ``=``)
    keeps a value that contains ``=`` or begins with ``-`` intact.
    """
    argv: list[str] = []
    for key, value in params.items():
        if not key:
            raise ComfyCliError(
                f"invalid param key: {key!r} — expected a slot address (e.g. "
                "'6.text') or a slot name (e.g. 'prompt'), not an empty or "
                "option-like value."
            )
        _reject_option_like(
            "param key",
            key,
            expected="a slot address (e.g. '6.text') or a slot name (e.g. 'prompt')",
        )
        # `=` is the load-bearing one: comfy-cli splits the `--param` value on
        # its FIRST `=` to separate slot key from value, so a key carrying its
        # own `=` mis-splits. Refuse the shape rather than out-parse the
        # splitter. Whitespace is refused for a weaker reason — KEY rides inside
        # the single `--param=KEY=VALUE` token, so it is never argv-ambiguous,
        # but comfy-cli `.strip()`s the key and slot names come from node input
        # names (0 of the 574 gallery templates carry whitespace or a leading
        # dash in one). A clear error here beats the engine's "matches no slot".
        if "=" in key or any(ch.isspace() for ch in key):
            raise ComfyCliError(
                f"invalid param key: {key!r} — a slot key cannot contain '=' or "
                "whitespace. Pass the value as the dict value, not in the key."
            )
        _reject_nul(f"param key {key!r}", key)
        if value is None:
            continue
        # json.dumps escapes a NUL inside a string value (so it can't crash
        # subprocess the way a raw NUL in the KEY would), but a NUL slot value is
        # never intentional — refuse it explicitly, matching partner_generate.
        # Checked recursively: a NUL nested in a list/dict is the same mistake
        # and would otherwise land as a literal `\u0000` in the filled graph.
        _reject_nul_deep(f"value for param {key!r}", value)
        rendered = json.dumps(value)
        argv.append(f"--param={key}={rendered}")
    return argv


def _run_template_argv(
    name: str, param_args: list[str], *, wait: bool, timeout_seconds: float
) -> tuple[list[str], float]:
    """Build the ``run-template`` argv (sans consent/``--async``) + the parent budget.

    Shared by :func:`run_template` and :func:`generate_image` so the engine
    deadline rule lives in exactly one place. ``wait``'s budget is the caller's
    (already bounded) ``timeout_seconds``; a ``wait=False`` submit gets the fixed
    short :data:`_RUN_TEMPLATE_ASYNC_TIMEOUT` instead, since the run outlives the
    call.

    Hand the engine a deadline it can act on. Unlike ``comfy generate --timeout``,
    this one is PER-EVENT, not a whole-run bound, so the caller's total budget
    cannot simply be forwarded; it is used only to LOWER the engine's bound when
    that budget is smaller than comfy-cli's 120s default. Without it a short
    budget is consumed entirely inside the engine's own 120s server probe and the
    child is SIGKILLed with no diagnostic — e.g. ``wait=False`` had a 60s budget
    against a 120s probe. Never RAISED above the default: that would blunt stall
    detection on long runs. comfy-cli types this flag as an int, so a float is a
    parse error.
    """
    budget = timeout_seconds if wait else _RUN_TEMPLATE_ASYNC_TIMEOUT
    args = ["run-template", name, *param_args]
    args.append(f"--timeout={max(1, int(min(budget, _RUN_TEMPLATE_EVENT_TIMEOUT)))}")
    return args, budget


@mcp.tool()
async def run_template(
    name: str,
    params: dict[str, Any] | None = None,
    confirm_spend: bool = False,
    wait: bool = True,
    timeout_seconds: float = 600.0,
    ctx: Context | None = None,
) -> Any:
    """Run a gallery template on the LOCAL ComfyUI — fetch, fill params, execute.

    Thin passthrough to ``comfy run-template <name> [--param=KEY=VALUE]…`` (the
    engine fetches the template graph, fills its parameterized slots, and runs it
    through the same local run path as ``run_workflow``). Named ``run_template``
    for contract parity with the cloud MCP's ``run_template(name, params)``; this
    is the one-command alternative to the manual ``search_templates`` →
    ``fetch_template`` → ``run_workflow`` chain.

    ``params`` fills the template's parameterized slots — ``{slot: value}`` where
    a slot is an address (``"6.text"``) or a unique name (``"prompt"``). List a
    template's slots by fetching it (``fetch_template``) and inspecting the graph.
    Values are forwarded verbatim for comfy-cli to accept or reject.

    SPEND CONSENT — most gallery templates are free OSS graphs that run entirely
    on the user's own machine. SOME embed partner-API (paid) nodes, and running
    one spends the user's Comfy credits. comfy-cli gates that path and this
    wrapper only passes consent through (see
    :func:`_resolve_template_spend_consent`):

    - ``confirm_spend=False`` (the default) forwards nothing, so a paid template
      fails CLOSED (``spend_consent_required``, nothing spent) while a free
      template runs normally. Nothing can be spent, so the user is NOT prompted —
      free template runs stay a single, silent call.
    - ``confirm_spend=True`` asks to unlock spending, and on a client that
      supports MCP **elicitation** the USER is prompted per call before anything
      runs. Approve and ``--allow-spend`` is forwarded; decline and this raises
      :class:`ComfyCliError` without starting comfy-cli. Only on a client that
      cannot elicit does the argument stand on its own, as the fallback.

    So ``confirm_spend=True`` is a REQUEST to spend, not the consent itself: set
    it only when the user has actually agreed, never merely to clear the error
    you just hit. Spend consent is not tool permission — a host's "always allow
    this tool" toggle authorizes calling this tool, never spending the user's
    money, and is never read as consent here. This mirrors ``partner_generate``,
    with two differences that verb's shape forces: it always spends so it always
    prompts, and comfy-cli's durable ``comfy generate consent always`` is scoped
    to ``comfy generate`` — ``run-template`` does not read it, so it grants
    nothing here.

    Unlike ``partner_generate``, this does NOT probe for the interlock first. It
    does not need to: ``partner_generate``'s gate landed in comfy-cli *after* the
    verb it guards, so the presence of ``comfy generate`` could not prove the gate
    was there and :func:`_require_spend_gate` had to ask. Here the gate is inline
    in ``run-template``'s own command body and shipped in the same change as the
    verb, so THE VERB IS THE CAPABILITY SIGNAL — a comfy-cli with ``run-template``
    but without the gate does not exist, and one without ``run-template`` exits
    non-zero (raising :class:`ComfyCliError`) having spent nothing. Probing
    ``comfy generate consent`` here would test an unrelated subsystem and would
    wrongly refuse FREE, local-only template runs on a CLI that lacks it.

    ``timeout_seconds`` bounds this call's wall clock. comfy-cli's own
    ``--timeout`` for this verb is PER-EVENT rather than a whole-run deadline, so
    it is forwarded only to tighten the engine's bound when ``timeout_seconds``
    is shorter than comfy-cli's 120s default — that way a short deadline surfaces
    the engine's own error instead of a signal kill. For a long (e.g. video) run,
    prefer ``wait=False`` over a large ``timeout_seconds``: a run killed at the
    deadline may already be queued, and only the async path hands back the
    ``prompt_id`` needed to track it rather than re-running it.

    With ``wait=True`` (default) this waits until the run finishes and returns the
    result (``prompt_id`` + outputs); with ``wait=False`` it submits ``--async``
    and returns immediately with a ``prompt_id`` to poll via ``job_status`` /
    ``wait_for_job`` / ``watch_job`` — use that for long (e.g. video) runs that
    may exceed your MCP client's tool timeout. OSS templates need their referenced
    models installed locally; a missing model surfaces the run path's per-node
    error (see ``search_models`` / ``download_model``). Everything targets the
    LOCAL server (``--where local`` is injected by ``_run_comfy``).
    """
    if not name:
        raise ComfyCliError(
            f"invalid template name: {name!r} — expected a template name "
            "(e.g. 'image_flux2'), not an empty or option-like value."
        )
    # A leading-dash name is read by comfy-cli as an option, not the template
    # positional (the same guard partner_generate applies to its model).
    _reject_option_like(
        "template name", name, expected="a template name (e.g. 'image_flux2')"
    )
    _reject_nul("template name", name)
    timeout_seconds = _bounded_timeout(timeout_seconds, _MAX_RUN_TEMPLATE_TIMEOUT)
    # argv + the engine deadline are built by the shared helper (see
    # `_run_template_argv` for why `--timeout` is needed and never raised).
    args, budget = _run_template_argv(
        name,
        _run_template_param_args(params or {}),
        wait=wait,
        timeout_seconds=timeout_seconds,
    )
    if await _resolve_template_spend_consent(name, confirm_spend, ctx):
        # comfy-cli's paid-node consent for run-template; a bare boolean flag.
        args.append("--allow-spend")
    if not wait:
        # Fire-and-return: submit and hand back a prompt_id to poll.
        args.append("--async")
    # The parent stays the backstop only: a little slack past the budget so
    # comfy-cli reports its own error rather than dying to a signal. Runs off
    # the event loop in the dedicated pool for the same reason
    # `partner_generate` does: this blocks for as long as the template takes (up
    # to an hour) and the tool is async for the consent round-trip above.
    return await _in_generate_pool(
        _run_comfy, *args, timeout=budget + _RUN_TEMPLATE_TIMEOUT_GRACE
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
    # comfy-cli parses a leading-dash positional as an option/flag; reject it
    # rather than let `jobs status` misread the id (argument injection).
    _reject_option_like("prompt_id", prompt_id)

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
    # comfy-cli parses a leading-dash positional as an option/flag; reject it
    # rather than let `jobs watch` misread the id (argument injection).
    _reject_option_like("prompt_id", prompt_id)
    timeout_seconds = _bounded_timeout(timeout_seconds, _MAX_WATCH_TIMEOUT)
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


def _drop_cloud_jobs(data: Any) -> Any:
    """Return ``comfy jobs ls`` data with cloud-tracked rows removed.

    comfy-cli merges its on-disk job state files into ``jobs ls`` without
    scoping them to the requested ``--where``, so a listing this server asked
    for as ``--where local`` can still carry rows from a prior CLOUD run. This
    server is local-only, so those rows are noise at best and misleading at
    worst — drop them here rather than let the caller reason about jobs it
    cannot act on. Once comfy-cli scopes the merge itself this becomes a no-op.

    Deliberately defensive: this filter never raises and never reshapes a
    payload it does not recognize. Only a ``dict`` carrying a ``list`` of jobs
    is touched, only rows POSITIVELY marked ``"cloud"`` are dropped (a row with
    no ``where`` is a legacy local row and is kept), and the input object is
    returned unchanged when nothing was dropped.
    """
    if not isinstance(data, dict):
        return data
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        return data
    kept = [
        row
        for row in jobs
        if not (isinstance(row, dict) and row.get("where") == "cloud")
    ]
    if len(kept) == len(jobs):
        return data
    # Shallow copy: the caller's ``count`` must match the rows we return, and
    # mutating comfy-cli's parsed payload in place is not this helper's call.
    return {**data, "jobs": kept, "count": len(kept)}


@mcp.tool()
def get_queue() -> Any:
    """List known LOCAL jobs with their status (pending / running / completed).

    Wraps ``comfy jobs ls``. comfy-cli merges its on-disk job state with the
    running ComfyUI server's queue, so this returns both jobs still in the queue
    and recently completed ones — call it to find a ``prompt_id`` to inspect with
    ``job_status`` or cancel with ``cancel_job``.

    LOCAL ONLY: jobs comfy-cli tracks in its state store from a CLOUD run are
    filtered out of the listing, because this server drives the user's local
    ComfyUI and nothing else. Passing a cloud job's ``prompt_id`` to
    ``job_status`` / ``cancel_job`` would route locally regardless, so listing
    those ids here would only invite calls that cannot work.
    """
    return _drop_cloud_jobs(_run_comfy("jobs", "ls", timeout=60.0))


# Image suffixes we return inline from ``fetch_outputs`` — kept to the formats
# ``mcp.server.fastmcp.Image`` maps to a real ``image/*`` MIME type (an unknown
# suffix would fall back to ``application/octet-stream`` and not render).
_INLINE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")

# Bounds on what ``fetch_outputs(inline_images=True)`` base64-inlines into the
# reply, mirroring the module's other output caps (``_TRACEBACK_TAIL_MAX_CHARS``,
# ``_EXCEPTION_TEXT_MAX_CHARS``): a big batch or high-res render must not force an
# unbounded allocation / blow the agent's context. The on-disk copies in
# ``out_dir`` are untouched — only the inline preview is capped.
_INLINE_IMAGE_MAX_COUNT = 8
_INLINE_IMAGE_MAX_BYTES = 16 * 1024 * 1024


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


def _is_within(root: str, path: str) -> bool:
    """True if ``path`` (already realpath'd) is ``root`` itself or nested under it."""
    return path == root or path.startswith(root + os.sep)


def _collect_output_images(data: Any, out_dir: str) -> list[str]:
    """Resolve image files referenced by ``comfy download``'s data to on-disk paths.

    Walks every string in the envelope ``data``, keeps those with an image
    suffix, and returns the ones that resolve to a real file **inside**
    ``out_dir`` (deduped, order-preserving). ``comfy download -o out_dir`` writes
    every file it produces into ``out_dir``, so scoping to that directory is what
    keeps the inline preview honest: a bare/relative name binds to the copy just
    written rather than a same-named file in the process CWD, and an absolute or
    ``../``-traversal path that escapes ``out_dir`` (an input reference, a URL
    basename, or an outright traversal in the metadata) is rejected instead of
    read and inlined. Inline return is best-effort and never masks the on-disk
    copy.
    """
    out_root = os.path.realpath(out_dir)
    resolved: dict[str, None] = {}
    for value in _iter_strings(data):
        if not value.lower().endswith(_INLINE_IMAGE_SUFFIXES):
            continue
        # Most-specific form first (the value as given, then joined onto out_dir,
        # then bare basename in out_dir) — but every candidate must resolve to a
        # real file INSIDE out_dir. Containment is what neutralizes the CWD
        # shadow (a bare "gen.png" resolves to CWD/gen.png, outside out_dir, so
        # it's rejected in favor of the out_dir copy) and the `../` traversal.
        for candidate in (
            value,
            os.path.join(out_dir, value),
            os.path.join(out_dir, os.path.basename(value)),
        ):
            real = os.path.realpath(candidate)
            if _is_within(out_root, real) and os.path.isfile(real):
                resolved.setdefault(real, None)
                break
    return list(resolved)


def _select_inline_images(paths: list[str]) -> list[str]:
    """Cap the inlined set to ``_INLINE_IMAGE_MAX_COUNT`` files / aggregate bytes.

    Preserves order and stops as soon as either bound would be exceeded, so a
    large batch or a high-res render can't force an unbounded base64 payload into
    the reply. Unreadable files are skipped (the on-disk copy still stands).
    """
    selected: list[str] = []
    total = 0
    for path in paths:
        if len(selected) >= _INLINE_IMAGE_MAX_COUNT:
            break
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if selected and total + size > _INLINE_IMAGE_MAX_BYTES:
            break
        selected.append(path)
        total += size
    return selected


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
    yields no inline images. The inline preview is capped
    (``_INLINE_IMAGE_MAX_COUNT`` files / ``_INLINE_IMAGE_MAX_BYTES`` aggregate) so
    a large batch can't blow up the reply — the on-disk copies are never capped.
    """
    args = ["download", prompt_id, "-o", out_dir]
    if url_only:
        args.append("--url-only")
    data = _run_comfy(*args, timeout=300.0)
    # ``url_only=True`` downloads no bytes, so there is nothing on disk to inline
    # — short-circuit rather than let basename matching surface stale files from
    # a previous run into ``out_dir`` (which would contradict the docstring).
    if not inline_images or url_only:
        return data
    paths = _select_inline_images(_collect_output_images(data, out_dir))
    images = [Image(path=path) for path in paths]
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

    ``comfy launch --background`` prints human text and exits 0 without a JSON
    envelope, so on success this returns a synthesized ``{"ok": True, ...}``
    payload carrying that text (BE-2953); a launch failure (e.g. port in use)
    exits non-zero and still raises a :class:`ComfyCliError`.

    NOTE (temporary upstream caveat): ``comfy launch --background`` currently
    crashes on Python 3.14 (comfy-cli asyncio ``get_event_loop`` issue; a fix is
    in review upstream). On affected comfy-cli versions the crash surfaces here
    as a clean :class:`ComfyCliError` from the error envelope. Remove this note
    once the upstream fix ships.
    """
    args = ["launch", "--background"]
    if extra_args:
        args += ["--", *extra_args]
    return _run_comfy(*args, timeout=180.0, plain_ok=True)


@mcp.tool()
def stop_comfyui() -> Any:
    """Stop the LOCAL ComfyUI server that comfy-cli launched.

    Wraps ``comfy stop``. Ownership semantics: comfy-cli only kills the pid it
    recorded when IT launched the server via ``launch_comfyui`` /
    ``comfy launch --background``. It therefore cannot stop a ComfyUI started by
    the desktop app or by hand — in that case comfy-cli reports it has no
    recorded server and this tool raises a :class:`ComfyCliError` carrying that
    message, rather than killing an unrelated process.

    Like ``launch_comfyui``, ``comfy stop`` prints human text and exits 0 without
    a JSON envelope, so a successful stop returns a synthesized
    ``{"ok": True, ...}`` payload carrying that text (BE-2953).
    """
    return _run_comfy("stop", timeout=60.0, plain_ok=True)


@mcp.tool()
def restart_comfyui(extra_args: list[str] | None = None) -> Any:
    """Restart the LOCAL ComfyUI server: stop the running one, then launch a fresh one.

    Composes the existing :func:`stop_comfyui` and :func:`launch_comfyui` — there
    is no ``comfy restart`` subcommand, so this is a thin stop-then-launch over
    comfy-cli, not a new engine feature. ``extra_args`` are forwarded to the new
    ComfyUI exactly as :func:`launch_comfyui` forwards them (after a ``--``
    separator), so a restart is also how you relaunch with different flags.
    Returns the new server's status (``launch_comfyui``'s envelope data).

    The stop step is best-effort ONLY for the benign "nothing to stop" case: if
    comfy-cli has no recorded server (e.g. nothing is running, or ComfyUI was
    started outside comfy-cli) it returns the ``no_recorded_server`` code, which
    is swallowed so the restart still brings the server up. Any OTHER stop
    failure (a process that couldn't be killed, a permission error, a comfy-cli
    malfunction) is re-raised rather than silently masked behind the launch.
    """
    try:
        stop_comfyui()
    except ComfyCliError as exc:
        if not _is_no_recorded_server(exc):
            raise
    return launch_comfyui(extra_args)


# comfy-cli's `logs` reports this error code when no persisted log file exists
# yet (nothing has been launched in the background, so nothing was captured).
_NO_LOG_FILE_CODE = "no_log_file"

# Bounds for get_logs' caller-controlled `tail`: at least 1 line (a negative
# value would forward a malformed `--tail -N`), capped so an absurd request
# can't make comfy-cli read/return an enormous log slice.
_MIN_LOG_TAIL = 1
_MAX_LOG_TAIL = 10000

# Hard character cap on each returned log line. `_MAX_LOG_TAIL` bounds the line
# COUNT, but a single pathological line — a base64 blob or tensor dump from a
# buggy or hostile custom node — could still be megabytes and flood an agent's
# context. Cap each line individually, mirroring the `_cap_text` guard on
# get_execution_error's free-text fields. This is a TOTAL cap: the truncation
# marker is charged against it (see get_logs) so a capped line never exceeds it.
_MAX_LOG_LINE_CHARS = 4000


@mcp.tool()
def get_logs(tail: int = 200) -> Any:
    """Return the tail of the LOCAL background ComfyUI's captured log file.

    Wraps ``comfy logs --tail <tail>``. comfy-cli persists a background ComfyUI's
    stdout/stderr to ``<workspace>/user/comfyui_<port>.log`` (written when it is
    started via ``launch_comfyui`` / ``comfy launch --background``), so this
    closes the debugging loop after a detached launch — the server's output is
    otherwise invisible. Returns ``{lines, path, truncated}``: the last ``tail``
    log lines, the file they came from, and whether older lines were dropped.

    If no log file exists yet (nothing was launched in the background), comfy-cli
    returns a ``no_log_file`` error envelope; rather than raise, this tool returns
    it as data — ``{"error": "no_log_file", "message": ...}`` — so "no logs yet"
    reads as a normal answer instead of a failure. Every other error still raises.

    ``tail`` is clamped to ``[1, 10000]`` before forwarding, so a negative value
    can't produce a malformed ``--tail -N`` and an absurd value can't make
    comfy-cli read back an enormous log slice. Each returned line is also capped
    to ``_MAX_LOG_LINE_CHARS`` so a single pathological line (a base64 blob or
    tensor dump from a buggy node) can't flood the caller's context.
    """
    tail = max(_MIN_LOG_TAIL, min(int(tail), _MAX_LOG_TAIL))
    try:
        data = _run_comfy("logs", "--tail", str(tail), timeout=60.0)
    except ComfyCliError as exc:
        if exc.code == _NO_LOG_FILE_CODE:
            return {"error": _NO_LOG_FILE_CODE, "message": str(exc)}
        raise
    if isinstance(data, dict) and isinstance(data.get("lines"), list):
        # Charge the truncation marker against the cap so a capped line's TOTAL
        # length (content + marker) never exceeds `_MAX_LOG_LINE_CHARS`.
        content_limit = _MAX_LOG_LINE_CHARS - len(_TRACEBACK_TRUNCATION_MARKER)
        data["lines"] = [_cap_text(line, content_limit) for line in data["lines"]]
    return data


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


# The compact per-row projection returned by the listing. The full detail
# (tags / models / providers / category_title) is what ``get_template(name)``
# returns — keeping the listing slim is what stops the full 558-row catalog from
# blowing the MCP client's tool-output cap.
_TEMPLATE_LIST_FIELDS = ("name", "title", "description", "output_type")

# Upper bound on a single page so an oversized `limit` can't build a response
# that trips the MCP client's tool-output cap; callers page the rest via `offset`.
_TEMPLATE_LIST_MAX_LIMIT = 200


def _template_matches(row: dict, query_lower: str) -> bool:
    """True if ``query_lower`` (already lowercased) matches a template ``row``.

    Case-insensitive substring match over the free-text fields ``name`` /
    ``title`` / ``description`` plus the string items inside the ``tags`` and
    ``models`` list values — deliberately NOT every string value, so a query
    like ``"image"`` does not hit ``output_type`` on hundreds of rows.
    """
    for key in ("name", "title", "description"):
        value = row.get(key)
        if isinstance(value, str) and query_lower in value.lower():
            return True
    for key in ("tags", "models"):
        for item in row.get(key) or []:
            if isinstance(item, str) and query_lower in item.lower():
                return True
    return False


@mcp.tool()
def search_templates(
    query: str = "",
    limit: int = 25,
    offset: int = 0,
    tag: str = "",
    type: str = "",
    model: str = "",
    provider: str = "",
    exclude_api: bool = False,
) -> Any:
    """Search the built-in ComfyUI workflow-template gallery.

    Wraps ``comfy templates ls``, whose payload is
    ``{total_in_gallery, matched, shown, filters, rows: [...]}`` — one ``row``
    per template with ``name / title / output_type / category_title / tags /
    models / providers / description``. The full catalog is ~558 rows, far too
    large to return whole, so this narrows and pages it:

    - ``query`` — free-text, case-insensitive substring match applied
      client-side over each row's ``name`` / ``title`` / ``description`` and the
      items in its ``tags`` / ``models`` lists (comfy-cli's ``ls`` has no
      free-text search flag, so this narrowing happens here).
    - ``tag`` / ``type`` / ``model`` / ``provider`` — forwarded to comfy-cli as
      ``--tag`` / ``--type`` / ``--model`` / ``--provider`` gallery filters
      (``--tag`` and ``--type`` are exact-match, ``--model`` / ``--provider``
      substring). Combine with ``query`` for free text on top.
    - ``exclude_api=True`` — drop rows carrying the ``API`` tag (templates that
      call a hosted API and need a key), approximating "runnable locally".
      comfy-cli's ``--tag`` only includes, so this negation is applied here.
    - ``limit`` (default 25, capped at 200) / ``offset`` — page the filtered rows.

    Returns ``{"total", "shown", "offset", "rows"}`` where ``total`` is the
    filtered match count, ``rows`` is the current page projected down to
    ``name / title / description / output_type`` (page again with ``offset`` to
    see more), and ``get_template(name)`` is the full-detail path.

    Step 1 of the template on-ramp: pick a ``name`` from the results, inspect it
    with ``get_template(name)``, then ``fetch_template(name, out_path)`` to write
    a runnable workflow JSON and pass that path straight to ``run_workflow`` — a
    working generation without hand-authoring workflow JSON.
    """
    if limit < 0:
        raise ComfyCliError(f"invalid limit: {limit} (must be >= 0)")
    limit = min(limit, _TEMPLATE_LIST_MAX_LIMIT)

    args = ["templates", "ls"]
    for flag, value in (
        ("--tag", tag),
        ("--type", type),
        ("--model", model),
        ("--provider", provider),
    ):
        if value:
            # comfy-cli parses a leading-dash value as an option/flag; reject it
            # rather than let `templates ls` misread the filter (argument
            # injection).
            _reject_option_like(f"{flag} value", value)
            args += [flag, value]
    data = _run_comfy(*args, timeout=60.0)

    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        shape = (
            "keys {" + ", ".join(sorted(map(str, data))) + "}"
            if isinstance(data, dict)
            else data.__class__.__name__
        )
        raise ComfyCliError(
            "unexpected `comfy templates ls` payload: expected a dict with a "
            f"`rows` list, got {shape}. comfy-cli's output shape may have drifted."
        )

    rows = data["rows"]
    bad = sum(1 for r in rows if not isinstance(r, dict))
    if bad:
        # Fail loudly on shape drift rather than silently dropping rows (which
        # would undercount `total`), matching the payload guard above.
        raise ComfyCliError(
            f"unexpected `comfy templates ls` payload: {bad} of {len(rows)} rows "
            "are not objects. comfy-cli's output shape may have drifted."
        )
    if exclude_api:
        rows = [
            r
            for r in rows
            if not any(
                isinstance(t, str) and t.lower() == "api" for t in r.get("tags") or []
            )
        ]
    if query:
        q = query.lower()
        rows = [r for r in rows if _template_matches(r, q)]

    total = len(rows)
    offset = max(0, offset)
    page = rows[offset : offset + limit]
    projected = [{k: r.get(k) for k in _TEMPLATE_LIST_FIELDS} for r in page]
    return {
        "total": total,
        "shown": len(projected),
        "offset": offset,
        "rows": projected,
    }


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

    - ``query`` given → ``comfy models search --text <query>`` (match model
      filenames). ``--text`` is required: comfy-cli's ``search`` takes the query
      as an option, not a positional (a positional exits 2 with a usage error).
    - else ``folder`` given → ``comfy models list-folder <folder>`` (list one
      model folder, e.g. ``checkpoints``, ``loras``).
    - else (both empty) → ``comfy models list-folders`` (list the folder names).

    LOCAL DEGRADATION: unlike the cloud catalog, this returns only what is on
    disk — filenames, with no enrichment (no base-model / hash / description /
    download metadata). Agents should set expectations accordingly: it answers
    "which model files does this install have?", not "tell me about this model".
    """
    if query:
        return _run_comfy("models", "search", "--text", query, timeout=60.0)
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
    such search), so the caller must already have the direct model URL.

    ``comfy model download`` streams human progress text to stderr and exits 0
    WITHOUT emitting an ``envelope/1`` object, so on that clean-exit success this
    returns a synthesized payload — ``{"ok": True, "action": ..., "message":
    ..., "note": ...}`` whose ``message`` carries the CLI's printed text (the
    "Done in …" tail and saved-path line) — rather than envelope ``data``
    (BE-3345). If comfy-cli starts emitting an envelope for this verb, that real
    envelope wins and its ``data`` (the saved path / download metadata) is
    returned instead. A non-zero exit still raises :class:`ComfyCliError`.
    """
    # comfy-cli parses a leading-dash value as an option/flag; reject any so a
    # crafted argument can't be smuggled in as a CLI flag (argument injection).
    _reject_option_like("url", url)
    # Restrict to http(s): this is a remote fetch of a known model URL, so a
    # `file://` path or other scheme — an SSRF / local-file-read primitive whose
    # body would be written straight into the models dir — is never legitimate.
    if urlparse(url).scheme.lower() not in ("http", "https"):
        raise ComfyCliError(f"invalid url: {url!r} (scheme must be http/https)")
    # Optional args are treated as unset when falsy (None or ""), so an explicit
    # empty string is omitted rather than forwarded as `--relative-path ""`.
    if relative_path:
        _reject_option_like("relative_path", relative_path)
        # relative_path is a models-dir SUBFOLDER (e.g. `models/loras`); keep the
        # write inside the models dir by rejecting absolute paths and `..`.
        parts = relative_path.replace("\\", "/").split("/")
        if os.path.isabs(relative_path) or ".." in parts:
            raise ComfyCliError(
                f"invalid relative_path: {relative_path!r} (path traversal)"
            )
    if filename:
        _reject_option_like("filename", filename)
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
    # plain_ok=True: `comfy model download` exits 0 with human progress text and
    # no envelope, so treat a clean exit as success instead of raising the
    # "returned no JSON" false negative on a download that actually landed
    # (BE-3345). A real error envelope or a non-zero exit still raises.
    return _run_comfy(*args, timeout=1800.0, plain_ok=True)


@mcp.tool()
def upload_file(paths: list[str], overwrite: bool = False) -> Any:
    """Upload local files into the LOCAL ComfyUI ``input`` directory.

    Wraps ``comfy upload <files...> [--overwrite]``. Use this to stage source
    images/masks a workflow references by filename before running it — it is
    what unlocks img2img / inpaint workflows on a local ComfyUI. Pass
    ``overwrite=True`` to replace files that already exist in the input dir
    (otherwise comfy-cli skips or errors on collisions).
    """
    # Each path is splatted in as a positional, so a leading-dash entry is read
    # by comfy-cli as a flag instead — `paths=["--overwrite"]` would silently
    # become the overwrite flag rather than a (failing) upload.
    for p in paths:
        _reject_option_like(
            "upload path",
            p,
            expected="a file path (prefix a dash-leading name with './')",
        )
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

    Known blind spots (upstream comfy-cli, fixes in progress): a passing result
    does NOT currently guarantee the server will accept the workflow.

    1. Missing required inputs are not detected — a node lacking a required
       input (e.g. KSampler without ``seed``) validates clean, but the server
       rejects it with ``required_input_missing``.
    2. ``COMFY_DYNAMICCOMBO_V3`` inputs (e.g. ClaudeNode ``model``) are not
       checked — invalid selection keys, missing required dotted sub-inputs
       (``model.max_tokens``, …), and misspelled sub-keys all pass, yet the
       server rejects with ``required_input_missing``.
    3. Frontend/UI-export workflow files are not actually validated — wrapper
       keys produce benign ``non_node_key`` warnings, zero nodes are checked,
       and the result is vacuously valid. Ignore those ``non_node_key``
       warnings (do not "fix" the file); export API format (or rely on
       ``run_workflow``'s auto-conversion) if validation fidelity matters.

    Treat ``valid:true`` as necessary-not-sufficient and rely on
    ``run_workflow`` errors for final authority.
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
    # Each override is splatted in as a positional, so a leading-dash entry is
    # read by comfy-cli as a flag — e.g. `"--stdout"` would flip the
    # non-destructive/in-place behavior this tool's `stdout` argument owns.
    for o in overrides:
        _reject_option_like(
            "override",
            o,
            expected="an 'ADDR=VALUE' string (e.g. '6.text=a red bicycle')",
        )
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
    """Entry point: serve the MCP over stdio.

    A macOS protected-folder denial hit during startup (a config, log, or module
    the server itself reads from under ~/Documents, say) arrives as a bare
    :class:`PermissionError` that the MCP client would surface as a raw Python
    traceback. Translate it into the same actionable guidance the tool paths
    give, on stderr — where MCP clients collect server logs — and exit non-zero.
    Anything else propagates unchanged.

    One case is beyond reach on purpose: if THIS server's own interpreter cannot
    read its venv, CPython dies in ``init_import_site`` before any of our code
    runs. That failure is only catchable from the parent side, which is exactly
    what the ``comfy``-binary guards in :func:`_check_comfy_version` and
    :func:`_require_comfy_bin` do for the child process we spawn.
    """
    try:
        mcp.run()
    except PermissionError as exc:
        # Prefer the exception's structured `filename` over re-parsing its text:
        # it is the authoritative path, and it is present for errnos the text
        # signature alone would not claim (TCC can surface as EACCES too).
        path = getattr(exc, "filename", None) or _tcc_path_from(str(exc))
        if not _is_macos() or not (
            _looks_like_tcc_denial(str(exc)) or _macos_protected_dir(path) is not None
        ):
            raise
        print(
            f"comfy-local-mcp: {exc}\n\n{_tcc_guidance(path)}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
