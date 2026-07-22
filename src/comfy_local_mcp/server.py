"""comfy-local-mcp — a thin MCP wrapper over comfy-cli.

Every tool shells out to the ``comfy`` command (comfy-cli), pinned to the LOCAL
target, asks for JSON, parses comfy-cli's versioned ``envelope/1`` result, and
returns its ``data``. There is deliberately no HTTP client and no code shared
with the Comfy Cloud MCP — comfy-cli is the engine.

Tools so far: the run -> get-output core loop plus job management
(``job_status`` / ``wait_for_job`` / ``watch_job`` / ``get_execution_error`` /
``cancel_job`` / ``get_queue``), the ``launch_comfyui`` / ``stop_comfyui`` /
``restart_comfyui`` lifecycle trio (``comfy launch --background`` /
``comfy stop`` / stop-then-launch) with ``get_logs`` (``comfy logs``) to read a
detached launch's captured output, and the
``discover`` / ``which`` introspection pair (``comfy discover`` /
``comfy which``) that lets an agent learn the CLI's own contract and selection.

Requires comfy-cli >= 1.12.0 (the ``comfy logs`` verb + the ``envelope/1``
contract): :func:`_run_comfy` guards this once, up front, with an actionable
upgrade error so a stale install fails clearly rather than cryptically.

NOTE: the exact ``comfy`` invocation + envelope shape still need a smoke test
against a real comfy-cli install and a running local ComfyUI.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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
- Start from a template: `search_templates(query=...)` to find one (free-text
  search, paged 25 at a time via `limit`/`offset`; narrow with `tag`/`type`/
  `model`/`provider`, or `exclude_api=True` for templates that run without a
  hosted-API key), `fetch_template` to save its workflow JSON, then `run_workflow`
  on that file. To change the prompt / seed
  / steps / model of a fetched template before running, inspect its tweakable slots
  with `list_workflow_slots` and edit them with `set_workflow_slot` (non-destructive
  by default) — the loop is `fetch_template` -> `set_workflow_slot` -> `run_workflow`.
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

Everything targets the LOCAL server only — there is no cloud access here.
"""

mcp = FastMCP("comfy-local-mcp", instructions=INSTRUCTIONS)

# Allow overriding the binary (e.g. a venv path) without touching code.
COMFY_BIN = os.environ.get("COMFY_BIN", "comfy")

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
    ``_run_comfy_streaming``) cannot drift. Injected keys are placed AFTER
    ``os.environ`` so they win over any inherited values:

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


class ComfyCliError(RuntimeError):
    """comfy-cli was missing, timed out, or returned an error envelope.

    ``code`` carries the envelope's structured ``error.code`` when the failure
    came from an error envelope (used to drive the bounded credential retry in
    ``run_workflow``); it is ``None`` for local failures the wrapper raises
    itself (missing binary, timeout, no-JSON output), so callers can branch on a
    specific code without string-matching the message — e.g. ``get_logs``
    swallows ``no_log_file`` but re-raises the rest.
    """

    def __init__(self, *args: object, code: str | None = None) -> None:
        super().__init__(*args)
        self.code = code


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
    except (OSError, subprocess.SubprocessError):
        # A transient spawn failure fails OPEN for THIS call but is NOT latched —
        # a later call re-checks rather than permanently disabling the guard.
        return
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
    if shutil.which(COMFY_BIN) is None:
        raise ComfyCliError(
            f"`{COMFY_BIN}` not found on PATH. Install comfy-cli "
            "(`pip install comfy-cli`) or set the COMFY_BIN env var."
        )
    _check_comfy_version()
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
        raise ComfyCliError(
            f"comfy-cli timed out after {timeout}s: {' '.join(cmd)}. "
            f"stderr tail: {_tail(exc.stderr) or '<empty>'}; "
            f"stdout tail: {_tail(exc.stdout) or '<empty>'}"
        ) from exc

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
    real_envelope = (
        envelope if envelope and envelope.get("type") == "envelope" else None
    )
    if plain_ok and real_envelope is None and returncode == 0:
        return _synthesize_plain_result(args, stdout, stderr)
    # Enforce the envelope contract on the normal path too: pass `real_envelope`
    # (not `envelope`) so a stray non-envelope JSON line — e.g. an incidental
    # `{"ok": true, "data": ...}` diagnostic — can't be mis-unwrapped as a valid
    # response for a non-`plain_ok` tool; it raises the "returned no JSON" error
    # like any other missing envelope. A real error envelope still has
    # `type==envelope`, so it flows through and raises with its code as usual.
    return _unwrap_envelope(real_envelope, args, returncode, stderr)


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
    envelope: dict | None, args: tuple[str, ...], returncode: int | None, stderr: str
) -> Any:
    """Unwrap comfy-cli's ``envelope/1`` result, raising on error/absence.

    Shared by the plain (`--json`) and streaming (`--json-stream`) paths so both
    have identical terminal behavior: return ``data`` on success, and raise a
    :class:`ComfyCliError` carrying the envelope's ``error.code`` on failure.

    Also the envelope-version assertion: if comfy-cli declares an envelope
    ``schema`` whose major differs from :data:`ENVELOPE_SCHEMA_MAJOR`, the whole
    result shape is presumed incompatible and we refuse it with a clear error
    (rather than silently misreading a differently-shaped ``data``). An envelope
    with no declared schema is assumed compatible.
    """
    if envelope is None:
        raise ComfyCliError(
            f"comfy-cli returned no JSON (exit {returncode}). "
            f"stderr: {stderr.strip()[:500]}"
        )
    # A declared schema must be a recognized ``envelope/<N>`` whose major matches.
    # Absent schema -> assume compatible (older comfy-cli); declared-but-unparseable
    # or a different major -> refuse loudly rather than fail open on a shape we
    # can't vouch for.
    schema = _envelope_schema(envelope)
    if schema is not None and _envelope_major(envelope) != ENVELOPE_SCHEMA_MAJOR:
        raise ComfyCliError(
            f"incompatible comfy-cli envelope schema {schema!r}: "
            f"this server speaks envelope/{ENVELOPE_SCHEMA_MAJOR}. "
            "Upgrade or pin comfy-cli to a version whose envelope contract matches."
        )
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
        message = err.get("message") or stderr.strip()[:_MAX_ERROR_FIELD_CHARS]
        parts = [
            f"comfy {' '.join(args)} failed "
            f"[{code or 'unknown'}]: "
            f"{str(message)[:_MAX_ERROR_FIELD_CHARS]}"
        ]
        hint = err.get("hint")
        if hint:
            parts.append(f"hint: {str(hint)[:_MAX_ERROR_FIELD_CHARS]}")
        detail_str = _render_error_details(err.get("details"))
        if detail_str:
            parts.append(detail_str)
        raise ComfyCliError("\n".join(parts), code=code)
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
    # `_check_comfy_version` runs a synchronous `comfy --version` (up to 30s on
    # the first call per process); offload it so the async event loop is never
    # blocked while it runs.
    await asyncio.to_thread(_check_comfy_version)
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
        return False, _unwrap_envelope(
            _last_json_object("".join(lines)), args, returncode, stderr
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
            stderr_tail = ""
            if stderr_future is not None:
                try:
                    stderr_tail = _tail(await asyncio.wait_for(stderr_future, 2.0))
                except (Exception, asyncio.CancelledError):
                    # Diagnostics are best-effort: never let gathering the tail
                    # mask the timeout itself. CancelledError is a BaseException
                    # (not caught by `except Exception`) and DOES fire here: the
                    # outer wait_for cancels _drain while it awaits stderr_future,
                    # which cancels the future too — so awaiting it below re-raises
                    # CancelledError. Swallow it so we still raise ComfyCliError.
                    stderr_tail = ""
            raise ComfyCliError(
                f"comfy-cli timed out after {timeout}s: {' '.join(cmd)}. "
                f"Progress so far: {tracker.snapshot()}. The run may still be "
                "going — check `job_status`, or for long generations submit "
                "with `wait=False` and poll `wait_for_job` / `watch_job`. "
                f"stderr tail: {stderr_tail or '<empty>'}; "
                # Slice to the last lines before joining so a chatty child's full
                # stdout history isn't copied just to keep the 500-char tail.
                f"stdout tail: {_tail(''.join(lines[-500:])) or '<empty>'}"
            ) from exc

        # Envelope path: the authoritative result is already in `lines`, read
        # within the deadline. Give the child a brief grace to exit on a SEPARATE
        # budget; the `finally` kills a still-live one.
        envelope = _last_json_object("".join(lines))
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
        return _unwrap_envelope(envelope, args, proc.returncode, stderr)
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


def _freshness_report() -> Any:
    """Best-effort installed-vs-latest report via ``comfy outdated``.

    Returns the ``comfy outdated`` payload (``core`` install status, one row per
    custom node ``packs`` entry, ``checked_at``) on success. On ANY
    :class:`ComfyCliError` — the verb missing on an older comfy-cli, the network
    lookup down, a timeout — returns ``{"error": "<reason>"}`` instead of
    raising, so the probe can never take ``server_info`` down with it.
    ``OSError`` is caught for the same reason: a spawn failure on this second
    subprocess (the env probe already succeeded) is still just the freshness
    probe failing, never grounds to fail ``server_info``.
    """
    try:
        return _run_comfy("outdated", timeout=15.0)
    except (ComfyCliError, OSError) as exc:
        return {"error": str(exc)}


@mcp.tool()
def server_info() -> Any:
    """Report the local ComfyUI / comfy-cli environment and verify compatibility.

    Wraps ``comfy env``. Returns whether a local ComfyUI server is running and
    its URL, plus the selected workspace and Python info. Call this first to
    confirm a local ComfyUI is up before running a workflow.

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
    culprit. The probe is best-effort: on a comfy-cli without the ``outdated``
    verb, a network failure, or a timeout, ``freshness`` degrades to
    ``{"error": "<reason>"}`` — ``server_info`` itself still succeeds.
    """
    envelope, _stdout, args, returncode, stderr = _run_comfy_raw("env", timeout=60.0)
    # _unwrap_envelope has already raised if envelope was None, so it is non-None here.
    data = _unwrap_envelope(envelope, args, returncode, stderr)
    compat = _check_comfy_cli_version()
    compat["envelope_schema"] = _envelope_schema(envelope)
    freshness = _freshness_report()
    if isinstance(data, dict):
        return {**data, "compatibility": compat, "freshness": freshness}
    return {"env": data, "compatibility": compat, "freshness": freshness}


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


@mcp.tool()
async def generate_image(
    prompt: str,
    checkpoint: str | None = None,
    wait: bool = True,
    timeout_seconds: float = 600.0,
    ctx: Context | None = None,
) -> Any:
    """Generate an image from a text prompt on the LOCAL ComfyUI — the fast on-ramp.

    Wraps ``comfy generate --prompt <prompt>``: a single call that turns a text
    prompt into an image, so an agent does not have to hand-assemble a workflow
    graph. comfy-cli owns the text->workflow injection (which node/slot the
    prompt fills, the default graph, checkpoint selection); this tool is a pure
    passthrough to that verb. Returns the same envelope shape as
    ``run_workflow`` (``prompt_id`` + outputs).

    Pass ``checkpoint`` to pick a specific checkpoint model (forwarded to
    ``comfy generate --checkpoint``); omit it to let comfy-cli choose a default.
    With ``wait=True`` (default) this waits until the generation finishes and
    streams live progress as MCP progress notifications (per-node execution +
    sampler step counts) so a long generation is not a silent block; with
    ``wait=False`` it submits and returns immediately with a ``prompt_id`` to
    poll via ``job_status``. ``timeout_seconds`` only bounds the ``wait=True``
    streaming path; the ``wait=False`` submit-and-return branch uses a fixed
    short timeout, so callers should not expect it to govern that case.

    This is the quickest path to an image. For full control — choosing a
    template, editing its graph, or running a hand-authored workflow — use the
    ``search_templates`` -> ``fetch_template`` -> ``run_workflow`` chain instead.

    Everything targets the LOCAL server (``--where local`` is injected by
    ``_run_comfy``), so there is no cloud reachability here.
    """
    # Pass the free-form text as ``--flag=value`` so a prompt (or checkpoint)
    # that begins with ``-`` is read as the value rather than mis-parsed by
    # comfy-cli as an option token. The leading-dash guards elsewhere reject
    # such input, but a prompt legitimately can start with ``-``, so we keep it
    # instead of rejecting it.
    checkpoint_args = [f"--checkpoint={checkpoint}"] if checkpoint else []
    if not wait:
        # Fire-and-return: no stream to follow, so keep the plain --json path.
        return _run_comfy(
            "generate", f"--prompt={prompt}", *checkpoint_args, timeout=60.0
        )
    return await _run_comfy_streaming(
        "generate",
        f"--prompt={prompt}",
        *checkpoint_args,
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
            if value.startswith("-"):
                # comfy-cli parses a leading-dash value as an option/flag; reject
                # it rather than let `templates ls` misread the filter (argument
                # injection).
                raise ComfyCliError(f"invalid {flag} value: {value!r} (leading '-')")
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
