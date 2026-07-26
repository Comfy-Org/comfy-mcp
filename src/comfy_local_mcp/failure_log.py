"""The opt-in local failure log.

Leaf module over :mod:`comfy_local_mcp.textutil`: it owns the log's configuration,
its module-level state (``_FAILURE_LOG_PATH`` and the lazily-opened rotating
handler) and the single ``_log_failure`` entry point ``server`` calls before
each raise. Nothing here imports ``server``.

Tests that need the log on (or off) must patch ``failure_log._FAILURE_LOG_PATH``
— the state lives here, so patching a name on ``server`` would have no effect.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from .textutil import _redact_url, _stream_tail

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
