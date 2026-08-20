"""The opt-in local failure log.

Leaf module over :mod:`comfy_mcp.textutil`: it owns the log's configuration,
its module-level state (``_FAILURE_LOG_PATH`` and the lazily-opened rotating
handler), and the small failure-event publisher ``server`` calls before each
raise. The JSONL writer observes those events. Nothing here imports ``server``.

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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from .textutil import _redact_url, _stream_tail

# Mode bits for the log's directory and files. The JSONL trail holds comfy-cli
# argv and stdout/stderr tails, so on a shared host it must not land at the
# umask default (`0o777`/`0o666` minus the umask — typically group/world
# readable). Owner-only on both, applied at creation so an existing directory
# the operator chose (a custom path's parent may be `/tmp`) is never re-moded.
_FAILURE_LOG_DIR_MODE = 0o700
_FAILURE_LOG_FILE_MODE = 0o600

# `COMFY_MCP_DEBUG_LOG` turns on a rotating, local-only JSONL record of
# every comfy-cli failure this server surfaces, so a tester can zip up a durable
# diagnostic trail instead of scraping an MCP client's transcript after the fact.
# Failure-only (a successful call writes nothing), local-only (nothing is
# transmitted anywhere), and OFF by default — while disabled there are ZERO
# filesystem effects: no directory is created and no handler is ever opened.
#
#   unset / "" / "0"  -> disabled (the default)
#   "1"               -> enabled at the per-OS default path
#   anything else     -> enabled, and the value IS the log file path
_FAILURE_LOG_ENV = "COMFY_MCP_DEBUG_LOG"

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

# A dedicated logger with `propagate = False`. In stdio mode stdout is the MCP
# protocol transport, so inheriting an arbitrary root stdout handler could
# corrupt the session. Streamable HTTP does not need that protocol reservation, but
# keeps the same predictable policy: application/server logs go to stderr and
# this opt-in diagnostic trail goes only to its owner-only file.
_FAILURE_LOGGER_NAME = "comfy_mcp.failures"


@dataclass(frozen=True, slots=True)
class _FailureEvent:
    """One immutable runner failure delivered to diagnostic observers."""

    kind: str
    args: tuple[str, ...]
    exit_code: int | None = None
    error_code: str | None = None
    message: str = ""
    stdout: str | bytes | None = None
    stderr: str | bytes | None = None
    streaming: bool = False


def _default_failure_log_path() -> str:
    """Per-OS default path for the failure log. Creates nothing.

    Mirrors comfy-cli's own local-state convention (its ``constants.py``
    ``DEFAULT_CONFIG``) with a ``comfy-mcp`` leaf, hand-rolled rather than
    imported: comfy-cli is the *engine* this server shells out to, not a Python
    dependency of it (the MCP framework is), so there is nothing to import.

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
    return os.path.join(base, "comfy-mcp", "failures.jsonl")


def _resolve_failure_log_path(value: str | None) -> str | None:
    """The log path ``COMFY_MCP_DEBUG_LOG=<value>`` selects, else ``None``."""
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


def _scrub_url(url: str) -> str:
    """One URL, credential-masked, safe to record in the failure log.

    A URL gets two treatments:

    - :func:`_redact_url` masks ``user:pass@`` userinfo, exactly as it does for
      the config values the error messages echo;
    - the query string and fragment are dropped entirely. That mirrors
      comfy-cli's own ``tracking.py`` scrubber, which exists because a CivitAI
      model URL carries its credential as ``?token=…`` — a secret no amount of
      userinfo masking would catch.
    """
    scrubbed = _redact_url(url)
    # Cut at whichever of `?` / `#` comes first: a fragment can precede a
    # (meaningless, but present) query, and slicing on each in turn would leave
    # the earlier delimiter's contents behind.
    cut = min(
        (i for i in (scrubbed.find("?"), scrubbed.find("#")) if i != -1),
        default=-1,
    )
    return scrubbed if cut == -1 else scrubbed[:cut]


# A URL ANYWHERE in a recorded string — deliberately not anchored to the start.
# argv is not all bare URLs: `_run_template_param_args` / `_generate_param_args`
# emit combined-flag tokens (`--image_url=https://<user>:<pass>@host/x?token=…`,
# `--param=k={"https://…"}`) whose URL sits mid-token, so a start-anchored test
# would wave the credential straight through into `args`. Anchoring on the
# literal scheme still lets the engine skip ahead to a candidate rather than
# re-scan from every offset.
#
# The body is `\S`-equivalent but TEMPERED: a match ENDS where the next
# `https?://` begins, so two URLs glued together with no whitespace between them
# — a comma/semicolon-separated list inside one `--param=` token, a manifest
# value, a validator hint — are scrubbed INDEPENDENTLY. A plain `\S+` swallowed
# the pair as ONE match, and `_scrub_url_match` then masked only the first
# URL's userinfo while the second's went to the client and to disk verbatim.
# Tempering costs one bounded lookahead per character consumed and still
# consumes each character exactly once, so the match stays single-pass and
# linear — no backtracking risk on a multi-KB message.
#
# The tempering STOPS at the first `?`/`#`, after which the rest of the token is
# taken untempered: past that delimiter `_scrub_url` deletes everything anyway,
# so splitting there would REVIVE text the scrubber drops today. The case that
# bites is a query whose value is itself an absolute URL
# (`?next=https://h2/y&token=…`) — split there, the trailing `&token=…` lands in
# a second match with no `?` of its own to cut on, and a secret this scrubber
# masks today would start leaking. Over-redacting a second host out of a debug
# tail is the right side to err on; the comma-separated list above is
# unaffected, its delimiter not being `?`.
_URL_RE = re.compile(r"https?://(?:(?!https?://)[^\s?#])*(?:[?#]\S*)?", re.IGNORECASE)

# The first whitespace-delimited token of a clipped stream window — the one
# place a URL can appear with its `https://` already sliced off, so the one
# place `_URL_RE` above is structurally unable to help. See
# `_scrubbed_stream_tail`.
_LEADING_TOKEN_RE = re.compile(r"\S*")

# Closing punctuation, peeled off the END of a `_URL_RE` match before scrubbing
# and re-attached after. `_URL_RE` cannot tell a URL from the character that
# merely FOLLOWS it, and `_scrub_url` deletes everything from the first `?` — so
# without this, comfy-cli's `Invalid value for '--url': 'https://h/x?q=1'`
# comes back having lost its closing quote and the rest of the sentence glued
# to it, and `_render_error_details`' `", "`-joined entries merge into one
# token. These messages now go to the MCP CLIENT, not just to the log, so that
# rewrite is no longer something only a maintainer reading a JSONL trail sees.
# Re-attaching gives nothing back: none of these characters is credential
# material, and the peel is lossless for a URL with no query at all (they are
# scrubbed to themselves and then restored).
_URL_TRAILING_PUNCT = "'\"`,;:!.)]}>"


def _scrub_url_match(match: re.Match[str]) -> str:
    """Scrub one :data:`_URL_RE` hit, keeping the punctuation glued to its end."""
    token = match.group(0)
    kept = token.rstrip(_URL_TRAILING_PUNCT)
    # `_URL_RE` anchors on `https?://`, so `kept` can never strip to empty.
    return _scrub_url(kept) + token[len(kept) :]


def _scrub_text(text: str) -> str:
    """Apply :func:`_scrub_url` to every URL embedded anywhere in ``text``.

    Only URL-shaped substrings are touched; everything around them (a path, a
    subcommand, a flag name, the prose of a message — the point of keeping the
    field at all) is preserved byte-for-byte, down to the quote or comma that
    closes a URL the surrounding text quoted (:data:`_URL_TRAILING_PUNCT`).
    """
    return _URL_RE.sub(_scrub_url_match, text)


def _scrub_arg(arg: str) -> str:
    """A comfy-cli argv token, safe to record in the failure log."""
    return _scrub_text(arg)


def _scrub_message(message: str) -> str:
    """A recorded ``message`` / stream tail, safe to record in the failure log.

    Scrubbing ``args`` alone would be theatre: the error-envelope message this log
    records is built by :func:`_unwrap_envelope` as ``comfy <args> failed …``, so
    the RAW argv — including a signed ``?token=…`` model URL — is echoed right
    back into it. The captured stream tails are the same story from the other
    side: comfy-cli echoes the URL it is fetching to stderr. Those strings are
    already what the MCP client sees (unchanged here), but this log PERSISTS them
    to disk for a tester to zip up and share, so the same masking has to reach
    all three.
    """
    return _scrub_text(message)


def _scrubbed_stream_tail(stream: str | bytes | None, limit: int) -> str:
    """A bounded stream tail with the same URL masking ``message`` gets.

    Scrubbing has to happen BEFORE the final clip to ``limit``, not after: the
    tail keeps the END of a capture, so a URL straddling the cut would arrive
    here already shorn of its ``https://`` and slip past :data:`_URL_RE`
    entirely — leaving the ``<user>:<pass>@host`` remainder in the file. So bound
    generously first (``_stream_tail`` slices raw bytes before decoding, so the
    wider window is still cheap on a multi-MB capture), scrub the whole window,
    and only then clip.

    The generous bound is NOT on its own enough, which is why the head fragment
    is handled explicitly below rather than left for the re-clip to push out of
    range: scrubbing SHRINKS the window — ``_scrub_url`` deletes whole query
    strings and userinfo — so a URL-dense capture can come out of it already at
    or under ``limit``, take the early return, and be written with its
    scheme-shorn head still attached. The re-clip is safe by contrast:
    everything it can cut has been scrubbed already, so a URL it bisects has no
    userinfo or query left to leak.

    That head fragment is DROPPED, not masked. Masking it (running the token
    through ``_scrub_url`` as if it were a whole URL) covers a clip that landed
    before the ``?`` and no further: a window cut PAST it arrives as
    ``token=…&x=1`` — or, cut mid-value, as a bare suffix of the secret — with
    no delimiter left for ``_scrub_url`` to anchor on, so the query it exists to
    delete survives verbatim. A clipped leading token is a fragment of something
    we cannot identify, and the ``...`` marker already says so; the host it
    might have named is still legible in the same message, which renders the
    argv through :func:`_scrub_text`. The cost is one partial token of a debug
    tail, and a capture whose real first line happens to start with ``...``
    loses its first token the same way.

    Unless that token is the WHOLE window — a capture with no whitespace in its
    last ``limit * 2`` chars — in which case dropping would return a bare
    ``...`` and throw away the error the tail exists to keep (a single unbroken
    blob is exactly the shape a chatty child's last line takes). There it falls
    back to masking, accepting the narrower residual: a lone unbroken token that
    is a scheme-shorn URL fragment clipped past its ``?`` keeps that remnant.
    Losing the entire capture is the worse failure of the two, and every
    multi-token case — which is every real CLI stderr — takes the drop.
    """
    if limit <= 0:
        # Mirror `_stream_tail`'s own non-positive guard: `[-0:]` is the WHOLE
        # string, so the re-clip below would otherwise hand back the entire
        # (double-width) window — the opposite of a bound.
        return "<empty>"
    window = _stream_tail(stream, limit * 2)
    scrubbed = _scrub_message(window)
    if scrubbed.startswith("..."):
        # `_stream_tail` prefixes that marker onto a window it CLIPPED, and a
        # clipped window can begin part-way through a URL — past the `https://`
        # that `_URL_RE` anchors on, which is precisely why the scrub above
        # cannot see a credential sitting in that leading remainder. Drop the
        # token outright rather than trying to mask it: see the docstring for
        # why no anchor is left to mask it ON. Over-redacting one debug-log
        # token is the right side to err on — but only while something else
        # survives it, hence the `rest` test: a whitespace-free window is one
        # token, and dropping it would hand back a bare `...`.
        head = _LEADING_TOKEN_RE.match(scrubbed, 3).group(0)
        rest = scrubbed[3 + len(head) :]
        scrubbed = "..." + (rest if rest.strip() else _scrub_url(head) + rest)
    if len(scrubbed) <= limit:
        return scrubbed
    # Re-clipping loses the marker `_stream_tail` may have added, so re-add it:
    # this result is truncated either way.
    return "..." + scrubbed[-limit:]


class _OwnerOnlyRotatingFileHandler(RotatingFileHandler):
    """:class:`RotatingFileHandler` that keeps every file it opens owner-only.

    The stdlib handler opens through :func:`open`, so the log lands at
    ``0o666 & ~umask`` — group/world-readable under a typical umask, for a file
    holding comfy-cli argv and stderr tails. Rotation opens a fresh file too, so
    the mode has to be reapplied on every open rather than once at setup.
    """

    def _open(self):  # type: ignore[no-untyped-def]
        stream = super()._open()
        try:
            os.chmod(self.baseFilename, _FAILURE_LOG_FILE_MODE)
        except OSError:
            # Best-effort, like everything else here: a file we cannot chmod
            # (someone else's, or a filesystem with no mode bits) must still be
            # logged to rather than silently disabling the diagnostics.
            pass
        return stream


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
                # Cleared BEFORE the teardown and the (fallible) setup below, so
                # neither a `close()` that raises nor a handler that fails to
                # open can leave the global claiming a path nothing is actually
                # writing to — which, if `_FAILURE_LOG_PATH` were later
                # repointed BACK to it, would make this memo report a handler
                # that is no longer attached and silently drop every record.
                _failure_handler_path = None
                for existing in list(logger.handlers):
                    logger.removeHandler(existing)
                    try:
                        existing.close()
                    except Exception:  # teardown must reach every handler
                        # A handler that fails to close is already detached; it
                        # must not abort the teardown of the ones after it (two
                        # live handlers would duplicate every line). Reported on
                        # the MODULE logger rather than swallowed: that is a
                        # different logger from the non-propagating
                        # `comfy_mcp.failures` one being rebuilt here, so
                        # this cannot recurse into the handler that just failed.
                        logging.getLogger(__name__).debug(
                            "failure-log handler close failed", exc_info=True
                        )
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, mode=_FAILURE_LOG_DIR_MODE, exist_ok=True)
                handler = _OwnerOnlyRotatingFileHandler(
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


def _write_failure_event(event: _FailureEvent) -> None:
    """Observe one event and append its scrubbed JSONL record when opted in."""

    path = _FAILURE_LOG_PATH
    if path is None:
        return
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": event.kind,
        "args": [_scrub_arg(arg) for arg in event.args],
        "exit_code": event.exit_code,
        "error_code": event.error_code,
        # Scrub first, THEN cap. Capping first would cut a credential URL
        # before its `@`, leaving an unrecognized raw userinfo fragment.
        "message": _scrub_text(event.message)[:_FAILURE_LOG_MESSAGE_CHARS],
        "stdout_tail": _scrubbed_stream_tail(event.stdout, _FAILURE_LOG_TAIL_CHARS),
        "stderr_tail": _scrubbed_stream_tail(event.stderr, _FAILURE_LOG_TAIL_CHARS),
        "streaming": event.streaming,
    }
    _failure_logger(path).info(json.dumps(entry, ensure_ascii=False))


# One observer is enough: this is not a general event bus. Keeping it as a
# tuple makes the publisher testable and leaves room for a process embedding
# the server to observe failures without coupling the runners to file I/O.
_FAILURE_OBSERVERS: tuple[Callable[[_FailureEvent], None], ...] = (
    _write_failure_event,
)


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
    """Publish one runner failure to the registered best-effort observers.

    Called immediately before each raise, so every event corresponds to a
    failure a caller actually saw. The default observer writes JSONL only when
    ``COMFY_MCP_DEBUG_LOG`` selected a path; while disabled it has zero
    filesystem effects.

    Observer failures are isolated from one another and swallowed after a
    debug message. Diagnostics must never replace the real ``ComfyCliError``.
    """

    event = _FailureEvent(
        kind=kind,
        args=tuple(args),
        exit_code=exit_code,
        error_code=error_code,
        message=message,
        stdout=stdout,
        stderr=stderr,
        streaming=streaming,
    )
    for observer in _FAILURE_OBSERVERS:
        try:
            observer(event)
        except Exception:  # diagnostic observers must never mask the real error
            logging.getLogger(__name__).debug(
                "failure observer failed; the event was dropped",
                exc_info=True,
            )
