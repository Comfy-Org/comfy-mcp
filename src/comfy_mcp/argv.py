"""comfy-cli argv, path, and input guards — every tool-facing string's last stop before ``subprocess``.

Leaf module over :mod:`comfy_mcp.errors`: it owns the shared argument-injection
and OS-limit defenses (``_reject_nul``, ``_reject_option_like``,
``_reject_nul_deep``, ``_guard_arg_len``, ``_encode_argv``,
``_bounded_timeout``, ``_clip_for_error``) plus every per-domain guard built on
top of them — ``_guard_workflow_path``, ``_guard_prompt_id``,
``_guard_download_id``, ``_guard_extra_args``, ``_guard_version``,
``_guard_node_names``, ``_guard_log_port`` / ``_render_bad_port``,
``_guard_model_relative_path`` / ``_guard_model_filename``, and
``_validate_upload_paths`` — and their supporting length/shape constants and
regexes. The common thread across all of them: a caller-supplied string that is
too long, embeds a NUL, or opens with a leading dash reaches either
``subprocess`` (a bare, unconverted exception) or comfy-cli (misread as an
option) instead of failing as the :class:`errors.ComfyCliError` every other bad
input produces here — these guards are what makes that failure clean. Nothing
here imports ``server``.

Everything here is private and reached as ``argv._name`` — there is no public
name in this module.
"""

from __future__ import annotations

import math
import ntpath
import os
import re
import sys
from typing import Any

from . import errors
from .errors import ComfyCliError


def _bounded_timeout(timeout_seconds: float, ceiling: float) -> float:
    """Bound a caller-supplied timeout to ``(0, ceiling]``, rejecting NaN.

    ``min(max(t, 0.0), ceiling)`` looks like it does this and does not: every
    NaN comparison is False, so ``max(nan, 0.0)`` returns ``nan``, ``min`` keeps
    it, and it reaches ``Popen.communicate(timeout=nan)`` — where the selector
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
    ``subprocess.Popen`` raises a bare ``ValueError: embedded null byte`` on one —
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
    """Reject a leading-dash value comfy-cli would parse as an option, not data.

    Two different jobs share this helper, and the distinction is worth keeping
    straight when adding a call site:

    - **A bare POSITIONAL is an argument-injection vector.** A leading-dash entry
      IS read as a flag, and every later positional shifts up one slot — how
      ``upload_file(paths=["--overwrite"])`` becomes the overwrite flag, and how a
      dash-leading ``workflow_path`` would let the first ``set-slot`` override land
      in the path slot. Guarding these is mandatory.
    - **An option VALUE (``--flag VALUE``) is NOT.** comfy-cli is Click-backed, and
      Click takes the token after a value-taking option verbatim — even one that is
      itself a valid option name (``--out-dir --slot`` parses as
      ``out_dir="--slot"``, not as a missing value). So ``vary_workflow``'s
      ``slots`` / ``out_dir`` are already injection-safe before any guard runs.

    Option values ARE guarded anyway, nearly everywhere in the module
    (``search_templates``'s filters, ``download_model``'s ``relative_path`` /
    ``filename``, ``fetch_template``'s ``out_path``, ``vary_workflow``'s ``slots``
    / ``out_dir``, ``run_workflow`` / ``validate_workflow``'s ``workflow_path``).
    That is input hygiene rather than injection defense: a dash-leading provider
    filter or output filename is a caller mistake, and a named error beats
    comfy-cli matching zero rows, writing a file named ``-x``, or printing
    ``--help`` text that then fails envelope parsing. Read those as
    belt-and-braces, not as evidence that option values are unsafe — the
    distinction above still decides which guards are *mandatory*.

    Two deliberate over-rejections, so neither reads as an oversight:

    - **A lone ``-``.** Click treats it as a positional, not an option, so it is
      not an injection vector. It is refused anyway because it is not a valid
      value at any call site — not a path, template name, node class, connection
      type, or slot address — and "wrote a file named ``-``" is precisely the
      caller mistake the hygiene guards exist to name.
    - **A dash-leading slot ADDR** (``vary_workflow``'s ``slots``). An ``ADDR``
      begins with the node id, which comfy-cli surfaces non-negative via
      ``list_workflow_slots``, so no reachable address starts with ``-``. It also
      costs no capability: ``set_workflow_slot``'s overrides, the structured
      forms' ``address`` (:func:`params._slot_address_arg`), and both param marshalers
      (:func:`params._validate_param_key`) already refuse one, so every other way to
      name a slot in this module rejects it too.

    And one deliberate NON-rejection, which is what "nearly" above is doing:

    - **``search_models``'s ``--text`` query.** The hygiene guards all rest on a
      leading dash never being real DATA for that value, and each leaves an escape
      hatch — ``./-x`` names the same file as ``-x``, which is why the messages
      above suggest it. Neither holds for a free-form substring match over model
      *filenames*: ``-fp16`` / ``-fp8`` / ``-turbo`` are ordinary filename
      substrings, so a dash-leading query matches real rows, and there is no other
      way to spell that substring through ``--text``. Verified against comfy-cli:
      ``models search --text -fp16`` reaches the server call exactly as
      ``--text fp16`` does, so guarding it would refuse a working search rather
      than name a mistake. ``_reject_nul`` still applies there — a NUL cannot ride
      in argv at all. Weigh a new option-value call site the same way before
      copying the hygiene guard into it.

    The echoed value is rendered through :func:`_clip_for_error`, so this message
    honors the module's :data:`errors._MAX_ERROR_FIELD_CHARS` bound like every other
    field that quotes caller input. It is the widest-reach echo in the module —
    over twenty call sites, most of them values with no length cap of their own —
    and a bare ``{value!r}`` would mirror one back whole, ``repr`` escaping
    included (a control character expands 4-to-10x, so even a capped 4096-char
    value can render as ~16 KB). Output is byte-identical to the old ``!r`` for
    anything whose rendered form already fits the bound, which is every real
    argument.
    """
    if value.startswith("-"):
        hint = f" — expected {expected}" if expected else ""
        raise ComfyCliError(
            f"invalid {label}: {_clip_for_error(value)} (leading '-'){hint}"
        )
    return value


# Generous ceiling on EVERY path-shaped argv value this module forwards, set the
# same way :data:`_MAX_DOWNLOAD_ID_LEN` is: PATH_MAX is 4096 on Linux and 1024 on
# macOS, so a value past this ceiling can only name a path the filesystem would
# refuse anyway (`ENAMETOOLONG`) — while still catching the oversized string
# before it reaches argv, where the OS rejects the exec with an `OSError`
# (`E2BIG`) no caller converts to a :class:`ComfyCliError`.
#
# The full set it covers, all of them through :func:`_guard_arg_len`:
# ``run_workflow`` / ``validate_workflow`` / the four other ``workflow_path``
# tools, ``download_model``'s ``relative_path`` and ``filename``,
# ``fetch_template``'s / ``emit_partner_workflow``'s / ``partner_generate``'s
# ``out_path``, ``preview_media``'s ``media_path`` / ``out_path``,
# ``fetch_outputs``'s and ``vary_workflow``'s ``out_dir``, each entry of
# ``upload_file``'s ``paths``, and ``search_models``' ``folder``.
# ``download_model``'s ``url`` is the one path-adjacent value with a ceiling of
# its own (:data:`_MAX_URL_LEN`), passed to the same helper as an explicit
# ``limit``. Keep this list in sync — the SCOPE note at :data:`_MAX_URL_LEN`
# points here for exactly the values that are covered.
#
# ``upload_file``'s ``paths`` is the one entry that needs MORE than this: it is
# splatted into many argv slots, so a per-entry ceiling leaves the sum unbounded
# and :data:`_MAX_UPLOAD_PATHS` / :data:`_MAX_UPLOAD_PATHS_TOTAL_BYTES` bound
# the list itself, the way :data:`_MAX_EXTRA_ARGS` does for ``extra_args``.
#
# It stays this GENEROUS on purpose. It is an argv-safety guard, NOT an attempt
# to close the residual forgery window in :func:`clitext._phrase_is_only_the_caller_s` —
# that discount reads a bounded 500-char message tail, so only a cap BELOW 500
# characters would close it, and that trade-off (refusing legitimately deep
# paths to buy protection from a caller's own crafted argument) is recorded and
# declined in that function's docstring. Do not reopen it by tightening this.
#
# Measured in CHARACTERS, like every id cap above it and unlike
# :data:`params._MAX_PRECHECKED_SLOT_BYTES`, which encodes first because it is sized
# against the kernel's per-argument limit DIRECTLY and has to land on that exact
# boundary. Nothing here does: a character cap is strictly conservative for the
# hazard it guards. Worst case is 4 bytes/character, so 4096 characters is at
# most 16 KiB on the wire — an order of magnitude under Linux's 128 KiB
# `MAX_ARG_STRLEN`. The byte figures the paragraph above cites (`PATH_MAX`,
# `ENAMETOOLONG`) are therefore ANCHORS for where a generous ceiling belongs,
# not boundaries this check claims to sit exactly on: a multibyte path between
# 4096 characters and 4096 bytes rides through and fails as comfy-cli's own
# `ENAMETOOLONG`, which is an ordinary engine error, not the unconverted
# `OSError` this cap exists to prevent.
#
# `PATH_MAX` is also the POSIX figure. Windows with long paths enabled (or a
# `\\?\` prefix) admits up to 32,767 characters, so on that host this cap can
# refuse a path the filesystem would have opened. Accepted: a path that deep is
# not a real caller's, and the alternative — a per-platform ceiling — would make
# the same tool argument legal on one host and not another for no gain, since
# the argv hazard is what is being guarded and it does not vary that way.
_MAX_PATH_ARG_LEN = 4096


def _guard_arg_len(label: str, value: str, limit: int | None = None) -> str:
    """Refuse a caller-supplied argv string too long to survive ``execve``.

    The one length check every path-shaped tool argument runs, so the family
    stays uniform the way :func:`_guard_prompt_id` / :func:`_guard_download_id`
    do for the id-shaped ones. See :data:`_MAX_PATH_ARG_LEN` for what the
    default ceiling buys and which values carry it; ``limit`` exists for
    ``download_model``'s ``url``, whose ceiling is :data:`_MAX_URL_LEN` rather
    than the path one.

    ``None`` rather than ``limit: int = _MAX_PATH_ARG_LEN`` because a default
    argument is bound once at DEFINITION time: the constant would be copied into
    the signature, and this repo's documented convention of patching the owning
    module (``monkeypatch.setattr(server, "_MAX_PATH_ARG_LEN", …)``) would then
    move the name while leaving every call reading the old 4096. Resolving it in
    the body keeps :data:`_MAX_PATH_ARG_LEN` the single source of truth.

    Reports the LENGTH, never the value — echoing a megabyte-long "path" back is
    the same denial-of-legibility the cap exists to prevent, and every other
    guard at these sites names the value instead of its size, so an oversized
    one would otherwise come back described as the wrong mistake. That is also
    why callers run this FIRST, ahead of :func:`_reject_option_like` /
    :func:`_reject_nul`.

    Returns the value UNCHANGED when it fits, so it can wrap an assignment the
    way the other ``_guard_*`` helpers do.
    """
    if limit is None:
        limit = _MAX_PATH_ARG_LEN
    if len(value) > limit:
        raise ComfyCliError(
            f"invalid {label}: {len(value)} characters exceeds the "
            f"{limit}-character maximum."
        )
    # Length is not the only way a string fails to reach `execve`: one that
    # cannot be ENCODED never becomes an argv entry at all. Checked here so the
    # whole path-shaped family is covered in one place, and after the size check
    # so an oversized value is still named as a size.
    _encode_argv(label, value)
    return value


def _encode_argv(label: str, value: str) -> bytes:
    """Render a tool argument the way ``subprocess`` will, or refuse it.

    ``subprocess`` encodes POSIX argv with :func:`os.fsencode`, so this is both
    the CHECK that a value can become an argv entry and the only honest way to
    MEASURE one in the bytes the kernel counts. :func:`upload_file` uses it for
    the second — its aggregate is sized against ``ARG_MAX`` directly and cannot
    afford a proxy — and :func:`_guard_arg_len` for the first.

    Refusing matters because the alternative is not a bad answer but an
    unconverted one: an unencodable argument raises :class:`UnicodeEncodeError`
    inside ``Popen``, which sits OUTSIDE the ``try`` in :func:`_run_comfy_raw`,
    so it escapes as an internal error rather than the :class:`ComfyCliError`
    every other bad input produces. Exactly the reason :func:`_reject_nul`
    exists for the bare ``ValueError`` on an embedded NUL.

    POSIX only, and deliberately not papered over. On Windows ``subprocess``
    builds a UTF-16 command line for ``CreateProcessW`` and never calls
    :func:`os.fsencode` at all, and that platform's filesystem error handler is
    ``surrogatepass`` — so nothing here raises and the byte count is a UTF-8
    estimate of a limit Windows does not express in bytes. The guard is a no-op
    there rather than wrong, and the Windows command-line limit is out of scope
    for the same reason :data:`_MAX_UPLOAD_PATHS_TOTAL_BYTES` says it is.
    """
    try:
        return os.fsencode(value)
    except UnicodeEncodeError:
        # NOT diagnosed as "unpaired surrogate": `os.fsencode` uses the
        # interpreter's filesystem encoding, inherited from the environment this
        # server was launched in, so under a non-UTF-8 locale an ordinary
        # multibyte filename raises the same error. Name the encoding and offer
        # the usual cause rather than asserting one.
        raise ComfyCliError(
            f"invalid {label}: cannot be encoded with this system's filesystem "
            f"encoding ({sys.getfilesystemencoding()}), so it cannot be passed to "
            "comfy-cli as a command-line argument — usually an unpaired "
            "surrogate, or a character the current locale cannot represent."
        ) from None


def _guard_workflow_path(workflow_path: str, *, frontend: bool = False) -> str:
    """Run the shared argv guards on a ``workflow_path`` tool argument.

    ``frontend=True`` selects the error wording for the tools that require the
    frontend-format (UI export) workflow ``fetch_template`` writes;
    ``frontend=False`` for the tools that accept API-format or UI-export files
    (``run_workflow`` / ``validate_workflow``). The two wordings track that
    real format difference — do not merge them. WHY each site guards (bare
    positional = mandatory injection defence vs ``--workflow`` option value =
    input hygiene, see :func:`_reject_option_like`) stays in the per-site
    comments; this helper only owns WHAT runs.

    LENGTH runs first, ahead of both the wording branch and the value checks —
    see :data:`_MAX_PATH_ARG_LEN` for what the cap buys, and the ordering
    comment below for why it is first.
    """
    # BEFORE the two value guards: a dash-leading oversized path would otherwise
    # be named as a SHAPE mistake, and `_reject_option_like`'s echo — bounded by
    # `_clip_for_error`, but still a 500-character preview of a value whose only
    # problem is its size — tells the caller nothing the length does. One
    # wording for both `frontend` variants: a size error has nothing
    # format-specific to say.
    _guard_arg_len("workflow_path", workflow_path)
    if frontend:
        expected = (
            "a path to a frontend-format workflow JSON file "
            "(prefix a dash-leading name with './')"
        )
    else:
        expected = (
            "a path to a workflow JSON file (prefix a dash-leading name with './')"
        )
    _reject_option_like("workflow_path", workflow_path, expected=expected)
    _reject_nul("workflow_path", workflow_path)
    return workflow_path


# Generous ceiling on a `prompt_id`'s length. Real ids are the server's UUIDs
# (36 chars), and comfy-cli's own sanity check for one is `^[A-Za-z0-9_-]{1,128}$`
# — so this deliberately sits at twice the engine's ceiling and can only refuse
# input the engine would refuse anyway. What it buys: an oversized string reaches
# argv, where the OS (not comfy-cli) rejects the exec with an `OSError` no caller
# converts to a :class:`ComfyCliError`, and gets echoed back whole in the error.
_MAX_PROMPT_ID_LEN = 256


def _guard_prompt_id(prompt_id: str) -> str:
    """Reject a ``prompt_id`` comfy-cli would mis-read or ``subprocess`` can't carry.

    Shared by the six tools that take one, so the family stays uniform. Every
    ``jobs`` verb (and ``download``) takes the id as a bare positional — argv is
    a list and there is no shell, so the real hazard is not injection but
    *parsing*: a leading dash reaches comfy-cli as an option rather than an id,
    most sharply in ``fetch_outputs`` where it sits beside that command's own
    ``-o`` / ``--url-only``. An embedded NUL is a legal JSON (and so MCP) string
    but makes ``subprocess.Popen`` raise a bare ``ValueError``, which would surface
    as an internal error instead of the :class:`ComfyCliError` every other bad
    input produces. An empty id can only ever be a caller mistake, and is
    refused like every other positional this module guards. A wildly oversized
    id is the same story as the NUL: it fails in the exec, as an ``OSError``
    nobody converts, rather than as a clean error — see
    :data:`_MAX_PROMPT_ID_LEN`.
    """
    if not prompt_id or prompt_id.startswith("-"):
        raise ComfyCliError(f"invalid prompt_id: {prompt_id!r} (empty or leading '-')")
    if len(prompt_id) > _MAX_PROMPT_ID_LEN:
        # Report the length, not the value: echoing a megabyte-long "id" back is
        # the same denial-of-legibility the cap exists to prevent.
        raise ComfyCliError(
            f"invalid prompt_id: {len(prompt_id)} characters exceeds the "
            f"{_MAX_PROMPT_ID_LEN}-character maximum."
        )
    return _reject_nul("prompt_id", prompt_id)


# Generous ceiling on a `download_id`'s length, set the same way
# :data:`_MAX_PROMPT_ID_LEN` is: comfy-cli mints one as 12 hex characters and
# refuses to resolve anything outside `^[A-Za-z0-9_-]{1,64}$` when it opens the
# state file, so twice that ceiling can only refuse input the engine would refuse
# anyway — while still catching the oversized string before it reaches argv,
# where the OS rejects the exec with an `OSError` no caller converts to a
# :class:`ComfyCliError`.
_MAX_DOWNLOAD_ID_LEN = 128


def _guard_download_id(download_id: str) -> str:
    """Reject a ``download_id`` comfy-cli would mis-read or ``subprocess`` can't carry.

    The :func:`_guard_prompt_id` treatment for the download family
    (``download(action="status"/"wait"/"cancel")``, and the id
    ``download_model`` polls with). Same three hazards, for the same reasons:
    every ``model download-*`` verb takes the id as a bare positional, so a
    leading dash reaches comfy-cli as an option rather than an id; an embedded
    NUL is a legal MCP string that makes ``subprocess.Popen`` raise a bare
    ``ValueError`` instead of the :class:`ComfyCliError` every other bad input
    produces; and an empty id can only be a caller mistake.

    Deliberately NOT a hex-shape match. comfy-cli's ids are ``uuid4().hex[:12]``
    today, but the state store resolves any ``[A-Za-z0-9_-]{1,64}`` id, so
    pinning the format here would refuse a perfectly valid future id — and the
    only thing this guard has to buy is that the value reaches the engine as an
    ARGUMENT. Whether it names a real download is comfy-cli's answer to give
    (``download_not_found``), not this wrapper's to guess.
    """
    # Type FIRST, the way `_guard_version` does it: every test below is a `str`
    # method, so a non-string would leave as a bare `AttributeError` rather than
    # the `ComfyCliError` this guard exists to produce. MCP-level validation
    # keeps that off the tool paths, but the guard's contract is total for
    # in-process callers.
    if not isinstance(download_id, str):
        raise ComfyCliError(
            f"invalid download_id: expected a string, got {type(download_id).__name__}."
        )
    # LENGTH first among the value checks, so the error reports a size instead of
    # echoing an oversized value back through the tool response and the failure
    # log — see `_guard_prompt_id`. Ordered ahead of the shape checks because
    # those interpolate `{download_id!r}` in full, so a multi-megabyte blank
    # would otherwise be mirrored back verbatim by the branch below.
    if len(download_id) > _MAX_DOWNLOAD_ID_LEN:
        raise ComfyCliError(
            f"invalid download_id: {len(download_id)} characters exceeds the "
            f"{_MAX_DOWNLOAD_ID_LEN}-character maximum."
        )
    # Both shape tests read the STRIPPED value, so they cannot disagree about
    # what the id is. `strip()` rather than falsiness because a whitespace-only
    # id is the same caller mistake as an empty one — comfy-cli's store resolves
    # `[A-Za-z0-9_-]{1,64}`, so a blank can never name a real download — and
    # applying it to the dash test too keeps the docstring's invariant true for
    # `" -x"`, which Click only defuses by accident (it keys option detection on
    # the first character). The ORIGINAL value is what gets forwarded and
    # returned: this guard refuses input, it does not rewrite it.
    stripped = download_id.strip()
    if not stripped or stripped.startswith("-"):
        raise ComfyCliError(
            f"invalid download_id: {download_id!r} (empty or leading '-')"
        )
    return _reject_nul("download_id", download_id)


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


# Ceilings on a lifecycle tool's `extra_args`, set the way `_MAX_PROMPT_ID_LEN`
# is: generous next to any real use, but finite, because these entries go
# straight into argv and the kernel's own limit (`ARG_MAX`, ~256KB-1MB depending
# on the platform) surfaces as `OSError: Argument list too long` from
# `subprocess.Popen` — an internal error that no caller converts and that
# `_guard_extra_args`' contract promises not to raise. Bounding here also keeps
# `_network_exposing_args`' per-token scan (a `split(",")` and an
# `ipaddress` parse per part, on the event loop, before the `to_thread` hop)
# proportional to a plausible command line rather than to whatever the caller
# sent. ComfyUI's own longest flag list is a handful of short tokens plus the
# occasional filesystem path, so 64 x 4096 is orders of magnitude of headroom.
_MAX_EXTRA_ARGS = 64
_MAX_EXTRA_ARG_LEN = 4096


def _guard_extra_args(extra_args: list[str] | None) -> list[str]:
    """Validate a lifecycle tool's ``extra_args`` and return it as a plain list.

    Three jobs, all about the fact that these entries go STRAIGHT into argv:

    - **Shape.** ``None`` means "no extras". Anything that is not a list/tuple of
      strings is a caller mistake worth naming: a bare ``str`` would be splatted
      one CHARACTER per argv slot (``"--cpu"`` -> ``-``, ``-``, ``c``, …), and a
      non-string entry dies inside ``subprocess`` with a ``TypeError`` this
      module's error contract never promised.
    - **NUL.** :func:`_reject_nul`, for the reason that function documents:
      ``subprocess.Popen`` raises a bare ``ValueError: embedded null byte``, which
      would escape as an internal error instead of a :class:`ComfyCliError`.
    - **Size.** :data:`_MAX_EXTRA_ARGS` entries of :data:`_MAX_EXTRA_ARG_LEN`
      characters each, for the reason those constants document: past the kernel's
      ``ARG_MAX`` the failure is an unconverted ``OSError`` rather than a
      :class:`ComfyCliError`.

    Deliberately NOT rejected: a leading dash. These args are *meant* to be
    ComfyUI flags — that is the whole point of the ``--`` separator — so
    :func:`_reject_option_like` would refuse the intended input. Which of those
    flags need the user's consent is :func:`_network_exposing_args`' question,
    not this one's.
    """
    if extra_args is None:
        return []
    if isinstance(extra_args, str) or not isinstance(extra_args, (list, tuple)):
        raise ComfyCliError(
            "invalid extra_args: expected a list of strings, got "
            f"{type(extra_args).__name__}."
        )
    if len(extra_args) > _MAX_EXTRA_ARGS:
        raise ComfyCliError(
            f"invalid extra_args: {len(extra_args)} entries exceeds the "
            f"{_MAX_EXTRA_ARGS}-entry maximum (they are forwarded to ComfyUI as "
            "command-line arguments)."
        )
    guarded: list[str] = []
    for index, arg in enumerate(extra_args):
        if not isinstance(arg, str):
            raise ComfyCliError(
                f"invalid extra_args[{index}]: expected a string, got "
                f"{type(arg).__name__}."
            )
        if len(arg) > _MAX_EXTRA_ARG_LEN:
            raise ComfyCliError(
                f"invalid extra_args[{index}]: {len(arg)} characters exceeds the "
                f"{_MAX_EXTRA_ARG_LEN}-character maximum for one command-line "
                "argument."
            )
        guarded.append(_reject_nul(f"extra_args[{index}]", arg))
    return guarded


# The two moving targets `comfy update comfy --version` accepts alongside a
# pinned release. Matched case-insensitively and forwarded lowercased, the way
# `update_comfyui` normalizes its own target.
_VERSION_ALIASES = ("nightly", "latest")


# A pinned ComfyUI release: `MAJOR.MINOR.PATCH`, optionally `v`-prefixed (both
# `0.24.0` and `v0.24.0` name the same tag), with semver's optional prerelease
# AND build suffixes — both, in that order, because that is what comfy-cli's own
# validator accepts (it strips a leading `v` and hands the rest to
# `semver.VersionInfo.parse`). Deliberately no stricter than the engine: a
# pattern that refused a version comfy-cli would have taken would be this wrapper
# inventing a limitation rather than mirroring one.
#
# Anchored end-to-end, so a value that matches carries no whitespace, no NUL, no
# shell metacharacter, and no leading dash — which is why the interpolations
# below can quote it directly and why `_reject_nul` is not repeated here. It is a
# client-side SANITY check, not an authority on which tags exist: comfy-cli (and
# git) own that, and a well-formed version with no such release still fails
# there, with the engine's own message.
#
# Digits are `[0-9]`, NOT `\d`: on a `str` pattern `\d` is Unicode-aware and
# matches fullwidth or Arabic-Indic digits (`１.２.３`, `٠.٢٤.٠`), which would
# raise the destructive prompt and be forwarded verbatim to comfy-cli and git
# while this comment claimed the match was ASCII-safe. The `v` prefix is accepted
# in either case for the same reason the aliases are matched case-insensitively;
# `_guard_version` normalizes `V` down, because comfy-cli strips only a lowercase
# one.
_SEMVER_RE = re.compile(
    r"^[vV]?[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


# Generous ceiling on `version`, set the way :data:`_MAX_PROMPT_ID_LEN` is: the
# regex above would reject an oversized value anyway, but the length is checked
# FIRST so the error reports a size instead of echoing a megabyte-long "version"
# back at the caller.
_MAX_VERSION_LEN = 64


def _guard_version(version: str) -> str:
    """Validate a ``switch_comfyui_version`` target and return it normalized.

    Two jobs, in the order their error messages want to fire:

    - **Argument injection.** ``version`` is forwarded as the value of
      ``--version``, and while Click reads the token after a value-taking option
      verbatim (see :func:`_reject_option_like`), a dash-leading value is a caller
      mistake worth naming rather than a flag worth forwarding — the same
      treatment :func:`_guard_prompt_id` gives its positional.
    - **Format.** ``nightly`` / ``latest`` / a semver tag is the whole accepted
      set, checked here so a typo (``0.24``, ``main``, ``HEAD``) is refused
      BEFORE a subprocess is spawned that would stash the user's working tree
      and move HEAD before discovering the same thing.
    """
    if not isinstance(version, str):
        raise ComfyCliError(
            f"invalid version: expected a string, got {type(version).__name__}."
        )
    stripped = version.strip()
    if not stripped:
        raise ComfyCliError("invalid version: empty.")
    if len(stripped) > _MAX_VERSION_LEN:
        # Report the length, not the value — see `_MAX_VERSION_LEN`.
        raise ComfyCliError(
            f"invalid version: {len(stripped)} characters exceeds the "
            f"{_MAX_VERSION_LEN}-character maximum."
        )
    _reject_option_like(
        "version",
        stripped,
        expected="'nightly', 'latest', or a release like '0.24.0'",
    )
    lowered = stripped.lower()
    if lowered in _VERSION_ALIASES:
        return lowered
    if _SEMVER_RE.match(stripped):
        # comfy-cli's `validate_version` strips a leading `v` with a
        # case-SENSITIVE `startswith("v")` and hands the rest to
        # `semver.VersionInfo.parse`, so `V0.24.0` would die in the engine. The
        # aliases above are already case-insensitive and the refusal text
        # promises a leading `v` works, so normalize rather than refuse. Only the
        # prefix is lowered: semver prerelease/build identifiers are
        # case-sensitive and must reach the engine as written.
        return f"v{stripped[1:]}" if stripped.startswith("V") else stripped
    raise ComfyCliError(
        f"invalid version: {stripped!r} — expected 'nightly', 'latest', or a "
        "ComfyUI release like '0.24.0' (a leading 'v' is accepted). Nothing was "
        "changed."
    )


# How many packs one call may install. Not a comfy-cli limit — the engine takes
# any number — but the consent prompt has to be READABLE for the approval to mean
# anything, and a prompt echoing hundreds of ids is one the user scrolls past
# rather than reads. Refusing past this point keeps "the user approved these
# packs" true rather than nominal.
_MAX_NODE_PACK_NAMES = 32


# The same readability bound applied to the JOINED prompt text, because neither cap
# on its own reaches it: `_MAX_NODE_PACK_NAMES` bounds the COUNT and
# `_MAX_NODE_PACK_ID_LEN` bounds each ENTRY, but nothing bounded their product, and
# 32 ids of 128 characters is a 4,698-character prompt — the exact "scrolls past
# rather than reads" failure the count cap exists to prevent, reached by padding
# instead of by counting. 1024 characters clears any realistic batch (registry
# slugs run ~15-30 characters, so even a full 32-pack call of real ids lands near
# 900) while refusing the padded one.
#
# REFUSED rather than elided, which is where this parts company with
# `_display_extra_args`: eliding a long argument list still shows the user the
# flags that make it dangerous, so prompting beats refusing to prompt. Eliding a
# PACK LIST would do the opposite — it would collect an approval for names the
# prompt never showed, which is how a batch padded with look-alike slugs hides the
# one pack the user would have declined. The enumeration IS the consent here, so an
# over-long list never reaches the prompt at all.
_MAX_NODE_PACK_NAMES_CHARS = 1024


# A registry node id: an alphanumeric-led slug. Matched with `re.fullmatch` (see
# `_guard_node_names`), so a value that matches carries no whitespace, no NUL, no
# shell metacharacter, no leading dash, and — the load-bearing exclusion — no `/`,
# `:` or `@`, which is what keeps a git URL or a filesystem path off this argv.
#
# `fullmatch` rather than a `^…$` + `.match()` pair, because that pair does NOT
# carry the claim above: Python's `$` also matches just before a trailing newline,
# so `.match("pack\n")` succeeded. Nothing reached argv with a newline in it even
# then — `_guard_node_names` strips each entry first, and a full sweep of the code
# points that survive `str.strip()` found no other character this pattern would
# admit — but the invariant was being carried by that `strip()` while this comment
# credited the anchors. `fullmatch` puts it where the comment says it is.
#
# This is DELIBERATELY narrower than the engine, which is normally this repo's
# anti-pattern (see `_VERSION_RE`, which refuses to out-strict comfy-cli). The
# justification is the consent contract, not taste: the prompt below tells the
# user they are approving a download of a NAMED PACK FROM THE REGISTRY. If an
# arbitrary git URL could ride through here, that sentence would be false for the
# one call where it mattered most, and the approval it collected would have been
# given for something other than what happened. A wrapper may not narrow the
# engine to invent product behavior; it must narrow the engine where a wider
# input would make its own consent prompt lie.
_REGISTRY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _guard_node_names(names: Any) -> list[str]:
    """Validate ``install_node``'s pack list, or raise before anything is spawned.

    Ordered so the message a caller gets names the most useful problem: the list
    shape first, then each entry's LENGTH, then ``all``, then the id shape, and
    last the length of the whole JOINED list
    (:data:`_MAX_NODE_PACK_NAMES_CHARS`).

    **The length check must stay AHEAD of the id-shape check**, which is the one
    ordering here that is load-bearing rather than cosmetic. The shape refusal
    echoes the value so the caller can see what was wrong with it; the length
    refusal reports a SIZE and never echoes, so a megabyte-long "pack name" does
    not come back through the tool response and the failure log (see
    :data:`_MAX_NODE_PACK_ID_LEN`). Reversed, an oversized *invalid* value would
    take the echoing branch. The echo is clipped by :func:`_clip_for_error` as a
    second line of defense, but the bound stays where it is: a 4096-character
    preview of a caller's junk is not what the message is for.

    ``all`` is rejected here with its own message even though comfy-cli refuses it
    too (```install all` is not allowed``): the engine's refusal arrives as plain
    text on a non-zero exit, and a caller who meant "update everything I have"
    should be pointed at ``update_comfyui(target="all")`` rather than left to
    reverse-engineer that from a usage error.
    """
    if not isinstance(names, list) or not names:
        raise ComfyCliError(
            "invalid names: expected a non-empty list of registry pack ids "
            "(e.g. ['comfyui-impact-pack'])."
        )
    if len(names) > _MAX_NODE_PACK_NAMES:
        raise ComfyCliError(
            f"invalid names: {len(names)} packs exceeds the "
            f"{_MAX_NODE_PACK_NAMES}-pack maximum for one call. The confirmation "
            "prompt has to stay readable for the approval to mean anything; "
            "install them in smaller batches."
        )
    guarded: list[str] = []
    for entry in names:
        if not isinstance(entry, str) or not entry.strip():
            raise ComfyCliError(
                "invalid names: every entry must be a non-empty registry pack id "
                "(e.g. 'comfyui-impact-pack')."
            )
        value = entry.strip()
        # BEFORE the id-shape check below, which echoes the value — see the
        # docstring. Report the length, not the value: `_MAX_NODE_PACK_ID_LEN`.
        if len(value) > _MAX_NODE_PACK_ID_LEN:
            raise ComfyCliError(
                f"invalid names: {len(value)} characters exceeds the "
                f"{_MAX_NODE_PACK_ID_LEN}-character maximum for a pack id."
            )
        if value.lower() == "all":
            raise ComfyCliError(
                "invalid names: 'all' is not an installable pack id. To update "
                "every pack you ALREADY have, call "
                'update_comfyui(target="all") instead — this tool only installs '
                "packs you name."
            )
        _reject_option_like(
            "names", value, expected="a registry pack id (e.g. 'comfyui-impact-pack')"
        )
        _reject_nul("names", value)
        if not _REGISTRY_ID_RE.fullmatch(value):
            raise ComfyCliError(
                f"invalid names: {_clip_for_error(value)} is not a registry pack "
                "id. Expected a slug like 'comfyui-impact-pack' (letters, digits, "
                "'.', '_', '-'). A git URL, a filesystem path, or a "
                "'<pack>@<version>' pin is deliberately refused: the confirmation "
                "prompt tells the user they are approving a named pack from the "
                "registry, so it must not be able to carry anything else. To "
                "install from a URL, or to pin a specific pack version, run "
                "`comfy node install` in a terminal, where your shell is the "
                "confirmation."
            )
        guarded.append(value)
    # Last, because it is a property of the whole list rather than of any entry —
    # and the one bound that makes the prompt's enumeration readable rather than
    # merely finite. See `_MAX_NODE_PACK_NAMES_CHARS` for why this refuses instead
    # of truncating the display.
    joined = len(", ".join(guarded))
    if joined > _MAX_NODE_PACK_NAMES_CHARS:
        raise ComfyCliError(
            f"invalid names: {len(guarded)} pack ids join to {joined} characters, "
            f"past the {_MAX_NODE_PACK_NAMES_CHARS}-character maximum for one "
            "call. The confirmation prompt has to stay readable for the approval "
            "to mean anything, and it names every pack it installs; install them "
            "in smaller batches."
        )
    return guarded


# Bounds for get_logs' optional `port` hint — the IANA port range, the same one
# `_comfy_target` enforces on `COMFYUI_PORT`. The lower bound is what keeps an
# option-like value off argv: a negative port renders as `--port -8188`, which
# Click reads as another option rather than as this one's value.
_MIN_LOG_PORT = 1
_MAX_LOG_PORT = 65535


# How much of a rejected `port` the error message may quote back. Same reason
# `_guard_prompt_id` reports a length instead of the value: an in-process caller
# can pass a megabyte-long "port", and echoing it whole is the denial-of-legibility
# the guard exists to prevent.
_MAX_PORT_REPR_CHARS = 80


def _render_bad_port(port: Any) -> str:
    """Render a rejected ``port`` for an error message, bounded and total.

    Two hazards, both of which would otherwise escape :func:`_guard_log_port` as
    something other than the :class:`ComfyCliError` it exists to raise. An
    oversized value (a long string or list from an in-process caller) floods the
    caller's context, so it is truncated with its length reported — the shape
    :func:`_guard_prompt_id` and :func:`_guard_download_id` use. And an int with
    more than ``sys.get_int_max_str_digits()`` digits (4300 by default on 3.11+)
    raises ``ValueError`` on conversion to text, so ``port=10**5000`` would
    surface as an unhandled internal error instead of "out of range"; that case
    is caught and described by type rather than by value.
    """
    try:
        text = repr(port)
    except ValueError:
        # Narrow on purpose: the only in-practice raiser is the int-digit limit
        # above. A `__repr__` that fails some other way is a caller bug worth
        # seeing whole, not swallowing here.
        return f"<{type(port).__name__} too large to render>"
    if len(text) > _MAX_PORT_REPR_CHARS:
        return f"{text[:_MAX_PORT_REPR_CHARS]}… ({len(text)} characters)"
    return text


def _guard_log_port(port: Any) -> int:
    """Reject a ``get_logs`` ``port`` that has no business reaching argv.

    The numeric sibling of :func:`_guard_prompt_id`, and narrow for the same
    reason: the value is stringified straight into ``--port <n>``, so the hazard
    is parsing, not injection. A negative port renders as ``--port -1`` and is
    read by Click as an option; an out-of-range or non-integer one can only ever
    be a caller mistake, and refusing it here means the mistake is named instead
    of arriving as comfy-cli's usage dump. ``bool`` is excluded explicitly
    because it is an ``int`` subclass in Python, so ``port=True`` would otherwise
    forward ``--port 1``.

    Returns ``int(port)`` rather than the caller's object for the same
    argv-shape reason: an ``IntEnum``/``IntFlag`` member passes both the
    ``isinstance`` and the range check, but stringifies as ``Color.RED``, which
    would reach comfy-cli as ``--port Color.RED``. Normalizing here is what makes
    the call site's "forward the guarded int" true.

    Whether the port names a ComfyUI that ever ran is comfy-cli's answer to give
    (it reports the candidates it checked), not this wrapper's to guess.
    """
    if isinstance(port, bool) or not isinstance(port, int):
        raise ComfyCliError(
            f"invalid port: {_render_bad_port(port)} (expected an integer between "
            f"{_MIN_LOG_PORT} and {_MAX_LOG_PORT})."
        )
    if not (_MIN_LOG_PORT <= port <= _MAX_LOG_PORT):
        raise ComfyCliError(
            f"invalid port: {_render_bad_port(port)} is outside the valid range "
            f"{_MIN_LOG_PORT}-{_MAX_LOG_PORT}."
        )
    return int(port)


# Ceiling on `node_dependencies`' two id-shaped arguments, set the way
# :data:`_MAX_DOWNLOAD_ID_LEN` is. A pack id is a registry slug — tens of
# characters — and an oversized one is not a lookup that misses but a value that
# never reaches comfy-cli at all: past the platform's `ARG_MAX`, `execve` fails
# with an `OSError` no caller here converts to a `ComfyCliError`. Checked FIRST,
# ahead of `_reject_option_like`, so the error reports a size rather than echoing
# a megabyte-long "pack name" back through the tool response and the failure log.
_MAX_NODE_PACK_ID_LEN = 128


# Generous ceiling on `download_model`'s `url`, set the same way
# :data:`_MAX_PATH_ARG_LEN` is. 8 KiB sits past the request-line limit
# common HTTP servers enforce (nginx's `large_client_header_buffers` and
# Apache's `LimitRequestLine` both default to 8 KiB), so a URL past this ceiling
# can only be one the remote server would refuse anyway — while real
# HuggingFace / CivitAI URLs, signed and query-tokened forms included, sit far
# below it. What it buys is the same thing the id caps buy: the oversized string
# fails as a clean :class:`ComfyCliError` instead of reaching argv, where the OS
# rejects the exec with an `OSError` no caller converts.
#
# Characters, not bytes, for the reason :data:`_MAX_PATH_ARG_LEN` gives:
# 8192 characters is at most 32 KiB on the wire, still well under the 128 KiB
# per-argument argv limit, so the character count is the conservative measure
# and the 8 KiB request-line figure is an anchor rather than a boundary this
# check claims to sit on. (A URL is percent-encoded ASCII on the wire anyway, so
# the two counts coincide for anything a server would actually accept.)
#
# SCOPE: this and `_MAX_PATH_ARG_LEN` cover the PATH-SHAPED values, which are
# enumerated at `_MAX_PATH_ARG_LEN` rather than described, so the list can be
# checked. `search_models`' `folder` is on it too — a models-subfolder positional
# the sweep's own ticket did not name, added because it is the same shape and the
# same hazard as `relative_path`.
#
# What remains deliberately UNCAPPED is the FREE-FORM half: `search_models`'
# `--text` query, the enumerated `search_templates` / `nodes` filter
# values, the params `_generate_param_args` renders, and `vary_workflow`'s /
# `set_workflow_slot`'s JSON slot. Those are prompt- and query-shaped rather
# than path-shaped, so a path-sized ceiling would be the wrong instrument — it
# would refuse a legitimately long prompt to buy a bound the argv hazard does
# not, on its own, justify putting there.
#
# Be precise about what that leaves open, because the obvious defense of it is
# WRONG: "a caller can spend the same bytes by sending more, smaller values" is
# true of the TOTAL `ARG_MAX` and false of Linux's PER-ARGUMENT `MAX_ARG_STRLEN`
# (128 KiB), which one oversized string trips on its own and which no number of
# smaller ones can reach. So a >128 KiB query, filter value, param or slot does
# still die as the unconverted `OSError` — the residue of this sweep, not
# something it covers. Leaving it is a scope call (these are the values a
# path-sized ceiling would misjudge, and the right bound for them is a separate
# decision), NOT a claim that they are safe. `_MAX_PRECHECKED_SLOT_BYTES` is not
# that bound either: it gates only whether `_reject_non_json_array_slot` bothers
# to PARSE, ABSTAINING above it, and the slot still reaches argv. The
# identifier-shaped names (`fetch_template`'s `name`, a node class name, a slot
# `address`) sit in the same residue for the same reason.
#
# SIZE is not the only axis, and the residue is the same shape on the other one:
# a string that cannot be ENCODED never becomes an argv entry either.
# `_guard_arg_len` runs `_encode_argv` for every value listed at
# `_MAX_PATH_ARG_LEN`, so the path-shaped family is covered on BOTH axes; the
# free-form and identifier-shaped values above are uncovered on both. Converting
# the spawn failure itself is what would close the residue on either axis at
# once, and that is deliberately a separate change from this sweep.
_MAX_URL_LEN = 8192


def _guard_model_relative_path(relative_path: str) -> str:
    """Reject a ``relative_path`` that would write outside the models tree.

    ``download_model``'s destination guard, lifted out of the tool so the
    policy sits under a name like the module's other ``_guard_*`` helpers.
    Three refusals, in the order they run: a value that can climb OUT of the
    workspace (an absolute or root-relative path, a Windows drive prefix, a
    ``..``-shaped dot-run component), a value that stays in the workspace but
    lands outside the models tree anyway (``custom_nodes/pwn`` +
    ``__init__.py`` is code ComfyUI executes on its next start), and the
    backslash separator spelling that those checks read as ``/`` while the
    forwarded value would not. That order is load-bearing, and the moved
    comments below say why — along with the guard's KNOWN LIMITATION, that it
    is a lexical check on the string and so cannot see a symlink already
    planted inside the models tree.

    Raises :class:`ComfyCliError` on any of the three; otherwise returns the
    value UNCHANGED — this guard never rewrites an untrusted argument, for the
    reason the bare-``loras`` comment gives.

    LENGTH runs first, ahead of all three — see :data:`_MAX_PATH_ARG_LEN`,
    whose ceiling this reuses: both are path-shaped values headed for argv, and
    a NAME_MAX-tight cap of its own would be tighter than the argv hazard needs.
    """
    # First, for the reason `_guard_arg_len` gives: every check below names the
    # value instead of its size, so an oversized one would come back described
    # as the wrong mistake — with a 500-character preview that says nothing.
    _guard_arg_len("relative_path", relative_path)
    _reject_option_like("relative_path", relative_path)
    _reject_nul("relative_path", relative_path)
    # relative_path is a models-dir SUBFOLDER (e.g. `models/loras`), enforced
    # in two stages. FIRST, below: keep the value inside the WORKSPACE by
    # rejecting absolute paths, `..`, and a Windows drive prefix (`C:evil`
    # has no separator but is drive-relative on that platform, same escape as
    # the bare `filename` case below). That is a containment check, not a
    # destination check — it says where the value cannot go, not where it
    # must land — so a SECOND stage after it confines the write to the models
    # tree itself (and refuses the `\` spelling that would let the forwarded
    # value miss that tree anyway); see those checks for why both are needed.
    #
    # Decide this the same way on every host rather than deferring to
    # `os.path.isabs`: the guard runs wherever the MCP server runs, but the
    # write happens wherever comfy-cli runs, so a Windows-shaped escape has
    # to be refused from a POSIX server too (`os.path` is `posixpath` there
    # and sees `\\server\share` as an ordinary directory name).
    #
    # An empty leading segment means the value opened with `/` or `\` — an
    # absolute POSIX path, a UNC root, or a Windows root-relative path like
    # `\evil` (which lands at the root of the *current drive*, outside the
    # models dir). Testing the split segment rather than `ntpath.isabs` also
    # keeps the answer stable across Python versions: 3.13 stopped reporting
    # single-separator paths as absolute, so `ntpath.isabs("\\evil")` is True
    # on 3.10 and False on 3.14 — and CI runs both.
    #
    # `splitdrive` then covers what no separator reveals: a drive-relative
    # `C:evil` resolves against drive C:'s own working directory.
    #
    # Segments are matched as a DOT RUN rather than by `seg == ".."`, because
    # Windows strips trailing spaces and periods from every path component at
    # syscall time (`normpath` does not — it leaves them in place, which is
    # exactly why the string check has to). So `".. "` and `"..."` reach the
    # filesystem as `..` and traverse out of the models dir while comparing
    # unequal to it. `strip(" .")` leaves something behind for any real name
    # — `".hidden"`, `"v1.5"`, `"loras"` — so only those disguised forms fall
    # out empty. Empty segments are skipped so a doubled or trailing slash
    # (`models//loras`, `models/loras/`) keeps working as before; a LEADING
    # empty segment is the separator case `parts[0] == ""` already catches.
    #
    # This deliberately sweeps in a bare `.` component too (`./models`), one
    # step wider than the escape strictly requires. Drawing the line exactly
    # where Windows' stripping lands would mean modelling how many trailing
    # dots survive per component — precisely the reasoning you do not want
    # load-bearing in a traversal guard — and a `.` segment carries no
    # meaning here that plain `models/loras` does not, so the caller loses
    # nothing but a spelling.
    parts = relative_path.replace("\\", "/").split("/")
    if (
        parts[0] == ""
        or ntpath.splitdrive(relative_path)[0] != ""
        or any(seg and not seg.strip(" .") for seg in parts)
        or ":" in relative_path
    ):
        raise ComfyCliError(
            f"invalid relative_path: {relative_path!r} (path traversal)"
        )
    # The checks above only prove the value cannot climb OUT of the
    # workspace — they never required it to land in the models tree.
    # comfy-cli joins `--relative-path` to the WORKSPACE ROOT, not to the
    # models dir (`local_filepath = get_workspace() / relative_path /
    # local_filename`, defaulting to `DEFAULT_COMFY_MODEL_PATH = "models"`),
    # which is why the documented shape is `models/loras` and not a bare
    # `loras`. So a value that is perfectly traversal-clean can still write
    # anywhere in the workspace: `custom_nodes/pwn` + `__init__.py` puts
    # attacker-controlled code on ComfyUI's import path, which it executes on
    # the next start. Require the first segment to be `models` so the write
    # lands where the tool says it does.
    #
    # Ordered AFTER the traversal checks so a traversal string still reports
    # as traversal, and safe to state as a plain segment comparison only
    # because those checks have already run: by here the value has no leading
    # separator, no drive prefix, and no dot-run component, so the first
    # non-empty segment really is the first directory the write enters.
    #
    # Empty segments are dropped for the same reason they are skipped above —
    # `models/loras/` and `models//loras` are ordinary spellings of a real
    # subfolder. Matched exactly, not case-folded: a case-insensitive match
    # would only ever LOOSEN a security guard, and the error text names the
    # shape to use, so the caller loses nothing but a spelling.
    #
    # A bare `loras` is REJECTED rather than normalized to `models/loras`.
    # Normalizing would have to rewrite an untrusted argument, and it would
    # quietly turn the sibling-dir names this check exists to refuse into
    # accepted-but-nonsense paths (`input` -> `models/input`) instead of an
    # error the caller can act on.
    #
    # KNOWN LIMITATION, stated rather than silently trusted: this is a
    # LEXICAL check on the string, so it cannot see a symlink or junction
    # that already exists inside the models tree. If `models/link` is already
    # a link to `custom_nodes`, then `models/link/pwn` passes here and the
    # write follows it out. Resolving the path for real is deliberately NOT
    # done: the guard runs wherever the MCP server runs while the write
    # happens wherever comfy-cli runs, so resolution would consult the wrong
    # filesystem (the same host-independence the traversal checks above are
    # built around). Planting that link already requires write access inside
    # the models tree, which is the thing this tool is allowed to grant.
    segs = [seg for seg in parts if seg]
    if not segs or segs[0] != "models":
        raise ComfyCliError(
            f"invalid relative_path: {relative_path!r} "
            "(must be the models dir or a subfolder of it, e.g. 'models/loras')"
        )
    # Every check above reads `\` as a separator (`parts` splits on it), but
    # the value is forwarded VERBATIM — so on a POSIX host the check and the
    # write disagree. `models\loras` validates as segments `models` + `loras`,
    # then pathlib treats the backslash as an ordinary character and writes to
    # a workspace-root directory literally NAMED `models\loras` — a sibling of
    # the models dir, outside the tree the check just claimed to enforce.
    # (Only the guard's invariant breaks, not containment: the literal name is
    # forced to start with `models`, and a `models\..\custom_nodes` escape is
    # already dead because the same `\`->`/` split turns it into a `..` the
    # traversal check rejects.)
    #
    # Refuse the separator rather than rewrite the argument, matching both the
    # bare-`loras` reasoning above and `filename` below, which rejects `\` too.
    # It costs no capability on any host: pathlib on Windows resolves
    # `models/loras` and `models\loras` to the same directory, so a Windows
    # caller loses a spelling, not a destination — and the message names the
    # spelling to use. Ordered LAST so the security-meaningful diagnoses win:
    # `models\..\evil` still reports as traversal and `custom_nodes\pwn` still
    # reports as outside the models tree.
    if "\\" in relative_path:
        raise ComfyCliError(
            f"invalid relative_path: {relative_path!r} "
            "(use '/' as the path separator, e.g. 'models/loras')"
        )
    return relative_path


def _guard_model_filename(filename: str) -> str:
    """Reject a ``filename`` that is a path rather than a bare output name.

    The :func:`_guard_model_relative_path` treatment for the value that names
    the file itself: separators in either spelling, a Windows drive prefix, and
    the ``.``/``..`` dot runs — each of which would redirect the write out of
    the directory ``relative_path`` was just made to pin down. Why a bare name
    needs no separate ``ntpath.splitdrive`` test, and why the dot case is
    matched as a run rather than compared to ``..``, are in the moved comment
    below.

    Raises :class:`ComfyCliError` when the value is anything but a bare
    filename; otherwise returns it UNCHANGED, like the guard above.

    LENGTH runs first here too, against the same :data:`_MAX_PATH_ARG_LEN`
    ceiling the guard above uses — deliberately not a NAME_MAX-tight cap of its
    own, which would be tighter than the argv hazard requires.
    """
    # Length before shape, reporting the size — see the guard above.
    _guard_arg_len("filename", filename)
    _reject_option_like("filename", filename)
    _reject_nul("filename", filename)
    # filename is a single output name, not a path; reject separators, `..`,
    # and `:` (a Windows drive prefix like `C:evil.dll` has no separator but
    # still escapes the models dir via `os.path.join` on that platform) so it
    # can't redirect the write out of the target directory. These already
    # subsume an `ntpath.splitdrive` test — every string it reports a drive
    # for is either `X:…` (caught by `:`) or a UNC `\\host\share…` (caught by
    # `\`) — so a bare name needs no separate drive check. The `.`/`..` case
    # is matched as a dot run for the same reason as `relative_path` above:
    # Windows strips a component's trailing spaces and periods at syscall
    # time, so `".. "` and `"..."` arrive as `..` yet compare unequal to it.
    if (
        not filename.strip(" .")
        or "/" in filename
        or "\\" in filename
        or ":" in filename
    ):
        raise ComfyCliError(f"invalid filename: {filename!r} (must be a bare filename)")
    return filename


# Ceilings on `upload_file`'s `paths` LIST, alongside the per-entry
# `_MAX_PATH_ARG_LEN` each path carries. The per-entry cap alone does not bound
# the argv this tool builds: `paths` is the one caller-supplied value here that
# is splatted into MANY argv slots, so a list of individually-legal paths still
# sums past the kernel's TOTAL `ARG_MAX` (256 KiB on macOS, typically 2 MiB on
# Linux) and dies as the `OSError` (`E2BIG`) `_run_comfy_raw` never converts —
# its `try` wraps `communicate()`, not the `Popen(...)` that raises. Bounding
# both the count and the size of a splatted list is the same thing
# `_guard_extra_args` does for `extra_args`, for the same reason.
#
# TWO of them, because either alone leaves the other open: the aggregate bounds
# the path BYTES, and the count bounds the per-entry overhead the aggregate
# cannot see, since `execve` charges a pointer and a terminator per entry (a
# list of a million EMPTY strings sums to zero bytes and still blows the limit).
# `_guard_extra_args` splits the same job differently — count plus PER-ENTRY
# length, no aggregate — so the two are the same pair of concerns rather than
# literally the same pair of caps; its own worst case is a residue of the kind
# the last paragraph describes.
#
# BYTES, not characters, and this is the one place in this module where that
# distinction bites. The per-value ceilings above count characters and say why:
# each sits so far under the limit it is guarding that a 4x UTF-8 expansion
# still cannot reach it. An AGGREGATE has no such headroom — it is sized against
# `ARG_MAX` directly, so 128 KiB of CHARACTERS would be up to 512 KiB of
# multibyte path on the wire and would sail past the very limit it exists to
# hold. Measured with `os.fsencode`, which is what `subprocess` itself uses to
# render argv, so the count is the kernel's rather than a near-enough proxy:
# `surrogatepass` would charge 3 bytes for each surrogate in the
# `surrogateescape` range that `os.fsencode` renders back as the SINGLE
# undecodable filename byte it came from, over-counting a path with non-UTF-8
# bytes in its name threefold and refusing a batch that fits.
#
# NOT a proof that the spawn fits, and the comment must not pretend otherwise.
# `execve` charges the inherited ENVIRONMENT against the same budget and
# `_comfy_env` forwards a copy of `os.environ`, which on a loaded shell is tens
# of KiB on its own; Windows does not work like this at all, capping the whole
# rendered command line at 32,767 UTF-16 code units. A ceiling that actually
# held under all of that would have to be computed per-spawn against the live
# environment, which is a different mechanism from a tool-argument guard. What
# these two DO buy is what the id caps buy: a runaway list is refused as a clean
# `ComfyCliError` naming the mistake, rather than reaching a spawn that fails
# with an unconverted `OSError`. Converting that `OSError` at the spawn site is
# the airtight backstop underneath them, and is tracked separately.
#
# The AGGREGATE is the one that binds in practice, and the count is the backstop
# under it rather than a second ceiling a caller meets. At a typical absolute
# path — `/home/user/Pictures/comfy-inputs/IMG_20260804_123456.png` is about 58
# bytes — 128 KiB is reached around 2,200 files, well before the 4096-entry cap,
# so that entry count is only reachable with short paths (a `/tmp/frames/NNNN.png`
# sequence is ~20 bytes and does fit). Read the count cap as what stops a list of
# a million tiny or empty strings, not as a promise that 4096 real files fit.
# Clearing these caps is not a promise the upload COMPLETES —
# `upload_file`'s 300s timeout is the binding constraint on a batch that large,
# and it is comfy-cli's transfer being timed, not this guard's business. A
# caller who does reach one is told to split the batch; the tool takes any
# number of files across several calls, so neither cap makes an upload
# impossible, only unbatched.
_MAX_UPLOAD_PATHS = 4096
_MAX_UPLOAD_PATHS_TOTAL_BYTES = 128 * 1024


def _validate_upload_paths(paths: Any) -> None:
    """Every list-level and per-entry guard for :func:`upload_file`.

    Synchronous ON PURPOSE: the tool is async, and these guards scan up to
    :data:`_MAX_UPLOAD_PATHS` entries of :data:`_MAX_PATH_ARG_LEN` characters
    each (including an ``os.fsencode`` per entry) before the first await —
    inline, that stalls every other in-flight call and progress notification on
    the event loop. :func:`upload_file` runs this through ``asyncio.to_thread``,
    the same offload `_parse_deps_manifest` gets for the same class of work.
    """
    # Each path is splatted in as a positional, so a leading-dash entry is read
    # by comfy-cli as a flag instead — `paths=["--overwrite"]` would silently
    # become the overwrite flag rather than a (failing) upload. NUL is orthogonal
    # to that (it is refused wherever it rides, positional or not): `subprocess`
    # raises a bare `ValueError` on one, which would escape as an internal error
    # instead of the `ComfyCliError` every other bad input produces.
    # SHAPE first, the way `_guard_extra_args` checks it: a bare `str` would pass
    # `len()` and then be splatted one CHARACTER per argv slot (`"a.png"` ->
    # `a`, `.`, `p`, …), and a non-string entry dies inside `subprocess` with a
    # `TypeError` this module's error contract never promised. This is the block
    # where list-level mistakes are named, so they are both named here.
    if isinstance(paths, str) or not isinstance(paths, (list, tuple)):
        raise ComfyCliError(
            f"invalid paths: expected a list of strings, got {type(paths).__name__}."
        )
    # The emptiest list-level mistake, and the one every cap below waves through:
    # `[]` builds `comfy upload` with no positionals, so the caller gets either a
    # success-shaped result for an upload that moved nothing or comfy-cli's raw
    # Click usage dump. Name it here like the rest.
    if not paths:
        raise ComfyCliError(
            "invalid paths: empty list — pass at least one file to upload "
            "(e.g. ['/tmp/source.png'])."
        )
    # COUNT next, ahead of the per-entry loop: it is the whole-list mistake, and
    # checking it first is what keeps the loop below proportional to a plausible
    # command line rather than to whatever the caller sent. See
    # `_MAX_UPLOAD_PATHS`.
    if len(paths) > _MAX_UPLOAD_PATHS:
        raise ComfyCliError(
            f"invalid paths: {len(paths)} entries exceeds the "
            f"{_MAX_UPLOAD_PATHS}-entry maximum (they are forwarded to comfy-cli "
            "as command-line arguments) — upload in batches."
        )
    # Every per-entry failure names the INDEX the way `_reject_non_json_array_slot`
    # names `slots[i]`: the list is splatted, so "which of the paths" is the first
    # thing a caller with several needs to know, and that is as true of a
    # dash-leading or NUL-bearing entry as of an oversized one. Size runs ahead
    # of the two value guards for the ordering reason `_guard_arg_len` gives.
    total = 0
    for index, p in enumerate(paths):
        label = f"upload path paths[{index}]"
        if not isinstance(p, str):
            raise ComfyCliError(
                f"invalid {label}: expected a string, got {type(p).__name__}."
            )
        _guard_arg_len(label, p)
        _reject_option_like(
            label,
            p,
            expected="a file path (prefix a dash-leading name with './')",
        )
        _reject_nul(label, p)
        # Encoded exactly as `subprocess` will encode it, so the running total is
        # the kernel's count rather than a proxy — see
        # `_MAX_UPLOAD_PATHS_TOTAL_BYTES` for why this aggregate cannot afford
        # one. The encodability REFUSAL already happened inside `_guard_arg_len`
        # above, which runs `_encode_argv` for every path-shaped value; this is
        # the same call for its byte count.
        total += len(_encode_argv(label, p))
    # AGGREGATE last, once every entry is known to be a legal path: reporting it
    # before the per-entry checks would name the sum for a list whose real
    # problem is one bad member. Reports the total, never the paths.
    if total > _MAX_UPLOAD_PATHS_TOTAL_BYTES:
        raise ComfyCliError(
            f"invalid paths: {len(paths)} entries totalling {total} bytes exceeds "
            f"the {_MAX_UPLOAD_PATHS_TOTAL_BYTES}-byte maximum for the whole list "
            "(the paths are forwarded to comfy-cli as command-line arguments) — "
            "upload in batches."
        )


def _clip_for_error(text: str) -> str:
    """Render a caller-supplied fragment for an error message, bounded.

    A slot's value list is caller-sized — a sweep over long prompts is KBs of
    text — and the whole point of naming the offending entry is lost if the
    message it rides in is unreadable. Same per-field cap the envelope errors use.
    :func:`_reject_option_like` renders through this too, which is what applies
    the bound to the module's widest echo: most of its call sites guard values
    with no length cap of their own.

    This quotes the fragment itself rather than leaving that to an ``!r`` at the
    call site, because the cap has to apply to what is actually RENDERED. ``repr``
    expands a control or non-printable character into a 4-to-10-character escape
    (``\\x00``, ``\\uXXXX``, ``\\U000XXXXX``), so clipping the raw text first and
    quoting after would let 500 such characters land as ~2000+ in the message —
    the bound would read as if it held while the field blew past it. Clipping the
    rendered form is what :func:`errors._render_error_details` does too.

    The source text is sliced BEFORE ``repr`` so an MB-sized value never
    materializes an expanded copy just to have it thrown away. That is safe for
    the characters shown: every source character contributes at least one
    character to the repr, so the escaping of the retained prefix cannot depend
    on anything past the cap. The one thing it does change is ``repr``'s choice
    of surrounding quote — it switches to double quotes for a string containing
    an apostrophe and no double quote, and that decision is now made over the
    slice, so an apostrophe past the cap flips it. Cosmetic, in a preview that
    is already truncated. The ellipsis is counted inside the cap, so the
    returned field never exceeds it.
    """
    rendered = repr(text[: errors._MAX_ERROR_FIELD_CHARS])
    if len(rendered) <= errors._MAX_ERROR_FIELD_CHARS:
        return rendered
    return rendered[: errors._MAX_ERROR_FIELD_CHARS - 1] + "…"
