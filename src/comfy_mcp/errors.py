"""comfy-cli failure types and error-message rendering.

Leaf module over :mod:`comfy_mcp.failure_log`: it owns the one exception every
comfy-cli-facing call raises (:class:`ComfyCliError`), the benign-"nothing
recorded to stop" detector ``restart_comfyui`` relies on, and the
``error.details`` renderer + per-field char cap that bound every error message
this server returns. Nothing here imports ``server``.

:class:`ComfyCliError` is this module's one PUBLIC name: ``server`` name-imports
it (``from .errors import ComfyCliError``) rather than reaching it
module-qualified, per the carve-out AGENTS.md documents for public exception
and model types — it rides hundreds of ``except ComfyCliError`` sites and
``isinstance`` checks, and name-importing keeps that text unchanged. Everything
else here is private and reached as ``errors._name`` — including
``_MAX_ERROR_FIELD_CHARS``, which staying code in ``server`` also reads, so
both sides must go through the module (``errors._MAX_ERROR_FIELD_CHARS`` at
call time), never an import-bound copy.
"""

from __future__ import annotations

import re
from typing import Any

from . import failure_log


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
    :func:`clitext._is_missing_verb_error` is exactly that caller.

    ``returncode`` is the child's exit status wherever :func:`_unwrap_envelope`
    knows it — on the no-envelope path AND on an error envelope — so it is
    genuinely independent of ``no_envelope`` rather than a proxy for it. It
    distinguishes *how* comfy-cli failed: a usage error the argument parser
    rejected before dispatch versus a failure partway through a command it did
    accept, which the message text alone cannot tell you. It stays ``None`` for
    the failures raised without ever reading a child's status (missing binary,
    timeout).

    ``timed_out`` marks the one failure that is not comfy-cli misbehaving but us
    running out of patience: the child's whole process group was killed at the
    ``timeout=`` we handed ``communicate``. A caller that *chose* that budget
    can then tell its own deadline firing from a genuine comfy-cli error without
    matching on the message — :func:`wait_for_job`, which caps each poll to the
    time left on the caller's bound, is exactly that caller.

    ``data`` is the failed envelope's own ``data`` payload, for the commands that
    carry a STRUCTURED result alongside a negative verdict: ``comfy validate``
    emits its full ``{valid, errors, warnings}`` report as ``data`` and sets the
    envelope's ``ok`` to ``valid``, so "the workflow does not fit this install"
    arrives here as an error whose payload is the actual answer. It is ``None``
    for every failure that has no payload, which is what lets a caller tell a
    real verdict from a check that could not run at all
    (:func:`_local_template_check`).
    """

    def __init__(
        self,
        *args: object,
        code: str | None = None,
        no_envelope: bool = False,
        returncode: int | None = None,
        timed_out: bool = False,
        data: Any = None,
    ) -> None:
        super().__init__(*args)
        self.code = code
        self.no_envelope = no_envelope
        self.returncode = returncode
        self.timed_out = timed_out
        self.data = data


# comfy-cli's error code for "I have no server pid recorded to stop" — the one
# stop failure ``restart_comfyui`` treats as benign (see its docstring).
_NO_RECORDED_SERVER_CODE = "no_recorded_server"

# The same marker as it appears rendered INSIDE a message (e.g. "comfy stop
# failed [no_recorded_server]: none"), for the failures that carry it as text
# without a structured ``code``. Word-bounded rather than a bare substring test
# so a longer, unrelated code that merely starts with it — ``no_recorded_server_pid``
# — is not read as this one. (``_`` is a word character, so ``\b`` does not fire
# between "server" and "_pid".)
_NO_RECORDED_SERVER_CODE_RE = re.compile(rf"\b{re.escape(_NO_RECORDED_SERVER_CODE)}\b")

# The SAME condition as comfy-cli actually prints it in the common case: `comfy
# stop` with no recorded background server prints "No ComfyUI is running in the
# background." and exits 1 WITHOUT an envelope (comfy-cli 1.12.0 `cmdline.stop`),
# so there is no structured ``code`` for the check above to see and the literal
# marker string never appears in the message either. That gap is what stopped
# ``restart_comfyui`` from recycling a server it did not background-launch — a
# foreground ``comfy launch``, the desktop app, a manual ``python main.py``, or
# nothing running at all — even though its docstring has always promised to
# swallow "nothing to stop".
#
# Matched with a case-insensitive REGEX on the stable part of the phrase rather
# than by equality against the exact sentence: this is comfy-cli's human output,
# free to drift in capitalization, punctuation, or an inserted word ("No ComfyUI
# *server* is running in the background"), and pinning the exact bytes is what
# made the original check brittle in the first place. It is deliberately still
# narrow — it requires BOTH halves, a negated ComfyUI subject AND "running in the
# background", within one short clause — so it identifies "nothing was recorded to
# stop" and nothing else. A permission error, a process that could not be killed,
# or any other comfy-cli malfunction still re-raises: none of them claim no
# ComfyUI is running.
#
# Two details keep "one short clause" honest, because the string it runs against
# is the wrapper's ``stderr: … | stdout: …`` rendering of BOTH streams, not one
# tidy sentence:
#
#   * The subject must OPEN a message, a line, or a field — start-of-string, a
#     newline, the ``:`` / ``|`` the wrapper delimits streams with, or the
#     ``...`` ``textutil._stream_tail`` prefixes a clipped capture with (a
#     truncation marker is where a field begins, not prose). comfy-cli prints
#     this sentence on its own; a hint buried mid-sentence in some other failure
#     ("…, and ensure no ComfyUI is running in the background") is advice, not a
#     report that nothing was recorded, and must not be swallowed.
#   * The two halves must be joined by the GRAMMAR of the sentence, not merely
#     sit near each other: at most two inserted words and an optional copula
#     ("No ComfyUI *server is* running…", "No ComfyUI running…"). Because that
#     gap is built from ``\s`` and ``\w`` only, it cannot cross ANY punctuation —
#     so the halves can never be stitched out of two different streams
#     (``… | stdout: …``), two different clauses ("No ComfyUI process could be
#     stopped; it is still running in the background"), or across a conjunction
#     or dash ("No ComfyUI process was stopped and remains running in the
#     background"). Every one of those reports a stop that FAILED — the exact
#     opposite case — and none of them survives this shape.
#
#     A character-class gap was tried first and is what this replaces: excluding
#     punctuation one mark at a time is whack-a-mole, whereas admitting only
#     word characters is closed by construction. Newlines still pass (``\s``
#     covers them), so a Rich soft-wrap inside the sentence still matches — a
#     wrap is not a clause break.
_NO_RECORDED_SERVER_TEXT_RE = re.compile(
    r"(?:\A|[\n|:]|\.\.\.)\s*no\s+comfyui\b"
    r"(?:\s+\w+){0,2}(?:\s+(?:is|was))?\s+running\s+in\s+the\s+background\b",
    re.IGNORECASE,
)


def _is_no_recorded_server(exc: ComfyCliError) -> bool:
    """True when ``exc`` is comfy-cli's benign 'nothing recorded to stop' error.

    Prefers the structured ``code`` and falls back to the message so it also
    recognizes the error when only the human-readable string carries it — either
    as the literal marker code, or as comfy-cli's own printed phrasing on the
    bare non-zero exit that emits no envelope (see
    :data:`_NO_RECORDED_SERVER_TEXT_RE`).

    The text fallback searches the whole rendered message rather than a single
    stream, and is deliberately NOT gated on ``exc.no_envelope``: the phrase
    itself is the signal, and which reporting path comfy-cli happens to use for
    it is exactly the detail that should not matter here. It genuinely varies —
    comfy-cli prints this one through Rich, i.e. on STDOUT, while the wrapper's
    no-envelope message renders stdout and stderr side by side, and an envelope
    that carries the sentence but omits ``error.code`` is the same benign case.

    BOTH text reads — the marker and the phrase — are gated on the two signals
    that outrank anything in the message, because reading text over them would
    let a real failure be swallowed:

    * ``exc.code`` set to something else. comfy-cli told us structurally what
      went wrong; text in the message does not overrule it (the
      ``code == _NO_RECORDED_SERVER_CODE`` branch above already took the benign
      case).
    * ``exc.timed_out``. We killed the stop at our own deadline, so whatever it
      printed before dying says nothing about whether a server is recorded — and
      a stop that never finished is precisely the case ``restart_comfyui`` must
      not relaunch over.

    The gate therefore sits ABOVE both reads rather than between them: a
    timed-out stop whose output happens to quote the marker is still a timeout.
    """
    if exc.code == _NO_RECORDED_SERVER_CODE:
        return True
    if exc.code is not None or exc.timed_out:
        return False
    message = str(exc)
    return (
        _NO_RECORDED_SERVER_CODE_RE.search(message) is not None
        or _NO_RECORDED_SERVER_TEXT_RE.search(message) is not None
    )


# Error-envelope ``error.details`` keys worth surfacing verbatim in the raised
# message. ``partner_nodes`` names the offending nodes on a partner-credential
# failure; keep the set small so a large envelope can't bloat the message.
_SURFACED_DETAIL_KEYS = ("partner_nodes",)

# Per-field cap for the rendered error message (mirrors the stderr cap) so a
# multi-KB `message`/`hint` or a huge `partner_nodes` array can't produce an
# unbounded error string in the MCP client / logs.
_MAX_ERROR_FIELD_CHARS = 500


def _render_error_details(details: Any) -> str | None:
    """Render the useful keys of an envelope's ``error.details`` for the message.

    Scrubbed before the cap like every other envelope-derived field
    :func:`_unwrap_envelope` renders — today ``_SURFACED_DETAIL_KEYS`` is only
    node names, so it masks nothing in practice, but this string lands in the
    same client-facing sentence as ``error.message``/``hint`` and a later key
    added to that tuple should not have to remember to redact separately.
    """
    if not isinstance(details, dict):
        return None
    parts: list[str] = []
    for key in _SURFACED_DETAIL_KEYS:
        value = details.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        rendered = failure_log._scrub_text(str(value))[:_MAX_ERROR_FIELD_CHARS]
        parts.append(f"{key}: {rendered}")
    return "; ".join(parts) if parts else None
