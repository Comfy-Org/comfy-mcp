"""comfy-cli human-output parsing — the text channel for verbs with no envelope.

Leaf module over :mod:`comfy_mcp.failure_log` and :mod:`comfy_mcp.errors`: it
owns every parser that reads comfy-cli's PRINTED text rather than its
``envelope/1`` JSON, because a handful of verbs (``comfy generate``, ``comfy
node install``, the lifecycle verbs) emit human-readable output and exit 0
instead of a structured result. Nothing here imports ``server``.

Two of these — :func:`_extract_install_failures` and
:func:`_extract_saved_paths` — are the AGENTS.md-documented cm-cli contract:
the ONLY channel this wrapper has for ``comfy node install``'s real per-pack
verdict and ``comfy generate``'s resolved output paths, matched against
comfy-cli's/cm-cli's own printed sentences rather than guessed at. Everything
here is private and reached as ``clitext._name`` — there is no public name in
this module.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from . import failure_log
from .errors import ComfyCliError

# The header `comfy generate` prints above the files it wrote (comfy-cli's
# `command/generate/output.py::print_saved`), followed by one INDENTED line per
# saved path.
_SAVED_MARKER = "Saved:"

# rich's Console width when its output is not a terminal — which it never is
# here, since every spawn pipes stdout (`_run_comfy_raw`).
_RICH_DEFAULT_WIDTH = 80

# Ceiling on how many saved paths a synthesized result reports. A partner model
# returns a handful of assets; anything past this is a runaway. The overflow is
# DROPPED outright, and `message` (a 1000-char tail) is not a reliable second
# copy of it — which is why the ceiling is set far above any real run rather
# than tight enough to need one.
_MAX_SAVED_PATHS = 50

# Longest path this will report. `PATH_MAX` is 4096 on Linux (1024 on macOS), so
# anything past it cannot name a real file. Over-long entries are DROPPED rather
# than sliced: a truncated path is a *different*, plausible-looking path, and a
# caller that acts on it writes to — or reports — the wrong place. Absent is
# legible; wrong is not.
_MAX_SAVED_PATH_CHARS = 4096

# How much text past the `Saved:` header is parsed. `_MAX_SAVED_PATHS` paths of
# `_MAX_SAVED_PATH_CHARS` each, plus fold slack, fits inside this comfortably.
# The bound exists because `model download` streams megabytes of rich progress
# through this same `plain_ok` synthesis: `splitlines()` also splits on `\r`, so
# a redrawing progress bar would otherwise materialize millions of tiny strings
# and the reassembly's `+=` would run over them quadratically.
_MAX_SAVED_BLOCK_CHARS = 256_000


def _cell_len(text: str) -> int:
    """Terminal CELLS ``text`` occupies — the unit rich measures its folds in.

    rich folds on cell width (``rich.cells.cell_len`` / ``chop_cells``), not on
    code points, so a path carrying CJK or emoji reaches the console edge after
    FEWER characters than ``len`` reports. Measuring with ``len`` made every such
    fold look unfoldable, and the continuation was then dropped — leaving a
    silently truncated path in ``saved_paths``.

    rich carries a generated width table; this approximates it with the two rules
    that matter for a filename — East-Asian Wide/Fullwidth is two cells, a
    combining mark is zero — and short-circuits the ASCII case, which is every
    path on a normal install.
    """
    if text.isascii():
        return len(text)
    total = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        total += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return total


def _child_console_width() -> int:
    """Column count rich uses in a comfy-cli child, for un-folding its output.

    rich resolves a non-terminal Console's width from ``COLUMNS`` and falls back
    to 80. ``_comfy_env`` forwards ``os.environ`` wholesale, so the value read
    here is the value the child actually rendered at — this is the same lookup,
    not a guess about it. Read per call rather than latched: nothing stops the
    host from re-exporting ``COLUMNS`` mid-session.

    The ``isdigit()`` test is rich's own, and mirroring it is the whole point:
    ``int()`` alone also accepts ``" 120 "``, ``"+120"``, ``"1_20"`` and
    ``"120\\n"``, none of which rich honours — so for any of those the parent
    would measure folds against a width the child never rendered at, and
    mis-assemble or truncate the paths. Anything rich ignores falls back to
    rich's own default, exactly as rich does.
    """
    columns = os.environ.get("COLUMNS", "")
    if not columns.isdigit():
        return _RICH_DEFAULT_WIDTH
    try:
        width = int(columns)
    except ValueError:
        # `isdigit()` is true for characters `int()` rejects (superscripts such
        # as "²"). rich has the same gap and crashes; fall back instead.
        return _RICH_DEFAULT_WIDTH
    return width if width > 0 else _RICH_DEFAULT_WIDTH


def _extract_saved_paths(text: str) -> list[str]:
    """Resolved output paths from a ``Saved:`` block in comfy-cli's printed text.

    ``comfy generate`` prints where it put each asset as a ``Saved:`` header
    followed by one indented path per line — the only place the resolved path
    appears when the command emits no envelope. Callers had to scrape it out of
    the human-readable blob (or shell out to ``ls`` to confirm anything landed),
    which is not reliably possible: comfy-cli prints through rich, whose Console
    is a fixed width off a TTY, so a path longer than that is FOLDED across
    physical lines — mid-filename, with no indent on the continuation (observed:
    ``…/partner-jellyfish.p`` / ``ng``).

    Three signals reassemble it, all read off how that block is rendered rather
    than guessed at:

    - Every path line is INDENTED (``rprint(f"  {p}")``) and a continuation never
      is, so an indented line starts a new path and an unindented one may
      continue the previous.
    - rich breaks a line only when the next thing does not FIT, so an unindented
      line continues a path only when the line above it had no room for it —
      measured in CELLS (:func:`_cell_len`), at the width
      :func:`_child_console_width` resolves the same way rich does. Anything that
      would have fit ends the block, which is what keeps unrelated trailing
      output from being glued onto a real path.
    - rich breaks in two different ways and they rejoin differently. A FOLD
      chops mid-token and loses nothing, so the pieces concatenate directly; a
      WORD WRAP breaks at whitespace and eats one space, so the pieces rejoin
      with a space put back. They are told apart by whether the line above had
      room for even the first character of the continuation: if it did not, the
      break was a fold; if it had room for a character but not for the whole next
      word, it was a wrap.

    That third rule is what makes a path containing SPACES survive — the common
    macOS ``/Users/me/My Pictures/…`` shape, which rich wraps at the space into a
    first line SHORTER than the console width. Reading only exact-width lines as
    continuations dropped the rest of the block and returned the leading
    fragment (``/Users/me/My``) as if it were a resolved destination.

    A blank line and the end of the text also end the block; a later ``Saved:``
    header starts another. Paths are returned in printed order, and a path that
    is left UNFINISHED — the block ended while the last line was still exactly
    full-width, so more of it was coming — is dropped rather than reported as a
    prefix, on the same reasoning as ``_MAX_SAVED_PATH_CHARS``.

    What the rendered text still cannot express: a path that happens to fill the
    console width EXACTLY is indistinguishable from a folded one, so it is
    dropped by the unfinished rule above, and unrelated output printed
    immediately after a path — with no blank line, and long enough that it could
    not have fit on that line — is read as a continuation. comfy-cli prints
    nothing there today (``print_saved`` is the last thing on the success path).
    Both are why the raw ``message`` is kept alongside this field rather than
    replaced by it. The real fix is upstream — an envelope for ``comfy
    generate`` — at which point ``_run_comfy`` takes the envelope path and this
    synthesis is bypassed entirely.
    """
    # Bound the parse to the block itself BEFORE splitting: this runs on the full
    # uncapped stdout+stderr of every `plain_ok` verb, and `model download`'s is
    # a multi-megabyte progress stream. Cheap `find` first so the overwhelmingly
    # common "no block here" case never splits anything. See
    # `_MAX_SAVED_BLOCK_CHARS`.
    marker_at = text.find(_SAVED_MARKER)
    if marker_at < 0:
        return []
    # Back up to the start of the marker's own physical line so the block's first
    # line is intact; `\r` counts, because `splitlines` treats it as a break.
    line_start = max(text.rfind("\n", 0, marker_at), text.rfind("\r", 0, marker_at)) + 1
    width = _child_console_width()
    lines = text[line_start : line_start + _MAX_SAVED_BLOCK_CHARS].splitlines()
    paths: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != _SAVED_MARKER:
            index += 1
            continue
        index += 1
        previous_width = 0
        # Where THIS block's paths start, so a stray unindented first line can
        # never be appended to a path the PREVIOUS block left behind.
        block_start = len(paths)
        while index < len(lines) and lines[index].strip():
            line = lines[index]
            if line[:1].isspace():
                paths.append(line.strip())
            elif len(paths) > block_start:
                # Continuation, and which KIND of break produced it. `max(…, 1)`
                # because a zero-width leading character still occupies a slot
                # rich had to have room for.
                head = max(_cell_len(line[:1]), 1)
                if previous_width + head > width:
                    # No room for even one more character: rich chopped mid-token
                    # and nothing was lost between the pieces.
                    paths[-1] += line.strip()
                elif previous_width + 1 + _cell_len(line.split(maxsplit=1)[0]) > width:
                    # Room for a character but not for the next whole word: rich
                    # wrapped at the space, and the space is not in either line.
                    paths[-1] += " " + line.strip()
                else:
                    break
            else:
                break
            previous_width = _cell_len(line)
            index += 1
        if previous_width == width and len(paths) > block_start:
            # The block ended on a full-width line, so the path was still being
            # folded when the text ran out: what we hold is a prefix, not a
            # destination. Drop it rather than hand back a plausible wrong path.
            paths.pop()
    return [path for path in paths if len(path) <= _MAX_SAVED_PATH_CHARS]


# The subcommand path whose exit status is NOT its verdict. See
# `_extract_install_failures`; kept as a tuple so the check in
# `_synthesize_plain_result` is an equality rather than a pair of `startswith`es.
_NODE_INSTALL_ACTION = ("node", "install")

# Cheap pre-test before the copy that `_extract_install_failures` needs. `node
# install` streams every pack's `pip` output through this synthesis, so a run can
# carry megabytes; `str.find` scans that without allocating, and the fold below
# then only ever runs on output that could actually contain a failure. Lowercase
# and case-SENSITIVE, matching the pattern below EXACTLY — the two are the same
# cm-cli literal deliberately, so the gate can never hide a sentence the pattern
# would have matched. (That is also why the pattern is not `IGNORECASE`: a gate
# stricter than its own pattern is a silent miss, and the honest way to keep them
# in step is to make both key off the one literal cm-cli prints.) The word is
# mid-sentence in `An error occurred while installing`, whereas pip's frequent
# line is capital-I `Installing collected packages` — so the common no-failure
# run does not pay for the fold. A single word because it is the longest fragment
# rich cannot break: rich wraps at spaces, so any longer gate could be split
# across two lines by a narrow console and missed.
_NODE_INSTALL_FAILURE_GATE = "installing"

# cm-cli's own per-pack failure sentence, and the ONE signal this wrapper has for
# a `node install` that did not install what it was asked to. Two things about
# the shape it matches, both read off ComfyUI-Manager's `cm-cli.py` rather than
# guessed: the sentence is printed identically by BOTH failure branches of its
# `install_node` (the URL clone and the registry resolve), and it is printed
# BEFORE `exit_on_fail` is consulted at all — which is why it, not the exit
# status, is what this wrapper reads.
#
# `pack` is whatever cm-cli quoted, capped; `reason` is the sentence it prints on
# the next line (`res.msg`, e.g. ``Node 'x@unknown' not found in [default,
# remote]``). `reason` is a fixed-width WINDOW rather than a pattern that stops at
# rich's closing `[/bold red]` tag, and that is the whole difference between
# reporting the reason and losing the failure: the tag is only present when markup
# was not rendered, so a pattern that REQUIRED it would fail to match at all on a
# rendered run, and one that stopped at any `[` would truncate this very message
# at its `[<channel>, <mode>]` suffix. The tag is trimmed off the window below.
#
# Two things the window does NOT do, both load-bearing:
#
# * It never runs into the NEXT failure. The window is TEMPERED against the
#   sentence's own opening literal, because `finditer`'s scan resumes at the end
#   of the whole match: an untempered 200-character window over two consecutive
#   per-pack failures (the folded remainder of the first is well under 200
#   characters for typical ids) would swallow the second one, leaving that pack
#   inside `installed` with `restart_required: true` — the exact false success
#   this parse exists to remove. Tempering also keeps `_PACK_NOT_FOUND_RE` off a
#   NEIGHBOURING pack's `Node '…' not found in` line, which cm-cli only ever
#   prints as the `res.msg` following that pack's own failure sentence. The
#   optional `ERROR: ` in that lookahead is cm-cli's own prefix on the line, cut
#   so a reason does not end with the first crumb of the next failure.
# * It does not require the quoted id to be CAPTURABLE. The opening quote is
#   still required — the sentence is only cm-cli's when it names something — but
#   the id and its closing quote are one optional group, so a quoted value longer
#   than the capture bound leaves `pack` empty and pushes the id into `reason`
#   instead of making the whole sentence unmatchable, which would drop the failure
#   entirely and keep `ok: True`. Same one-directional bias as everything else
#   here: an unnamed failure is still a failure.
_NODE_INSTALL_FAILURE_RE = re.compile(
    r"An error occurred while installing '(?:(?P<pack>[^']{0,160})'\.?)?"
    r"(?P<reason>(?:(?!(?:ERROR: )?An error occurred while installing).){0,200})"
)

# rich's closing markup tag, which a piped child emits literally. Trimmed off the
# reason window above. Matched as a TAG (`[/bold red]`, `[/red]`) rather than as a
# bare `[/`: cm-cli's own message can carry a bracketed absolute path — `not found
# in [/srv/channels/local, local]`, `pip install failed in [/home/u/venv]` — and
# trimming at the first `[/` anywhere would cut that message off mid-sentence, or
# to nothing at all when it opens with one. A tag is `[/` plus a word, so the two
# cannot be confused: no rich style name contains a `/`. `[/]` — rich's "close the
# last style" spelling — is matched too, since it is a tag by the same reasoning.
_RICH_CLOSE_TAG_RE = re.compile(r"\[/(?:[A-Za-z][\w ]*)?\]")

# Ceiling on reported failures. One record per pack cm-cli names, and a call can
# name at most `_MAX_NODE_PACK_NAMES` packs, so anything past this is a runaway
# (or a pack's own output quoting the sentence) rather than a real result.
_MAX_INSTALL_FAILURES = 32


def _dedupe_install_failures(
    failures: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    """Distinct failure records, first spelling wins, capped.

    Dedup is not tidiness: the same sentence reaches this parse twice routinely —
    a pack's own install log echoing it, or rich writing to both streams — and the
    cap is what makes that dangerous. Capping a CONCATENATION (what the two
    streams used to be) lets ``_MAX_INSTALL_FAILURES`` copies of one echoed
    sentence evict every genuine record, and the packs the engine really rejected
    then come back inside ``installed``. Deduping first spends the ceiling on
    distinct failures, which is the only thing it was ever meant to bound.

    The cap is enforced here rather than by the caller so it can stop consuming
    the (lazy) scan behind it, and so no path can reach the payload uncapped.
    """
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for failure in failures:
        key = (failure["pack"], failure["reason"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(failure)
        if len(unique) >= _MAX_INSTALL_FAILURES:
            break
    return unique


def _trim_install_reason(reason: str) -> str:
    """cm-cli's message, with rich's closing tag taken off. See :data:`_RICH_CLOSE_TAG_RE`."""
    closed = _RICH_CLOSE_TAG_RE.search(reason)
    return (reason[: closed.start()] if closed else reason).strip()


def _extract_install_failures(text: str) -> list[dict[str, str]]:
    """Packs ``comfy node install`` reported as FAILED, out of its printed text.

    This exists because ``node install``'s exit status is not its verdict, which
    is not an assumption but an observation: a `node install` naming a pack that
    is not in the registry prints cm-cli's failure sentence and still exits 0,
    so the wrapper's ``plain_ok`` synthesis — whose whole premise is "a clean
    exit is success" — reported an install that never happened as a success, with
    the failure buried in prose no structured consumer reads.

    Reading it out of the printed text is the only option available, and it is
    the same move :func:`_extract_saved_paths` makes for ``comfy generate``: this
    verb emits no envelope, so its text is not a second-best channel, it is the
    ONLY channel comfy-cli offers. The verdict returned here is still comfy-cli's
    own — this parses its sentence, it does not decide anything about the pack.
    The real fix is upstream (an envelope for the verb, or an exit status that
    reflects the outcome), at which point ``_run_comfy`` takes the envelope path
    and this is bypassed entirely.

    Detection is DELIBERATELY one-directional: a pack is failed only when cm-cli
    said so in as many words, and everything else is left alone. The opposite
    design — confirming each pack against cm-cli's ``[INSTALLED]`` / ``[ SKIP ]``
    / ``[ ENABLED ]`` lines and failing whatever is unconfirmed — is a strictly
    worse trade here, because a future wording change would then report every
    successful install as a failure. This way the same change regresses to the
    pre-existing behaviour: noisy but not actively wrong.

    Both fields are scrubbed (:func:`failure_log._scrub_text`) before they are
    returned: ``reason`` relays cm-cli's own message, which for a pack installed
    from a private channel can quote a repository URL carrying credentials. The
    scrub runs after the window is taken, which is safe in the one direction that
    matters: the window is cut at its END, and ``_URL_RE`` anchors on the
    ``https://`` at a URL's HEAD, so a credential URL cut short by the cap is
    still matched (and its remaining tail is not credential material).

    ``pack`` may be ``""`` when the quoted value did not survive the capture
    bound — the failure is still reported, because "something failed and we could
    not name it" must not read as success.
    """
    if _NODE_INSTALL_FAILURE_GATE not in text:
        return []
    # Fold only now, and for the same reason `_normalize_cli_text` folds: rich
    # wraps this sentence at the console width, so a pack id long enough to push
    # it past that width would otherwise put a newline between the words and
    # defeat the match. Not lowercased — the pack id is compared against the
    # caller's own names, and the reason is relayed verbatim.
    folded = _PANEL_NOISE_RE.sub(" ", _ANSI_RE.sub("", text))
    # A generator, so `_dedupe_install_failures`' cap stops the scan rather than
    # trimming a list that was already built.
    return _dedupe_install_failures(
        {
            "pack": failure_log._scrub_text((match.group("pack") or "").strip()),
            "reason": failure_log._scrub_text(
                _trim_install_reason(match.group("reason"))
            ),
        }
        for match in _NODE_INSTALL_FAILURE_RE.finditer(folded)
    )


def _synthesize_plain_result(args: tuple[str, ...], stdout: str, stderr: str) -> dict:
    """Success payload for a ``plain_ok`` command that exited 0 without an envelope.

    Some comfy-cli commands print human-readable text and exit 0 instead of
    emitting an ``envelope/1`` object: the lifecycle verbs ``launch`` / ``stop``
    and ``model download``, whose stderr carries the progress
    tail (e.g. ``Done in 55.8s``) and the saved-path text. For those a clean exit
    IS the success signal, so we return a result dict carrying whatever text
    comfy-cli printed (preferring stderr, per the CLI's logging) rather than
    raising on the absent envelope — a false negative that would invite a retry
    of an action that already succeeded (a non-idempotent lifecycle change, or a
    bandwidth-expensive multi-GB refetch).

    The synthesized ``message`` carries the printed text verbatim (capped) and
    is always present. When that text contains a ``Saved:`` block — today only
    ``comfy generate``, i.e. :func:`partner_generate` — the resolved output paths
    are ALSO returned as ``saved_paths``, so a caller learns where the asset
    landed without scraping prose that rich may have wrapped mid-filename (see
    :func:`_extract_saved_paths`). The key is omitted when there is no such
    block, so ``launch`` / ``stop`` / ``model download`` are unchanged, and it
    never replaces ``message``: the text stays the fallback for anything the
    parse cannot recover.

    This path is a stopgap: once comfy-cli emits an envelope for a verb, a real
    envelope always wins in the ``_run_comfy`` fast-path and this synthesis is
    bypassed.
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
    # Omitting the raw args is not on its own enough, because the captured text
    # above is the OTHER side of the same credential: comfy-cli echoes the URL
    # it is fetching (`Downloading <url>`) to stderr, so `model download`'s
    # legacy foreground fallback and `partner_generate` would hand a signed
    # CivitAI/HuggingFace URL straight back on the SUCCESS path — the one path
    # the failure-side scrubbing this module does never sees. Scrubbed BEFORE
    # the tail cap below, never after: capping first can bisect a URL so its
    # `https://` is gone and `failure_log._URL_RE` can no longer see the
    # credential remainder (the ordering `failure_log._scrubbed_stream_tail`
    # documents). `saved_paths` below is parsed from the RAW text on purpose —
    # those are local filesystem paths, and the caller needs them exact.
    message = failure_log._scrub_text(message)
    result = {
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
    # Parsed from the UNCAPPED text — the cap above is a tail slice that would
    # otherwise cut a long `Saved:` block mid-path. Each stream is scanned on its
    # OWN rather than concatenated: it keeps a stderr tail from ever sharing a
    # physical line with the block and defeating its indent test (what the
    # newline join used to buy), and it skips the join's full copy of a
    # multi-megabyte `model download` progress stream.
    saved_paths: list[str] = []
    for part in (stderr, stdout):
        if part.strip():
            saved_paths.extend(_extract_saved_paths(part))
    if saved_paths:
        # Bounded like every other field here: a pathological run must not turn
        # a success payload into an unbounded response. The per-path bound lives
        # in `_extract_saved_paths` (over-long entries are dropped, not sliced).
        result["saved_paths"] = saved_paths[:_MAX_SAVED_PATHS]
    # The one verb whose exit 0 is not a success signal, so the `ok` above is a
    # claim this synthesis is not entitled to make for it. Scoped to the exact
    # subcommand rather than applied to every `plain_ok` verb, so that `launch` /
    # `stop` / `model download` / `generate` cannot be flipped by a nested pack's
    # output quoting the sentence. Read off the UNCAPPED streams for the same
    # reason `saved_paths` is: the `message` below keeps only a 1000-character
    # tail, and in a multi-pack install the pack that failed FIRST is pushed out
    # of it by everything the later packs print — which is exactly the partial
    # failure a caller most needs to be told about.
    # stdout first, and deduped across the two: cm-cli writes this sentence to
    # whichever stream rich holds, and a pack's own install log can echo it to the
    # other — so the same failure arrives twice, and `_MAX_INSTALL_FAILURES`
    # applied to the raw concatenation would let those copies evict the genuine
    # records. See `_dedupe_install_failures`.
    if tuple(action_parts[:2]) == _NODE_INSTALL_ACTION:
        failures = _dedupe_install_failures(
            _extract_install_failures(stdout) + _extract_install_failures(stderr)
        )
        if failures:
            result["ok"] = False
            result["failures"] = failures
            result["note"] = (
                "comfy-cli emitted no JSON envelope for this command and exited "
                "0, but its output names packs that FAILED to install — for this "
                "command the exit status is not the verdict, so the printed "
                "failure is."
            )
    return result


# The sentence :func:`_synthesize_plain_result` INVENTS when the child printed
# nothing at all — ``comfy <action> completed (exit 0).``. It is the wrapper's
# own words, not comfy-cli's, so :func:`_plain_message` reports it as absent:
# its callers quote the result as "comfy-cli's own output", and a silent child
# has to read as ``<empty>`` rather than as a wrapper line dressed up as the
# engine's. Genuine output that IS this sentence verbatim is indistinguishable
# from the placeholder and loses nothing by being treated as it.
_SYNTHESIZED_SILENT_RE = re.compile(r"\Acomfy [\w .-]*completed \(exit 0\)\.\Z")


def _plain_message(result: Any) -> str:
    """The text comfy-cli PRINTED for a :func:`_synthesize_plain_result` payload.

    The reader half of the synthesizer, for a caller whose real answer is a side
    effect rather than the return value (:func:`workflow_deps`) and which needs
    comfy-cli's printed output only to explain a failure. Everything is checked
    rather than assumed, because the same call site sees a REAL envelope's
    ``data`` the moment comfy-cli grows one for that verb — an arbitrary payload
    with no ``message``, and possibly not a dict at all. The text is already
    scrubbed and capped by the synthesizer.

    Returns ``""`` for the synthesizer's own placeholder as well as for a
    missing/non-string ``message`` — see :data:`_SYNTHESIZED_SILENT_RE`. A
    caller can therefore treat a falsy result as "the child said nothing" and
    substitute its own wording.
    """
    if isinstance(result, dict):
        message = result.get("message")
        if isinstance(message, str) and not _SYNTHESIZED_SILENT_RE.match(message):
            return message
    return ""


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


# Click's "No such option: --background" usage error — the OPTION-shaped sibling
# of `_MISSING_VERB_RE_TEMPLATE` above, built and read the same way (see
# `_is_missing_option_error`). The separator run is `\W{1,8}` because Click
# writes a colon and a space there and rich may wrap the panel mid-phrase; the
# option name is `re.escape`d, so its own leading dashes are matched literally
# rather than absorbed by that run.
_MISSING_OPTION_RE_TEMPLATE = r"no\s+such\s+option\W{{1,8}}{option}(?![\w.:/-])"


def _is_missing_option_error(exc: ComfyCliError, option: str) -> bool:
    """Is *exc* comfy-cli rejecting ``option`` as unknown, rather than *running* it?

    The option-level counterpart to :func:`_is_missing_verb_error`, for a verb
    that exists but does not yet take the flag we passed —
    ``comfy model download --background`` against a comfy-cli released before
    the background download landed. Click raises ``NoSuchOption`` while parsing,
    which is a ``UsageError``: exit 2, and no envelope, because nothing was ever
    dispatched.

    Both of that function's conditions are required here for exactly its
    reasons, and the stakes are the same: the caller's degrade silently reruns
    the download synchronously, so a false positive would turn a genuine submit
    failure into a second, blocking transfer. ``no_envelope`` keeps a relayed
    "no such option" — from a nested pip/git call comfy-cli made, or a registry
    message it echoed — from reaching the degrade, and the usage-exit status
    narrows it to "the parser rejected the command line". The option name must
    then appear itself, ending at a real delimiter, so ``--background`` does not
    match a longer ``--background-worker``.
    """
    if not exc.no_envelope or exc.returncode != _CLICK_USAGE_ERROR_EXIT:
        return False
    pattern = _MISSING_OPTION_RE_TEMPLATE.format(option=re.escape(option))
    normalized = _normalize_cli_text(str(exc))
    return re.search(pattern, normalized, re.IGNORECASE) is not None


# The two error codes `install_node` reports per failed pack. `pack_not_found` is
# the one a caller can act on without reading prose — it means the id is not in
# the registry channel this install reads, so retrying is pointless and the fix is
# a different id (`nodes(action="search")` / `workflow_deps` are where a real one comes
# from). Everything else — a clone failure, a dependency conflict, an unreachable
# registry — is `install_failed`, where the engine's own `error` text is the only
# thing that distinguishes them and a retry may well work.
_PACK_NOT_FOUND_CODE = "pack_not_found"
_INSTALL_FAILED_CODE = "install_failed"

# cm-cli's `install_by_id` message for an id its channel does not carry:
# ``Node '<id>@<version>' not found in [<channel>, <mode>]``. Matched on the
# distinctive middle rather than the whole line so a future change to the bracket
# suffix degrades to `install_failed` — a weaker code on a still-correct failure —
# rather than to a missed failure.
_PACK_NOT_FOUND_RE = re.compile(r"Node '[^']{0,160}' not found in", re.IGNORECASE)

# Whitespace, removed from BOTH sides of the attribution compare below. A pack id
# is a registry slug (`_REGISTRY_ID_RE`: letters, digits, `.`, `_`, `-`), so it
# never carries whitespace of its own and this can only ever remove whitespace
# that was not in the id — which is exactly the case it exists for: rich hard-wraps
# a long id at the console width, `_extract_install_failures`' fold turns that
# break into a SPACE, and an exact compare would then miss a pack the caller
# named, leave it inside `installed`, and keep `restart_required` true.
_PACK_KEY_NOISE_RE = re.compile(r"\s+")


def _install_pack_key(value: str) -> str:
    """A pack id reduced to what attribution compares. See :data:`_PACK_KEY_NOISE_RE`."""
    return _PACK_KEY_NOISE_RE.sub("", value).lower()


def _classify_install_result(names: list[str], result: Any) -> dict[str, Any]:
    """``install_node``'s payload, with the engine's own per-pack verdict applied.

    The whole point is that ``installed`` must list only packs that were actually
    installed. Previously it echoed the caller's own argument, so a pack that
    cm-cli refused came back inside ``installed`` alongside ``ok: true``, and an
    agent reading those two structured fields told the user to restart a ComfyUI
    that had gained nothing — a failure reporting itself as a success, which is
    worse than a plain error because nothing downstream has a reason to doubt it.

    ``result["failures"]`` is :func:`_extract_install_failures`' output, put there
    by :func:`_synthesize_plain_result`. Its absence means one of two very
    different things, and BOTH correctly yield the unchanged success payload: the
    run printed no failure sentence, or comfy-cli has since grown a real envelope
    for the verb (in which case ``result`` is that envelope's ``data``, this
    parse is bypassed, and the engine's own structured verdict is what the caller
    sees). Hence the defensive ``isinstance`` rather than a bare ``.get``.

    Attribution is by NAME against the caller's own ``names``, compared through
    :func:`_install_pack_key` — case-insensitively because cm-cli echoes back
    whatever spelling it resolved, and whitespace-insensitively because rich may
    have wrapped a long id and the fold upstream turned that break into a space.
    A failure whose pack matches none of them — an unparseable name, or a
    dependency pack cm-cli pulled in and named itself — is still reported, under
    cm-cli's own spelling and without removing anything from ``installed``. That
    direction is deliberate: it cannot silently drop a pack the caller asked for,
    and it cannot silently swallow a failure either.

    ``restart_required`` becomes ``False`` when NOTHING was installed. It is
    otherwise unchanged (always ``True``), and the reason for the exception is the
    same one this function exists for: its documented meaning is "the running
    ComfyUI will not see the new nodes until it restarts", and with no new nodes
    that sentence is not merely vacuous but is the specific bad advice the false
    success produced — the user restarts, the nodes are still missing, and the
    tool call that sent them there looked clean.
    """
    failures = result.get("failures") if isinstance(result, dict) else None
    if not isinstance(failures, list) or not failures:
        return {"installed": names, "result": result, "restart_required": True}
    requested = {_install_pack_key(name): name for name in names}
    failed: list[dict[str, str]] = []
    failed_keys: set[str] = set()
    attributed = 0
    for failure in failures:
        # Every field is checked rather than assumed for the second reason above:
        # the moment comfy-cli grows a real envelope for this verb, `result` is
        # that envelope's `data` and a `failures` key in it is the ENGINE's, of
        # whatever shape it likes. An `AttributeError` there would turn a
        # perfectly reportable install into an unhandled internal error — the
        # opposite of what this function exists to do.
        if not isinstance(failure, dict):
            continue
        pack = failure.get("pack")
        reason = failure.get("reason")
        pack = pack if isinstance(pack, str) else ""
        reason = reason if isinstance(reason, str) else ""
        key = _install_pack_key(pack)
        if key in requested:
            failed_keys.add(key)
            attributed += 1
            reported = requested[key]
        else:
            # Not one of ours, or not parseable. Report it verbatim; `""` is left
            # as-is so the record still says a failure happened when cm-cli's
            # sentence carried no usable name.
            reported = pack
        failed.append(
            {
                "name": reported,
                "code": (
                    _PACK_NOT_FOUND_CODE
                    if _PACK_NOT_FOUND_RE.search(reason)
                    else _INSTALL_FAILED_CODE
                ),
                "error": reason,
            }
        )
    if not failed:
        # Every entry was unreadable, so nothing was reported as failed and the
        # success payload is still the right answer. Without this the branch would
        # emit "0 of N pack(s) failed to install" alongside an empty `failed`,
        # which is a worse lie than the one this function was written to remove.
        return {"installed": names, "result": result, "restart_required": True}
    installed = [name for name in names if _install_pack_key(name) not in failed_keys]
    # The ratio counts only the failures attributed to a pack the caller ASKED
    # for, because `len(names)` is its denominator: a record can name a pack
    # outside `names` (a dependency cm-cli resolved itself, or a sentence whose id
    # did not survive), and counting those against the requested total reads as
    # "2 of 1 pack(s) failed" — or claims one of them failed while that very pack
    # is still listed in `installed`. The rest are reported, in their own clause.
    unattributed = len(failed) - attributed
    return {
        "installed": installed,
        "failed": failed,
        "error": (
            f"install_node: {len(failed_keys)} of {len(names)} requested pack(s) "
            "failed to install"
            + (
                f", and the engine reported {unattributed} further failure(s) it "
                "did not attribute to any of them"
                if unattributed
                else ""
            )
            + ". comfy-cli exited 0 anyway — for `comfy node install` the "
            "exit status is not the verdict — so `installed` lists only the packs "
            "it did NOT report as failed. See `failed` for the engine's own "
            "message per pack; a `pack_not_found` code means the id is not in "
            "this install's registry channel, so retrying the same id will not "
            "help."
        ),
        "result": result,
        # See the docstring: no new nodes, nothing for a restart to pick up.
        "restart_required": bool(installed),
    }


# The missing-verb/-option matchers are deliberately strict because their
# degrade asserts NOTHING IS BROKEN (see `_is_missing_verb_error`), and their
# `no_envelope` condition exists to keep a RELAYED "no such command" out. This
# closes the one door that condition cannot: text the CALLER put on the command
# line. Click echoes an offending value verbatim in its usage error (`Invalid
# value for '[PACK]': …`), which lands on the same exit 2 with no envelope the
# matchers read — so a caller passing `pack="no such command 'deps'"` could
# otherwise forge the parser's own message about `deps` and convert a genuine
# usage failure into "your comfy-cli is just too old".
#
# The degrade sites, by what their argv carries — the VERB-shaped ones first:
# `outdated` (`_freshness_report`) and `system-stats` / `free`
# (`_resource_verb_upgrade_error`) take no caller text at all; `notes`
# (`list_workflow_notes`) carries `workflow_path`; `download-status` /
# `download-cancel` (`_download_verb_unsupported`) carry `download_id`; `node
# deps` carries `pack` and `registry_id`. Then the OPTION-shaped siblings, which
# read `_is_missing_option_error` and are in the same set for the same reason:
# `--registry` (`node_dependencies`) carries those same two values, and
# `--background` (`download_model`) carries `url` / `relative_path` /
# `filename`. The remaining two option sites need no subtraction because their
# values CANNOT spell the phrase, not because nobody supplied them: `--port`
# (`get_logs`) forwards a `_guard_log_port`-normalized int, and `--version`
# (`_run_version_switch`) a `_guard_version`-constrained token (`nightly` /
# `latest` / a semver tag).
#
# EVERY site whose argv can carry the phrase applies this subtraction, passing
# its own values — the matchers see only the exception and have no way to know
# what their caller put on the command line, which is why the values are a
# per-site argument rather than folded in.
#
# For `notes`, the download verbs and `--background` this is forward cover
# rather than a live hole: on comfy-cli main those parameters are plain `str`
# Typer params validated in the command body, so the failure is an enveloped
# exit 1 and Click never echoes them at parse time. Retyping one to a
# parse-time-validated type (`Path`, a `Choice`) is a one-line comfy-cli change
# that needs no coordination with this repo — which is exactly the change that
# would open the door here.
def _phrase_is_only_the_caller_s(
    exc: ComfyCliError, pattern: str, *values: str
) -> bool:
    """Does *pattern* match only text the CALLER supplied, not comfy-cli's own?

    Removes every qualifying entry of *values* from the normalized message and
    re-runs *pattern*. No match afterwards means the sole occurrence came from an
    echoed argument, and the degrade must not fire. Only entries that match
    *pattern* THEMSELVES are subtracted — see below for why that qualifier is
    the load-bearing part rather than an optimization.

    A value that is itself a substring of comfy-cli's genuine message deletes
    that occurrence too, so a caller who passes the parser's exact wording gets
    the raw passthrough instead of the degrade. That is the same one-directional
    trade :func:`_is_missing_verb_error` documents — noisy but honest beats a
    false "nothing is broken" — and here it is also the only self-inflicted way
    to reach it.

    Only a value that MATCHES *pattern* on its own is subtracted, which is the
    load-bearing half of this function. ``str.replace`` is global, so
    subtracting unconditionally would delete a caller's value from comfy-cli's
    OWN phrase whenever the value happened to be a substring of it — and the
    values that collide are ordinary, not crafted: ``filename="background"``
    erases the flag out of ``No such option: --background``,
    ``workflow_path="notes"`` and ``download(action="status",
    download_id="a")`` do the same to ``No such command 'notes'`` and
    ``'download-status'``. A value that does not
    contain the phrase cannot have forged the phrase, so there is nothing to
    discount and it is left alone. The cost of getting this wrong is not
    symmetric: at the verb sites an over-subtraction only costs the friendly
    message, but at ``download_model``'s ``--background`` the suppressed degrade
    is the only path that PERFORMS the transfer, so a benign ``filename`` would
    have turned a working legacy-CLI download into an outright failure.

    Matching is on the echo being VERBATIM (modulo the normalization both sides
    get), which is what Click's ``repr`` of an offending value gives for the
    shapes that can carry the phrase at all — the value needs a quote, and
    ``repr`` answers a single-quoted value with double quotes rather than
    backslashes. Three ways the echo can fail to be verbatim are known and
    accepted; in each the subtraction is a no-op, so the result is exactly the
    behaviour that site had BEFORE this discount existed, and reaching it at all
    is self-inflicted — the degrade a caller forges is returned to the same
    caller that crafted the argument.

    - A value carrying BOTH quote styles comes back re-escaped, so the raw value
      is no longer a substring of the message.
    - Click does not always echo the value byte-for-byte. The retype this is
      forward cover for is exactly where that shows: a Typer
      ``Path(..., resolve_path=True)`` echoes the RESOLVED path, and ``repr``
      doubles a backslash.
    - The message this reads is a bounded 500-char tail, so a value longer than
      the tail arrives clipped. The id-shaped values are already capped well
      under that bound (``_MAX_DOWNLOAD_ID_LEN``, ``_MAX_NODE_PACK_ID_LEN``);
      every PATH-shaped value carries only the far more generous argv-safety
      ceiling ``_MAX_PATH_ARG_LEN`` — ``workflow_path``, ``download_model``'s
      ``relative_path`` and ``filename``, the three ``out_path`` values, the two
      ``out_dir`` values, each entry of ``upload_file``'s ``paths`` and
      ``search_models``' ``folder`` — as ``download_model``'s ``url`` carries
      ``_MAX_URL_LEN``. (That list and the one at :data:`argv._MAX_PATH_ARG_LEN` are
      the same set, and are meant to stay in sync.) Both ceilings sit
      thousands of characters ABOVE the tail and so leave this window exactly as
      open as it was uncapped. That is deliberate — a cap tight enough to close
      the gap would have to sit under 500 characters, and would start refusing
      legitimately deep paths and long CivitAI/HuggingFace URLs to buy nothing
      but protection from a caller's own crafted argument.
      That tail is also URL-SCRUBBED on its way to the client
      (``failure_log._scrubbed_stream_tail``), so a value whose echo carries
      credential material — userinfo or a query string, the only two things
      ``failure_log._scrub_url`` rewrites — comes back masked rather than
      verbatim. That is NOT a fourth way to reach the no-op, because the
      subtraction below scrubs the value too, so the two sides are rewritten
      identically and still cancel. Only ``download_model``'s ``url`` is
      URL-shaped at all; the other sites pass identifiers and filesystem paths,
      which need the ``https?://`` scheme ``failure_log._URL_RE`` anchors on and
      so survive byte-for-byte.

    One residual runs the other way and is also accepted: requiring the value to
    carry the phrase whole means a value holding only a FRAGMENT is never
    discounted, so the phrase could be assembled ACROSS a join that neither the
    value nor this check sees whole. Three joins exist and none is reachable
    today:

    - Click's own template, if its fixed wording ended mid-phrase exactly where
      it interpolates the value. None of its usage templates do.
    - Two caller values landing adjacently in one message. Click echoes only the
      ONE value it rejected, so the sites passing two (``pack`` /
      ``registry_id``, and ``download_model``'s three) never get both echoed.
    - The wrapper's own ``"stderr: … | stdout: …"`` framing, whose ``" | "``
      :func:`_normalize_cli_text` folds to a single space. Splitting the phrase
      there needs a fragment at the very END of the stderr tail and its
      completion at the very START of stdout — and the caller supplies argv, not
      the child's stdout, so it does not control both sides.

    All three would need the retype this is forward cover for to have happened
    first, since without it Click never echoes these values at parse time at all.
    """
    normalized = _normalize_cli_text(str(exc))
    for value in values:
        # SCRUBBED before normalizing, because the message being subtracted from
        # is scrubbed too: `download_model`'s `url` is the one caller value that
        # is URL-shaped, and `failure_log._scrub_url` rewrites it (userinfo
        # masked, query dropped) on its way into `str(exc)`. Subtracting the RAW
        # value would then miss its own echo — `str.replace` finds nothing — and
        # the forged phrase would survive the discount, which at this site means
        # a crafted `url` re-enables the legacy foreground transfer the
        # `--background` degrade exists to gate. Both sides through the same two
        # passes is what keeps the comparison honest.
        echoed = _normalize_cli_text(failure_log._scrub_text(value))
        # A value that does not carry the phrase itself could not have forged
        # it — see above. This also disposes of the empty-normalization case
        # (`" "`, which normalizes to `""`): `str.replace("", " ")` would
        # interleave a space between EVERY character of the message, and an
        # empty string never matches *pattern*, so it is skipped here.
        if echoed and re.search(pattern, echoed, re.IGNORECASE):
            normalized = normalized.replace(echoed, " ")
    return re.search(pattern, normalized, re.IGNORECASE) is None


# ComfyUI-Manager's absence, as `comfy node deps-in-workflow` reports it. The verb
# shells out to Manager's `cm-cli`, and comfy-cli's `execute_cm_cli` hard-exits
# (status 1, this line on stderr, no envelope) when the module is not importable
# from the workspace Python. Matched on the two halves of that sentence with
# `.{0,80}` between them so rich's panel wrapping — folded to single spaces by
# `_normalize_cli_text` — cannot separate them, while an unrelated line that
# merely says "not found" much later in the stream still fails to match.
_MANAGER_MISSING_RE = r"comfyui-manager not found.{0,80}cm-cli.{0,40}not available"


def _is_manager_missing_error(exc: ComfyCliError, *caller_values: str) -> bool:
    """Is *exc* comfy-cli refusing because ComfyUI-Manager is not installed?

    The sibling of :func:`_is_missing_verb_error` for a DEPENDENCY gap rather
    than a version one, and narrow for the same reason: the caller's degrade
    tells the agent that nothing is broken except a missing prerequisite.

    ``exc.no_envelope`` is required exactly as it is there — comfy-cli aborts
    before any envelope is emitted, so an error envelope that merely quotes this
    sentence (a pack's own output relayed through a verb that DID run) is not
    this. What is deliberately NOT required is Click's usage status: this abort
    happens after dispatch, inside the command body, so it exits 1 rather than 2
    and the missing-verb matcher's second condition does not apply.

    That missing status check makes the echoed-argument door wider here than it
    is for ``node deps``, so :func:`_phrase_is_only_the_caller_s` is not optional
    on this path: a ``workflow_path`` carrying the sentence comes back verbatim
    in cm-cli's own ``File not found: <path>`` line, on the same exit 1 with no
    envelope, and would otherwise forge this degrade.
    """
    return (
        exc.no_envelope
        and re.search(_MANAGER_MISSING_RE, _normalize_cli_text(str(exc))) is not None
        and not _phrase_is_only_the_caller_s(exc, _MANAGER_MISSING_RE, *caller_values)
    )
