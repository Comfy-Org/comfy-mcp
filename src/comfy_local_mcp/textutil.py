"""Pure text helpers shared across the package.

Leaf module by design: it imports nothing from this package, so anything may
depend on it. It holds the two bounded-tail helpers every comfy-cli failure
message is built from and the URL redactor those messages (and the opt-in
failure log) run config values through.
"""

from __future__ import annotations


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
        window = text[-4 * limit :]
        if len(window) < len(text) and not window.strip():
            # The window landed wholly inside a run of trailing whitespace (a
            # progress bar's padding, say), so the strip below would return
            # nothing and silently drop the real error sitting just before it.
            # The str branch never has this problem — it strips first — so only
            # in this case pay for a full-buffer rstrip and re-window.
            text = text.rstrip()
            window = text[-4 * limit :]
        text = window.decode("utf-8", errors="replace")
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
