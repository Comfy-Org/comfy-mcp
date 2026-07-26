"""macOS protected-folder (TCC) diagnostics.

Leaf module: pure detection + message building over ``os``/``re``/``sys``, with
no dependency on the rest of the package. ``server`` calls into it wherever a
comfy-cli failure could actually be a macOS privacy denial.
"""

from __future__ import annotations

import os
import re
import sys

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
