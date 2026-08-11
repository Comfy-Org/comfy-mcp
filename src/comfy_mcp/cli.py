"""The console script's own argv surface — ``--help`` and ``--version``.

Leaf module over :mod:`comfy_mcp` (the package ``__init__`` alone, for the
fallback version); nothing here imports ``server``. It exists because
``comfy-mcp`` is a *stdio* server: a first-time user who types the command in a
terminal gets no banner, no prompt and no error — the server starts, waits for
JSON-RPC on stdin, and exits silently at EOF, which reads as "the install is
broken". These two flags are the one place this program talks to a HUMAN, so
they say what it is and how to launch it properly.

Printing to **stdout** here is deliberate and is not a breach of the rule that
stdout belongs to JSON-RPC (see :mod:`comfy_mcp.failure_log`): that rule holds
while the server is serving, and the whole point of these flags is that they
return before ``mcp.run()`` is ever reached, so no client is listening on the
other end. ``--help`` on stdout is also what a pipe into ``less``/``grep``
expects.

Everything here is private and reached as ``cli._name`` — there is no public
name in this module.
"""

from __future__ import annotations

import os
import sys
from importlib import metadata

from . import __version__

# The installed distribution's name (``[project] name`` in pyproject.toml).
_DISTRIBUTION = "comfy-mcp"

_HELP_FLAGS = frozenset({"-h", "--help"})
_VERSION_FLAGS = frozenset({"--version", "-V"})

# POSIX end-of-options marker: everything after it is an operand, never a flag.
_END_OF_OPTIONS = "--"


def _version() -> str:
    """The running version, preferring installed package metadata.

    Metadata is the authoritative answer for the thing the user actually
    installed — it is what ``pip show comfy-mcp`` reports and what a
    ``serverInfo.version`` should agree with. It is unavailable in exactly one
    situation: running from a source tree that was never installed (no
    ``.dist-info``), where :data:`comfy_mcp.__version__` is the source of truth
    the release checks pin (``tests/test_packaging.py``). So try metadata, fall
    back to the literal — never the other way round, which would report the
    checkout's number for an install that shipped a different one.

    The fallback is deliberately wider than "no ``.dist-info``": a
    ``.dist-info`` that EXISTS but is corrupt, unreadable, or carries no
    ``Version`` field fails as something other than
    ``PackageNotFoundError`` (an ``OSError`` reading METADATA) or does not fail
    at all and hands back ``None``. That broken install is precisely the case
    the README sends users here to diagnose, so every one of those answers the
    source literal rather than a traceback or ``comfy-mcp None``.
    """
    try:
        installed = metadata.version(_DISTRIBUTION)
    except Exception:  # noqa: BLE001 - a broken install must still answer
        return __version__
    return installed or __version__


def _print(text: str) -> None:
    """Write ``text`` to stdout, tolerating a reader that has already left.

    The module docstring advertises piping ``--help`` into ``less``/``grep``,
    and those readers close the pipe early as a matter of course (``| head``,
    or ``q`` in a pager). The write — or the interpreter's own flush at
    shutdown — then raises ``BrokenPipeError``, which is an ``OSError`` and so
    is NOT the ``PermissionError`` :func:`server.main` translates: the user
    gets a traceback plus an "Exception ignored" epilogue for the crime of
    quitting a pager. Python's documented remedy is to redirect the fd at
    ``devnull`` so that shutdown flush has somewhere harmless to go.
    """
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:  # no real fd to salvage (a wrapped stdout); exit quietly
            pass


def _usage(min_comfy_cli: str) -> str:
    """The ``--help`` text.

    ``min_comfy_cli`` is passed in rather than imported so the floor keeps a
    single source of truth — ``server._MIN_COMFY_CLI_STR``, the same string the
    runtime guard rejects an older comfy-cli with. Importing it here would
    point a leaf module at ``server``; duplicating it would let the advertised
    floor drift away from the enforced one.

    The text is deliberately pure ASCII. It is printed through whatever
    encoding the locale gives stdout once that stdout is a file or a pipe
    rather than a terminal, so a single em dash is enough to turn
    ``comfy-mcp --help > out.txt`` into a ``UnicodeEncodeError`` traceback with
    half the message written — on a code page that cannot represent it (cp932)
    or a strict-ASCII C locale. The one screen a confused user gets has to
    render everywhere; ``test_usage_is_pure_ascii`` pins that.
    """
    return f"""\
comfy-mcp {_version()} - MCP server for ComfyUI (stdio transport).

Not meant to be run interactively: it speaks MCP over stdin/stdout and is
launched by an MCP client as a subprocess. Register it with your client
instead, for example:

  claude mcp add comfy-mcp -- comfy-mcp

Run with no arguments (as a client does) it serves MCP on stdio and exits
when stdin closes, so a bare `comfy-mcp` in a terminal looks like it did
nothing. That is expected.

Options:
  -h, --help     show this message and exit
  -V, --version  show the version and exit

Requires comfy-cli >= {min_comfy_cli} on PATH (`pip install "comfy-cli>={min_comfy_cli}"`);
set COMFY_BIN to its absolute path if it is not on the PATH your client uses.

Docs: https://github.com/Comfy-Org/comfy-mcp
"""


def _handle_argv(argv: list[str], min_comfy_cli: str) -> bool:
    """Answer ``--help`` / ``--version`` if present; report whether we did.

    ``True`` means the caller has already been served and must NOT start the
    server. ``False`` means "carry on and serve" — including for arguments this
    does not recognise, which is deliberate: argv was ignored entirely before
    this existed, so rejecting an unknown flag would newly break any client
    registration that passes one. A flag anywhere in the OPTION section counts
    (``-h`` is the request even behind something else), and ``--help`` wins
    over ``--version`` so a user asking for both gets the more useful answer.

    The option section ends at the first bare ``--``, the POSIX end-of-options
    marker: past it a token is an operand, so ``comfy-mcp -- --help`` serves
    rather than printing help. That matters because answering on stdout and
    exiting 0 is the wrong reply to a *client*, which is listening for JSON-RPC
    on that same stdout — it would see prose and a clean exit instead of a
    diagnosable failure. Honouring ``--`` gives any caller that must pass such
    a token an escape hatch; nothing else here is a flag that takes a value, so
    there is no other way for one to arrive as somebody else's argument.

    ``argv`` is a ``list``, not a ``Sequence``, and the guard below enforces it:
    a bare ``str`` IS a ``Sequence[str]``, so ``_handle_argv("--help", …)``
    would iterate CHARACTERS, match nothing, and silently start the server —
    the exact silent-serve failure this module exists to remove. It fails loudly
    instead.
    """
    if not isinstance(argv, list):
        raise TypeError(f"argv must be a list of arguments, got {type(argv).__name__}")
    end = argv.index(_END_OF_OPTIONS) if _END_OF_OPTIONS in argv else len(argv)
    options = set(argv[:end])
    if options & _HELP_FLAGS:
        _print(_usage(min_comfy_cli))
        return True
    if options & _VERSION_FLAGS:
        _print(f"comfy-mcp {_version()}\n")
        return True
    return False
