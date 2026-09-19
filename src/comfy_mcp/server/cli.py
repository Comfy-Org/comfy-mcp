"""The console script's human argv surface and HTTP listener configuration.

Leaf module inside :mod:`comfy_mcp.server`; it imports only the package version
for its fallback and the adjacent listener config. It exists because
No arguments remains the legacy stdio server. ``serve`` is the explicit,
long-running Streamable HTTP mode; parsing it here keeps listener configuration
out of the business/tool module.

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

import logging
import os
import sys
from argparse import ArgumentParser
from importlib import metadata

from pydantic import ValidationError

from .. import __version__
from .config import LogLevel, RemoteServerConfig, RemoteTransport

# The distribution name from ``pyproject.toml`` — what ``pip install`` records.
# Spelled out rather than taken from ``__package__``: that is the IMPORT name
# (``comfy_mcp``), a different string that resolves here only because
# ``importlib.metadata`` normalizes ``_`` to ``-`` before it looks a
# distribution up.
_DISTRIBUTION = "comfy-mcp"

_HELP_FLAGS = frozenset({"-h", "--help"})
_VERSION_FLAGS = frozenset({"--version", "-V"})

# POSIX end-of-options marker: everything after it is an operand, never a flag.
_END_OF_OPTIONS = "--"


def _version() -> str:
    """The running version, preferring installed package metadata.

    The single answer to "which release is this?", for BOTH callers that ask:
    ``comfy-mcp --version`` here, and the private server runtime's
    ``_server_version`` for the
    ``initialize`` handshake's ``serverInfo.version``. One implementation on
    purpose — two would drift into two answers for the same install, and the
    string a client displays has to be the string the user reads off their
    terminal, or a bug report correlates against the wrong release.

    Metadata is the authoritative answer for the thing the user actually
    installed — it is what ``pip show comfy-mcp`` reports. It is unavailable in
    exactly one ordinary situation: running from a source tree that was never
    installed (no ``.dist-info``), where :data:`comfy_mcp.__version__` is the
    source of truth the release checks pin (``tests/test_packaging.py``). So try
    metadata, fall back to the literal — never the other way round, which would
    report the checkout's number for an install that shipped a different one.

    The fallback is deliberately wider than "no ``.dist-info``": a
    ``.dist-info`` that EXISTS but is corrupt, unreadable, or carries no
    ``Version`` field fails as something other than ``PackageNotFoundError`` (an
    ``OSError`` reading METADATA) or does not fail at all and hands back
    ``None``. Both must still answer — a broken install is precisely what the
    README sends users to ``--version`` to diagnose, and on the handshake side
    this runs at IMPORT time, so raising would take the whole server down over a
    display string. Only the unreadable case is WARNED about: a source checkout
    has no metadata by design and is the normal dev path, while a
    half-written ``.dist-info`` is a real fault worth a line in the log.
    """
    try:
        installed = metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        # Not installed as a distribution at all — a source checkout on
        # `PYTHONPATH`. The normal dev case, not an error.
        return __version__
    except Exception:  # a broken install must still answer
        logging.getLogger(__name__).warning(
            "could not read installed %s metadata; reporting the source version",
            _DISTRIBUTION,
            exc_info=True,
        )
        return __version__
    # `or`, not a bare return: metadata with no `Version:` field yields None,
    # which would reach a client as the very empty version this exists to fix.
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
    single source of truth — ``server._internal._MIN_COMFY_CLI_STR``, the same string the
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
comfy-mcp {_version()} - MCP server for ComfyUI (stdio or Streamable HTTP).

With no arguments it speaks MCP over stdin/stdout and is launched by an MCP
client as a subprocess. Register that mode with your client, for example:

  claude mcp add comfy-mcp -- comfy-mcp

A bare interactive invocation exits when stdin closes; use the explicit
`serve` command for a persistent network listener.

For a long-running Streamable HTTP server that does not depend on stdin:

  comfy-mcp serve --host 127.0.0.1 --port 8000

The HTTP endpoint is http://127.0.0.1:8000/mcp by default. A non-loopback
bind requires at least one explicit --allowed-host pattern.

Options:
  -h, --help               show this message and exit
  -V, --version            show the version and exit

serve options (COMFY_MCP_* environment variables provide the defaults):
  --transport VALUE        streamable-http (the only HTTP transport)
  --host HOST              MCP listener host, default 127.0.0.1
  --port PORT              MCP listener port, default 8000
  --path PATH              MCP route, default /mcp
  --log-level LEVEL        DEBUG, INFO, WARNING, ERROR, or CRITICAL
  --allowed-host PATTERN   accepted Host header; repeat for multiple patterns

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


def _serve_config(argv: list[str]) -> RemoteServerConfig | None:
    """Parse ``serve`` or return ``None`` for the legacy stdio invocation.

    Only an explicit leading ``serve`` opts into HTTP. Unknown historical argv
    therefore retains the old fall-through behavior instead of accidentally
    turning a client subprocess registration into a network listener.
    """

    if not isinstance(argv, list):
        raise TypeError(f"argv must be a list of arguments, got {type(argv).__name__}")
    if not argv or argv[0] != "serve":
        return None

    parser = ArgumentParser(prog="comfy-mcp serve", add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--transport",
        choices=[transport.value for transport in RemoteTransport],
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--path")
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=[level.value for level in LogLevel],
    )
    parser.add_argument("--allowed-host", action="append", dest="allowed_hosts")
    parsed = parser.parse_args(argv[1:])

    try:
        base = RemoteServerConfig.from_env()
        return RemoteServerConfig(
            transport=(
                parsed.transport if parsed.transport is not None else base.transport
            ),
            host=parsed.host if parsed.host is not None else base.host,
            port=parsed.port if parsed.port is not None else base.port,
            path=parsed.path if parsed.path is not None else base.path,
            log_level=(
                parsed.log_level if parsed.log_level is not None else base.log_level
            ),
            allowed_hosts=(
                tuple(parsed.allowed_hosts)
                if parsed.allowed_hosts is not None
                else base.allowed_hosts
            ),
        )
    except ValidationError as exc:
        parser.error(str(exc))
    return None  # pragma: no cover - ArgumentParser.error raises SystemExit
