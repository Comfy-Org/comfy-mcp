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

from collections.abc import Sequence
from importlib import metadata

from . import __version__

# The installed distribution's name (``[project] name`` in pyproject.toml).
_DISTRIBUTION = "comfy-mcp"

_HELP_FLAGS = frozenset({"-h", "--help"})
_VERSION_FLAGS = frozenset({"--version", "-V"})


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
    """
    try:
        return metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return __version__


def _usage(min_comfy_cli: str) -> str:
    """The ``--help`` text.

    ``min_comfy_cli`` is passed in rather than imported so the floor keeps a
    single source of truth — ``server._MIN_COMFY_CLI_STR``, the same string the
    runtime guard rejects an older comfy-cli with. Importing it here would
    point a leaf module at ``server``; duplicating it would let the advertised
    floor drift away from the enforced one.
    """
    return f"""\
comfy-mcp {_version()} — MCP server for ComfyUI (stdio transport).

Not meant to be run interactively: it speaks MCP over stdin/stdout and is
launched by an MCP client as a subprocess. Register it with your client
instead, for example:

  claude mcp add comfy-mcp -- comfy-mcp

Run with no arguments (as a client does) it serves MCP on stdio and exits
when stdin closes — so a bare `comfy-mcp` in a terminal looks like it did
nothing. That is expected.

Options:
  -h, --help     show this message and exit
  -V, --version  show the version and exit

Requires comfy-cli >= {min_comfy_cli} on PATH (`pip install "comfy-cli>={min_comfy_cli}"`);
set COMFY_BIN to its absolute path if it is not on the PATH your client uses.

Docs: https://github.com/Comfy-Org/comfy-mcp
"""


def _handle_argv(argv: Sequence[str], min_comfy_cli: str) -> bool:
    """Answer ``--help`` / ``--version`` if present; report whether we did.

    ``True`` means the caller has already been served and must NOT start the
    server. ``False`` means "carry on and serve" — including for arguments this
    does not recognise, which is deliberate: argv was ignored entirely before
    this existed, so rejecting an unknown flag would newly break any client
    registration that passes one. A flag anywhere in ``argv`` counts (``-h`` is
    the request even behind something else), and ``--help`` wins over
    ``--version`` so a user asking for both gets the more useful answer.
    """
    args = set(argv)
    if args & _HELP_FLAGS:
        print(_usage(min_comfy_cli), end="", flush=True)
        return True
    if args & _VERSION_FLAGS:
        print(f"comfy-mcp {_version()}", flush=True)
        return True
    return False
