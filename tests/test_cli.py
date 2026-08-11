"""The console script's `--help` / `--version` surface (`comfy_mcp.cli`).

A stdio server that ignores argv gives a first-time user nothing: `comfy-mcp
--help` starts the server, blocks on stdin, and exits silently at EOF, which is
indistinguishable from a broken install. These pin the two flags that answer
instead — and, just as importantly, that everything else still reaches the
server, since argv was previously ignored outright and a client registration is
free to pass arguments this program does not know about.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

import comfy_mcp
from comfy_mcp import cli, server

_FLOOR = server._MIN_COMFY_CLI_STR
_ROOT = Path(__file__).resolve().parents[1]


def test_help_prints_usage_to_stdout(capsys):
    handled = cli._handle_argv(["--help"], _FLOOR)
    out = capsys.readouterr()
    assert handled is True
    assert out.err == ""
    assert out.out.startswith("comfy-mcp ")
    # The three things a confused user needs: what it is, that it is not an
    # interactive program, and the one command that registers it properly.
    assert "stdio" in out.out
    assert "Not meant to be run interactively" in out.out
    assert "claude mcp add comfy-mcp -- comfy-mcp" in out.out


def test_help_advertises_the_enforced_comfy_cli_floor(capsys):
    """The help text's floor is the runtime guard's, not a second copy of it."""
    cli._handle_argv(["-h"], _FLOOR)
    out = capsys.readouterr().out
    assert f"comfy-cli >= {_FLOOR}" in out
    assert f'pip install "comfy-cli>={_FLOOR}"' in out


def test_version_prints_the_running_version(capsys):
    handled = cli._handle_argv(["--version"], _FLOOR)
    out = capsys.readouterr()
    assert handled is True
    assert out.err == ""
    # Compared against `cli._version()` rather than `comfy_mcp.__version__`:
    # an editable install carries the version recorded when it was installed,
    # so a stale local env legitimately disagrees with the source literal (the
    # same reason `test_packaging.py` reads the files instead of metadata).
    assert out.out == f"comfy-mcp {cli._version()}\n"
    assert re.fullmatch(r"comfy-mcp \d+\.\d+.*\n", out.out)


def test_help_wins_over_version(capsys):
    """Asking for both gets the answer that contains the other one."""
    assert cli._handle_argv(["--version", "--help"], _FLOOR) is True
    out = capsys.readouterr().out
    assert "Options:" in out


@pytest.mark.parametrize("argv", [[], ["--transport", "stdio"], ["-x"], ["help"]])
def test_everything_else_falls_through_to_the_server(argv, capsys):
    """Unrecognised argv must still serve — argv used to be ignored entirely."""
    assert cli._handle_argv(argv, _FLOOR) is False
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "argv",
    [["--", "--help"], ["--", "-h"], ["--", "-V"], ["--transport", "--", "--version"]],
)
def test_end_of_options_marker_is_honoured(argv, capsys):
    """Past a bare `--` a token is an operand, so it must not print to stdout.

    Answering a *client* on stdout is worse than not answering: that stream is
    the JSON-RPC channel, so prose plus exit 0 reads as protocol garbage rather
    than a diagnosable failure. `--` is the escape hatch for a caller that has
    to pass such a token along.
    """
    assert cli._handle_argv(argv, _FLOOR) is False
    assert capsys.readouterr() == ("", "")


def test_a_bare_string_is_rejected_rather_than_read_as_characters():
    """`main("--help")` must not iterate letters and silently serve."""
    with pytest.raises(TypeError, match="list of arguments"):
        cli._handle_argv("--help", _FLOOR)  # type: ignore[arg-type]


def test_usage_is_pure_ascii():
    """Non-ASCII would make `comfy-mcp --help > out.txt` a UnicodeEncodeError.

    Redirected or piped, stdout carries the locale's encoding, and a code page
    that cannot represent an em dash raises after writing half the message.
    """
    usage = cli._usage(_FLOOR)
    usage.encode("ascii")  # raises UnicodeEncodeError if a stray glyph creeps in


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_a_reader_that_left_early_does_not_traceback(flag, monkeypatch):
    """`comfy-mcp --help | head -1`, or `q` in a pager, closes the pipe on us."""

    class _ClosedPipe:
        def write(self, text):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

        def fileno(self):
            raise io.UnsupportedOperation("fileno")

    monkeypatch.setattr(sys, "stdout", _ClosedPipe())
    assert cli._handle_argv([flag], _FLOOR) is True


def test_version_prefers_installed_metadata(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda name: f"9.9.9-{name}")
    assert cli._version() == "9.9.9-comfy-mcp"


def test_version_falls_back_to_the_source_literal(monkeypatch):
    """No `.dist-info` (a source tree that was never installed) still answers."""

    def _missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _missing)
    assert cli._version() == comfy_mcp.__version__


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(OSError(13, "Permission denied"), id="unreadable-metadata"),
        pytest.param(
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"), id="corrupt-metadata"
        ),
    ],
)
def test_version_falls_back_when_the_dist_info_is_broken(broken, monkeypatch):
    """A `.dist-info` that exists but is unreadable fails as something else.

    That is the broken install the README sends users here to diagnose, so it
    must answer rather than traceback.
    """

    def _broken(name):
        raise broken

    monkeypatch.setattr(metadata, "version", _broken)
    assert cli._version() == comfy_mcp.__version__


def test_version_falls_back_when_metadata_has_no_version_field(monkeypatch):
    """METADATA with no `Version:` hands back `None`, not an exception.

    Printing `comfy-mcp None` would both mislead and violate the declared
    `-> str`.
    """
    monkeypatch.setattr(metadata, "version", lambda name: None)
    assert cli._version() == comfy_mcp.__version__


def test_the_terminal_and_the_handshake_report_the_same_version(monkeypatch):
    """One lookup, so `--version` and `serverInfo.version` cannot drift apart.

    They answer the same question about the same install, and a bug report
    correlates the string a user read off their terminal against the string
    their client displayed. `server._server_version` delegates here rather than
    repeating the metadata read; this fails if someone forks it back into two.
    Checked on the fallback branch too, since that is where two copies diverged
    before they were collapsed.
    """
    assert server._server_version() == cli._version()

    def _absent(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _absent)
    assert server._server_version() == cli._version() == comfy_mcp.__version__


def test_main_answers_help_without_starting_the_server(monkeypatch, capsys):
    """`main()` returns before any startup probing or `mcp.run()`."""

    def _boom(*args, **kwargs):
        raise AssertionError("the server must not start for --help")

    monkeypatch.setattr(server, "_apply_startup_instructions", _boom)
    monkeypatch.setattr(server.mcp, "run", _boom)
    server.main(["--help"])
    assert capsys.readouterr().out.startswith("comfy-mcp ")


def test_main_serves_when_given_no_flags(monkeypatch):
    """The default path is unchanged: no arguments still starts stdio."""
    calls = []
    monkeypatch.setattr(
        server, "_apply_startup_instructions", lambda: calls.append("i")
    )
    monkeypatch.setattr(server.mcp, "run", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["comfy-mcp"])
    server.main()
    assert calls == ["i", {"transport": "stdio"}]


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_entry_point_exits_zero_with_output(flag):
    """End-to-end through a real process: prints on stdout, exits 0.

    `python -m comfy_mcp.server` reaches the same `main()` the `comfy-mcp`
    console script does, so this covers the real `sys.argv` read and the exit
    status a user sees — without needing the script on PATH.

    `stdin` is `DEVNULL` so a regression fails fast: if flag interception ever
    stops firing, the child becomes a stdio MCP server, and on pytest's
    inherited stdin it would block for the whole timeout — and pass or hang
    depending on whether that handle happened to be at EOF.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_mcp.server", flag],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_ROOT,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("comfy-mcp ")
