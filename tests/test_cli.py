"""The console script's `--help` / `--version` surface (`comfy_mcp.cli`).

A stdio server that ignores argv gives a first-time user nothing: `comfy-mcp
--help` starts the server, blocks on stdin, and exits silently at EOF, which is
indistinguishable from a broken install. These pin the two flags that answer
instead — and, just as importantly, that everything else still reaches the
server, since argv was previously ignored outright and a client registration is
free to pass arguments this program does not know about.
"""

from __future__ import annotations

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


def test_version_prefers_installed_metadata(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda name: f"9.9.9-{name}")
    assert cli._version() == "9.9.9-comfy-mcp"


def test_version_falls_back_to_the_source_literal(monkeypatch):
    """No `.dist-info` (a source tree that was never installed) still answers."""

    def _missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _missing)
    assert cli._version() == comfy_mcp.__version__


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
    """
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_mcp.server", flag],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("comfy-mcp ")
