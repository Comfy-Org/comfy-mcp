"""The `initialize` handshake reports the installed comfy-mcp version.

FastMCP defaults its version to ``None``; this application passes the installed
distribution version explicitly so clients can correlate reports with a release.

It has three layers on purpose:

* the WIRE check — a real `initialize` over a real stdio subprocess, the only
  thing that proves the string reaches a CLIENT rather than merely reaching a
  constructor;
* the SOURCE check — where that string comes from, including the fallback the
  wire check (which always runs installed) can never exercise;
* the framework check — that the constructor keyword remains ``version``.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import subprocess
import sys

import pytest
from fastmcp import FastMCP

import comfy_mcp
from comfy_mcp.server import _internal as server

# Bounds the stdio handshake below. Generous because it covers interpreter
# startup plus importing the SDK, not the handshake itself (milliseconds); a
# cold CI runner is the slow case.
_HANDSHAKE_TIMEOUT_S = 120.0

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "comfy-mcp-tests", "version": "0"},
    },
}


def _stdio_server_info() -> dict:
    """Run the real stdio server, send `initialize`, return its `serverInfo`."""
    proc = subprocess.Popen(
        # Via the same entry point the console script uses, since that script is
        # not guaranteed to be on PATH under every install layout.
        [sys.executable, "-c", "from comfy_mcp.server import main; main()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Point the startup machine-snapshot probe at a binary that cannot
        # exist. The probe is best-effort and falls open either way, but letting
        # it resolve a real `comfy` would make this test's runtime depend on
        # whether the machine has comfy-cli installed and how fast it answers.
        env={**os.environ, "COMFY_BIN": "/nonexistent/comfy-mcp-test-binary"},
    )
    try:
        # Closing stdin after the one request is what ends the server: it reads
        # EOF on its JSON-RPC channel and exits, so the success path never needs
        # a kill.
        stdout, stderr = proc.communicate(
            json.dumps(_INITIALIZE) + "\n", timeout=_HANDSHAKE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:  # pragma: no cover - only on a wedged server
        proc.kill()
        _, stderr = proc.communicate()
        pytest.fail(
            f"stdio server did not answer initialize in {_HANDSHAKE_TIMEOUT_S}s. "
            f"stderr: {stderr[-2000:]}"
        )
    for line in stdout.splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            # stdout is the JSON-RPC channel, but be forgiving about anything
            # else landing on it rather than turning noise into a confusing
            # failure.
            continue
        if message.get("id") == 1 and "result" in message:
            return message["result"]["serverInfo"]
    pytest.fail(
        f"no initialize response on stdout.\n"
        f"stdout: {stdout[:2000]}\nstderr: {stderr[-2000:]}"
    )


def test_initialize_reports_a_real_version_over_stdio():
    """The regression itself: a client saw `"version": ""`."""
    info = _stdio_server_info()
    assert info["name"] == "comfy-mcp"
    assert info["version"], "serverInfo.version is empty — a client has nothing to show"
    assert info["version"] == importlib.metadata.version("comfy-mcp")


def test_the_configured_version_is_the_installed_distribution_version():
    """Cheap, no-subprocess restatement of the above, for a fast failure."""
    assert server.mcp.version == importlib.metadata.version("comfy-mcp")


def test_version_falls_back_to_the_source_literal_without_distribution_metadata(
    monkeypatch,
):
    """A checkout run straight off `PYTHONPATH=src` has no installed metadata.

    Unreachable from the wire test, which always runs against an install — but
    it is the branch that keeps `_server_version` from raising during import and
    taking the whole server down, where a missing version used to merely look
    untidy.
    """

    def _absent(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _absent)
    assert server._server_version() == comfy_mcp.__version__


def test_unreadable_distribution_metadata_falls_back_instead_of_raising(
    monkeypatch, caplog
):
    """`_server_version` runs at import: raising would stop the server starting.

    A half-written or truncated `.dist-info` is the realistic way
    `importlib.metadata` fails with something other than
    `PackageNotFoundError`, and a display string is not worth a dead server.
    """

    def _broken(name):
        raise ValueError(f"unreadable metadata for {name}")

    monkeypatch.setattr(importlib.metadata, "version", _broken)
    with caplog.at_level("WARNING"):
        assert server._server_version() == comfy_mcp.__version__
    assert "could not read installed comfy-mcp metadata" in caplog.text


def test_metadata_without_a_version_field_falls_back(monkeypatch):
    """`Distribution.version` is `None` when the `Version:` field is missing.

    Passing that through would reproduce the exact empty `serverInfo.version`
    this whole file exists to prevent, one layer down.
    """
    monkeypatch.setattr(importlib.metadata, "version", lambda name: None)
    assert server._server_version() == comfy_mcp.__version__


def test_sdk_still_spells_the_handshake_version_argument_version():
    """A renamed SDK keyword would silently restore the empty default.

    Same reasoning as `test_sdk_conformance.py`. An outright rename would in
    fact fail loudly at import (`TypeError: unexpected keyword argument`); what
    this pins is the other half — that the DEFAULT is still the empty string
    this fix exists to displace, so an SDK that starts deriving a version does
    not leave the argument sitting here unexamined.
    """
    parameter = inspect.signature(FastMCP.__init__).parameters.get("version")
    assert parameter is not None, "FastMCP no longer takes a `version` argument"
    assert parameter.default is None
