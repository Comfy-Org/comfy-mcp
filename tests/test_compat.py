"""Tests for the comfy-cli compatibility gate (BE-2997).

comfy-cli is unpinned — it comes from PATH at whatever version — so the whole
contract (envelope shape, flag ordering, error codes) rests on an unversioned
dependency. These lock in the runtime checks that guard it:

1. ``_unwrap_envelope`` refuses an envelope whose declared ``schema`` major
   differs from the ``envelope/N`` this server parses, and passes an envelope
   that declares no schema (backward compatible with older comfy-cli builds).
2. ``server_info`` reports a ``compatibility`` block and, when a
   ``COMFY_CLI_MIN_VERSION`` floor is configured, rejects an older comfy-cli.
"""

from __future__ import annotations

import subprocess

import pytest

from comfy_local_mcp import server


# --- envelope-version assertion (_unwrap_envelope) --------------------------


def test_unwrap_accepts_matching_envelope_major():
    """An ``envelope/1`` schema (the version we speak) unwraps to ``data``."""
    envelope = {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": True,
        "data": {"x": 1},
    }
    assert server._unwrap_envelope(envelope, ("env",), 0, "") == {"x": 1}


def test_unwrap_accepts_schemaless_envelope():
    """An envelope with no ``schema`` is assumed compatible (older comfy-cli)."""
    envelope = {"type": "envelope", "ok": True, "data": {"x": 1}}
    assert server._unwrap_envelope(envelope, ("env",), 0, "") == {"x": 1}


def test_unwrap_rejects_incompatible_envelope_major():
    """A future ``envelope/2`` is a breaking contract change -> refused loudly."""
    envelope = {"schema": "envelope/2", "type": "envelope", "ok": True, "data": {}}
    with pytest.raises(server.ComfyCliError, match="incompatible comfy-cli envelope"):
        server._unwrap_envelope(envelope, ("env",), 0, "")


@pytest.mark.parametrize("schema", ["envelope-v2", "envelope/1-foo", "v2", "envelope/"])
def test_unwrap_rejects_declared_but_unparseable_schema(schema):
    """A declared schema that isn't a bare ``envelope/<N>`` fails closed, not open."""
    envelope = {"schema": schema, "type": "envelope", "ok": True, "data": {"x": 1}}
    with pytest.raises(server.ComfyCliError, match="incompatible comfy-cli envelope"):
        server._unwrap_envelope(envelope, ("env",), 0, "")


def test_unwrap_rejects_incompatible_major_even_on_error_envelope():
    """An incompatible schema is reported as such, not as its (untrusted) error."""
    envelope = {
        "schema": "envelope/2",
        "type": "envelope",
        "ok": False,
        "error": {"code": "whatever", "message": "boom"},
    }
    with pytest.raises(server.ComfyCliError, match="incompatible comfy-cli envelope"):
        server._unwrap_envelope(envelope, ("env",), 0, "")


def test_incompatible_envelope_propagates_through_run_comfy(monkeypatch):
    """The assertion guards every tool: a bad-schema envelope raises via _run_comfy."""

    def fake(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
        import json

        out = json.dumps({"schema": "envelope/99", "type": "envelope", "ok": True})
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match="incompatible comfy-cli envelope"):
        server._run_comfy("env")


# --- version parsing / detection --------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("comfy-cli 1.12.0", (1, 12, 0)),
        ("comfy version 1.5", (1, 5, 0)),
        ("v2.0.3\n", (2, 0, 3)),
        ("no version here", None),
    ],
)
def test_parse_version(text, expected):
    assert server._parse_version(text) == expected


def test_detect_version_none_when_binary_missing(monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    assert server._detect_comfy_cli_version() is None


def test_detect_version_parses_cli_output(monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def fake(cmd, capture_output, text, errors, timeout, check):  # noqa: ARG001
        assert cmd == [server.COMFY_BIN, "--version"]
        return subprocess.CompletedProcess(
            cmd, 0, stdout="comfy-cli, 1.12.0\n", stderr=""
        )

    monkeypatch.setattr(server.subprocess, "run", fake)
    assert server._detect_comfy_cli_version() == "1.12.0"


def test_detect_version_ignores_stderr_and_nonzero_exit(monkeypatch):
    """Only stdout on a clean exit is trusted; a stderr number / bad exit -> None."""
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def fake(cmd, capture_output, text, errors, timeout, check):  # noqa: ARG001
        # Non-zero exit, and a misleading dotted number only on stderr.
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="Python 3.11.7 error\n"
        )

    monkeypatch.setattr(server.subprocess, "run", fake)
    assert server._detect_comfy_cli_version() is None


def test_detect_version_none_on_subprocess_error(monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def boom(*a, **k):
        raise OSError("cannot exec")

    monkeypatch.setattr(server.subprocess, "run", boom)
    assert server._detect_comfy_cli_version() is None


# --- version floor (_check_comfy_cli_version) -------------------------------


def test_check_reports_version_without_floor(monkeypatch):
    """No floor configured: report the version, no warning, never raise."""
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.12.0")

    report = server._check_comfy_cli_version()

    assert report["comfy_cli_version"] == "1.12.0"
    assert report["min_comfy_cli_version"] is None
    assert report["envelope_schema_major"] == server.ENVELOPE_SCHEMA_MAJOR
    assert report["warnings"] == []


def test_check_warns_when_version_unknown_and_no_floor(monkeypatch):
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: None)

    report = server._check_comfy_cli_version()

    assert report["comfy_cli_version"] is None
    assert report["warnings"]  # a non-empty warning about the unknown version


def test_check_passes_when_version_meets_floor(monkeypatch):
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", "1.5.0")
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.12.0")

    report = server._check_comfy_cli_version()

    assert report["min_comfy_cli_version"] == "1.5.0"
    assert report["warnings"] == []


def test_check_raises_when_version_below_floor(monkeypatch):
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", "1.10.0")
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.5.0")

    with pytest.raises(server.ComfyCliError, match="older than the required minimum"):
        server._check_comfy_cli_version()


@pytest.mark.parametrize("bad_floor", ["2", "latest", "v-next"])
def test_check_warns_when_floor_is_unparseable(monkeypatch, bad_floor):
    """A misconfigured floor must not silently no-op: warn instead of failing open."""
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", bad_floor)
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.12.0")

    report = server._check_comfy_cli_version()

    assert report["warnings"]  # warns the floor was unparseable / not enforced
    assert any("not a parseable" in w for w in report["warnings"])


def test_check_warns_but_does_not_raise_when_floor_set_and_version_unknown(monkeypatch):
    """An undetectable version never hard-fails, even with a floor configured."""
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", "1.10.0")
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: None)

    report = server._check_comfy_cli_version()

    assert report["comfy_cli_version"] is None
    assert report["warnings"]  # warns that the floor could not be verified


# --- server_info integration ------------------------------------------------


@pytest.fixture
def patched_env(monkeypatch):
    """Patch comfy-cli so ``server_info`` sees a given ``comfy env`` envelope."""

    def setup(envelope: dict, version: str | None = "1.12.0") -> list[dict]:
        import json

        calls: list[dict] = []

        def fake(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
            calls.append({"cmd": cmd})
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(envelope), stderr=""
            )

        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(server.subprocess, "run", fake)
        monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: version)
        return calls

    return setup


def test_server_info_attaches_compatibility_block(patched_env, monkeypatch):
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    calls = patched_env(
        {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": True,
            "data": {"running": True, "url": "http://127.0.0.1:8188"},
        }
    )

    result = server.server_info()

    assert result["running"] is True  # original comfy env data preserved
    compat = result["compatibility"]
    assert compat["comfy_cli_version"] == "1.12.0"
    assert compat["envelope_schema"] == "envelope/1"
    assert compat["envelope_schema_major"] == server.ENVELOPE_SCHEMA_MAJOR
    assert calls[0]["cmd"][4:] == ["env"]  # still `comfy env`


def test_server_info_raises_on_incompatible_envelope(patched_env, monkeypatch):
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    patched_env({"schema": "envelope/2", "type": "envelope", "ok": True, "data": {}})

    with pytest.raises(server.ComfyCliError, match="incompatible comfy-cli envelope"):
        server.server_info()


def test_server_info_raises_on_version_below_floor(patched_env, monkeypatch):
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", "2.0.0")
    patched_env(
        {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": True,
            "data": {"running": False},
        },
        version="1.12.0",
    )

    with pytest.raises(server.ComfyCliError, match="older than the required minimum"):
        server.server_info()


def test_server_info_wraps_non_dict_env_data(patched_env, monkeypatch):
    """If `comfy env` ever returns non-dict data, it is still returned with compat."""
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    patched_env({"schema": "envelope/1", "type": "envelope", "ok": True, "data": "raw"})

    result = server.server_info()

    assert result["env"] == "raw"
    assert result["compatibility"]["envelope_schema"] == "envelope/1"
