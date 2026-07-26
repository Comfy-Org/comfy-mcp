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

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
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

        def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
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


# --- server_info freshness block (`comfy outdated`) --------------------------


_ENV_ENVELOPE = {
    "schema": "envelope/1",
    "type": "envelope",
    "ok": True,
    "data": {"running": True, "url": "http://127.0.0.1:8188"},
}

_OUTDATED_DATA = {
    "core": {
        "installed": "v0.3.40",
        "commit": "abc1234",
        "latest": "v0.3.41",
        "outdated": True,
    },
    "packs": [
        {
            "name": "comfyui-seedream",
            "source": "registry",
            "installed": "1.0.0",
            "latest": "1.2.0",
            "outdated": True,
        }
    ],
    "checked_at": "2026-07-22T00:00:00Z",
}


@pytest.fixture
def patched_env_then_outdated(monkeypatch):
    """Patch comfy-cli so ``server_info``'s two runs each see their own reply.

    ``server_info`` shells out twice — ``comfy env`` then ``comfy outdated`` —
    so unlike ``patched_env`` (same envelope every call) this queues ONE reply
    per call: each entry is a ``CompletedProcess``-shaping tuple
    ``(returncode, stdout, stderr)`` or an exception instance to raise.
    """

    def setup(replies: list) -> list[dict]:
        calls: list[dict] = []

        def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
            calls.append({"cmd": cmd, "timeout": timeout})
            reply = replies[len(calls) - 1]
            if isinstance(reply, BaseException):
                raise reply
            returncode, stdout, stderr = reply
            return subprocess.CompletedProcess(
                cmd, returncode, stdout=stdout, stderr=stderr
            )

        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(server.subprocess, "run", fake)
        monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.12.0")
        monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
        return calls

    return setup


def test_server_info_attaches_freshness_block(patched_env_then_outdated):
    """The `comfy outdated` payload passes through as the `freshness` key."""
    import json

    outdated_envelope = {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": True,
        "data": _OUTDATED_DATA,
    }
    calls = patched_env_then_outdated(
        [(0, json.dumps(_ENV_ENVELOPE), ""), (0, json.dumps(outdated_envelope), "")]
    )

    result = server.server_info()

    assert result["running"] is True  # comfy env data preserved
    assert result["freshness"] == _OUTDATED_DATA  # verbatim passthrough
    assert result["freshness"]["core"]["outdated"] is True
    # Second spawn is `comfy --json --where local outdated` with the 15s budget.
    assert calls[1]["cmd"][4:] == ["outdated"]
    assert calls[1]["timeout"] == 15.0


def test_server_info_freshness_degrades_on_missing_verb(patched_env_then_outdated):
    """A comfy-cli without `outdated` -> the purpose-built `unsupported` degrade.

    Every released comfy-cli (through 1.12.0) lacks the verb, so this is the
    COMMON path, not an edge case. It must read as a capability gap rather than
    a failure of this server: no raw Click/Typer usage dump, no "returned no
    JSON" wrapper text, and a machine-readable `unsupported` flag so a client
    can branch without string-matching.
    """
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            (2, "", "Usage: comfy [OPTIONS] COMMAND\nNo such command 'outdated'."),
        ]
    )

    result = server.server_info()

    assert result["running"] is True  # env data intact — the probe never breaks it
    assert "compatibility" in result
    assert result["freshness"]["unsupported"] is True
    assert "freshness unavailable" in result["freshness"]["error"]
    # The raw wrapper/CLI text no longer leaks through.
    assert "returned no JSON" not in result["freshness"]["error"]
    assert "No such command" not in result["freshness"]["error"]
    assert "Usage: comfy" not in result["freshness"]["error"]


def test_server_info_freshness_missing_verb_match_is_case_insensitive(
    patched_env_then_outdated,
):
    """`Error: no such command 'outdated'` (lowercase) also degrades cleanly."""
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            (2, "", "Error: no such command 'outdated'"),
        ]
    )

    result = server.server_info()

    assert result["freshness"]["unsupported"] is True
    assert "freshness unavailable" in result["freshness"]["error"]


def test_server_info_freshness_missing_verb_survives_panel_wrapping(
    patched_env_then_outdated,
):
    """A rich panel that wraps the phrase mid-line still degrades cleanly.

    Typer renders errors inside a bordered rich panel and wraps at the terminal
    width, so `No such command` and `'outdated'` can land on separate lines with
    box-drawing glyphs between them. Matching a literal one-line phrase would
    miss this and leak the raw usage dump — the exact outcome this degrade
    exists to prevent.
    """
    import json

    wrapped_panel = (
        "╭─ Error ─────────────────────────╮\n"
        "│ No such command\n"
        "│ 'outdated'.                     │\n"
        "╰─────────────────────────────────╯"
    )
    patched_env_then_outdated(
        [(0, json.dumps(_ENV_ENVELOPE), ""), (2, "", wrapped_panel)]
    )

    result = server.server_info()

    assert result["freshness"]["unsupported"] is True
    assert "freshness unavailable" in result["freshness"]["error"]


def test_server_info_freshness_relayed_phrase_is_not_unsupported(
    patched_env_then_outdated,
):
    """A REAL failure relaying "no such command" keeps its raw diagnostic.

    A comfy-cli that HAS the verb can fail with an error envelope whose message
    quotes a nested tool's own "no such command" (a git/pip call, a custom-node
    pack's install script, a relayed registry response). Classifying that as
    `unsupported` would tell the agent to skip staleness advice and reassure the
    user that nothing is broken — masking a genuine failure. The envelope's
    structured `error.code` proves the verb ran, so it must pass through.
    """
    import json

    error_envelope = {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": False,
        "error": {
            "code": "pack_probe_failed",
            "message": "git: 'no such command' while probing pack 'outdated-notifier'",
        },
    }
    patched_env_then_outdated(
        [(0, json.dumps(_ENV_ENVELOPE), ""), (1, json.dumps(error_envelope), "")]
    )

    result = server.server_info()

    assert "unsupported" not in result["freshness"]
    assert "pack_probe_failed" in result["freshness"]["error"]
    assert "freshness unavailable" not in result["freshness"]["error"]


def test_server_info_freshness_unknown_other_verb_is_not_unsupported(
    patched_env_then_outdated,
):
    """ "No such command" naming a DIFFERENT verb is not our capability gap."""
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            (2, "", "Error: No such command 'git-lfs'."),
        ]
    )

    result = server.server_info()

    assert "unsupported" not in result["freshness"]
    assert "git-lfs" in result["freshness"]["error"]


def test_server_info_freshness_passes_through_other_errors(patched_env_then_outdated):
    """A NON-missing-verb spawn failure keeps the raw reason and no `unsupported`.

    The special case above is deliberately narrow: for a network failure the raw
    reason IS the diagnostic, so it must still reach the caller verbatim.
    """
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            (1, "", "network unreachable"),
        ]
    )

    result = server.server_info()

    assert result["running"] is True
    assert set(result["freshness"]) == {"error"}  # no `unsupported` key
    assert "unsupported" not in result["freshness"]
    assert "network unreachable" in result["freshness"]["error"]
    assert "freshness unavailable" not in result["freshness"]["error"]


def test_server_info_freshness_degrades_on_error_envelope(patched_env_then_outdated):
    """A structured `comfy outdated` failure (e.g. network) -> freshness.error."""
    import json

    error_envelope = {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": False,
        "error": {"code": "network_error", "message": "registry lookup failed"},
    }
    patched_env_then_outdated(
        [(0, json.dumps(_ENV_ENVELOPE), ""), (1, json.dumps(error_envelope), "")]
    )

    result = server.server_info()

    assert result["running"] is True
    assert "registry lookup failed" in result["freshness"]["error"]


def test_server_info_freshness_degrades_on_timeout(patched_env_then_outdated):
    """A hung `comfy outdated` -> freshness.error, not a server_info timeout raise."""
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            subprocess.TimeoutExpired(cmd=["comfy", "outdated"], timeout=15.0),
        ]
    )

    result = server.server_info()

    assert result["running"] is True
    assert "timed out" in result["freshness"]["error"]


def test_server_info_freshness_degrades_on_decode_error(patched_env_then_outdated):
    """Non-UTF-8 bytes from `comfy outdated` -> freshness.error, not a raise.

    `_run_comfy_raw` decodes the child's stdout with strict `encoding="utf-8"`
    (no `errors="replace"`), so a pack name/path with non-UTF-8 bytes in the
    user's live custom-node install can make `subprocess.run` itself raise
    `UnicodeDecodeError` — a `ValueError` subclass, not `OSError`/`ComfyCliError`.
    """
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        ]
    )

    result = server.server_info()

    assert result["running"] is True  # env data intact — the probe never breaks it
    assert set(result["freshness"]) == {"error"}
    assert "invalid start byte" in result["freshness"]["error"]


def test_server_info_docstring_teaches_freshness_and_update_commands():
    """The tool docstring documents `freshness` + the exact update commands."""
    doc = server.server_info.__doc__ or ""
    assert "freshness" in doc
    assert "comfy update comfy" in doc
    assert "comfy node update" in doc
