"""Tests for the comfy-cli compatibility gate.

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

import ast
import re
import subprocess
from pathlib import Path

import pytest

from comfy_mcp.server import _internal as server

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


def test_incompatible_envelope_propagates_through_run_comfy(patched_run):
    """The assertion guards every tool: a bad-schema envelope raises via _run_comfy."""
    # Deliberately NOT conftest's `envelope()` — the whole point is the major
    # this server does not speak, which that builder cannot emit.
    patched_run({"schema": "envelope/99", "type": "envelope", "ok": True})

    with pytest.raises(server.ComfyCliError, match="incompatible comfy-cli envelope"):
        server._run_comfy("env")


# --- version parsing / detection --------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("comfy-cli 1.13.0", (1, 13, 0)),
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

    def fake(cmd, capture_output, text, errors, timeout, check, cwd=None):
        assert cmd == [server.COMFY_BIN, "--version"]
        return subprocess.CompletedProcess(
            cmd, 0, stdout="comfy-cli, 1.13.0\n", stderr=""
        )

    monkeypatch.setattr(server.subprocess, "run", fake)
    assert server._detect_comfy_cli_version() == "1.13.0"


def test_detect_version_ignores_stderr_and_nonzero_exit(monkeypatch):
    """Only stdout on a clean exit is trusted; a stderr number / bad exit -> None."""
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def fake(cmd, capture_output, text, errors, timeout, check, cwd=None):
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
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.13.0")

    report = server._check_comfy_cli_version()

    assert report["comfy_cli_version"] == "1.13.0"
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
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.13.0")

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
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.13.0")

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
def patched_env(monkeypatch, patched_run):
    """Patch comfy-cli so ``server_info`` sees a given ``comfy env`` envelope."""

    def setup(payload: dict, version: str | None = "1.13.0") -> list[dict]:
        calls = patched_run(payload)
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
    assert compat["comfy_cli_version"] == "1.13.0"
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
        version="1.13.0",
    )

    with pytest.raises(server.ComfyCliError, match="older than the required minimum"):
        server.server_info()


def test_server_info_rejects_a_stray_non_envelope_json(patched_env, monkeypatch):
    """An incidental non-envelope JSON line from `comfy env` is not server info.

    `_run_comfy_raw` hands back `_last_json_object`'s answer, which falls back to
    ANY JSON object on stdout. `server_info` must apply the same `type==envelope`
    contract `_run_comfy` does — otherwise a diagnostic line that happens to
    carry `ok: true` would be reported as a valid environment report, and the
    no-envelope diagnostics (both stream tails) would never be shown.
    """
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    patched_env({"ok": True, "data": {"running": True}})

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.server_info()

    msg = str(excinfo.value)
    assert "returned no JSON" in msg
    assert "stdout: " in msg  # the raw line is surfaced, not swallowed


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
def patched_env_then_outdated(monkeypatch, patched_run_sequence):
    """Patch comfy-cli so ``server_info``'s two runs each see their own reply.

    ``server_info`` shells out twice — ``comfy env`` then ``comfy outdated`` —
    so unlike ``patched_env`` (same envelope every call) this queues ONE reply
    per call: each entry is a result-shaping tuple ``(returncode, stdout,
    stderr)`` or an exception instance to raise.

    The ordered subprocess implementation comes from the shared conftest
    fixture; this local fixture only configures the compatibility-specific
    version and floor state.
    """
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.13.0")
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    return patched_run_sequence


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

    `comfy outdated` ships in comfy-cli 1.13.0, below this server's floor, so
    a compliant install answers the probe and this is now the RARE path — it
    survives because the version guard fails OPEN, letting an install with an
    unparseable `--version` (a source build, a fork) reach here below the floor.
    It must still read as a capability gap rather than a failure of this server:
    no raw Click/Typer usage dump, no "returned no JSON" wrapper text, and a
    machine-readable `unsupported` flag so a client can branch without
    string-matching.
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
    user that nothing is broken — masking a genuine failure. The presence of an
    envelope at all proves the verb ran, so it must pass through.
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


def test_server_info_freshness_envelope_at_exit_two_is_not_unsupported(
    patched_env_then_outdated,
):
    """An error envelope at Click's usage-error status still proves the verb ran.

    Pins the `no_envelope` half of the gate INDEPENDENTLY of the exit code:
    every other envelope-present negative here exits 1, so the exit-code check
    alone would reject them and the provenance condition would go untested. A
    comfy-cli that HAS `outdated` can reject one of its options and report that
    structurally at exit 2 — only the absence of an envelope means the parser
    never dispatched.
    """
    import json

    error_envelope = {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": False,
        "error": {
            "code": "bad_option",
            "message": "No such command 'outdated' handler registered for --json",
        },
    }
    patched_env_then_outdated(
        [(0, json.dumps(_ENV_ENVELOPE), ""), (2, json.dumps(error_envelope), "")]
    )

    result = server.server_info()

    assert "unsupported" not in result["freshness"]
    assert "bad_option" in result["freshness"]["error"]


def test_server_info_freshness_dotted_command_is_not_unsupported(
    patched_env_then_outdated,
):
    """`No such command 'outdated.foo'` is a different command, not our verb.

    The lookahead must reject every character a command name can continue with,
    not just word characters and the hyphen — a `.` would otherwise let the
    `outdated` prefix match and discard a real diagnostic.
    """
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            (2, "", "Error: No such command 'outdated.foo'."),
        ]
    )

    result = server.server_info()

    assert "unsupported" not in result["freshness"]
    assert "outdated.foo" in result["freshness"]["error"]


def test_server_info_freshness_midrun_crash_is_not_unsupported(
    patched_env_then_outdated,
):
    """A crash in a verb comfy-cli DID accept keeps its raw diagnostic.

    Emitting no envelope only proves comfy-cli died before it could report
    structurally — a recognized verb can do that too by crashing mid-run. Click
    exits 2 only when its parser rejected the command line before dispatch, so a
    no-envelope failure at any other status is a real failure, even when its
    output happens to quote the missing-verb phrase.
    """
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            (1, "", "Traceback...\nRuntimeError: No such command 'outdated' in hook"),
        ]
    )

    result = server.server_info()

    assert "unsupported" not in result["freshness"]
    assert "Traceback" in result["freshness"]["error"]
    assert "freshness unavailable" not in result["freshness"]["error"]


def test_server_info_freshness_missing_verb_survives_colon_sgr(
    patched_env_then_outdated,
):
    """Colon-separated SGR (`\\x1b[38:5:130m`) is stripped like any other CSI.

    ECMA-48 allows `:<=>` in CSI parameter bytes, and terminals do emit the
    colon form for true-colour. A parameter class of only `[0-9;]` would leave
    those bytes in place, defeating the match and leaking the raw usage dump.
    """
    import json

    coloured = (
        "\x1b[38:5:130mError\x1b[0m: No such \x1b[38:2:255:0:0mcommand\x1b[0m "
        "'outdated'."
    )
    patched_env_then_outdated([(0, json.dumps(_ENV_ENVELOPE), ""), (2, "", coloured)])

    result = server.server_info()

    assert result["freshness"]["unsupported"] is True
    assert "freshness unavailable" in result["freshness"]["error"]


def test_server_info_freshness_missing_verb_on_stdout(patched_env_then_outdated):
    """The usage error degrades cleanly whichever stream Click wrote it to.

    `_unwrap_envelope` renders both streams because comfy-cli splits its
    diagnostics unpredictably — a Typer/click usage error can land on stdout
    rather than stderr. The missing-verb classifier reads the raised message, so
    it must not care which half carried the text.
    """
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            (2, "Usage: comfy [OPTIONS] COMMAND\nNo such command 'outdated'.", ""),
        ]
    )

    result = server.server_info()

    assert result["freshness"]["unsupported"] is True
    assert "freshness unavailable" in result["freshness"]["error"]


def test_server_info_freshness_codeless_envelope_is_not_unsupported(
    patched_env_then_outdated,
):
    """An error envelope that OMITS `error.code` still proves the verb ran.

    A null `code` is not evidence of a missing verb — `_unwrap_envelope` leaves
    it `None` whenever an otherwise well-formed error envelope has no `code`
    field. Gating on `code is None` would misread this genuine failure as the
    benign capability gap; `no_envelope` is the signal that actually holds.
    """
    import json

    codeless_envelope = {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": False,
        "error": {"message": "No such command 'outdated' in the pack's hook script"},
    }
    patched_env_then_outdated(
        [(0, json.dumps(_ENV_ENVELOPE), ""), (1, json.dumps(codeless_envelope), "")]
    )

    result = server.server_info()

    assert "unsupported" not in result["freshness"]
    assert "hook script" in result["freshness"]["error"]
    assert "freshness unavailable" not in result["freshness"]["error"]


def test_server_info_freshness_hyphenated_command_is_not_unsupported(
    patched_env_then_outdated,
):
    """`No such command 'outdated-notifier'` is a DIFFERENT command, not our verb.

    A trailing `\\b` would treat the hyphen as a word boundary and match the
    `outdated` prefix, discarding a real diagnostic about some other command.
    """
    import json

    patched_env_then_outdated(
        [
            (0, json.dumps(_ENV_ENVELOPE), ""),
            (2, "", "Error: No such command 'outdated-notifier'."),
        ]
    )

    result = server.server_info()

    assert "unsupported" not in result["freshness"]
    assert "outdated-notifier" in result["freshness"]["error"]


def test_server_info_freshness_missing_verb_survives_ansi_colour(
    patched_env_then_outdated,
):
    """Colourized rich output still degrades cleanly.

    Rich styles its error panel with ANSI escapes, whose bytes include word
    characters (digits, `m`). Left in place they land between the matched words
    and defeat the pattern, leaking the raw usage dump.
    """
    import json

    coloured = (
        "\x1b[31mError\x1b[0m: \x1b[1mNo such\x1b[0m \x1b[1mcommand\x1b[0m "
        "\x1b[33m'outdated'\x1b[0m."
    )
    patched_env_then_outdated([(0, json.dumps(_ENV_ENVELOPE), ""), (2, "", coloured)])

    result = server.server_info()

    assert result["freshness"]["unsupported"] is True
    assert "freshness unavailable" in result["freshness"]["error"]


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


# --- floor-vs-prose consistency (the desync that outlives a floor raise) -----

# Matches a capability claim that hedges a version: "requires a comfy-cli NEWER
# than 1.13.0", "the verb landed after comfy-cli 1.13.0", "landed in comfy-cli
# after 1.13.0", "ships in releases after 1.13.0". The captured version is what
# the prose says you need to be ABOVE, so it is a bug whenever it is at or below
# the floor.
#
# The `comfy-cli` prefix stays OPTIONAL because the live offenders mostly omit it
# ("the verb ships in releases after 1.14.0" named no product at all). The cost
# of that reach is a false positive on a hedge about some OTHER component's
# version — ComfyUI's own, which `switch_comfyui_version` discusses in 0.x
# numbers that trivially sort below the 1.14.0 floor — so `lead` captures the
# sentence fragment in front of the hedge and `_hedged_versions` drops a match
# whose subject is ComfyUI rather than comfy-cli. A spurious failure here blocks
# unrelated PRs, so the guard has to be able to tell the two apart.
_HEDGE_RE = re.compile(
    r"(?P<lead>[^.\n]{0,80}?)"
    r"(?:newer than|(?:landed|shipped|ships?)\s+(?:in\s+)?(?:comfy-cli\s+)?"
    r"(?:releases\s+)?after|releases after)"
    r"\s+(?:comfy-cli\s+)?v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?",
    re.IGNORECASE,
)

# A hint that interpolated the floor would render this phrase; it must not appear.
_MIN_STR_INTERPOLATION_LEAKED = f"NEWER than {server._MIN_COMFY_CLI_STR}"


def _hedged_versions(text: str) -> list[tuple[int, int, int]]:
    """Every version a "you need something newer than this" phrase names."""
    found = []
    for match in _HEDGE_RE.finditer(text):
        subject = f"{match.group('lead')}{match.group(0)}".lower()
        # "a ComfyUI newer than 0.24.0" is a claim about the app, not the CLI.
        if "comfyui" in subject and "comfy-cli" not in subject:
            continue
        found.append(
            (
                int(match.group("major")),
                int(match.group("minor")),
                int(match.group("patch") or 0),
            )
        )
    return found


def _rendered_strings(source: str) -> list[str]:
    """Every string literal in ``source`` as PYTHON will build it at runtime.

    The raw-source scan below cannot see the shape that actually shipped the bug
    this guard exists for. A degrade message is written as several adjacent
    literals — implicit concatenation — with the floor interpolated:

        "... (the verb ships in releases "
        f"after {_MIN_COMFY_CLI_STR}). Nothing else is affected. "

    In the source text there are no digits after "releases after" (just a brace),
    and the phrase is split across a ``" \\n f"`` seam that no ``\\s+`` can
    bridge — so the regex is structurally blind to it while the caller reads
    "ships in releases after 1.14.0". Parsing instead of grepping fixes both at
    once: the parser joins implicit concatenation for us, and a
    ``{_MIN_COMFY_CLI_STR}`` placeholder is resolved to the floor it renders as.
    Any other interpolation becomes ``{…}``, which no version pattern matches —
    this guard only claims to catch the floor leaking into a hedge.
    """

    def render(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(render(part) for part in node.values)
        if isinstance(node, ast.FormattedValue):
            expr = node.value
            if isinstance(expr, ast.Name) and expr.id == "_MIN_COMFY_CLI_STR":
                return server._MIN_COMFY_CLI_STR
            return "{…}"
        return "{…}"

    return [
        render(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.JoinedStr)
        or (isinstance(node, ast.Constant) and isinstance(node.value, str))
    ]


def test_the_hedge_detector_actually_detects():
    """Self-test: a clean sweep below must mean "clean", not "regex rotted"."""
    assert _hedged_versions("requires a comfy-cli NEWER than 1.13.0") == [(1, 13, 0)]
    assert _hedged_versions("the verb landed after comfy-cli 1.13.0") == [(1, 13, 0)]
    assert _hedged_versions("landed in comfy-cli after 1.13.0") == [(1, 13, 0)]
    assert _hedged_versions("ships in releases after 1.9") == [(1, 9, 0)]
    assert _hedged_versions("the TTL shipped in v1.14.0") == []
    # A hedge about ComfyUI's own version is not a comfy-cli claim, and its 0.x
    # numbering would otherwise sort below the floor and fail CI for nothing.
    assert _hedged_versions("pin a ComfyUI newer than 0.24.0") == []
    assert _hedged_versions("needs comfy-cli newer than 1.13.0 to drive ComfyUI") == [
        (1, 13, 0)
    ]


def test_the_rendered_string_reader_sees_through_concat_and_interpolation():
    """Self-test for the renderer: the blind spot must actually be covered.

    Both halves matter. Joining without resolving leaves a brace where the
    version goes; resolving without joining leaves the phrase split at the seam.
    Only doing both reproduces what the caller reads.
    """
    source = (
        'x = ("the verb ships in releases "\n'
        '     f"after {_MIN_COMFY_CLI_STR}). Nothing else is affected.")\n'
    )
    rendered = _rendered_strings(source)
    joined = [s for s in rendered if "releases after" in s]
    assert joined, f"implicit concatenation not joined: {rendered}"
    assert _hedged_versions(joined[0]) == [server._MIN_COMFY_CLI]
    # Any other interpolation stays opaque rather than being guessed at.
    assert "{…}" in "".join(_rendered_strings('y = f"ships after {some_other_var}"'))


def test_no_capability_claim_hedges_a_version_the_floor_already_covers():
    """No docstring may say "you need a comfy-cli newer than X" for X <= floor.

    This is the class of staleness a floor raise creates and nothing else
    catches. Every such sentence was TRUE when written — the verb really did land
    after the then-current floor — and every one of them silently became either
    vacuous or false the moment the floor moved to the release carrying the verb.
    An agent reading "node_dependencies requires a comfy-cli newer than 1.13.0"
    on a server that refuses to start below 1.14.0 is being told to upgrade to
    something it already has.

    The fix at each site is to name the release that HAS the capability, spelled
    out rather than interpolated from `_MIN_COMFY_CLI_STR` (which is the FLOOR,
    so interpolating it re-creates exactly this contradiction), and to say what
    the degrade really protects against now: a build that slipped past the
    fail-OPEN version guard, or a dependency outside comfy-cli.

    Two passes, because the offenders come in two shapes. The RAW source pass
    covers prose that is literally in the file — comments and docstrings, which
    the parser does not reassemble. The RENDERED pass (`_rendered_strings`)
    covers the shape that actually ships to a caller: a degrade message built
    from adjacent literals with the floor interpolated, which the raw pass is
    structurally blind to (no digits in the source, and the phrase split across
    the concatenation seam). Without the second pass this test asserted only
    that nobody typed the contradiction by hand.
    """
    source = Path(server.__file__).read_text(encoding="utf-8")
    hedged = _hedged_versions(source)
    for rendered in _rendered_strings(source):
        hedged.extend(_hedged_versions(rendered))
    offenders = [v for v in hedged if v <= server._MIN_COMFY_CLI]
    assert offenders == [], (
        f"{len(offenders)} capability claim(s) in server/_internal.py hedge a comfy-cli "
        f"version at or below the {server._MIN_COMFY_CLI_STR} floor "
        f"({', '.join('.'.join(map(str, v)) for v in offenders)}) — the floor "
        "already guarantees it, so the sentence is vacuous or false. Name the "
        "release that carries the capability instead."
    )


def test_the_spelled_out_upgrade_hints_name_a_release_the_floor_guarantees():
    """The two hand-written upgrade hints must not outlive the floor either.

    `_RESOURCE_VERB_UPGRADE_HINT` and `_LOG_PORT_UPGRADE_HINT` deliberately spell
    their version out instead of interpolating `_MIN_COMFY_CLI_STR`, because that
    constant is the floor and these sentences are about the release a verb landed
    in. That is correct, and it is also exactly why they cannot be reflowed
    automatically when the floor moves — so pin them here. Both name 1.14.0,
    which the floor covers, so the message reads as "this build is not a
    published release" rather than "upgrade to what you already run".
    """
    for hint in (server._RESOURCE_VERB_UPGRADE_HINT, server._LOG_PORT_UPGRADE_HINT):
        assert "1.14.0" in hint
        assert _MIN_STR_INTERPOLATION_LEAKED not in hint
        assert _hedged_versions(hint) == []


def test_the_degrade_messages_name_the_release_that_carries_their_verb():
    """The two version-naming degrade payloads are pinned for the same reason.

    Neither is a module constant, so the test above cannot reach them — they are
    built inline where the degrade returns. Both are read ONLY by an install that
    got past the fail-OPEN version guard from BELOW the floor, so each has to
    name the release that actually carries its verb rather than the floor.

    They point in OPPOSITE directions from that floor, which is precisely why no
    single interpolated constant can serve both: `workflow notes` ships AT the
    floor (1.14.0), while `comfy outdated` shipped one release BELOW it (1.13.0),
    and a 1.13.x source build reading "upgrade to >= 1.14.0" for `outdated` is
    being told to clear a bar it does not need to. The hedge guard above catches
    neither mistake — ">= 1.14.0" hedges nothing — so pin the versions here.
    """
    rendered = _rendered_strings(Path(server.__file__).read_text(encoding="utf-8"))
    for prefix, expected in (
        ("workflow notes unavailable", "1.14.0 and newer"),
        ("freshness unavailable", "comfy-cli 1.13.0 and newer"),
    ):
        matches = [text for text in rendered if text.startswith(prefix)]
        assert matches, f"no {prefix!r} degrade message found in server/_internal.py"
        # `ast.walk` yields the joined string AND the leading fragment it was
        # concatenated from, so take the longest — the fully assembled message.
        assert expected in max(matches, key=len)
