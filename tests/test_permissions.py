"""macOS protected-folder (TCC) diagnostics.

A ComfyUI install under ~/Documents, ~/Desktop or ~/Downloads is unreadable by
an MCP client that lacks Full Disk Access — and by every process it spawns. The
`comfy` binary then dies inside CPython's own startup:

    Fatal Python error: init_import_site: Failed to import the site module
    PermissionError: [Errno 1] Operation not permitted: '.../venv/pyvenv.cfg'

These tests lock in that we recognize that signature wherever it can reach us
and answer with the fix (Full Disk Access, or move the folder) instead of
relaying the raw traceback — and, just as importantly, that we do NOT rewrite
unrelated failures or non-macOS ones.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from comfy_local_mcp import server

# The child's stderr when its venv sits in a TCC-protected folder.
_DENIED_PATH = os.path.join(
    os.path.expanduser("~"), "Documents", "ComfyUI", "venv", "pyvenv.cfg"
)
_FATAL_STDERR = (
    "Fatal Python error: init_import_site: Failed to import the site module\n"
    "Python runtime state: core initialized\n"
    "Traceback (most recent call last):\n"
    '  File "<frozen site>", line 762, in <module>\n'
    f"PermissionError: [Errno 1] Operation not permitted: '{_DENIED_PATH}'\n"
)


@pytest.fixture
def on_macos(monkeypatch):
    """Run the body as if on macOS (the diagnostics are macOS-only by design)."""
    monkeypatch.setattr(server.sys, "platform", "darwin")


@pytest.fixture
def on_linux(monkeypatch):
    monkeypatch.setattr(server.sys, "platform", "linux")


def _assert_actionable(message: str) -> None:
    """Every rewritten message must carry both escape hatches."""
    assert "Full Disk Access" in message
    assert "System Settings" in message
    assert "comfy set-default" in message


# --- path classification -----------------------------------------------------


@pytest.mark.parametrize("folder", ["Documents", "Desktop", "Downloads"])
def test_protected_dir_detects_each_folder(folder):
    path = os.path.join(os.path.expanduser("~"), folder, "ComfyUI", "venv", "bin")
    assert server._macos_protected_dir(path) == folder


def test_protected_dir_ignores_unprotected_and_prefix_collisions():
    home = os.path.expanduser("~")
    assert server._macos_protected_dir(os.path.join(home, "ComfyUI")) is None
    # "~/DocumentsArchive" merely starts with "~/Documents" — not the same folder.
    assert server._macos_protected_dir(os.path.join(home, "DocumentsArchive")) is None
    assert server._macos_protected_dir("/opt/comfy/venv") is None
    assert server._macos_protected_dir(None) is None


def test_guidance_names_the_folder_when_the_path_is_known(on_macos):
    message = server._tcc_guidance(_DENIED_PATH)
    assert "~/Documents" in message
    assert _DENIED_PATH in message
    _assert_actionable(message)


def test_guidance_stays_general_without_a_path(on_macos):
    message = server._tcc_guidance(None)
    # No location is asserted as fact; all three protected folders are listed.
    for folder in ("~/Documents", "~/Desktop", "~/Downloads"):
        assert folder in message
    _assert_actionable(message)


def test_denial_signature_is_macos_only(on_macos, monkeypatch):
    assert server._looks_like_tcc_denial(_FATAL_STDERR) is True
    assert server._looks_like_tcc_denial("connection refused") is False
    assert server._looks_like_tcc_denial("") is False
    monkeypatch.setattr(server.sys, "platform", "linux")
    assert server._looks_like_tcc_denial(_FATAL_STDERR) is False


def test_denied_path_is_parsed_out_of_the_traceback():
    assert server._tcc_path_from(_FATAL_STDERR) == _DENIED_PATH
    assert server._tcc_path_from("no path here") is None


# --- binary resolution -------------------------------------------------------


def test_unreadable_comfy_bin_reports_the_permission_not_a_missing_install(
    on_macos, monkeypatch
):
    """`shutil.which` returns None for BOTH cases — say which one it is."""
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        server.os, "stat", lambda _: (_ for _ in ()).throw(PermissionError(1, "EPERM"))
    )
    monkeypatch.setattr(server, "COMFY_BIN", _DENIED_PATH)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._require_comfy_bin()

    message = str(excinfo.value)
    assert "not found on PATH" not in message
    _assert_actionable(message)


def test_genuinely_missing_comfy_bin_keeps_the_install_message(on_macos, monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        server.os, "stat", lambda _: (_ for _ in ()).throw(FileNotFoundError())
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._require_comfy_bin()

    assert "not found on PATH" in str(excinfo.value)
    assert "Full Disk Access" not in str(excinfo.value)


def test_resolvable_comfy_bin_passes(on_macos, monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    server._require_comfy_bin()  # no raise


# --- the version guard (first shell-out of the process) ----------------------


def _patch_version_probe(monkeypatch, returncode: int, stderr: str) -> None:
    """Re-enable the memoized version guard and make `comfy --version` fail."""
    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def fake_run(cmd, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(server.subprocess, "run", fake_run)


def test_version_guard_translates_the_fatal_startup_error(on_macos, monkeypatch):
    """The earliest catchable point: `comfy --version` before the first tool call."""
    _patch_version_probe(monkeypatch, 1, _FATAL_STDERR)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._check_comfy_version()

    message = str(excinfo.value)
    _assert_actionable(message)
    assert "~/Documents" in message
    # The raw error is preserved, just demoted below the fix.
    assert "init_import_site" in message
    # Not memoized: granting access and retrying in-process must re-check.
    assert server._version_checked is False


def test_version_guard_leaves_a_non_macos_failure_alone(on_linux, monkeypatch):
    """EPERM means something else on Linux — no macOS guidance there."""
    _patch_version_probe(monkeypatch, 1, _FATAL_STDERR)
    server._check_comfy_version()  # fails open, exactly as before
    assert server._version_checked is True


def test_version_guard_ignores_a_healthy_comfy_cli(on_macos, monkeypatch):
    """A supported comfy-cli passes through untouched."""
    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(  # noqa: ARG005
            cmd, 0, stdout="comfy-cli, version 1.12.0\n", stderr=""
        ),
    )
    server._check_comfy_version()
    assert server._version_checked is True


# --- the envelope path (a denial that reaches us mid-run) --------------------


def _patch_failing_run(monkeypatch, stderr: str) -> None:
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def fake_run(cmd, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr)

    monkeypatch.setattr(server.subprocess, "run", fake_run)


def test_tool_call_surfaces_the_fix_instead_of_returned_no_json(on_macos, monkeypatch):
    _patch_failing_run(monkeypatch, _FATAL_STDERR)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env", timeout=1.0)

    message = str(excinfo.value)
    assert "returned no JSON" not in message
    _assert_actionable(message)
    assert "init_import_site" in message


def test_an_unrelated_failure_keeps_its_original_message(on_macos, monkeypatch):
    _patch_failing_run(monkeypatch, "Traceback: ConnectionRefusedError")

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("env", timeout=1.0)

    assert "returned no JSON" in str(excinfo.value)
    assert "Full Disk Access" not in str(excinfo.value)


def test_a_real_error_envelope_is_untouched(on_macos, monkeypatch):
    """An envelope means comfy-cli ran — its own error must not be reinterpreted."""
    envelope = json.dumps(
        {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": False,
            "error": {"code": "workflow_unknown_nodes", "message": "nope"},
        }
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(  # noqa: ARG005
            cmd, 1, stdout=envelope, stderr=_FATAL_STDERR
        ),
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._run_comfy("validate", timeout=1.0)

    assert excinfo.value.code == "workflow_unknown_nodes"
    assert "Full Disk Access" not in str(excinfo.value)


# --- process startup ---------------------------------------------------------


def test_main_reports_a_startup_denial_instead_of_a_traceback(
    on_macos, monkeypatch, capsys
):
    def boom():
        raise PermissionError(1, "Operation not permitted", _DENIED_PATH)

    monkeypatch.setattr(server.mcp, "run", boom)

    with pytest.raises(SystemExit) as excinfo:
        server.main()

    assert excinfo.value.code == 1
    message = capsys.readouterr().err
    _assert_actionable(message)
    assert "~/Documents" in message


def test_main_propagates_an_unrelated_permission_error(on_macos, monkeypatch):
    def boom():
        raise PermissionError(13, "Permission denied", "/etc/shadow")

    monkeypatch.setattr(server.mcp, "run", boom)

    with pytest.raises(PermissionError):
        server.main()


def test_main_propagates_on_non_macos(on_linux, monkeypatch):
    def boom():
        raise PermissionError(1, "Operation not permitted", _DENIED_PATH)

    monkeypatch.setattr(server.mcp, "run", boom)

    with pytest.raises(PermissionError):
        server.main()
