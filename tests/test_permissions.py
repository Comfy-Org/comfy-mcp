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
from conftest import _FakeRunProc

from comfy_local_mcp import server, tcc

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
    monkeypatch.setattr(tcc.sys, "platform", "darwin")


@pytest.fixture
def on_linux(monkeypatch):
    monkeypatch.setattr(tcc.sys, "platform", "linux")


def _assert_actionable(message: str) -> None:
    """Every rewritten message must carry both escape hatches."""
    assert "Full Disk Access" in message
    assert "System Settings" in message
    assert "comfy set-default" in message


# --- path classification -----------------------------------------------------


@pytest.mark.parametrize("folder", ["Documents", "Desktop", "Downloads"])
def test_protected_dir_detects_each_folder(folder):
    path = os.path.join(os.path.expanduser("~"), folder, "ComfyUI", "venv", "bin")
    assert tcc._macos_protected_dir(path) == folder


def test_protected_dir_ignores_unprotected_and_prefix_collisions():
    home = os.path.expanduser("~")
    assert tcc._macos_protected_dir(os.path.join(home, "ComfyUI")) is None
    # "~/DocumentsArchive" merely starts with "~/Documents" — not the same folder.
    assert tcc._macos_protected_dir(os.path.join(home, "DocumentsArchive")) is None
    assert tcc._macos_protected_dir("/opt/comfy/venv") is None
    assert tcc._macos_protected_dir(None) is None


def test_protected_dir_matches_case_insensitively():
    """macOS volumes are case-insensitive by default: ~/downloads IS ~/Downloads."""
    path = os.path.join(os.path.expanduser("~"), "downloads", "ComfyUI")
    assert tcc._macos_protected_dir(path) == "Downloads"


def test_protected_dir_accepts_a_bytes_path():
    """An OSError from a bytes-path syscall carries a bytes `filename`."""
    path = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI")
    assert tcc._macos_protected_dir(os.fsencode(path)) == "Documents"


def test_guidance_names_the_folder_when_the_path_is_known(on_macos):
    message = tcc._tcc_guidance(_DENIED_PATH)
    assert "~/Documents" in message
    assert _DENIED_PATH in message
    _assert_actionable(message)


def test_guidance_stays_general_without_a_path(on_macos):
    message = tcc._tcc_guidance(None)
    # No location is asserted as fact; all three protected folders are listed.
    for folder in ("~/Documents", "~/Desktop", "~/Downloads"):
        assert folder in message
    _assert_actionable(message)


def test_denial_signature_is_macos_only(on_macos, monkeypatch):
    assert tcc._looks_like_tcc_denial(_FATAL_STDERR) is True
    assert tcc._looks_like_tcc_denial("connection refused") is False
    assert tcc._looks_like_tcc_denial("") is False
    monkeypatch.setattr(tcc.sys, "platform", "linux")
    assert tcc._looks_like_tcc_denial(_FATAL_STDERR) is False


def test_denial_survives_a_localized_strerror(on_macos):
    """`Operation not permitted` is libc text macOS translates; `[Errno 1]` isn't."""
    localized = (
        "Fatal Python error: init_import_site: Failed to import the site module\n"
        f"PermissionError: [Errno 1] Opération non permise: '{_DENIED_PATH}'\n"
    )
    assert tcc._looks_like_tcc_denial(localized) is True
    # …and the path still resolves, so the message names the folder.
    assert tcc._tcc_path_from(localized) == _DENIED_PATH
    assert "~/Documents" in tcc._tcc_guidance(tcc._tcc_path_from(localized))
    # A different errno must not be swept in by the `[Errno 1]` marker.
    assert (
        tcc._looks_like_tcc_denial("PermissionError: [Errno 13] Accès refusé") is False
    )


def test_eperm_alone_is_not_enough_to_claim_tcc(on_macos):
    """macOS raises EPERM for SIP, sandboxing and more — don't rewrite those.

    Without either corroborating signal (the startup-crash marker, or a denied
    path that really is under a protected folder) the original message stands.
    """
    sip = "OSError: [Errno 1] Operation not permitted: '/usr/lib/dyld'"
    assert tcc._looks_like_tcc_denial(sip) is False
    assert tcc._looks_like_tcc_denial("Operation not permitted") is False
    # …but a denied path inside a protected folder needs no startup marker.
    mid_run = f"OSError: [Errno 1] Operation not permitted: '{_DENIED_PATH}'"
    assert tcc._looks_like_tcc_denial(mid_run) is True


def test_denied_path_is_parsed_out_of_the_traceback():
    assert tcc._tcc_path_from(_FATAL_STDERR) == _DENIED_PATH
    assert tcc._tcc_path_from("no path here") is None


def test_denied_path_is_parsed_when_repr_used_double_quotes(on_macos):
    """`repr` switches to double quotes for a path containing an apostrophe."""
    path = os.path.join(
        os.path.expanduser("~"), "Documents", "Conan's App", "venv", "pyvenv.cfg"
    )
    text = f'PermissionError: [Errno 1] Operation not permitted: "{path}"'
    assert tcc._tcc_path_from(text) == path
    assert tcc._looks_like_tcc_denial(text) is True


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


def test_bare_comfy_bin_is_resolved_against_path_not_the_cwd(on_macos, monkeypatch):
    """The default `COMFY_BIN` is a bare name: a plain stat() would check $PWD.

    A `comfy` installed on PATH *inside* a protected folder is exactly the case
    this feature exists for, so it must be found by walking PATH.
    """
    protected_bin_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "ComfyUI", "venv", "bin"
    )
    monkeypatch.setattr(server, "COMFY_BIN", "comfy")
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        server.os, "get_exec_path", lambda: ["/usr/bin", protected_bin_dir]
    )

    def fake_stat(path):
        if path == os.path.join(protected_bin_dir, "comfy"):
            raise PermissionError(1, "Operation not permitted", path)
        raise FileNotFoundError()

    monkeypatch.setattr(server.os, "stat", fake_stat)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._require_comfy_bin()

    message = str(excinfo.value)
    assert "not found on PATH" not in message
    assert "~/Documents" in message
    _assert_actionable(message)


def test_an_unprotected_permission_failure_is_not_called_tcc(on_macos, monkeypatch):
    """A restrictive mode/ACL on a COMFY_BIN elsewhere is not a Full Disk Access
    problem — mislabelling it sends the user off fixing the wrong setting."""
    monkeypatch.setattr(server, "COMFY_BIN", "/opt/comfy/bin/comfy")
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        server.os,
        "stat",
        lambda p: (_ for _ in ()).throw(PermissionError(13, "Permission denied", p)),
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._require_comfy_bin()

    assert "Full Disk Access" not in str(excinfo.value)


def test_genuinely_missing_comfy_bin_keeps_the_install_message(on_macos, monkeypatch):
    monkeypatch.setattr(server, "COMFY_BIN", _DENIED_PATH)
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

    def fake_run(cmd, **kwargs):
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


def test_version_guard_translates_a_denied_spawn(on_macos, monkeypatch):
    """The probe can be denied at exec time, never reaching a returncode.

    The generic OSError handler would fail open here and let the raw EPERM
    escape unexplained from the real spawn a moment later.
    """
    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda cmd, **kw: (_ for _ in ()).throw(
            PermissionError(1, "Operation not permitted", _DENIED_PATH)
        ),
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._check_comfy_version()

    _assert_actionable(str(excinfo.value))
    assert "~/Documents" in str(excinfo.value)
    assert server._version_checked is False


def test_version_guard_fails_open_on_an_unrelated_denied_spawn(on_macos, monkeypatch):
    """A non-TCC permission failure keeps the pre-existing fail-OPEN behavior."""
    monkeypatch.setattr(server, "_version_checked", False)
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda cmd, **kw: (_ for _ in ()).throw(
            PermissionError(13, "Permission denied", "/opt/comfy/bin/comfy")
        ),
    )

    server._check_comfy_version()  # no raise
    assert server._version_checked is False  # not latched, exactly as before


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
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout="comfy-cli, version 1.13.0\n", stderr=""
        ),
    )
    server._check_comfy_version()
    assert server._version_checked is True


# --- the envelope path (a denial that reaches us mid-run) --------------------


def _patch_failing_run(monkeypatch, stderr: str) -> None:
    """A comfy-cli spawn that exits 1 with ``stderr`` and no envelope."""
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def fake_popen(cmd, **kwargs):
        return _FakeRunProc(
            cmd, {}, stdout="", stderr=stderr, returncode=1, raises=None
        )

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)


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
        "Popen",
        lambda cmd, **kw: _FakeRunProc(
            cmd, {}, stdout=envelope, stderr=_FATAL_STDERR, returncode=1, raises=None
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


def test_main_handles_a_bytes_filename(on_macos, monkeypatch, capsys):
    """A denial on a bytes path carries a bytes `filename` — decode, don't crash."""

    def boom():
        raise PermissionError(1, "Operation not permitted", os.fsencode(_DENIED_PATH))

    monkeypatch.setattr(server.mcp, "run", boom)

    with pytest.raises(SystemExit):
        server.main()

    message = capsys.readouterr().err
    assert "~/Documents" in message
    assert _DENIED_PATH in message  # decoded, not rendered as b'...'


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
