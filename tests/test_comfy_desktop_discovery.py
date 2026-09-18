"""Tests for the Comfy Desktop port-lock discovery fallback.

Comfy Desktop (the Electron app) picks its port DYNAMICALLY per launch —
falling back to another one when its preferred port is taken — and writes a
``port-<N>.json`` lock file per port it currently holds
(``{"pid", "installationName", "timestamp"}``) under its own per-OS
``port-locks`` directory. ``_comfy_target`` has no way to know that port ahead
of time: with neither ``COMFYUI_URL`` nor ``COMFYUI_HOST`` set, it used to fall
straight through to comfy-cli's hardcoded ``127.0.0.1:8188`` default, which is
simply wrong whenever Desktop landed on a different port — a config that was
correct once and silently goes stale on the app's next relaunch, with nothing
in this server's output pointing at why.

These lock in:

1. ``_comfy_desktop_port_locks_dir`` — the per-OS path, and "doesn't exist" ->
   None.
2. ``_pid_is_alive`` — a real, running pid is alive; a definitely-nonexistent
   one is not.
3. ``_discover_comfy_desktop_port`` — exactly one LIVE lock resolves; zero,
   more than one, a STALE (dead-pid) lock, and a malformed lock file all
   resolve to None rather than guessing.
4. ``_comfy_target`` wiring — the fallback only applies when neither
   ``COMFYUI_URL`` nor ``COMFYUI_HOST`` is set (both still take unconditional
   precedence, unchanged), and byte-identical ``None`` when discovery itself
   finds nothing — the existing local-only default keeps governing.
"""

from __future__ import annotations

import json
import os

import pytest

from comfy_mcp import target

# Captured at import time, before any per-test fixture (including this file's
# own ``conftest._isolate_comfy_desktop_discovery``, which patches this exact
# name to a stub for every OTHER test in the suite) has a chance to shadow it.
# The tests below that exercise `_comfy_desktop_port_locks_dir` directly
# restore this real function via `monkeypatch.setattr` so they see the actual
# path-construction logic rather than the safety-net stub.
_REAL_LOCKS_DIR = target._comfy_desktop_port_locks_dir


# --- _comfy_desktop_port_locks_dir ------------------------------------------


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("darwin", ("Library", "Application Support", "Comfy Desktop", "port-locks")),
        ("win32", ("Comfy Desktop", "port-locks")),
        ("linux", (".config", "Comfy Desktop", "port-locks")),
    ],
)
def test_locks_dir_mirrors_electron_appdata_convention(
    monkeypatch, tmp_path, platform, expected
):
    """The per-OS path follows Electron's own ``app.getPath("appData")`` layout."""
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", _REAL_LOCKS_DIR)
    monkeypatch.setattr(target.sys, "platform", platform)
    monkeypatch.setattr(target.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    # Directory must actually exist for the function to return it.
    if platform == "win32":
        d = tmp_path / "AppData" / "Roaming" / "Comfy Desktop" / "port-locks"
    elif platform == "darwin":
        d = (
            tmp_path
            / "Library"
            / "Application Support"
            / "Comfy Desktop"
            / "port-locks"
        )
    else:
        d = tmp_path / ".config" / "Comfy Desktop" / "port-locks"
    d.mkdir(parents=True)

    result = target._comfy_desktop_port_locks_dir()

    assert result == d
    for segment in expected:
        assert segment in str(result)


def test_locks_dir_none_when_missing(monkeypatch, tmp_path):
    """No Comfy Desktop install (or it has never run) -> None, not a raise."""
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", _REAL_LOCKS_DIR)
    monkeypatch.setattr(target.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path))  # empty; no "Comfy Desktop" subdir

    assert target._comfy_desktop_port_locks_dir() is None


def test_locks_dir_none_when_appdata_unset(monkeypatch):
    """Windows with no ``APPDATA`` in the environment -> None, not a KeyError."""
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", _REAL_LOCKS_DIR)
    monkeypatch.setattr(target.sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)

    assert target._comfy_desktop_port_locks_dir() is None


# --- _pid_is_alive -----------------------------------------------------------


def test_pid_is_alive_for_this_process():
    assert target._pid_is_alive(os.getpid()) is True


def test_pid_is_alive_false_for_a_pid_that_does_not_exist():
    # PID 0 is reserved/invalid on every real OS; a value this large is not a
    # real pid on any platform this runs on either — belt and suspenders.
    assert target._pid_is_alive(2**31 - 1) is False


# --- _discover_comfy_desktop_port -------------------------------------------


def _write_lock(locks_dir, port: int, *, pid: int) -> None:
    locks_dir.mkdir(parents=True, exist_ok=True)
    (locks_dir / f"port-{port}.json").write_text(
        json.dumps({"pid": pid, "installationName": "ComfyUI", "timestamp": 0}),
        encoding="utf-8",
    )


def test_discover_none_when_locks_dir_missing(monkeypatch):
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", lambda: None)

    assert target._discover_comfy_desktop_port() is None


def test_discover_none_when_no_locks(monkeypatch, tmp_path):
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", lambda: tmp_path)

    assert target._discover_comfy_desktop_port() is None


def test_discover_resolves_the_single_live_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", lambda: tmp_path)
    _write_lock(tmp_path, 8003, pid=os.getpid())

    assert target._discover_comfy_desktop_port() == 8003


def test_discover_none_for_a_stale_lock(monkeypatch, tmp_path):
    """A lock naming a dead pid (the app crashed instead of exiting cleanly)."""
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", lambda: tmp_path)
    _write_lock(tmp_path, 8003, pid=2**31 - 1)

    assert target._discover_comfy_desktop_port() is None


def test_discover_none_for_multiple_live_locks(monkeypatch, tmp_path):
    """Two live instances at once -> no principled way to pick one; don't guess."""
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", lambda: tmp_path)
    _write_lock(tmp_path, 8003, pid=os.getpid())
    _write_lock(tmp_path, 8004, pid=os.getpid())

    assert target._discover_comfy_desktop_port() is None


def test_discover_skips_malformed_lock_and_still_resolves_the_valid_one(
    monkeypatch, tmp_path
):
    """A malformed sibling lock is skipped individually, not fatal to the scan."""
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", lambda: tmp_path)
    (tmp_path / "port-9999.json").write_text("not json", encoding="utf-8")
    (tmp_path / "port-not-a-port.json").write_text("{}", encoding="utf-8")
    _write_lock(tmp_path, 8003, pid=os.getpid())

    assert target._discover_comfy_desktop_port() == 8003


def test_discover_ignores_a_lock_missing_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", lambda: tmp_path)
    (tmp_path / "port-9999.json").write_text(
        json.dumps({"installationName": "ComfyUI"}), encoding="utf-8"
    )

    assert target._discover_comfy_desktop_port() is None


# --- _comfy_target integration -----------------------------------------------


def test_target_falls_back_to_comfy_desktop_port(monkeypatch):
    """Nothing configured, but Desktop has one live instance -> use it."""
    monkeypatch.setattr(target, "_discover_comfy_desktop_port", lambda: 8003)

    assert target._comfy_target() == (
        "127.0.0.1",
        8003,
        target._COMFY_DESKTOP_PORT_LOCK_SOURCE,
    )


def test_target_none_when_discovery_finds_nothing(monkeypatch):
    """Nothing configured, no Comfy Desktop instance either -> unchanged None."""
    monkeypatch.setattr(target, "_discover_comfy_desktop_port", lambda: None)

    assert target._comfy_target() is None


def test_comfyui_url_wins_over_desktop_discovery(monkeypatch):
    """An explicit COMFYUI_URL is never overridden by the discovery fallback."""
    monkeypatch.setattr(target, "_discover_comfy_desktop_port", lambda: 8003)
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")

    assert target._comfy_target() == ("gpu.example", 9001, "COMFYUI_URL")


def test_comfyui_host_wins_over_desktop_discovery(monkeypatch):
    """An explicit COMFYUI_HOST is never overridden by the discovery fallback."""
    monkeypatch.setattr(target, "_discover_comfy_desktop_port", lambda: 8003)
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")

    assert target._comfy_target() == (
        "gpu.example",
        target.DEFAULT_COMFYUI_PORT,
        "COMFYUI_HOST",
    )
