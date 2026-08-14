"""Tests for the pid ``launch_comfyui`` / ``server_info`` report.

``comfy launch --background`` records the pid of the WRAPPER it spawned — the
detached ``comfy … launch`` re-invocation — not the ``python main.py`` that
wrapper starts and that binds the port. That recorded value is what both the
launch envelope's ``pid`` and ``comfy env``'s ``config.background.pid`` carry,
so anything acting on the number this server hands out acts on the parent:
killing it leaves ComfyUI running and still holding the port.

What is under test is the reconciliation that fixes it, and it is still only a
passthrough: ``comfy stop --port <p> --dry-run`` — the same probe the untracked
kill gate reads — names the process comfy-cli itself verified is on the port,
and that is the pid reported. The recorded value is kept beside it as
``recorded_pid`` because it is still the pid ``comfy stop`` acts on. So:

1. The correction: reported pid == the port holder comfy-cli vouched for, on
   both tools, with the wrapper pid preserved and labelled.
2. The degrade: an engine that will not vouch (nothing listening, a comfy-cli
   predating ``comfy stop --port``) keeps the recorded pid and SAYS it is
   unconfirmed, rather than reporting it as the listener.
3. The cost: no pid/port pair to reconcile means no extra subprocess at all.

comfy-cli is mocked throughout; no process is ever looked at or signalled.
"""

from __future__ import annotations

import json

import pytest
from conftest import _FakeRunProc, _raises_at_spawn, envelope

from comfy_mcp import server

#: The pid comfy-cli records: the `comfy … launch` wrapper, reparented to init.
WRAPPER_PID = 64063

#: Its child — the `python main.py` that actually holds the port.
LISTENER_PID = 64067

PORT = 8188

#: The listener's argv, as `comfy stop --port --dry-run` reports it.
_CMDLINE = ["/ComfyUI/.venv/bin/python", "main.py", "--port", str(PORT)]


def _dry_run(pid: int = LISTENER_PID, port: int = PORT, **overrides) -> dict:
    """``comfy stop --port <p> --dry-run``'s payload for a verified ComfyUI.

    Mirrors comfy-cli's ``stop.json`` schema (``stopped`` false because nothing
    was stopped, ``dry_run``/``verified``/``untracked`` true, plus the identity),
    so a test overriding one field is overriding a real one.
    """
    data = {
        "stopped": False,
        "dry_run": True,
        "verified": True,
        "untracked": True,
        "pid": pid,
        "port": port,
        "cmdline": list(_CMDLINE),
    }
    data.update(overrides)
    return data


def _env_data(background: object = _dry_run, **overrides) -> dict:
    """``comfy env``'s data payload, with a background record under ``config``.

    ``comfy env`` reports the record comfy-cli wrote at launch, which is where
    the wrapper pid surfaces to a caller. Pass ``background=None`` for an env
    with no recorded server.
    """
    record = {"host": "127.0.0.1", "port": PORT, "pid": WRAPPER_PID}
    if background is not _dry_run:
        record = background  # type: ignore[assignment]
    data = {
        "python": {"version": "3.12.0", "executable": "/usr/bin/python3"},
        "config": {"path": "/home/user/.config/comfy-cli/config.ini"},
        "server": {"running": True, "url": "http://127.0.0.1:8188"},
    }
    data["config"]["background"] = record  # type: ignore[index]
    data.update(overrides)
    return data


def _line(payload) -> str:
    """One ``envelope/1`` stdout line carrying ``payload`` as its data."""
    return json.dumps(envelope(data=payload))


@pytest.fixture
def sequenced(monkeypatch):
    """Answer each spawn from a queue of ``(returncode, stdout, stderr)`` replies.

    The reconciliation makes a SECOND comfy-cli call after the one the tool
    itself made, so these paths shell out more than once and the shared
    single-reply fakes cannot express them — the sequenced-replies carve-out
    AGENTS.md allows for a local stub. It still mirrors the shared fake's spawn
    signature and reuses its :class:`_FakeRunProc`, so a change to how ``server``
    shells out breaks here loudly rather than drifting. A queue that runs out
    fails the test, which is how "no extra subprocess" is asserted.
    """

    def setup(replies: list) -> list[dict]:
        calls: list[dict] = []

        def fake(
            cmd, stdout, stderr, stdin, text, encoding, env, start_new_session, cwd
        ):
            record = {"cmd": cmd, "timeout": None, "cwd": cwd}
            calls.append(record)
            try:
                reply = replies[len(calls) - 1]
            except IndexError:
                raise AssertionError(f"unexpected comfy-cli call: {cmd}") from None
            failed = isinstance(reply, BaseException)
            if failed and _raises_at_spawn(reply):
                raise reply
            returncode, out, err = (None, None, None) if failed else reply
            return _FakeRunProc(
                cmd,
                record,
                stdout=out,
                stderr=err,
                returncode=returncode,
                raises=reply if failed else None,
            )

        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(server.subprocess, "Popen", fake)
        monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.15.0")
        monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
        # `server_info` also probes `comfy outdated`; these tests are about the
        # pid, so keep that off the queue and out of the call assertions.
        monkeypatch.setattr(server, "_freshness_report", lambda: {})
        return calls

    return setup


# --- server_info: the recorded pid is reconciled before it is reported -------


def test_server_info_reports_the_pid_that_holds_the_port(sequenced):
    """The ticket's acceptance: reported pid == the verified port holder's pid."""
    calls = sequenced([(0, _line(_env_data()), ""), (0, _line(_dry_run()), "")])

    background = server.server_info()["config"]["background"]

    assert background["pid"] == LISTENER_PID
    assert background["pid"] != WRAPPER_PID
    assert background["recorded_pid"] == WRAPPER_PID
    assert background["pid_source"] == "port-listener"
    # The rest of the record passes through untouched.
    assert background["host"] == "127.0.0.1"
    assert background["port"] == PORT
    # Second spawn is the dry run, on the port the record names, and it is a
    # DRY run: without `--dry-run` this reconciliation would stop the server.
    assert calls[1]["cmd"][4:] == ["stop", "--port", str(PORT), "--dry-run"]


def test_server_info_keeps_a_pid_the_engine_will_not_vouch_for_but_flags_it(
    sequenced,
):
    """Nothing on the port -> the recorded pid stays, labelled unconfirmed.

    The degrade must not silently promote the wrapper pid: `pid_source` says it
    came from the CLI's record, and the note says not to act on it.
    """
    calls = sequenced(
        [
            (0, _line(_env_data()), ""),
            (1, _line({}), "Nothing is listening on port 8188."),
        ]
    )

    background = server.server_info()["config"]["background"]

    assert background["pid"] == WRAPPER_PID
    assert background["pid_source"] == "cli-record"
    assert "recorded_pid" not in background
    assert "could NOT be confirmed" in background["pid_note"]
    assert len(calls) == 2


def test_server_info_flags_the_pid_on_a_comfy_cli_without_stop_port(sequenced):
    """A comfy-cli predating `comfy stop --port` degrades the same way.

    Click rejects the unknown option while PARSING (exit 2, no envelope), so
    nothing was ever dispatched and nothing could have been stopped.
    """
    sequenced(
        [
            (0, _line(_env_data()), ""),
            (2, "", "Error: No such option: --port"),
        ]
    )

    background = server.server_info()["config"]["background"]

    assert background["pid"] == WRAPPER_PID
    assert background["pid_source"] == "cli-record"
    assert "1.15.0" in background["pid_note"]


def test_server_info_does_not_probe_when_nothing_is_recorded(sequenced):
    """No background record -> no reconciliation, and no second subprocess.

    The queue holds ONE reply, so a second spawn fails the test rather than
    quietly costing a `comfy stop --port` on every `server_info`.
    """
    calls = sequenced([(0, _line(_env_data(background=None)), "")])

    result = server.server_info()

    assert result["config"]["background"] is None
    assert len(calls) == 1


def test_server_info_reconciles_a_port_recorded_as_a_string(sequenced):
    """The record round-trips through a config file, so its port can be text."""
    data = _env_data()
    data["config"]["background"]["port"] = str(PORT)
    calls = sequenced([(0, _line(data), ""), (0, _line(_dry_run()), "")])

    background = server.server_info()["config"]["background"]

    assert background["pid"] == LISTENER_PID
    assert calls[1]["cmd"][4:] == ["stop", "--port", str(PORT), "--dry-run"]


def test_server_info_leaves_an_unreconcilable_record_alone(sequenced):
    """A record with no usable port is passed through, unprobed and unlabelled."""
    calls = sequenced(
        [
            (
                0,
                _line(_env_data(background={"host": "127.0.0.1", "pid": WRAPPER_PID})),
                "",
            )
        ]
    )

    background = server.server_info()["config"]["background"]

    assert background == {"host": "127.0.0.1", "pid": WRAPPER_PID}
    assert len(calls) == 1


def test_server_info_ignores_an_engine_that_reports_an_unverified_listener(sequenced):
    """`verified` is comfy-cli's own judgment; without it there is no answer."""
    sequenced(
        [
            (0, _line(_env_data()), ""),
            (0, _line(_dry_run(verified=False)), ""),
        ]
    )

    background = server.server_info()["config"]["background"]

    assert background["pid"] == WRAPPER_PID
    assert background["pid_source"] == "cli-record"


def test_server_info_reports_an_already_correct_pid_as_verified(sequenced):
    """An engine that records the listener itself needs no correction.

    The fix is "report the port holder", not "always rewrite the pid": when the
    two agree the pid stands and is simply reported as confirmed.
    """
    sequenced(
        [
            (0, _line(_env_data()), ""),
            (0, _line(_dry_run(pid=WRAPPER_PID)), ""),
        ]
    )

    background = server.server_info()["config"]["background"]

    assert background["pid"] == WRAPPER_PID
    assert background["recorded_pid"] == WRAPPER_PID
    assert background["pid_source"] == "port-listener"


# --- launch_comfyui: the same correction on the launch envelope --------------


def test_launch_reports_the_pid_that_holds_the_port(sequenced):
    """`comfy launch`'s own envelope carries the wrapper pid; correct it too."""
    launch_data = {
        "background": True,
        "listen": "127.0.0.1",
        "port": PORT,
        "url": f"http://127.0.0.1:{PORT}",
        "pid": WRAPPER_PID,
    }
    calls = sequenced([(0, _line(launch_data), ""), (0, _line(_dry_run()), "")])

    result = server._launch_comfyui_sync([])

    assert result["pid"] == LISTENER_PID
    assert result["recorded_pid"] == WRAPPER_PID
    assert result["pid_source"] == "port-listener"
    assert result["url"] == f"http://127.0.0.1:{PORT}"
    assert calls[1]["cmd"][4:] == ["stop", "--port", str(PORT), "--dry-run"]


def test_launch_without_a_reported_pid_is_untouched(sequenced):
    """The plain-text launch synthesis carries no pid, so there is none to fix.

    A comfy-cli that prints human text and emits no envelope reports no pid at
    all — nothing wrong is being handed out, and probing for one would be new
    behavior rather than a correction. One queued reply, so a probe would fail.
    """
    calls = sequenced([(0, "", "ComfyUI is successfully launched in the background.")])

    result = server._launch_comfyui_sync([])

    assert result["ok"] is True
    assert "pid" not in result
    assert len(calls) == 1


def test_restart_reports_the_corrected_pid_too(monkeypatch, sequenced):
    """`restart_comfyui` composes the launch, so the correction rides along.

    Asserted through `_restart_comfyui_locked` — the composition itself —
    rather than by re-testing the launch half, so a future restart path that
    stopped going through `_launch_comfyui_sync` would fail here.
    """
    monkeypatch.setattr(server, "stop_comfyui", lambda: {"ok": True})
    launch_data = {"background": True, "port": PORT, "pid": WRAPPER_PID}
    sequenced([(0, _line(launch_data), ""), (0, _line(_dry_run()), "")])

    result = server._restart_comfyui_locked([])

    assert result["pid"] == LISTENER_PID
    assert result["recorded_pid"] == WRAPPER_PID
