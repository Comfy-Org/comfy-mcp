"""`auth_login` — spawn `comfy cloud login`, hand its OAuth URL to the agent.

Unlike the rest of the suite, these tests run a REAL child process against a
fake `comfy` binary on disk rather than a patched ``subprocess`` fake. That is
deliberate and load-bearing: `auth_login` is the only tool that uses
``asyncio.create_subprocess_exec``, and the whole contract it implements —
return as soon as the `login_url` line is FLUSHED, while the child keeps running
— only exists across a real pipe. A `subprocess` stand-in that hands back a
pre-filled buffer would satisfy every assertion here while proving none of it.

The fake CLI is a two-file pair (`tests/conftest.py`'s fixtures cover the two
*synchronous* spawn paths and can't help here): a tiny `/bin/sh` shim named
`comfy` that appends its argv to a log and then `exec`s the scenario script
under this interpreter. The shim is what makes the exact-argv and
single-spawn assertions possible; the `exec` keeps the scenario a normal Python
file instead of a quoting exercise, and keeps it in the shim's process so a
kill of the process group reaches it.

Every test drives the tool inside ONE ``asyncio.run`` — the parked child, its
reader task and its pipes all belong to the loop that created them, so a second
``asyncio.run`` would tear the parked login down between calls and quietly test
nothing.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from comfy_mcp import server

_needs_posix_exec = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX exec bit + /bin/sh shim; the fake CLI can't run on Windows",
)

pytestmark = _needs_posix_exec

_URL = "https://api.comfy.org/oauth/authorize?client_id=comfy-cli&code_challenge=abc"

_LOGIN_URL_LINE = json.dumps(
    {"schema": "event/1", "type": "login_url", "url": _URL, "timeout_s": 600}
)

_SUCCESS_ENVELOPE = json.dumps(
    {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": True,
        # comfy-cli redacts the session itself; kept here so the test proves
        # `auth_login` echoes nothing out of `data` even when `data` is present.
        "data": {"action": "login", "session": {"access_token": "<redacted>"}},
    }
)

_OAUTH_TIMEOUT_ENVELOPE = json.dumps(
    {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": False,
        "error": {
            "code": "oauth_timeout",
            "message": "timed out waiting for the browser callback",
            "hint": "re-run `comfy cloud login` and finish sign-in in the browser",
        },
    }
)

_EMIT_URL = f"_w({_LOGIN_URL_LINE!r})\n"


def _fake_comfy(tmp_path: Path, body: str) -> tuple[Path, Path]:
    """Write an executable fake `comfy` + its scenario; return (binary, argv log).

    ``body`` is Python appended to a preamble that defines ``_w(line)`` (write +
    flush, because the parent is reading a pipe line-by-line and an unflushed
    write is indistinguishable from a CLI that never emitted the event).
    """
    scenario = tmp_path / "fake_login.py"
    scenario.write_text(
        "import sys, time\n"
        "def _w(line):\n"
        "    sys.stdout.write(line + '\\n')\n"
        "    sys.stdout.flush()\n"
        f"{body}"
    )
    argv_log = tmp_path / "argv.log"
    exe = tmp_path / "comfy"
    # `$*` (not `$@`) so each spawn is exactly ONE line in the log — the
    # single-spawn assertion counts lines.
    exe.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{argv_log}"\n'
        f'exec "{sys.executable}" "{scenario}" "$@"\n'
    )
    exe.chmod(0o755)
    return exe, argv_log


def _spawn_argvs(argv_log: Path) -> list[str]:
    if not argv_log.exists():
        return []
    return [line for line in argv_log.read_text().splitlines() if line]


@pytest.fixture(autouse=True)
def _isolated_login_state(monkeypatch):
    """Give each test a clean module state and never leak a login child.

    `auth_login` parks its child in module globals on purpose (that is how a
    second call finds the first flow), so without this a test would inherit the
    previous one's child and a long-sleeping fake would outlive the run.
    """
    monkeypatch.setattr(server, "_login_child", None, raising=False)
    monkeypatch.setattr(server, "_login_lock", None, raising=False)
    monkeypatch.setattr(server, "_login_lock_loop", None, raising=False)
    yield
    child = server._login_child
    server._login_child = None
    if child is not None:
        server._kill_proc_tree_async(child.proc)


async def _shutdown_login() -> None:
    """Reap the parked child from inside the loop that owns it."""
    child = server._login_child
    server._login_child = None
    if child is not None:
        await server._abandon_login_child(child)


async def _await_child_exit(child, timeout: float = 10.0) -> None:
    """Wait for the reader task to park the child's terminal result."""
    await asyncio.wait_for(asyncio.shield(child.reader), timeout)


def test_login_url_event_returns_awaiting_browser(tmp_path, monkeypatch):
    """The headline case: URL emitted, child still blocked on the callback."""
    exe, argv_log = _fake_comfy(tmp_path, _EMIT_URL + "time.sleep(120)\n")
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))

    async def scenario():
        try:
            return await server.auth_login()
        finally:
            await _shutdown_login()

    result = asyncio.run(scenario())

    assert result["status"] == "awaiting_browser"
    assert result["login_url"] == _URL
    # Counted down from the child's own reported deadline, so it is at most the
    # 600 the tool asks for — never a constant echoed back.
    assert 590 <= result["expires_in_s"] <= 600
    assert "auth_status" in result["next"]
    # No token, and nothing lifted out of the envelope's `data`.
    assert "session" not in result and "access_token" not in json.dumps(result)
    # The exact spawn the ticket pins: global `--json` BEFORE the subcommand,
    # `--no-browser` (this server may be headless relative to the user), and no
    # `--where` (a cloud verb is not local-targetable).
    assert _spawn_argvs(argv_log) == ["--json cloud login --no-browser --timeout 600"]


def test_error_envelope_before_url_raises_unwrapped(tmp_path, monkeypatch):
    """A child that dies before a URL surfaces comfy-cli's own code + hint."""
    exe, argv_log = _fake_comfy(
        tmp_path,
        f"_w({_OAUTH_TIMEOUT_ENVELOPE!r})\nsys.exit(1)\n",
    )
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))

    async def scenario():
        try:
            return await server.auth_login()
        finally:
            await _shutdown_login()

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(scenario())

    assert excinfo.value.code == "oauth_timeout"
    message = str(excinfo.value)
    assert "timed out waiting for the browser callback" in message
    # The hint is the actionable half; dropping it was the whole point of
    # routing this through `_unwrap_envelope` rather than reporting an exit code.
    assert "finish sign-in in the browser" in message
    # A failed login parks nothing — the next call starts a fresh flow.
    assert server._login_child is None
    assert len(_spawn_argvs(argv_log)) == 1


def test_second_call_while_pending_reuses_the_same_flow(tmp_path, monkeypatch):
    """One concurrent login: the second call re-reports, it does not re-spawn."""
    exe, argv_log = _fake_comfy(tmp_path, _EMIT_URL + "time.sleep(120)\n")
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))

    async def scenario():
        try:
            first = await server.auth_login()
            second = await server.auth_login()
            return first, second
        finally:
            await _shutdown_login()

    first, second = asyncio.run(scenario())

    assert first["status"] == second["status"] == "awaiting_browser"
    assert first["login_url"] == second["login_url"] == _URL
    # The load-bearing assertion: a second child would bind its own loopback
    # listener and hand the user a URL for a race loser.
    assert len(_spawn_argvs(argv_log)) == 1


def test_concurrent_calls_spawn_exactly_one_child(tmp_path, monkeypatch):
    """The same guard under a genuine race — both calls awaited together.

    The check-then-spawn in `auth_login` is two awaits wide, so without its lock
    two concurrent calls both see "no child" and both spawn. The sequential test
    above cannot catch that; this one can.
    """
    exe, argv_log = _fake_comfy(tmp_path, _EMIT_URL + "time.sleep(120)\n")
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))

    async def scenario():
        try:
            return await asyncio.gather(server.auth_login(), server.auth_login())
        finally:
            await _shutdown_login()

    first, second = asyncio.run(scenario())

    assert first["login_url"] == second["login_url"] == _URL
    assert len(_spawn_argvs(argv_log)) == 1


def test_completed_flow_is_reported_once_then_cleared(tmp_path, monkeypatch):
    """After the child succeeds, the next call reports it and drops the state."""
    exe, argv_log = _fake_comfy(tmp_path, _EMIT_URL + f"_w({_SUCCESS_ENVELOPE!r})\n")
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))

    async def scenario():
        try:
            pending = await server.auth_login()
            await _await_child_exit(server._login_child)
            done = await server.auth_login()
            # State cleared: a third call has nothing parked to report, so it
            # would start a fresh sign-in.
            return pending, done, server._login_child
        finally:
            await _shutdown_login()

    pending, done, parked = asyncio.run(scenario())

    assert pending["status"] == "awaiting_browser"
    assert done["status"] == "completed"
    assert "auth_status" in done["next"]
    # Terminal report carries status only — the envelope's `data.session` (which
    # comfy-cli has already redacted) is never echoed.
    assert "login_url" not in done
    assert "access_token" not in json.dumps(done)
    assert parked is None
    assert len(_spawn_argvs(argv_log)) == 1


def test_failed_flow_is_reported_once_then_cleared(tmp_path, monkeypatch):
    """The completion twin: a child that fails AFTER handing out the URL.

    This is the real oauth-timeout shape — comfy-cli emits `login_url`, blocks
    on the callback, and only then errors — so the failure lands on the terminal
    report rather than on the raising path exercised above.
    """
    exe, _argv_log = _fake_comfy(
        tmp_path,
        _EMIT_URL + f"_w({_OAUTH_TIMEOUT_ENVELOPE!r})\nsys.exit(1)\n",
    )
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))

    async def scenario():
        try:
            await server.auth_login()
            await _await_child_exit(server._login_child)
            return await server.auth_login(), server._login_child
        finally:
            await _shutdown_login()

    failed, parked = asyncio.run(scenario())

    assert failed["status"] == "failed"
    assert failed["error_code"] == "oauth_timeout"
    assert "timed out waiting for the browser callback" in failed["message"]
    assert parked is None


def test_wedged_child_past_its_deadline_is_replaced(tmp_path, monkeypatch):
    """A child alive well past its own deadline must not hold the login slot.

    Only one sign-in may be in flight, so without the overdue check a comfy-cli
    that never exits would make `auth_login` hand back the same long-dead URL
    for the rest of the process — a dead end with no way for the agent or the
    user to reset it.
    """
    exe, argv_log = _fake_comfy(tmp_path, _EMIT_URL + "time.sleep(120)\n")
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))

    async def scenario():
        try:
            first = await server.auth_login()
            wedged = server._login_child
            # Backdate the deadline instead of sleeping past a real one: the
            # child's own `--timeout` is what we are pretending elapsed.
            wedged.expires_at -= server._LOGIN_TIMEOUT_S + (
                server._LOGIN_OVERDUE_GRACE_S + 1
            )
            assert wedged.is_overdue()
            second = await server.auth_login()
            return first, second, wedged
        finally:
            await _shutdown_login()

    first, second, wedged = asyncio.run(scenario())

    assert first["status"] == second["status"] == "awaiting_browser"
    # A genuinely fresh flow: a second spawn, and the wedged child reaped rather
    # than left holding its loopback port.
    assert len(_spawn_argvs(argv_log)) == 2
    assert wedged.proc.returncode is not None
    assert server._login_child is None


def test_pending_child_within_its_deadline_is_not_overdue(tmp_path, monkeypatch):
    """The other half of the wedge check — a live flow is never pre-empted."""
    exe, _argv_log = _fake_comfy(tmp_path, _EMIT_URL + "time.sleep(120)\n")
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))

    async def scenario():
        try:
            await server.auth_login()
            child = server._login_child
            # Past the deadline but inside the shutdown grace: the child may
            # still be writing its terminal envelope, so it stays.
            child.expires_at -= server._LOGIN_TIMEOUT_S + (
                server._LOGIN_OVERDUE_GRACE_S / 2
            )
            return child.is_overdue()
        finally:
            await _shutdown_login()

    assert asyncio.run(scenario()) is False


def test_no_login_url_within_budget_is_an_actionable_error(tmp_path, monkeypatch):
    """A comfy-cli too old to emit the event must not leave a stranded flow.

    The child holds a loopback listener the agent has no URL for, so the tool
    kills it and points at the path that still works instead of parking a login
    nobody can complete.
    """
    exe, _argv_log = _fake_comfy(tmp_path, "time.sleep(120)\n")
    monkeypatch.setattr(server, "COMFY_BIN", str(exe))
    monkeypatch.setattr(server, "_LOGIN_URL_WAIT_S", 1.0)

    async def scenario():
        try:
            return await server.auth_login()
        finally:
            await _shutdown_login()

    with pytest.raises(server.ComfyCliError) as excinfo:
        asyncio.run(scenario())

    message = str(excinfo.value)
    assert "login_url" in message
    assert "upgrade comfy-cli" in message.lower()
    assert "comfy cloud login" in message
    assert server._login_child is None


def test_missing_binary_refuses_before_spawning(tmp_path, monkeypatch):
    """The shared `_require_comfy_bin` guard runs on this path too."""
    monkeypatch.setattr(server, "COMFY_BIN", str(tmp_path / "nope" / "comfy"))

    with pytest.raises(server.ComfyCliError, match="not found on PATH"):
        asyncio.run(server.auth_login())

    assert server._login_child is None
