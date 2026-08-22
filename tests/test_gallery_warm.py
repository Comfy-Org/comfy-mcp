"""The startup pre-warm of comfy-cli's template gallery cache.

``search_templates`` wraps ``comfy templates ls``, and comfy-cli fetches the
gallery index SYNCHRONOUSLY inside that command when its cache is absent or
corrupt — so on a fresh machine the first search pays a cold comfy-cli start
plus a network fetch inside its own request window, which QA watched stack into
client-side transport timeouts under parallel load. ``main()`` therefore starts
one fire-and-forget ``templates ls`` before serving. These tests pin the whole
mechanism: the no-binary short-circuit (no spawn, no failure-log noise), the
fail-silent on a comfy-cli error, the exact argv + timeout of the warm itself,
and ``main()`` starting it on a daemon thread it never joins.

The warm is stubbed out suite-wide by conftest's ``_skip_template_gallery_warm``
(``main()`` is called by other test modules), so this one restores the real
function the way ``test_machine_snapshot.py`` restores the real snapshot probe.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
from conftest import envelope

from comfy_mcp import failure_log
from comfy_mcp.server import _internal as server

# Captured at import, BEFORE conftest's autouse ``_skip_template_gallery_warm``
# replaces the module attribute — the same re-enable pattern the snapshot tests
# use. Every test here runs the REAL warm against a stubbed spawn.
_REAL_WARM = server._warm_template_gallery


@pytest.fixture(autouse=True)
def _real_warm(monkeypatch):
    """Undo conftest's no-op stub — these tests exist to exercise the warm."""
    monkeypatch.setattr(server, "_warm_template_gallery", _REAL_WARM)


def test_warm_runs_one_templates_ls_with_its_own_timeout(patched_run):
    """The warm is exactly one ``templates ls``, capped at 60s, result discarded."""
    calls = patched_run(envelope(data={"rows": [], "total": 0}))

    assert server._warm_template_gallery() is None

    (call,) = calls
    assert call["cmd"][1:4] == ["--json", "--where", "local"]
    # `templates ls` and NOT `templates refresh`: `ls` is a no-op fetch-wise on a
    # cache that is present and fresh, while `refresh` forces a network fetch on
    # every single startup.
    assert call["cmd"][4:] == ["templates", "ls"]
    # The SAME budget `search_templates` gives the identical command, not a
    # shorter one: the warm is the cold-cache invocation, so it is the one most
    # likely to need the full fetch, and an expiry SIGKILLs the process group
    # mid-write of comfy-cli's index.json.
    assert call["timeout"] == 60.0


def test_search_templates_and_the_warm_share_one_timeout(patched_run):
    """Neither can drift into a shorter budget than the other for `templates ls`."""
    warm_calls = patched_run(envelope(data={"rows": [], "total": 0}))
    server._warm_template_gallery()
    search_calls = patched_run(envelope(data={"rows": [], "total": 0}))
    server.search_templates()

    # Pinned to the value, not just to each other: an equality that both sides
    # could satisfy with `None` would pass while asserting nothing.
    assert warm_calls[0]["timeout"] == search_calls[0]["timeout"] == 60.0


def test_warm_without_a_binary_never_spawns_or_logs(monkeypatch, tmp_path):
    """No comfy-cli means no spawn AND no ``binary_missing`` record.

    ``_require_comfy_bin`` inside ``_run_comfy`` would write a failure-log entry
    for a call the user never made; the warm is purely opportunistic, so it
    checks first. The log is enabled here on purpose — its silence is the
    assertion.
    """
    path = tmp_path / "state" / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(path))
    monkeypatch.setattr(server.shutil, "which", lambda _: None)
    calls: list[tuple] = []
    monkeypatch.setattr(
        server, "_run_comfy", lambda *args, **kwargs: calls.append(args)
    )

    assert server._warm_template_gallery() is None

    assert calls == []
    assert not path.exists(), [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


@pytest.mark.parametrize(
    "failure",
    [
        server.ComfyCliError("comfy-cli said no"),
        OSError("spawn failed"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
    ids=["comfy-cli-error", "os-error", "undecodable-output"],
)
def test_warm_swallows_every_tolerated_failure(monkeypatch, failure):
    """A failed warm propagates nothing — the caller just pays what it pays today.

    Same tolerated set as ``_machine_snapshot_block``. Anything raised out of
    here would land on ``main()``'s startup thread with nobody to catch it.
    """
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def boom(*args, **kwargs):
        raise failure

    monkeypatch.setattr(server, "_run_comfy", boom)

    assert server._warm_template_gallery() is None


def test_main_starts_the_warm_on_a_daemon_thread_it_never_joins(monkeypatch):
    """``main()`` fires the warm and hands off to ``mcp.run`` without waiting.

    Unlike the snapshot probe there is no result to collect, so there is no
    bounded join either — the warm can never delay the initialize response,
    however long comfy-cli takes. A warm that blocks for the whole test proves
    it: ``mcp.run`` must still have been reached.
    """
    started = threading.Event()
    release = threading.Event()
    threads: list[threading.Thread] = []
    order: list[str] = []

    def blocking_warm():
        threads.append(threading.current_thread())
        started.set()
        release.wait(10)
        order.append("warm-finished")

    monkeypatch.setattr(server, "_warm_template_gallery", blocking_warm)
    monkeypatch.setattr(
        server.mcp,
        "run",
        lambda *, transport, show_banner: order.append(f"run:{transport}"),
    )

    began = time.monotonic()
    server.main()
    elapsed = time.monotonic() - began

    try:
        assert started.wait(5), "the warm thread never started"
        assert elapsed < 5, f"main blocked on the warm: {elapsed:.1f}s"
        # The warm is still parked in `release.wait` — that `mcp.run` already
        # ran is the whole point.
        assert order == ["run:stdio"]
        (thread,) = threads
        assert thread.name == "comfy-mcp-gallery-warm"
        assert thread.daemon, "a non-daemon warm would hold the process open"
    finally:
        release.set()


def test_warm_writes_no_failure_log_record_for_any_failure(monkeypatch, tmp_path):
    """EVERY failure mode of the warm is silent in the log, not just no-binary.

    The `shutil.which` short-circuit only ever suppressed ``binary_missing``;
    ``timeout`` / ``spawn_failed`` / ``no_json`` / ``error_envelope`` each write
    a record from inside ``_run_comfy``, and on an offline host that is one
    entry per startup spending the log's fixed rotation budget on a call no
    reader can correlate with anything they did. The warm runs inside
    ``failure_log._unattributed_calls()`` so the whole set is quiet.
    """
    path = tmp_path / "state" / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(path))
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")

    def timing_out(*args, **kwargs):
        failure_log._log_failure("timeout", args, message="comfy-cli timed out")
        raise server.ComfyCliError("comfy-cli timed out")

    monkeypatch.setattr(server, "_run_comfy", timing_out)

    assert server._warm_template_gallery() is None

    assert not path.exists(), [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def test_suppression_is_scoped_to_the_warm_thread(monkeypatch, tmp_path):
    """A real tool call's failure is still recorded, during and after a warm.

    Thread-local, not a global mute: the warm must not quiet the failures
    another thread is reporting at the same moment, and it must restore the
    previous state when it returns.
    """
    path = tmp_path / "state" / "failures.jsonl"
    monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", str(path))
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    inside = threading.Event()
    release = threading.Event()

    def parked(*args, **kwargs):
        failure_log._log_failure("timeout", args, message="warm timed out")
        inside.set()
        release.wait(10)
        raise server.ComfyCliError("warm timed out")

    monkeypatch.setattr(server, "_run_comfy", parked)
    warm = threading.Thread(target=server._warm_template_gallery, daemon=True)
    warm.start()
    try:
        assert inside.wait(5), "the warm never reached the suppressed call"
        # A different thread, mid-warm: this one IS a call the user made.
        failure_log._log_failure("timeout", ("jobs", "ls"), message="tool timed out")
    finally:
        release.set()
        warm.join(10)

    # ...and the suppression is unwound once the warm returns.
    failure_log._log_failure("no_json", ("env",), message="after the warm")

    kinds = [
        json.loads(line)["message"]
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert kinds == ["tool timed out", "after the warm"]


def test_main_survives_a_thread_start_failure(monkeypatch):
    """Thread exhaustion drops the warm; it never aborts the server.

    ``main()``'s only other handler translates ``PermissionError``, so an
    unhandled ``RuntimeError("can't start new thread")`` from the warm — the
    one part of startup documented as best-effort in every direction — would be
    the thing that stops the server from starting at all.
    """
    order: list[str] = []
    real_thread = threading.Thread

    class _ExhaustedForTheWarm(real_thread):
        """Real threads for everything else; only the warm's start() fails.

        Scoped by name rather than blanket-patching ``threading.Thread``: the
        snapshot probe above it builds one too, and a test that broke BOTH
        would not show which start() ``main()`` actually tolerates.
        """

        def start(self):
            if self.name == "comfy-mcp-gallery-warm":
                raise RuntimeError("can't start new thread")
            return super().start()

    monkeypatch.setattr(server.threading, "Thread", _ExhaustedForTheWarm)
    monkeypatch.setattr(
        server.mcp,
        "run",
        lambda *, transport, show_banner: order.append(f"run:{transport}"),
    )

    server.main()

    assert order == ["run:stdio"]
