"""Tests for ``get_execution_error`` — the failure-diagnostics companion tool.

It wraps ``comfy jobs status <id>`` and normalizes ComfyUI's raw
``execution_error`` payload (carried under the snapshot's ``error`` field) into a
compact verdict: the failing node + exception + a bounded traceback tail. These
lock in the three shapes the tool must handle: a failed status with a multi-frame
traceback (fields extracted, tail bounded), a healthy status (explicit
``error: None``, no raise), and a leading-dash prompt_id (rejected).
"""

from __future__ import annotations

import pytest

from comfy_local_mcp import server


def test_get_execution_error_extracts_fields_and_caps_traceback(monkeypatch):
    """A failed status → flat fields extracted and the traceback tailed + capped."""
    # 50 frames, each padded to ~500 chars, so the joined tail blows the byte cap
    # and forces both the frame-count tail (last 20) and the char cap to bite.
    frames = [f"frame {i}: " + "x" * 500 for i in range(50)]
    status = {
        "status": "error",
        "error": {
            "exception_message": "Tensor size mismatch",
            "exception_type": "RuntimeError",
            "node_id": 7,  # server may send an int; contract coerces to str
            "node_type": "KSampler",
            "traceback": frames,
        },
    }
    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return status

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    result = server.get_execution_error("pid-123")

    assert calls[0] == ("jobs", "status", "pid-123")  # wraps `comfy jobs status <id>`
    assert result["prompt_id"] == "pid-123"
    assert result["status"] == "error"
    assert result["exception_message"] == "Tensor size mismatch"
    assert result["exception_type"] == "RuntimeError"
    assert result["node_id"] == "7"  # coerced to str
    assert result["node_type"] == "KSampler"

    tail = result["traceback_tail"]
    # Frame-count tail keeps only the newest frames (well under all 50)...
    assert len(tail) <= server._TRACEBACK_TAIL_FRAMES + 1  # +1 for the marker
    # ...and the char cap dropped leading frames, leaving a truncation marker.
    assert tail[0] == "...(truncated)"
    joined = "\n".join(tail[1:])
    assert len(joined) <= server._TRACEBACK_TAIL_MAX_CHARS
    # The newest frame (the actual failure site) survives.
    assert tail[-1] == frames[-1]


def test_get_execution_error_small_traceback_is_not_marked(monkeypatch):
    """A short traceback passes through untouched — no cap marker, str-coerced."""
    status = {
        "status": "error",
        "error": {
            "exception_message": "boom",
            "exception_type": "ValueError",
            "node_id": None,
            "node_type": "LoadImage",
            "traceback": "single frame string",  # str, not a list
        },
    }
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: status)

    result = server.get_execution_error("pid")

    assert result["node_id"] is None
    assert result["traceback_tail"] == ["single frame string"]  # wrapped, not marked


def test_get_execution_error_tolerates_non_dict_error(monkeypatch):
    """A bare-string ``error`` is normalized, not crashed on (safe speculative call)."""
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {"status": "error", "error": "raw failure text"},
    )

    result = server.get_execution_error("pid")

    assert result["status"] == "error"
    assert result["exception_message"] == "raw failure text"
    assert result["node_id"] is None
    assert result["traceback_tail"] == []


def test_get_execution_error_no_error_returns_explicit_none(monkeypatch):
    """A completed status (no ``error``) returns ``error: None`` instead of raising."""
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {"status": "completed", "outputs": ["/tmp/gen.png"]},
    )

    result = server.get_execution_error("pid")

    assert result == {"prompt_id": "pid", "status": "completed", "error": None}


def test_get_execution_error_error_status_with_empty_payload_is_not_healthy(
    monkeypatch,
):
    """``status: error`` with a falsy ``error`` field must not masquerade as healthy."""
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {"status": "error", "error": {}},  # failed but no details
    )

    result = server.get_execution_error("pid")

    # Distinguished from the no-error branch: status stays "error", not error=None.
    assert result["status"] == "error"
    assert "error" not in result  # flat failure shape, not the healthy shape
    assert result["exception_message"] is None
    assert result["traceback_tail"] == []


def test_get_execution_error_tolerates_non_sequence_traceback(monkeypatch):
    """A malformed non-sequence ``traceback`` (dict/int) is dropped, not crashed on."""
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {
            "status": "error",
            "error": {
                "exception_message": "boom",
                "traceback": {"unexpected": "shape"},
            },
        },
    )

    result = server.get_execution_error("pid")

    assert result["exception_message"] == "boom"
    assert result["traceback_tail"] == []  # garbage traceback dropped, no TypeError


def test_get_execution_error_caps_oversized_exception_message(monkeypatch):
    """A multi-megabyte ``exception_message`` is bounded, not dumped whole."""
    huge = "z" * (server._EXCEPTION_TEXT_MAX_CHARS + 10_000)
    monkeypatch.setattr(
        server,
        "_run_comfy",
        lambda *a, **k: {"status": "error", "error": {"exception_message": huge}},
    )

    result = server.get_execution_error("pid")

    msg = result["exception_message"]
    assert len(msg) <= server._EXCEPTION_TEXT_MAX_CHARS + len("...(truncated)")
    assert msg.endswith("...(truncated)")


def test_get_execution_error_rejects_leading_dash(monkeypatch):
    """A leading-dash prompt_id is rejected before any comfy-cli call."""
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.get_execution_error("-rf")
    assert not called  # guarded before shelling out


def test_cap_traceback_tail_hard_truncates_single_oversized_frame():
    """One frame longer than the cap is character-truncated, keeping its tail."""
    frame = "y" * (server._TRACEBACK_TAIL_MAX_CHARS + 500)
    capped = server._cap_traceback_tail([frame])

    assert capped[0] == "...(truncated)"
    assert len(capped) == 2
    # The marker + separator are charged to the budget, so the *joined* result
    # (marker + frame) stays within the documented cap — not just the frame.
    joined = "\n".join(capped)
    assert len(joined) <= server._TRACEBACK_TAIL_MAX_CHARS
    # The tail of the oversized frame (the failure site) is what survives.
    assert capped[1] == frame[-len(capped[1]) :]
