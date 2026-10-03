"""Unit tests for the comfy-cli envelope parser."""

import json
import logging

import pytest

from comfy_mcp.errors import ComfyCliError
from comfy_mcp.server import (
    _last_json_object,
    _log_no_json_shape,
    _real_envelope,
    _run_comfy,
    _unwrap_envelope,
)


def _slots_envelope() -> dict:
    """A realistic ``workflow slots`` payload in the 10–20 KB range."""
    slots = [
        {
            "address": f"{index}.text",
            "node_id": str(index),
            "class_type": "CLIPTextEncode",
            "name": "text",
            "type": "STRING",
            "value": ("a prompt line " * 12) + "\u2028",
        }
        for index in range(48)
    ]
    return {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": True,
        "data": {"slots": slots},
    }


def test_prefers_envelope_over_plain_json():
    out = '{"foo": 1}\n{"type": "envelope", "ok": true, "data": {"x": 1}}\n'
    assert _last_json_object(out) == {
        "type": "envelope",
        "ok": True,
        "data": {"x": 1},
    }


def test_ignores_non_json_noise():
    out = 'loading...\nprogress 50%\n{"type": "envelope", "ok": true, "data": null}\n'
    assert _last_json_object(out) == {"type": "envelope", "ok": True, "data": None}


def test_falls_back_to_last_json_when_no_envelope():
    assert _last_json_object('{"a": 1}\n{"b": 2}\n') == {"b": 2}


def test_returns_none_when_no_json():
    assert _last_json_object("just text\nmore text\n") is None


def test_whole_slots_document_survives_a_line_separator():
    """A complete envelope stays one document when splitlines() cuts it."""
    stdout = json.dumps(_slots_envelope(), ensure_ascii=False) + "\n"
    assert 10_000 <= len(stdout.encode("utf-8")) <= 20_000
    assert stdout.count("\n") == 1
    assert len(stdout.splitlines()) > 1
    found = _last_json_object(stdout)
    assert found is not None
    assert _real_envelope(found) is not None
    assert found["schema"] == "envelope/1"
    assert found["ok"] is True
    assert len(found["data"]["slots"]) == 48


def test_compact_single_line_envelope():
    stdout = '{"schema":"envelope/1","type":"envelope","ok":true,"data":{"n":1}}\n'
    found = _last_json_object(stdout)
    assert found == {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": True,
        "data": {"n": 1},
    }
    assert _real_envelope(found) is found


def test_pretty_multiline_envelope():
    stdout = json.dumps(
        {"schema": "envelope/1", "type": "envelope", "ok": True, "data": {}},
        indent=2,
    )
    assert len(stdout.splitlines()) > 1
    found = _last_json_object(stdout)
    assert _real_envelope(found) == {
        "schema": "envelope/1",
        "type": "envelope",
        "ok": True,
        "data": {},
    }


def test_carriage_return_whitespace_is_one_document():
    body = json.dumps(
        {"schema": "envelope/1", "type": "envelope", "ok": True, "data": {"k": 1}}
    )
    stdout = body.replace(": ", ":\r", 1) + "\n"
    assert stdout.count("\n") == 1
    assert len(stdout.splitlines()) == 2
    assert json.loads(stdout)["type"] == "envelope"
    found = _last_json_object(stdout)
    assert _real_envelope(found) is not None
    assert found["data"] == {"k": 1}


def test_ndjson_still_prefers_the_latest_envelope():
    stdout = (
        '{"type":"progress","value":1}\n'
        '{"type":"envelope","schema":"envelope/1","ok":true,"data":{"first":1}}\n'
        '{"type":"progress","value":2}\n'
        '{"type":"envelope","schema":"envelope/1","ok":true,"data":{"last":1}}\n'
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(stdout)
    found = _last_json_object(stdout)
    assert found["data"] == {"last": 1}
    assert _real_envelope(found) is found


def test_mixed_diagnostic_text_still_uses_the_json_line():
    stdout = (
        "loading custom nodes\n"
        "progress 50%\n"
        '{"type":"envelope","schema":"envelope/1","ok":true,"data":{"ok":1}}\n'
    )
    found = _last_json_object(stdout)
    assert found["data"] == {"ok": 1}
    assert _real_envelope(found) is not None


def test_empty_output_returns_none():
    assert _last_json_object("") is None
    assert _last_json_object(" \n\t") is None
    assert _real_envelope(None) is None


def test_invalid_json_returns_none():
    assert _last_json_object("{not json") is None
    assert _last_json_object("}\n{also bad\n") is None


@pytest.mark.parametrize(
    "stdout",
    ["[1, 2, 3]", "42", '"hello"', "true", "null", "[\n  1,\n  2\n]"],
)
def test_top_level_non_object_is_not_an_envelope(stdout):
    assert _last_json_object(stdout) is None
    assert _real_envelope(_last_json_object(stdout)) is None


def test_run_comfy_returns_a_whole_slots_document(patched_run):
    stdout = json.dumps(_slots_envelope(), ensure_ascii=False) + "\n"
    patched_run(stdout)
    assert len(_run_comfy("workflow", "slots", "graph.json")["slots"]) == 48


def test_no_json_logs_whole_document_shape_without_the_body(caplog):
    secret = "SHOULD_NOT_APPEAR_IN_LOG"
    stdout = (
        json.dumps(
            {
                "schema": "envelope/1",
                "type": "envelope",
                "ok": True,
                "data": {"value": secret},
            }
        ).replace(": ", ":\r", 1)
        + "\n"
    )
    caplog.set_level(logging.WARNING, logger="comfy_mcp.server")
    with pytest.raises(ComfyCliError, match="returned no JSON"):
        _unwrap_envelope(None, ("workflow", "slots"), 0, "", stdout=stdout)
    text = caplog.text
    assert "whole_json_valid=true" in text
    assert "top_level_type=dict" in text
    assert "json_type=envelope" in text
    assert "schema=envelope/1" in text
    assert "stdout_chars=" in text
    assert "stdout_bytes=" in text
    assert "splitlines=" in text
    assert secret not in text
    assert stdout not in text


def test_no_json_shape_marks_invalid_and_non_object(caplog):
    caplog.set_level(logging.WARNING, logger="comfy_mcp.server")
    _log_no_json_shape("not json")
    assert "whole_json_valid=false" in caplog.text
    assert "top_level_type=invalid" in caplog.text
    caplog.clear()
    _log_no_json_shape("[1, 2]")
    assert "whole_json_valid=true" in caplog.text
    assert "top_level_type=list" in caplog.text
    assert "json_type=-" in caplog.text
