"""Unit tests for the comfy-cli envelope parser."""

from comfy_mcp.server._internal import _last_json_object


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
